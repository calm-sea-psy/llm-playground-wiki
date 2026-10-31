"""에이전트 노드의 LLM 판단: 질문을 하위 질문으로 나누고, 모은 근거로 답할 수 있는지 · 무엇을 더 찾을지 정한다.

노드 프롬프트는 컨텍스트 조립기의 에이전트 칸 배치(`agent_layout`)로 만든다: system 은 노드 공통 고정 지시(base),
user 는 질문 → 하위 질문 → 근거 → 쓴 검색어 → 메모 → 노드 지시 → 질문 상기. 에이전트가 쓴 글(하위 질문 · 검색어 ·
메모)도 검색 문서처럼 태그 안 data 칸에만 들어간다 — 문서 속 지시가 판단 · 새 검색어로 옮겨 오는 길을 좁힌다.
"""

import asyncio
import functools
import logging
from collections.abc import Callable
from typing import Self

from playground_llm import (
    LLM,
    GenerationError,
    Message,
    ModelNotFoundError,
    ModelSpec,
    PromptLibrary,
)

from playground_wiki.agent.evidence import ranked_candidates
from playground_wiki.agent.schemas import generate_judgment, generate_sub_questions
from playground_wiki.agent.state import DecomposeStatus, Judgment
from playground_wiki.generation import PROMPT_DIR
from playground_wiki.generation.context import (
    Candidate,
    CharTokenCounter,
    ContextAssembler,
    WorkingMemory,
    memory_candidates,
)
from playground_wiki.generation.context_layouts import (
    DOCUMENTS,
    NODE,
    NOTES,
    QUERIES,
    QUESTION,
    REMINDER,
    SUB_QUESTIONS,
    SYSTEM,
    agent_layout,
)
from playground_wiki.generation.messages import context_budget

logger = logging.getLogger(__name__)

AGENT_PROMPT_DIR = PROMPT_DIR / "agent"
"""노드 지시 프롬프트 (`agent/{노드}/{버전}.md`). 답변 프롬프트(PROMPT_DIR)와 따로 읽는다."""
DECOMPOSE_PROMPT = "decompose@v1"
JUDGE_PROMPT = "judge@v1"
MAX_SUB_QUESTIONS = 4  # 복합 세트 v1 의 hop 은 2~3 — 넉넉히 4. 하위 질문마다 검색 한 번이라 검색 횟수 · 지연의 상한이다
MAX_NEW_QUERIES = (
    2  # 판단 한 번에 새 검색어 둘까지 — 빠진 hop 은 보통 하나, 반복 상한 3 과 곱해 검색 수를 묶는다
)
# 노드 출력 길이 상한. 하위 질문 4개 × 40자 ≈ 110토큰 + JSON 틀 — 두 배 여유
DECOMPOSE_MAX_TOKENS = 256
# found · missing 한 문장씩(≈ 각 100자) + 새 검색어 둘 + JSON 틀 ≈ 200토큰 — 두 배 여유
JUDGE_MAX_TOKENS = 384
# 노드 하나의 벽시계 상한. 답변 단계(60초 · 512~2048토큰)보다 출력이 짧아 절반. 넘기면 원문 하나 · 모은 근거로 답하기로 간다
# (동기 HTTP 요청이라 스레드는 서버가 끝낼 때까지 남는다 — Ollama 한 번에 하나라 다음 호출이 그만큼 기다린다)
NODE_TIMEOUT_SEC = 30.0
NODE_TEMPERATURE = 0.0  # 판단 · 검색어는 같은 입력에 같은 출력이 낫다 (답변 단계의 0.1 과 다르다)
_FAILURES = (GenerationError, ModelNotFoundError, TimeoutError)


class Reasoner:
    """분해 · 판단 노드가 부르는 LLM 판단. 실패는 예외로 올리지 않고 상태(error)로 돌려준다.

    예)
        reasoner = Reasoner.from_spec(llm, policy.profiles["grounded"].spec, low=0.06)
        sub_questions, status = await reasoner.decompose(question, system=base_text)
    """

    def __init__(
        self,
        llm: LLM,
        library: PromptLibrary,
        *,
        decompose_spec: ModelSpec,
        judge_spec: ModelSpec,
        low: float,
        timeout_sec: float = NODE_TIMEOUT_SEC,
    ) -> None:
        self._llm = llm
        self._library = library
        self._decompose_spec = decompose_spec
        self._judge_spec = judge_spec
        self._low = low
        self._timeout_sec = timeout_sec

    @classmethod
    def from_spec(
        cls, llm: LLM, spec: ModelSpec, *, low: float, timeout_sec: float = NODE_TIMEOUT_SEC
    ) -> Self:
        """답변 단계의 모델 설정(gemma4:12b · think off · num_ctx)에서 노드별 출력 길이 · temperature 만 바꿔 만든다.

        Raises:
          PromptError: 노드 프롬프트(AGENT_PROMPT_DIR)가 없을 때.
        """
        library = PromptLibrary.load(AGENT_PROMPT_DIR)
        for ref in (DECOMPOSE_PROMPT, JUDGE_PROMPT):
            library.get(ref)
        return cls(
            llm,
            library,
            decompose_spec=spec.with_params(
                max_tokens=DECOMPOSE_MAX_TOKENS, temperature=NODE_TEMPERATURE
            ),
            judge_spec=spec.with_params(max_tokens=JUDGE_MAX_TOKENS, temperature=NODE_TEMPERATURE),
            low=low,
            timeout_sec=timeout_sec,
        )

    @property
    def prompt_versions(self) -> tuple[str, ...]:
        """노드 지시 프롬프트의 `이름@버전:해시` (분해 · 판단)."""
        return tuple(
            f"{template.ref}:{template.digest}"
            for template in (
                self._library.get(DECOMPOSE_PROMPT),
                self._library.get(JUDGE_PROMPT),
            )
        )

    async def decompose(
        self, question: str, *, system: str
    ) -> tuple[tuple[str, ...], DecomposeStatus]:
        """질문을 나눈 하위 질문과 결과 상태를 돌려준다. 하나도 안 나오거나 실패하면 원문 하나."""
        node = self._library.get(DECOMPOSE_PROMPT).render(max_sub_questions=str(MAX_SUB_QUESTIONS))
        messages = self.node_messages(WorkingMemory(question), system=system, node=node)
        sub_questions = await self._call(
            functools.partial(
                generate_sub_questions,
                self._llm,
                self._decompose_spec,
                messages,
                limit=MAX_SUB_QUESTIONS,
            )
        )
        if sub_questions is None:
            return (question,), "error"
        return (sub_questions, "ok") if sub_questions else ((question,), "fallback")

    async def judge(self, memory: WorkingMemory, *, system: str) -> Judgment:
        """모은 근거로 질문에 답할 수 있는지 판단한 결과를 돌려준다. 실패하면 status 가 error (모은 근거로 답하러 간다)."""
        node = self._library.get(JUDGE_PROMPT).render(max_queries=str(MAX_NEW_QUERIES))
        messages = self.node_messages(memory, system=system, node=node)
        judgment = await self._call(
            functools.partial(generate_judgment, self._llm, self._judge_spec, messages)
        )
        if judgment is None:
            return Judgment(sufficient=False, found="", missing="", queries=(), status="error")
        return judgment

    def node_messages(
        self, memory: WorkingMemory, *, system: str, node: str
    ) -> tuple[Message, ...]:
        """노드 하나에 보낼 대화를 돌려준다 (에이전트 칸 배치). 근거는 low 이상만, 검색어마다 순위를 돌아가며 고른다.

        Raises:
          BudgetError: 지시가 칸 예산을 넘을 때.
        """
        candidates: list[Candidate] = memory_candidates(
            memory, question=QUESTION, sub_questions=SUB_QUESTIONS, queries=QUERIES, notes=NOTES
        )
        candidates += ranked_candidates(memory, slot=DOCUMENTS, low=self._low)
        candidates += memory_candidates(memory, question=REMINDER)
        # 답 길이를 뺀 나머지가 칸 예산 (context-design §6) — 노드는 출력이 짧아 근거 칸 예산이 먼저 닿는다
        budget = context_budget(self._judge_spec) - JUDGE_MAX_TOKENS
        context = ContextAssembler(agent_layout(budget), CharTokenCounter()).assemble(
            {SYSTEM: system, NODE: node}, candidates
        )
        return context.messages

    async def _call[T](self, call: Callable[[], T]) -> T | None:
        """동기 LLM 호출을 작업 스레드에서 시간 제한 안에 돌린 결과를 돌려준다. 실패 · 시간 초과면 None (원인은 로그에만)."""
        try:
            async with asyncio.timeout(self._timeout_sec):
                return await asyncio.to_thread(call)
        except _FAILURES as e:
            logger.warning("agent node call failed: %s: %s", type(e).__name__, e)
            return None
