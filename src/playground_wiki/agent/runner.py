"""에이전트 실행: 질문 하나로 그래프를 돌려 진행 알림(`StepStarted`)을 흘리고 마지막에 결과(`AgentResult`)를 낸다."""

import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Self, cast

from langgraph.graph.state import CompiledStateGraph
from playground_llm import LLM, ModelSpec
from playground_retrieval.core import Hit
from playground_retrieval.search import SearchResult

from playground_wiki.agent.graph import build_graph
from playground_wiki.agent.nodes import AgentDeps, QueryScreen, Searcher
from playground_wiki.agent.reasoner import Reasoner
from playground_wiki.agent.state import (
    AgentState,
    DecomposeStatus,
    Judgment,
    ScreenHit,
    StepRecord,
    StepStarted,
    StopReason,
    initial_state,
)
from playground_wiki.generation.messages import context_budget

_STREAM_MODES = [
    "custom",
    "values",
]  # custom = 노드의 StepStarted, values = 노드가 끝날 때마다의 전체 상태


@dataclass(frozen=True)
class AgentResult:
    """질문 하나를 그래프로 돌린 결과. 답 생성은 부르는 쪽(답변 서비스)이 answer_hits 로 한다."""

    sub_questions: tuple[str, ...]
    queries: tuple[str, ...]
    """검색한 검색어 (하위 질문 + 새 검색어), 쓴 순서대로."""
    searches: int
    """검색 서비스를 부른 횟수."""
    iterations: int
    """끝낸 검색 반복 수 (1~MAX_ITERATIONS)."""
    evidence: int
    """모은 청크 수 (중복 없이, 점수와 상관없이)."""
    answer_hits: tuple[Hit, ...]
    """답변 근거로 고른 청크 (low 이상, 칸 예산 안). 비었으면 refuse."""
    decompose: DecomposeStatus
    judgments: tuple[Judgment, ...]
    stop: StopReason
    steps: tuple[StepRecord, ...]
    prompt_versions: tuple[str, ...]
    """노드 지시 프롬프트 `이름@버전:해시` (분해 · 판단)."""
    seed_hits: int = 0
    """원문 검색에서 이어 받은 청크 수."""
    blocked: ScreenHit | None = None
    """하위 질문 · 새 검색어가 걸린 관문 근거 (stop 이 gate_sub 일 때). 부르는 쪽이 그 규칙의 고정 답을 보낸다."""

    @property
    def search_ms(self) -> float:
        """검색 노드에 걸린 시간의 합 (ms)."""
        return round(sum(step.ms for step in self.steps if step.step == "search"), 1)

    def diagnostics(self) -> dict[str, object]:
        """진단 칸(done 이벤트 · 로그의 agent)을 돌려준다. 수와 상태만 — 하위 질문 · 검색어는 질문을 옮겨 담아 넣지 않는다 (ADR-027)."""
        timings: dict[str, float] = {}
        for step in self.steps:
            timings[step.step] = round(timings.get(step.step, 0.0) + step.ms, 1)
        return {
            "steps": [step.step for step in self.steps],
            "decompose": self.decompose,
            "sub_questions": len(self.sub_questions),
            "searches": self.searches,
            "iterations": self.iterations,
            "evidence": self.evidence,
            "answer_documents": len(self.answer_hits),
            "judgments": [_verdict(judgment) for judgment in self.judgments],
            "stop": self.stop,
            "prompt_versions": list(self.prompt_versions),
            "timings_ms": timings,
            "seed_hits": self.seed_hits,
        }


class AgentRunner:
    """에이전트 그래프를 질문마다 돌린다. 서버가 시작할 때 한 번 만든다.

    예)
        runner = AgentRunner.from_spec(search=search_service, llm=llm, spec=grounded.spec, low=0.06,
                                       max_query_chars=300)
        async for item in runner.run("「소나기」 작가가 다닌 대학을 세운 사람은?", system=base_text):
            ...  # StepStarted … 마지막에 AgentResult
    """

    def __init__(self, graph: CompiledStateGraph, *, prompt_versions: tuple[str, ...]) -> None:
        self._graph = graph
        self._prompt_versions = prompt_versions

    @classmethod
    def from_spec(
        cls,
        *,
        search: Searcher,
        llm: LLM,
        spec: ModelSpec,
        low: float,
        max_query_chars: int,
        clock: Callable[[], float] = time.perf_counter,
        screen: QueryScreen | None = None,
    ) -> Self:
        """답변 단계의 모델 설정 · 답변 정책의 low 로 노드 판단(`Reasoner`)과 그래프를 만든다.

        Args:
          search: 검색 툴 — 단발 /ask 와 같은 검색 서비스.
          llm: 노드 판단에 쓸 LLM 클라이언트 (답변 단계와 같은 것).
          spec: 답변 단계(grounded)의 모델 설정. 노드는 출력 길이 · temperature 만 바꿔 쓴다.
          low: 답변 정책의 low.
          max_query_chars: 검색어 길이 상한 (/search · /ask 질문 상한과 같은 값).
          clock: 노드 시간을 잴 시계 (초).
          screen: 하위 질문 · 새 검색어를 검색 전에 볼 관문 (원문 관문의 안전 규칙). None 이면 보지 않는다.

        Raises:
          PromptError: 노드 프롬프트가 없을 때.
        """
        reasoner = Reasoner.from_spec(llm, spec, low=low)
        deps = AgentDeps(
            search=search,
            reasoner=reasoner,
            low=low,
            budget=context_budget(spec),
            max_query_chars=max_query_chars,
            clock=clock,
            screen=screen,
        )
        return cls(build_graph(deps), prompt_versions=reasoner.prompt_versions)

    async def run(
        self, question: str, *, system: str, seed: SearchResult | None = None
    ) -> AsyncIterator[StepStarted | AgentResult]:
        """질문 하나의 진행 알림을 흘리고 마지막에 결과를 한 번 낸다.

        Args:
          question: 사용자 질문 원문.
          system: 노드 공통 고정 지시 (답변 프롬프트의 base 조각을 오늘 · 덤프 날짜로 채운 것).
          seed: 원문 질문으로 이미 찾은 검색 결과 — 근거에 이어 쓰고 원문은 다시 찾지 않는다.
        """
        hits = () if seed is None else seed.hits
        final: AgentState | None = None
        async for mode, chunk in self._graph.astream(
            initial_state(question, system=system, seed=hits), stream_mode=_STREAM_MODES
        ):
            if mode == "custom":
                yield cast("StepStarted", chunk)
            else:
                final = cast("AgentState", chunk)
        if final is None:
            raise RuntimeError("agent graph ended without a state")
        yield self._result(final)

    def _result(self, state: AgentState) -> AgentResult:
        memory = state["memory"]
        stop, decompose = state["stop"], state["decompose"]
        if stop is None or decompose is None:
            raise RuntimeError(
                f"agent graph ended before compose: {[s.step for s in state['steps']]}"
            )
        return AgentResult(
            sub_questions=memory.sub_questions,
            queries=memory.queries,
            searches=state["searches"],
            iterations=memory.iterations,
            evidence=len(memory.evidence),
            answer_hits=state["answer_hits"],
            decompose=decompose,
            judgments=state["judgments"],
            stop=stop,
            steps=state["steps"],
            prompt_versions=self._prompt_versions,
            seed_hits=state["seed_hits"],
            blocked=state["blocked"],
        )


def _verdict(judgment: Judgment) -> str:
    if judgment.status == "error":
        return "error"
    return "sufficient" if judgment.sufficient else "insufficient"
