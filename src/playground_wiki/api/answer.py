"""질문 하나에 답한다: 관문 → 검색 → 답변 정책 → 생성(캐스케이드)을 이어 SSE 이벤트로 흘린다 (ADR-024 · ADR-027).

관문 · 정책 · 생성은 하위 패키지마다 따로 있고, 셋을 잇는 곳은 레이어 규칙상 api 다 (generation 은 gate 를 모른다).
단계 하나의 생성은 `api.stage` 가 맡고, 여기서는 그 결과로 다음 단계를 고르고 이벤트 · 진단을 만든다.
답 조각(delta)을 하나라도 보낸 뒤에는 단계 전환도 error 도 하지 않는다 — 지금까지 보낸 것으로 done 을 닫는다.
"""

import hashlib
import hmac
import json
import logging
import secrets
import time
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
from types import MappingProxyType
from typing import Literal, Protocol, Self

from playground_llm import LLM, DoneReason, PromptLibrary
from playground_retrieval.core import Hit
from playground_retrieval.search import SearchResult

from playground_wiki.api.config import AnswerSetup
from playground_wiki.api.stage import StageOutcome, generate_stage
from playground_wiki.gate import GateMatch, QueryGate, fixed_reply
from playground_wiki.generation import (
    AnswerPolicy,
    ChatPrompt,
    Decision,
    MarkerStatus,
    Profile,
    Stage,
    build_prompt,
    korean_date,
    sources,
)

logger = logging.getLogger(__name__)

EventName = Literal["stage", "sources", "delta", "done", "error"]
REFUSAL = "주어진 위키백과 문서에서 이 질문에 답할 근거를 찾지 못했습니다. 시도한 검색어: «{query}»"
"""refuse 단계 · 마지막 단계가 빈 답일 때의 고정 문구. 질문을 바꿔 다시 검색하지 않으니 검색어는 질문 그대로."""
CASCADE_MESSAGE = "문서에서 답을 찾지 못해 일반 지식으로 답하는 중입니다."
STAGE_MESSAGES: Mapping[str, str] = MappingProxyType(
    {
        "gate": "질문 관문에서 고정 답을 보냅니다.",
        "grounded": "위키백과 문서에서 답을 찾는 중입니다.",
        "partial": "문서 내용과 일반 지식을 나눠 답하는 중입니다.",
        "refuse": "답할 근거가 되는 문서를 찾지 못했습니다.",
    }
)
"""stage 이벤트 이름(관문 + 답변 정책 단계 `Stage`) → 화면에 보일 문구."""
# 로그의 질문 해시에 섞는 salt 길이. 서비스를 만들 때(서버 실행마다) 새로 뽑는다 — salt 없는 SHA-256 은
# 짧거나 틀이 알려진 질문을 되짚을 수 있다 (보안 점검 2026-10-25). 같은 실행 안에서만 같은 질문을 묶을 수 있다
QUESTION_SALT_BYTES = 16
ErrorCode = Literal["generation", "timeout"]
"""error 이벤트의 code."""
ERROR_MESSAGES: Mapping[ErrorCode, str] = MappingProxyType(
    {
        "generation": "답을 만드는 중에 문제가 생겼습니다. 잠시 뒤 다시 시도해 주세요.",
        "timeout": "답을 만드는 데 시간이 너무 오래 걸렸습니다. 잠시 뒤 다시 시도해 주세요.",
    }
)
"""error code → 화면에 보일 문구 (서버 사정은 로그에만)."""


@dataclass(frozen=True)
class AnswerEvent:
    """SSE 이벤트 하나."""

    name: EventName
    data: dict[str, object]

    def to_sse(self) -> str:
        """`event:` · `data:` 줄로 (JSON 한 줄)."""
        return f"event: {self.name}\ndata: {json.dumps(self.data, ensure_ascii=False)}\n\n"


class Searcher(Protocol):
    """질문으로 청크를 찾는다 (SearchService)."""

    async def search(self, query: str) -> SearchResult:
        """리랭커 점수가 붙은 상위 청크."""
        ...


@dataclass
class _Trace:
    """요청 하나의 진단. done 의 diagnostics 이자 로그 한 줄 (ADR-027).

    요청이 진행되며 채운다 — 클라이언트가 중간에 끊어도 그때까지의 진단을 로그로 남겨야 해서 결과 객체가 아니다.
    """

    question_chars: int
    question_sha256: str
    """질문의 HMAC-SHA256 (키 = 실행마다 새로 뽑는 salt). 질문 원문은 남기지 않는다."""
    started: float = 0.0
    stage_path: list[str] = field(default_factory=list)
    gate: GateMatch | None = None
    top_score: float | None = None
    marker: MarkerStatus | None = None
    done_reasons: dict[str, DoneReason | None] = field(default_factory=dict)
    """단계 → 생성이 멈춘 이유 (stop · length, 끊었거나 서버가 주지 않았으면 None). 단계마다 따로라 grounded 가
    잘려 partial 로 넘어간 것도 남는다 — done 의 truncated 는 마지막 단계 것뿐이다."""
    spec_id: str | None = None
    prompt_version: str | None = None
    search_ms: float | None = None
    first_delta_ms: float | None = None
    total_ms: float | None = None


class AnswerService:
    """질문을 받아 SSE 이벤트를 흘린다. 서버가 시작할 때 한 번 만들어 요청마다 `answer` 를 부른다.

    관문(`check_gate`) · 단계 프롬프트(`stage_prompt`) · 단계 생성(`api.stage.generate_stage`) ·
    다음 단계 고르기(`next_stage`)는 따로 부를 수 있다 — 에이전트 경로가 같은 조각을 쓴다.

    예)
        setup = AnswerSetup.from_config(ApiConfig.from_env(), tokenizer=KiwiTokenizer(kiwi))
        service = AnswerService.from_config(setup, search=search_service, llm=llm)
        async for event in service.answer("경복궁을 중건한 사람은?"):
            send(event.to_sse())
    """

    def __init__(
        self,
        *,
        search: Searcher,
        policy: AnswerPolicy,
        library: PromptLibrary,
        llm: LLM,
        gate: QueryGate | None,
        dump_date: str,
        log_questions: bool = False,
        question_salt: bytes | None = None,
        today: Callable[[], date] = date.today,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._search = search
        self._policy = policy
        self._library = library
        self._llm = llm
        self._gate = gate
        self._dump_date = dump_date
        self._log_questions = log_questions
        self._question_salt = (
            secrets.token_bytes(QUESTION_SALT_BYTES) if question_salt is None else question_salt
        )
        self._today = today
        self._clock = clock

    @classmethod
    def from_config(cls, setup: AnswerSetup, *, search: Searcher, llm: LLM) -> Self:
        """설정 폴더에서 읽은 답변 구성에 검색기 · LLM 을 붙여 만든다 — 서버와 평가 스크립트가 같은 조립을 쓴다.

        Args:
          setup: 정책 · 프롬프트 · 관문 · 덤프 날짜 · 질문 기록 설정 (`AnswerSetup.from_config`).
          search: 검색기. 서버는 SearchService, 평가는 그것을 감싼 기록용 검색기 · 저장 문맥 검색기.
          llm: 생성 클라이언트. 서버는 OllamaLLM, 평가는 메시지를 고쳐 쓰는 래퍼 · 시험의 가짜.
        """
        return cls(
            search=search,
            policy=setup.policy,
            library=setup.library,
            llm=llm,
            gate=setup.gate,
            dump_date=setup.dump_date,
            log_questions=setup.log_questions,
        )

    async def answer(self, question: str) -> AsyncIterator[AnswerEvent]:
        """질문 하나의 이벤트: stage → (sources) → delta… → done, 또는 아무것도 못 보낸 실패면 error."""
        start = self._clock()
        trace = _Trace(len(question), question_digest(self._question_salt, question), start)
        try:
            async for event in self._answer(question, trace, start):
                yield event
        finally:
            trace.total_ms = _ms(self._clock() - start)
            extra = {"question": question} if self._log_questions else {}
            logger.info("ask %s", json.dumps(_diagnostics(trace) | extra, ensure_ascii=False))

    def check_gate(self, question: str) -> GateMatch | None:
        """관문에 걸린 근거. 걸리지 않았거나 관문이 꺼져 있으면 None (검색으로)."""
        return None if self._gate is None else self._gate.check(question)

    def stage_prompt(self, profile: Profile, question: str, context: Sequence[Hit]) -> ChatPrompt:
        """단계 프로필 · 질문 · 문맥으로 LLM 에 보낼 대화. 오늘 날짜 · 덤프 날짜를 넣는다."""
        return build_prompt(
            profile,
            self._library,
            question=question,
            context=context,
            today=korean_date(self._today()),
            dump_date=self._dump_date,
        )

    async def _answer(
        self, question: str, trace: _Trace, start: float
    ) -> AsyncIterator[AnswerEvent]:
        if (match := self.check_gate(question)) is not None:
            trace.gate = match
            for event in self._fixed("gate", fixed_reply(match, dump_date=self._dump_date), trace):
                yield event
            return
        searched = self._clock()
        result = await self._search.search(question)
        trace.search_ms = _ms(self._clock() - searched)
        decision = self._policy.decide(result.hits)
        trace.top_score = decision.top_score
        if decision.profile is None:
            for event in self._fixed("refuse", REFUSAL.format(query=question), trace):
                yield event
            return
        yield AnswerEvent("sources", {"sources": [asdict(s) for s in sources(decision.context)]})
        async for event in self._cascade(question, decision, trace, start):
            yield event

    async def _cascade(
        self, question: str, decision: Decision, trace: _Trace, start: float
    ) -> AsyncIterator[AnswerEvent]:
        """정책이 고른 단계부터 생성한다. 아무것도 보내지 못했으면 `next_stage` 로 넘어간다."""
        stage: Stage = decision.stage
        message = STAGE_MESSAGES[stage]
        while True:
            trace.stage_path.append(stage)
            yield _stage(stage, message)
            profile = self._policy.profiles[stage]
            prompt = self.stage_prompt(profile, question, decision.context)
            outcome: StageOutcome | None = None
            first_delta_ms: float | None = None
            async for item in generate_stage(
                self._llm,
                profile,
                prompt,
                context_count=len(decision.context),
                read_marker=stage == "grounded",
                clock=self._clock,
            ):
                if isinstance(item, StageOutcome):
                    outcome = item
                    continue
                if first_delta_ms is None:
                    first_delta_ms = _ms(self._clock() - start)
                yield _delta(item)
            if outcome is None:
                raise RuntimeError(f"stage {stage!r} ended without an outcome")
            trace.marker = trace.marker or outcome.marker
            trace.done_reasons[stage] = outcome.done_reason
            trace.spec_id, trace.prompt_version = profile.spec.spec_id, prompt.version
            trace.first_delta_ms = trace.first_delta_ms or first_delta_ms
            following = next_stage(stage, outcome)
            if following is None:
                for event in self._close(question, outcome, trace):
                    yield event
                return
            stage, message = following, CASCADE_MESSAGE

    def _fixed(self, stage: str, text: str, trace: _Trace) -> Iterator[AnswerEvent]:
        """LLM 없이 고정 문구로 끝나는 길(관문 · refuse): stage → delta → done."""
        trace.stage_path.append(stage)
        yield _stage(stage)
        yield _delta(text)
        yield _done(trace, None, self._clock())

    def _close(self, question: str, outcome: StageOutcome, trace: _Trace) -> Iterator[AnswerEvent]:
        """마지막으로 생성한 단계의 결과로 요청을 닫는다: 보낸 답 → done, 실패 · 시간 초과 → error, 빈 답 → 고정 문구."""
        if outcome.sent:
            yield _done(trace, outcome, self._clock())
        elif outcome.error is not None:
            yield _error("generation", outcome.error)
        elif outcome.timed_out:
            yield _error("timeout", "partial timed out before any answer text")
        else:
            yield _delta(REFUSAL.format(query=question))  # 마지막 단계의 빈 답
            # 화면에 나간 글은 생성이 아니라 고정 문구다 — 관문 · refuse 처럼 잘림 · 출처 번호를 싣지 않는다
            # (한도에 닿았다는 사실은 diagnostics 의 done_reason 에 남는다)
            yield _done(trace, None, self._clock())


def next_stage(stage: Stage, outcome: StageOutcome) -> Stage | None:
    """생성을 마친 단계 다음에 생성할 단계. 이 단계로 요청을 닫아야 하면 None.

    답을 보냈거나 생성이 실패했으면 단계를 바꾸지 않는다. grounded 가 못 찾음 · 빈 답 · 표지 전에 잘림 ·
    시간 초과로 아무것도 보내지 않았으면 partial 로 넘어간다 (캐스케이드). partial 다음 단계는 없다.
    """
    if outcome.sent or outcome.error is not None or stage != "grounded":
        return None
    return "partial"


def question_digest(salt: bytes, question: str) -> str:
    """로그에 남기는 질문 해시 — salt 를 키로 한 HMAC-SHA256 (16진수 64자)."""
    return hmac.new(salt, question.encode("utf-8"), hashlib.sha256).hexdigest()


def _stage(stage: str, message: str | None = None) -> AnswerEvent:
    return AnswerEvent("stage", {"stage": stage, "message": message or STAGE_MESSAGES[stage]})


def _delta(text: str) -> AnswerEvent:
    return AnswerEvent("delta", {"text": text})


def _done(trace: _Trace, outcome: StageOutcome | None, now: float) -> AnswerEvent:
    trace.total_ms = _ms(now - trace.started)
    data: dict[str, object] = {
        "stage_path": list(trace.stage_path),
        "cited": list(outcome.cited) if outcome else [],
        "dropped": list(outcome.dropped) if outcome else [],
        "truncated": bool(outcome and outcome.truncated),
        "timed_out": bool(outcome and outcome.timed_out),
        "diagnostics": _diagnostics(trace),
    }
    return AnswerEvent("done", data)


def _error(code: ErrorCode, detail: str) -> AnswerEvent:
    logger.warning("ask error %s: %s", code, detail)
    return AnswerEvent("error", {"code": code, "message": ERROR_MESSAGES[code]})


def _diagnostics(trace: _Trace) -> dict[str, object]:
    return {
        "stage_path": list(trace.stage_path),
        "gate": None if trace.gate is None else _gate_diagnostics(trace.gate),
        "top_score": trace.top_score,
        "marker": trace.marker,
        "done_reason": dict(trace.done_reasons),
        "spec_id": trace.spec_id,
        "prompt_version": trace.prompt_version,
        "question_chars": trace.question_chars,
        "question_sha256": trace.question_sha256,
        "timings_ms": {
            "search": trace.search_ms,
            "first_delta": trace.first_delta_ms,
            "total": trace.total_ms,
        },
    }


def _gate_diagnostics(match: GateMatch) -> dict[str, object]:
    # 근거는 규칙 사전의 말뿐이다 (잡담은 사전 말과 같아진 정규화 질문) — 질문 원문이 섞이지 않는다 (ADR-027)
    return {"rule": match.rule, "kind": match.kind, "matched": list(match.matched)}


def _ms(seconds: float) -> float:
    return round(seconds * 1000, 1)
