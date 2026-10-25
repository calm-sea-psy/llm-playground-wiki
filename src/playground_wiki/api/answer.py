"""질문 하나에 답한다: 관문 → 검색 → 답변 정책 → 생성(캐스케이드)을 이어 SSE 이벤트로 흘린다 (ADR-024 · ADR-027).

관문 · 정책 · 생성은 하위 패키지마다 따로 있고, 셋을 잇는 곳은 레이어 규칙상 api 다 (generation 은 gate 를 모른다).
답 조각(delta)을 하나라도 보낸 뒤에는 단계 전환도 error 도 하지 않는다 — 지금까지 보낸 것으로 done 을 닫는다.
"""

import asyncio
import contextlib
import hashlib
import json
import logging
import threading
import time
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Literal, Protocol, cast

from playground_llm import (
    LLM,
    DoneReason,
    Generation,
    GenerationError,
    Message,
    ModelSpec,
    PromptLibrary,
    StreamDelta,
)
from playground_retrieval.core import Hit
from playground_retrieval.search import SearchResult

from playground_wiki.gate import QueryGate, fixed_reply
from playground_wiki.generation import (
    AnswerPolicy,
    CitationStream,
    MarkerReader,
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
STAGE_MESSAGES = {
    "gate": "질문 관문에서 고정 답을 보냅니다.",
    "grounded": "위키백과 문서에서 답을 찾는 중입니다.",
    "partial": "문서 내용과 일반 지식을 나눠 답하는 중입니다.",
    "refuse": "답할 근거가 되는 문서를 찾지 못했습니다.",
}
ERROR_MESSAGES = {
    "generation": "답을 만드는 중에 문제가 생겼습니다. 잠시 뒤 다시 시도해 주세요.",
    "timeout": "답을 만드는 데 시간이 너무 오래 걸렸습니다. 잠시 뒤 다시 시도해 주세요.",
}


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
class _Outcome:
    """한 단계 생성의 결과. 단계 전환 판단에 쓴다."""

    sent: bool = False
    """답 조각을 하나라도 보냈나 — 보냈으면 단계를 바꾸지 않는다."""
    marker: MarkerStatus | None = None
    done_reason: DoneReason | None = None
    """생성이 멈춘 이유. 시간 초과 · 실패 · 못 찾음으로 끊었거나 서버가 주지 않았으면 None."""
    truncated: bool = False
    """생성이 max_tokens 에 닿아 멈췄다 (`is_truncated`)."""
    timed_out: bool = False
    error: str | None = None
    cited: tuple[int, ...] = ()
    dropped: tuple[int, ...] = ()
    first_delta_ms: float | None = None
    spec_id: str = ""
    prompt_version: str = ""


@dataclass
class _Trace:
    """요청 하나의 진단. done 의 diagnostics 이자 로그 한 줄 (ADR-027)."""

    question_chars: int
    question_sha256: str
    started: float = 0.0
    stage_path: list[str] = field(default_factory=list)
    gate: dict[str, object] | None = None
    top_score: float | None = None
    marker: MarkerStatus | None = None
    done_reasons: dict[str, DoneReason | None] = field(default_factory=dict)
    """단계 → 생성이 멈춘 이유. 로그 한 줄에만 싣는다 (done 의 diagnostics 필드는 그대로)."""
    spec_id: str | None = None
    prompt_version: str | None = None
    search_ms: float | None = None
    first_delta_ms: float | None = None
    total_ms: float | None = None


class AnswerService:
    """질문을 받아 SSE 이벤트를 흘린다. 서버가 시작할 때 한 번 만들어 요청마다 `answer` 를 부른다.

    예)
        service = AnswerService(search=search_service, policy=policy, library=library, llm=llm,
                                gate=gate, dump_date="2026년 10월 1일")
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
        self._today = today
        self._clock = clock

    async def answer(self, question: str) -> AsyncIterator[AnswerEvent]:
        """질문 하나의 이벤트: stage → (sources) → delta… → done, 또는 아무것도 못 보낸 실패면 error."""
        start = self._clock()
        trace = _Trace(len(question), hashlib.sha256(question.encode("utf-8")).hexdigest(), start)
        try:
            async for event in self._answer(question, trace, start):
                yield event
        finally:
            trace.total_ms = _ms(self._clock() - start)
            extra: dict[str, object] = {"done_reason": dict(trace.done_reasons)}
            if self._log_questions:
                extra["question"] = question
            logger.info("ask %s", json.dumps(_diagnostics(trace) | extra, ensure_ascii=False))

    async def _answer(
        self, question: str, trace: _Trace, start: float
    ) -> AsyncIterator[AnswerEvent]:
        if self._gate is not None and (match := self._gate.check(question)) is not None:
            trace.stage_path.append("gate")
            trace.gate = {"rule": match.rule, "kind": match.kind, "matched": list(match.matched)}
            yield _stage("gate")
            yield AnswerEvent("delta", {"text": fixed_reply(match, dump_date=self._dump_date)})
            yield _done(trace, None, self._clock())
            return
        searched = self._clock()
        result = await self._search.search(question)
        trace.search_ms = _ms(self._clock() - searched)
        decision = self._policy.decide(result.hits)
        trace.top_score = decision.top_score
        if decision.profile is None:
            trace.stage_path.append("refuse")
            yield _stage("refuse")
            yield AnswerEvent("delta", {"text": REFUSAL.format(query=question)})
            yield _done(trace, None, self._clock())
            return
        yield AnswerEvent("sources", {"sources": [asdict(s) for s in sources(decision.context)]})
        stage: Stage = decision.stage
        message = STAGE_MESSAGES[stage]
        while True:
            trace.stage_path.append(stage)
            yield _stage(stage, message)
            outcome = _Outcome()
            async for event in self._generate(stage, question, decision.context, outcome, start):
                yield event
            trace.marker = trace.marker or outcome.marker
            trace.done_reasons[stage] = outcome.done_reason
            trace.spec_id, trace.prompt_version = outcome.spec_id, outcome.prompt_version
            trace.first_delta_ms = trace.first_delta_ms or outcome.first_delta_ms
            if outcome.sent:
                yield _done(trace, outcome, self._clock())
                return
            if outcome.error is not None:
                yield _error("generation", outcome.error)
                return
            # 못 찾음 · 빈 답 · 표지 전에 잘림 · 시간 초과 — 아무것도 안 보냈으니 partial 로
            if stage == "grounded":
                stage, message = "partial", CASCADE_MESSAGE
                continue
            if outcome.timed_out:
                yield _error("timeout", "partial timed out before any answer text")
                return
            yield AnswerEvent(
                "delta", {"text": REFUSAL.format(query=question)}
            )  # 마지막 단계의 빈 답
            # 화면에 나간 글은 생성이 아니라 고정 문구다 — 관문 · refuse 처럼 잘림 · 출처 번호를 싣지 않는다
            # (한도에 닿았다는 사실은 로그의 done_reason 에 남는다. 시간 초과는 위에서 error 로 끝났다)
            yield _done(trace, None, self._clock())
            return

    async def _generate(
        self, stage: Stage, question: str, context: Sequence[Hit], outcome: _Outcome, start: float
    ) -> AsyncIterator[AnswerEvent]:
        """한 단계를 생성하며 답 조각을 흘리고, 결과를 outcome 에 적는다."""
        profile: Profile = self._policy.profiles[stage]
        prompt = build_prompt(
            profile,
            self._library,
            question=question,
            context=context,
            today=korean_date(self._today()),
            dump_date=self._dump_date,
        )
        outcome.spec_id, outcome.prompt_version = profile.spec.spec_id, prompt.version
        reader = MarkerReader() if stage == "grounded" else None
        citations = CitationStream(len(context))
        deadline = self._clock() + profile.timeout_sec
        stream = _ThreadedStream(self._llm, profile.spec, prompt.messages)
        final: Generation | None = None
        try:
            while True:
                # 단계 전체의 벽시계 제한 — 남은 시간이 없으면 바로 TimeoutError
                async with asyncio.timeout(max(0.0, deadline - self._clock())):
                    item = await stream.next()
                if item is None:
                    break
                if isinstance(item, Generation):
                    final = item
                    continue
                if item.kind != "text":
                    continue  # thinking 조각은 화면에 보내지 않는다
                text = item.text
                if reader is not None:
                    if reader.result is None:
                        if (marked := reader.feed(text)) is None:
                            continue
                        outcome.marker = marked.status
                        if marked.status == "not_found":
                            return  # finally 가 응답 스트림을 닫는다
                        text = marked.rest
                    else:
                        text = reader.body(text)
                for event in self._deltas(citations.feed(text), outcome, start):
                    yield event
            if final is not None:
                outcome.done_reason, outcome.truncated = final.done_reason, is_truncated(final)
            if reader is not None and reader.result is None:
                marked = reader.finish()
                outcome.marker = marked.status
                # 표지를 정하기 전에 한도에 닿아 끝났으면 "표지 없으면 찾음"을 쓰지 않는다 —
                # 붙잡아 둔 글은 끊긴 표지일 수 있으니 못 찾음처럼 아무것도 보내지 않고 partial 로 (ADR-024)
                if marked.status == "not_found" or (
                    marked.status == "missing" and outcome.truncated
                ):
                    return
                for event in self._deltas(citations.feed(marked.rest), outcome, start):
                    yield event
            for event in self._deltas(citations.finish(), outcome, start):
                yield event
        except TimeoutError:
            outcome.timed_out = True
            for event in self._flush(citations, outcome, start):
                yield event
        except GenerationError as e:
            outcome.error = str(e)
            for event in self._flush(citations, outcome, start):
                yield event
        finally:
            stream.stop()
            outcome.cited, outcome.dropped = citations.cited, citations.dropped

    def _flush(
        self, citations: CitationStream, outcome: _Outcome, start: float
    ) -> list[AnswerEvent]:
        # 중간에 끊겼어도 이미 답을 보내고 있었다면 붙잡아 둔 꼬리(공백 · 출처 번호)까지 내보낸다
        return self._deltas(citations.finish(), outcome, start) if outcome.sent else []

    def _deltas(self, text: str, outcome: _Outcome, start: float) -> list[AnswerEvent]:
        # 첫 글자가 나오기 전의 공백만 남은 조각은 보내지 않는다 — 보내면 "답을 보냈다"가 되어 단계를 못 바꾼다
        if not text or (not outcome.sent and not text.strip()):
            return []
        if not outcome.sent:
            outcome.sent = True
            outcome.first_delta_ms = _ms(self._clock() - start)
        return [AnswerEvent("delta", {"text": text})]


class _ThreadedStream:
    """동기 `llm.stream` 을 작업 스레드에서 돌려 조각을 asyncio 큐로 넘긴다.

    stop() 하면 작업 스레드가 다음 조각을 받는 즉시 응답 스트림을 닫는다 — 큐 소비만 멈추면 스레드와 GPU 가
    계속 생성한다. 스트림 객체는 그 스레드 안에서만 다룬다 (돌고 있는 제너레이터를 다른 스레드에서 닫을 수 없다).
    """

    _END = object()

    def __init__(self, llm: LLM, spec: ModelSpec, messages: Sequence[Message]) -> None:
        self._loop = asyncio.get_running_loop()
        self._queue: asyncio.Queue[object] = asyncio.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(llm, spec, messages), daemon=True)
        self._thread.start()

    def _run(self, llm: LLM, spec: ModelSpec, messages: Sequence[Message]) -> None:
        stream = llm.stream(spec, messages)
        try:
            for item in stream:
                if self._stop.is_set():
                    break
                self._put(item)
        except Exception as e:  # noqa: BLE001 — 받는 쪽(이벤트 루프)에 그대로 넘긴다
            self._put(e)
        finally:
            stream.close()  # 응답 스트림을 닫아 Ollama 쪽 생성을 끊는다
            self._put(self._END)

    def _put(self, item: object) -> None:
        # 서버가 내려가 이벤트 루프가 닫혔으면 받을 쪽이 없다
        with contextlib.suppress(RuntimeError):
            self._loop.call_soon_threadsafe(self._queue.put_nowait, item)

    async def next(self) -> StreamDelta | Generation | None:
        """다음 조각. 끝났으면 None. 시간 제한은 부르는 쪽이 `asyncio.timeout` 으로 건다.

        Raises:
          GenerationError: 생성이 실패했을 때 (작업 스레드에서 난 것을 그대로).
        """
        item = await self._queue.get()
        if item is self._END:
            return None
        if isinstance(item, Exception):
            raise item
        return cast("StreamDelta | Generation", item)

    def stop(self) -> None:
        """생성을 끊는다 (작업 스레드가 다음 조각에서 스트림을 닫는다)."""
        self._stop.set()


def is_truncated(generation: Generation) -> bool:
    """생성이 max_tokens 에 닿아 멈췄나 — 서버가 준 done_reason 이 length 인지.

    서버가 done_reason 을 주지 않았으면(None) 생성 토큰 수 ≥ max_tokens 로 짐작한다. None 을 잘리지 않음으로
    보면 그런 서버에서 잘림을 놓친다. stop 이면 토큰 수가 한도와 같아도 잘리지 않은 것이다.
    """
    if generation.done_reason is not None:
        return generation.done_reason == "length"
    max_tokens = generation.model.params.get("max_tokens")
    return max_tokens is not None and generation.usage.completion_tokens >= int(max_tokens)


def _stage(stage: str, message: str | None = None) -> AnswerEvent:
    return AnswerEvent("stage", {"stage": stage, "message": message or STAGE_MESSAGES[stage]})


def _done(trace: _Trace, outcome: _Outcome | None, now: float) -> AnswerEvent:
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


def _error(code: str, detail: str) -> AnswerEvent:
    logger.warning("ask error %s: %s", code, detail)
    return AnswerEvent("error", {"code": code, "message": ERROR_MESSAGES[code]})


def _diagnostics(trace: _Trace) -> dict[str, object]:
    return {
        "stage_path": list(trace.stage_path),
        "gate": trace.gate,
        "top_score": trace.top_score,
        "marker": trace.marker,
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


def _ms(seconds: float) -> float:
    return round(seconds * 1000, 1)
