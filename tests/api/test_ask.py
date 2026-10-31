import asyncio
import hashlib
import json
import logging
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import replace
from datetime import date
from typing import get_args

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from playground_corpus import Chunk
from playground_llm import (
    DoneReason,
    FakeLLM,
    Generation,
    GenerationError,
    Message,
    ModelNotFoundError,
    ModelSpec,
    PromptLibrary,
    StreamDelta,
    Usage,
)
from playground_retrieval.core import Hit
from playground_retrieval.search import SearchResult

from playground_wiki.api.answer import (
    CASCADE_MESSAGE,
    ERROR_MESSAGES,
    REFUSAL,
    STAGE_MESSAGES,
    AnswerEvent,
    AnswerService,
    ErrorCode,
    question_digest,
)
from playground_wiki.api.app import create_app
from playground_wiki.api.config import AnswerSetup
from playground_wiki.api.schemas import MAX_QUERY_CHARS
from playground_wiki.gate import QueryGate, Token
from playground_wiki.generation import PROMPT_DIR, AnswerPolicy, Profile, Stage, Thresholds

LIBRARY = PromptLibrary.load(PROMPT_DIR)
SPEC = ModelSpec("ollama", "fake", {"think": "off", "max_tokens": 64})
BUDGET = int(SPEC.params["max_tokens"])
GROUNDED = Profile(SPEC, system=("base@v4", "grounded@v2"), user="question@v1", timeout_sec=5)
PARTIAL = Profile(SPEC, system=("base@v4", "partial@v2"), user="question@v1", timeout_sec=5)
QUESTION = "경복궁을 중건한 사람은?"
PRIVATE = "010-1234-5678"
"""질문에 섞인 개인정보 — 로그 · done 진단 · error 이벤트 어디에도 나오면 안 된다 (ADR-027)."""
SERVER_DETAIL = "fake: ConnectError: http://10.0.0.5:11434"
"""LLM 호출 실패의 원인. 서버 로그 WARNING 에만 남고 클라이언트에는 화면 문구만 간다."""
# done 진단 항목 — 늘리면 질문 원문 · 비밀값이 새 항목으로 들어가지 않았는지 보고 여기에 더한다 (ADR-027)
DIAGNOSTICS_KEYS = frozenset(
    {
        "stage_path",
        "gate",
        "top_score",
        "marker",
        "done_reason",
        "spec_id",
        "prompt_version",
        "question_chars",
        "question_sha256",
        "timings_ms",
    }
)


class _NoTokens:
    """형태소가 없는 분석기 — 실시간 규칙은 걸리지 않고 잡담 사전만 본다 (Kiwi 를 올리지 않는다)."""

    def tokenize(self, text: str) -> list[Token]:
        return []


class _Words:
    """띄어쓰기로 나눈 낱말을 일반 명사로 — Kiwi 없이 실시간 규칙(때 표지 + 대상어)을 건다."""

    def tokenize(self, text: str) -> list[Token]:
        return [Token(word, "NNG") for word in text.split()]


class FakeSearch:
    def __init__(self, score: float | None) -> None:
        self.score = score
        self.calls: list[str] = []

    async def search(self, query: str) -> SearchResult:
        self.calls.append(query)
        if self.score is None:
            return SearchResult(hits=(), took_ms=1.0)
        chunk = Chunk(
            id="1-0",
            doc_id="1",
            title="경복궁",
            section="역사",
            text="흥선대원군이 중건했다.",
            seq=0,
            token_count=4,
        )
        return SearchResult(hits=(Hit(chunk=chunk, score=self.score),), took_ms=1.0)


def _by_stage(
    grounded: str, partial: str = "### 위키백과 문서에서\n직접적인 내용은 없습니다."
) -> FakeLLM:
    """system 프롬프트로 단계를 알아보고 단계마다 정해 둔 답을 낸다."""
    return FakeLLM(
        [lambda spec, messages: grounded if "근거 답변" in messages[0].content else partial]
    )


def _service(
    llm: object, score: float | None = 0.95, **kwargs: object
) -> tuple[AnswerService, FakeSearch]:
    search = FakeSearch(score)
    policy = AnswerPolicy(Thresholds(high=0.9, low=0.5), {"grounded": GROUNDED, "partial": PARTIAL})
    service = AnswerService(
        search=search,
        policy=policy,
        library=LIBRARY,
        llm=llm,
        gate=QueryGate(_NoTokens()),
        dump_date="2026년 10월 1일",
        today=lambda: date(2026, 10, 24),
        **kwargs,
    )
    return service, search


def _events(service: AnswerService, question: str = QUESTION) -> list[AnswerEvent]:
    async def collect() -> list[AnswerEvent]:
        return [event async for event in service.answer(question)]

    return asyncio.run(collect())


def _names(events: list[AnswerEvent]) -> list[str]:
    return [event.name for event in events]


def _text(events: list[AnswerEvent]) -> str:
    return "".join(str(event.data["text"]) for event in events if event.name == "delta")


def _done(events: list[AnswerEvent]) -> dict:
    return next(event.data for event in events if event.name == "done")


def test_messages_cover_every_stage_and_error_code():
    assert set(STAGE_MESSAGES) == {"gate", *get_args(Stage)}
    assert set(ERROR_MESSAGES) == set(get_args(ErrorCode))


def test_smalltalk_is_answered_by_the_gate_without_search():
    service, search = _service(_by_stage("찾음\n답"))

    events = _events(service, "안녕!")

    assert _names(events) == ["stage", "delta", "done"]
    assert events[0].data["stage"] == "gate"
    assert search.calls == []
    assert _done(events)["diagnostics"]["gate"]["kind"] == "greeting"


def test_no_hits_is_a_fixed_refusal_with_the_query():
    service, _ = _service(_by_stage("찾음\n답"), score=None)

    events = _events(service)

    assert _names(events) == ["stage", "delta", "done"]
    assert events[0].data["stage"] == "refuse"
    assert _text(events) == REFUSAL.format(query=QUESTION)


def test_grounded_answer_drops_the_marker_and_out_of_range_citations():
    service, _ = _service(_by_stage("찾음\n흥선대원군이 중건했다 [1][7]."))

    events = _events(service)

    assert _names(events)[:3] == ["sources", "stage", "delta"]
    assert events[0].data["sources"][0]["title"] == "경복궁"
    assert _text(events) == "흥선대원군이 중건했다 [1]."
    done = _done(events)
    assert (done["stage_path"], done["cited"], done["dropped"]) == (["grounded"], [1], [7])
    assert done["diagnostics"]["marker"] == "found"


def test_not_found_cascades_to_partial_before_any_text():
    llm = _by_stage("못 찾음", partial="### 위키백과 문서에서\n직접적인 내용은 없습니다.")
    service, _ = _service(llm)

    events = _events(service)

    stages = [event.data for event in events if event.name == "stage"]
    assert [s["stage"] for s in stages] == ["grounded", "partial"]
    assert stages[1]["message"] == CASCADE_MESSAGE
    assert _text(events).startswith("### 위키백과 문서에서")
    assert _done(events)["stage_path"] == ["grounded", "partial"]
    assert len(llm.calls) == 2


def test_empty_grounded_answer_also_cascades():
    service, _ = _service(_by_stage("찾음\n\n"))

    assert _done(_events(service))["stage_path"] == ["grounded", "partial"]


def test_empty_partial_answer_ends_with_the_fixed_refusal():
    service, _ = _service(
        _by_stage("찾음\n답", partial=""), score=0.6
    )  # low ≤ 0.6 < high → partial

    events = _events(service)

    assert _done(events)["stage_path"] == ["partial"]
    assert _text(events) == REFUSAL.format(query=QUESTION)
    assert _done(events)["diagnostics"]["timings_ms"]["first_delta"] is None


class ScriptedLLM:
    """조각을 정해 둔 대로(사이에 쉬기도 하며) 흘리는 대역. 응답 스트림이 닫혔는지 기록한다.

    끝에 보내는 Generation 의 done_reason · 생성 토큰 수도 정한다 (FakeLLM 은 늘 stop).
    """

    def __init__(
        self,
        pieces: Sequence[str | float | Exception],
        *,
        done_reason: DoneReason | None = None,
        completion_tokens: int = 1,
    ) -> None:
        self.pieces = pieces
        self.done_reason = done_reason
        self.completion_tokens = completion_tokens
        self.closed = threading.Event()
        self.produced = 0

    def stream(
        self, spec: ModelSpec, messages: Sequence[Message]
    ) -> Iterator[StreamDelta | Generation]:
        try:
            for piece in self.pieces:
                if isinstance(piece, float):
                    time.sleep(piece)
                elif isinstance(piece, Exception):
                    raise piece
                else:
                    self.produced += 1
                    yield StreamDelta("text", piece)
            yield Generation(
                text="",
                model=spec,
                usage=Usage(prompt_tokens=1, completion_tokens=self.completion_tokens, total_ms=0),
                done_reason=self.done_reason,
            )
        finally:
            self.closed.set()


def test_failure_before_any_text_is_an_error_event():
    service, _ = _service(ScriptedLLM([GenerationError("server down")]))

    events = _events(service)

    assert _names(events)[-1] == "error"
    assert "done" not in _names(events)


def test_missing_model_is_an_error_event_not_a_dropped_stream():
    llm = ScriptedLLM([ModelNotFoundError("model 'gemma4:12b' not found")])
    service, _ = _service(llm)

    events = _events(service)

    assert _names(events)[-1] == "error"
    assert events[-1].data["code"] == "generation"
    assert "gemma4" not in json.dumps(events[-1].data, ensure_ascii=False)  # 서버 사정은 로그에만
    assert "done" not in _names(events)


def test_after_text_is_sent_a_timeout_closes_with_done_not_a_new_stage():
    slow = ScriptedLLM(["찾음\n", "흥선대원군이 ", 0.5, "중건했다."])
    profile = Profile(SPEC, system=GROUNDED.system, user=GROUNDED.user, timeout_sec=0.2)
    service, _ = _service(slow)
    service._policy = AnswerPolicy(
        Thresholds(high=0.9, low=0.5), {"grounded": profile, "partial": PARTIAL}
    )

    events = _events(service)

    assert [e.data["stage"] for e in events if e.name == "stage"] == [
        "grounded"
    ]  # partial 로 넘어가지 않는다
    assert "error" not in _names(events)
    done = _done(events)
    assert (done["timed_out"], done["stage_path"]) == (True, ["grounded"])
    assert _text(events) == "흥선대원군이 "


def test_not_found_closes_the_model_stream_instead_of_reading_it_to_the_end():
    # 못 찾음 뒤에 긴 출력이 이어져도 끝까지 읽지 않고 응답 스트림을 닫는다 — 작업 스레드가 다음 조각에서 멈춘다
    chatty = ScriptedLLM(["못 찾음\n", *["더 쓰는 말 "] * 200])
    service, _ = _service(chatty)
    service._llm = _Switch(
        grounded=chatty, partial=_by_stage("", partial="### 위키백과 문서에서\n없음")
    )

    _events(service)

    assert chatty.closed.wait(2)
    assert chatty.produced < 200


class _Switch:
    """단계마다 다른 LLM 대역을 쓴다 (system 프롬프트로 단계를 알아본다)."""

    def __init__(self, grounded: object, partial: object) -> None:
        self.grounded, self.partial = grounded, partial

    def stream(
        self, spec: ModelSpec, messages: Sequence[Message]
    ) -> Iterator[StreamDelta | Generation]:
        target = self.grounded if "근거 답변" in messages[0].content else self.partial
        return target.stream(spec, messages)


@pytest.mark.parametrize(
    ("done_reason", "truncated"), [("length", True), ("stop", False), (None, True)]
)
def test_done_truncated_follows_done_reason(done_reason: DoneReason | None, truncated: bool):
    llm = ScriptedLLM(
        ["찾음\n", "흥선대원군이 중건했다."], done_reason=done_reason, completion_tokens=BUDGET
    )
    service, _ = _service(llm)

    done = _done(_events(service))

    assert (done["truncated"], done["stage_path"]) == (truncated, ["grounded"])


# 표지 앞부분("못 찾")만 쓰고 생성이 끝났다 — 표지를 정하지 못한 채 끝나는 건 붙잡아 둔 글이 표지의 앞부분일 때뿐이다
@pytest.mark.parametrize("done_reason", ["length", None])
def test_grounded_cut_at_the_limit_before_the_marker_is_read_cascades_to_partial(
    done_reason: DoneReason | None,
):
    cut = ScriptedLLM(["못 찾"], done_reason=done_reason, completion_tokens=BUDGET)
    service, _ = _service(_Switch(grounded=cut, partial=_by_stage("", partial="### 문서 밖\n답")))

    events = _events(service)

    assert [e.data["stage"] for e in events if e.name == "stage"] == ["grounded", "partial"]
    assert _text(events).startswith("### 문서 밖")
    assert "못 찾" not in _text(events)
    assert _done(events)["diagnostics"]["marker"] == "missing"


def test_grounded_stop_before_the_marker_is_read_still_flows_as_found():
    service, _ = _service(ScriptedLLM(["못 찾"], done_reason="stop", completion_tokens=BUDGET))

    events = _events(service)

    assert _text(events) == "못 찾"
    done = _done(events)
    assert (done["stage_path"], done["truncated"], done["diagnostics"]["marker"]) == (
        ["grounded"],
        False,
        "missing",
    )


@pytest.mark.parametrize(
    "pieces", [["찾음\n", "흥선대원군이 "], ["흥선대원군이 "]], ids=["found", "missing"]
)
def test_after_text_is_sent_hitting_the_limit_closes_with_done_not_a_new_stage(
    pieces: list[str],
):
    service, _ = _service(ScriptedLLM(pieces, done_reason="length", completion_tokens=BUDGET))

    events = _events(service)

    assert [e.data["stage"] for e in events if e.name == "stage"] == ["grounded"]
    assert "error" not in _names(events)
    assert _text(events) == "흥선대원군이 "
    done = _done(events)
    assert (done["truncated"], done["timed_out"]) == (True, False)


def test_partial_cut_at_the_limit_with_nothing_sent_ends_with_the_fixed_refusal(
    caplog: pytest.LogCaptureFixture,
):
    # thinking 이 한도를 다 써 답 글이 하나도 없다 — score 0.6 은 low ≤ 점수 < high 라 partial 부터
    caplog.set_level(logging.INFO, logger="playground_wiki.api.answer")
    cut = ScriptedLLM([], done_reason="length", completion_tokens=BUDGET)
    service, _ = _service(cut, score=0.6)

    events = _events(service)
    record = json.loads(caplog.records[-1].getMessage().removeprefix("ask "))

    assert _text(events) == REFUSAL.format(query=QUESTION)
    done = _done(events)
    # 화면에 나간 건 고정 문구라 잘린 답이 아니다 — 한도에 닿은 사실은 diagnostics · 로그의 done_reason 에
    assert (done["stage_path"], done["truncated"], done["timed_out"]) == (["partial"], False, False)
    assert done["diagnostics"]["done_reason"] == record["done_reason"] == {"partial": "length"}
    # 생성 조각을 하나도 보내지 않았다 — F9 은 이것으로 고정 문구를 LLM 이 쓴 답과 가른다
    assert done["diagnostics"]["timings_ms"]["first_delta"] is None


def test_done_reason_per_stage_is_in_diagnostics_and_the_log_line(
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.INFO, logger="playground_wiki.api.answer")
    cut = ScriptedLLM(["못 찾"], done_reason="length", completion_tokens=BUDGET)
    service, _ = _service(_Switch(grounded=cut, partial=_by_stage("", partial="답")))

    events = _events(service)
    record = json.loads(caplog.records[-1].getMessage().removeprefix("ask "))

    done = _done(events)
    # grounded 가 잘려 partial 로 넘어간 것도 남는다 — done.truncated 는 마지막 단계(partial) 것이다
    assert done["diagnostics"]["done_reason"] == {"grounded": "length", "partial": "stop"}
    assert record["done_reason"] == done["diagnostics"]["done_reason"]
    assert done["truncated"] is False
    assert set(done) == {"stage_path", "cited", "dropped", "truncated", "timed_out", "diagnostics"}


def _limits(
    service: AnswerService, grounded: Profile = GROUNDED, partial: Profile = PARTIAL
) -> None:
    """단계 프로필(시간 제한 · max_tokens)을 바꿔 끼운다."""
    service._policy = AnswerPolicy(
        Thresholds(high=0.9, low=0.5), {"grounded": grounded, "partial": partial}
    )


def test_grounded_timeout_before_any_text_cascades_and_closes_the_model_stream():
    # 첫 조각이 시간 제한(0.2초)보다 늦다 — 보낸 게 없으니 partial 로 넘어가고, grounded 생성은 다음 조각에서 끊는다
    stuck = ScriptedLLM([0.5, "찾음\n", *["늦은 말 "] * 200])
    service, _ = _service(_Switch(grounded=stuck, partial=_by_stage("", partial="### 문서 밖\n답")))
    _limits(service, grounded=replace(GROUNDED, timeout_sec=0.2))

    events = _events(service)

    assert [e.data["stage"] for e in events if e.name == "stage"] == ["grounded", "partial"]
    assert _text(events).startswith("### 문서 밖")
    assert "늦은 말" not in _text(events)
    assert _done(events)["diagnostics"]["done_reason"]["grounded"] is None  # 끝나기 전에 끊었다
    assert stuck.closed.wait(2)
    assert stuck.produced < 200


def test_partial_timeout_before_any_text_is_a_timeout_error():
    stuck = ScriptedLLM([0.5, "늦은 답"])
    service, _ = _service(stuck, score=0.6)  # low ≤ 0.6 < high → partial 부터
    _limits(service, partial=replace(PARTIAL, timeout_sec=0.2))

    events = _events(service)

    assert _names(events) == ["sources", "stage", "error"]
    assert events[-1].data == {"code": "timeout", "message": ERROR_MESSAGES["timeout"]}
    assert stuck.closed.wait(2)


def test_stage_timeout_is_a_wall_clock_limit_not_a_gap_between_pieces():
    # 조각 사이 0.05초는 시간 제한(0.3초)보다 짧다 — 그래도 단계 전체로 0.3초에서 끊는다 (끝까지 가면 2초)
    pieces: list[str | float] = ["찾음\n"]
    for _ in range(40):
        pieces += [0.05, "말 "]
    trickle = ScriptedLLM(pieces)
    service, _ = _service(trickle)
    _limits(service, grounded=replace(GROUNDED, timeout_sec=0.3))

    events = _events(service)

    done = _done(events)
    assert (done["timed_out"], done["stage_path"]) == (True, ["grounded"])
    assert _text(events).count("말") < 40
    assert trickle.closed.wait(2)
    assert trickle.produced < 1 + 40  # 조각을 끝까지 만들지 않았다


def test_each_stage_calls_the_model_with_its_own_max_tokens():
    # max_tokens 는 Ollama 가 끊는다 — 단계마다 제 프로필의 한도를 넘기는지 본다 (요청 본문은 test_answer_config)
    llm = _by_stage("못 찾음")
    service, _ = _service(llm)
    partial_spec = ModelSpec("ollama", "fake", {"think": "off", "max_tokens": BUDGET * 4})
    _limits(service, partial=replace(PARTIAL, spec=partial_spec))

    _events(service)

    assert [call.spec.params["max_tokens"] for call in llm.calls] == [BUDGET, BUDGET * 4]


def test_question_text_is_not_logged_unless_asked(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO, logger="playground_wiki.api.answer")
    quiet, _ = _service(_by_stage("찾음\n답"))
    loud, _ = _service(_by_stage("찾음\n답"), log_questions=True)

    _events(quiet)
    quiet_record = json.loads(caplog.records[-1].getMessage().removeprefix("ask "))
    _events(loud)
    loud_record = json.loads(caplog.records[-1].getMessage().removeprefix("ask "))

    assert "question" not in quiet_record
    assert quiet_record["question_chars"] == len(QUESTION)
    assert len(quiet_record["question_sha256"]) == 64
    assert loud_record["question"] == QUESTION


def test_from_config_wires_the_setup_search_and_llm(caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.INFO, logger="playground_wiki.api.answer")
    search = FakeSearch(0.95)
    setup = AnswerSetup(
        policy=AnswerPolicy(
            Thresholds(high=0.9, low=0.5), {"grounded": GROUNDED, "partial": PARTIAL}
        ),
        library=LIBRARY,
        gate=QueryGate(_NoTokens()),
        dump_date="2026년 10월 1일",
        log_questions=True,
    )
    service = AnswerService.from_config(
        setup, search=search, llm=_by_stage("찾음\n흥선대원군이 중건했다 [1].")
    )

    gated = _events(service, "안녕!")
    answered = _events(service)

    assert _names(gated) == ["stage", "delta", "done"]
    assert _done(answered)["stage_path"] == ["grounded"]
    assert _text(answered) == "흥선대원군이 중건했다 [1]."
    assert search.calls == [QUESTION]
    assert json.loads(caplog.records[-1].getMessage().removeprefix("ask "))["question"] == QUESTION


def _private(path: str) -> tuple[AnswerService, str]:
    """개인정보가 섞인 질문과, 그 질문이 path(관문 · refuse · grounded · partial · error)로 끝나게 짠 서비스."""
    if path == "gate":
        service, _ = _service(_by_stage("찾음\n답"))
        service._gate = QueryGate(_Words())  # 실시간 규칙 — '오늘' + '날씨'
        return service, f"오늘 날씨 알려 줘. 제 번호는 {PRIVATE}"
    llms = {
        "refuse": _by_stage("찾음\n답"),  # 검색 결과가 없다
        "grounded": _by_stage("찾음\n흥선대원군이 중건했다 [1]."),
        "partial": _by_stage("못 찾음"),  # grounded → partial
        "error": ScriptedLLM([GenerationError(SERVER_DETAIL)]),
    }
    service, _ = _service(llms[path], score=None if path == "refuse" else 0.95)
    return service, f"제 번호는 {PRIVATE} 인데, 경복궁을 중건한 사람은?"


@pytest.mark.parametrize("path", ["gate", "refuse", "grounded", "partial", "error"])
def test_question_text_is_in_no_log_line_by_default(path: str, caplog: pytest.LogCaptureFixture):
    caplog.set_level(logging.DEBUG)  # 이 로거만이 아니라 모든 로거 · 모든 수준
    service, question = _private(path)

    _events(service, question)

    lines = [record.getMessage() for record in caplog.records]
    assert [line for line in lines if PRIVATE in line] == []
    ask = json.loads(next(line for line in lines if line.startswith("ask {")).removeprefix("ask "))
    assert (ask["question_chars"], ask["question_sha256"]) == (
        len(question),
        question_digest(service._question_salt, question),
    )
    # salt 없는 SHA-256 이 아니다 — 짧은 질문을 사전으로 되짚을 수 없게
    assert ask["question_sha256"] != hashlib.sha256(question.encode("utf-8")).hexdigest()


def test_question_hash_is_stable_within_a_run_and_changes_across_runs(
    caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.INFO, logger="playground_wiki.api.answer")
    first_run, _ = _service(_by_stage("찾음\n답"))
    second_run, _ = _service(_by_stage("찾음\n답"))

    hashes = []
    for service in (first_run, first_run, second_run):
        _events(service)
        record = json.loads(caplog.records[-1].getMessage().removeprefix("ask "))
        hashes.append(record["question_sha256"])

    assert hashes[0] == hashes[1]
    assert hashes[0] != hashes[2]
    assert len(hashes[0]) == 64


@pytest.mark.parametrize("path", ["gate", "refuse", "grounded", "partial"])
def test_done_diagnostics_carry_no_question_text(path: str):
    service, question = _private(path)

    done = _done(_events(service, question))

    assert set(done["diagnostics"]) == DIAGNOSTICS_KEYS
    assert PRIVATE not in json.dumps(done, ensure_ascii=False)


def test_gate_diagnostics_name_rule_words_not_the_question():
    # 실시간 규칙의 근거는 사전의 때 표지 · 대상어뿐이다 (잡담은 질문 전체가 사전 말과 같을 때만 걸린다)
    service, question = _private("gate")

    gate = _done(_events(service, question))["diagnostics"]["gate"]

    assert gate == {"rule": "realtime", "kind": "realtime", "matched": ["오늘", "날씨"]}


def test_error_event_sends_only_the_fixed_message():
    service, question = _private("error")

    events = _events(service, question)

    # 원인(서버 주소가 든 예외 문구)은 서버 로그에만 — 클라이언트에는 화면 문구만 간다
    assert events[-1] == AnswerEvent(
        "error", {"code": "generation", "message": ERROR_MESSAGES["generation"]}
    )
    assert SERVER_DETAIL not in "".join(event.to_sse() for event in events)


def test_ask_route_rejects_a_question_over_the_limit_without_searching():
    app = create_app()
    app.state.answer_service, search = _service(_by_stage("찾음\n답"))
    client = TestClient(app)  # with 없이 — lifespan(모델 · 저장소)을 돌리지 않는다

    response = client.post("/ask", json={"question": "가" * (MAX_QUERY_CHARS + 1)})

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
    assert response.headers["content-type"] == "application/json"  # 스트림을 열지 않는다
    assert search.calls == []


def test_ask_route_takes_the_longest_question_after_stripping_whitespace():
    question = "가" * MAX_QUERY_CHARS
    app = create_app()
    app.state.answer_service, search = _service(_by_stage("찾음\n답"))
    client = TestClient(app)

    response = client.post("/ask", json={"question": f"  {question}  "})

    assert response.status_code == status.HTTP_200_OK
    assert search.calls == [question]


def test_ask_route_streams_server_sent_events():
    app = create_app()
    app.state.answer_service, _ = _service(_by_stage("찾음\n흥선대원군이 중건했다 [1]."))
    client = TestClient(app)  # with 없이 — lifespan(모델 · 저장소)을 돌리지 않는다

    response = client.post("/ask", json={"question": QUESTION})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: done" in response.text
    assert client.post("/ask", json={"question": "  "}).status_code == 422
