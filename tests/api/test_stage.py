import asyncio
from collections.abc import Iterator, Sequence

import pytest
from playground_llm import (
    DoneReason,
    FakeLLM,
    Generation,
    GenerationError,
    Message,
    ModelSpec,
    StreamDelta,
    Usage,
)

from playground_wiki.api.answer import next_stage
from playground_wiki.api.stage import StageOutcome, generate_stage, is_truncated
from playground_wiki.api.threaded_stream import ThreadedStream
from playground_wiki.generation import ChatPrompt, Profile

SPEC = ModelSpec("ollama", "fake", {"think": "off", "max_tokens": 64})
BUDGET = int(SPEC.params["max_tokens"])
PROFILE = Profile(SPEC, system=("base@v4",), user="question@v1", timeout_sec=5)
PROMPT = ChatPrompt(messages=(Message("user", "질문"),), version="question@v1:0")


def _outcome(**changes: object) -> StageOutcome:
    fields: dict[str, object] = {
        "sent": False,
        "marker": None,
        "done_reason": "stop",
        "truncated": False,
        "timed_out": False,
        "error": None,
        "cited": (),
        "dropped": (),
    }
    return StageOutcome(**(fields | changes))


def _run(llm: object, *, read_marker: bool, count: int = 3) -> tuple[list[str], StageOutcome]:
    async def collect() -> list[str | StageOutcome]:
        stream = generate_stage(llm, PROFILE, PROMPT, context_count=count, read_marker=read_marker)
        return [item async for item in stream]

    *pieces, outcome = asyncio.run(collect())
    assert isinstance(outcome, StageOutcome)
    return [str(piece) for piece in pieces], outcome


@pytest.mark.parametrize(
    ("done_reason", "completion_tokens", "expected"),
    [
        ("length", BUDGET, True),
        ("stop", BUDGET, False),  # 한도와 같은 수에서 스스로 끝냈다
        (None, BUDGET, True),  # 서버가 주지 않으면 토큰 수로 짐작
        (None, BUDGET - 1, False),
    ],
)
def test_is_truncated_reads_done_reason_before_the_token_count(
    done_reason: DoneReason | None, completion_tokens: int, expected: bool
):
    usage = Usage(prompt_tokens=1, completion_tokens=completion_tokens, total_ms=0)
    generation = Generation(text="", model=SPEC, usage=usage, done_reason=done_reason)

    assert is_truncated(generation) is expected


def test_generate_stage_streams_text_then_one_outcome():
    pieces, outcome = _run(FakeLLM(["찾음\n흥선대원군이 중건했다 [1][7]."]), read_marker=True)

    assert "".join(pieces) == "흥선대원군이 중건했다 [1]."
    assert outcome == _outcome(sent=True, marker="found", cited=(1,), dropped=(7,))


def test_generate_stage_sends_nothing_when_the_marker_says_not_found():
    pieces, outcome = _run(FakeLLM(["못 찾음\n덧붙인 말"]), read_marker=True)

    assert pieces == []
    # 끝까지 읽지 않고 끊었으니 멈춘 이유는 없다
    assert outcome == _outcome(marker="not_found", done_reason=None)


def test_generate_stage_without_marker_passes_marker_words_through():
    pieces, outcome = _run(FakeLLM(["못 찾음은 표지가 아니다"]), read_marker=False)

    assert "".join(pieces) == "못 찾음은 표지가 아니다"
    assert (outcome.sent, outcome.marker) == (True, None)


class _Failing:
    def stream(
        self, spec: ModelSpec, messages: Sequence[Message]
    ) -> Iterator[StreamDelta | Generation]:
        yield StreamDelta("text", "답 [1")
        raise GenerationError("server down")


def test_generate_stage_puts_a_failure_in_the_outcome_and_flushes_the_held_tail():
    pieces, outcome = _run(_Failing(), read_marker=False)

    assert "".join(pieces) == "답 [1"  # 붙잡아 둔 덜 닫힌 번호까지 내보낸다
    assert (outcome.sent, outcome.error, outcome.done_reason) == (True, "server down", None)


@pytest.mark.parametrize(
    ("stage", "outcome", "expected"),
    [
        ("grounded", _outcome(), "partial"),  # 아무것도 못 보냈다 — 캐스케이드
        ("grounded", _outcome(timed_out=True), "partial"),
        ("grounded", _outcome(sent=True), None),
        ("grounded", _outcome(error="down"), None),
        ("partial", _outcome(), None),  # 마지막 단계
    ],
)
def test_next_stage_cascades_only_from_an_empty_grounded(
    stage: str, outcome: StageOutcome, expected: str | None
):
    assert next_stage(stage, outcome) == expected


def test_threaded_stream_hands_over_pieces_and_ends_with_none():
    async def collect() -> list[object]:
        stream = ThreadedStream(FakeLLM(["답"]), SPEC, PROMPT.messages)
        items = []
        while (item := await stream.next()) is not None:
            items.append(item)
        stream.stop()
        return items

    items = asyncio.run(collect())

    assert [type(item) for item in items][-1] is Generation
    assert "".join(item.text for item in items if isinstance(item, StreamDelta)) == "답"
