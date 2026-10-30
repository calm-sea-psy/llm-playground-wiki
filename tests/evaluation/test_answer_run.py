import asyncio
import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pytest
from playground_corpus import Chunk
from playground_llm import (
    DoneReason,
    FakeLLM,
    Generation,
    GenerationError,
    Message,
    ModelSpec,
    PromptLibrary,
    StreamDelta,
    Usage,
)
from playground_retrieval.core import Hit
from playground_retrieval.search import SearchResult

from playground_wiki.api.answer import REFUSAL, AnswerService
from playground_wiki.evaluation.answer_run import (
    MAX_CONSECUTIVE_ERRORS,
    AnswerRecord,
    AnswerRun,
    EvalItem,
    Searcher,
    build_record,
    read_records,
    tagged,
)
from playground_wiki.gate import QueryGate, Token
from playground_wiki.generation import PROMPT_DIR, AnswerPolicy, Profile, Thresholds

LIBRARY = PromptLibrary.load(PROMPT_DIR)
SPEC = ModelSpec("ollama", "fake", {"think": "off", "max_tokens": 64})
GROUNDED = Profile(SPEC, system=("base@v4", "grounded@v2"), user="question@v1", timeout_sec=5)
PARTIAL = Profile(SPEC, system=("base@v4", "partial@v2"), user="question@v1", timeout_sec=5)
POLICY = AnswerPolicy(Thresholds(high=0.9, low=0.5), {"grounded": GROUNDED, "partial": PARTIAL})

PALACE = EvalItem(
    id="k1",
    source="korquad",
    question="경복궁을 중건한 사람은?",
    gold_title="경복궁",
    gold_answer="흥선대원군",
)
HELLO = EvalItem(
    id="o1", source="offwiki", question="안녕!", expected="refuse", category="smalltalk"
)
LEAVE = EvalItem(
    id="o2",
    source="offwiki",
    question="우리 회사 연차 규정은?",
    expected="refuse",
    category="personal",
)
SEJONG = EvalItem(
    id="k2",
    source="korquad",
    question="훈민정음을 만든 왕은?",
    gold_title="세종",
    gold_answer="세종",
)


class _NoTokens:
    """형태소가 없는 분석기 — 실시간 규칙은 걸리지 않고 잡담 사전만 본다 (Kiwi 를 올리지 않는다)."""

    def tokenize(self, text: str) -> list[Token]:
        return []


class FakeSearch:
    """정답 청크와 다른 청크 하나를 돌려준다. no_hits 에 든 질문은 결과가 없다."""

    def __init__(self, no_hits: Sequence[str] = ()) -> None:
        self.no_hits = set(no_hits)
        self.calls: list[str] = []

    async def search(self, query: str) -> SearchResult:
        self.calls.append(query)
        if query in self.no_hits:
            return SearchResult(hits=(), took_ms=1.0)
        hits = (
            Hit(chunk=_chunk("1-0", "경복궁", "경복궁 > 역사\n흥선대원군이 중건했다."), score=0.95),
            Hit(chunk=_chunk("2-0", "창덕궁", "창덕궁 > \n태종이 지었다."), score=0.85),
        )
        return SearchResult(hits=hits, took_ms=1.0)


def _chunk(chunk_id: str, title: str, text: str) -> Chunk:
    doc_id, seq = chunk_id.split("-")
    return Chunk(
        id=chunk_id, doc_id=doc_id, title=title, section="", text=text, seq=int(seq), token_count=4
    )


def _by_stage(grounded: str, partial: str = "### 위키백과 문서에서\n없음 [2].") -> FakeLLM:
    """system 프롬프트로 단계를 알아보고 단계마다 정해 둔 답을 낸다."""
    return FakeLLM(
        [lambda spec, messages: grounded if "근거 답변" in messages[0].content else partial]
    )


def _fail(spec: ModelSpec, messages: Sequence[Message]) -> str:
    raise GenerationError("server down")


def _runner(llm: object, search: FakeSearch | None = None) -> AnswerRun:
    def make_answerer(searcher: Searcher) -> AnswerService:
        return AnswerService(
            search=searcher,
            policy=POLICY,
            library=LIBRARY,
            llm=llm,
            gate=QueryGate(_NoTokens()),
            dump_date="2026년 10월 1일",
            today=lambda: date(2026, 10, 24),
        )

    return AnswerRun(search or FakeSearch(no_hits=[LEAVE.question]), make_answerer)


def _run(runner: AnswerRun, items: Sequence[EvalItem], path: Path, **kwargs: object):
    return asyncio.run(runner.run(items, path, **kwargs))


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_grounded_answer_is_recorded_with_context_and_citations(tmp_path: Path):
    path = tmp_path / "answers.jsonl"

    _run(_runner(_by_stage("찾음\n흥선대원군이 중건했다 [1][7].")), [PALACE], path)

    (record,) = read_records(path)
    assert record.item == PALACE
    assert record.stage_path == ("grounded",)
    assert record.marker == "found"
    assert record.answer == "흥선대원군이 중건했다 [1]."
    assert (record.cited, record.dropped) == ((1,), (7,))
    assert record.context_count == 2
    assert [hit.chunk_id for hit in record.hits] == ["1-0", "2-0"]
    assert record.hits[0].text == "경복궁 > 역사\n흥선대원군이 중건했다."
    assert record.top_score == 0.95
    assert record.spec_id == SPEC.spec_id
    assert record.prompt_version.startswith("base@v4")
    assert (record.truncated, record.timed_out, record.error) == (False, False, None)
    assert record.first_delta_ms is not None
    assert record.total_ms is not None


def test_not_found_is_recorded_as_a_cascade(tmp_path: Path):
    path = tmp_path / "answers.jsonl"

    _run(_runner(_by_stage("못 찾음")), [PALACE], path)

    (record,) = read_records(path)
    assert record.stage_path == ("grounded", "partial")
    assert record.marker == "not_found"
    assert record.answer.startswith("### 위키백과 문서에서")
    assert record.cited == (2,)


def test_gate_and_refusal_paths_are_recorded_without_llm(tmp_path: Path):
    path = tmp_path / "answers.jsonl"
    llm = _by_stage("찾음\n답")

    _run(_runner(llm), [HELLO, LEAVE], path)

    hello, leave = read_records(path)
    assert hello.stage_path == ("gate",)
    assert hello.gate_rule is not None
    assert (hello.hits, hello.top_score, hello.marker, hello.search_ms) == ((), None, None, None)
    assert leave.stage_path == ("refuse",)
    assert leave.answer == REFUSAL.format(query=LEAVE.question)
    assert (leave.context_count, leave.spec_id) == (0, None)
    assert llm.calls == []


def test_rerun_skips_items_that_are_already_recorded(tmp_path: Path):
    path = tmp_path / "answers.jsonl"
    _run(_runner(_by_stage("찾음\n답 [1].")), [PALACE, HELLO], path)
    search = FakeSearch()
    llm = _by_stage("찾음\n다른 답 [1].")

    summary = _run(_runner(llm, search), [PALACE, HELLO, SEJONG], path)

    assert (summary.total, summary.skipped, summary.recorded) == (3, 2, 1)
    assert search.calls == [SEJONG.question]
    assert [row["item"]["id"] for row in _lines(path)] == ["k1", "o1", "k2"]
    assert read_records(path)[0].answer == "답 [1]."  # 앞 기록을 고치지 않는다


def test_limit_runs_only_that_many_pending_items(tmp_path: Path):
    path = tmp_path / "answers.jsonl"
    runner = _runner(_by_stage("찾음\n답"))

    first = _run(runner, [PALACE, HELLO, SEJONG], path, limit=1)
    second = _run(runner, [PALACE, HELLO, SEJONG], path, limit=1)

    assert (first.recorded, second.skipped, second.recorded) == (1, 1, 1)
    assert [row["item"]["id"] for row in _lines(path)] == ["k1", "o1"]


def test_generation_failures_are_retried_on_the_next_run(tmp_path: Path):
    path = tmp_path / "answers.jsonl"
    _run(_runner(FakeLLM([_fail])), [PALACE, HELLO], path)
    assert [row["error"] for row in _lines(path)] == ["generation", None]

    summary = _run(_runner(_by_stage("찾음\n흥선대원군 [1].")), [PALACE, HELLO], path)

    assert (summary.retried, summary.skipped, summary.recorded) == (1, 1, 1)
    records = read_records(path)
    assert [record.item.id for record in records] == ["o1", "k1"]
    assert records[1].error is None


def test_consecutive_generation_failures_stop_the_run(tmp_path: Path):
    path = tmp_path / "answers.jsonl"
    items = [
        EvalItem(
            id=f"k{i}", source="korquad", question=f"질문 {i}", gold_title="t", gold_answer="a"
        )
        for i in range(MAX_CONSECUTIVE_ERRORS + 2)
    ]

    with pytest.raises(RuntimeError, match="in a row"):
        _run(_runner(FakeLLM([_fail])), items, path)

    assert len(_lines(path)) == MAX_CONSECUTIVE_ERRORS


def test_a_partial_last_line_is_dropped_and_its_item_runs_again(tmp_path: Path):
    path = tmp_path / "answers.jsonl"
    _run(_runner(_by_stage("찾음\n답")), [PALACE], path)
    with path.open("a", encoding="utf-8") as f:
        f.write('{"item": {"id": "o1"')  # 쓰다가 끊긴 줄

    summary = _run(_runner(_by_stage("찾음\n답")), [PALACE, HELLO], path)

    assert (summary.skipped, summary.recorded) == (1, 1)
    assert [row["item"]["id"] for row in _lines(path)] == ["k1", "o1"]


def test_record_survives_a_json_round_trip(tmp_path: Path):
    path = tmp_path / "answers.jsonl"
    _run(_runner(_by_stage("찾음\n흥선대원군이 중건했다 [1].")), [PALACE], path)
    (record,) = read_records(path)

    assert AnswerRecord.from_json(json.loads(json.dumps(record.to_json()))) == record


def test_old_record_lines_without_done_reason_still_load(tmp_path: Path):
    path = tmp_path / "answers.jsonl"
    _run(_runner(_by_stage("찾음\n흥선대원군이 중건했다 [1].")), [PALACE], path)
    row = _lines(path)[0]
    del row["done_reason"]  # 이 필드가 생기기 전에 쓴 줄
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    (record,) = read_records(path)

    assert record.done_reason == {}
    assert record.stage_path == ("grounded",)


class StagedLLM:
    """단계마다 정해 둔 조각과 done_reason 으로 끝나는 대역 (FakeLLM 은 늘 stop 이라 잘림을 못 낸다)."""

    def __init__(
        self,
        grounded: tuple[Sequence[str], DoneReason | None],
        partial: tuple[Sequence[str], DoneReason | None],
    ) -> None:
        self.grounded, self.partial = grounded, partial

    def stream(self, spec: ModelSpec, messages: Sequence[Message]):
        pieces, reason = self.grounded if "근거 답변" in messages[0].content else self.partial
        for piece in pieces:
            yield StreamDelta("text", piece)
        usage = Usage(prompt_tokens=1, completion_tokens=64, total_ms=0)
        yield Generation(text="".join(pieces), model=spec, usage=usage, done_reason=reason)


def test_done_reason_per_stage_is_recorded_when_grounded_is_cut_before_the_marker(
    tmp_path: Path,
):
    path = tmp_path / "answers.jsonl"
    llm = StagedLLM(grounded=(["못 찾"], "length"), partial=(["### 문서 밖\n답"], "stop"))

    _run(_runner(llm), [PALACE], path)

    (record,) = read_records(path)
    assert record.stage_path == ("grounded", "partial")
    assert record.marker == "missing"
    assert record.done_reason == {"grounded": "length", "partial": "stop"}
    assert record.truncated is False  # done.truncated 는 마지막 단계(partial) 것
    assert record.answer == "### 문서 밖\n답"


def test_partial_ending_with_the_fixed_refusal_has_no_first_delta(tmp_path: Path):
    path = tmp_path / "answers.jsonl"
    llm = StagedLLM(grounded=(["못 찾음\n"], "stop"), partial=([], "length"))

    _run(_runner(llm), [PALACE], path)

    (record,) = read_records(path)
    assert record.answer == REFUSAL.format(query=PALACE.question)
    assert record.first_delta_ms is None
    assert (record.truncated, record.cited) == (False, ())
    # grounded 는 못 찾음에서 끊어 이유가 없고, partial 은 한도에 닿았다
    assert record.done_reason == {"grounded": None, "partial": "length"}


class _Event:
    def __init__(self, name: str, data: dict) -> None:
        self.name, self.data = name, data


def test_build_record_needs_exactly_one_ending_event():
    events = [_Event("stage", {"stage": "grounded"}), _Event("delta", {"text": "답"})]

    with pytest.raises(ValueError, match="done or error"):
        build_record(PALACE, events, None, wall_ms=1.0)


def test_build_record_rejects_sources_out_of_search_order():
    result = asyncio.run(FakeSearch().search(PALACE.question))
    events = [
        _Event("sources", {"sources": [{"number": 1, "title": "창덕궁"}]}),
        _Event("error", {"code": "generation"}),
    ]

    with pytest.raises(ValueError, match="search order"):
        build_record(PALACE, events, result, wall_ms=1.0)


def test_tagged_appends_the_tag_to_the_file_name_and_keeps_the_base_path():
    path = Path("data/eval/f9-answers-v1.jsonl")

    assert tagged(path, None) == path
    assert tagged(path, "grounded-v3") == Path("data/eval/f9-answers-v1-grounded-v3.jsonl")
    meta = tagged(path, "grounded-v3").with_suffix(".meta.json")
    assert meta.name == "f9-answers-v1-grounded-v3.meta.json"
