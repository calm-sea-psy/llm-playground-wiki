import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

from playground_corpus import Chunk
from playground_retrieval.core import Hit
from playground_retrieval.search import SearchResult

from playground_wiki.evaluation.agent_eval import (
    AgentEvalRun,
    AgentItem,
    AgentRecord,
    Pairing,
    RecordingSearch,
    SearchCall,
    build_record,
    compound_items,
    contains_answer,
    correct,
    f9_items,
    hop_coverage,
    meets_expected,
    read_records,
    render_report,
)
from playground_wiki.evaluation.answer_metrics import contains_gold
from playground_wiki.evaluation.answer_run import AnswerRecord, EvalItem

COMPOUND_ROW = {
    "id": "cq-001",
    "kind": "bridge",
    "question": "「소나기」 작가가 교수로 있던 대학교를 세운 사람은?",
    "answer": "조영식",
    "aliases": ["조 영식"],
    "hops": [
        {"sub_question": "a", "answer": "황순원", "title": "소나기 (소설)", "evidence": "e"},
        {"sub_question": "b", "answer": "경희대학교", "title": "황순원", "evidence": "e"},
        {"sub_question": "c", "answer": "조영식", "title": "경희대학교", "evidence": "e"},
    ],
    "single_doc_check": {"bm25_top10": [], "answerable_from_one": False},
}
ITEM = compound_items([COMPOUND_ROW])[0]


@dataclass(frozen=True)
class _Event:
    name: str
    data: dict


def _done(**agent: object) -> _Event:
    diagnostics = {
        "gate": None,
        "top_score": 0.9,
        "timings_ms": {"first_delta": 1200.0, "total": 3000.0},
    }
    if agent:
        diagnostics["agent"] = agent
    return _Event("done", {"diagnostics": diagnostics})


def _events(
    answer: str, *, agent: bool = False, titles: tuple[str, ...] = ("경희대학교",)
) -> list[_Event]:
    stages = ["agent_decompose", "agent_search", "agent_judge"] if agent else []
    events = [_Event("stage", {"stage": stage}) for stage in stages]
    events.append(_Event("sources", {"sources": [{"title": title} for title in titles]}))
    events.append(_Event("stage", {"stage": "grounded"}))
    events.append(_Event("delta", {"text": answer}))
    diag = {
        "steps": ["decompose", "search", "judge", "compose"],
        "iterations": 2,
        "stop": "sufficient",
    }
    events.append(_done(**diag) if agent else _done())
    return events


def _record(
    answer: str, mode: str = "single", item: AgentItem = ITEM, **kwargs: object
) -> AgentRecord:
    base = build_record(
        item,
        "agent" if mode == "agent" else "single",
        _events(answer, agent=mode == "agent"),
        [],
        wall_ms=3000.0,
    )
    return replace(base, **kwargs)


def _hit(title: str) -> Hit:
    chunk = Chunk(id="1-0", doc_id="1", title=title, section="", text="본문", seq=0, token_count=1)
    return Hit(chunk=chunk, score=0.9)


class _Search:
    async def search(self, query: str) -> SearchResult:
        return SearchResult(hits=(_hit(f"{query} 문서"),), took_ms=1.0)


class _Answerer:
    """질문마다 검색 두 번 · 정해 둔 답. 'down' 이 들어간 질문은 생성 실패(error)."""

    def __init__(self, search: RecordingSearch) -> None:
        self.search = search
        self.modes: list[str] = []

    async def answer(self, question: str, *, mode: str = "single") -> AsyncIterator[_Event]:
        self.modes.append(mode)
        await self.search.search(f"{question}-1")
        await self.search.search(f"{question}-2")
        if "down" in question:
            yield _Event("error", {"code": "generation"})
            return
        for event in _events("조영식", agent=mode == "agent"):
            yield event


def _items(*ids: str) -> list[AgentItem]:
    return [replace(ITEM, id=item_id, question=item_id) for item_id in ids]


def _run(items: list[AgentItem], path: Path, mode: str = "agent", limit: int | None = None):
    search = RecordingSearch(_Search())
    answerer = _Answerer(search)
    summary = asyncio.run(
        AgentEvalRun(search, answerer, clock=lambda: 0.0).run(items, mode, path, limit=limit)
    )
    return summary, answerer


def test_compound_items_keep_answers_and_normalized_hop_titles():
    assert ITEM.answers == ("조영식", "조 영식")
    assert ITEM.hop_titles == ("소나기 (소설)", "황순원", "경희대학교")
    assert ITEM.kind == "bridge"


def test_f9_items_split_korquad_and_offwiki_rows():
    rows = [
        {"id": "k1", "source": "korquad", "question": "q", "gold_title": "t", "gold_answer": "a"},
        {
            "id": "o1",
            "source": "offwiki",
            "question": "q",
            "expected": "refuse",
            "category": "personal",
            "trap": False,
        },
    ]

    assert [item.id for item in f9_items(rows, "f9")] == ["k1"]
    offwiki = f9_items(rows, "offwiki")
    assert [(item.id, item.set_name, item.expected) for item in offwiki] == [
        ("o1", "offwiki", "refuse")
    ]


def test_record_keeps_answer_path_apart_from_agent_steps():
    record = build_record(
        ITEM,
        "agent",
        _events("조영식이다", agent=True),
        [SearchCall("q", ("황순원",))],
        wall_ms=3210.04,
    )

    assert record.stage_path == ("agent_decompose", "agent_search", "agent_judge", "grounded")
    assert record.answer_path == ("grounded",)
    assert record.final_stage == "grounded"
    assert record.iterations == 2
    assert record.source_titles == ("경희대학교",)
    assert record.wall_ms == 3210.0
    assert (
        AgentRecord.from_json(json.loads(json.dumps(record.to_json(), ensure_ascii=False)))
        == record
    )


def test_contains_answer_follows_the_f9_contains_gold_rule():
    item = EvalItem(
        id="k", source="korquad", question="q", gold_title="t", gold_answer="흥선 대원군"
    )
    agent_item = AgentItem.from_eval_item(item)
    for answer in ("흥선대원군이 중건했다", "모른다"):
        old = AnswerRecord(
            item=item,
            stage_path=("grounded",),
            answer=answer,
            hits=(),
            context_count=0,
            marker=None,
            cited=(),
            dropped=(),
            truncated=False,
            timed_out=False,
            error=None,
            top_score=None,
            gate_rule=None,
            spec_id=None,
            prompt_version=None,
            search_ms=None,
            first_delta_ms=None,
            total_ms=None,
            wall_ms=0.0,
        )

        assert contains_answer(_record(answer, item=agent_item)) == contains_gold(old)


def test_offwiki_is_judged_by_the_expected_stage():
    item = replace(ITEM, set_name="offwiki", answers=(), expected="partial", hop_titles=())
    grounded = _record("답", item=item)
    partial = replace(grounded, stage_path=("partial",))

    assert contains_answer(grounded) is None
    assert meets_expected(grounded) is False
    assert correct(partial) is True


def test_hop_coverage_counts_answer_sources_and_searched_titles():
    record = replace(
        _record("조영식"),
        source_titles=("경희대학교", "황순원"),
        searches=(SearchCall("a", ("소나기 (소설)",)),),
    )

    assert hop_coverage(record) == (2, 3)
    assert hop_coverage(record, searched=True) == (1, 3)


def test_run_records_searches_and_resumes_after_existing_ids(tmp_path: Path):
    path = tmp_path / "compound-agent.jsonl"

    first, answerer = _run(_items("a", "b", "c"), path, limit=2)
    second, _ = _run(_items("a", "b", "c"), path)

    assert (first.recorded, second.recorded, second.skipped) == (2, 1, 2)
    assert answerer.modes == ["agent", "agent"]
    records = read_records(path)
    assert [record.item.id for record in records] == ["a", "b", "c"]
    assert [call.query for call in records[0].searches] == ["a-1", "a-2"]  # 문항마다 새로 모은다


def test_run_drops_a_partial_last_line_and_retries_generation_errors(tmp_path: Path):
    path = tmp_path / "f9-single.jsonl"
    _run(_items("a", "down"), path, mode="single")
    with path.open("a", encoding="utf-8") as f:
        f.write('{"item": ')  # 끊긴 줄

    summary, _ = _run(_items("a", "down"), path, mode="single")

    assert (summary.skipped, summary.recorded, summary.retried) == (1, 1, 1)
    assert [record.item.id for record in read_records(path)] == ["a", "down"]


def test_report_pairs_by_id_and_lists_only_ids():
    single = [_record("모른다"), replace(_record("조영식"), item=replace(ITEM, id="cq-002"))]
    agent = [
        _record("조영식", mode="agent"),
        replace(_record("조영식", mode="agent"), item=replace(ITEM, id="cq-002")),
        replace(_record("x", mode="agent"), item=replace(ITEM, id="cq-999")),  # 단발 기록이 없다
    ]

    pairing = Pairing.match("compound", single, agent)
    text = render_report([pairing], today=date(2026, 11, 1), settings={"grounded": "spec"})

    assert [record.item.id for record in pairing.agent] == ["cq-001", "cq-002"]
    assert "| 2 | 1 | 0 | 1 | 0 | 1 |" in text  # 짝 · 둘 다 · 단발만 · 에이전트만 · 둘 다 틀림 · p
    assert "에이전트만 맞힘: cq-001" in text
    assert "decompose → search → judge → compose" in text
    assert ITEM.question not in text  # 문항 원문은 리포트에 없다
    assert "hop 모두 덮음" in text


def test_route_records_count_the_original_search_as_an_iteration():
    def agent(route: str | None, iterations: int) -> AgentRecord:
        diagnostics = {"iterations": iterations} | ({} if route is None else {"route": route})
        return _record("조영식", mode="agent", agent=diagnostics)

    assert (agent("escalated_partial", 2).route, agent("escalated_partial", 2).iterations) == (
        "escalated_partial",
        3,
    )
    assert agent("single_found", 0).iterations == 1
    assert agent("gate", 0).iterations == 0
    assert (agent(None, 2).route, agent(None, 2).iterations) == (None, 2)  # route 전 기록
