"""scripts/eval_agent.py 의 인자 · 경로 · 리포트를 모델 없이 — GPU · DB 는 쓰지 않는다."""

import json
from datetime import date
from pathlib import Path

import pytest
from conftest import load_script

from playground_wiki.evaluation.agent_eval import AgentRecord, compound_items

eval_agent = load_script("eval_agent")
ROW = {
    "id": "cq-001",
    "kind": "bridge",
    "question": "비밀 질문 원문",
    "answer": "조영식",
    "aliases": [],
    "hops": [
        {"sub_question": "a", "answer": "x", "title": "황순원", "evidence": "e"},
        {"sub_question": "b", "answer": "y", "title": "경희대학교", "evidence": "e"},
    ],
    "single_doc_check": {"bm25_top10": [], "answerable_from_one": False},
}


def _record(mode: str, answer: str) -> dict:
    item = compound_items([ROW])[0]
    return AgentRecord(
        item=item,
        mode=mode,
        stage_path=("grounded",),
        answer=answer,
        source_titles=("경희대학교",),
        searches=(),
        error=None,
        first_delta_ms=1000.0,
        total_ms=2000.0,
        wall_ms=2000.0,
    ).to_json()


def test_run_needs_set_and_mode():
    with pytest.raises(SystemExit):
        eval_agent._parse_args(["run", "--set", "compound"])

    args = eval_agent._parse_args(["run", "--set", "f9", "--mode", "agent", "--limit", "3"])
    assert (args.set, args.mode, args.limit) == ("f9", "agent", 3)


def test_records_path_appends_the_tag():
    assert eval_agent.records_path("compound", "agent") == Path(
        "data/eval/agent/compound-agent.jsonl"
    )
    assert eval_agent.records_path("f9", "single", "t1").name == "f9-single-t1.jsonl"


def test_report_writes_aggregates_without_question_text(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for mode, answer in (("single", "모른다"), ("agent", "조영식이다")):
        path = eval_agent.records_path("compound", mode)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_record(mode, answer), ensure_ascii=False) + "\n", encoding="utf-8"
        )

    eval_agent.report(None, date(2026, 11, 1))

    text = (tmp_path / "reports" / "2026-11-01-agent.md").read_text(encoding="utf-8")
    assert "## compound (1 문항)" in text
    assert "에이전트만 맞힘: cq-001" in text
    assert "비밀 질문 원문" not in text


def test_report_tag_pairs_untagged_single_with_tagged_agent(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runs = (("single", None, "모른다"), ("agent", "v2", "조영식이다"), ("agent", None, "모른다"))
    for mode, tag, answer in runs:
        path = eval_agent.records_path("compound", mode, tag)
        path.parent.mkdir(parents=True, exist_ok=True)
        row = _record(mode, answer) | {"agent": {"route": "escalated_partial", "iterations": 1}}
        path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    eval_agent.report("v2", date(2026, 11, 2))

    text = (tmp_path / "reports" / "2026-11-02-agent-v2.md").read_text(encoding="utf-8")
    assert "에이전트만 맞힘: cq-001" in text  # 꼬리표 없는 에이전트가 아니라 v2 와 짝지었다
    assert "길(route): escalated_partial 1" in text
