"""scripts/eval_agent.py 의 인자 · 경로 · 리포트를 모델 없이 — GPU · DB 는 쓰지 않는다."""

import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

import pytest
from conftest import load_script

from playground_wiki.evaluation.agent_eval import AgentItem, AgentRecord, compound_items
from playground_wiki.evaluation.agent_safety import f4_agent_item
from playground_wiki.evaluation.answer_run import ContextChunk
from playground_wiki.evaluation.f4_prose import F4Item

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


# --- 안전 회귀 ---

CHUNK = ContextChunk(chunk_id="1-0", title="문서", section="", text="본문", score=0.6)
HARMFUL_ROWS = [
    {"id": "hz-01", "kind": "harmful", "pair": "hc-01", "question": "유해 질문 원문"},
    {"id": "hc-01", "kind": "contrast", "pair": "hz-01", "question": "대조 질문 원문"},
]


def test_safety_sets_refuse_single_runs_and_need_a_round_tag():
    with pytest.raises(SystemExit):
        eval_agent._parse_args(["run", "--set", "harmful", "--mode", "single", "--tag", "s-1"])
    with pytest.raises(SystemExit):
        eval_agent._parse_args(["run", "--set", "procedure", "--mode", "agent"])
    with pytest.raises(SystemExit):
        eval_agent._parse_args(["safety-report"])
    with pytest.raises(SystemExit):
        eval_agent._parse_args(
            ["run", "--set", "f4", "--mode", "agent", "--tag", "s-1", "--kinds", "x"]
        )

    args = eval_agent._parse_args(["run", "--set", "f4", "--mode", "single", "--tag", "safety-1"])
    assert (args.set, args.mode, args.tag) == ("f4", "single", "safety-1")


def _agent_row(item_id: str, kind: str, stage: str, route: str) -> dict:
    return AgentRecord(
        item=AgentItem(id=item_id, set_name="harmful", question=f"{item_id} 원문", kind=kind),
        mode="agent",
        stage_path=("agent_decompose", stage),
        answer=f"{item_id} 답 원문",
        source_titles=("문서",),
        searches=(),
        error=None,
        first_delta_ms=10.0,
        total_ms=20.0,
        wall_ms=20.0,
        agent={"route": route},
        context=(CHUNK,),
    ).to_json()


def _single_row(item_id: str, kind: str, stage: str) -> dict:
    return {
        "item": {
            "id": item_id,
            "source": "harmful",
            "question": f"{item_id} 원문",
            "category": kind,
        },
        "stage_path": [stage],
        "answer": f"{item_id} 단발 답",
        "hits": [asdict(CHUNK)],
        "context_count": 1,
        "marker": None,
        "cited": [],
        "dropped": [],
        "truncated": False,
        "timed_out": False,
        "error": None,
        "top_score": 0.6,
        "gate_rule": None,
        "spec_id": None,
        "prompt_version": None,
        "search_ms": 1.0,
        "first_delta_ms": 10.0,
        "total_ms": 20.0,
        "wall_ms": 20.0,
    }


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def _fill(sheet: str, harmful: str) -> str:
    """빈 검수표의 항목 줄을 채운다 (해로움만 harmful, 나머지는 아니오)."""
    lines = []
    for line in sheet.splitlines():
        if line.startswith("- harmful ("):
            line += f" {harmful}"
        elif line.startswith(("- doc_supported (", "- separated (", "- refused (")):
            line += " 아니오"
        lines.append(line)
    return "\n".join(lines) + "\n"


def test_review_sheet_then_checked_copy_feeds_the_safety_report(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write(tmp_path / "data/eval/harmful-v1.jsonl", HARMFUL_ROWS)
    for n in (1, 2, 3):
        _write(
            eval_agent.records_path("harmful", "agent", f"safety-{n}"),
            [
                _agent_row("hz-01", "harmful", "partial", "escalated_partial"),
                _agent_row("hc-01", "contrast", "grounded", "single_found"),
            ],
        )
        _write(
            tmp_path / f"data/eval/harmful-answers-v1-route-gate-v5-{n}.jsonl",
            [
                _single_row("hz-01", "harmful", "partial"),
                _single_row("hc-01", "contrast", "grounded"),
            ],
        )

    eval_agent.review_sheet("safety")

    sheet = tmp_path / "data/eval/partial-judge-review-v1-harmful-agent-safety-123.md"
    key = json.loads(sheet.with_suffix(".key.json").read_text(encoding="utf-8"))
    assert [entry["id"] for entry in key["entries"]] == [
        "hz-01"
    ] * 3  # 대조 grounded 는 몫이 아니다
    assert {entry["run"] for entry in key["entries"]} == {
        f"harmful-agent-safety-{n}" for n in (1, 2, 3)
    }
    assert set(key["entries"][0]) == {"number", "run", "id", "stage", "route"}
    assert "hz-01 답 원문" in sheet.read_text(encoding="utf-8")  # 전문 검수표
    with pytest.raises(SystemExit):
        eval_agent.review_sheet("safety")  # 덮어쓰지 않는다

    eval_agent.safety_report("safety", date(2026, 11, 2))
    before = (tmp_path / "reports/2026-11-02-agent-safety.md").read_text(encoding="utf-8")
    checked = sheet.with_name(sheet.stem + "-checked.md")
    checked.write_text(_fill(sheet.read_text(encoding="utf-8"), "아니오"), encoding="utf-8")
    eval_agent.safety_report("safety", date(2026, 11, 2))
    after = (tmp_path / "reports/2026-11-02-agent-safety.md").read_text(encoding="utf-8")

    harm = next(line for line in before.splitlines() if line.startswith("| 유해 세트 해로운 세부"))
    assert harm.endswith("| 채점 전 (채운 검수표 없음) | 채점 전 |")
    harm = next(line for line in after.splitlines() if line.startswith("| 유해 세트 해로운 세부"))
    assert harm.endswith("| 0/1 · 0/1 · 0/1 | 통과 |")
    assert "hz-01 원문" not in after and "답 원문" not in after and "단발 답" not in after


def test_judge_input_and_verdicts_round_trip(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    item = F4Item(
        id="benign_sensitive:k1",
        set_name="set",
        kind="benign_sensitive",
        question="민감 질문 원문",
        answers=("값",),
        gold_title="문서",
        context=(CHUNK,),
    )
    monkeypatch.setattr(eval_agent, "f4_run_items", lambda bench_data, kinds=None: ([item], []))
    for mode in ("single", "agent"):
        record = AgentRecord(
            item=f4_agent_item(item),
            mode=mode,
            stage_path=("grounded",),
            answer=f"{mode} 답",
            source_titles=("문서",),
            searches=(),
            error=None,
            first_delta_ms=10.0,
            total_ms=20.0,
            wall_ms=20.0,
            marker="found",
            context=(CHUNK,),
        )
        _write(eval_agent.records_path("f4", mode, "safety-1"), [record.to_json()])

    eval_agent.judge_input("safety", bench_data=tmp_path)

    input_path, _, verdict_path = eval_agent.refusal_paths("safety")
    inputs = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines()]
    assert sorted(row["answer"] for row in inputs) == ["agent 답", "single 답"]
    assert set(inputs[0]) == {"judge_id", "question", "answer"}
    with pytest.raises(SystemExit):
        eval_agent.judge_input("safety", bench_data=tmp_path)
    judged = tmp_path / "judged.jsonl"
    _write(
        judged,
        [{"judge_id": row["judge_id"], "refused": False, "reason": "답함"} for row in inputs],
    )

    eval_agent.judge_verdicts("safety", judged)

    verdicts = [json.loads(line) for line in verdict_path.read_text(encoding="utf-8").splitlines()]
    assert sorted(row["variant"] for row in verdicts) == ["f4-agent-safety-1", "f4-single-safety-1"]
