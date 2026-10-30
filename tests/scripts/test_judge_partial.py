"""scripts/judge_partial.py 의 세트 검사 · 정책 · 표본 · 일치율 · 리포트를 합성 입력으로 — 데이터 · 모델 없이 돈다."""

import importlib.util
import json
from collections import Counter
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest
from playground_corpus import Chunk
from playground_llm import ModelSpec, Think
from playground_llm.fake import FakeLLM
from playground_retrieval.core import Hit

from playground_wiki.evaluation.answer_run import AnswerRecord, ContextChunk, EvalItem
from playground_wiki.evaluation.judge_review import Agreement
from playground_wiki.evaluation.partial_judge import DOC_HEADING, GENERAL_HEADING
from playground_wiki.generation import Profile, Thresholds

_SCRIPT = Path(__file__).parents[2] / "scripts" / "judge_partial.py"
_spec = importlib.util.spec_from_file_location("judge_partial", _SCRIPT)
assert _spec is not None
assert _spec.loader is not None
judge_partial = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(judge_partial)

QUESTION_MARK = "질문원문표지"
ANSWER_MARK = "답원문표지"
CHUNK_MARK = "청크원문표지"
REASON_MARK = "판정근거표지"
PROFILE = Profile(
    ModelSpec("ollama", "gemma4:12b", {"think": Think.OFF}),
    system=("base@v4", "partial@v2"),
    user="question@v1",
    timeout_sec=60,
)
CHUNK = ContextChunk(
    chunk_id="1-0", title="문서", section="", text=f"문서\n{CHUNK_MARK}", score=0.5
)
BASE = AnswerRecord(
    item=EvalItem(id="x", source="offwiki", question=f"{QUESTION_MARK}?", expected="partial"),
    stage_path=("partial",),
    answer=f"{DOC_HEADING}\n{ANSWER_MARK}[1]\n{GENERAL_HEADING}\n알 수 없습니다.",
    hits=(CHUNK,),
    context_count=1,
    marker=None,
    cited=(1,),
    dropped=(),
    truncated=False,
    timed_out=False,
    error=None,
    top_score=0.5,
    gate_rule=None,
    spec_id="ollama:fake",
    prompt_version="base@v4:x+partial@v2:y+question@v1:z",
    search_ms=50.0,
    first_delta_ms=900.0,
    total_ms=2000.0,
    wall_ms=2001.0,
    done_reason={"partial": "stop"},
)
VERDICT = {
    "doc_supported": True,
    "separated": True,
    "harmful": False,
    "refused": True,
    "reason": REASON_MARK,
}


def _set_row(row_id: str, kind: str, pair: str, question: str = "질문?") -> dict:
    return {
        "id": row_id,
        "kind": kind,
        "pair": pair,
        "topic": "주제",
        "question": question,
        "related_title": None,
    }


def _record(item_id: str, *, source: str = "harmful", category: str | None = None, **changes):
    item = EvalItem(
        id=item_id,
        source=source,
        question=f"{QUESTION_MARK} {item_id}?",
        category=category,
        expected=changes.pop("expected", None),
        gold_title=changes.pop("gold_title", None),
        gold_answer=changes.pop("gold_answer", None),
    )
    return replace(BASE, item=item, **changes)


def _hit(score: float) -> Hit:
    chunk = Chunk(id="1-0", doc_id="1", title="문서", section="", text="문서", seq=0, token_count=1)
    return Hit(chunk=chunk, score=score)


def _judged(run: str, item_id: str, **verdict) -> dict:
    return {
        "run": run,
        "id": item_id,
        "judge": "ollama:gpt-oss:20b",
        "judge_prompt": judge_partial.JUDGE_PROMPT_ID,
        "format": {},
        "verdict": {**VERDICT, **verdict},
        "error": None,
    }


def test_answer_path_keeps_the_base_record_and_tags_a_rerun():
    base = judge_partial.RUNS["harmful-route"]

    assert judge_partial.answer_path("route", None) == base
    tagged = judge_partial.answer_path("route", "grounded-v3")
    assert tagged.parent == base.parent
    assert tagged.name == "harmful-answers-v1-route-grounded-v3.jsonl"


def test_harmful_items_turn_paired_rows_into_items():
    rows = [
        _set_row("hz-01", "harmful", "ct-01"),
        _set_row("ct-01", "contrast", "hz-01", " 대조? "),
    ]

    items = judge_partial.harmful_items(rows)

    assert [(item.id, item.source, item.category, item.question) for item in items] == [
        ("hz-01", "harmful", "harmful", "질문?"),
        ("ct-01", "harmful", "contrast", "대조?"),
    ]


def test_harmful_items_report_every_problem_with_its_id():
    rows = [
        _set_row("hz-01", "harmful", "ct-01", question=" "),  # 빈 질문
        _set_row("ct-01", "contrast", "hz-01"),
        _set_row("hz-02", "harmful", "hz-03"),  # 같은 kind 끼리 짝
        _set_row("hz-03", "harmful", "hz-02"),
        _set_row("hz-04", "danger", "missing"),  # 모르는 kind · 없는 짝
    ]

    with pytest.raises(ValueError) as caught:
        judge_partial.harmful_items(rows)

    message = str(caught.value)
    assert "hz-01: question" in message
    assert "hz-02: pair" in message
    assert "hz-04: kind" in message
    assert "hz-04: pair" in message
    assert "ct-01" not in message


def test_harmful_items_reject_duplicate_ids():
    rows = [_set_row("hz-01", "harmful", "ct-01"), _set_row("hz-01", "contrast", "hz-01")]

    with pytest.raises(ValueError, match="hz-01: id"):
        judge_partial.harmful_items(rows)


def test_partial_only_policy_sends_every_score_to_partial_with_production_context():
    policy = judge_partial.PartialOnlyPolicy(
        Thresholds(high=0.8, low=0.1), {"grounded": replace(PROFILE), "partial": PROFILE}
    )

    high = policy.decide([_hit(0.95), _hit(0.05)])
    low = policy.decide([_hit(0.05)])

    assert (high.stage, high.profile, [hit.score for hit in high.context]) == (
        "partial",
        PROFILE,
        [0.95],
    )
    assert (low.stage, low.profile, low.context) == ("partial", PROFILE, ())


def test_f9_review_keys_follow_group_sizes():
    records = [
        *(
            _record(f"k{i}", source="korquad", gold_title="문서", gold_answer="답")
            for i in range(6)
        ),
        *(_record(f"p{i}", source="offwiki", expected="partial") for i in range(9)),
        *(_record(f"r{i}", source="offwiki", expected="refuse") for i in range(3)),
    ]

    keys = judge_partial.f9_review_keys(records, 6, seed=0)

    assert Counter(key[0] for key in keys) == {"k": 2, "p": 3, "r": 1}


def test_harmful_review_keys_take_both_members_of_answered_pairs():
    pairs = {"hz-01": "ct-01", "ct-01": "hz-01", "hz-02": "ct-02", "ct-02": "hz-02"}
    pairs |= {"hz-03": "ct-03", "ct-03": "hz-03"}
    records = [
        _record("hz-01", category="harmful"),
        _record("ct-01", category="contrast"),
        _record("hz-02", category="harmful"),  # 짝 ct-02 의 partial 답이 없다
        _record("hz-03", category="harmful"),
        _record("ct-03", category="contrast"),
    ]

    keys = judge_partial.harmful_review_keys(records, pairs, 5, seed=0)

    assert sorted(keys) == ["ct-01", "ct-03", "hz-01", "hz-03"]


def test_judged_row_keeps_format_and_verdict():
    llm = FakeLLM([json.dumps(VERDICT, ensure_ascii=False)])

    row = judge_partial.judged_row("f9", BASE, llm)

    assert (row["run"], row["id"], row["error"]) == ("f9", "x", None)
    assert row["format"]["has_doc"] is True
    assert row["verdict"]["refused"] is True
    assert CHUNK_MARK in llm.calls[0].messages[1].content


def test_stale_judgements_are_the_ones_from_another_judge_prompt():
    current = _judged("f9", "a")
    stale = {**_judged("f9", "b"), "judge_prompt": "11111111"}

    assert judge_partial._current_prompt(current)
    assert not judge_partial._current_prompt(stale)
    assert not judge_partial._current_prompt({"run": "f9", "id": "c"})  # 지시 해시가 없던 기록


def test_judged_row_records_a_bad_reply_as_error():
    row = judge_partial.judged_row("f9", BASE, FakeLLM(["판정할 수 없습니다"]))

    assert row["verdict"] is None
    assert "not JSON" in str(row["error"])


def test_review_keys_collect_both_sheets_from_their_key_files(tmp_path, monkeypatch):
    paths = {}
    for part, entries in (
        ("f9", [{"number": 1, "run": "f9", "id": "a"}, {"number": 2, "run": "f9", "id": "b"}]),
        ("harmful", [{"number": 1, "run": "harmful-forced", "id": "hz-01"}]),
    ):
        sheet = tmp_path / f"review-{part}.md"
        sheet.with_suffix(".key.json").write_text(
            json.dumps({"entries": entries}), encoding="utf-8"
        )
        paths[part] = sheet
    paths["missing"] = tmp_path / "review-missing.md"  # 아직 안 만든 검수표
    monkeypatch.setattr(judge_partial, "REVIEW_PATHS", paths)

    assert judge_partial.review_keys() == {
        ("f9", "a"),
        ("f9", "b"),
        ("harmful-forced", "hz-01"),
    }


def test_agreements_for_match_sheet_numbers_to_judged_answers():
    entries = [
        {"number": 1, "run": "f9", "id": "a"},
        {"number": 2, "run": "f9", "id": "b"},
        {"number": 3, "run": "f9", "id": "c"},  # 판정이 없다
    ]
    judged = {
        ("f9", "a"): _judged("f9", "a", harmful=True),
        ("f9", "b"): _judged("f9", "b"),
    }
    ratings = {
        1: {"doc_supported": True, "separated": True, "harmful": True, "refused": None},
        2: {"doc_supported": True, "separated": False, "harmful": True, "refused": True},
        3: {"doc_supported": True, "separated": True, "harmful": True, "refused": True},
    }

    tables = judge_partial.agreements_for(ratings, entries, judged)

    assert tables["harmful"] == Agreement(both=1, human_only=1, judge_only=0, neither=0)
    assert tables["separated"] == Agreement(both=1, human_only=0, judge_only=1, neither=0)
    assert tables["refused"] == Agreement(both=1, human_only=0, judge_only=0, neither=0)


def test_render_report_counts_without_question_answer_chunk_or_reason_text():
    f9 = [
        _record("k1", source="korquad", gold_title="문서", gold_answer="답"),
        _record("p1", source="offwiki", expected="partial"),
        _record("r1", source="offwiki", expected="refuse"),
        _record("g1", source="offwiki", expected="refuse", stage_path=("gate",)),
    ]
    forced = [
        _record("hz-01", category="harmful"),
        _record("ct-01", category="contrast"),
    ]
    judged = {
        ("f9", "k1"): _judged("f9", "k1"),
        ("f9", "p1"): _judged("f9", "p1", separated=False),
        ("harmful-forced", "hz-01"): _judged("harmful-forced", "hz-01", harmful=True),
        ("harmful-forced", "ct-01"): _judged("harmful-forced", "ct-01", refused=False),
    }
    agreements = {"f9": {key: Agreement(1, 0, 0, 1) for key in judge_partial.JUDGE_ITEMS}}
    # 사람 채점은 판정기와 엇갈리게 둔다 — 해로움의 기준 수치가 사람 쪽에서 나오는지 보려고
    human = {
        ("harmful-forced", "hz-01"): {"harmful": False, "refused": True},
        ("harmful-forced", "ct-01"): {"harmful": False, "refused": False},
    }

    text = judge_partial.render_report(
        {"f9": f9, "harmful-route": [], "harmful-forced": forced},
        judged,
        agreements,
        human,
        today=date(2026, 10, 29),
    )

    for mark in (QUESTION_MARK, ANSWER_MARK, CHUNK_MARK, REASON_MARK):
        assert mark not in text
    assert "| **전체** | 3 | 3/3 (100.0%) | 0/3 (0.0%) |" in text
    assert "1/2 (50.0%)" in text  # F9 나눠 씀 — 판정이 있는 두 답 중 하나
    assert "아직 돌리지 않았다 (`answer --mode route`)" in text
    assert "| harmful | 1 | 1/1 (100.0%) |" in text  # 판정기
    assert "| harmful | 1 | 0/1 (0.0%) | 97.5% | 1/1 (100.0%) |" in text  # 사람 채점
    assert "| f9 | 나눠 씀 | 2 | 100.0% |" in text


def test_render_report_says_when_review_sheets_are_not_filled():
    text = judge_partial.render_report({}, {}, {}, {}, today=date(2026, 10, 29))

    assert "검수표를 아직 채우지 않았다" in text
    assert "기록이 없다" in text
