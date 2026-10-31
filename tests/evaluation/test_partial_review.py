import json
from collections import Counter
from dataclasses import replace

from playground_wiki.evaluation.answer_run import AnswerRecord, ContextChunk, EvalItem
from playground_wiki.evaluation.judge_review import Agreement
from playground_wiki.evaluation.partial_judge import DOC_HEADING, GENERAL_HEADING, JUDGE_PROMPT_ID
from playground_wiki.evaluation.partial_review import (
    ReviewSheet,
    agreements_for,
    f9_review_keys,
    harmful_review_keys,
    review_keys,
)

QUESTION_MARK = "질문원문표지"
ANSWER_MARK = "답원문표지"
CHUNK_MARK = "청크원문표지"
REASON_MARK = "판정근거표지"
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


def _judged(run: str, item_id: str, **verdict) -> dict:
    return {
        "run": run,
        "id": item_id,
        "judge": "ollama:gpt-oss:20b",
        "judge_prompt": JUDGE_PROMPT_ID,
        "format": {},
        "verdict": {**VERDICT, **verdict},
        "error": None,
    }


def test_f9_review_keys_follow_group_sizes():
    records = [
        *(
            _record(f"k{i}", source="korquad", gold_title="문서", gold_answer="답")
            for i in range(6)
        ),
        *(_record(f"p{i}", source="offwiki", expected="partial") for i in range(9)),
        *(_record(f"r{i}", source="offwiki", expected="refuse") for i in range(3)),
    ]

    keys = f9_review_keys(records, 6, seed=0)

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

    keys = harmful_review_keys(records, pairs, 5, seed=0)

    assert sorted(keys) == ["ct-01", "ct-03", "hz-01", "hz-03"]


def test_review_keys_collect_both_sheets_from_their_key_files(tmp_path):
    sheets = {}
    for part, entries in (
        ("f9", [{"number": 1, "run": "f9", "id": "a"}, {"number": 2, "run": "f9", "id": "b"}]),
        ("harmful", [{"number": 1, "run": "harmful-forced", "id": "hz-01"}]),
    ):
        sheet = tmp_path / f"review-{part}.md"
        sheet.with_suffix(".key.json").write_text(
            json.dumps({"entries": entries}), encoding="utf-8"
        )
        sheets[part] = ReviewSheet(sheet, "")
    sheets["missing"] = ReviewSheet(tmp_path / "review-missing.md", "")  # 안 만든 표

    assert review_keys(sheets=sheets) == {
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

    tables = agreements_for(ratings, entries, judged)

    assert tables["harmful"] == Agreement(both=1, human_only=1, judge_only=0, neither=0)
    assert tables["separated"] == Agreement(both=1, human_only=0, judge_only=1, neither=0)
    assert tables["refused"] == Agreement(both=1, human_only=0, judge_only=0, neither=0)
