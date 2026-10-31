from dataclasses import replace

import pytest
from playground_evalkit import KorQuADItem

from playground_wiki.evaluation import split_of
from playground_wiki.evaluation.answer_run import AnswerRecord, ContextChunk, EvalItem
from playground_wiki.evaluation.f9_sample import (
    extend_rows,
    longtail_items,
    sample_rows,
    select_items,
)

QUESTION_MARK = "질문원문표지"
ANSWER_MARK = "답원문표지"
CHUNK_MARK = "청크원문표지"


def _paragraphs_in(split: str, count: int) -> list[str]:
    keys = (f"문서{i}#0" for i in range(1000))
    return [key for key in keys if split_of(key) == split][:count]


def _item(qid: str, title: str, answer: str) -> KorQuADItem:
    return KorQuADItem(id=qid, title=title, question=f"{qid}?", answer=answer, source_title=title)


def _record(item: EvalItem, **fields: object) -> AnswerRecord:
    base = AnswerRecord(
        item=item,
        stage_path=("grounded",),
        answer=f"흥선대원군 {ANSWER_MARK} [1].",
        hits=(ContextChunk("1-0", "경복궁", "", f"흥선대원군 {CHUNK_MARK}", 0.97),),
        context_count=1,
        marker="found",
        cited=(1,),
        dropped=(),
        truncated=False,
        timed_out=False,
        error=None,
        top_score=0.97,
        gate_rule=None,
        spec_id="ollama:fake",
        prompt_version="base@v4:x+grounded@v2:y+question@v1:z",
        search_ms=50.0,
        first_delta_ms=900.0,
        total_ms=2000.0,
        wall_ms=2001.0,
        done_reason={"grounded": "stop"},
    )
    return replace(base, **fields)


def _records() -> list[AnswerRecord]:
    palace = EvalItem(
        id="k1",
        source="korquad",
        question=f"경복궁 {QUESTION_MARK}",
        gold_title="경복궁",
        gold_answer="흥선대원군",
    )
    personal = EvalItem(
        id="o1", source="offwiki", question=QUESTION_MARK, expected="refuse", category="personal"
    )
    no_llm = {"marker": None, "spec_id": None, "first_delta_ms": None, "done_reason": {}}
    return [
        _record(palace),
        # 표지 전 잘림 → partial (done.truncated 는 partial 것이라 False)
        _record(
            replace(palace, id="k2"),
            stage_path=("grounded", "partial"),
            marker="missing",
            top_score=0.9,
            done_reason={"grounded": "length", "partial": "stop"},
        ),
        _record(replace(palace, id="k3"), error="generation", total_ms=None, **no_llm),
        # 답을 보내던 중 한도에 닿음
        _record(replace(palace, id="k4"), truncated=True, done_reason={"grounded": "length"}),
        _record(personal, stage_path=("gate",), hits=(), top_score=None, **no_llm),
        # partial 이 빈 채 잘려 고정 거절 문구
        _record(
            replace(personal, id="o2"),
            stage_path=("partial",),
            top_score=0.3,
            cited=(),
            **{**no_llm, "done_reason": {"partial": "length"}},
        ),
    ]


def test_extend_rows_appends_the_revived_document_and_keeps_existing_rows():
    eval_a, eval_new = _paragraphs_in("eval", 2)
    existing = [
        {
            "id": "a",
            "source": "korquad",
            "question": "a?",
            "gold_title": "가",
            "gold_answer": "정답",
            "paragraph": eval_a,
        },
        {"id": "ow-001", "source": "offwiki", "question": "q"},
    ]
    items = [_item("n", "아가씨 (영화)", "박찬욱"), _item("a", "아가씨 (영화)", "정답")]
    paragraphs = {"n": eval_new, "a": eval_a}
    texts = {"아가씨 (영화)": "박찬욱 감독의 영화"}

    rows, added = extend_rows(existing, items, paragraphs, texts)

    assert added == 1
    assert [row["id"] for row in rows] == ["a", "n", "ow-001"]
    assert rows[0] == existing[0]  # 있는 줄은 그대로
    assert extend_rows(rows, items, paragraphs, texts) == (rows, 0)  # 두 번 돌려도 같다


def test_sample_rows_keep_eval_items_whose_answer_is_still_in_the_document():
    eval_a, eval_b, eval_c = _paragraphs_in("eval", 3)
    (bench,) = _paragraphs_in("bench", 1)
    items = [
        _item("a", "가", "정답"),  # 남음
        _item("b", "나", "옛 정답"),  # 문서가 고쳐져 정답 문장이 없다
        _item("c", "다", "정답"),  # 문서가 코퍼스에 없다
        _item("d", "가", "정답"),  # bench 몫
    ]
    paragraphs = {"a": eval_a, "b": eval_b, "c": eval_c, "d": bench}
    texts = {"가": "본문에 정 답이 있다", "나": "본문에 새 값만 있다"}
    offwiki = [
        {
            "id": "ow-001",
            "question": "q",
            "expected": "refuse",
            "category": "personal",
            "trap": False,
            "related_title": None,
            "note": "",
        }
    ]

    rows, picked = sample_rows(items, paragraphs, texts, offwiki)

    assert picked == 2  # a · b (c 는 문서가 없어 고르기 전에 빠진다)
    assert [(row["id"], row["source"]) for row in rows] == [("a", "korquad"), ("ow-001", "offwiki")]
    assert EvalItem.from_json(rows[0]).gold_answer == "정답"
    assert EvalItem.from_json(rows[1]).expected == "refuse"


def test_select_items_keeps_the_sample_order_and_rejects_unknown_ids():
    items = [record.item for record in _records()]

    picked = select_items(items, ["o1", "k2"])

    assert [item.id for item in picked] == ["k2", "o1"]
    with pytest.raises(ValueError, match="not in the sample"):
        select_items(items, ["k1", "ow-404"])


def test_longtail_items_keep_answered_items_with_every_gold_form():
    rows = [
        {
            "id": "no_fact:Q1-P577",
            "kind": "no_fact",
            "question": "q",
            "answers": ["1815년"],
            "gold_title": "가",
            "strata": {"property": "발표일"},
        },
        {
            "id": "longtail:Q2-P57",
            "kind": "longtail",
            "question": "q",
            "answers": ["마르셀 레르비에", "마르셀 샤를 레르비에"],
            "gold_title": "가행복",
            "strata": {"property": "감독"},
        },
    ]

    (item,) = longtail_items(rows)

    assert (item.id, item.source, item.category) == ("longtail:Q2-P57", "longtail", "감독")
    assert item.gold_answers == ("마르셀 레르비에", "마르셀 샤를 레르비에")
    assert (
        EvalItem.from_json(_record(item).to_json()["item"]) == item
    )  # 기록에서 다시 읽어도 표기가 남는다
