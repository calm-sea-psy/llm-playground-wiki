import asyncio
import json
from pathlib import Path

import pytest

from playground_wiki.evaluation.answer_run import EvalItem
from playground_wiki.evaluation.f4_prose import F4Item, StoredSearch, load_f4_items


def _row(item_id: str, kind: str, **extra: object) -> dict:
    # 저장 순서와 순위가 엇갈려도 순위 순으로 읽어야 한다
    return {
        "id": item_id,
        "kind": kind,
        "question": f"{item_id} 질문?",
        "answers": ["정답", "정답 표기"],
        "gold_title": "문서",
        "strata": "x",
        "context": [
            {
                "chunk_id": "12-3",
                "rank": 2,
                "score": 0.4,
                "title": "다른",
                "section": "",
                "text": "다른 본문",
            },
            {
                "chunk_id": "7-0",
                "rank": 1,
                "score": 0.97,
                "title": "문서",
                "section": "역사",
                "text": "정답 본문",
            },
        ],
        **extra,
    }


def _write_set(root: Path, name: str, rows: list[dict]) -> None:
    (root / name).mkdir(parents=True)
    lines = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    (root / name / "items.jsonl").write_text(lines, encoding="utf-8")


def test_item_reads_context_in_rank_order_and_becomes_a_search_result():
    item = F4Item.from_json(_row("korquad-nodoc:1", "no_doc"), "set-a")

    result = item.search_result()

    assert [chunk.chunk_id for chunk in item.context] == ["7-0", "12-3"]
    assert [(hit.chunk.doc_id, hit.chunk.seq, hit.score) for hit in result.hits] == [
        ("7", 0, 0.97),
        ("12", 3, 0.4),
    ]
    assert result.hits[0].chunk.text == "정답 본문"


def test_item_becomes_an_f4_eval_item_with_kind_as_category():
    item = F4Item.from_json(_row("korquad-nodoc:1", "no_doc"), "set-a")

    assert item.eval_item() == EvalItem(
        id="korquad-nodoc:1",
        source="f4",
        question="korquad-nodoc:1 질문?",
        gold_title="문서",
        gold_answer="정답",
        category="no_doc",
    )


def test_item_needs_kind_specific_fields():
    with pytest.raises(ValueError, match="original_answers"):
        F4Item.from_json(_row("counterfactual:1", "counterfactual"), "set-a")
    with pytest.raises(ValueError, match="canary"):
        F4Item.from_json(_row("injection_direct:1", "injection_direct"), "set-a")
    with pytest.raises(ValueError, match="unknown F4 kind"):
        F4Item.from_json(_row("sensitive:1", "sensitive"), "set-a")


def test_f4_eval_item_needs_gold():
    with pytest.raises(ValueError, match="gold_title and gold_answer"):
        EvalItem(id="x", source="f4", question="q?")


def test_load_picks_kinds_per_set_and_records_hash_and_counts(tmp_path: Path):
    _write_set(
        tmp_path,
        "set-a",
        [
            _row("korquad:1", "answerable"),
            _row("korquad-nodoc:1", "no_doc"),
            _row("lt:1", "longtail"),
        ],
    )
    _write_set(tmp_path, "set-b", [_row("cf:1", "counterfactual", original_answers=["원래"])])

    items, sets = load_f4_items(
        tmp_path, {"set-a": frozenset({"no_doc"}), "set-b": frozenset({"counterfactual"})}
    )

    assert [(item.id, item.set_name) for item in items] == [
        ("korquad-nodoc:1", "set-a"),
        ("cf:1", "set-b"),
    ]
    assert [dict(f4_set.counts) for f4_set in sets] == [{"no_doc": 1}, {"counterfactual": 1}]
    assert len(sets[0].sha256) == 64
    assert json.dumps(sets[0].to_json()) == json.dumps(
        {"name": "set-a", "sha256": sets[0].sha256, "counts": {"no_doc": 1}}
    )
    with pytest.raises(TypeError):
        sets[0].counts["no_doc"] = 2  # 읽기 전용인지 본다


def test_load_refuses_missing_kinds_and_duplicate_ids(tmp_path: Path):
    _write_set(tmp_path, "set-a", [_row("x:1", "no_doc")])
    _write_set(tmp_path, "set-b", [_row("x:1", "no_fact")])

    with pytest.raises(ValueError, match="no items of kind"):
        load_f4_items(tmp_path, {"set-a": frozenset({"no_fact"})})
    with pytest.raises(ValueError, match="unique"):
        load_f4_items(tmp_path, {"set-a": frozenset({"no_doc"}), "set-b": frozenset({"no_fact"})})


def test_stored_search_returns_the_item_in_use_and_checks_the_question():
    first = F4Item.from_json(_row("a:1", "no_doc"), "set-a")
    second = F4Item.from_json(_row("a:2", "no_doc"), "set-a")
    search = StoredSearch()

    with pytest.raises(RuntimeError, match="use"):
        asyncio.run(search.search(first.question))
    search.use(second)
    with pytest.raises(ValueError, match="a:2"):
        asyncio.run(search.search(first.question))

    assert asyncio.run(search.search(second.question)) == second.search_result()
