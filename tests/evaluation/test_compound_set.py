import json
from pathlib import Path
from typing import Literal, get_args

from playground_evalkit import squash

# 복합 질문 세트 — 근거 문장이 덤프 본문에 있는지는 덤프가 있어야 해서 여기서 재지 않는다 (compound-v1.md)
SET = Path(__file__).parent / "compound-v1.jsonl"

Kind = Literal["bridge", "comparison", "intersection", "aggregation", "temporal"]
KINDS: tuple[str, ...] = get_args(Kind)
KIND_COUNTS: dict[str, int] = {
    "bridge": 20,
    "comparison": 12,
    "intersection": 8,
    "aggregation": 5,
    "temporal": 5,
}
TOP_DOCS = 10  # 한 문서 검사는 질문 그대로 BM25 상위 10 문서


def _rows() -> list[dict]:
    return [json.loads(line) for line in SET.read_text(encoding="utf-8").splitlines() if line]


def _document_hops(row: dict) -> list[dict]:
    return [hop for hop in row["hops"] if hop["title"] is not None]


def test_set_has_fifty_unique_ids():
    ids = [row["id"] for row in _rows()]

    assert len(ids) == 50
    assert len(set(ids)) == len(ids)


def test_kind_counts_cover_every_kind():
    assert set(KIND_COUNTS) == set(KINDS)


def test_kinds_are_known_and_counted():
    rows = _rows()

    assert {row["kind"] for row in rows} <= set(KINDS)
    for kind, count in KIND_COUNTS.items():
        assert sum(row["kind"] == kind for row in rows) == count, kind


def test_every_item_joins_two_or_more_documents():
    for row in _rows():
        hops = _document_hops(row)

        assert len(row["hops"]) >= 2, row["id"]
        assert len({hop["title"] for hop in hops}) >= 2, row["id"]


def test_document_hops_have_evidence_and_calculation_hops_do_not():
    for row in _rows():
        for hop in row["hops"]:
            if hop["title"] is None:
                assert hop["evidence"] is None, row["id"]
            else:
                assert hop["evidence"].strip(), row["id"]


def test_answers_are_not_empty():
    for row in _rows():
        assert row["answer"].strip(), row["id"]
        assert all(alias.strip() for alias in row["aliases"]), row["id"]
        assert all(hop["answer"].strip() for hop in row["hops"]), row["id"]
        assert all(hop["sub_question"].strip() for hop in row["hops"]), row["id"]


def test_no_single_document_answers_any_question():
    for row in _rows():
        check = row["single_doc_check"]

        assert check["answerable_from_one"] is False, row["id"]
        assert len(check["bm25_top10"]) == TOP_DOCS, row["id"]


def test_bridge_question_does_not_leak_first_hop_answer():
    for row in _rows():
        if row["kind"] != "bridge":
            continue

        assert squash(row["hops"][0]["answer"]) not in squash(row["question"]), row["id"]
