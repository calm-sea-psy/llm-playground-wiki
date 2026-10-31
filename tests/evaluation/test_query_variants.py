from collections.abc import Mapping, Sequence

import numpy as np
import pytest
from numpy.typing import NDArray
from playground_corpus import Chunk
from playground_retrieval.core import Filters, Hit
from playground_retrieval.search import HybridRetriever, Reranker

from playground_wiki.evaluation.query_rewrite import Rewrite
from playground_wiki.evaluation.query_variants import (
    VARIANTS,
    QueryItem,
    VariantRecord,
    VariantSearcher,
    VariantSettings,
    compound_items,
    longtail_query_items,
    offwiki_items,
    paraphrase_items,
    query_plan,
)

QUESTION = "경복궁을 다시 지은 사람은?"
REWRITTEN = "경복궁 중건 인물"
MULTI = ("경복궁 중건을 이끈 사람", "경복궁 재건 주도")
SETTINGS = VariantSettings(candidates=4, rrf_k=60, rerank_candidates=3, top_k=2)


def _chunk(chunk_id: str, title: str = "경복궁", text: str = "") -> Chunk:
    return Chunk(
        id=chunk_id,
        doc_id=chunk_id.split("-", maxsplit=1)[0],
        title=title,
        section="",
        text=text or f"{title} 본문 {chunk_id}",
        seq=0,
        token_count=1,
    )


CHUNKS = {
    "a-0": _chunk("a-0"),
    "b-0": _chunk("b-0", "광화문"),
    "c-0": _chunk("c-0", "근정전"),
    "d-0": _chunk("d-0", "경복궁", "경복궁 > 역사\n흥선대원군이 중건을 이끌었다"),
    "e-0": _chunk("e-0", "창덕궁"),
    "f-0": _chunk("f-0", "경회루"),
}


def _hits(*chunk_ids: str) -> list[Hit]:
    return [
        Hit(chunk=CHUNKS[chunk_id], score=1.0 / rank)
        for rank, chunk_id in enumerate(chunk_ids, start=1)
    ]


class FakeRetriever:
    """질의마다 정해 둔 순위를 돌려주고 받은 호출을 남기는 검색기."""

    def __init__(self, rankings: Mapping[str, Sequence[str]]) -> None:
        self._rankings = rankings
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, k: int, filters: Filters | None = None) -> list[Hit]:
        del filters  # 범위 제한은 쓰지 않는다
        self.calls.append((query, k))
        return _hits(*self._rankings.get(query, ()))[:k]


class FakeScorer:
    """(질의, 청크 id) 마다 정해 둔 점수 — 없으면 0. 받은 쌍을 남긴다."""

    def __init__(self, scores: Mapping[tuple[str, str], float]) -> None:
        self._scores = scores
        self.calls: list[list[tuple[str, str]]] = []

    def score(self, pairs: Sequence[tuple[str, str]]) -> NDArray[np.float32]:
        self.calls.append(list(pairs))
        by_text = {chunk.text: chunk_id for chunk_id, chunk in CHUNKS.items()}
        return np.asarray(
            [self._scores.get((query, by_text[text]), 0.0) for query, text in pairs],
            dtype=np.float32,
        )


ITEM = QueryItem(id="k1", question=QUESTION, gold_titles=("경복궁",), answers=("흥선대원군",))
REWRITE = Rewrite(
    id="k1",
    question=QUESTION,
    rewrite=REWRITTEN,
    queries=MULTI,
    rewrite_ms=120.0,
    multi_ms=180.0,
)
DENSE = FakeRetriever(
    {
        QUESTION: ["a-0", "b-0", "c-0"],
        REWRITTEN: ["d-0", "a-0"],
        MULTI[0]: ["e-0", "d-0"],
        MULTI[1]: ["f-0"],
    }
)
BM25 = FakeRetriever(
    {QUESTION: ["b-0", "a-0"], REWRITTEN: ["d-0"], MULTI[0]: ["e-0"], MULTI[1]: ["f-0", "a-0"]}
)
# 원문 질문 점수: d 가 가장 높지만 원문 검색 후보(a · b · c)에는 없다
ORIGINAL_SCORES = {
    (QUESTION, "a-0"): 0.3,
    (QUESTION, "b-0"): 0.5,
    (QUESTION, "c-0"): 0.1,
    (QUESTION, "d-0"): 0.9,
}


def _searcher(scores: Mapping[tuple[str, str], float] = ORIGINAL_SCORES) -> VariantSearcher:
    return VariantSearcher(
        FakeRetriever(DENSE._rankings),
        FakeRetriever(BM25._rankings),
        FakeScorer(scores),
        settings=SETTINGS,
    )


def _by_variant(results: Sequence) -> dict:
    return {result.variant: result for result in results}


def test_search_returns_every_variant_in_order():
    results = _searcher().search(ITEM, REWRITE)

    assert tuple(result.variant for result in results) == VARIANTS


def test_original_matches_the_product_hybrid_and_reranker():
    dense, bm25 = FakeRetriever(DENSE._rankings), FakeRetriever(BM25._rankings)
    product = Reranker(
        HybridRetriever([dense, bm25], rrf_k=SETTINGS.rrf_k, candidates=SETTINGS.candidates),
        FakeScorer(ORIGINAL_SCORES),
        candidates=SETTINGS.rerank_candidates,
    ).search(QUESTION, SETTINGS.top_k)

    original = _by_variant(_searcher().search(ITEM, REWRITE))["original"]

    assert original.chunk_ids == tuple(hit.chunk.id for hit in product)
    assert original.scores == pytest.approx([hit.score for hit in product])
    assert original.gold_rank is None
    assert original.queries == 1
    assert original.llm_ms == 0.0


def test_rewrite_adds_candidates_but_ranks_with_the_original_question():
    searcher = _searcher()

    rewrite = _by_variant(searcher.search(ITEM, REWRITE))["rewrite"]

    assert rewrite.chunk_ids == ("d-0", "b-0")
    assert rewrite.gold_rank == 1
    assert rewrite.top_score == pytest.approx(0.9)
    assert rewrite.queries == 2
    assert rewrite.pairs == 4  # 원문 후보 a · b · c + 고친 질의 후보 d (a 는 겹침)
    assert rewrite.llm_ms == REWRITE.rewrite_ms
    rewrite_pairs = searcher._scorer.calls[1]
    assert {query for query, _ in rewrite_pairs} == {QUESTION}


def test_rewrite_max_takes_the_higher_of_original_and_rewritten_scores():
    scores = {**ORIGINAL_SCORES, (REWRITTEN, "c-0"): 0.95, (QUESTION, "d-0"): 0.2}

    best = _by_variant(_searcher(scores).search(ITEM, REWRITE))["rewrite_max"]

    assert best.chunk_ids == ("c-0", "b-0")
    assert best.scores == pytest.approx([0.95, 0.5])
    assert best.pairs == 8


def test_multi_unions_the_hybrid_candidates_of_every_query():
    multi = _by_variant(_searcher().search(ITEM, REWRITE))["multi"]

    # 원문 a · b · c, 여러 질의 1 의 e · d, 여러 질의 2 의 f (a 는 겹침) — 고친 질의(②)는 쓰지 않는다
    assert multi.pairs == 6
    assert multi.queries == 3
    assert multi.chunk_ids == ("d-0", "b-0")
    assert multi.llm_ms == REWRITE.multi_ms


def test_fusion_is_rrf_over_query_by_retriever_rankings_without_reranker():
    fusion = _by_variant(_searcher().search(ITEM, REWRITE))["fusion"]

    # a: 원문 Dense 1 · BM25 2, 여러 질의 2 BM25 2 / b: 원문 Dense 2 · BM25 1 / e: 여러 질의 1 Dense 1 · BM25 1
    expected = {
        "a-0": 1 / 61 + 1 / 62 + 1 / 62,
        "b-0": 1 / 62 + 1 / 61,
        "e-0": 1 / 61 + 1 / 61,
    }
    assert fusion.chunk_ids == ("a-0", "e-0")
    assert fusion.scores == pytest.approx([expected["a-0"], expected["e-0"]])
    assert fusion.pairs == 0
    assert fusion.rerank_ms == 0.0


def test_fusion_rerank_reranks_the_rrf_top_candidates_with_the_original_question():
    scores = {**ORIGINAL_SCORES, (QUESTION, "e-0"): 0.7, (QUESTION, "f-0"): 0.4}

    reranked = _by_variant(_searcher(scores).search(ITEM, REWRITE))["fusion_rerank"]

    # RRF 상위 3 은 a · e · f — b(0.5) · d(0.9) 는 RRF 3위 안에 들지 못해 원문 점수가 높아도 후보가 아니다
    assert reranked.pairs == SETTINGS.rerank_candidates
    assert reranked.chunk_ids == ("e-0", "f-0")
    assert reranked.scores == pytest.approx([0.7, 0.4])


def test_fusion_with_only_the_original_query_keeps_the_hybrid_order():
    searcher = _searcher()
    plain = Rewrite(id="k1", question=QUESTION, rewrite=None, queries=(), rewrite_ms=0, multi_ms=0)

    results = _by_variant(searcher.search(ITEM, plain))

    hybrid = HybridRetriever(
        [FakeRetriever(DENSE._rankings), FakeRetriever(BM25._rankings)],
        rrf_k=SETTINGS.rrf_k,
        candidates=SETTINGS.candidates,
    ).search(QUESTION, SETTINGS.top_k)
    assert results["fusion"].chunk_ids == tuple(hit.chunk.id for hit in hybrid)
    assert results["rewrite"].chunk_ids == results["original"].chunk_ids
    assert results["multi"].chunk_ids == results["original"].chunk_ids


def test_each_query_is_retrieved_once_per_retriever():
    searcher = _searcher()
    repeated = Rewrite(
        id="k1", question=QUESTION, rewrite=MULTI[0], queries=MULTI, rewrite_ms=1, multi_ms=1
    )

    searcher.search(ITEM, repeated)

    assert searcher._dense.calls == [(QUESTION, 4), (MULTI[0], 4), (MULTI[1], 4)]
    assert searcher._bm25.calls == searcher._dense.calls


def test_query_plan_falls_back_to_the_question():
    assert query_plan(QUESTION, None).distinct == (QUESTION,)
    same = Rewrite(
        id="k1",
        question=QUESTION,
        rewrite=" 경복궁을  다시 지은 사람은? ",
        queries=(),
        rewrite_ms=1,
        multi_ms=2,
    )
    plan = query_plan(QUESTION, same)
    assert plan.rewrite_queries == (QUESTION,)
    assert plan.multi_queries == (QUESTION,)
    assert (plan.rewrite_ms, plan.multi_ms) == (1, 2)


def test_query_plan_rejects_a_rewrite_of_another_question():
    with pytest.raises(ValueError, match="different question"):
        query_plan("다른 질문", REWRITE)


def test_gold_rank_needs_the_gold_title_and_an_answer_string():
    other_doc = Hit(chunk=_chunk("x-0", "창덕궁", "흥선대원군"), score=1.0)
    gold = Hit(chunk=CHUNKS["d-0"], score=0.5)
    alias_item = QueryItem(
        id="k2", question="q", gold_titles=("경복궁",), answers=("없는 표기", "흥선 대원군")
    )

    assert ITEM.gold_rank([other_doc, gold]) == 2
    assert alias_item.gold_rank([gold]) == 1
    assert QueryItem(id="o", question="q").gold_rank([gold]) is None


def test_hops_found_counts_distinct_gold_documents():
    item = QueryItem(
        id="c1", question="q", gold_titles=("경복궁", "광화문", "종묘"), answers=("x",)
    )

    assert item.hops_found(_hits("a-0", "d-0", "b-0", "e-0")) == 2


def test_query_item_needs_titles_and_answers_together():
    with pytest.raises(ValueError, match="both"):
        QueryItem(id="x", question="q", gold_titles=("경복궁",))


def test_paraphrase_items_pick_the_question_text_and_first_answer():
    rows = [
        {
            "id": "p1",
            "title": "경복궁_(조선)",
            "question": "원문",
            "paraphrase": "바꾼 말",
            "answers": ["흥선대원군", "대원군"],
        }
    ]

    (original,) = paraphrase_items(rows, paraphrased=False)
    (paraphrased,) = paraphrase_items(rows, paraphrased=True)

    assert original.question == "원문"
    assert paraphrased.question == "바꾼 말"
    assert paraphrased.gold_titles == ("경복궁 (조선)",)
    assert paraphrased.answers == ("흥선대원군",)


def test_compound_items_take_every_hop_title_and_aliases():
    rows = [
        {
            "id": "c1",
            "question": "q",
            "answer": "서울",
            "aliases": ["한양"],
            "hops": [{"title": "조선_왕조"}, {"title": "경복궁"}, {"title": "조선 왕조"}],
        }
    ]

    (item,) = compound_items(rows)

    assert item.gold_titles == ("조선 왕조", "경복궁")
    assert item.answers == ("서울", "한양")


def test_offwiki_and_longtail_items():
    (offwiki,) = offwiki_items(
        [{"id": "ow-1", "question": "q", "expected": "refuse", "category": "personal"}]
    )
    longtail = longtail_query_items(
        [
            {
                "id": "l1",
                "kind": "longtail",
                "question": "q",
                "gold_title": "가",
                "answers": ["a", "b"],
                "strata": {"property": "p"},
            },
            {
                "id": "l2",
                "kind": "no_fact",
                "question": "q",
                "gold_title": "나",
                "answers": ["c"],
                "strata": {"property": "p"},
            },
        ]
    )

    assert not offwiki.has_gold
    assert offwiki.expected == "refuse"
    assert [item.id for item in longtail] == ["l1"]
    assert longtail[0].answers == ("a", "b")


def test_variant_record_round_trips_through_json():
    results = _searcher().search(ITEM, REWRITE)
    record = VariantRecord(id="k1", gate=None, results=results)

    assert VariantRecord.from_json(record.to_json()) == record
    assert VariantRecord.from_json(VariantRecord(id="g", gate="chitchat").to_json()).results == ()


def test_variant_record_result_raises_for_a_gated_item():
    with pytest.raises(KeyError):
        VariantRecord(id="g", gate="chitchat").result("original")
