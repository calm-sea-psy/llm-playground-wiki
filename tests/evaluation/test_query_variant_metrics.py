import pytest

from playground_wiki.evaluation.query_rewrite import Rewrite
from playground_wiki.evaluation.query_variant_metrics import (
    GROUNDED,
    rank_agreement,
    rewrite_flags,
    summarize,
)
from playground_wiki.evaluation.query_variants import (
    VARIANTS,
    QueryItem,
    Variant,
    VariantRecord,
    VariantResult,
)
from playground_wiki.generation import STAGES, Thresholds

RULE = Thresholds(high=0.8, low=0.1)
K = 5


def _result(
    variant: Variant,
    *,
    rank: int | None = None,
    score: float = 0.5,
    hops: int = 1,
    llm_ms: float = 0.0,
) -> VariantResult:
    return VariantResult(
        variant=variant,
        chunk_ids=("c",),
        titles=("t",),
        scores=(score,),
        gold_rank=rank,
        hops_found=hops,
        queries=1 if variant == "original" else 3,
        pairs=20,
        search_ms=10.0,
        rerank_ms=5.0,
        llm_ms=llm_ms,
    )


def _record(
    item_id: str, base: VariantResult, other: VariantResult, *, gate: str | None = None
) -> VariantRecord:
    results = [base] + [
        other if variant == other.variant else _result(variant) for variant in VARIANTS[1:]
    ]
    return VariantRecord(id=item_id, gate=gate, results=() if gate else tuple(results))


def _gold(item_id: str, titles: tuple[str, ...] = ("t",)) -> QueryItem:
    return QueryItem(id=item_id, question="q", gold_titles=titles, answers=("a",))


def test_grounded_matches_the_policy_stage_name():
    assert GROUNDED in STAGES


def test_summarize_counts_hits_mrr_and_pairs_with_the_baseline():
    items = {f"k{i}": _gold(f"k{i}") for i in range(4)}
    records = [
        _record("k0", _result("original", rank=1), _result("multi", rank=1)),
        _record("k1", _result("original", rank=None), _result("multi", rank=2)),
        _record("k2", _result("original", rank=3), _result("multi", rank=None)),
        _record("k3", _result("original", rank=None), _result("multi", rank=1)),
    ]

    summary = summarize(items, records, "multi", rule=RULE, k=K)
    baseline = summarize(items, records, "original", rule=RULE, k=K)

    assert (summary.hit1.count, summary.hit1.total) == (2, 4)
    assert summary.hitk.count == 3
    assert summary.mrr == pytest.approx((1 + 0.5 + 0 + 1) / 4)
    assert summary.hit1_test is not None
    assert (summary.hit1_test.only_a, summary.hit1_test.only_b) == (0, 1)
    assert summary.hitk_test is not None
    assert (summary.hitk_test.only_a, summary.hitk_test.only_b) == (1, 2)
    assert baseline.hit1_test is None
    assert baseline.mrr == pytest.approx((1 + 0 + 1 / 3 + 0) / 4)
    assert summary.queries == 3
    assert summary.pairs == 20


def test_summarize_leaves_out_gated_items():
    items = {"k0": _gold("k0"), "g": _gold("g")}
    records = [
        _record("k0", _result("original", rank=1), _result("rewrite", rank=1)),
        _record("g", _result("original"), _result("rewrite"), gate="chitchat"),
    ]

    summary = summarize(items, records, "rewrite", rule=RULE, k=K)

    assert summary.searched == 1
    assert summary.hit1.total == 1


def test_stage_moves_count_items_that_climb_to_grounded():
    items = {f"ow-{i}": QueryItem(id=f"ow-{i}", question="q", expected="partial") for i in range(3)}
    records = [
        _record("ow-0", _result("original", score=0.5), _result("rewrite", score=0.9)),
        _record("ow-1", _result("original", score=0.9), _result("rewrite", score=0.05)),
        _record("ow-2", _result("original", score=0.5), _result("rewrite", score=0.6)),
    ]

    summary = summarize(items, records, "rewrite", rule=RULE, k=K)

    assert summary.to_grounded == ("ow-0",)
    assert summary.from_grounded == ("ow-1",)
    assert summary.changed == 2
    assert dict(summary.stages) == {"grounded": 1, "refuse": 1, "partial": 1}
    assert summary.hit1.total == 0
    assert summary.mrr is None
    assert summary.hit1_test is None


def test_fusion_has_no_stages():
    items = {"ow-0": QueryItem(id="ow-0", question="q", expected="refuse")}
    records = [_record("ow-0", _result("original", score=0.5), _result("fusion", score=0.9))]

    summary = summarize(items, records, "fusion", rule=RULE, k=K)

    assert dict(summary.stages) == {}
    assert summary.to_grounded == ()


def test_hop_coverage_divides_by_the_number_of_gold_documents():
    items = {"c1": _gold("c1", ("가", "나")), "c2": _gold("c2", ("가", "나", "다", "라"))}
    records = [
        _record("c1", _result("original"), _result("multi", hops=2)),
        _record("c2", _result("original"), _result("multi", hops=1)),
    ]

    summary = summarize(items, records, "multi", rule=RULE, k=K)

    assert summary.hop_coverage == pytest.approx((1.0 + 0.25) / 2)


def test_latency_adds_llm_search_and_rerank():
    items = {"k0": _gold("k0")}
    records = [_record("k0", _result("original"), _result("rewrite", llm_ms=100.0))]

    summary = summarize(items, records, "rewrite", rule=RULE, k=K)

    assert summary.total.percentiles[50] == pytest.approx(115.0)
    assert summary.llm.percentiles[50] == pytest.approx(100.0)


def test_rewrite_flags_pick_items_to_read():
    rows = [
        Rewrite("k1", "1997년 경복궁", "경복궁 중건", ("1997년 경복궁 중건", "1867년 중건"), 1, 1),
        Rewrite("k2", "경복궁 중건", " 경복궁  중건", ("경복궁 다시 지음",), 1, 1),
        Rewrite("k3", "q", None, (), 1, 1, rewrite_error="format: x", multi_error="generation: y"),
    ]

    flags = rewrite_flags(rows)

    assert flags.total == 3
    assert flags.rewrite_dropped == ("k1",)
    assert flags.rewrite_added == ()
    assert flags.multi_dropped == ("k1",)
    assert flags.multi_added == ("k1",)
    assert flags.unchanged == ("k2",)
    assert flags.rewrite_failed == ("k3",)
    assert flags.multi_failed == ("k3",)
    assert dict(flags.query_counts) == {0: 1, 1: 1, 2: 1}


def test_rank_agreement_compares_the_baseline_rank():
    records = [
        _record("k0", _result("original", rank=1), _result("rewrite")),
        _record("k1", _result("original", rank=None), _result("rewrite")),
        _record("k2", _result("original", rank=2), _result("rewrite")),
        _record("g", _result("original"), _result("rewrite"), gate="chitchat"),
    ]

    agreement, differ = rank_agreement(records, {"k0": 1, "k1": None, "k2": 3, "g": 1})

    assert (agreement.count, agreement.total) == (2, 3)
    assert differ == ("k2",)
