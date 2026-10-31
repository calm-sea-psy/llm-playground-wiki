"""질의 변형 실험 집계 — 변형별 hit@1 · hit@k · MRR@k, ① 과 같은 문항 짝 McNemar, 1위 점수로 본 단계 경로 변화, 비용.

정답 판정은 검색 단계가 기록한 gold_rank (query_variants.QueryItem.gold_rank) 를 그대로 쓴다.
"""

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from playground_evalkit import McNemarResult, mcnemar_exact, mrr, squash

from playground_wiki.evaluation.answer_metrics import Latency, Share, latency, share
from playground_wiki.evaluation.query_rewrite import Rewrite, added_numbers, dropped_numbers
from playground_wiki.evaluation.query_variants import (
    UNSCORED,
    QueryItem,
    Variant,
    VariantRecord,
    VariantResult,
)

GROUNDED = "grounded"
"""답변 정책의 grounded 단계 이름 (generation.policy.Stage) — evaluation 은 generation 을 import 하지 않는다
(tests/test_layers.py). 같은지는 시험이 본다."""
BASELINE: Variant = "original"
"""모든 변형을 같은 문항 짝으로 견주는 기준 — ① 원문 한 번 (지금 운영 검색)."""


class StageRule(Protocol):
    """리랭커 1위 점수로 단계를 고르는 규칙 (generation.Thresholds)."""

    def stage(self, score: float | None) -> str:
        """1위 점수의 단계 (grounded · partial · refuse). 결과가 없으면(None) refuse."""
        ...


@dataclass(frozen=True)
class VariantSummary:
    """세트 하나 · 변형 하나의 집계."""

    variant: Variant
    searched: int
    """검색한 문항 수 (관문에서 멈춘 문항 제외)."""
    hit1: Share
    """정답 문항 중 1위가 정답 근거 청크인 것."""
    hitk: Share
    """정답 문항 중 상위 k 에 정답 근거 청크가 있는 것."""
    mrr: float | None
    """MRR@k (상위 k 밖이면 0). 정답 문항이 없으면 None."""
    hit1_test: McNemarResult | None
    """① 과 같은 문항 짝 hit@1 McNemar (a = ①, b = 이 변형). ① 자신이거나 정답 문항이 없으면 None."""
    hitk_test: McNemarResult | None
    stages: Mapping[str, int]
    """1위 점수로 고른 단계별 문항 수. 단계를 고를 수 없는 변형(UNSCORED)이면 비어 있다."""
    changed: int
    """① 과 단계가 다른 문항 수."""
    to_grounded: tuple[str, ...]
    """① 에서 grounded 가 아니었는데 이 변형에서 grounded 가 된 문항 id — 위키 밖 질문이면 위험으로 센다."""
    from_grounded: tuple[str, ...]
    """① 에서 grounded 였는데 이 변형에서 아니게 된 문항 id."""
    hop_coverage: float | None
    """정답 문서를 덮은 비율의 평균 (상위 k 가 덮은 정답 문서 수 / 정답 문서 수). 정답 문항이 없으면 None."""
    queries: float
    """문항당 검색 질의 수 평균 (원문 포함)."""
    pairs: float
    """문항당 리랭커 쌍 수 평균."""
    llm: Latency
    search: Latency
    rerank: Latency
    total: Latency
    """질의 만들기 + 검색 + 리랭커 (답변 생성 전까지)."""


@dataclass(frozen=True)
class RewriteFlags:
    """고친 질의 · 여러 질의를 눈으로 볼 문항 — 숫자 조건이 빠졌거나 새로 생긴 것, 실패, 그대로 둔 것."""

    total: int
    rewrite_failed: tuple[str, ...]
    multi_failed: tuple[str, ...]
    unchanged: tuple[str, ...]
    """고친 질의가 원문과 같은 문항 (② 가 ① 과 같은 검색이 된다)."""
    rewrite_dropped: tuple[str, ...]
    """원문의 숫자(날짜 · 수량 · 순번)가 고친 질의에서 빠진 문항."""
    rewrite_added: tuple[str, ...]
    """고친 질의에 원문에 없는 숫자가 생긴 문항."""
    multi_dropped: tuple[str, ...]
    """여러 질의 중 하나라도 원문의 숫자가 빠진 문항."""
    multi_added: tuple[str, ...]
    query_counts: Mapping[int, int]
    """여러 질의 수(원문 제외) → 문항 수."""


def summarize(
    items: Mapping[str, QueryItem],
    records: Sequence[VariantRecord],
    variant: Variant,
    *,
    rule: StageRule,
    k: int,
) -> VariantSummary:
    """세트 기록에서 변형 하나의 집계. 관문에서 멈춘 문항은 뺀다.

    Args:
      items: 문항 id → 문항 (정답 문서 수 · 정답 유무).
      records: 세트의 검색 기록.
      variant: 집계할 변형.
      rule: 1위 점수 → 단계.
      k: 상위 몇 개까지 보는지 (기록한 top_k 이하).

    Raises:
      KeyError: 기록의 문항이 items 에 없을 때.
    """
    searched = [record for record in records if record.gate is None]
    pairs = [(record.result(BASELINE), record.result(variant)) for record in searched]
    gold = [
        (base, other)
        for (base, other), record in zip(pairs, searched, strict=True)
        if items[record.id].has_gold
    ]
    hit1 = share(_hit(other, 1) for _, other in gold)
    hitk = share(_hit(other, k) for _, other in gold)
    tests = (None, None)
    if variant != BASELINE and gold:
        tests = (
            mcnemar_exact(
                [_hit(base, 1) for base, _ in gold], [_hit(other, 1) for _, other in gold]
            ),
            mcnemar_exact(
                [_hit(base, k) for base, _ in gold], [_hit(other, k) for _, other in gold]
            ),
        )
    stages, changed, to_grounded, from_grounded = _stage_moves(searched, pairs, variant, rule)
    coverage = [
        other.hops_found / len(items[record.id].gold_titles)
        for (_, other), record in zip(pairs, searched, strict=True)
        if items[record.id].has_gold
    ]
    results = [other for _, other in pairs]
    return VariantSummary(
        variant=variant,
        searched=len(searched),
        hit1=hit1,
        hitk=hitk,
        mrr=mrr([_relevance(other, k) for _, other in gold]) if gold else None,
        hit1_test=tests[0],
        hitk_test=tests[1],
        stages=stages,
        changed=changed,
        to_grounded=to_grounded,
        from_grounded=from_grounded,
        hop_coverage=sum(coverage) / len(coverage) if coverage else None,
        queries=_mean(result.queries for result in results),
        pairs=_mean(result.pairs for result in results),
        llm=latency(result.llm_ms for result in results),
        search=latency(result.search_ms for result in results),
        rerank=latency(result.rerank_ms for result in results),
        total=latency(result.total_ms for result in results),
    )


def rewrite_flags(rewrites: Iterable[Rewrite]) -> RewriteFlags:
    """고친 질의 기록에서 눈으로 볼 문항 id (기록 순서)."""
    rows = list(rewrites)
    unchanged, rewrite_dropped, rewrite_added, multi_dropped, multi_added = [], [], [], [], []
    for row in rows:
        if row.rewrite is not None:
            if squash(row.rewrite) == squash(row.question):
                unchanged.append(row.id)
            if dropped_numbers(row.question, row.rewrite):
                rewrite_dropped.append(row.id)
            if added_numbers(row.question, row.rewrite):
                rewrite_added.append(row.id)
        if any(dropped_numbers(row.question, query) for query in row.queries):
            multi_dropped.append(row.id)
        if any(added_numbers(row.question, query) for query in row.queries):
            multi_added.append(row.id)
    return RewriteFlags(
        total=len(rows),
        rewrite_failed=tuple(row.id for row in rows if row.rewrite_error is not None),
        multi_failed=tuple(row.id for row in rows if row.multi_error is not None),
        unchanged=tuple(unchanged),
        rewrite_dropped=tuple(rewrite_dropped),
        rewrite_added=tuple(rewrite_added),
        multi_dropped=tuple(multi_dropped),
        multi_added=tuple(multi_added),
        query_counts=MappingProxyType(
            dict(sorted(Counter(len(row.queries) for row in rows).items()))
        ),
    )


def rank_agreement(
    records: Iterable[VariantRecord], expected: Mapping[str, int | None]
) -> tuple[Share, tuple[str, ...]]:
    """① 의 정답 순위가 다른 측정(F9 민감도 기록)의 순위와 같은 문항 비율과 다른 문항 id — 조립이 운영 검색과 같은지 본다.

    expected 에 없는 문항 · 관문에서 멈춘 문항은 뺀다.
    """
    same: list[bool] = []
    differ: list[str] = []
    for record in records:
        if record.gate is not None or record.id not in expected:
            continue
        agree = record.result(BASELINE).gold_rank == expected[record.id]
        same.append(agree)
        if not agree:
            differ.append(record.id)
    return share(same), tuple(differ)


def _stage_moves(
    searched: Sequence[VariantRecord],
    pairs: Sequence[tuple[VariantResult, VariantResult]],
    variant: Variant,
    rule: StageRule,
) -> tuple[Mapping[str, int], int, tuple[str, ...], tuple[str, ...]]:
    """(단계별 수, ① 과 다른 수, grounded 로 올라간 id, grounded 에서 내려간 id). UNSCORED 변형이면 비운다."""
    if variant in UNSCORED:
        return MappingProxyType({}), 0, (), ()
    counts: Counter[str] = Counter()
    changed = 0
    to_grounded: list[str] = []
    from_grounded: list[str] = []
    for record, (base, other) in zip(searched, pairs, strict=True):
        before, after = rule.stage(base.top_score), rule.stage(other.top_score)
        counts[after] += 1
        changed += before != after
        if after == GROUNDED and before != GROUNDED:
            to_grounded.append(record.id)
        if before == GROUNDED and after != GROUNDED:
            from_grounded.append(record.id)
    return MappingProxyType(dict(counts)), changed, tuple(to_grounded), tuple(from_grounded)


def _hit(result: VariantResult, k: int) -> bool:
    return result.gold_rank is not None and result.gold_rank <= k


def _relevance(result: VariantResult, k: int) -> list[bool]:
    return [result.gold_rank == position for position in range(1, k + 1)]


def _mean(values: Iterable[int]) -> float:
    counted = list(values)
    return sum(counted) / len(counted) if counted else 0.0
