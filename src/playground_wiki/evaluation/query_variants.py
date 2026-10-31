"""질의 변형 실험(검색만): 원문 · 고친 질의 · 여러 질의의 검색 후보를 변형마다 합치고 순위를 매겨 문항별 상위 k 를 남긴다.

변형 (docs 계획 Phase 3 — 같은 문항에서 나란히):
  original       ① 원문 한 번 — 지금 /search 와 같다 (Hybrid RRF 상위 rerank_candidates → 원문 리랭커 → 상위 k)
  rewrite        ② 원문 + 고친 질의의 Hybrid 후보 합집합 → 원문 리랭커
  rewrite_max    ② 의 비교 열 — 같은 후보를 원문 · 고친 질의로 매겨 높은 점수로 (이쪽이 크게 이기면 기준값 재보정을 정한다)
  multi          ③-a 원문 + 여러 질의의 Hybrid 후보 합집합 → 원문 리랭커
  fusion         ③-b RAG-Fusion 원형 — 질의 × 검색기(Dense · BM25) 순위를 RRF 로 합친 상위 k, 리랭커 없음
  fusion_rerank  ③-c ③-b 의 RRF 상위 rerank_candidates → 원문 리랭커

RRF 와 리랭커는 검색 라이브러리의 HybridRetriever · Reranker 를 그대로 쓴다 (미리 받은 순위를 검색기 꼴로 넣어) —
① 이 제품 검색과 같은 계산이 되게. 질의마다 Dense · BM25 결과를 한 번만 받아 변형끼리 나눠 쓰고, 변형의 검색 시간은
그 변형이 쓰는 질의의 검색 시간을 더해 낸다 (제품은 질의를 한 번에 하나씩 돌린다). 리랭커는 변형마다 따로 불러 잰다.
"""

import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Self, cast, get_args

import numpy as np
from playground_evalkit import KorQuADItem, contains_answer, normalize_title, squash
from playground_retrieval.core import Filters, Hit, Retriever
from playground_retrieval.search import HybridRetriever, PairScorer, Reranker, SearchSettings
from playground_retrieval.search.hybrid import CANDIDATES, RRF_K
from playground_retrieval.search.rerank import CANDIDATES as RERANK_CANDIDATES
from playground_retrieval.search.service import TOP_K

from playground_wiki.evaluation.f9_sample import longtail_items
from playground_wiki.evaluation.query_rewrite import Rewrite

Variant = Literal["original", "rewrite", "rewrite_max", "multi", "fusion", "fusion_rerank"]
VARIANTS: tuple[Variant, ...] = get_args(Variant)
VARIANT_LABELS: Mapping[Variant, str] = {
    "original": "① 원문 한 번",
    "rewrite": "② 고쳐 쓰기 + 원문 리랭커",
    "rewrite_max": "② 비교 — 원문 · 고친 질의 점수 중 높은 값",
    "multi": "③-a 여러 질의 + 원문 리랭커",
    "fusion": "③-b RAG-Fusion (리랭커 없음)",
    "fusion_rerank": "③-c RAG-Fusion + 원문 리랭커",
}
UNSCORED: frozenset[Variant] = frozenset({"fusion"})
"""점수가 RRF(순위로 만든 값)라 리랭커 기준값(high · low)으로 단계를 고를 수 없는 변형 — 적중률 비교에만 쓴다."""


@dataclass(frozen=True)
class VariantSettings:
    """변형을 만드는 검색 수치. 기본값은 검색 라이브러리 기본값이고, 실행은 운영 설정(retrieval.yaml)으로 만든다."""

    candidates: int = CANDIDATES
    """질의마다 Dense · BM25 가 각각 가져오는 후보 수."""
    rrf_k: int = RRF_K
    rerank_candidates: int = RERANK_CANDIDATES
    """질의 하나의 Hybrid 후보 수 · ③-c 가 리랭커에 넘기는 RRF 후보 수."""
    top_k: int = TOP_K
    """변형마다 남기는 상위 건수 — 답변에 넘기는 수와 같다."""

    @classmethod
    def from_search(cls, settings: SearchSettings) -> Self:
        """검색 서비스 구성(ApiConfig.search)과 같은 수치로."""
        return cls(
            candidates=settings.candidates,
            rrf_k=settings.rrf_k,
            rerank_candidates=settings.rerank_candidates,
            top_k=settings.top_k,
        )

    def to_json(self) -> dict[str, int]:
        """실행 meta 에 남길 값."""
        return {
            "candidates": self.candidates,
            "rrf_k": self.rrf_k,
            "rerank_candidates": self.rerank_candidates,
            "top_k": self.top_k,
        }


@dataclass(frozen=True)
class QueryItem:
    """질의 변형 실험 문항 하나."""

    id: str
    question: str
    """검색에 넣는 질문 — 리랭커도 이 질문으로 매긴다 (패러프레이즈 세트면 패러프레이즈가 '원문')."""
    gold_titles: tuple[str, ...] = ()
    """정답 문서 제목(정규화). 복합 질문은 hop 문서 전부. 위키 밖 질문이면 비어 있다."""
    answers: tuple[str, ...] = ()
    """정답 표기 (첫째가 대표). 정답 근거 청크는 정답 문서이면서 이 중 하나를 담은 청크."""
    expected: str | None = None
    """위키 밖 질문의 기대 단계 (partial · refuse). 정답이 있는 문항이면 None."""

    def __post_init__(self) -> None:
        if bool(self.gold_titles) != bool(self.answers):
            raise ValueError(f"gold_titles and answers must be both set or both empty, {self.id=}")

    @property
    def has_gold(self) -> bool:
        """정답 문서가 있는 문항인가 (위키 밖 질문이면 False)."""
        return bool(self.gold_titles)

    def gold_rank(self, hits: Sequence[Hit]) -> int | None:
        """검색 결과에서 정답 근거 청크의 첫 순위 (1부터). 없거나 정답이 없는 문항이면 None — 판정은 evalkit `contains_answer`."""
        items = [
            KorQuADItem(
                id=self.id, title=title, question=self.question, answer=answer, source_title=title
            )
            for title in self.gold_titles
            for answer in self.answers
        ]
        return next(
            (
                rank
                for rank, hit in enumerate(hits, start=1)
                if any(contains_answer(hit.chunk.title, hit.chunk.text, item) for item in items)
            ),
            None,
        )

    def hops_found(self, hits: Sequence[Hit]) -> int:
        """검색 결과에 청크가 하나라도 든 정답 문서 수 — 복합 질문이 hop 문서를 몇 개 덮었나."""
        return len({normalize_title(hit.chunk.title) for hit in hits} & set(self.gold_titles))


@dataclass(frozen=True)
class QueryPlan:
    """문항 하나에서 변형마다 검색하는 질의."""

    question: str
    rewrite_queries: tuple[str, ...]
    """② 가 검색하는 질의 (원문 먼저). 고친 질의가 없거나 원문과 같으면 원문 하나."""
    multi_queries: tuple[str, ...]
    """③ 이 검색하는 질의 (원문 먼저)."""
    rewrite_ms: float = 0.0
    multi_ms: float = 0.0

    @property
    def distinct(self) -> tuple[str, ...]:
        """검색할 서로 다른 질의 (처음 나온 순서)."""
        return tuple(dict.fromkeys((*self.rewrite_queries, *self.multi_queries)))


@dataclass(frozen=True)
class QueryHits:
    """질의 하나의 검색기별 순위."""

    query: str
    dense: tuple[Hit, ...]
    bm25: tuple[Hit, ...]
    took_ms: float
    """Dense · BM25 검색 시간 합."""


@dataclass(frozen=True)
class VariantResult:
    """문항 하나 · 변형 하나의 상위 k 와 비용."""

    variant: Variant
    chunk_ids: tuple[str, ...]
    """상위 청크 id (순위 순)."""
    titles: tuple[str, ...]
    """그 청크의 문서 제목."""
    scores: tuple[float, ...]
    """리랭커 점수 (0~1, rewrite_max 는 원문 · 고친 질의 중 높은 값). fusion 은 RRF 점수."""
    gold_rank: int | None
    """정답 근거 청크의 첫 순위 (1부터). 상위 k 에 없거나 정답이 없는 문항이면 None."""
    hops_found: int
    """상위 k 가 덮은 정답 문서 수."""
    queries: int
    """검색한 질의 수 (원문 포함)."""
    pairs: int
    """리랭커에 넣은 (질의, 청크) 쌍 수. fusion 은 0."""
    search_ms: float
    """검색한 질의들의 Dense · BM25 시간 합."""
    rerank_ms: float
    llm_ms: float
    """질의를 만든 LLM 호출 시간 (② 는 고쳐 쓰기, ③ 은 여러 질의, ① 은 0)."""

    @property
    def top_score(self) -> float | None:
        """1위 점수. 결과가 없으면 None."""
        return self.scores[0] if self.scores else None

    @property
    def total_ms(self) -> float:
        """질의 만들기 + 검색 + 리랭커 시간 (답변 생성 전까지)."""
        return self.llm_ms + self.search_ms + self.rerank_ms

    def to_json(self) -> dict[str, Any]:
        """기록 한 줄의 results 한 칸."""
        return {
            "variant": self.variant,
            "chunk_ids": list(self.chunk_ids),
            "titles": list(self.titles),
            "scores": list(self.scores),
            "gold_rank": self.gold_rank,
            "hops_found": self.hops_found,
            "queries": self.queries,
            "pairs": self.pairs,
            "search_ms": self.search_ms,
            "rerank_ms": self.rerank_ms,
            "llm_ms": self.llm_ms,
        }

    @classmethod
    def from_json(cls, row: Mapping[str, Any]) -> Self:
        """기록의 results 한 칸에서."""
        if row["variant"] not in VARIANTS:
            raise ValueError(f"unknown variant {row['variant']!r}")
        return cls(
            variant=cast("Variant", row["variant"]),
            chunk_ids=tuple(row["chunk_ids"]),
            titles=tuple(row["titles"]),
            scores=tuple(float(score) for score in row["scores"]),
            gold_rank=row["gold_rank"],
            hops_found=int(row["hops_found"]),
            queries=int(row["queries"]),
            pairs=int(row["pairs"]),
            search_ms=float(row["search_ms"]),
            rerank_ms=float(row["rerank_ms"]),
            llm_ms=float(row["llm_ms"]),
        )


@dataclass(frozen=True)
class VariantRecord:
    """문항 하나의 변형별 결과 (검색 기록 한 줄)."""

    id: str
    gate: str | None
    """관문이 멈춘 규칙 종류. 멈췄으면 운영 경로처럼 검색하지 않아 results 가 비어 있다."""
    results: tuple[VariantResult, ...] = ()

    def result(self, variant: Variant) -> VariantResult:
        """그 변형의 결과.

        Raises:
          KeyError: 관문에서 멈춰 결과가 없을 때.
        """
        for result in self.results:
            if result.variant == variant:
                return result
        raise KeyError(f"{self.id}: no result for {variant!r}")

    def to_json(self) -> dict[str, Any]:
        """JSONL 한 줄."""
        return {
            "id": self.id,
            "gate": self.gate,
            "results": [result.to_json() for result in self.results],
        }

    @classmethod
    def from_json(cls, row: Mapping[str, Any]) -> Self:
        """JSONL 한 줄에서."""
        return cls(
            id=str(row["id"]),
            gate=row["gate"],
            results=tuple(VariantResult.from_json(result) for result in row["results"]),
        )


class VariantSearcher:
    """문항 하나를 질의마다 Dense · BM25 로 한 번씩 찾고, 그 순위로 변형 여섯의 상위 k 를 만든다.

    리랭커는 늘 원문 질문으로 매긴다 (rewrite_max 만 고친 질의 점수도 매겨 높은 값을 쓴다).

    예)
        searcher = VariantSearcher(dense, bm25, scorer, settings=VariantSettings.from_search(config.search))
        results = searcher.search(item, rewrite)
    """

    def __init__(
        self,
        dense: Retriever,
        bm25: Retriever,
        scorer: PairScorer,
        *,
        settings: VariantSettings | None = None,
    ) -> None:
        self._dense = dense
        self._bm25 = bm25
        self._scorer = scorer
        self.settings = settings or VariantSettings()

    def search(self, item: QueryItem, rewrite: Rewrite | None) -> tuple[VariantResult, ...]:
        """변형마다 결과 (VARIANTS 순서). rewrite 가 None 이면 ② · ③ 도 원문 하나로 검색한다."""
        plan = query_plan(item.question, rewrite)
        retrieved = {query: self._retrieve(query) for query in plan.distinct}
        question = item.question
        rewrite_pool = _union(self._hybrid(retrieved[query]) for query in plan.rewrite_queries)
        multi_pool = _union(self._hybrid(retrieved[query]) for query in plan.multi_queries)
        fused_top = self._fuse(question, plan.multi_queries, retrieved, self.settings.top_k)
        fused_pool = self._fuse(
            question, plan.multi_queries, retrieved, self.settings.rerank_candidates
        )
        original_pool = self._hybrid(retrieved[question])

        def build(
            variant: Variant,
            hits: Sequence[Hit],
            queries: Sequence[str],
            *,
            pairs: int,
            rerank_ms: float,
            llm_ms: float,
        ) -> VariantResult:
            return _result(
                item,
                variant,
                hits,
                queries=len(queries),
                pairs=pairs,
                search_ms=sum(retrieved[query].took_ms for query in queries),
                rerank_ms=rerank_ms,
                llm_ms=llm_ms,
            )

        original, original_ms = self._rerank(question, original_pool)
        rewritten, rewrite_ms = self._rerank(question, rewrite_pool)
        best, best_ms = self._rerank_max(plan.rewrite_queries, rewrite_pool)
        multi, multi_ms = self._rerank(question, multi_pool)
        fused_reranked, fused_ms = self._rerank(question, fused_pool)
        rewrite_pairs = len(rewrite_pool)
        return (
            build(
                "original",
                original,
                (question,),
                pairs=len(original_pool),
                rerank_ms=original_ms,
                llm_ms=0.0,
            ),
            build(
                "rewrite",
                rewritten,
                plan.rewrite_queries,
                pairs=rewrite_pairs,
                rerank_ms=rewrite_ms,
                llm_ms=plan.rewrite_ms,
            ),
            build(
                "rewrite_max",
                best,
                plan.rewrite_queries,
                pairs=rewrite_pairs * len(plan.rewrite_queries),
                rerank_ms=best_ms,
                llm_ms=plan.rewrite_ms,
            ),
            build(
                "multi",
                multi,
                plan.multi_queries,
                pairs=len(multi_pool),
                rerank_ms=multi_ms,
                llm_ms=plan.multi_ms,
            ),
            build(
                "fusion",
                fused_top,
                plan.multi_queries,
                pairs=0,
                rerank_ms=0.0,
                llm_ms=plan.multi_ms,
            ),
            build(
                "fusion_rerank",
                fused_reranked,
                plan.multi_queries,
                pairs=len(fused_pool),
                rerank_ms=fused_ms,
                llm_ms=plan.multi_ms,
            ),
        )

    def _retrieve(self, query: str) -> QueryHits:
        start = time.perf_counter()
        dense = self._dense.search(query, self.settings.candidates)
        bm25 = self._bm25.search(query, self.settings.candidates)
        took_ms = round((time.perf_counter() - start) * 1000, 1)
        return QueryHits(query=query, dense=tuple(dense), bm25=tuple(bm25), took_ms=took_ms)

    def _hybrid(self, hits: QueryHits) -> list[Hit]:
        # 제품 SearchService 와 같은 검색기 순서 (Dense 먼저) — RRF 동점이면 먼저 나온 검색기 순서가 남는다
        fused = HybridRetriever(
            [_RankedList(hits.dense), _RankedList(hits.bm25)],
            rrf_k=self.settings.rrf_k,
            candidates=self.settings.candidates,
        )
        return fused.search(hits.query, self.settings.rerank_candidates)

    def _fuse(
        self,
        question: str,
        queries: Sequence[str],
        retrieved: Mapping[str, QueryHits],
        k: int,
    ) -> list[Hit]:
        """질의 × 검색기 순위 전부를 RRF 로 (질의 순서, 그 안에서 Dense · BM25 순서)."""
        rankings: list[Retriever] = []
        for query in queries:
            rankings.append(_RankedList(retrieved[query].dense))
            rankings.append(_RankedList(retrieved[query].bm25))
        fused = HybridRetriever(
            rankings, rrf_k=self.settings.rrf_k, candidates=self.settings.candidates
        )
        return fused.search(question, k)

    def _rerank(self, question: str, candidates: Sequence[Hit]) -> tuple[list[Hit], float]:
        if not candidates:
            return [], 0.0
        reranker = Reranker(_RankedList(candidates), self._scorer, candidates=len(candidates))
        start = time.perf_counter()
        hits = reranker.search(question, self.settings.top_k)
        return hits, round((time.perf_counter() - start) * 1000, 1)

    def _rerank_max(
        self, queries: Sequence[str], candidates: Sequence[Hit]
    ) -> tuple[list[Hit], float]:
        if not candidates:
            return [], 0.0
        start = time.perf_counter()
        per_query = [
            self._scorer.score([(query, hit.chunk.text) for hit in candidates]) for query in queries
        ]
        scores = np.max(np.stack(per_query), axis=0)
        # Reranker 와 같은 정렬 — 동점이면 후보 순서
        order = np.argsort(-scores, kind="stable")[: self.settings.top_k]
        hits = [Hit(chunk=candidates[i].chunk, score=float(scores[i])) for i in order]
        return hits, round((time.perf_counter() - start) * 1000, 1)


def query_plan(question: str, rewrite: Rewrite | None) -> QueryPlan:
    """변형마다 검색할 질의. 고친 질의가 없거나 원문과 같으면(공백 무시) ② 는 원문 하나, 여러 질의가 없으면 ③ 도 원문 하나.

    Raises:
      ValueError: rewrite 가 다른 질문의 기록일 때 (세트 · 기록 파일을 잘못 짝지었을 때).
    """
    if rewrite is None:
        return QueryPlan(question=question, rewrite_queries=(question,), multi_queries=(question,))
    if rewrite.question != question:
        raise ValueError(f"{rewrite.id}: rewrite was made for a different question")
    rewritten = rewrite.rewrite
    same = rewritten is None or squash(rewritten) == squash(question)
    return QueryPlan(
        question=question,
        rewrite_queries=(question,) if same else (question, cast("str", rewritten)),
        multi_queries=(question, *rewrite.queries),
        rewrite_ms=rewrite.rewrite_ms,
        multi_ms=rewrite.multi_ms,
    )


def paraphrase_items(rows: Iterable[Mapping[str, Any]], *, paraphrased: bool) -> list[QueryItem]:
    """패러프레이즈 세트 줄의 문항 — paraphrased 면 패러프레이즈를, 아니면 KorQuAD 원문을 질문으로.

    정답 판정은 F9 민감도 측정(scripts/eval_answer.py retrieval-sensitivity)과 같다 — 정규화한 제목 · 첫 번째 정답.
    """
    return [
        QueryItem(
            id=str(row["id"]),
            question=str(row["paraphrase"] if paraphrased else row["question"]),
            gold_titles=(normalize_title(str(row["title"])),),
            answers=(str(row["answers"][0]),),
        )
        for row in rows
    ]


def longtail_query_items(rows: Iterable[Mapping[str, Any]]) -> list[QueryItem]:
    """F4 결선 롱테일 세트 줄 중 답 있음 문항 (`f9_sample.longtail_items` 와 같은 문항 · 정답 표기)."""
    return [
        QueryItem(
            id=item.id,
            question=item.question,
            gold_titles=(cast("str", item.gold_title),),
            answers=item.gold_answers,
        )
        for item in longtail_items(rows)
    ]


def offwiki_items(rows: Iterable[Mapping[str, Any]]) -> list[QueryItem]:
    """위키 밖 질문 세트 줄의 문항 (정답 문서 없음)."""
    return [
        QueryItem(id=str(row["id"]), question=str(row["question"]), expected=str(row["expected"]))
        for row in rows
    ]


def compound_items(rows: Iterable[Mapping[str, Any]]) -> list[QueryItem]:
    """복합 질문 세트 줄(id · question · answer · aliases · hops[{title}])의 문항. 정답 문서는 hop 문서 전부."""
    return [
        QueryItem(
            id=str(row["id"]),
            question=str(row["question"]),
            gold_titles=tuple(
                dict.fromkeys(normalize_title(str(hop["title"])) for hop in row["hops"])
            ),
            answers=(str(row["answer"]), *(str(alias) for alias in row.get("aliases", ()))),
        )
        for row in rows
    ]


class _RankedList:
    """미리 받은 순위를 그대로 내놓는 검색기 — 라이브러리 RRF · 리랭커에 받아 둔 결과를 다시 넣을 때."""

    def __init__(self, hits: Sequence[Hit]) -> None:
        self._hits = tuple(hits)

    def search(self, query: str, k: int, filters: Filters | None = None) -> list[Hit]:
        del query, filters  # 이미 받은 순위라 질의 · 범위를 다시 보지 않는다
        return list(self._hits[:k])


def _union(rankings: Iterable[Sequence[Hit]]) -> list[Hit]:
    """여러 순위의 청크를 처음 나온 순서로 한 번씩 (앞 순위 · 앞 질의가 먼저)."""
    seen: dict[str, Hit] = {}
    for hits in rankings:
        for hit in hits:
            seen.setdefault(hit.chunk.id, hit)
    return list(seen.values())


def _result(
    item: QueryItem,
    variant: Variant,
    hits: Sequence[Hit],
    *,
    queries: int,
    pairs: int,
    search_ms: float,
    rerank_ms: float,
    llm_ms: float,
) -> VariantResult:
    return VariantResult(
        variant=variant,
        chunk_ids=tuple(hit.chunk.id for hit in hits),
        titles=tuple(hit.chunk.title for hit in hits),
        scores=tuple(float(hit.score) for hit in hits),
        gold_rank=item.gold_rank(hits),
        hops_found=item.hops_found(hits),
        queries=queries,
        pairs=pairs,
        search_ms=round(search_ms, 1),
        rerank_ms=rerank_ms,
        llm_ms=llm_ms,
    )
