"""검색기를 KorQuAD dev 로 평가한다: hit@1/5/10 · MRR@10 · 문서 hit@5 · 지연.

서브셋을 청킹해 검색기를 만들고, 서브셋에 정답 문서가 있는 질문으로 잰다.
정답 판정은 playground_evalkit (제목 일치 + 공백 무시 정답 포함) — CORPUS 임베딩 벤치와 같은 규칙이다.
서브셋 수치는 검색기 간 상대 비교 전용이다.

출력:
  data/runs/{run_id}/records-{retriever}.jsonl   문항별 점수 (git 제외)
  reports/{run_id}.md                            집계 표 (커밋)

사용법:
  uv run python scripts/eval_retrieval.py --retriever bm25
  uv run python scripts/eval_retrieval.py --retriever bm25 --max-tokens 256 384 512
  uv run python scripts/eval_retrieval.py --retriever hybrid-rerank --from-store --collection kowiki_kure-v1 --model kure-v1
"""

import argparse
import contextlib
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

from playground_corpus import (
    EMBEDDING_MODELS,
    Chunker,
    ChunkingConfig,
    HFTokenizer,
    SentenceTransformerEmbedder,
    read_documents,
)
from playground_evalkit import (
    KorQuADItem,
    ResultRecord,
    contains_answer,
    hit_at_k,
    load_korquad,
    mrr,
    render_table,
    write_records,
)
from playground_retrieval.core import Retriever
from playground_retrieval.search import (
    Bm25Index,
    Bm25Retriever,
    CrossEncoderScorer,
    DenseRetriever,
    HybridRetriever,
    KiwiAnalyzer,
    Reranker,
)
from playground_retrieval.store import PgConfig, PgStore, QdrantStore

from playground_wiki.api.config import BM25_INDEX_DIR
from playground_wiki.evaluation.paths import REPORT_DIR, SUBSET
from playground_wiki.sources.wiki import SOURCE_NAME

TOKENIZER = "BAAI/bge-m3"  # 후보 임베딩 3종이 같은 XLM-R 토크나이저라 청크 경계가 같다
TOP_K = 10
RERANK_SUFFIX = "-rerank"
# hybrid = dense + bm25 RRF, *-rerank = 그 검색기의 후보 20개를 bge-reranker-v2-m3 로 다시 매김
RETRIEVERS = ("bm25", "dense", "hybrid", "bm25-rerank", "dense-rerank", "hybrid-rerank")


def main() -> None:
    args = _parse_args()
    needs_dense = any(
        name.removesuffix(RERANK_SUFFIX) in {"dense", "hybrid"} for name in args.retriever
    )
    if needs_dense and (not args.collection or len(args.max_tokens) != 1):
        raise SystemExit(
            "dense · hybrid 는 --model · --collection 과 그 컬렉션의 --max-tokens 하나가 필요하다"
        )
    if args.from_store and (not args.collection or len(args.max_tokens) != 1):
        raise SystemExit("--from-store 는 --collection 과 그 컬렉션의 --max-tokens 하나가 필요하다")
    run_id = args.run_id or f"{datetime.now(tz=UTC):%Y-%m-%d}-retrieval-{'-'.join(args.retriever)}"
    summary = json.loads((args.subset / "subset.json").read_text(encoding="utf-8"))
    with contextlib.ExitStack() as stack:
        pg = (
            stack.enter_context(PgStore.connect(PgConfig.from_env()))
            if needs_dense or args.from_store
            else None
        )
        if pg is not None and args.from_store:
            documents = []
            titles = pg.collection_titles(args.collection, SOURCE_NAME)
        else:
            documents = list(read_documents(args.subset / "docs-00000.parquet"))
            if args.limit_docs:  # scripts/ingest.py --limit-docs 와 같은 규칙 (앞에서부터)
                documents = documents[: args.limit_docs]
            titles = {doc.title for doc in documents}
        items = [
            item
            for item in load_korquad(
                Path(summary["korquad_dev"]), title_aliases=summary["korquad_title_map"]
            )
            if item.title in titles
        ]
        tokenizer = HFTokenizer.from_pretrained(TOKENIZER)
        print(f"documents={len(titles)} questions={len(items)} run_id={run_id}")

        dense = _dense(args, pg) if pg is not None and needs_dense else None
        scorer = (
            CrossEncoderScorer.load()
            if any(n.endswith(RERANK_SUFFIX) for n in args.retriever)
            else None
        )
        rows = []
        for max_tokens in args.max_tokens:
            start = time.perf_counter()
            if pg is not None and args.from_store:
                # 저장한 색인을 읽으므로 build_sec 는 읽는 시간 (만든 시간은 색인 meta.json)
                bm25 = Bm25Retriever(
                    Bm25Index.load(BM25_INDEX_DIR / args.collection),
                    KiwiAnalyzer(),
                    pg,
                    collection=args.collection,
                    source=SOURCE_NAME,
                )
            else:
                chunker = Chunker(tokenizer, ChunkingConfig(max_tokens=max_tokens))
                chunks = [chunk for doc in documents for chunk in chunker.chunk(doc)]
                bm25 = Bm25Retriever.from_chunks(chunks, KiwiAnalyzer())
            bm25_build_sec = time.perf_counter() - start
            chunk_count = len(bm25.index)
            for name in args.retriever:
                retriever = _retriever(name, bm25, dense, scorer)
                # dense 색인은 ingest 에서 만든다
                build_sec = 0.0 if name.startswith("dense") else bm25_build_sec
                row, records = _evaluate(run_id, name, max_tokens, retriever, items)
                write_records(
                    Path("data/runs") / run_id / f"records-{name}-{max_tokens}.jsonl", records
                )
                rows.append({**row, "chunks": chunk_count, "build_sec": round(build_sec, 1)})
                print(
                    f"{name} {max_tokens}: hit@5={row['hit_at_5']:.3f} mrr={row['mrr_at_10']:.3f} "
                    f"doc_hit@5={row['doc_hit_at_5']:.3f} p50={row['p50_ms']}ms build={build_sec:.0f}s"
                )

        _write_report(run_id, rows, len(titles), len(items), args)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--retriever", nargs="+", default=["bm25"], choices=RETRIEVERS)
    parser.add_argument("--subset", type=Path, default=SUBSET)
    parser.add_argument("--max-tokens", nargs="+", type=int, default=[384])
    parser.add_argument(
        "--limit-docs", type=int, default=0, help="앞에서부터 이만큼 문서만 (ingest 와 맞춘다)"
    )
    parser.add_argument(
        "--model", default="bge-m3", choices=list(EMBEDDING_MODELS), help="dense 질의 임베딩 모델"
    )
    parser.add_argument(
        "--collection", default="", help="dense 가 검색할 Qdrant 컬렉션 (scripts/ingest.py 로 적재)"
    )
    parser.add_argument(
        "--from-store",
        action="store_true",
        help="서브셋 parquet 대신 적재한 컬렉션(PG 청크 · 저장한 BM25 색인)으로. 전체 코퍼스 평가용",
    )
    parser.add_argument("--run-id", default="")
    return parser.parse_args()


def _dense(args: argparse.Namespace, pg: PgStore) -> DenseRetriever:
    """적재된 컬렉션을 검색하는 Dense 검색기. PG 접속은 스크립트가 끝날 때까지 연다."""
    embedder = SentenceTransformerEmbedder.load(EMBEDDING_MODELS[args.model])
    return DenseRetriever(
        embedder, QdrantStore.from_env(), pg, collection=args.collection, source=SOURCE_NAME
    )


def _retriever(
    name: str, bm25: Bm25Retriever, dense: DenseRetriever | None, scorer: CrossEncoderScorer | None
) -> Retriever:
    base_name = name.removesuffix(RERANK_SUFFIX)
    if base_name == "bm25":
        base: Retriever = bm25
    elif dense is None:
        raise SystemExit(f"{name} 는 --collection 이 필요하다")
    else:
        base = dense if base_name == "dense" else HybridRetriever([dense, bm25])
    if name.endswith(RERANK_SUFFIX):
        if scorer is None:
            raise SystemExit(f"{name} 는 리랭커가 필요하다")
        return Reranker(base, scorer)
    return base


def _evaluate(
    run_id: str, name: str, max_tokens: int, retriever: Retriever, items: list[KorQuADItem]
) -> tuple[dict[str, float | str | int], list[ResultRecord]]:
    relevance: list[list[bool]] = []
    doc_relevance: list[list[bool]] = []
    latencies: list[float] = []
    top_scores: list[float] = []
    for item in items:
        start = time.perf_counter()
        hits = retriever.search(item.question, k=TOP_K)
        latencies.append((time.perf_counter() - start) * 1000)
        relevance.append([contains_answer(hit.chunk.title, hit.chunk.text, item) for hit in hits])
        doc_relevance.append([hit.chunk.title == item.title for hit in hits])
        top_scores.append(hits[0].score if hits else 0.0)
    created_at = datetime.now(tz=UTC).isoformat(timespec="seconds")
    records = [
        ResultRecord(
            run_id=run_id,
            spec_id=f"{name}/{max_tokens}",
            item_id=item.id,
            scores={
                "hit@1": float(any(rel[:1])),
                "hit@5": float(any(rel[:5])),
                "hit@10": float(any(rel[:10])),
                "doc_hit@5": float(any(doc[:5])),
                "top1_score": top,
            },
            spec={"retriever": name, "max_tokens": max_tokens, "tokenizer": TOKENIZER},
            created_at=created_at,
        )
        for item, rel, doc, top in zip(items, relevance, doc_relevance, top_scores, strict=True)
    ]
    # 1위 점수 분포: 1위가 정답일 때와 아닐 때 — 리랭커 점수로 답변 정책 기준값을 정할 수 있는지 보는 용도
    hit_scores = [top for top, rel in zip(top_scores, relevance, strict=True) if any(rel[:1])]
    miss_scores = [top for top, rel in zip(top_scores, relevance, strict=True) if not any(rel[:1])]
    latencies.sort()
    row = {
        "retriever": name,
        "max_tokens": max_tokens,
        "hit_at_1": hit_at_k(relevance, 1),
        "hit_at_5": hit_at_k(relevance, 5),
        "hit_at_10": hit_at_k(relevance, 10),
        "mrr_at_10": mrr(relevance),
        "doc_hit_at_5": hit_at_k(doc_relevance, 5),
        "p50_ms": round(latencies[len(latencies) // 2], 1),
        "p99_ms": round(latencies[int(len(latencies) * 0.99)], 1),
        "top1_hit_median": statistics.median(hit_scores) if hit_scores else 0.0,
        "top1_miss_median": statistics.median(miss_scores) if miss_scores else 0.0,
    }
    return row, records


def _write_report(
    run_id: str,
    rows: list[dict[str, float | str | int]],
    documents: int,
    questions: int,
    args: argparse.Namespace,
) -> None:
    columns = [
        "retriever",
        "max_tokens",
        "chunks",
        "hit_at_1",
        "hit_at_5",
        "hit_at_10",
        "mrr_at_10",
        "doc_hit_at_5",
        "p50_ms",
        "p99_ms",
        "build_sec",
        "top1_hit_median",
        "top1_miss_median",
    ]
    if args.from_store:
        scope = (
            f"- 컬렉션: `{args.collection}` 문서 {documents:,}개 · KorQuAD dev 질문 {questions:,}개 "
            "(정답 문서가 컬렉션에 있는 질문). BM25 는 저장한 색인, 청크는 PG\n"
        )
    else:
        scope = (
            f"- 서브셋: `{args.subset.name}` 문서 {documents:,}개 · KorQuAD dev 질문 {questions:,}개 (정답 문서가 서브셋에 있는 질문)\n"
            "- **서브셋 수치는 상대 비교 전용** — 방해 문서가 적어 전체 코퍼스보다 높게 나온다\n"
        )
    report = (
        f"# 검색 평가 — {run_id}\n\n"
        + scope
        + f"- 청킹: `{TOKENIZER}` 토크나이저 기준 (후보 임베딩 3종과 같은 청크 경계)\n"
        "- 정답 판정: evalkit `contains_answer` (제목 일치 + 공백 무시 정답 포함) — CORPUS 임베딩 벤치와 같은 규칙\n"
        "- 지연은 질의 하나의 검색 시간 (색인은 미리 만든 상태)\n"
        "- `top1_hit_median` · `top1_miss_median`: 1위가 정답일 때 · 아닐 때의 1위 점수 중앙값 "
        "(검색기마다 척도가 다르다 — 리랭커는 0~1)\n\n" + render_table(rows, columns)
    )
    REPORT_DIR.mkdir(exist_ok=True)
    # Windows 에서도 LF 로 (리포트는 커밋하고 .gitattributes 가 LF)
    (REPORT_DIR / f"{run_id}.md").write_text(report, encoding="utf-8", newline="\n")
    print(f"report: reports/{run_id}.md")


if __name__ == "__main__":
    main()
