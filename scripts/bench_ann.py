"""ANN 벤치마크: FAISS(Flat · HNSW · IVF · IVF+SQ8 · IVF+PQ) vs Qdrant(전수 · HNSW float · HNSW int8).

정답은 FAISS Flat 전수 검색의 상위 10건이다. 지표는 정답 대비 recall@10, KorQuAD hit@5 · MRR@10(evalkit 매칭),
질의 하나씩 보낸 지연 p50 · p99, 질의를 한꺼번에 보낸 처리량(QPS), 색인 크기, 색인 시간.

- 문서 벡터는 scripts/ingest.py 가 남긴 임베딩 캐시를, 청크 원문은 PG 를 읽는다 (같은 컬렉션)
- 질문은 정답 문서가 컬렉션에 있는 KorQuAD dev 질문 중 무작위 --queries 개. 질의 벡터는 캐시에 남긴다
- FAISS 지연은 스레드 1개, 처리량은 전체 스레드. Qdrant 지연은 gRPC 왕복 포함
- 서브셋 수치는 1차 확인용. 전체 규모는 Phase 1.5 (reports/ann-benchmark.md)

출력:
  reports/{run_id}.md    결과 표 (커밋)
  reports/{run_id}.svg   recall@10 – QPS 그래프 (커밋)

사용법:
  uv run python scripts/bench_ann.py --collection kowiki-subset-10000_kure-v1
  QDRANT_TIMEOUT=300 uv run python scripts/bench_ann.py --collection kowiki_kure-v1 --skip-qdrant-exact --nlist 4096 --run-id ann-benchmark
  (전체 규모 Qdrant 는 100개 묶음 요청이 기본 시간 제한 5초를 넘는다)
"""

import argparse
import json
import math
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import faiss
import numpy as np
from numpy.typing import NDArray
from playground_corpus import (
    EMBEDDING_MODELS,
    Chunk,
    EmbeddingCache,
    SentenceTransformerEmbedder,
)
from playground_evalkit import (
    KorQuADItem,
    contains_answer,
    hit_at_k,
    load_korquad,
    mrr,
    render_table,
)
from playground_retrieval.eval import latency_summary, recall_at_k, recall_qps_svg
from playground_retrieval.search import FaissIndex
from playground_retrieval.store import PgConfig, PgStore, QdrantStore
from qdrant_client import models

from playground_wiki.sources.wiki import SOURCE_NAME

SUBSET = Path("data/parsed/kowiki-subset-10000")
EMB_DIR = Path("data/emb")
TOP_K = 10
WARMUP = 20  # 측정 전에 버리는 질의 수 (캐시 · 스레드 풀 준비)
QDRANT_BATCH = 100  # 처리량 측정 때 요청 하나에 담는 질의 수
HNSW_EF = (16, 32, 64, 128, 256)
NPROBE = (1, 4, 16, 64)
QDRANT_EF = (16, 32, 64, 128, 256)
RUNS_DIR = Path("data/runs")  # FAISS 결과 저장 (git 제외)
# --qdrant-variants 이름 → (리포트 표기, 양자화 검색 인자)
QDRANT_VARIANTS = {
    "float": ("HNSW float", models.QuantizationSearchParams(ignore=True)),  # 원본 벡터(on_disk)로
    "int8": ("HNSW int8", models.QuantizationSearchParams(ignore=False, rescore=False)),
    # 재채점은 후보 안의 순서만 바꾸므로 후보를 2배로 뽑아야 recall 이 오른다
    "int8-rescore": (
        "HNSW int8+rescore x2",
        models.QuantizationSearchParams(ignore=False, rescore=True, oversampling=2.0),
    ),
}
COLUMNS = [
    "engine",
    "index",
    "param",
    "recall_at_10",
    "hit_at_5",
    "mrr_at_10",
    "p50_ms",
    "p99_ms",
    "qps",
    "build_sec",
    "index_mb",
]


@dataclass(frozen=True)
class RunResult:
    """색인 설정 하나로 질의 전부를 돌린 결과."""

    found: list[list[str]]
    """질의마다 청크 id 상위 10 (순위순)."""
    latencies: list[float]
    """질의 하나씩 보낸 검색 시간 (ms)."""
    qps: float
    """질의를 한꺼번에 보냈을 때 초당 질의 수."""


def main() -> None:
    args = _parse_args()
    run_id = args.run_id or f"{datetime.now(tz=UTC):%Y-%m-%d}-ann-subset"
    chunk_ids, vectors = EmbeddingCache(
        EMB_DIR / args.model / f"{args.collection}-{args.max_tokens}"
    ).load()
    pg = PgStore.connect(PgConfig.from_env())
    chunks = _ChunkCache(pg, args.collection)
    items = _sample_questions(args, pg.collection_titles(args.collection, SOURCE_NAME))
    queries = _query_vectors(args, items)
    qdrant = QdrantStore.from_env()
    status = qdrant.index_status(args.collection)
    if status.indexed < status.points:  # HNSW 를 만드는 중이면 전수 검색이 섞여 지연이 틀린다
        raise SystemExit(f"Qdrant 색인 중: {status} — 끝난 뒤 다시 실행")
    print(
        f"vectors={len(chunk_ids):,} queries={len(items):,} threads={faiss.omp_get_max_threads()}"
    )

    def judge(found: list[list[str]]) -> list[list[bool]]:
        by_id = chunks.get([c for ids in found for c in ids])
        return [
            [contains_answer(by_id[c].title, by_id[c].text, item) for c in ids]
            for item, ids in zip(items, found, strict=True)
        ]

    # FAISS 결과는 저장해 두고 다시 쓴다 — 전체 규모에서 25분 걸리므로 Qdrant 쪽이 실패해도 다시 돌리지 않게
    checkpoint = RUNS_DIR / run_id / "faiss.json"
    if checkpoint.exists() and not args.redo_faiss:
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        rows, truth = saved["rows"], saved["truth"]
        print(f"faiss: {checkpoint} 재사용 ({len(rows)}행)")
    else:
        rows, truth = _faiss_rows(args, chunk_ids, vectors, queries, judge)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text(json.dumps({"rows": rows, "truth": truth}), encoding="utf-8")
    del vectors  # Qdrant 단계에서는 쓰지 않는다 (전체 규모 약 10GB)

    specs = _qdrant_specs(
        exact=not args.skip_qdrant_exact, variants=args.qdrant_variants, float_ef=args.float_ef
    )
    for label, param, params in specs:
        run = _qdrant_run(qdrant, args.collection, queries, params, batch=args.qdrant_batch)
        row = _row("qdrant", label, param, run, truth=truth, judge=judge)
        rows.append({**row, "build_sec": "-", "index_mb": "-"})
        _print(row)

    _write_report(run_id, rows, len(chunk_ids), len(items), args)


def _faiss_rows(
    args: argparse.Namespace,
    chunk_ids: Sequence[str],
    vectors: NDArray[np.float32],
    queries: NDArray[np.float32],
    judge: Callable[[list[list[str]]], list[list[bool]]],
) -> tuple[list[dict[str, float | str | int]], list[list[str]]]:
    """FAISS 색인마다 만들고 검색 인자를 바꿔 가며 잰 행 목록과 정답(Flat 상위 10)."""
    doc_ids = [chunk_id.rsplit("-", 1)[0] for chunk_id in chunk_ids]  # 청크 id = {문서 id}-{순서}
    rows: list[dict[str, float | str | int]] = []
    truth: list[list[str]] = []
    for factory, param, values in _faiss_specs(args.nlist or _nlist(len(chunk_ids))):
        start = time.perf_counter()
        index = FaissIndex.build(
            vectors, chunk_ids, doc_ids, collection=args.collection, factory=factory
        )
        build_sec = time.perf_counter() - start
        index_mb = faiss.serialize_index(index.index).nbytes / 2**20
        for value in values:
            if param:
                faiss.ParameterSpace().set_index_parameter(index.index, param, value)
            run = _faiss_run(index, queries)
            if factory == "Flat":
                truth = run.found
            label = f"{param}={value}" if param else "-"
            row = _row("faiss", factory, label, run, truth=truth, judge=judge)
            rows.append({**row, "build_sec": round(build_sec, 1), "index_mb": round(index_mb, 1)})
            _print(row)
    return rows, truth


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--collection", required=True, help="scripts/ingest.py 로 적재한 컬렉션")
    parser.add_argument("--model", default="kure-v1", choices=list(EMBEDDING_MODELS))
    parser.add_argument(
        "--max-tokens", type=int, default=384, help="임베딩 캐시 폴더 이름의 청크 크기"
    )
    parser.add_argument(
        "--subset",
        type=Path,
        default=SUBSET,
        help="subset.json 이 있는 폴더 (KorQuAD 경로 · 제목 보정)",
    )
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--nlist", type=int, default=0, help="IVF 목록 수. 0 이면 4·√n 에 가까운 2의 거듭제곱"
    )
    parser.add_argument(
        "--skip-qdrant-exact",
        action="store_true",
        help="Qdrant 전수 검색을 건너뛴다 (전체 규모에서 질의마다 원본 벡터를 디스크에서 다 읽는다)",
    )
    parser.add_argument(
        "--qdrant-variants",
        nargs="+",
        default=list(QDRANT_VARIANTS),
        choices=list(QDRANT_VARIANTS),
        help="잴 Qdrant 설정. float 는 원본 벡터(on_disk)로 찾아 전체 규모에서 질의당 수백 ms",
    )
    parser.add_argument(
        "--float-ef", nargs="+", type=int, default=list(QDRANT_EF), help="float 설정의 hnsw_ef 값들"
    )
    parser.add_argument(
        "--qdrant-batch", type=int, default=QDRANT_BATCH, help="처리량 측정 때 요청 하나의 질의 수"
    )
    parser.add_argument(
        "--redo-faiss", action="store_true", help="저장한 FAISS 결과가 있어도 다시 잰다"
    )
    parser.add_argument(
        "--machine",
        default="RTX 5080 Laptop · GPU 전력 상한 95W · CPU 24스레드 (FAISS · Qdrant 측정은 CPU)",
        help="리포트에 적는 장비 · 전원 상태 (지연 · QPS 는 전원 모드에 따라 달라진다)",
    )
    parser.add_argument("--run-id", default="")
    return parser.parse_args()


def _sample_questions(args: argparse.Namespace, titles: set[str]) -> list[KorQuADItem]:
    summary = json.loads((args.subset / "subset.json").read_text(encoding="utf-8"))
    items = [
        item
        for item in load_korquad(
            Path(summary["korquad_dev"]), title_aliases=summary["korquad_title_map"]
        )
        if item.title in titles
    ]
    return random.Random(args.seed).sample(items, min(args.queries, len(items)))


def _query_vectors(args: argparse.Namespace, items: Sequence[KorQuADItem]) -> NDArray[np.float32]:
    """질의 벡터. 같은 질문 · 모델이면 캐시를 다시 쓴다."""
    cache = EmbeddingCache(EMB_DIR / args.model / f"korquad-dev-q{len(items)}-seed{args.seed}")
    ids = [item.id for item in items]
    if not (cache.directory / "manifest.json").exists():
        embedder = SentenceTransformerEmbedder.load(EMBEDDING_MODELS[args.model])
        cache.write(embedder, ids, [item.question for item in items], kind="query")
    cached_ids, vectors = cache.load()
    if cached_ids != ids:
        raise SystemExit(f"질의 캐시가 질문과 다르다: {cache.directory} 를 지우고 다시 실행")
    return vectors


def _faiss_run(index: FaissIndex, queries: NDArray[np.float32]) -> RunResult:
    normalized = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    threads = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(1)
    try:
        index.index.search(normalized[:WARMUP], TOP_K)
        latencies = _timed(lambda q: index.index.search(q.reshape(1, -1), TOP_K), normalized)
    finally:
        faiss.omp_set_num_threads(threads)
    start = time.perf_counter()
    _, rows = index.index.search(normalized, TOP_K)
    qps = len(normalized) / (time.perf_counter() - start)
    found = [[index.chunk_ids[row] for row in query_rows if row >= 0] for query_rows in rows]
    return RunResult(found=found, latencies=latencies, qps=qps)


def _nlist(n: int) -> int:
    """4·√n 에 가장 가까운 2의 거듭제곱 (FAISS 권장 범위 4~16·√n 의 아래쪽 — 학습 시간을 줄인다)."""
    target = 4 * math.sqrt(n)
    return 2 ** round(math.log2(target))


def _faiss_specs(nlist: int) -> list[tuple[str, str, tuple[int, ...]]]:
    """(index_factory 문자열, 바꿔 볼 검색 인자, 값들)."""
    return [
        ("Flat", "", (0,)),
        ("HNSW32", "efSearch", HNSW_EF),
        (f"IVF{nlist},Flat", "nprobe", NPROBE),
        (f"IVF{nlist},SQ8", "nprobe", NPROBE[1:]),
        (f"IVF{nlist},PQ128", "nprobe", NPROBE[1:]),  # 벡터당 128바이트 (float32 의 1/32)
    ]


class _ChunkCache:
    """판정에 필요한 청크만 PG 에서 꺼내 두는 캐시 (전체 규모에서 청크 전부를 올리지 않는다)."""

    def __init__(self, pg: PgStore, collection: str) -> None:
        self._pg = pg
        self._collection = collection
        self._by_id: dict[str, Chunk] = {}

    def get(self, chunk_ids: Sequence[str]) -> dict[str, Chunk]:
        missing = list(dict.fromkeys(c for c in chunk_ids if c not in self._by_id))
        for chunk in self._pg.get_chunks(self._collection, SOURCE_NAME, missing):
            self._by_id[chunk.id] = chunk
        if lost := [c for c in missing if c not in self._by_id]:
            raise SystemExit(
                f"PG 에 없는 청크 {len(lost)}개 (예: {lost[0]}) — 같은 컬렉션인지 확인"
            )
        return self._by_id


def _qdrant_specs(
    *, exact: bool, variants: Sequence[str], float_ef: Sequence[int]
) -> list[tuple[str, str, models.SearchParams]]:
    """(색인 이름, 인자 표기, 검색 인자). 컬렉션은 HNSW(m=16) + int8 스칼라 양자화, 원본 벡터 on_disk."""
    original = models.QuantizationSearchParams(ignore=True)
    specs = (
        [("exact float", "-", models.SearchParams(exact=True, quantization=original))]
        if exact
        else []
    )
    for variant in variants:
        label, quantization = QDRANT_VARIANTS[variant]
        efs = float_ef if variant == "float" else QDRANT_EF
        specs.extend(
            (label, f"hnsw_ef={ef}", models.SearchParams(hnsw_ef=ef, quantization=quantization))
            for ef in efs
        )
    return specs


def _qdrant_run(
    store: QdrantStore,
    collection: str,
    queries: NDArray[np.float32],
    params: models.SearchParams,
    *,
    batch: int,
) -> RunResult:
    for query in queries[:WARMUP]:
        store.search(collection, query, TOP_K, params=params)
    latencies = _timed(lambda q: store.search(collection, q, TOP_K, params=params), queries)
    start = time.perf_counter()
    found: list[list[str]] = []
    for offset in range(0, len(queries), batch):
        results = store.search_batch(
            collection, queries[offset : offset + batch], TOP_K, params=params
        )
        found.extend([hit.chunk_id for hit in hits] for hits in results)
    qps = len(queries) / (time.perf_counter() - start)
    return RunResult(found=found, latencies=latencies, qps=qps)


def _timed(
    search: Callable[[NDArray[np.float32]], object], queries: NDArray[np.float32]
) -> list[float]:
    latencies = []
    for query in queries:
        start = time.perf_counter()
        search(query)
        latencies.append((time.perf_counter() - start) * 1000)
    return latencies


def _row(
    engine: str,
    index: str,
    param: str,
    run: RunResult,
    *,
    truth: list[list[str]],
    judge: Callable[[list[list[str]]], list[list[bool]]],
) -> dict[str, float | str | int]:
    relevance = judge(run.found)
    latency = latency_summary(run.latencies)
    return {
        "engine": engine,
        "index": index,
        "param": param,
        "recall_at_10": recall_at_k(run.found, truth, TOP_K),
        "hit_at_5": hit_at_k(relevance, 5),
        "mrr_at_10": mrr(relevance),
        "p50_ms": latency.p50_ms,
        "p99_ms": latency.p99_ms,
        "qps": round(run.qps),
    }


def _print(row: dict[str, float | str | int]) -> None:
    print(
        f"{row['engine']} {row['index']} {row['param']}: recall@10={row['recall_at_10']:.3f} "
        f"hit@5={row['hit_at_5']:.3f} p50={row['p50_ms']}ms qps={row['qps']}"
    )


def _write_report(
    run_id: str,
    rows: list[dict[str, float | str | int]],
    vectors: int,
    questions: int,
    args: argparse.Namespace,
) -> None:
    report = (
        f"# ANN 벤치마크 — {run_id}\n\n"
        f"- 컬렉션: `{args.collection}` 벡터 {vectors:,}개 ({args.model}, 청크 {args.max_tokens}토큰) · "
        f"KorQuAD dev 질문 {questions:,}개 (정답 문서가 컬렉션에 있는 질문 중 무작위, seed {args.seed})\n"
        "- `recall_at_10`: FAISS Flat(전수 비교) 상위 10건 대비. `hit_at_5` · `mrr_at_10`: evalkit `contains_answer` 정답 판정\n"
        f"- FAISS: CPU, 지연은 스레드 1개 · 질의 하나씩, `qps` 는 질의 {questions:,}개 일괄 · 스레드 {faiss.omp_get_max_threads()}개. "
        "`index_mb` 는 직렬화한 색인 크기\n"
        f"- Qdrant: 로컬 Docker, HNSW m=16 · ef_construct=100 + int8 스칼라 양자화(RAM) · 원본 벡터 on_disk. "
        f"지연은 gRPC 왕복 포함 (REST 는 질의마다 약 16ms 더 든다), `qps` 는 요청 하나에 {args.qdrant_batch}개씩 묶어 보냄. "
        "색인 시간 · 크기는 적재 때라 비움\n"
        f"- 전원 · 장비: {args.machine}\n\n"
        f"![recall@10 – QPS]({run_id}.svg)\n\n"
        "오른쪽 위로 갈수록 좋다 (같은 recall 에서 더 빠름). 각 점은 아래 표의 한 줄이다.\n\n"
        + render_table(rows, COLUMNS)
    )
    Path("reports").mkdir(exist_ok=True)
    # Windows 에서도 LF 로 (리포트는 커밋하고 .gitattributes 가 LF)
    (Path("reports") / f"{run_id}.md").write_text(report, encoding="utf-8", newline="\n")
    chart = recall_qps_svg(rows, title=f"recall@10 – QPS ({vectors:,} 벡터)")
    (Path("reports") / f"{run_id}.svg").write_text(chart, encoding="utf-8", newline="\n")
    print(f"report: reports/{run_id}.md · {run_id}.svg")


if __name__ == "__main__":
    main()
