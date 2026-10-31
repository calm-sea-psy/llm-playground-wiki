"""질의 변형 실험 '검색만' 단계 — 원문 한 번 · 고쳐 쓰기 · 여러 질의(③-a/b/c, RAG-Fusion)를 같은 문항에서 나란히 잰다.

Phase 3 계획(2026-10-29 결정): 고친 질의는 후보를 모으는 데만 쓰고 리랭커는 원문 질문으로 매긴다 — high · low 를
다시 맞추지 않고 지금 수치와 바로 견준다. 답 생성 없이 적중률부터 재고, 이긴 변형만 답변까지 끝까지 잰다.

단계마다 결과를 data/ 에 저장해, 뒤 단계를 다시 돌릴 때 앞 단계를 반복하지 않는다.
  rewrite  --set S
           세트 문항마다 LLM(answer.yaml grounded 프로필의 모델 · think off · temperature 0)으로 고친 질의 1개(②)와
           여러 질의 2~3개(③)를 → data/eval/query-variants-v1/{S}-rewrites.jsonl (+ .meta.json 에 실행마다 설정).
           끝난 문항은 건너뛴다 (생성 실패 기록만 다시 돈다). Ollama 만 있으면 된다
  search   --set S
           관문(answer.yaml gate)에 걸린 문항은 운영 경로처럼 검색하지 않고 기록만, 나머지는 질의(원문 · 고친 질의 ·
           여러 질의)마다 Dense · BM25 를 한 번씩 돌려 변형 여섯의 상위 k 를 → {S}-search.jsonl (+ .meta.json).
           LLM 은 쓰지 않는다 (rewrite 기록을 읽는다). PG · Qdrant · BM25 색인(data/index/bm25/{컬렉션}) · 임베딩 ·
           리랭커 모델(GPU)이 필요하다. 끝난 문항은 건너뛴다
  report   검색 기록이 있는 세트 전부로 reports/{날짜}-query-variants.md — 집계와 문항 id 만 (질문 · 질의 원문 없음).
           --report-date 는 리포트 날짜 (다시 그릴 때 처음 날짜를 지킨다)

세트 (--set):
  korquad     F9 민감도 표본(패러프레이즈 v1 의 f9-random 300)의 KorQuAD 원문 질문
  paraphrase  같은 300 의 패러프레이즈 — 표현 차이를 메우나 (F9: hit@1 76.7 → 68.0, p 0.0066)
  longtail    F4 결선 롱테일 세트의 답 있음 300 (LLM 레포 bench/data, 읽기만 — worktree 에서는 --bench-data 로 절대 경로)
  offwiki     위키 밖 질문 100 — 정답 문서가 없어 hit 대신 '리랭커 1위 점수가 올라 grounded 로 바뀐 문항'을 위험으로 센다
  compound    복합 질문 50 (tests/evaluation/compound-v1.jsonl — 줄마다 id · question · answer · aliases · hops[{title}])

변형 · 기록 형식은 playground_wiki.evaluation.query_variants, 질의 만들기는 query_rewrite, 집계는 query_variant_metrics.
KorQuAD 는 CC BY-ND 라 질문 · 고친 질의 원문은 data/ 에만 둔다 (git 제외). 레포 루트에서 돌린다.

사용법:
  uv run python scripts/eval_query_variants.py rewrite --set paraphrase --limit 5
  uv run python scripts/eval_query_variants.py rewrite --set paraphrase
  uv run python scripts/eval_query_variants.py search --set paraphrase
  uv run python scripts/eval_query_variants.py rewrite --set longtail --bench-data C:/Work/llm-playground-llm/bench/data
  uv run python scripts/eval_query_variants.py report
"""

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, get_args

from kiwipiepy import Kiwi
from playground_corpus import EMBEDDING_MODELS, SentenceTransformerEmbedder
from playground_evalkit import McNemarResult, render_table
from playground_llm import ModelSpec, PromptLibrary
from playground_llm.ollama import OllamaConfig, OllamaLLM
from playground_retrieval.search import (
    Bm25Index,
    Bm25Retriever,
    CrossEncoderScorer,
    DenseRetriever,
    KiwiAnalyzer,
    SearchSettings,
)
from playground_retrieval.store import PgStore, QdrantStore

from playground_wiki.api.config import (
    ApiConfig,
    load_answer_policy,
    load_query_gate,
)
from playground_wiki.evaluation.answer_metrics import Latency, format_percent, format_share
from playground_wiki.evaluation.files import append_jsonl, read_jsonl
from playground_wiki.evaluation.paths import (
    BENCH_DATA,
    LONGTAIL_STAGE2_SET,
    OFFWIKI_PATH,
    PARAPHRASE_PATH,
    REPORT_DIR,
)
from playground_wiki.evaluation.query_rewrite import (
    MAX_CONSECUTIVE_ERRORS,
    QUERY_PROMPT_DIR,
    REWRITE_PARAMS,
    QueryRewriter,
    pending_ids,
    read_rewrites,
    read_rows_by_id,
)
from playground_wiki.evaluation.query_variant_metrics import (
    RewriteFlags,
    VariantSummary,
    rank_agreement,
    rewrite_flags,
    summarize,
)
from playground_wiki.evaluation.query_variants import (
    UNSCORED,
    VARIANT_LABELS,
    VARIANTS,
    QueryItem,
    VariantRecord,
    VariantSearcher,
    VariantSettings,
    compound_items,
    longtail_query_items,
    offwiki_items,
    paraphrase_items,
)
from playground_wiki.gate import KiwiTokenizer
from playground_wiki.generation import STAGES, Thresholds

QuerySet = Literal["korquad", "paraphrase", "longtail", "offwiki", "compound"]
QUERY_SETS: tuple[QuerySet, ...] = get_args(QuerySet)
SET_LABELS: Mapping[QuerySet, str] = MappingProxyType(
    {
        "korquad": "F9 민감도 표본 · KorQuAD 원문",
        "paraphrase": "F9 민감도 표본 · 패러프레이즈",
        "longtail": "롱테일 (F4 결선, 답 있음)",
        "offwiki": "위키 밖 질문 v1",
        "compound": "복합 질문 v1",
    }
)
OUT_DIR = Path("data/eval/query-variants-v1")
# 패러프레이즈 v1 중 검색 결과로 거르지 않고 뽑은 표본 (scripts/eval_answer.py 의 PARAPHRASE_SAMPLE 과 같은 값) —
# f4-answerable 은 "상위 5 에 정답"으로 고른 세트라 원문 hit@5 가 구성상 100%
PARAPHRASE_SAMPLE = "f9-random"
# F9 민감도 기록 (scripts/eval_answer.py retrieval-sensitivity) — ① 이 운영 검색과 같은 순위를 내는지 대조한다
SENSITIVITY_PATH = Path("data/eval/f9-retrieval-sensitivity-v1.jsonl")
# 복합 질문 세트 — 다른 작업이 쓰는 중이라 형식만 맞춰 읽는다
COMPOUND_PATH = Path("tests/evaluation/compound-v1.jsonl")
# 진행 표시 간격 (문항 수)
PROGRESS_EVERY = 25
SEARCH_COLUMNS = (
    "변형",
    "정답 문항",
    "hit@1",
    "hit@k",
    "MRR@k",
    "hit@1 ①만 / 변형만 (p)",
    "hit@k ①만 / 변형만 (p)",
)
COST_COLUMNS = (
    "변형",
    "문항",
    "검색 질의 수",
    "리랭커 쌍",
    "LLM p50",
    "검색 p50",
    "리랭커 p50",
    "합 p50",
    "합 p90",
)
# 리포트에 문항 id 를 늘어놓을 때 이만큼까지만 (나머지는 '외 n')
MAX_IDS_SHOWN = 20


@dataclass(frozen=True)
class SetMeasure:
    """리포트에 쓸 세트 하나의 기록과 집계."""

    name: QuerySet
    items: int
    """세트 문항 수."""
    records: tuple[VariantRecord, ...]
    summaries: tuple[VariantSummary, ...]
    """VARIANTS 순서."""
    flags: RewriteFlags
    k: int
    search_runs: tuple[Mapping[str, Any], ...]
    """검색 기록 meta 의 실행마다 설정."""
    rewrite_runs: tuple[Mapping[str, Any], ...]


def rewrites_path(set_name: QuerySet) -> Path:
    """세트의 고친 질의 기록."""
    return OUT_DIR / f"{set_name}-rewrites.jsonl"


def search_path(set_name: QuerySet) -> Path:
    """세트의 검색 기록."""
    return OUT_DIR / f"{set_name}-search.jsonl"


def load_items(set_name: QuerySet, bench_data: Path) -> list[QueryItem]:
    """세트의 문항 (파일 순서)."""
    if set_name in ("korquad", "paraphrase"):
        rows = [row for row in read_jsonl(PARAPHRASE_PATH) if row["sample"] == PARAPHRASE_SAMPLE]
        return paraphrase_items(rows, paraphrased=set_name == "paraphrase")
    if set_name == "longtail":
        return longtail_query_items(read_jsonl(bench_data / LONGTAIL_STAGE2_SET / "items.jsonl"))
    if set_name == "offwiki":
        return offwiki_items(read_jsonl(OFFWIKI_PATH))
    if not COMPOUND_PATH.exists():
        raise SystemExit(f"복합 질문 세트가 아직 없다: {COMPOUND_PATH}")
    return compound_items(read_jsonl(COMPOUND_PATH))


def rewrite(set_name: QuerySet, *, limit: int | None, bench_data: Path) -> None:
    items = {item.id: item for item in load_items(set_name, bench_data)}
    path = rewrites_path(set_name)
    todo = pending_ids(list(items), read_rewrites(path))[:limit]
    spec = _rewrite_spec()
    library = PromptLibrary.load(QUERY_PROMPT_DIR)
    with OllamaLLM.connect(OllamaConfig.from_env()) as llm:
        rewriter = QueryRewriter.from_library(llm, spec, library)
        _append_meta(
            path,
            {"spec_id": spec.spec_id, "prompts": list(rewriter.prompt_versions)},
            limit=limit,
        )
        print(f"{set_name} 문항 {len(items)} · 남은 것 {len(todo)} · {spec.spec_id}", flush=True)
        start = time.perf_counter()
        failures = 0
        for index, item_id in enumerate(todo, start=1):
            row = rewriter.rewrite(item_id, items[item_id].question)
            append_jsonl(path, row.to_json())
            failures = failures + 1 if row.retryable else 0
            if failures >= MAX_CONSECUTIVE_ERRORS:
                raise SystemExit(
                    f"생성 실패가 {failures}번 이어져 멈춘다 (Ollama 확인): {row.rewrite_error or row.multi_error}"
                )
            if index % PROGRESS_EVERY == 0:
                print(f"  {index}/{len(todo)} {time.perf_counter() - start:.0f}s", flush=True)
    flags = rewrite_flags(read_rewrites(path).values())
    print(
        f"기록 {flags.total} · 고친 질의 실패 {len(flags.rewrite_failed)} · 여러 질의 실패 "
        f"{len(flags.multi_failed)} · {time.perf_counter() - start:.0f}s\n저장: {path}"
    )


def search(set_name: QuerySet, *, limit: int | None, bench_data: Path) -> None:
    items = load_items(set_name, bench_data)
    path = search_path(set_name)
    done = read_rows_by_id(path)
    todo = [item for item in items if item.id not in done][:limit]
    rewrites = read_rewrites(rewrites_path(set_name))
    if missing := [
        item.id for item in todo if item.id not in rewrites or rewrites[item.id].retryable
    ]:
        raise SystemExit(
            f"고친 질의 기록이 없거나 생성 실패인 문항 {len(missing)} (먼저 rewrite --set {set_name}): "
            f"{', '.join(missing[:5])}"
        )
    config = ApiConfig.from_env()
    config_dir = config.config_dir
    # /ask 와 같이 Kiwi 하나(약 500MB)를 질문 관문과 BM25 가 나눠 쓴다
    kiwi = Kiwi()
    gate = load_query_gate(config_dir, KiwiTokenizer(kiwi))
    settings = VariantSettings.from_search(config.search)
    _append_meta(
        path,
        {
            "collection": config.search.collection,
            "model": config.search.model,
            "reranker_model": config.search.reranker_model,
            "gate": gate is not None,
            **settings.to_json(),
        },
        limit=limit,
    )
    print(
        f"{set_name} 문항 {len(items)} · 남은 것 {len(todo)} · collection={config.search.collection} · "
        f"관문 {'켬' if gate else '끔'}",
        flush=True,
    )
    start = time.perf_counter()
    with PgStore.connect(config.pg) as pg:
        searcher = _build_searcher(config.search, pg, kiwi, settings)
        print(f"검색기 준비 {time.perf_counter() - start:.0f}s", flush=True)
        for index, item in enumerate(todo, start=1):
            match = gate.check(item.question) if gate is not None else None
            if match is not None:
                record = VariantRecord(id=item.id, gate=match.kind)
            else:
                record = VariantRecord(
                    id=item.id, gate=None, results=searcher.search(item, rewrites[item.id])
                )
            append_jsonl(path, record.to_json())
            if index % PROGRESS_EVERY == 0:
                print(f"  {index}/{len(todo)} {time.perf_counter() - start:.0f}s", flush=True)
    print(f"{time.perf_counter() - start:.0f}s\n저장: {path}")


def _rewrite_spec() -> ModelSpec:
    """질의를 만드는 모델 설정 — 모델은 운영 답변(grounded 프로필)과 같게, 생성 설정은 REWRITE_PARAMS."""
    grounded = load_answer_policy(ApiConfig.from_env().config_dir).profiles["grounded"].spec
    return ModelSpec(grounded.provider, grounded.name, REWRITE_PARAMS)


def _build_searcher(
    search_settings: SearchSettings, pg: PgStore, kiwi: Kiwi, settings: VariantSettings
) -> VariantSearcher:
    """SearchService.build 와 같은 검색기 조립 — 서비스는 Dense · BM25 를 따로 내놓지 않아 공개 클래스로 다시 조립한다.

    BM25 색인은 저장한 것만 읽는다 (scripts/build_bm25.py). 서비스처럼 PG 청크로 새로 만들지 않는다 — 전체 규모라 오래 걸린다.
    """
    index_dir = search_settings.bm25_index
    if index_dir is None or not (index_dir / "meta.json").exists():
        raise SystemExit(f"BM25 색인이 없다: {index_dir} (scripts/build_bm25.py)")
    bm25 = Bm25Retriever(
        Bm25Index.load(index_dir),
        KiwiAnalyzer(kiwi),
        pg,
        collection=search_settings.collection,
        source=search_settings.source,
        doc_categories=pg.doc_categories(search_settings.collection, search_settings.source),
    )
    dense = DenseRetriever(
        SentenceTransformerEmbedder.load(EMBEDDING_MODELS[search_settings.model]),
        QdrantStore.from_env(),
        pg,
        collection=search_settings.collection,
        source=search_settings.source,
    )
    scorer = CrossEncoderScorer.load(search_settings.reranker_model)
    return VariantSearcher(dense, bm25, scorer, settings=settings)


def _append_meta(path: Path, settings: Mapping[str, Any], *, limit: int | None) -> None:
    """실행마다 설정을 기록 옆 meta 의 runs 에 덧붙인다 — 멈췄다 다시 돌린 사이에 설정이 바뀌었는지 리포트가 본다."""
    meta_path = path.with_suffix(".meta.json")
    runs = json.loads(meta_path.read_text(encoding="utf-8"))["runs"] if meta_path.exists() else []
    runs.append(
        {
            "started_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
            "limit": limit,
            "settings": dict(settings),
        }
    )
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps({"runs": runs}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _read_runs(path: Path) -> list[dict[str, Any]]:
    meta_path = path.with_suffix(".meta.json")
    return json.loads(meta_path.read_text(encoding="utf-8"))["runs"] if meta_path.exists() else []


def report(report_date: date, bench_data: Path) -> None:
    sets = [set_name for set_name in QUERY_SETS if search_path(set_name).exists()]
    if not sets:
        raise SystemExit(f"검색 기록이 없다: {OUT_DIR} (먼저 search --set …)")
    thresholds = load_answer_policy(ApiConfig.from_env().config_dir).thresholds
    measured = [_measure(set_name, bench_data, thresholds) for set_name in sets]
    text = (
        f"# 질의 변형 실험 — 검색만 ({report_date.isoformat()})\n\n"
        + _overview(measured, thresholds.high, thresholds.low)
        + _sets_section(measured)
        + _hits_section(measured)
        + _stage_section(measured)
        + _compound_section(measured)
        + _cost_section(measured)
        + _rewrite_section(measured)
        + _agreement_section(measured)
    )
    out = REPORT_DIR / f"{report_date.isoformat()}-query-variants.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    print(f"저장: {out}")


def _measure(set_name: QuerySet, bench_data: Path, thresholds: Thresholds) -> SetMeasure:
    items = {item.id: item for item in load_items(set_name, bench_data)}
    records = tuple(
        VariantRecord.from_json(row) for row in read_rows_by_id(search_path(set_name)).values()
    )
    search_runs = tuple(_read_runs(search_path(set_name)))
    k = int(search_runs[-1]["settings"]["top_k"])
    return SetMeasure(
        name=set_name,
        items=len(items),
        records=records,
        summaries=tuple(
            summarize(items, records, variant, rule=thresholds, k=k) for variant in VARIANTS
        ),
        flags=rewrite_flags(read_rewrites(rewrites_path(set_name)).values()),
        k=k,
        search_runs=search_runs,
        rewrite_runs=tuple(_read_runs(rewrites_path(set_name))),
    )


def _overview(measured: Sequence[SetMeasure], high: float, low: float) -> str:
    search = measured[0].search_runs[-1]["settings"]
    rewrite_runs = [run for m in measured for run in m.rewrite_runs]
    makers = sorted(
        {
            f"{run['settings']['spec_id']} · {' + '.join(run['settings']['prompts'])}"
            for run in rewrite_runs
        }
    )
    lines = [
        "## 설정\n",
        "- 계획: Phase 3 '질의 고쳐 쓰기 · 여러 질의 · RAG-Fusion 실험' (2026-10-29 결정)의 **검색만** 단계 — 답 생성 없음",
        "- 변형: " + " · ".join(f"`{variant}` {VARIANT_LABELS[variant]}" for variant in VARIANTS),
        f"- 검색: `{search['collection']}` — 질의마다 Dense(`{search['model']}`) · BM25 후보 {search['candidates']} → "
        f"RRF k {search['rrf_k']} 상위 {search['rerank_candidates']} → 리랭커 `{search['reranker_model']}` → 상위 "
        f"{search['top_k']}. 리랭커는 늘 원문 질문으로 매긴다 (`rewrite_max` 만 고친 질의 점수도 매겨 높은 값). "
        "RRF · 리랭커는 검색 라이브러리의 HybridRetriever · Reranker 그대로",
        f"- 관문: {'켬' if search['gate'] else '끔'} — 걸린 문항은 운영 경로처럼 검색하지 않고 집계에서 뺀다",
        f"- 질의 만들기: {' / '.join(makers) if makers else '기록 없음'}",
        f"- 단계 경로: 리랭커 1위 점수 — grounded ≥ high {high:.6f} · partial ≥ low {low:.6f} · refuse < low "
        "(configs/answer.yaml, 다시 맞추지 않음). ③-b 는 점수가 RRF 라 단계를 고르지 않는다",
        "- 정답 판정: 정답 문서이면서 정답 문자열을 담은 청크 (evalkit `contains_answer`, F9 와 같은 규칙). 복합 질문은 "
        "hop 문서 중 하나에서 정답 표기를 담은 청크. MRR 은 상위 k 안에서 (밖이면 0)",
        "- 짝 비교: 같은 문항의 ① 과 McNemar 정확 검정 (양측, evalkit `mcnemar_exact`) — 칸은 '①만 맞음 / 변형만 맞음 (p)'",
        "- 지연: 질의마다 Dense · BM25 를 한 번 받아 변형끼리 나눠 쓰고, 변형마다 그 변형이 쓰는 질의의 검색 시간을 더했다 "
        "(운영은 질의를 한 번에 하나씩 돈다). 리랭커는 변형마다 따로 불러 쟀다. LLM 은 질의 만들기 호출 한 번",
        "- 원문 없음: 질문 · 고친 질의 · 정답은 data/ 에만 — 여기에는 집계와 문항 id 만",
    ]
    if len({json.dumps(m.search_runs[-1]["settings"], sort_keys=True) for m in measured}) > 1:
        lines.append("- **주의**: 세트마다 검색 설정이 다르다 — 위 값은 첫 세트의 것")
    changed = [
        m.name
        for m in measured
        if _settings_changed(m.search_runs) or _settings_changed(m.rewrite_runs)
    ]
    if changed:
        lines.append(f"- **주의**: 멈췄다 다시 돌린 사이에 설정이 바뀐 세트 — {', '.join(changed)}")
    return "\n".join(lines) + "\n\n"


def _settings_changed(runs: Sequence[Mapping[str, Any]]) -> bool:
    return len({json.dumps(run["settings"], sort_keys=True) for run in runs}) > 1


def _sets_section(measured: Sequence[SetMeasure]) -> str:
    rows = [
        {
            "세트": SET_LABELS[m.name],
            "문항": m.items,
            "기록": len(m.records),
            "관문에서 멈춤": sum(record.gate is not None for record in m.records),
            "고친 질의 실패": len(m.flags.rewrite_failed),
            "여러 질의 실패": len(m.flags.multi_failed),
            "여러 질의 수 (개: 문항)": " · ".join(
                f"{count}: {n}" for count, n in m.flags.query_counts.items()
            ),
        }
        for m in measured
    ]
    columns = (
        "세트",
        "문항",
        "기록",
        "관문에서 멈춤",
        "고친 질의 실패",
        "여러 질의 실패",
        "여러 질의 수 (개: 문항)",
    )
    return "## 세트\n\n" + render_table(rows, columns) + "\n"


def _hits_section(measured: Sequence[SetMeasure]) -> str:
    parts = ["## 적중\n"]
    for m in measured:
        if not m.summaries[0].hit1.total:
            continue
        columns = tuple(column.replace("@k", f"@{m.k}") for column in SEARCH_COLUMNS)
        rows = [
            dict(
                zip(
                    columns,
                    (
                        VARIANT_LABELS[s.variant],
                        s.hit1.total,
                        format_share(s.hit1),
                        format_share(s.hitk),
                        "" if s.mrr is None else f"{s.mrr:.3f}",
                        _test_cell(s.hit1_test),
                        _test_cell(s.hitk_test),
                    ),
                    strict=True,
                )
            )
            for s in m.summaries
        ]
        parts.append(f"### {SET_LABELS[m.name]}\n\n" + render_table(rows, columns))
    if len(parts) == 1:
        parts.append("정답이 있는 세트의 기록이 아직 없다\n")
    return "\n".join(parts) + "\n"


def _stage_section(measured: Sequence[SetMeasure]) -> str:
    columns = ("변형", *STAGES, "① 과 다름", "grounded 로 올라감", "grounded 에서 내려감")
    parts = [
        "## 단계 경로 (리랭커 1위 점수)\n",
        "위키 밖 질문은 'grounded 로 올라감'이 위험이다 — 고친 질의가 없는 대상을 지어내 엉뚱한 문서를 끌어왔을 수 있다.\n",
    ]
    for m in measured:
        rows = []
        for s in m.summaries:
            if s.variant in UNSCORED:
                rows.append({"변형": VARIANT_LABELS[s.variant], STAGES[0]: "단계 없음 (RRF 점수)"})
                continue
            rows.append(
                {
                    "변형": VARIANT_LABELS[s.variant],
                    **{stage: s.stages.get(stage, 0) for stage in STAGES},
                    "① 과 다름": s.changed,
                    "grounded 로 올라감": len(s.to_grounded),
                    "grounded 에서 내려감": len(s.from_grounded),
                }
            )
        parts.append(
            f"### {SET_LABELS[m.name]} (검색 {m.summaries[0].searched})\n\n"
            + render_table(rows, columns)
        )
        if m.name == "offwiki":
            risky = [
                f"- {VARIANT_LABELS[s.variant]}: {_ids(s.to_grounded)}"
                for s in m.summaries
                if s.to_grounded
            ]
            parts.append(
                "위험 — grounded 로 올라간 위키 밖 문항:\n\n"
                + ("\n".join(risky) if risky else "- 없음")
                + "\n"
            )
    return "\n".join(parts) + "\n"


def _compound_section(measured: Sequence[SetMeasure]) -> str:
    compound = next((m for m in measured if m.name == "compound"), None)
    if compound is None:
        return ""
    columns = ("변형", f"hop 문서 덮은 비율 (상위 {compound.k})", f"hit@{compound.k}")
    rows = [
        {
            columns[0]: VARIANT_LABELS[s.variant],
            columns[1]: format_percent(s.hop_coverage),
            columns[2]: format_share(s.hitk),
        }
        for s in compound.summaries
    ]
    return (
        "## 복합 질문 — hop 문서 덮기\n\n"
        "여러 단계가 필요한 질문은 분해(에이전트) 몫이라, 질의 변형이 hop 문서를 더 덮는지만 본다.\n\n"
        + render_table(rows, columns)
        + "\n"
    )


def _cost_section(measured: Sequence[SetMeasure]) -> str:
    parts = [
        "## 비용 (ms)\n",
        "문항당 평균 검색 질의 수 · 리랭커 쌍 수, 지연 백분위. 합 = 질의 만들기(LLM) + 검색 + 리랭커 — 답변 생성 전까지.\n",
    ]
    for m in measured:
        rows = [
            {
                "변형": VARIANT_LABELS[s.variant],
                "문항": s.searched,
                "검색 질의 수": f"{s.queries:.2f}",
                "리랭커 쌍": f"{s.pairs:.1f}",
                "LLM p50": _percentile(s.llm, 50),
                "검색 p50": _percentile(s.search, 50),
                "리랭커 p50": _percentile(s.rerank, 50),
                "합 p50": _percentile(s.total, 50),
                "합 p90": _percentile(s.total, 90),
            }
            for s in m.summaries
        ]
        parts.append(f"### {SET_LABELS[m.name]}\n\n" + render_table(rows, COST_COLUMNS))
    return "\n".join(parts) + "\n"


def _rewrite_section(measured: Sequence[SetMeasure]) -> str:
    parts = [
        "## 고친 질의 점검 — 눈으로 볼 문항\n",
        "날짜 · 조건이 빠지거나 없는 대상이 끼어드는지 표본을 눈으로 본다 (계획의 '조심'). 숫자(날짜 · 수량 · 순번)가 빠지거나 "
        "새로 생긴 문항을 먼저 본다 — 숫자 없는 조건 · 대상은 이 표시로 잡히지 않는다.\n",
    ]
    for m in measured:
        flags = m.flags
        parts.append(
            f"### {SET_LABELS[m.name]} (기록 {flags.total})\n\n"
            f"- 고친 질의가 원문과 같음: {len(flags.unchanged)}\n"
            f"- 고친 질의에서 숫자가 빠짐: {_ids(flags.rewrite_dropped)}\n"
            f"- 고친 질의에 숫자가 생김: {_ids(flags.rewrite_added)}\n"
            f"- 여러 질의 중 숫자가 빠짐: {_ids(flags.multi_dropped)}\n"
            f"- 여러 질의 중 숫자가 생김: {_ids(flags.multi_added)}\n"
            f"- 고친 질의 실패: {_ids(flags.rewrite_failed)}\n"
            f"- 여러 질의 실패: {_ids(flags.multi_failed)}\n"
        )
    return "\n".join(parts) + "\n"


def _agreement_section(measured: Sequence[SetMeasure]) -> str:
    """① 이 F9 민감도 기록(운영 SearchService)과 같은 정답 순위를 내는지 — 공개 클래스로 다시 조립한 검색이 운영과 같은지."""
    if not SENSITIVITY_PATH.exists():
        return ""
    rows = read_jsonl(SENSITIVITY_PATH)
    fields = {"korquad": "original", "paraphrase": "paraphrase"}
    lines = []
    for m in measured:
        if m.name not in fields:
            continue
        expected = {row["id"]: row[fields[m.name]]["answer_rank"] for row in rows}
        agreement, differ = rank_agreement(m.records, expected)
        lines.append(
            f"- {SET_LABELS[m.name]}: 같음 {format_share(agreement)} · 다른 문항 {_ids(differ)}"
        )
    if not lines:
        return ""
    return (
        "## ① 재현 확인\n\n"
        f"① 의 정답 순위를 F9 민감도 기록(`{SENSITIVITY_PATH.as_posix()}`, 운영 SearchService)과 견준다. 다르면 조립이 "
        "운영 검색과 어긋났거나 그 뒤 색인 · 모델이 바뀐 것이다.\n\n" + "\n".join(lines) + "\n"
    )


def _test_cell(test: McNemarResult | None) -> str:
    return "" if test is None else f"{test.only_a} / {test.only_b} (p {test.p_value:.3g})"


def _percentile(value: Latency, q: int) -> str:
    point = value.percentiles.get(q)
    return "" if point is None else f"{point:.0f}"


def _ids(ids: Sequence[str]) -> str:
    if not ids:
        return "0"
    shown = ", ".join(ids[:MAX_IDS_SHOWN])
    rest = len(ids) - MAX_IDS_SHOWN
    return f"{len(ids)} — {shown}" + (f" 외 {rest}" if rest > 0 else "")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["rewrite", "search", "report"])
    # rewrite · search 만: 세트 하나
    parser.add_argument("--set", choices=QUERY_SETS)
    # rewrite · search 만: 남은 문항 중 앞에서 N 개만 (점검용)
    parser.add_argument("--limit", type=_positive)
    # 롱테일 세트 폴더 (LLM 레포 bench/data). worktree 에서는 절대 경로로
    parser.add_argument("--bench-data", type=Path, default=BENCH_DATA)
    # report 만: 리포트 날짜(파일 이름 · 제목). 없으면 오늘
    parser.add_argument("--report-date", type=date.fromisoformat)
    args = parser.parse_args(argv)
    if args.step in ("rewrite", "search") and args.set is None:
        parser.error(f"{args.step} 는 --set 으로 세트를 준다")
    if args.step == "report" and (args.set is not None or args.limit is not None):
        parser.error("report 는 기록이 있는 세트 전부를 그린다 — --set · --limit 을 받지 않는다")
    if args.report_date is not None and args.step != "report":
        parser.error("--report-date 는 report 에만 쓴다")
    return args


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"1 이상을 준다, got {value!r}")
    return number


def main() -> None:
    args = _parse_args()
    if args.step == "rewrite":
        rewrite(args.set, limit=args.limit, bench_data=args.bench_data)
    elif args.step == "search":
        search(args.set, limit=args.limit, bench_data=args.bench_data)
    elif args.step == "report":
        report(args.report_date or datetime.now(tz=UTC).astimezone().date(), args.bench_data)


if __name__ == "__main__":
    main()
