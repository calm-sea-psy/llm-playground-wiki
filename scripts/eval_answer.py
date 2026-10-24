"""F9 답변 평가 — 실제 답변 경로(`/ask` 와 같은 AnswerService)로 평가 문항을 돌려 자동으로 셀 수 있는 지표를 모은다.

단계별로 돌린다 — 단계마다 결과를 data/ 에 저장해, 뒤 단계를 다시 돌릴 때 앞 단계를 반복하지 않는다.
  sample   KorQuAD dev 의 eval 몫(문단 해시 40%)에서 문단마다 한 문항을 고르고, 정답 문서가 코퍼스에 있고 정답 문장이
           지금 문서에 남은 것만 + 위키 밖 질문(data/eval/offwiki-v1.jsonl) 전부 → data/eval/f9-sample-v1.jsonl
           (calibrate_threshold.py sample 과 같은 방식에 몫만 다르다 — 문단 해시라 calibrate · F4 와 겹치지 않는다)
  run      표본을 서버 없이 /ask 와 같은 조립(관문 · 검색 · 답변 정책 · grounded → partial 캐스케이드)으로 돌려
           문항마다 SSE 이벤트를 기록 한 줄로 → data/eval/f9-answers-v1.jsonl (+ .meta.json 에 실행마다 설정).
           단계 경로 · 표지 · 답 원문 · 검색 청크 · 출처 번호(cited · dropped) · 잘림 · 시간 초과 · 진단 시간 · 1위 점수.
           끝난 문항은 건너뛰어 멈췄다 다시 돌려도 된다 (생성 실패 기록은 지우고 다시 돈다). --limit N 은 N 문항만 (점검용).
           PG · Qdrant(docker compose up) · BM25 색인(data/index/bm25/{컬렉션}) · Ollama 가 필요하다
  retrieval-sensitivity
           패러프레이즈 v1(data/bench/paraphrase-v1.jsonl)의 f9-random 300 을 원문 · 패러프레이즈로 같은 검색
           (Hybrid + 리랭커, 상위 5)에 돌려 정답 근거 청크 순위를 → data/eval/f9-retrieval-sensitivity-v1.jsonl (+ .meta.json).
           LLM 은 쓰지 않는다. PG · Qdrant · BM25 색인이 필요하다
  report   reports/{날짜}-answer-eval.md 에 집계만 (질문 · 답 원문 없음). 민감도 결과가 없으면 그 절은 "아직"으로 둔다.
           민감도 절에는 패러프레이즈 사람 확인 요약과, 뜻이 조금 달라진 문항(PARAPHRASE_SHIFTED)을 뺀 hit 도 싣는다

지표 규칙은 playground_wiki.evaluation.answer_metrics, 실행 기록은 playground_wiki.evaluation.answer_run.
KorQuAD 는 CC BY-ND 라 질문 · 답 원문 · 패러프레이즈는 data/ 에만 둔다 (git 제외). 레포 루트에서 돌린다.

사용법:
  uv run python scripts/eval_answer.py sample
  uv run python scripts/eval_answer.py run --limit 5
  uv run python scripts/eval_answer.py run
  uv run python scripts/eval_answer.py retrieval-sensitivity
  uv run python scripts/eval_answer.py report
"""

import argparse
import asyncio
import json
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path

import pyarrow.parquet as pq
from kiwipiepy import Kiwi
from playground_evalkit import (
    KorQuADItem,
    contains_answer,
    load_korquad,
    normalize_title,
    render_table,
)
from playground_llm import PromptLibrary
from playground_llm.ollama import OllamaConfig, OllamaLLM
from playground_retrieval.search import KiwiAnalyzer, SearchResult, SearchService
from playground_retrieval.store import PgStore, QdrantStore

from playground_wiki.api.answer import AnswerService
from playground_wiki.api.config import (
    ApiConfig,
    config_dir_from_env,
    load_answer_policy,
    load_dump_date,
    load_query_gate,
)
from playground_wiki.evaluation import clopper_pearson, korquad_paragraphs, sample_per_paragraph
from playground_wiki.evaluation.answer_metrics import (
    ANSWER_STAGES,
    FINAL_STAGES,
    LATENCY_PERCENTILES,
    MARKER_MISSING_LIMIT,
    SCORE_BAND_EDGES,
    Latency,
    Share,
    answer_rank,
    cites_gold,
    contains_gold,
    final_stage,
    gold_in_hits,
    has_dropped,
    latency,
    marker_mismatch,
    marker_missing,
    matches_expected,
    paired_hits,
    path_counts,
    score_band,
    score_bands,
    share,
    stage_counts,
    stage_path,
)
from playground_wiki.evaluation.answer_run import (
    AnswerRecord,
    AnswerRun,
    EvalItem,
    Searcher,
    read_records,
)
from playground_wiki.gate import KiwiTokenizer
from playground_wiki.generation import PROMPT_DIR, AnswerPolicy, check_profiles

SUBSET = Path("data/parsed/kowiki-subset-10000")  # subset.json: KorQuAD dev 경로 · 제목 보정표 (D4)
# 전체 덤프 파싱 결과 — 적재한 kowiki_kure-v1 컬렉션과 같은 문서
PARSED = Path("data/parsed/kowiki")
OFFWIKI_PATH = Path("data/eval/offwiki-v1.jsonl")
SAMPLE_PATH = Path("data/eval/f9-sample-v1.jsonl")
ANSWERS_PATH = Path("data/eval/f9-answers-v1.jsonl")
PARAPHRASE_PATH = Path("data/bench/paraphrase-v1.jsonl")
# 검색 결과로 거르지 않고 뽑은 표본 — F4 답 있음 300 은 "상위 5 에 정답"으로 고른 세트라 원문 hit@5 가 구성상 100%
PARAPHRASE_SAMPLE = "f9-random"
# 사람 확인 (2026-10-24, data/bench/paraphrase-v1-review-checked.md — 표본마다 seed 0 으로 25개). 뜻이 조금 달라진 것은
# 빼지 않고 전부로 재되, 뺀 hit 차이도 한 줄 같이 낸다
PARAPHRASE_REVIEW = "확인 25개 중 정답 바뀜 0 · 정답이 질문에 샘 0 · 뜻이 조금 달라짐 3"
PARAPHRASE_SHIFTED = frozenset(
    {
        "6589969-9-1",  # '소송을 걸었는가' → '고소한', '신문사' → '언론사' — 범위가 넓어짐
        "6559027-10-2",  # '함장 중 누구였나' — 함장이 여럿인 것처럼 읽힘
        "6507936-13-1",  # '모두 합쳐' 가 붙어 범위가 넓어짐
    }
)
SENSITIVITY_PATH = Path("data/eval/f9-retrieval-sensitivity-v1.jsonl")
REPORT_DIR = Path("reports")
SEED = 0
NEXT_STEPS = (
    "분리 답변(partial) 50개 판정 — 근거와 일반 지식을 나눠 썼는지 · 해로운 답이 있는지 (수동 또는 판정 모델)",
    "유해 정보 20개 — partial 이 위키 수준을 넘는 위험한 절차를 덧붙이는지, 과잉 거절 문항과 함께",
    "판정 모델 검증 — 판정기를 쓰는 곳(분리 답변 · 유해 정보 · 줄글 거절)마다 사람 채점 50개와 일치율",
    "간접 인젝션 끝까지 — 원본 위키 마크업에 심은 지시가 정제 → 적재 → 검색 → /ask 까지 살아남는지",
    "점수 구간 층화 표본과 grounded 지어냄 판정 — 0.83~0.94 구간 약 100개로 high 확정 (ADR-024)",
    "짧은 답 필드를 따로 받아 EM · F1, 검색기별(BM25 · Dense · Hybrid) 표현 민감도",
)


def main() -> None:
    args = _parse_args()
    if args.step == "sample":
        sample()
    elif args.step == "run":
        run(limit=args.limit)
    elif args.step == "retrieval-sensitivity":
        retrieval_sensitivity()
    elif args.step == "report":
        report()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["sample", "run", "retrieval-sensitivity", "report"])
    # run 만: 이번에 돌릴 최대 문항 수 (전체를 돌리기 전 점검용 — 끝난 문항은 다음 실행이 건너뛴다)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def sample() -> None:
    summary = json.loads((SUBSET / "subset.json").read_text(encoding="utf-8"))
    korquad = Path(summary["korquad_dev"])
    items = load_korquad(korquad, title_aliases=summary["korquad_title_map"])
    paragraphs = korquad_paragraphs(korquad)
    texts = _document_texts({item.title for item in items})
    offwiki = _read_jsonl(OFFWIKI_PATH)
    rows, picked = sample_rows(items, paragraphs, texts, offwiki)
    _write_jsonl(SAMPLE_PATH, rows)

    kept = sum(row["source"] == "korquad" for row in rows)
    print(
        f"KorQuAD dev 질문 {len(items):,} → eval 몫 문단마다 한 문항 {picked} "
        f"→ 정답이 코퍼스에 남은 것 {kept} (빠짐 {picked - kept}, {1 - kept / picked:.1%})\n"
        f"위키 밖 질문 {len(offwiki)}\n"
        f"저장: {SAMPLE_PATH} ({len(rows)}줄)"
    )


def sample_rows(
    items: Sequence[KorQuADItem],
    paragraphs: Mapping[str, str],
    texts: Mapping[str, str],
    offwiki: Iterable[Mapping[str, object]],
) -> tuple[list[dict], int]:
    """표본 줄(KorQuAD 다음 위키 밖)과, 정답 문장을 대조하기 전에 문단마다 고른 KorQuAD 문항 수."""
    in_corpus = [item for item in items if item.title in texts]
    picked = sample_per_paragraph(in_corpus, paragraphs, "eval", seed=SEED)
    # 문서는 있어도 정답 문장이 2018년 이후 고쳐졌으면 뺀다 — 검색이 아니라 문자열 대조라 선택 편향이 없다
    kept = [item for item in picked if contains_answer(item.title, texts[item.title], item)]
    rows: list[dict] = [
        {
            "id": item.id,
            "source": "korquad",
            "question": item.question,
            "gold_title": item.title,
            "gold_answer": item.answer,
            "paragraph": paragraphs[item.id],
        }
        for item in kept
    ]
    rows += [
        {
            "id": row["id"],
            "source": "offwiki",
            "question": row["question"],
            "expected": row["expected"],
            "category": row["category"],
            "trap": row["trap"],
            "related_title": row["related_title"],
        }
        for row in offwiki
    ]
    return rows, len(picked)


def run(limit: int | None) -> None:
    items = [EvalItem.from_json(row) for row in _read_jsonl(SAMPLE_PATH)]
    config = ApiConfig.from_env()
    config_dir = config_dir_from_env()
    policy = load_answer_policy(config_dir)
    library = PromptLibrary.load(PROMPT_DIR)
    check_profiles(policy.profiles.values(), library)
    # /ask 와 같이 Kiwi 하나(약 500MB)를 질문 관문과 BM25 가 나눠 쓴다
    kiwi = Kiwi()
    gate = load_query_gate(config_dir, KiwiTokenizer(kiwi))
    dump_date = load_dump_date(config_dir)
    _append_run_meta(
        config, policy, gate_enabled=gate is not None, dump_date=dump_date, limit=limit
    )
    print(
        f"문항 {len(items)} · collection={config.search.collection} · 관문 {'켬' if gate else '끔'}",
        flush=True,
    )
    start = time.perf_counter()
    with PgStore.connect(config.pg) as pg, OllamaLLM.connect(OllamaConfig.from_env()) as llm:
        service = SearchService.build(
            config.search, pg, QdrantStore.from_env(), analyzer=KiwiAnalyzer(kiwi)
        )
        print(f"검색 서비스 준비 {time.perf_counter() - start:.0f}s", flush=True)

        def make_answerer(search: Searcher) -> AnswerService:
            return AnswerService(
                search=search,
                policy=policy,
                library=library,
                llm=llm,
                gate=gate,
                dump_date=dump_date,
            )

        runner = AnswerRun(service, make_answerer)
        summary = asyncio.run(runner.run(items, ANSWERS_PATH, limit=limit, on_record=_progress))
    print(
        f"이번 실행 {summary.recorded} · 건너뜀 {summary.skipped} · 다시 돈 생성 실패 {summary.retried} "
        f"/ 문항 {summary.total} · {time.perf_counter() - start:.0f}s\n저장: {ANSWERS_PATH}"
    )


def _progress(index: int, total: int, record: AnswerRecord) -> None:
    print(
        f"  {index}/{total} {record.item.id} {stage_path(record)} {record.wall_ms / 1000:.1f}s",
        flush=True,
    )


def _append_run_meta(
    config: ApiConfig,
    policy: AnswerPolicy,
    *,
    gate_enabled: bool,
    dump_date: str,
    limit: int | None,
) -> None:
    """실행마다 설정을 meta 의 runs 에 덧붙인다 — 멈췄다 다시 돌린 사이에 설정이 바뀌었는지 리포트가 본다."""
    meta_path = ANSWERS_PATH.with_suffix(".meta.json")
    runs = json.loads(meta_path.read_text(encoding="utf-8"))["runs"] if meta_path.exists() else []
    runs.append(
        {
            "started_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
            "limit": limit,
            "settings": {
                "collection": config.search.collection,
                "model": config.search.model,
                "candidates": config.search.candidates,
                "rerank_candidates": config.search.rerank_candidates,
                "reranker_model": config.search.reranker_model,
                "top_k": config.search.top_k,
                "thresholds": {"high": policy.thresholds.high, "low": policy.thresholds.low},
                "gate": gate_enabled,
                "dump_date": dump_date,
                "profiles": {
                    stage: {
                        "spec_id": profile.spec.spec_id,
                        "system": list(profile.system),
                        "user": profile.user,
                        "timeout_sec": profile.timeout_sec,
                    }
                    for stage, profile in sorted(policy.profiles.items())
                },
                "inputs": {"sample": str(SAMPLE_PATH)},
            },
        }
    )
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps({"runs": runs}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def retrieval_sensitivity() -> None:
    rows = [row for row in _read_jsonl(PARAPHRASE_PATH) if row["sample"] == PARAPHRASE_SAMPLE]
    config = ApiConfig.from_env()
    print(
        f"collection={config.search.collection} · 문항 {len(rows)} × 원문 · 패러프레이즈",
        flush=True,
    )
    start = time.perf_counter()
    with PgStore.connect(config.pg) as pg:
        service = SearchService.build(config.search, pg, QdrantStore.from_env())
        print(f"검색 서비스 준비 {time.perf_counter() - start:.0f}s", flush=True)
        originals = asyncio.run(_search_all(service, [row["question"] for row in rows]))
        paraphrases = asyncio.run(_search_all(service, [row["paraphrase"] for row in rows]))

    out = [
        sensitivity_row(row, original, paraphrase)
        for row, original, paraphrase in zip(rows, originals, paraphrases, strict=True)
    ]
    _write_jsonl(SENSITIVITY_PATH, out)
    meta = {
        "created_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "collection": config.search.collection,
        "model": config.search.model,
        "candidates": config.search.candidates,
        "rerank_candidates": config.search.rerank_candidates,
        "reranker_model": config.search.reranker_model,
        "top_k": config.search.top_k,
        "inputs": {"paraphrase": str(PARAPHRASE_PATH), "sample": PARAPHRASE_SAMPLE},
    }
    meta_path = SENSITIVITY_PATH.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    table = paired_hits(_hit_pairs(out, config.search.top_k))
    print(
        f"hit@{config.search.top_k} 원문 {_percent(table.original_rate)} · "
        f"패러프레이즈 {_percent(table.paraphrase_rate)} "
        f"(원문만 {table.original_only} · 패러프레이즈만 {table.paraphrase_only}, "
        f"McNemar p={table.mcnemar_p():.3g})\n저장: {SENSITIVITY_PATH} · {meta_path}"
    )


async def _search_all(service: SearchService, queries: list[str]) -> list[SearchResult]:
    # 서비스가 질의를 한 번에 하나씩 돌리므로(GPU 모델 · PG 접속 공유) 순서대로 기다린다
    results = []
    for index, query in enumerate(queries, start=1):
        results.append(await service.search(query))
        if index % 50 == 0:
            print(f"  {index}/{len(queries)}", flush=True)
    return results


def sensitivity_row(row: Mapping, original: SearchResult, paraphrase: SearchResult) -> dict:
    """패러프레이즈 한 줄의 원문 · 패러프레이즈 검색 결과 (정답 근거 청크 순위 — 질문 원문은 싣지 않는다)."""
    item = KorQuADItem(
        id=row["id"],
        title=normalize_title(row["title"]),
        question=row["question"],
        answer=row["answers"][0],
        source_title=row["title"],
    )
    return {
        "id": row["id"],
        "title": item.title,
        "original": _search_fields(original, item),
        "paraphrase": _search_fields(paraphrase, item),
    }


def _search_fields(result: SearchResult, item: KorQuADItem) -> dict:
    return {
        "answer_rank": answer_rank(result.hits, item),
        "top_score": result.hits[0].score if result.hits else None,
        "chunk_ids": [hit.chunk.id for hit in result.hits],
        "took_ms": result.took_ms,
    }


def _hit_pairs(rows: Iterable[Mapping], k: int) -> list[tuple[bool, bool]]:
    return [
        (_hit(row["original"]["answer_rank"], k), _hit(row["paraphrase"]["answer_rank"], k))
        for row in rows
    ]


def _hit(rank: int | None, k: int) -> bool:
    return rank is not None and rank <= k


def report() -> None:
    if not ANSWERS_PATH.exists():
        raise SystemExit(f"{ANSWERS_PATH} 가 없다 — run 을 먼저 돌린다")
    records = read_records(ANSWERS_PATH)
    runs = json.loads(ANSWERS_PATH.with_suffix(".meta.json").read_text(encoding="utf-8"))["runs"]
    sample_count = len(_read_jsonl(SAMPLE_PATH))
    sensitivity: tuple[list[dict], dict] | None = None
    if SENSITIVITY_PATH.exists():
        meta_path = SENSITIVITY_PATH.with_suffix(".meta.json")
        sensitivity = (
            _read_jsonl(SENSITIVITY_PATH),
            json.loads(meta_path.read_text(encoding="utf-8")),
        )
    today = datetime.now(tz=UTC).date()
    text = render_report(records, runs, sample_count, sensitivity, today=today)
    REPORT_DIR.mkdir(exist_ok=True)
    path = REPORT_DIR / f"{today:%Y-%m-%d}-answer-eval.md"
    # Windows 에서도 LF 로 (리포트는 커밋하고 .gitattributes 가 LF)
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"기록 {len(records)} / 표본 {sample_count} · 리포트: {path}")


def render_report(
    records: Sequence[AnswerRecord],
    runs: Sequence[Mapping],
    sample_count: int,
    sensitivity: tuple[Sequence[Mapping], Mapping] | None,
    *,
    today: date,
) -> str:
    """리포트 markdown — 집계만 (질문 · 답 원문 · 청크 본문을 싣지 않는다)."""
    korquad = [record for record in records if record.item.source == "korquad"]
    offwiki = [record for record in records if record.item.source == "offwiki"]
    return (
        f"# F9 답변 평가 — 자동 지표 ({today:%Y-%m-%d})\n\n"
        + _overview(records, korquad, offwiki, runs, sample_count)
        + "\n## 생성 설정 — 기록의 spec_id · 프롬프트 버전\n\n"
        + _spec_table(records)
        + "\n## 단계 경로\n\n"
        + _path_section(korquad, offwiki)
        + "\n## KorQuAD — 근거 답변\n\n"
        + _korquad_section(korquad)
        + "\n## 리랭커 1위 점수 구간 — KorQuAD (ADR-024)\n\n"
        + _band_section(korquad)
        + "\n## 표지 · 출처 번호 · 잘림 (LLM 을 부른 문항 전부)\n\n"
        + _marker_section(records)
        + "\n## 지연 (초) — 질문을 받은 때부터\n\n"
        + _latency_section(records)
        + "\n## 위키 밖 질문 — 기대 단계와 실제 경로\n\n"
        + _offwiki_section(offwiki)
        + "\n## 검색의 표현 민감도 — 원문 vs 패러프레이즈\n\n"
        + _sensitivity_section(sensitivity)
        + "\n## 다음 일\n\n"
        + "".join(f"- {step}\n" for step in NEXT_STEPS)
    )


def _overview(
    records: Sequence[AnswerRecord],
    korquad: Sequence[AnswerRecord],
    offwiki: Sequence[AnswerRecord],
    runs: Sequence[Mapping],
    sample_count: int,
) -> str:
    settings = runs[-1]["settings"]
    distinct = {json.dumps(run["settings"], sort_keys=True) for run in runs}
    changed = (
        f"- **실행 {len(runs)}번 사이에 설정이 바뀌었다** ({len(distinct)}가지) — 아래 수치는 섞여 있다. "
        "생성 설정 표의 spec_id · 프롬프트 버전으로 나눠 본다\n"
        if len(distinct) > 1
        else ""
    )
    thresholds = settings["thresholds"]
    return (
        f"- 입력: KorQuAD dev eval 몫(문단 해시 40%, 문단마다 한 문항, 정답 문서가 코퍼스에 있고 정답 문장이 지금 "
        f"문서에 남은 것) {len(korquad)} + 위키 밖 질문 offwiki v1 {len(offwiki)} — 표본 {sample_count} 중 "
        f"기록 {len(records)}\n"
        f"- 답변 경로: `/ask` 와 같은 조립을 서버 없이 — 관문 {'켬' if settings['gate'] else '끔'} → 검색 "
        f"`{settings['collection']}` Hybrid(후보 {settings['candidates']}) + 리랭커 `{settings['reranker_model']}`"
        f"(후보 {settings['rerank_candidates']}) 상위 {settings['top_k']} → 답변 정책(high {thresholds['high']:.6f} · "
        f"low {thresholds['low']:.6f}) → grounded@v2 첫 줄 표지, 못 찾음 · 빈 답 · 시간 초과면 partial "
        "(ADR-024 · ADR-027)\n"
        + changed
        + "- 정답 포함: 답에 KorQuAD 정답 문자열이 있음 (공백 무시 — evalkit `contains_answer` 규칙). "
        "답이 문장이라 EM · F1 은 쓰지 않는다\n"
        "- 정답 근거 청크: 정답 문서의 청크이면서 정답 문자열을 담은 것 (검색 평가와 같은 기준)\n"
        "- 위키 밖 v1 은 임계값을 고를 때(calibrate) 쓴 세트라 검증이 아니라 경로 확인용이다 — 검증은 offwiki v2\n"
        "- 질문 · 답 원문은 로컬 기록(data/eval/f9-answers-v1.jsonl)에만 둔다 (KorQuAD CC BY-ND)\n"
    )


def _spec_table(records: Sequence[AnswerRecord]) -> str:
    counts = Counter(
        (final_stage(record), record.spec_id, record.prompt_version)
        for record in records
        if record.spec_id is not None
    )
    rows = [
        {"끝난 단계": stage, "spec_id": spec_id, "프롬프트 버전": f"`{version}`", "문항": count}
        for (stage, spec_id, version), count in sorted(counts.items())
    ]
    if not rows:
        return "LLM 을 부른 기록이 없다\n"
    return render_table(rows, ["끝난 단계", "spec_id", "프롬프트 버전", "문항"])


def _path_section(korquad: Sequence[AnswerRecord], offwiki: Sequence[AnswerRecord]) -> str:
    groups = {
        "KorQuAD · 검색 상위에 정답 근거": [r for r in korquad if gold_in_hits(r)],
        "KorQuAD · 상위에 없음": [r for r in korquad if not gold_in_hits(r)],
        "위키 밖": list(offwiki),
    }
    counts = {name: path_counts(group) for name, group in groups.items()}
    rows = [
        {"경로": path, **{name: counts[name].get(path, 0) for name in groups}}
        for path in path_counts([*korquad, *offwiki])
    ]
    rows.append({"경로": "합계", **{name: len(group) for name, group in groups.items()}})
    return (
        render_table(rows, ["경로", *groups])
        + "\n'검색 상위' 는 답변 경로가 받은 상위 결과 전부 (low 미만이라 문맥에서 뺀 것 포함). "
        "→ 는 캐스케이드(못 찾음 · 빈 답 · 시간 초과로 다음 단계), 끝의 error 는 답을 하나도 못 보낸 실패\n"
    )


def _korquad_section(korquad: Sequence[AnswerRecord]) -> str:
    grounded = [r for r in korquad if final_stage(r) == "grounded"]
    partial = [r for r in korquad if final_stage(r) == "partial"]
    rows = [
        {
            "지표": "검색 상위에 정답 근거 청크 (답변 경로의 hit@k)",
            "값": _fmt(share(gold_in_hits(r) for r in korquad)),
            "분모": "KorQuAD 전부",
        },
        {
            "지표": "grounded 로 끝남",
            "값": _fmt(Share(len(grounded), len(korquad))),
            "분모": "KorQuAD 전부",
        },
        {
            "지표": "정답 포함",
            "값": _fmt(share(contains_gold(r) for r in grounded)),
            "분모": "grounded 로 끝난 답",
        },
        {
            "지표": "정답 근거 청크를 출처 번호로 짚음",
            "값": _fmt(share(cites_gold(r) for r in grounded)),
            "분모": "grounded 로 끝나고 문맥에 정답 근거 청크가 있는 답",
        },
        {
            "지표": "출처 번호가 하나도 없음",
            "값": _fmt(share(not r.cited for r in grounded)),
            "분모": "grounded 로 끝난 답",
        },
        {
            "지표": "정답 포함 (partial)",
            "값": _fmt(share(contains_gold(r) for r in partial)),
            "분모": "partial 로 끝난 답 (캐스케이드 포함) — 일반 지식이나 다른 문서로 맞힌 것",
        },
    ]
    return render_table(rows, ["지표", "값", "분모"])


def _band_section(korquad: Sequence[AnswerRecord]) -> str:
    groups: dict[str, list[AnswerRecord]] = defaultdict(list)
    for record in korquad:
        groups[score_band(record.top_score)].append(record)
    rows = []
    for band in score_bands():
        group = groups.get(band, [])
        if not group:
            continue
        grounded = [r for r in group if final_stage(r) == "grounded"]
        rows.append(
            {
                "1위 점수": band,
                "문항": len(group),
                "grounded 로 끝남": len(grounded),
                "그중 정답 포함": _fmt(share(contains_gold(r) for r in grounded)),
                "grounded → partial": sum(r.stage_path == ("grounded", "partial") for r in group),
                "partial 만": sum(r.stage_path == ("partial",) for r in group),
                "그 밖": sum(
                    r.stage_path not in {("grounded",), ("grounded", "partial"), ("partial",)}
                    for r in group
                ),
            }
        )
    lower, upper = SCORE_BAND_EDGES
    middle = [r for r in korquad if r.top_score is not None and lower <= r.top_score < upper]
    middle_grounded = sum(final_stage(r) == "grounded" for r in middle)
    return (
        render_table(
            rows,
            [
                "1위 점수",
                "문항",
                "grounded 로 끝남",
                "그중 정답 포함",
                "grounded → partial",
                "partial 만",
                "그 밖",
            ],
        )
        + f"\n- {lower:g}~{upper:g} 구간에서 grounded 로 끝난 답 {middle_grounded}개 — 구간끼리 지어냄 비율을 가르려면 "
        "구간당 약 100개가 필요하다 (ADR-024). 이 표본은 점수로 층화하지 않아 적다 → 층화 표본 · 지어냄 판정은 다음 일\n"
        "- 정답 미포함이 곧 지어냄은 아니다 (같은 사실을 다른 말로 · 일부만 답함). 지어냄은 문맥과 대조해 판정한다\n"
    )


def _marker_section(records: Sequence[AnswerRecord]) -> str:
    missing = share(marker_missing(r) for r in records)
    _, missing_upper = clopper_pearson(missing.count, missing.total) if missing.total else (0, 1)
    mismatch = share(marker_mismatch(r) for r in records)
    mismatch_gold = sum(bool(marker_mismatch(r)) and bool(contains_gold(r)) for r in records)
    called = [r for r in records if ANSWER_STAGES.intersection(r.stage_path)]
    answered = [r for r in called if final_stage(r) in ANSWER_STAGES]
    errors = Counter(r.error for r in records if r.error is not None)
    rows = [
        {
            "지표": "표지 누락",
            "값": _fmt(missing),
            "분모": "grounded 표지를 읽은 문항",
            "메모": f"95% 상한 {missing_upper:.1%} — {MARKER_MISSING_LIMIT:.0%} 미만이면 "
            "'표지 없으면 찾음' 유지, 이상이면 반대로 (ADR-024)",
        },
        {
            "지표": "표지 '찾음' · 본문 '정보 없음' 어긋남",
            "값": _fmt(mismatch),
            "분모": "grounded 가 '찾음'으로 답을 낸 문항",
            "메모": f"정규식이라 상한 (곁가지 '정보 없음'도 걸린다). 그중 KorQuAD 정답 포함 {mismatch_gold}",
        },
        {
            "지표": "범위 밖 출처 번호를 뗀 답",
            "값": _fmt(share(has_dropped(r) for r in records)),
            "분모": "LLM 이 답을 쓴 문항",
            "메모": "서버가 떼고 로그에 남긴다 (ADR-026)",
        },
        {
            "지표": "잘림 (max_tokens 도달)",
            "값": _fmt(Share(sum(r.truncated for r in answered), len(answered))),
            "분모": "LLM 이 답을 쓴 문항",
            "메모": "답을 낸 단계만 — 잘려 빈 답이 된 grounded 는 캐스케이드로 넘어가 여기 안 잡힌다",
        },
        {
            "지표": "시간 초과로 끊김",
            "값": _fmt(Share(sum(r.timed_out for r in answered), len(answered))),
            "분모": "LLM 이 답을 쓴 문항",
            "메모": "보낸 데까지로 done",
        },
        {
            "지표": "답을 못 보낸 실패 (error)",
            "값": _fmt(Share(sum(errors.values()), len(called))),
            "분모": "LLM 을 부른 문항",
            "메모": " · ".join(f"{code} {count}" for code, count in sorted(errors.items()))
            or "없음",
        },
    ]
    return render_table(rows, ["지표", "값", "분모", "메모"])


def _latency_section(records: Sequence[AnswerRecord]) -> str:
    groups: dict[str, list[AnswerRecord]] = defaultdict(list)
    for record in records:
        groups[stage_path(record)].append(record)
    rows = [
        _latency_row(path, groups[path])
        for path in path_counts(records)
        if not path.endswith("error")
    ]
    columns = [
        "경로",
        "문항",
        *(f"첫 글자 p{q}" for q in LATENCY_PERCENTILES),
        *(f"전체 p{q}" for q in LATENCY_PERCENTILES),
    ]
    return (
        render_table(rows, columns)
        + "\n첫 글자 = 첫 답 조각을 보낸 때 (관문 · 거절은 고정 문구라 비운다), 전체 = done 까지. "
        "done 진단의 시간 — 실패(error)로 끝난 문항은 진단이 없어 뺀다\n"
    )


def _latency_row(path: str, group: Sequence[AnswerRecord]) -> dict:
    first = latency(r.first_delta_ms for r in group)
    total = latency(r.total_ms for r in group)
    return {
        "경로": path,
        "문항": len(group),
        **{f"첫 글자 p{q}": _seconds(first, q) for q in LATENCY_PERCENTILES},
        **{f"전체 p{q}": _seconds(total, q) for q in LATENCY_PERCENTILES},
    }


def _offwiki_section(offwiki: Sequence[AnswerRecord]) -> str:
    if not offwiki:
        return "기록이 없다\n"
    groups: dict[str, list[AnswerRecord]] = defaultdict(list)
    totals: dict[str, list[AnswerRecord]] = defaultdict(list)
    for record in offwiki:
        name = f"{record.item.expected}/{record.item.category}"
        groups[name + (" (trap)" if record.item.trap else "")].append(record)
        totals[f"**{record.item.expected} 전체**"].append(record)
    rows = [_offwiki_row(name, group) for name, group in sorted(groups.items())]
    rows += [_offwiki_row(name, group) for name, group in sorted(totals.items())]
    return (
        render_table(rows, ["묶음", "문항", *FINAL_STAGES, "기대대로"])
        + "\n- 기대대로: refuse → 관문 · 거절, partial → partial. trap = 이름이 같은 문서가 있어 검색에 걸리기 쉬운 질문\n"
        "- refuse 기대가 partial 로 끝난 것은 설계상 허용이다 (ADR-025 — 거절은 점수가 아니라 partial 프롬프트의 "
        "'알 수 없음'과 관문이 맡는다). 그 답이 괜찮은지는 분리 답변 판정(다음 일)이 본다\n"
        "- partial 기대가 grounded 로 끝난 것은 '찾음' 표지로 문서 근거라고 답한 것 — 지어냄 후보다\n"
    )


def _offwiki_row(name: str, group: Sequence[AnswerRecord]) -> dict:
    return {
        "묶음": name,
        "문항": len(group),
        **stage_counts(group),
        "기대대로": _fmt(share(matches_expected(r) for r in group)),
    }


def _sensitivity_section(sensitivity: tuple[Sequence[Mapping], Mapping] | None) -> str:
    if sensitivity is None:
        return "아직 돌리지 않았다 (`retrieval-sensitivity`)\n"
    rows, meta = sensitivity
    k = int(meta["top_k"])
    table_rows = [_sensitivity_row(f"hit@{cutoff}", rows, cutoff) for cutoff in sorted({1, k})]
    kept = [row for row in rows if row["id"] not in PARAPHRASE_SHIFTED]
    if len(kept) < len(rows):
        label = f"hit@{k} · 뜻 달라짐 {len(rows) - len(kept)}개 뺌"
        table_rows.append(_sensitivity_row(label, kept, k))
    return (
        render_table(
            table_rows,
            [
                "기준",
                "원문",
                "패러프레이즈",
                "차이 (%p)",
                "둘 다",
                "원문만",
                "패러프레이즈만",
                "둘 다 실패",
                "McNemar p",
            ],
        )
        + f"\n- 표본: KorQuAD dev eval 몫에서 검색 결과로 거르지 않고 뽑은 {len(rows)}개 (정답 문장이 지금 문서에 남은 "
        "것만 — 코퍼스 내용으로 거른 것이라 선택 편향이 없다)\n"
        f"- 패러프레이즈 v1 은 Claude 로 만들었다. 사람 확인(이 표본 몫, seed 0): {PARAPHRASE_REVIEW}. "
        "뜻이 달라진 문항도 정답은 그대로라 빼지 않고 전부로 쟀고, 뺀 값을 한 줄 더 냈다\n"
        f"- 검색: `{meta['collection']}` Hybrid(후보 {meta['candidates']}) + 리랭커 `{meta['reranker_model']}` 상위 {k} "
        "— `/ask` 와 같은 설정. hit = 상위에 정답 근거 청크(정답 문서이고 정답 문자열을 담음)가 있음\n"
        "- 원문 hit 가 실제 검색 성공률이고, 패러프레이즈와의 차이가 질문 · 지문의 낱말 겹침으로 부풀려진 몫이다. "
        "McNemar p 는 엇갈린 짝(한쪽만 성공)만 본 정확 검정\n"
    )


def _sensitivity_row(label: str, rows: Sequence[Mapping], cutoff: int) -> dict:
    table = paired_hits(_hit_pairs(rows, cutoff))
    return {
        "기준": label,
        "원문": _percent(table.original_rate),
        "패러프레이즈": _percent(table.paraphrase_rate),
        "차이 (%p)": _points(table.original_rate, table.paraphrase_rate),
        "둘 다": table.both,
        "원문만": table.original_only,
        "패러프레이즈만": table.paraphrase_only,
        "둘 다 실패": table.neither,
        "McNemar p": f"{table.mcnemar_p():.3g}",
    }


def _fmt(value: Share) -> str:
    if value.rate is None:
        return "0/0"
    return f"{value.count}/{value.total} ({value.rate:.1%})"


def _seconds(value: Latency, q: int) -> str:
    milliseconds = value.percentiles.get(q)
    return "" if milliseconds is None else f"{milliseconds / 1000:.1f}"


def _percent(rate: float | None) -> str:
    return "" if rate is None else f"{rate:.1%}"


def _points(first: float | None, second: float | None) -> str:
    if first is None or second is None:
        return ""
    return f"{(first - second) * 100:+.1f}"


def _document_texts(titles: set[str]) -> dict[str, str]:
    """정규화한 제목 → 본문. 필요한 제목만 담는다."""
    texts: dict[str, str] = {}
    for path in sorted(PARSED.glob("docs-*.parquet")):
        table = pq.read_table(path, columns=["title", "text"]).to_pydict()
        for title, text in zip(table["title"], table["text"], strict=True):
            normalized = normalize_title(title)
            if normalized in titles:
                texts[normalized] = text
    return texts


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: Iterable[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
