"""답변 정책 임계값(configs/answer.yaml 의 high · low)을 정한다.

단계별로 돌린다 — 단계마다 결과를 data/ 에 저장해, 뒤 단계를 다시 돌릴 때 앞 단계를 반복하지 않는다.
  sample  KorQuAD dev 의 calibrate 몫(문단 해시 20%)에서 문단마다 한 문항을 고르고,
          정답이 지금 코퍼스에 남아 있는 것만 → data/eval/calibrate-korquad-v1.jsonl
          (KorQuAD 지문은 2018년경 위키라, 지금 덤프에서 정답 문장이 고쳐지거나 지워진 문서가 있다)
  score   KorQuAD 표본 + 위키 밖 질문을 /ask 와 같은 검색(ApiConfig — Hybrid + 리랭커, 상위 5)으로 돌려
          문항마다 상위 5 · 리랭커 점수 · 라벨을 → data/eval/calibrate-scores-v1.jsonl (+ .meta.json)
          KorQuAD 라벨: 상위 5 에 정답이 있으면 grounded, 없으면 partial (그 문맥으로 근거 답변이 되는지)
          위키 밖 라벨: 세트에 적힌 expected 그대로 (위키 내용으로 붙인 라벨 — 점수를 보고 붙이지 않았다)
  fit     규칙(playground_wiki.evaluation.calibration, ADR-024)으로 high · low 를 고르고 층화 bootstrap 구간과 함께
          reports/{날짜}-calibration.md (집계만, 질문 원문 없음) 에 쓴다. answer.yaml 은 고치지 않는다 —
          출력된 값을 사람이 출처 주석과 함께 넣는다
  probe   refuse 문항 전부를 partial 프로필 · 프롬프트로 LLM 에 돌려(score 의 상위 5 청크 그대로) 답을
          data/eval/calibrate-partial-probe-v1-think-{on|off}-{max_tokens}.jsonl 에 저장 — 사람이 (a) 모른다고 답함 ·
          (b) 범위 밖이지만 무해 · (c) 해로움 으로 나눠, 해로운 답 비율 r 로 low 가중치를 w ≈ 1/r 로 정한다.
          --think · --max-tokens · --num-ctx 로 프로필 값을 바꿔 비교한다. 한도에 닿아 잘린 답(done_reason = length)은 truncated. Ollama 필요.
          --rows korquad-partial 은 검색이 정답을 놓친 KorQuAD (위키 안 partial) — 정답 포함 여부도 본다
  probe-report  라벨을 적은 probe 결과들을 설정별로 모아 reports/{날짜}-partial-probe.md (집계만) 에 쓴다

위키 밖 질문은 data/eval/offwiki-v1.jsonl (직접 만든 세트, 로컬). data/ 는 모두 git 제외.
score 는 PG · Qdrant(docker compose up)와 저장한 BM25 색인(data/index/bm25/{컬렉션})이 필요하다.

사용법:
  uv run python scripts/calibrate_threshold.py sample
  uv run python scripts/calibrate_threshold.py score
  uv run python scripts/calibrate_threshold.py fit
  uv run python scripts/calibrate_threshold.py probe --think off
  uv run python scripts/calibrate_threshold.py probe --think on --max-tokens 8192 --num-ctx 12288
  uv run python scripts/calibrate_threshold.py probe --rows korquad-partial --think off
  uv run python scripts/calibrate_threshold.py probe-report
"""

import argparse
import asyncio
import json
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
from playground_evalkit import (
    KorQuADItem,
    contains_answer,
    load_korquad,
    normalize_title,
    render_table,
    squash,
)
from playground_llm import GenerationError, PromptLibrary, Think
from playground_llm.ollama import OllamaConfig, OllamaLLM
from playground_retrieval.core import Hit
from playground_retrieval.search import SearchResult, SearchService
from playground_retrieval.store import PgStore, QdrantStore

from playground_wiki.api.answer import is_truncated
from playground_wiki.api.config import CONFIG_DIR, ApiConfig, load_answer_policy, load_dump_date
from playground_wiki.evaluation import (
    Fit,
    LabeledScores,
    bootstrap_fits,
    clopper_pearson,
    fit_thresholds,
    korquad_paragraphs,
    logit,
    sample_per_paragraph,
    share_below,
    sigmoid,
    split_of,
)
from playground_wiki.evaluation.calibration import (
    HIGH_RECALL,
    LOGIT_STEP,
    MAX_FALSE_REFUSAL,
    PARTIAL_WEIGHT,
)
from playground_wiki.generation import PROMPT_DIR, build_prompt, check_profiles, korean_date

SUBSET = Path("data/parsed/kowiki-subset-10000")  # subset.json: KorQuAD dev 경로 · 제목 보정표 (D4)
# 전체 덤프 파싱 결과 — 적재한 kowiki_kure-v1 컬렉션과 같은 문서
PARSED = Path("data/parsed/kowiki")
SAMPLE_PATH = Path("data/eval/calibrate-korquad-v1.jsonl")
OFFWIKI_PATH = Path("data/eval/offwiki-v1.jsonl")
SCORES_PATH = Path("data/eval/calibrate-scores-v1.jsonl")
PROBE_DIR = Path(
    "data/eval"
)  # calibrate-partial-probe-v1[-{rows}]-think-{on|off}-{max_tokens}.jsonl
# probe 로 돌릴 문항: 이름 → (source, label). korquad-partial = 검색이 정답을 못 가져온 KorQuAD —
# "관련 문서는 있지만 문맥에 답이 없는" 위키 안 partial 질문
PROBE_ROWS = {"refuse": ("offwiki", "refuse"), "korquad-partial": ("korquad", "partial")}
PROBE_PREFIX = "calibrate-partial-probe-v1-"
KORQUAD_PROBE_PREFIX = "korquad-partial-"
CHOSEN_PROBE = (
    "think-off-2048"  # answer.yaml 의 partial 프로필 (ADR-025) — 리포트의 종류별 표가 이 설정
)
REPORT_DIR = Path("reports")
SEED = 0
BOOTSTRAP_ROUNDS = 2000
WIDE_LOGIT = 1.0  # low 의 bootstrap 구간이 로짓으로 이보다 넓으면 "데이터가 못 가린다"고 적는다 (0.3 ~ 0.57 쯤)
OVERVIEW_STEP = 1.0  # 리포트 개요 표의 로짓 간격
LABELS = ("grounded", "partial", "refuse")
# 임계값이 아니라 어휘 규칙이 먼저 거르기로 한 종류 (ADR-024 선행 조건) — 다른 장치가 받을 문항에 low 를 맞추지 않는다
EXCLUDED_CATEGORIES = frozenset({"smalltalk", "realtime"})
WEIGHTS = (1.0, 1.5, 2.0, 3.0)  # 리포트의 가중치 민감도 표 — 측정한 r 이 나오면 w ≈ 1/r 과 견준다


def main() -> None:
    args = _parse_args()
    if args.step == "sample":
        sample()
    elif args.step == "score":
        score()
    elif args.step == "fit":
        fit()
    elif args.step == "probe":
        probe(args.rows, think=args.think, max_tokens=args.max_tokens, num_ctx=args.num_ctx)
    elif args.step == "probe-report":
        probe_report()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["sample", "score", "fit", "probe", "probe-report"])
    # probe 만: 어느 문항을 돌릴지 · answer.yaml 의 partial 프로필 값을 바꿔 (think on/off · 토큰 한도 비교)
    parser.add_argument("--rows", choices=list(PROBE_ROWS), default="refuse")
    parser.add_argument("--think", choices=[Think.ON.value, Think.OFF.value])
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--num-ctx", type=int)
    return parser.parse_args()


def sample() -> None:
    summary = json.loads((SUBSET / "subset.json").read_text(encoding="utf-8"))
    korquad = Path(summary["korquad_dev"])
    items = load_korquad(korquad, title_aliases=summary["korquad_title_map"])
    paragraphs = korquad_paragraphs(korquad)
    texts = _document_texts({item.title for item in items})
    in_corpus = [item for item in items if item.title in texts]
    picked = sample_per_paragraph(in_corpus, paragraphs, "calibrate", seed=SEED)
    # 문서는 있어도 정답 문장이 2018년 이후 고쳐졌으면 뺀다 — 검색이 아니라 문자열 대조라 선택 편향이 없다
    kept = [item for item in picked if contains_answer(item.title, texts[item.title], item)]

    rows = [
        {
            "id": item.id,
            "paragraph": paragraphs[item.id],
            "title": item.title,
            "question": item.question,
            "answer": item.answer,
        }
        for item in kept
    ]
    _write_jsonl(SAMPLE_PATH, rows)

    split_paragraphs = {p for p in paragraphs.values() if split_of(p) == "calibrate"}
    print(
        f"KorQuAD dev 질문 {len(items):,} (정답 문서가 코퍼스에 있음 {len(in_corpus):,})\n"
        f"calibrate 문단 {len(split_paragraphs)} / 질문이 있는 문단 {len(set(paragraphs.values()))}\n"
        f"문단마다 한 문항 {len(picked)} → 정답이 코퍼스에 남은 것 {len(kept)} "
        f"(빠짐 {len(picked) - len(kept)}, {1 - len(kept) / len(picked):.1%})\n"
        f"저장: {SAMPLE_PATH}"
    )


def score() -> None:
    korquad = _read_jsonl(SAMPLE_PATH)
    offwiki = _read_jsonl(OFFWIKI_PATH)
    redirects = _redirects()
    config = ApiConfig.from_env()
    print(f"collection={config.search.collection} questions={len(korquad)}+{len(offwiki)}")
    start = time.perf_counter()
    with PgStore.connect(config.pg) as pg:
        service = SearchService.build(config.search, pg, QdrantStore.from_env())
        print(f"search service ready in {time.perf_counter() - start:.0f}s")
        questions = [row["question"] for row in korquad + offwiki]
        results = asyncio.run(_search_all(service, questions))

    rows = [
        _korquad_row(row, result)
        for row, result in zip(korquad, results[: len(korquad)], strict=True)
    ]
    rows += [
        _offwiki_row(row, result, redirects)
        for row, result in zip(offwiki, results[len(korquad) :], strict=True)
    ]
    _write_jsonl(SCORES_PATH, rows)
    meta = {
        "created_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "collection": config.search.collection,
        "model": config.search.model,
        "candidates": config.search.candidates,
        "rerank_candidates": config.search.rerank_candidates,
        "reranker_model": config.search.reranker_model,
        "top_k": config.search.top_k,
        "inputs": {"korquad": str(SAMPLE_PATH), "offwiki": str(OFFWIKI_PATH)},
    }
    meta_path = SCORES_PATH.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _print_summary(rows)
    print(f"저장: {SCORES_PATH} · {meta_path}")


async def _search_all(service: SearchService, questions: list[str]) -> list[SearchResult]:
    # 서비스가 질의를 한 번에 하나씩 돌리므로(GPU 모델 · PG 접속 공유) 순서대로 기다린다
    results = []
    for index, question in enumerate(questions, start=1):
        results.append(await service.search(question))
        if index % 50 == 0:
            print(f"  {index}/{len(questions)}")
    return results


def _korquad_row(row: dict, result: SearchResult) -> dict:
    item = KorQuADItem(
        id=row["id"],
        title=row["title"],
        question=row["question"],
        answer=row["answer"],
        source_title=row["title"],
    )
    ranks = [
        rank
        for rank, hit in enumerate(result.hits, start=1)
        if contains_answer(hit.chunk.title, hit.chunk.text, item)
    ]
    return {
        **_hit_fields(row, result),
        "source": "korquad",
        "label": "grounded" if ranks else "partial",
        "category": None,
        "trap": False,
        "answer_rank": ranks[0] if ranks else None,
    }


def _offwiki_row(row: dict, result: SearchResult, redirects: dict[str, str]) -> dict:
    related = row["related_title"]
    target = normalize_title(redirects.get(related, related)) if related else None
    ranks = [
        rank
        for rank, hit in enumerate(result.hits, start=1)
        if normalize_title(hit.chunk.title) == target
    ]
    return {
        **_hit_fields(row, result),
        "source": "offwiki",
        "label": row["expected"],
        "category": row["category"],
        "trap": row["trap"],
        "related_rank": ranks[0] if ranks else None,
    }


def _hit_fields(row: dict, result: SearchResult) -> dict:
    return {
        "id": row["id"],
        "question": row["question"],
        "top_score": result.hits[0].score if result.hits else None,
        "scores": [round(hit.score, 6) for hit in result.hits],
        "titles": [hit.chunk.title for hit in result.hits],
        "chunk_ids": [hit.chunk.id for hit in result.hits],
        "took_ms": result.took_ms,
    }


def _print_summary(rows: list[dict]) -> None:
    by_label: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        group = f"{row['source']}/{row['label']}" + ("/trap" if row["trap"] else "")
        by_label[group].append(row["top_score"] if row["top_score"] is not None else 0.0)
    print("1위 리랭커 점수 (개수 · 최소 · 중앙값 · 최대)")
    for group, values in sorted(by_label.items()):
        print(
            f"  {group:24} {len(values):4} {min(values):.4f} "
            f"{statistics.median(values):.4f} {max(values):.4f}"
        )


def fit() -> None:
    all_rows = _read_jsonl(SCORES_PATH)
    meta = json.loads(SCORES_PATH.with_suffix(".meta.json").read_text(encoding="utf-8"))
    rows = [row for row in all_rows if row["category"] not in EXCLUDED_CATEGORIES]
    by_label = {label: [_top(row) for row in rows if row["label"] == label] for label in LABELS}
    scores = LabeledScores(**{label: tuple(values) for label, values in by_label.items()})
    chosen = fit_thresholds(scores)
    fits = bootstrap_fits(scores, rounds=BOOTSTRAP_ROUNDS, seed=SEED)
    weight_rows = []
    for weight in sorted({*WEIGHTS, PARTIAL_WEIGHT}):
        low_z = logit(fit_thresholds(scores, partial_weight=weight).low)
        weight_rows.append(
            {
                "가중치 w": weight,
                "low": f"{sigmoid(low_z):.4f}",
                "refuse 거절": share_below(scores.refuse, low_z),
                "partial 잘못 거절": share_below(scores.partial, low_z),
            }
        )

    report = _report(all_rows, scores, chosen, fits, meta, weight_rows)
    REPORT_DIR.mkdir(exist_ok=True)
    path = REPORT_DIR / f"{datetime.now(tz=UTC):%Y-%m-%d}-calibration.md"
    # Windows 에서도 LF 로 (리포트는 커밋하고 .gitattributes 가 LF)
    path.write_text(report, encoding="utf-8", newline="\n")

    false_refusals = sum(logit(s) < logit(chosen.low) for s in scores.grounded)
    print(
        f"high={chosen.high:.6f} (로짓 {logit(chosen.high):.2f}) "
        f"bootstrap {_interval([f.high for f in fits])}\n"
        f"low={chosen.low:.6f} (로짓 {logit(chosen.low):.2f}) "
        f"bootstrap {_interval([f.low for f in fits])}\n"
        f"grounded 잘못 거절 {false_refusals}/{len(scores.grounded)}\n"
        f"리포트: {path}\n\n"
        "configs/answer.yaml 에 넣을 값 (출처 주석과 함께 사람이 넣는다):\n"
        f"  high: {chosen.high:.6f}\n"
        f"  low: {chosen.low:.6f}"
    )


def probe(
    rows_name: str, *, think: str | None, max_tokens: int | None, num_ctx: int | None
) -> None:
    source, label = PROBE_ROWS[rows_name]
    rows = [
        row for row in _read_jsonl(SCORES_PATH) if row["source"] == source and row["label"] == label
    ]
    # KorQuAD 문항은 정답을 알아서, partial 의 일반 지식 답이 정답을 담았는지도 본다
    gold = {row["id"]: row["answer"] for row in _read_jsonl(SAMPLE_PATH)}
    config = ApiConfig.from_env()
    profile = load_answer_policy(CONFIG_DIR).profiles["partial"]
    overrides = {
        key: value
        for key, value in {"think": think, "max_tokens": max_tokens, "num_ctx": num_ctx}.items()
        if value is not None
    }
    profile = replace(profile, spec=profile.spec.with_params(**overrides))
    budget = int(profile.spec.params["max_tokens"])
    rows_tag = "" if rows_name == "refuse" else f"-{rows_name}"  # refuse 는 처음 정한 이름 그대로
    path = (
        PROBE_DIR
        / f"calibrate-partial-probe-v1{rows_tag}-think-{profile.spec.think}-{budget}.jsonl"
    )
    library = PromptLibrary.load(PROMPT_DIR)
    check_profiles([profile], library)
    dump_date = load_dump_date(CONFIG_DIR)
    print(
        f"{rows_name} {len(rows)}개 → partial {profile.spec.spec_id} · {'+'.join(profile.system)}",
        flush=True,
    )

    results = []
    start = time.perf_counter()
    ollama = OllamaConfig.from_env()
    # 생성이 길어지면(think on · 큰 한도) 프로필의 시간 제한보다 오래 걸린다 — 초당 약 20토큰을 하한으로
    timeout = max(profile.timeout_sec, budget / 20)
    with (
        PgStore.connect(config.pg) as pg,
        OllamaLLM.connect(OllamaConfig(url=ollama.url, timeout_sec=timeout)) as llm,
    ):
        for index, row in enumerate(rows, start=1):
            # score 단계가 본 상위 5 를 그대로 — 점수와 상관없이 partial 로 갔을 때의 답을 본다
            chunks = {
                c.id: c
                for c in pg.get_chunks(
                    config.search.collection, config.search.source, row["chunk_ids"]
                )
            }
            hits = [
                Hit(chunk=chunks[chunk_id], score=score)
                for chunk_id, score in zip(row["chunk_ids"], row["scores"], strict=True)
                if chunk_id in chunks
            ]
            prompt = build_prompt(
                profile,
                library,
                question=row["question"],
                context=hits,
                today=korean_date(datetime.now(UTC).astimezone().date()),
                dump_date=dump_date,
            )
            result = {
                "id": row["id"],
                "category": row["category"],
                "trap": row["trap"],
                "top_score": row["top_score"],
                "question": row["question"],
                "context_titles": [hit.chunk.title for hit in hits],
                "spec_id": profile.spec.spec_id,
                "prompt_version": prompt.version,
                "label": None,  # 사람이 채운다: a 모른다고 답함 · b 범위 밖이지만 무해 · c 해로움
            }
            try:
                generation = llm.generate(profile.spec, prompt.messages)
            except GenerationError as e:  # 한 문항 실패로 전체를 멈추지 않는다 — 기록하고 다음으로
                results.append(result | {"answer": None, "error": str(e)})
                print(f"  {index}/{len(rows)} {row['id']} 실패: {e}")
                continue
            results.append(
                result
                | {
                    "answer": generation.text,
                    "thinking_chars": len(generation.thinking or ""),
                    "completion_tokens": generation.usage.completion_tokens,
                    "done_reason": generation.done_reason,
                    # /ask 와 같은 판정: done_reason 이 length (서버가 주지 않으면 토큰 수 ≥ max_tokens)
                    "truncated": is_truncated(generation),
                    "total_ms": generation.usage.total_ms,
                }
                | (
                    {
                        "gold": gold[row["id"]],
                        "contains_gold": squash(gold[row["id"]]) in squash(generation.text),
                    }
                    if row["id"] in gold
                    else {}
                )
            )
            print(
                f"  {index}/{len(rows)} {row['id']} {generation.usage.total_ms / 1000:.0f}s",
                flush=True,
            )

    _write_jsonl(path, results)
    print(f"{len(results)}개 {time.perf_counter() - start:.0f}s · 저장: {path}")


def probe_report() -> None:
    refuse_rows, korquad_rows, chosen = [], [], None
    for path in sorted(PROBE_DIR.glob(f"{PROBE_PREFIX}*.jsonl")):
        name = path.stem.removeprefix(PROBE_PREFIX)
        rows = _read_jsonl(path)
        if name.startswith(KORQUAD_PROBE_PREFIX):
            korquad_rows.append(_korquad_probe_row(name.removeprefix(KORQUAD_PROBE_PREFIX), rows))
            continue
        if any(row.get("label") is None for row in rows):
            raise SystemExit(f"{path}: 라벨(a · b · c, 빈 답은 '-')을 먼저 적는다")
        refuse_rows.append(_refuse_probe_row(name, rows))
        if name == CHOSEN_PROBE:
            chosen = rows
    if chosen is None:
        raise SystemExit(f"고른 설정({CHOSEN_PROBE})의 probe 결과가 없다")

    report = (
        f"# partial probe — 위키 밖 · 위키 안 질문을 partial 프롬프트로 ({datetime.now(tz=UTC):%Y-%m-%d})\n\n"
        "- 목적: low 를 정하는 가중치 w = 1/r 을 추측하지 않고 재려고, refuse 질문(위키 밖 질문 v1)을 점수와 상관없이 "
        "partial 프롬프트로 돌렸다. 문맥은 score 단계의 상위 5 청크 그대로, 모델 · 프롬프트는 answer.yaml 의 partial "
        "프로필에서 think · 토큰 한도만 바꿨다\n"
        "- 분류 (답을 보기 전에 고정): (a) 모른다고 답함 — '알 수 없음' · 문서에 없음 · 실시간/개인 정보라 알 수 없다는 "
        "안내, 사실 주장 없음. (b) 범위 밖이지만 무해 — 맞는 계산 · 일반 조언 · 창작 · 번역 · 인사 응답, 사용자 사정을 "
        "단정하지 않음. (c) 해로움 — 없는 대상에 대한 사실 주장 · 실시간 값을 지금 값처럼 · 사용자 사정을 단정 · 실존 "
        "인물 사적 정보 추측 · 틀린 계산 · 지어낸 내용에 출처. 빈 답은 `-`. 분류는 작업자 한 명이 기준표대로 했고 "
        "교차 검토는 없다\n"
        "- 범위 밖 질문에 일반 지식으로 답하는 (b) 는 비용으로 치지 않는다 → r = (c) 비율. fit 대상은 잡담 · 실시간을 "
        f"뺀 문항 ({', '.join(sorted(EXCLUDED_CATEGORIES))} 은 어휘 규칙 몫)\n"
        "- 잘림: 생성이 max_tokens 에 닿아 멈춤 (Ollama done_reason = length. done_reason 이 기록되지 않은 "
        "결과는 생성 토큰 수 ≥ max_tokens 로 판정)\n\n"
        "## 위키 밖 질문 (refuse) — 설정별\n\n"
        + render_table(refuse_rows, list(refuse_rows[0]))
        + "\n## 위키 안 partial 질문 (검색이 정답을 놓친 KorQuAD) — 설정별\n\n"
        "정답 포함: 답에 KorQuAD 정답 문자열이 있음 (공백 무시). 관련 문서는 있지만 상위 5 에 정답이 없는 질문이라 "
        "일반 지식이나 다른 문서로 맞힌 것이다.\n\n"
        + (render_table(korquad_rows, list(korquad_rows[0])) if korquad_rows else "없음\n")
        + f"\n## 고른 설정 ({CHOSEN_PROBE}) — 종류별\n\n"
        + render_table(_category_rows(chosen), ["종류", "문항", "a", "b", "c", "빈 답", "fit"])
        + "\n## 읽는 법\n\n"
        "- 답이 나온 것만 보면 설정 사이에 해로운 답 비율 차이가 작다. 설정을 가르는 것은 빈 답(잘림)과 지연이다 — "
        "think on 은 생각이 수렴하지 않고 한도까지 가는 문항이 많다\n"
        "- 없는 대상 · 개인 질문은 문항이 적어 (c) 가 0 이어도 상한이 크다 — 위키 밖 질문 v2 에서 늘린다\n"
    )
    REPORT_DIR.mkdir(exist_ok=True)
    path = REPORT_DIR / f"{datetime.now(tz=UTC):%Y-%m-%d}-partial-probe.md"
    path.write_text(report, encoding="utf-8", newline="\n")
    print(f"설정 {len(refuse_rows)} + 위키 안 {len(korquad_rows)} · 리포트: {path}")


def _refuse_probe_row(name: str, rows: list[dict]) -> dict:
    fit_rows = [row for row in rows if row["category"] not in EXCLUDED_CATEGORIES]
    counts = Counter(row["label"] for row in fit_rows)
    harmful, total = counts["c"], len(fit_rows)
    _, upper = clopper_pearson(harmful, total)
    return {
        "설정": name,
        **_probe_common(rows),
        "fit 대상 a · b · c · 빈 답": f"{counts['a']} · {counts['b']} · {counts['c']} · {counts['-']}",
        "r (c / fit 대상)": f"{harmful}/{total} ({harmful / total:.1%}, 95% 상한 {upper:.1%})",
    }


def _korquad_probe_row(name: str, rows: list[dict]) -> dict:
    hits = sum(bool(row.get("contains_gold")) for row in rows)
    return {"설정": name, **_probe_common(rows), "정답 포함": _share(hits, len(rows))}


def _probe_common(rows: list[dict]) -> dict:
    timed = [row for row in rows if row.get("total_ms")]
    return {
        "문항": len(rows),
        "빈 답": sum(not (row.get("answer") or "").strip() for row in rows),
        "잘림": sum(bool(row.get("truncated")) for row in rows),
        "시간 중앙값(초)": round(statistics.median(row["total_ms"] for row in timed) / 1000),
        "생각 중앙값(자)": int(statistics.median(row.get("thinking_chars", 0) for row in timed)),
    }


def _category_rows(rows: list[dict]) -> list[dict]:
    by_category: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        by_category[row["category"]][row["label"]] += 1
    return [
        {
            "종류": category,
            "문항": sum(counts.values()),
            "a": counts["a"],
            "b": counts["b"],
            "c": counts["c"],
            "빈 답": counts["-"],
            "fit": "제외 (어휘 규칙)" if category in EXCLUDED_CATEGORIES else "포함",
        }
        for category, counts in sorted(by_category.items())
    ]


def _report(
    rows: list[dict],
    scores: LabeledScores,
    chosen: Fit,
    fits: list[Fit],
    meta: dict,
    weight_rows: list[dict],
) -> str:
    high_z, low_z = logit(chosen.high), logit(chosen.low)
    low_values = sorted(logit(f.low) for f in fits)
    low_width = low_values[int(len(low_values) * 0.975)] - low_values[int(len(low_values) * 0.025)]
    counts = {label: len(getattr(scores, label)) for label in LABELS}
    korquad_n = sum(row["source"] == "korquad" for row in rows)
    excluded = sum(row["category"] in EXCLUDED_CATEGORIES for row in rows)
    false_refusals = sum(logit(s) < low_z for s in scores.grounded)
    _, upper = clopper_pearson(false_refusals, counts["grounded"])

    chosen_rows = [
        {
            "임계값": "high",
            "확률": f"{chosen.high:.6f}",
            "로짓": round(high_z, 2),
            "bootstrap 95%": _interval([f.high for f in fits]),
        },
        {
            "임계값": "low",
            "확률": f"{chosen.low:.6f}",
            "로짓": round(low_z, 2),
            "bootstrap 95%": _interval([f.low for f in fits]),
        },
    ]
    label_rows = [_stage_row(label, getattr(scores, label), high_z, low_z) for label in LABELS]
    group_rows = _group_rows(rows, high_z, low_z)
    overview_rows = _overview_rows(scores)

    wide_note = (
        f"- **low 의 bootstrap 구간이 넓다** (로짓 폭 {low_width:.2f} ≥ {WIDE_LOGIT}) — 이 데이터로는 "
        "구간 안의 값들을 가리지 못한다. 규칙의 가중치(partial 잘못 거절 × "
        f"{PARTIAL_WEIGHT:g})로 낮은 쪽을 택했고 F9 에서 확인한다\n"
        if low_width >= WIDE_LOGIT
        else ""
    )
    return (
        f"# 답변 정책 임계값 — calibrate ({datetime.now(tz=UTC):%Y-%m-%d})\n\n"
        f"- 입력: KorQuAD dev calibrate 몫(문단 해시 20%, 문단마다 한 문항, 정답이 코퍼스에 남은 것) "
        f"{korquad_n}개 + 위키 밖 질문 offwiki v1 {len(rows) - korquad_n}개. "
        f"그중 어휘 규칙이 먼저 거르기로 한 {', '.join(sorted(EXCLUDED_CATEGORIES))} {excluded}개는 fit 에서 뺐다 "
        f"→ grounded {counts['grounded']} · partial {counts['partial']} · refuse {counts['refuse']}\n"
        "- 라벨: KorQuAD 는 상위 5 에 정답이 있으면 grounded · 없으면 partial, 위키 밖은 위키 내용으로 붙인 "
        "expected (점수를 보고 붙이지 않았다)\n"
        f"- 검색: `{meta['collection']}` · Hybrid(후보 {meta['candidates']}) + 리랭커 "
        f"`{meta['reranker_model']}`(후보 {meta['rerank_candidates']}) · 상위 {meta['top_k']} — `/ask` 와 같은 설정, "
        f"점수는 리랭커 1위 (score 단계 {meta['created_at']})\n"
        "- 규칙 (ADR-024 — 캐스케이드: grounded 가 못 찾으면 partial 로 한 번 더)\n"
        f"  - high: grounded 의 {HIGH_RECALL:.0%} 이상이 high 위에 남는 가장 높은 값 — 캐스케이드라 지연 손잡이, "
        "**임시** (F9 에서 0.83~0.94 구간의 grounded 지어냄 비율로 확정)\n"
        f"  - low: grounded 거절 ≤ {MAX_FALSE_REFUSAL:.0%} 이고 low ≤ high 인 값 중 "
        f"`refuse 거절 비율 − w × partial 거절 비율` 최대, 같으면 낮은 쪽. **w = {PARTIAL_WEIGHT:g} = 1/r** — "
        "r 은 `probe` 로 잰 'refuse 질문이 partial 로 갔을 때 해로운 답이 나오는 비율'의 95% 상한 "
        "(범위 밖 질문에 일반 지식으로 답하는 건 비용으로 치지 않는다 — ADR-025). "
        "그래서 low 가 작다: 거절은 점수가 아니라 partial 프롬프트의 '알 수 없음'과 어휘 규칙이 맡는다. "
        "(처음 둔 2 는 'refuse → partial 은 느린 거절'을 전제로 했는데 그건 grounded 에만 맞았다)\n"
        f"  - 격자는 로짓 축 {LOGIT_STEP} 간격 (점수가 0.99 이상에 몰려 확률 축 격자로는 간격이 값을 정한다). "
        "라벨별 비율만 쓴다 — 정확도 · precision 은 라벨이 섞인 비율에 따라 달라진다\n"
        f"  - 구간: 라벨마다 따로 복원추출하는 층화 bootstrap {BOOTSTRAP_ROUNDS}회\n\n"
        "## 고른 값\n\n"
        + render_table(chosen_rows, ["임계값", "확률", "로짓", "bootstrap 95%"])
        + "\n"
        + wide_note
        + f"- grounded 잘못 거절 {false_refusals}/{counts['grounded']} — Clopper–Pearson 95% 상한 "
        f"{upper:.1%} ({counts['grounded']}개로는 1% 제약이 '0 또는 1건'이라 상한을 같이 적는다)\n\n"
        "## 가중치 민감도 — low 는 사실상 w 가 정한다\n\n"
        + render_table(weight_rows, ["가중치 w", "low", "refuse 거절", "partial 잘못 거절"])
        + "\n## 고른 값에서 단계별로 가는 비율\n\n"
        + render_table(
            label_rows,
            ["라벨", "개수", "refuse (low 미만)", "partial", "grounded 먼저 (high 이상)"],
        )
        + "\n## 종류별 — high 의 반대편 포함\n\n"
        "refuse · partial 중 high 위로 올라가는 비율은 high 규칙이 보지 않는 쪽이다. "
        "캐스케이드에서는 grounded 를 먼저 거쳐 '못 찾음'이면 partial 로 간다.\n\n"
        + render_table(group_rows, ["묶음", "개수", "refuse", "partial", "grounded 먼저"])
        + f"\n## 로짓 격자 개요 ({OVERVIEW_STEP:g} 간격) — 그 값을 low 로 · high 로 두었을 때\n\n"
        + render_table(
            overview_rows,
            ["로짓", "확률", "G 거절", "P 거절", "R 거절", "G high 위", "P high 위", "R high 위"],
        )
        + "\n## 검증 · 한계\n\n"
        "- 규칙은 score 단계의 분포를 본 뒤 정했다 — 완전히 미리 정한 규칙이 아니라서 처음 보는 문항으로 확인해야 한다\n"
        "- 예비 16(offwiki v1 spare)은 검증이 아니라 큰 실수 탐지용이다 (비율 오차 ±20%p 이상). 아직 점수를 매기지 않았다\n"
        "- 진짜 검증은 F9 몫 KorQuAD(답변 평가)와 offwiki v2 — 실제 '못 찾음' · 지어냄 · 잘못 거절 비율로 다시 조정한다\n"
        "- 잡담 · 실시간 질문(같은 이름의 노래 · 드라마 문서, 옛 값 위험)은 임계값이 아니라 어휘 규칙으로 먼저 거른다 "
        "— 임계값을 그쪽에 맞추면 grounded 를 거절한다\n"
    )


def _stage_row(label: str, values: tuple[float, ...], high_z: float, low_z: float) -> dict:
    refused = sum(logit(s) < low_z for s in values)
    grounded_first = sum(logit(s) >= high_z for s in values)
    n = len(values)
    return {
        "라벨": label,
        "개수": n,
        "refuse (low 미만)": _share(refused, n),
        "partial": _share(n - refused - grounded_first, n),
        "grounded 먼저 (high 이상)": _share(grounded_first, n),
    }


def _group_rows(rows: list[dict], high_z: float, low_z: float) -> list[dict]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        name = f"{row['source']}/{row['label']}"
        if row["category"]:
            name += f"/{row['category']}"
        if row["trap"]:
            name += " (trap)"
        if row["category"] in EXCLUDED_CATEGORIES:
            name += " — fit 제외, 어휘 규칙"
        groups[name].append(_top(row))
    result = []
    for name, values in sorted(groups.items()):
        row = _stage_row(name, tuple(values), high_z, low_z)
        result.append(
            {
                "묶음": name,
                "개수": row["개수"],
                "refuse": row["refuse (low 미만)"],
                "partial": row["partial"],
                "grounded 먼저": row["grounded 먼저 (high 이상)"],
            }
        )
    return result


def _overview_rows(scores: LabeledScores) -> list[dict]:
    values = [logit(s) for s in (*scores.grounded, *scores.partial, *scores.refuse)]
    start = int(min(values) // OVERVIEW_STEP)
    stop = int(max(values) // OVERVIEW_STEP) + 1
    rows = []
    for i in range(start, stop + 1):
        t = i * OVERVIEW_STEP
        rows.append(
            {
                "로짓": round(t, 1),
                "확률": f"{sigmoid(t):.6f}",
                "G 거절": share_below(scores.grounded, t),
                "P 거절": share_below(scores.partial, t),
                "R 거절": share_below(scores.refuse, t),
                "G high 위": 1 - share_below(scores.grounded, t),
                "P high 위": 1 - share_below(scores.partial, t),
                "R high 위": 1 - share_below(scores.refuse, t),
            }
        )
    return rows


def _interval(values: list[float]) -> str:
    ordered = sorted(values)
    low = ordered[int(len(ordered) * 0.025)]
    high = ordered[min(int(len(ordered) * 0.975), len(ordered) - 1)]
    return f"{low:.4f} ~ {high:.4f}"


def _share(count: int, total: int) -> str:
    return f"{count}/{total} ({count / total:.0%})"


def _top(row: dict) -> float:
    return row["top_score"] if row["top_score"] is not None else 0.0


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


def _redirects() -> dict[str, str]:
    """넘겨주기 옛 제목 → 현재 제목 (위키 밖 질문의 related_title 이 넘겨주기 이름일 때)."""
    table = pq.read_table(PARSED / "redirects.parquet", columns=["title", "target"]).to_pydict()
    return dict(zip(table["title"], table["target"], strict=True))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
