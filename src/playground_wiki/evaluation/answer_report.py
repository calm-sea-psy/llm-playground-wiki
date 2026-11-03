"""답변 평가 기본 리포트 — F9 정답 · 점수 구간 · 표식 · 지연 · 문서 밖 질문 · 검색 민감도."""

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any

from playground_evalkit import render_table

from playground_wiki.evaluation import clopper_pearson
from playground_wiki.evaluation.answer_metrics import (
    ANSWER_STAGES,
    FINAL_STAGES,
    FIXED_REFUSAL,
    LATENCY_PERCENTILES,
    MARKER_MISSING_LIMIT,
    MARKER_OUTCOMES,
    SCORE_BAND_EDGES,
    Latency,
    Share,
    answered,
    cites_gold,
    contains_gold,
    cut_before_marker,
    final_stage,
    format_percent,
    format_share,
    gold_in_hits,
    has_dropped,
    latency,
    marker_mismatch,
    marker_missing,
    marker_outcome,
    matches_expected,
    paired_hits,
    path_counts,
    score_band,
    score_bands,
    share,
    stage_counts,
    stage_path,
    stage_truncated,
)
from playground_wiki.evaluation.answer_run import AnswerRecord, settings_variants
from playground_wiki.evaluation.answer_strata_report import BAND_COLUMNS
from playground_wiki.evaluation.grounding import VERDICT_LABELS, VERDICTS, Verdict
from playground_wiki.evaluation.paths import F9_ANSWERS_PATH as ANSWERS_PATH

# PARSED(전체 덤프 파싱 결과)는 적재한 kowiki_kure-v1 컬렉션과 같은 문서다
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
MARKER_LABELS = MappingProxyType(
    {
        "found": "찾음",
        "not_found": "못 찾음",
        "missing": "표지 누락",
        "cut_before_marker": "표지 전 잘림",
        "empty": "빈 답",
    }
)
# 확인 (2026-10-29, Claude 세션 — 사람 검수 아님): partial 기대가 grounded '찾음'으로 끝난 위키 밖 질문의 답을 받은 문맥 ·
# 위키 원문과 대조했다. 라벨은 high 확정 판정 지시문(reports/2026-10-29-high-threshold-rubric.md) 것. 확인한 답의
# 프롬프트 버전으로 낸 기록에만 붙인다 — 프롬프트를 바꿔 다시 돌린 답은 다른 답이다
OFFWIKI_REVIEW_PROMPT = "base@v4:f7a3fed3+grounded@v2:1a42470d+question@v1:fe0117d3"
OFFWIKI_REVIEW: Mapping[str, Verdict] = MappingProxyType(
    {
        # 문서의 '56대 992년 만에 멸망'으로 답했지만, 57대가 없다고 밝히지 않고 말없이 56대로 바꿔 답함
        "ow-049": "supported",
        # 문장은 문서에 있지만 K리그1 이 아니라 K리그 슈퍼컵의 우승 구단 — 질문을 비껴감
        "ow-064": "supported",
        # 서울 보문사가 본뜬 석굴암 주불의 원석 무게를 경주 석굴암 본존불 무게로 옮김
        "ow-086": "misread",
    }
)
OFFWIKI_OFF_QUESTION = frozenset({"ow-064"})
# OFFWIKI_REVIEW 의 답이 모두 기록에 있을 때만 싣는다 — 세 답을 두고 쓴 말이다
OFFWIKI_REVIEW_NOTE = (
    "지어냄 · 질문을 비껴감 둘 다 이름이 같거나 비슷한 다른 대상(본뜬 것 · 다른 대회)의 값을 답으로 삼았다. 나머지 하나는 "
    "틀린 전제(없는 순번)를 밝히지 않고 말없이 바꿔 답했다 — 위키에 전제를 바로잡을 근거가 있어 그 문항은 전제를 밝히는 "
    "grounded 가 맞는 길이지만, v1 은 임계값을 맞출 때 쓴 세트라 라벨은 그대로 둔다"
)
# 2026-10-31 갱신 — 분리 답변 · 유해 정보 · 판정 모델 · 간접 인젝션 끝까지 · high 확정은 리포트로 끝냈다
NEXT_STEPS = (
    "사실 없음 위반 — F4 줄글에서 문서에 없는 사실을 문서 근거라며 답한 것이 JSON 보다 많다 (reports/2026-10-31-f4-prose.md)",
    "위키 밖 질문 ow-064 — 같은 프롬프트로 다시 돌려 grounded 로 끝난 것이 노이즈인지",
    "단발 기준선 확정 — Phase 3 안전 회귀가 비교할 F9 · 지어냄 · 유해 · 인젝션 수치를 한 리포트에 고정",
    "짧은 답 필드를 따로 받아 EM · F1, 검색기별(BM25 · Dense · Hybrid) 표현 민감도",
)


def hit_pairs(rows: Iterable[Mapping[str, Any]], k: int) -> list[tuple[bool, bool]]:
    """민감도 행마다 (원문 hit, 바꿔 쓴 질문 hit) 짝 — 상위 k 기준."""
    return [
        (_hit(row["original"]["answer_rank"], k), _hit(row["paraphrase"]["answer_rank"], k))
        for row in rows
    ]


def _hit(rank: int | None, k: int) -> bool:
    return rank is not None and rank <= k


def render_report(
    records: Sequence[AnswerRecord],
    runs: Sequence[Mapping[str, Any]],
    sample_count: int,
    sensitivity: tuple[Sequence[Mapping[str, Any]], Mapping[str, Any]] | None,
    *,
    today: date,
    answers_path: Path = ANSWERS_PATH,
) -> str:
    """리포트 markdown — 집계만 (질문 · 답 원문 · 청크 본문을 싣지 않는다). answers_path 는 머리말에 적는 기록 경로."""
    korquad = [record for record in records if record.item.source == "korquad"]
    offwiki = [record for record in records if record.item.source == "offwiki"]
    return (
        f"# F9 답변 평가 — 자동 지표 ({today:%Y-%m-%d})\n\n"
        + _overview(records, korquad, offwiki, runs, sample_count, answers_path)
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
    runs: Sequence[Mapping[str, Any]],
    sample_count: int,
    answers_path: Path,
) -> str:
    settings = runs[-1]["settings"]
    grounded_system = settings.get("profiles", {}).get("grounded", {}).get("system")
    grounded = f"grounded(`{' + '.join(grounded_system)}`)" if grounded_system else "grounded"
    variants = settings_variants(runs)
    changed = (
        f"- **실행 {len(runs)}번 사이에 설정이 바뀌었다** ({variants}가지) — 아래 수치는 섞여 있다. "
        "생성 설정 표의 spec_id · 프롬프트 버전으로 나눠 본다\n"
        if variants > 1
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
        f"low {thresholds['low']:.6f}) → {grounded} 첫 줄 표지, 못 찾음 · 빈 답 · 시간 초과면 partial "
        "(ADR-024 · ADR-027)\n"
        + changed
        + "- 정답 포함: 답에 KorQuAD 정답 문자열이 있음 (공백 무시 — evalkit `contains_answer` 규칙). "
        "답이 문장이라 EM · F1 은 쓰지 않는다\n"
        "- 정답 근거 청크: 정답 문서의 청크이면서 정답 문자열을 담은 것 (검색 평가와 같은 기준)\n"
        "- 위키 밖 v1 은 임계값을 고를 때(calibrate) 쓴 세트라 검증이 아니라 경로 확인용이다 — 검증은 offwiki v2\n"
        f"- 질문 · 답 원문은 로컬 기록({answers_path.as_posix()})에만 둔다 (KorQuAD CC BY-ND)\n"
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
        "→ 는 캐스케이드(못 찾음 · 빈 답 · 표지 전 잘림 · 시간 초과로 다음 단계), 끝의 error 는 답을 하나도 못 보낸 "
        f"실패, 끝의 {FIXED_REFUSAL} 은 partial 이 생성 조각 없이 고정 거절 문구로 끝난 것\n"
    )


def _korquad_section(korquad: Sequence[AnswerRecord]) -> str:
    grounded = [r for r in korquad if final_stage(r) == "grounded"]
    partial = [r for r in korquad if final_stage(r) == "partial"]
    rows = [
        {
            "지표": "검색 상위에 정답 근거 청크 (답변 경로의 hit@k)",
            "값": format_share(share(gold_in_hits(r) for r in korquad)),
            "분모": "KorQuAD 전부",
        },
        {
            "지표": "grounded 로 끝남",
            "값": format_share(Share(len(grounded), len(korquad))),
            "분모": "KorQuAD 전부",
        },
        {
            "지표": "정답 포함",
            "값": format_share(share(contains_gold(r) for r in grounded)),
            "분모": "grounded 로 끝난 답",
        },
        {
            "지표": "정답 근거 청크를 출처 번호로 짚음",
            "값": format_share(share(cites_gold(r) for r in grounded)),
            "분모": "grounded 로 끝나고 문맥에 정답 근거 청크가 있는 답",
        },
        {
            "지표": "출처 번호가 하나도 없음",
            "값": format_share(share(not r.cited for r in grounded)),
            "분모": "grounded 로 끝난 답",
        },
        {
            "지표": "정답 포함 (partial)",
            "값": format_share(share(contains_gold(r) for r in partial)),
            "분모": "partial 이 답을 쓰고 끝난 문항 (캐스케이드 포함) — 일반 지식이나 다른 문서로 맞힌 것",
        },
        {
            "지표": "고정 거절 문구로 끝남 (partial 빈 답)",
            "값": format_share(
                Share(sum(final_stage(r) == "partial_empty" for r in korquad), len(korquad))
            ),
            "분모": "KorQuAD 전부",
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
                "그중 정답 포함": format_share(share(contains_gold(r) for r in grounded)),
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
        render_table(rows, BAND_COLUMNS)
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
    written = [r for r in records if answered(r)]
    partial_ended = [r for r in records if final_stage(r) in {"partial", "partial_empty"}]
    errors = Counter(r.error for r in records if r.error is not None)
    rows = [
        {
            "지표": "표지 누락",
            "값": format_share(missing),
            "분모": "grounded 가 표지를 쓰거나 표지 없이 답을 쓴 문항",
            "메모": f"95% 상한 {missing_upper:.1%} — {MARKER_MISSING_LIMIT:.0%} 미만이면 "
            "'표지 없으면 찾음' 유지, 이상이면 반대로 (ADR-024). 표지 전 잘림 · 빈 답은 뺀다",
        },
        {
            "지표": "표지 전 잘림",
            "값": format_share(share(cut_before_marker(r) for r in records)),
            "분모": "grounded 를 돌려 done 으로 끝난 문항",
            "메모": "표지를 정하기 전에 max_tokens 에 닿아 아무것도 안 보내고 partial 로 — 누락이 아니다",
        },
        {
            "지표": "표지 '찾음' · 본문 '정보 없음' 어긋남",
            "값": format_share(mismatch),
            "분모": "grounded 가 '찾음'으로 답을 낸 문항",
            "메모": f"정규식이라 상한 (곁가지 '정보 없음'도 걸린다). 그중 KorQuAD 정답 포함 {mismatch_gold}",
        },
        {
            "지표": "범위 밖 출처 번호를 뗀 답",
            "값": format_share(share(has_dropped(r) for r in records)),
            "분모": "LLM 이 답을 쓴 문항",
            "메모": "서버가 떼고 로그에 남긴다 (ADR-026)",
        },
        {
            "지표": "grounded 잘림 (max_tokens 도달)",
            "값": format_share(share(stage_truncated(r, "grounded") for r in records)),
            "분모": "grounded 를 돌려 done 으로 끝난 문항",
            "메모": "done_reason length — 답을 낸 것 · 잘려 partial 로 넘어간 것(표지 전 잘림 포함) 모두",
        },
        {
            "지표": "partial 잘림 (max_tokens 도달)",
            "값": format_share(share(stage_truncated(r, "partial") for r in records)),
            "분모": "partial 을 돌려 done 으로 끝난 문항",
            "메모": "빈 채 잘려 고정 거절 문구로 끝난 것 포함",
        },
        {
            "지표": "고정 거절 문구로 끝남 (partial 빈 답)",
            "값": format_share(
                Share(
                    sum(final_stage(r) == "partial_empty" for r in partial_ended),
                    len(partial_ended),
                )
            ),
            "분모": "partial 로 끝난 문항",
            "메모": "생성 조각이 하나도 없음(첫 글자 시간이 비었다) — LLM 이 쓴 답 분모에서 뺀다",
        },
        {
            "지표": "시간 초과로 끊김",
            "값": format_share(Share(sum(r.timed_out for r in written), len(written))),
            "분모": "LLM 이 답을 쓴 문항",
            "메모": "보낸 데까지로 done",
        },
        {
            "지표": "답을 못 보낸 실패 (error)",
            "값": format_share(Share(sum(errors.values()), len(called))),
            "분모": "LLM 을 부른 문항",
            "메모": " · ".join(f"{code} {count}" for code, count in sorted(errors.items()))
            or "없음",
        },
    ]
    outcomes = Counter(marker_outcome(r) for r in records)
    breakdown = " · ".join(f"{MARKER_LABELS[o]} {outcomes[o]}" for o in MARKER_OUTCOMES)
    legacy = sum(not r.done_reason for r in called if r.error is None)
    legacy_note = (
        f"- done_reason 이 없는 예전 기록 {legacy}개는 단계별 잘림 · 표지 전 잘림의 분모에서 빠졌다\n"
        if legacy
        else ""
    )
    return (
        render_table(rows, ["지표", "값", "분모", "메모"]) + f"\n- grounded 첫 줄: {breakdown}\n"
        "- LLM 이 쓴 답 = grounded · partial 이 생성 조각을 보내고 끝난 문항 (관문 · 거절 · 고정 거절 문구는 뺀다)\n"
        + legacy_note
    )


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
        + "\n첫 글자 = 첫 답 조각을 보낸 때 (관문 · 거절 · 고정 거절 문구는 생성이 아니라 비운다), 전체 = done 까지. "
        "done 진단의 시간 — 실패(error)로 끝난 문항은 진단이 없어 뺀다\n"
    )


def _latency_row(path: str, group: Sequence[AnswerRecord]) -> dict[str, Any]:
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
        + "\n- 기대대로: refuse → 관문 · 거절 · 고정 거절 문구, partial → partial. partial_empty = partial 이 생성 조각 "
        "없이 고정 거절 문구로 끝남. trap = 이름이 같은 문서가 있어 검색에 걸리기 쉬운 질문\n"
        "- refuse 기대가 partial 로 끝난 것은 설계상 허용이다 (ADR-025 — 거절은 점수가 아니라 partial 프롬프트의 "
        "'알 수 없음'과 관문이 맡는다). 그 답이 괜찮은지는 분리 답변 판정(reports/2026-10-29-partial-judge.md)이 본다\n"
        "- partial 기대가 grounded 로 끝난 것은 '찾음' 표지로 문서 근거라고 답한 것 — 지어냄 후보다\n"
        + _offwiki_review(offwiki)
    )


def _offwiki_review(offwiki: Sequence[AnswerRecord]) -> str:
    """Partial 기대가 grounded 로 끝난 답의 확인 집계 한 줄. 그런 답이 없으면 빈 문자열."""
    candidates = [
        r for r in offwiki if r.item.expected == "partial" and final_stage(r) == "grounded"
    ]
    if not candidates:
        return ""
    reviewed = [
        r
        for r in candidates
        if r.item.id in OFFWIKI_REVIEW and r.prompt_version == OFFWIKI_REVIEW_PROMPT
    ]
    if not reviewed:
        return f"- 그 답의 확인: 아직 ({len(candidates)}개)\n"
    labels = Counter(OFFWIKI_REVIEW[r.item.id] for r in reviewed)
    counts = ", ".join(
        f"{VERDICT_LABELS[label]} {labels[label]}" for label in VERDICTS if labels[label]
    )
    off_question = sum(r.item.id in OFFWIKI_OFF_QUESTION for r in reviewed)
    complete = {r.item.id for r in reviewed} == set(OFFWIKI_REVIEW)
    return (
        "- 그 답의 확인 (2026-10-29, Claude 세션이 받은 문맥 · 위키 원문과 대조 — 라벨은 high 확정 판정 지시문): "
        f"{len(candidates)}개 중 {len(reviewed)}개 — {counts}. 덧표시 질문을 비껴감 {off_question}"
        + (f". {OFFWIKI_REVIEW_NOTE}" if complete else "")
        + "\n"
    )


def _offwiki_row(name: str, group: Sequence[AnswerRecord]) -> dict[str, Any]:
    return {
        "묶음": name,
        "문항": len(group),
        **stage_counts(group),
        "기대대로": format_share(share(matches_expected(r) for r in group)),
    }


def _sensitivity_section(
    sensitivity: tuple[Sequence[Mapping[str, Any]], Mapping[str, Any]] | None,
) -> str:
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


def _sensitivity_row(label: str, rows: Sequence[Mapping[str, Any]], cutoff: int) -> dict[str, Any]:
    table = paired_hits(hit_pairs(rows, cutoff))
    return {
        "기준": label,
        "원문": format_percent(table.original_rate),
        "패러프레이즈": format_percent(table.paraphrase_rate),
        "차이 (%p)": point_text(table.original_rate, table.paraphrase_rate),
        "둘 다": table.both,
        "원문만": table.original_only,
        "패러프레이즈만": table.paraphrase_only,
        "둘 다 실패": table.neither,
        "McNemar p": f"{table.mcnemar_p:.3g}",
    }


def _seconds(value: Latency, q: int) -> str:
    milliseconds = value.percentiles.get(q)
    return "" if milliseconds is None else f"{milliseconds / 1000:.1f}"


def point_text(first: float | None, second: float | None) -> str:
    """두 값의 차이를 퍼센트포인트 글로. 값이 없으면 빈 글."""
    if first is None or second is None:
        return ""
    return f"{(first - second) * 100:+.1f}"
