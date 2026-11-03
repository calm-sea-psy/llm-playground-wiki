"""층화 표본 리포트 — 점수 구간별 지어냄 · 구간 차이 · 검색 교차 표."""

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

from playground_evalkit import render_table

from playground_wiki.evaluation import split_of
from playground_wiki.evaluation.answer_measure import interval_text
from playground_wiki.evaluation.answer_metrics import (
    SCORE_BAND_EDGES,
    Share,
    contains_gold,
    final_stage,
    format_share,
    path_counts,
    rate_with_interval,
    score_band,
    score_bands,
    share,
)
from playground_wiki.evaluation.answer_run import AnswerRecord, settings_variants
from playground_wiki.evaluation.f9_sample import SEED
from playground_wiki.evaluation.grounding import (
    EXCLUDED,
    FABRICATED,
    UNSUPPORTED,
    VERDICT_LABELS,
    VERDICTS,
    GroundingVerdict,
    cluster_bootstrap_difference,
    newcombe_difference,
)
from playground_wiki.evaluation.strata import PoolRow

# 중간 구간 안에서 지어냄이 한쪽 끝에 몰리는지 — high 를 올린다면 어디서부터인지 (세부 구간마다 문항이 적어 참고용)
FINE_BAND_EDGES = (SCORE_BAND_EDGES[0], 0.86, 0.90, SCORE_BAND_EDGES[1])
# 사용자 승인 (2026-10-29) 뒤 답을 보기 전에 고정한 판정 규칙 — 리포트의 판정 방법 절
JUDGE_METHOD = (
    "판정자: Claude 하위 에이전트 (답을 만든 gemma4:12b 와 다른 계열). 판정 입력에는 가린 번호 · 질문 · 모델이 받은 "
    "문서(출처 번호 순) · 답만 실었다 — 1위 점수 · 구간 · KorQuAD 정답 · 문항 id 는 가리고 두 구간을 섞어 무작위 순서로",
    "절차: 답에서 질문에 답하는 핵심 주장(이름 · 수치 · 날짜 · 사건)을 뽑아 문서와 대조한다. 다른 말로 씀 · 표기나 단위만 "
    "바꿈 · 문서 값으로 한 단순 계산 · 일부만 답함 · 줄여 씀은 근거로 인정",
    "라벨: 근거 있음(핵심이 문서에 있다) · 곁가지 근거 없음(핵심은 있고 꼭 필요 없는 덧붙임만 문서에 없다) · "
    "지어냄 · 잘못 읽음(핵심이 문서와 어긋남 — 대상 · 수치 · 시점을 옮기거나 조건 · 부정을 뒤집음) · "
    "지어냄 · 문서에 없음(핵심이 문서 어디에도 없음, 일반 지식으로 맞아도) · 본문 거절. 덧표시로 출처 번호 어긋남 · "
    "질문을 비껴감",
    "검수: 메인 세션이 점수 · 구간을 가린 채 다시 봤다 — 지어냄 · 곁가지 판정 전부, 정답 문자열과 엇갈린 것(정답을 "
    "담았는데 지어냄 / 안 담았는데 근거 있음), 나머지 근거 있음에서 무작위 20개",
    "판정자에게 준 지시문 전문: reports/2026-10-29-high-threshold-rubric.md",
)
# 사용자와 정한 결론 (리포트 결론 절). 정하기 전에는 None
HIGH_DECISION: str | None = (
    "- **`high` 0.824914 유지 — 임시값을 확정값으로** (2026-10-29 사용자 결정, ADR-024)\n"
    "- 근거: 답을 보기 전에 정한 규칙(지어냄 차이의 Newcombe 95% 구간이 0 을 포함하면 유지)대로. 차이 +4.1%p, 구간 "
    "−0.3 ~ +9.6 으로 0 을 겨우 포함하는 경계 결과다 — '차이 없음'이 아니라, 차이가 9.6%p 를 넘지는 않지만 4%p 안팎의 "
    "차이가 있는지는 이 표본으로 가르지 못한다는 뜻이다\n"
    "- 문단 묶음 bootstrap 95% 는 +0.2 ~ +8.6 으로 0 을 뺐다 (2026-10-31 외부 리뷰 · 메인 확인으로 더함) — 같은 문단의 "
    "질문을 묶어 다시 뽑으면 차이가 0 위다. '유지'는 이 값이 아니라 답을 보기 전에 정한 Newcombe 규칙(−0.3 ~ +9.6, "
    "0 포함) 때문이다 — 결과를 본 뒤 규칙을 바꾸지 않았다\n"
    "- 지어냄은 두 구간 모두 잘못 읽음뿐이다 (다른 대상 · 수치 · 시점으로 옮김) — 문서에 없는 일반 지식을 끌어온 답은 "
    "없었다. 중간 구간에서는 첫 줄 표지 '못 찾음'이 이미 더 자주 partial 로 넘긴다 (20/150, ≥ 0.94 는 1/150)\n"
    "- 올렸을 때의 비용: 0.83~0.94 질문이 grounded 를 건너뛰고 partial 로 가, 근거 있는 답(119/128)도 일반 지식이 "
    "섞인 칸으로 나간다. 세부 구간에서 지어냄이 한쪽 끝에 몰리지 않아 0.94 말고 다른 경계 후보도 없었다\n"
    "- 다시 잴 때: grounded 프롬프트 · 답변 모델 · 리랭커가 바뀌거나, 운영 로그의 1위 점수 분포가 KorQuAD 와 크게 "
    "다르면 같은 단계(pool · strata · run --strata · judge-input · strata-report)로 다시 잰다\n"
)
# 리포트 표의 열 — 꼬리표 이름이 들어가는 표는 그 절에서 만든다
VERDICT_COLUMNS = (
    "1위 점수",
    "grounded 답",
    *(VERDICT_LABELS[label] for label in VERDICTS),
    "지어냄 (95% 구간)",
    "곁가지 포함 (95% 구간)",
    "정답 포함",
    "출처 번호 어긋남",
    "질문을 비껴감",
)
BAND_COLUMNS = (
    "1위 점수",
    "문항",
    "grounded 로 끝남",
    "그중 정답 포함",
    "grounded → partial",
    "partial 만",
    "그 밖",
)


def render_strata_report(
    records: Sequence[AnswerRecord],
    runs: Sequence[Mapping[str, Any]],
    pool_rows: Sequence[PoolRow],
    verdicts: Mapping[str, GroundingVerdict],
    *,
    today: date,
) -> str:
    """High 확정 리포트 markdown — 집계만 (질문 · 답 원문 · 청크 본문 · 판정 근거 구절을 싣지 않는다).

    Args:
      records: 층화 표본을 답변 경로로 돌린 기록.
      runs: 그 기록의 meta runs (실행마다 설정).
      pool_rows: 점수 풀 줄 전부.
      verdicts: 문항 id → 판정. grounded 로 끝난 답마다 하나.
      today: 리포트 날짜.
    """
    missing = [
        r.item.id for r in records if final_stage(r) == "grounded" and r.item.id not in verdicts
    ]
    if missing:
        raise ValueError(f"{len(missing)} grounded answers have no verdict, e.g. {missing[:3]}")
    pool_scores = {row.id: row.top_score for row in pool_rows}
    paragraphs = {row.id: row.paragraph for row in pool_rows}
    high = float(runs[-1]["settings"]["thresholds"]["high"])
    return (
        f"# high 확정 — 점수 구간 층화 표본의 grounded 지어냄 비율 ({today:%Y-%m-%d})\n\n"
        + _strata_overview(records, runs, pool_rows)
        + "\n## 점수 풀 — 리랭커 1위 점수 구간\n\n"
        + _pool_section(pool_rows, high=high)
        + "\n## 층화 표본 — 단계 경로\n\n"
        + _strata_path_section(records, pool_scores)
        + "\n## 지어냄 판정 — grounded 로 끝난 답\n\n"
        + _verdict_section(records, verdicts)
        + "\n## 두 구간 비교 (0.83–0.94 − ≥ 0.94)\n\n"
        + _difference_section(records, verdicts, paragraphs)
        + "\n## 지어냄 × 검색 상위 5 정답 근거\n\n"
        + _retrieval_cross_section(records, verdicts, pool_rows)
        + "\n## 판정 방법 · 검수\n\n"
        + judge_method_section(verdicts)
        + "\n## 결론\n\n"
        + (HIGH_DECISION or "사용자와 정한다 — 아직\n")
    )


def _strata_overview(
    records: Sequence[AnswerRecord], runs: Sequence[Mapping[str, Any]], pool_rows: Sequence[PoolRow]
) -> str:
    settings = runs[-1]["settings"]
    variants = settings_variants(runs)
    splits = Counter(split_of(row.paragraph) for row in pool_rows)
    pool_by_id = {row.id: row for row in pool_rows}
    # 점수 풀의 점수로 고른 구간 — 다시 검색한 점수로 나눈 아래 표들과 다를 수 있다
    composition = Counter(
        (
            split_of(pool_by_id[record.item.id].paragraph),
            score_band(pool_by_id[record.item.id].top_score),
        )
        for record in records
    )
    thresholds = settings["thresholds"]
    changed = (
        f"- **실행 {len(runs)}번 사이에 설정이 바뀌었다** ({variants}가지) — 아래 수치는 섞여 있다\n"
        if variants > 1
        else ""
    )
    lower, upper = SCORE_BAND_EDGES
    return (
        "- 질문 (ADR-024): 1위 점수 0.83~0.94 구간에서 grounded 가 지어내는 비율이 ≥ 0.94 구간보다 높은가 — "
        "높으면 `high` 를 0.94 로, 다르지 않으면 유지\n"
        "- 점수 풀: KorQuAD dev "
        + " · ".join(f"{split} 몫 {count}" for split, count in sorted(splits.items()))
        + " — 문단의 질문 전부, 코퍼스 · 정답 문장으로 거르지 않음 (지어냄은 정답이 아니라 문맥과 대조한다). "
        "`/ask` 와 같은 검색에만 돌려 1위 점수를 쟀다 (LLM 없음)\n"
        f"- 층화 표본 {len(records)}문항: {lower:g}~{upper:g} 구간을 eval 몫부터 담고 모자란 만큼 bench 몫에서, 몫마다 같은 "
        f"수를 ≥ {upper:g} 구간에서 seed {SEED} 로 — "
        + " · ".join(
            f"{split} {band} {count}" for (split, band), count in sorted(composition.items())
        )
        + ". id 순으로 돌려 두 구간이 섞여 돈다. 아래 표의 구간은 답변 경로가 다시 검색한 1위 점수(기록의 top_score)로 "
        "나눈다\n"
        f"- 답변 경로: F9 와 같음 — 관문 {'켬' if settings['gate'] else '끔'} → `{settings['collection']}` "
        f"Hybrid(후보 {settings['candidates']}) + 리랭커 `{settings['reranker_model']}`(후보 "
        f"{settings['rerank_candidates']}) 상위 {settings['top_k']} → 답변 정책(high {thresholds['high']:.6f} · "
        f"low {thresholds['low']:.6f}) → grounded@v2 첫 줄 표지, 못 찾음이면 partial "
        f"(`{settings['profiles']['grounded']['spec_id']}`)\n"
        + changed
        + "- 질문 · 답 · 문서 원문과 판정 근거 구절은 로컬(data/eval/f9-strata-*)에만 둔다 (KorQuAD CC BY-ND)\n"
    )


def _pool_section(pool_rows: Sequence[PoolRow], *, high: float) -> str:
    groups: dict[str, list[PoolRow]] = defaultdict(list)
    for row in pool_rows:
        groups[score_band(row.top_score)].append(row)
    rows = [
        {
            "1위 점수": band,
            "질문": len(groups[band]),
            "문단": len({row.paragraph for row in groups[band]}),
            "정답 문서가 코퍼스에": format_share(share(row.in_corpus for row in groups[band])),
            "정답 문장이 남음": format_share(share(row.answer_in_document for row in groups[band])),
            "상위에 정답 근거": format_share(
                share(row.answer_rank is not None for row in groups[band])
            ),
        }
        for band in score_bands()
        if groups.get(band)
    ]
    return render_table(
        rows,
        [
            "1위 점수",
            "질문",
            "문단",
            "정답 문서가 코퍼스에",
            "정답 문장이 남음",
            "상위에 정답 근거",
        ],
    ) + (
        "\n- 상위에 정답 근거 = 검색 상위 5에 정답 문서이면서 정답 문자열을 담은 청크 (검색 평가와 같은 기준). "
        "중간 구간은 검색이 정답을 놓친 문항이 몰려 있어 지어냄 위험이 큰 쪽이 자연스럽게 들어온다\n"
        + _below_band_note(pool_rows, high)
    )


def _below_band_note(pool_rows: Sequence[PoolRow], high: float) -> str:
    """High 와 표본 구간 아래 끝 사이 — 운영에서는 grounded 로 가는데 층화 표본이 재지 않은 점수. 사이가 없으면 빈 문자열."""
    lower = SCORE_BAND_EDGES[0]
    if high >= lower:
        return ""
    between = sum(row.top_score is not None and high <= row.top_score < lower for row in pool_rows)
    return (
        f"- [{high:.6f}, {lower:g}) 은 운영에서는 grounded 로 가지만(high 이상) 층화 표본 구간({lower:g}~{SCORE_BAND_EDGES[1]:g}) "
        f"밖이다 — 점수 풀 {between}문항 (위 표 < {lower:g} 행에 든다). 이 구간의 지어냄은 재지 않았다\n"
    )


def _strata_path_section(
    records: Sequence[AnswerRecord], pool_scores: Mapping[str, float | None]
) -> str:
    groups: dict[str, list[AnswerRecord]] = defaultdict(list)
    for record in records:
        groups[score_band(record.top_score)].append(record)
    rows = [
        {
            "1위 점수": band,
            "문항": len(groups[band]),
            "grounded 로 끝남": sum(final_stage(r) == "grounded" for r in groups[band]),
            "grounded → partial": sum(
                r.stage_path == ("grounded", "partial") for r in groups[band]
            ),
            "그 밖": " · ".join(
                f"{path} {count}"
                for path, count in path_counts(groups[band]).items()
                if path not in {"grounded", "grounded → partial"}
            ),
        }
        for band in score_bands()
        if groups.get(band)
    ]
    moved = sum(
        score_band(record.top_score) != score_band(pool_scores.get(record.item.id))
        for record in records
    )
    return render_table(
        rows, ["1위 점수", "문항", "grounded 로 끝남", "grounded → partial", "그 밖"]
    ) + (
        f"\n- 점수 풀과 답변 경로의 검색은 따로 돌았다 — 다시 검색한 1위 점수가 다른 구간에 든 문항 {moved}개\n"
        "- grounded → partial 은 첫 줄 표지 '못 찾음'으로 캐스케이드 — 지어냄 분모에서 빠진다 (답을 내지 않았다)\n"
    )


def _verdict_section(
    records: Sequence[AnswerRecord], verdicts: Mapping[str, GroundingVerdict]
) -> str:
    groups = _judged_by_band(records, verdicts)
    rows = []
    for band in score_bands():
        judged = groups.get(band)
        if not judged:
            continue
        labels = Counter(verdict.verdict for _, verdict in judged)
        counted = [verdict for _, verdict in judged if verdict.verdict not in EXCLUDED]
        rows.append(
            {
                "1위 점수": band,
                "grounded 답": len(judged),
                **{VERDICT_LABELS[label]: labels[label] for label in VERDICTS},
                "지어냄 (95% 구간)": rate_with_interval(
                    sum(v.verdict in FABRICATED for v in counted), len(counted)
                ),
                "곁가지 포함 (95% 구간)": rate_with_interval(
                    sum(v.verdict in UNSUPPORTED for v in counted), len(counted)
                ),
                "정답 포함": format_share(share(contains_gold(record) for record, _ in judged)),
                "출처 번호 어긋남": sum(verdict.wrong_citation for _, verdict in judged),
                "질문을 비껴감": sum(verdict.off_question for _, verdict in judged),
            }
        )
    crossing: Counter[tuple[bool, bool]] = Counter()
    for judged in groups.values():
        for record, verdict in judged:
            if verdict.verdict not in EXCLUDED:
                crossing[(bool(contains_gold(record)), verdict.verdict in FABRICATED)] += 1
    return render_table(rows, VERDICT_COLUMNS) + (
        f"\n- 지어냄 = ({VERDICT_LABELS['misread']} + {VERDICT_LABELS['not_in_context']}) ÷ "
        f"(grounded 답 − {VERDICT_LABELS['body_refusal']}). 곁가지 포함은 분자에 "
        f"{VERDICT_LABELS['extra_unsupported']}까지 더한 것. 95% 구간은 Wilson\n"
        f"- {VERDICT_LABELS['body_refusal']} = 표지는 '찾음'인데 본문이 문서에 답이 없다고 했다 — 답을 내지 않아 분모에서 뺀다\n"
        "- 정답 포함(KorQuAD 정답 문자열)과 판정의 교차 — 정답 포함 · 지어냄 "
        f"{crossing[(True, True)]}, 정답 포함 · 지어냄 아님 {crossing[(True, False)]}, "
        f"정답 미포함 · 지어냄 {crossing[(False, True)]}, 정답 미포함 · 지어냄 아님 {crossing[(False, False)]}. "
        "정답 미포함이 곧 지어냄은 아니다 (다른 말로 · 일부만 · 정답 문장이 고쳐진 문서)\n"
        f"- 정답 미포함 · 지어냄 아님 {crossing[(False, False)]}/{sum(crossing.values())} 에는 다른 말로 · 일부만 답한 "
        "것과 문서가 바뀌어 다른 값이 맞는 답에, 질문을 비껴간 답이 섞여 있다 — '정답 포함'이 맞는 답을 놓쳐 낮게 재는 "
        "몫의 상한이다\n"
        "\n중간 구간 안쪽 (참고 — 세부 구간마다 문항이 적다):\n\n"
        + _fine_band_table(records, verdicts)
    )


def _fine_band_table(
    records: Sequence[AnswerRecord], verdicts: Mapping[str, GroundingVerdict]
) -> str:
    groups: dict[str, list[AnswerRecord]] = defaultdict(list)
    for record in records:
        groups[score_band(record.top_score, FINE_BAND_EDGES)].append(record)
    rows = []
    for band in score_bands(FINE_BAND_EDGES):
        group = groups.get(band)
        if not group:
            continue
        # 판정은 grounded 로 끝난 답에만 있다
        fabricated = []
        for record in group:
            verdict = verdicts.get(record.item.id)
            if verdict is not None and verdict.verdict not in EXCLUDED:
                fabricated.append(verdict.verdict in FABRICATED)
        rows.append(
            {
                "1위 점수 (세부)": band,
                "문항": len(group),
                "grounded → partial": sum(r.stage_path == ("grounded", "partial") for r in group),
                "지어냄": format_share(share(fabricated)),
            }
        )
    return render_table(rows, ["1위 점수 (세부)", "문항", "grounded → partial", "지어냄"])


def _difference_section(
    records: Sequence[AnswerRecord],
    verdicts: Mapping[str, GroundingVerdict],
    paragraphs: Mapping[str, str],
) -> str:
    lower, upper = SCORE_BAND_EDGES
    groups = _judged_by_band(records, verdicts)
    middle = [pair for pair in groups.get(score_band(lower), []) if pair[1].verdict not in EXCLUDED]
    high = [pair for pair in groups.get(score_band(upper), []) if pair[1].verdict not in EXCLUDED]
    if not middle or not high:
        return "판정한 답이 한쪽 구간에 없다\n"
    rows = [
        _difference_row(name, labels, middle, high, paragraphs)
        for name, labels in (("지어냄", FABRICATED), ("곁가지 포함", UNSUPPORTED))
    ]
    return render_table(
        rows,
        ["지표", "0.83–0.94", "≥ 0.94", "차이 (%p)", "Newcombe 95%", "문단 묶음 bootstrap 95%"],
    ) + (
        f"\n- 분모: 0.83–0.94 {len(middle)} · ≥ 0.94 {len(high)} (본문 거절 뺌). 두 구간은 서로 다른 문항이라 짝 검정"
        "(McNemar)이 아니라 독립 두 비율의 차이 — Newcombe (1998) 방법 10 (각 비율의 Wilson 구간을 합침)\n"
        f"- 문단 묶음 bootstrap: 같은 문단의 질문은 문맥을 나눠 써서 독립이 아니다 — 구간마다 문단을 복원 추출 "
        f"(2000회 · seed {SEED})\n"
        "- 결정 규칙 (답을 보기 전에 고정): 지어냄 차이의 Newcombe 구간이 0 위면 중간 구간이 더 지어낸다 → `high` 를 "
        f"{upper:g} 로 올리자고 제안, 0 을 포함하면 유지 제안. 결론은 사용자와 정한다. 중간 구간 분모가 100 에 못 미치면 "
        "'못 잼' (ADR-024)\n"
    )


def _difference_row(
    name: str,
    labels: frozenset[str],
    middle: Sequence[tuple[AnswerRecord, GroundingVerdict]],
    high: Sequence[tuple[AnswerRecord, GroundingVerdict]],
    paragraphs: Mapping[str, str],
) -> dict[str, Any]:
    middle_count = sum(verdict.verdict in labels for _, verdict in middle)
    high_count = sum(verdict.verdict in labels for _, verdict in high)
    newcombe = newcombe_difference(middle_count, len(middle), high_count, len(high))
    cluster = cluster_bootstrap_difference(
        _paragraph_groups(middle, labels, paragraphs),
        _paragraph_groups(high, labels, paragraphs),
        seed=SEED,
    )
    return {
        "지표": name,
        "0.83–0.94": format_share(Share(middle_count, len(middle))),
        "≥ 0.94": format_share(Share(high_count, len(high))),
        "차이 (%p)": f"{newcombe.difference * 100:+.1f}",
        "Newcombe 95%": interval_text(newcombe.low, newcombe.high),
        "문단 묶음 bootstrap 95%": interval_text(cluster.low, cluster.high),
    }


def _paragraph_groups(
    judged: Sequence[tuple[AnswerRecord, GroundingVerdict]],
    labels: frozenset[str],
    paragraphs: Mapping[str, str],
) -> list[list[bool]]:
    groups: dict[str, list[bool]] = defaultdict(list)
    for record, verdict in judged:
        groups[paragraphs[record.item.id]].append(verdict.verdict in labels)
    return [groups[paragraph] for paragraph in sorted(groups)]


def _retrieval_cross_section(
    records: Sequence[AnswerRecord],
    verdicts: Mapping[str, GroundingVerdict],
    pool_rows: Sequence[PoolRow],
) -> str:
    # 정답 근거 = 점수 풀 검색의 상위 5 (answer_rank) — 구간은 위 표들처럼 답변 경로가 다시 검색한 1위 점수로
    ranks = {row.id: row.answer_rank for row in pool_rows}
    groups = _judged_by_band(records, verdicts)
    rows = []
    fabricated: dict[str, Counter[bool]] = {}
    for band in score_bands():
        judged = [pair for pair in groups.get(band, []) if pair[1].verdict not in EXCLUDED]
        if not judged:
            continue
        fabricated[band] = Counter(
            ranks[record.item.id] is not None
            for record, verdict in judged
            if verdict.verdict in FABRICATED
        )
        for found in (False, True):
            group = [
                verdict
                for record, verdict in judged
                if (ranks[record.item.id] is not None) == found
            ]
            rows.append(
                {
                    "1위 점수": band,
                    "상위 5 에 정답 근거": "있음" if found else "없음",
                    "grounded 답": len(group),
                    "지어냄 (95% 구간)": rate_with_interval(
                        sum(verdict.verdict in FABRICATED for verdict in group), len(group)
                    ),
                }
            )
    if not rows:
        return "판정한 답이 없다\n"
    lower = score_band(SCORE_BAND_EDGES[0])
    middle = fabricated.get(lower, Counter())
    return render_table(
        rows, ["1위 점수", "상위 5 에 정답 근거", "grounded 답", "지어냄 (95% 구간)"]
    ) + (
        f"\n- {lower} 구간 지어냄 {middle.total()} 중 {middle[False]}문항이 검색 상위 5 에 정답 근거가 없던 문항이다. "
        "분모는 지어냄 판정 표와 같다 (grounded 답 − 본문 거절)\n"
        "- 정답 근거 = 점수 풀 검색 상위 5 에 정답 문서이면서 정답 문자열을 담은 청크 (`answer_rank`, 점수 풀 표와 같은 "
        "기준). 구간마다 분모가 작아 참고값이다\n"
    )


def judge_method_section(verdicts: Mapping[str, GroundingVerdict]) -> str:
    """판정 방법 절 — 판정 수와 판정 규칙."""
    reviewed = [verdict for verdict in verdicts.values() if verdict.reviewed]
    changes = Counter(
        (VERDICT_LABELS[v.first_verdict], VERDICT_LABELS[v.verdict])
        for v in reviewed
        if v.first_verdict != v.verdict
    )
    change_text = (
        " · ".join(
            f"{first} → {final} {count}" for (first, final), count in sorted(changes.items())
        )
        or "없음"
    )
    return (
        "".join(f"- {line}\n" for line in JUDGE_METHOD)
        + f"- 검수 결과: 판정 {len(verdicts)} 중 메인 세션이 다시 본 것 {len(reviewed)}, 바뀐 판정 "
        f"{sum(changes.values())} ({change_text})\n"
    )


def _judged_by_band(
    records: Sequence[AnswerRecord], verdicts: Mapping[str, GroundingVerdict]
) -> dict[str, list[tuple[AnswerRecord, GroundingVerdict]]]:
    groups: dict[str, list[tuple[AnswerRecord, GroundingVerdict]]] = defaultdict(list)
    for record in records:
        if final_stage(record) == "grounded":
            groups[score_band(record.top_score)].append((record, verdicts[record.item.id]))
    return groups
