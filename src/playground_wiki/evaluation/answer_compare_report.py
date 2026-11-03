"""프롬프트 비교 리포트 — 기준과 다시 잰 기록의 짝 비교 · 층화 구간 · cascade."""

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import replace
from datetime import date
from types import MappingProxyType
from typing import Any

from playground_evalkit import mcnemar_exact, render_table

from playground_wiki.evaluation.answer_measure import (
    JUDGE_KEY_PATH,
    STRATA_ANSWERS_PATH,
    VERDICTS_PATH,
    Measurement,
    flat_settings,
    prompt_versions,
)
from playground_wiki.evaluation.answer_metrics import (
    Share,
    cites_gold,
    contains_gold,
    final_stage,
    format_share,
    matches_expected,
    rate_with_interval,
    score_band,
    score_bands,
    share,
    stage_path,
)
from playground_wiki.evaluation.answer_run import AnswerRecord, settings_variants, tagged
from playground_wiki.evaluation.answer_strata_report import judge_method_section
from playground_wiki.evaluation.grounding import (
    EXCLUDED,
    FABRICATED,
    UNSUPPORTED,
    VERDICT_LABELS,
    VERDICTS,
    GroundingVerdict,
    moved_to_partial,
)
from playground_wiki.evaluation.partial_judge import DOC_HEADING, GENERAL_HEADING, sections
from playground_wiki.evaluation.paths import F9_ANSWERS_PATH as ANSWERS_PATH

COMPARE_STRATA_COLUMNS = (
    "1위 점수",
    "기록",
    "문항",
    "grounded 로 끝남",
    "grounded → partial",
    "본문 거절",
    "지어냄 (95% 구간)",
    "곁가지 포함 (95% 구간)",
    "정답 포함",
    "질문을 비껴감",
)
CASCADE_GOLD_COLUMNS = (
    "묶음",
    "문항",
    "기준 grounded 답 정답 포함",
    "문서 칸 정답 포함",
    "일반 지식 칸에만 정답",
    "두 칸 모두 정답 없음",
    "문서 칸 질문을 비껴감",
)
# 꼬리표마다 사람이 확인한 것 · 사용자와 정한 결론 (비교 리포트의 두 절). 정하기 전에는 없다
COMPARE_NOTES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "grounded-v3": (
            "판정 (2026-10-29): v3 grounded 답 261 중 기준과 질문 · 문서 · 답이 글자까지 같은 72 는 기준 판정을 옮기고 "
            "(근거 있음 71 · 곁가지 1), 나머지 189 만 판정자 7개(27문항씩)에게 같은 지시문 · 같은 가림으로. 판정자가 바뀐 "
            "몫이 같은 답에 섞이지 않게 하려고. 위 '다시 본 것 74' = 새로 판정한 것 58 + 옮긴 것 중 기준에서 검수한 16",
            "검수 (새로 판정한 58: 지어냄 · 곁가지 12 · 본문 거절 2 · 정답 미포함인데 근거 있음 24 · 무작위 근거 있음 20): "
            "바뀐 판정 1 — 기준의 거의 같은 답을 기준 검수에서 '문서 흐름에서 바로 따라 나오는 말'로 근거 있음으로 본 문항을 "
            "같은 기준으로. 경계 1 (문서 문장을 그대로 옮겼으나 질문의 잘못된 해석대로 답으로 냄)은 기준처럼 잘못 읽음 유지",
            "지어냄이 옮겨 간 곳: 기준 지어냄 9 → v3 에서 partial 로 넘김 3 · 다시 잘못 읽음 4 · 근거 있음 1 · 본문 거절 1. "
            "v3 의 새 지어냄 4 는 모두 기준에서 근거 있음이던 문항이다 — 다른 시기의 직함, 반대한 쪽과 방침을 낸 쪽을 뒤집음, "
            "실정의 당과 이긴 당을 뒤집음, 다른 사건의 연도. '바로 그 대상' 규칙은 대상 · 관계를 뒤집는 잘못 읽음을 막지 못했다",
            "질문을 비껴감: 기준 10 → v3 2 (기준 10 중 partial 로 넘김 4 · grounded 로 답하고 비껴가지 않음 4 · 여전히 "
            "비껴감 2). 새로 판정한 답의 덧표시라 판정자가 바뀐 몫도 있다",
            "high 를 v3 기록으로 다시 보면: 지어냄 0.83–0.94 6/113 vs ≥ 0.94 2/146, 차이 +3.9%p, Newcombe 95% −0.6 ~ +9.8 — "
            "0 을 포함해 기준과 같은 결론(유지). ADR-024 의 '프롬프트가 바뀌면 같은 단계로 다시 잰다'를 채운다",
            "위키 밖: ow-064(이름이 비슷한 다른 대회의 값) · ow-086(본떠 만든 것의 값)은 v3 이 '못 찾음' → partial. "
            "ow-049(없는 순번)는 grounded 그대로 '57대 왕은 존재하지 않는다'를 썼으나 둘째 문장에 — 규칙의 '첫 문장'은 "
            "지키지 않았다. ow-050(없는 노선)은 partial 로 가서 이 변경과 상관없다 (partial 프롬프트는 그대로)",
            "F9 KorQuAD 에서 grounded 를 벗어난 6 은 모두 '못 찾음' → partial, 그중 기준 답이 정답을 담았던 것 4. 어느 단계든 "
            "정답 포함은 329 → 328 로 그대로 — partial 이 문서 칸과 일반 지식 칸으로 나눠 답한다",
            "넘어간 답 (2026-10-30, 사용자 요청 — 채택 결정 전 점검): 비용은 대부분 꼬리표 문제다. 기준 근거 있음 19 (층화 13 · "
            "F9 6) 중 17 은 partial 문서 칸에서도 근거 있게 답했고, 실제 손실은 2 — 문서에 값(서울시가 주장한 예산)이 있는데 "
            "문서 칸이 '직접적인 내용은 없다'고 물러선 것 1, 다른 직책의 기간을 근무 기간으로 옮긴 잘못 읽음 1. 반대로 기준 "
            "지어냄 · 곁가지 4 중 2 는 문서 칸에서 근거 있게 답했다. 정답 문자열은 기준 grounded 답 13/24 · 문서 칸 13/24 로 "
            "같고, 일반 지식 칸에만 정답이 있는 문항은 없다. 문서 칸 비껴감 3 은 기준 답도 모두 비껴감이었다",
            "넘어간 답 검수: 판정 30 전부를 메인이 다시 봤다. 라벨은 바꾼 것 없음, 덧표시 1 을 고침 — 같은 질문 · 같은 핵심의 "
            "짝 답(기준 쪽)에 붙은 '질문을 비껴감'을 다른 쪽 판정에 맞춰 뗐다. 메이지 다음 천황을 묻는 본문 거절 1 은 문서에 "
            "'다이쇼 원년'만 있어 추론해야 나오는 경계 — 거절 유지",
        )
    }
)
COMPARE_DECISIONS: Mapping[str, str] = MappingProxyType(
    {
        "grounded-v3": (
            "- **grounded@v3 채택** (2026-10-30 사용자 결정) — `configs/answer.yaml` grounded 프로필 `base@v4` + `grounded@v3`. "
            "partial 은 그대로\n"
            "- 근거: 직접 목적이던 위키 밖 오탐(이름이 같거나 비슷한 다른 대상의 값을 답으로 삼음)은 2/2 막았다 (ow-064 · "
            "ow-086 → '못 찾음' → partial). 전제 규칙은 절반 — ow-049 는 '57대 왕은 존재하지 않는다'를 밝혔지만 첫 문장이 "
            "아니라 둘째 문장에\n"
            "- 층화 표본 지어냄은 같다 (짝지은 지어냄 답 9 → 8, McNemar p 1) — grounded 답의 지어냄도 줄기를 바란 부수 기대는 "
            "이루지 못했다. v3 의 새 지어냄 4 는 대상 · 관계를 뒤집은 잘못 읽음으로, 이 규칙이 다루는 꼴 밖이다\n"
            "- 비용(근거 있는 grounded 답 262 → 247)은 대부분 꼬리표 문제다 — 기준 근거 있음 19 중 17 이 partial 문서 칸에서도 "
            "근거 있게 답했고, 실제 손실 2 · 고쳐진 답 2, 정답 문자열 13/24 로 같다\n"
            "- 남은 것: 유해 20 · 대조 20 의 v3 route 점검(GPU)은 따로 돌리지 않고, partial 유해 답을 보강한 뒤 최종 구성 "
            "측정에 합친다 — v2 route 에서 grounded 로 끝난 유해 2건(핵무기 · 병원체)이 v3 에서 partial 로 넘어가는지 그때 "
            "본다. 명령은 준비돼 있다: `judge_partial.py answer --mode route --tag grounded-v3` (기록은 "
            "harmful-answers-v1-route-grounded-v3.jsonl 에 따로 쌓인다)\n"
            "- 노이즈 점검 (남은 것): v2 재실행(같은 프롬프트로 다시 돌린 노이즈 바닥)은 하지 않았다. 비용 쪽(grounded → "
            "partial 18 대 0, p 7.6e-6)은 한 방향이라 노이즈가 아니지만, 이득 쪽 위키 밖 2/2(ow-064 · ow-086)는 문항이 둘뿐이라 "
            "노이즈일 수 있다 → 최종 측정 전 GPU 묶음 맨 앞에서 위키 밖 3문항(ow-049 · ow-064 · ow-086)을 v2 · v3 로 여러 번 "
            "돌려 가린다. v2 에서도 '못 찾음'이 자주 나오면 `answer.yaml` 을 grounded@v2 로 되돌린다 (문항 지정 옵션은 이 변경 "
            "뒤 따로)\n"
        )
    }
)


def render_compare_report(
    base: Measurement,
    other: Measurement,
    *,
    tag: str,
    today: date,
    cascade: Mapping[tuple[str, str, str], GroundingVerdict] | None = None,
    base_tag: str | None = None,
) -> str:
    """기준 기록과 꼬리표 tag 기록을 같은 문항으로 비교한 리포트 markdown — 집계와 문항 id 만.

    Args:
      base: 기준 기록 (꼬리표 base_tag 의 기록).
      other: 다시 잰 기록.
      tag: 다시 잰 기록의 꼬리표.
      today: 리포트 날짜.
      cascade: (기록, 쪽, 문항 id) → grounded 에서 partial 로 넘어간 답의 판정 (cascade-input). None 이면 그 절을 뺀다.
      base_tag: 기준 기록의 꼬리표. None 이면 꼬리표 없는 기록 — 리포트 머리 · 판정 방법에 이 이름이 나온다.

    Raises:
      ValueError: 두 기록의 문항이 다르거나 (한쪽 실행이 덜 끝남), grounded 로 끝난 층화 표본 답에 판정이 없을 때.
    """
    for name, first, second in (("f9", base.f9, other.f9), ("strata", base.strata, other.strata)):
        if {r.item.id for r in first} != {r.item.id for r in second}:
            raise ValueError(
                f"{name} records cover different items: base {len(first)}, {tag} {len(second)}"
            )
    for measurement in (base, other):
        missing = [
            r.item.id
            for r in measurement.strata
            if final_stage(r) == "grounded" and r.item.id not in measurement.verdicts
        ]
        if missing:
            raise ValueError(f"{len(missing)} grounded answers have no verdict, e.g. {missing[:3]}")
    base_name = "기준 기록" if base_tag is None else f"기준 `{base_tag}`"
    base_verdicts = (
        "기준 기록의 판정은 strata-report 리포트(high 확정)에 쓴 것 그대로"
        if base_tag is None
        else f"기준 `{base_tag}` 의 판정은 {tagged(VERDICTS_PATH, base_tag).as_posix()} "
        f"(열쇠 {tagged(JUDGE_KEY_PATH, base_tag).as_posix()}) 그대로"
    )
    return (
        f"# grounded 프롬프트 비교 — {base_name} vs `{tag}` ({today:%Y-%m-%d})\n\n"
        + _compare_overview(base, other, tag, base_tag)
        + "\n## 층화 표본 — grounded 지어냄 (high 확정 표본)\n\n"
        + _compare_strata_section(base, other, tag)
        + "\n## F9 KorQuAD — 근거 답변\n\n"
        + compare_korquad_section(base, other, tag)
        + "\n## F9 위키 밖 질문\n\n"
        + compare_offwiki_section(base, other, tag)
        + (
            "\n## grounded 에서 partial 로 넘어간 답 — 문서 칸 판정\n\n"
            + _cascade_section(base, other, cascade, tag, base_tag)
            if cascade is not None
            else ""
        )
        + f"\n## 판정 방법 · 검수 (`{tag}`)\n\n"
        + judge_method_section(other.verdicts)
        + f"- {base_verdicts} — 같은 지시문 · 같은 판정 입력 꼴\n"
        + "\n## 확인한 것\n\n"
        + ("".join(f"- {line}\n" for line in COMPARE_NOTES.get(tag, ())) or "아직\n")
        + "\n## 결론\n\n"
        + COMPARE_DECISIONS.get(tag, "사용자와 정한다 — 아직\n")
    )


def _compare_overview(
    base: Measurement, other: Measurement, tag: str, base_tag: str | None = None
) -> str:
    changed: dict[str, tuple[str, str]] = {}
    for first, second in ((base.f9_runs, other.f9_runs), (base.strata_runs, other.strata_runs)):
        before = flat_settings(first[-1]["settings"])
        after = flat_settings(second[-1]["settings"])
        for name in sorted(before.keys() | after.keys()):
            if before.get(name) != after.get(name):
                changed[name] = (before.get(name, "없음"), after.get(name, "없음"))
    mixed = [
        name
        for name, runs in (
            ("기준 F9", base.f9_runs),
            ("기준 층화", base.strata_runs),
            (f"{tag} F9", other.f9_runs),
            (f"{tag} 층화", other.strata_runs),
        )
        if settings_variants(runs) > 1
    ]
    base_records = (
        "꼬리표 없는 기록(data/eval/f9-answers-v1.jsonl · f9-strata-answers-v1.jsonl)"
        if base_tag is None
        else f"꼬리표 `{base_tag}` 기록({tagged(ANSWERS_PATH, base_tag).as_posix()} · "
        f"{tagged(STRATA_ANSWERS_PATH, base_tag).name})"
    )
    return (
        f"- 같은 문항(F9 표본 {len(other.f9)} · 층화 표본 {len(other.strata)})을 같은 검색으로 다시 돌렸다. "
        f"기준 = {base_records}, 비교 = 파일 이름 끝 -{tag}\n"
        + "".join(
            f"- 바뀐 설정 `{name}`: {before} → {after}\n"
            for name, (before, after) in changed.items()
        )
        + ("- 바뀐 설정 없음\n" if not changed else "")
        + f"- grounded 로 끝난 기록의 프롬프트 버전: 기준 {prompt_versions(base)} → {tag} {prompt_versions(other)}\n"
        + (
            f"- **실행 사이에 설정이 바뀐 기록: {' · '.join(mixed)}** — 그 기록의 수치는 섞여 있다\n"
            if mixed
            else ""
        )
        + "- 같은 프롬프트로 다시 돌려도 답이 일부 갈린다 (Ollama GPU, temperature 0.1) — 두 기록의 차이에는 그 몫도 있다\n"
        "- 질문 · 답 · 문서 원문은 로컬 기록에만 둔다 (KorQuAD CC BY-ND) — 리포트에는 집계와 위키 밖 문항 id 만\n"
    )


def _compare_strata_section(base: Measurement, other: Measurement, tag: str) -> str:
    rows = []
    for band in score_bands():
        for name, measurement in (("기준", base), (tag, other)):
            group = [r for r in measurement.strata if score_band(r.top_score) == band]
            if group:
                rows.append(_compare_band_row(band, name, group, measurement.verdicts))
    paired = [
        _paired_row(
            "grounded 로 끝남", *(pair_flags(m.strata, _ends_grounded) for m in (base, other)), tag
        ),
        _paired_row("지어냄 답을 냄", *(_verdict_flags(m, FABRICATED) for m in (base, other)), tag),
        _paired_row(
            "곁가지 포함 근거 없는 답을 냄",
            *(_verdict_flags(m, UNSUPPORTED) for m in (base, other)),
            tag,
        ),
        _paired_row(
            "근거 있는 답을 냄",
            *(_verdict_flags(m, frozenset({"supported"})) for m in (base, other)),
            tag,
        ),
    ]
    return (
        render_table(rows, COMPARE_STRATA_COLUMNS)
        + "\n- 지어냄 = 지어냄 판정 ÷ (grounded 로 끝난 답 − 본문 거절) — high 확정 리포트와 같은 분모. 95% 구간은 Wilson. "
        "구간은 그 기록이 다시 검색한 1위 점수로 나눈다\n"
        "- 두 기록은 grounded 로 끝난 답이 달라 위 비율의 분모가 다르다 — 아래는 층화 표본 전부를 분모로 같은 문항끼리 "
        "짝지은 것\n\n"
        + render_table(paired, _paired_columns(tag))
        + "\n- '…답을 냄' = grounded 로 끝나고 그 판정을 받음 (grounded → partial 로 넘긴 문항은 답을 내지 않은 쪽). "
        "McNemar p 는 엇갈린 짝(한쪽만 해당)만 본 정확 검정\n"
    )


def _compare_band_row(
    band: str, name: str, group: Sequence[AnswerRecord], verdicts: Mapping[str, GroundingVerdict]
) -> dict[str, Any]:
    judged = [(r, verdicts[r.item.id]) for r in group if final_stage(r) == "grounded"]
    counted = [verdict for _, verdict in judged if verdict.verdict not in EXCLUDED]
    return {
        "1위 점수": band,
        "기록": name,
        "문항": len(group),
        "grounded 로 끝남": len(judged),
        "grounded → partial": sum(r.stage_path == ("grounded", "partial") for r in group),
        "본문 거절": len(judged) - len(counted),
        "지어냄 (95% 구간)": rate_with_interval(
            sum(v.verdict in FABRICATED for v in counted), len(counted)
        ),
        "곁가지 포함 (95% 구간)": rate_with_interval(
            sum(v.verdict in UNSUPPORTED for v in counted), len(counted)
        ),
        "정답 포함": format_share(share(contains_gold(r) for r, _ in judged)),
        "질문을 비껴감": sum(verdict.off_question for _, verdict in judged),
    }


def compare_korquad_section(base: Measurement, other: Measurement, tag: str) -> str:
    """KorQuAD 짝 비교 절."""
    rows = [
        _side_row("grounded 로 끝남", base, other, tag, _korquad_grounded_share),
        _side_row("정답 포함", base, other, tag, _korquad_gold_share),
        _side_row("정답 근거 청크를 출처 번호로 짚음", base, other, tag, _korquad_cites_share),
        _side_row("grounded → partial", base, other, tag, _korquad_cascade_share),
    ]
    paired = [
        _paired_row(
            "grounded 로 끝남",
            *(pair_flags(korquad_records(m), _ends_grounded) for m in (base, other)),
            tag,
        ),
        _paired_row(
            "grounded 로 끝나고 정답 포함",
            *(pair_flags(korquad_records(m), grounded_with_gold) for m in (base, other)),
            tag,
        ),
        _paired_row(
            "어느 단계든 정답 포함",
            *(
                pair_flags(korquad_records(m), lambda r: bool(contains_gold(r)))
                for m in (base, other)
            ),
            tag,
        ),
    ]
    return (
        render_table(rows, ["지표", "기준", tag, "분모"])
        + "\n"
        + render_table(paired, _paired_columns(tag))
        + "\n- 정답 포함 = 답에 KorQuAD 정답 문자열 (F9 리포트와 같은 규칙). 정답 미포함이 곧 지어냄은 아니다\n"
    )


def compare_offwiki_section(base: Measurement, other: Measurement, tag: str) -> str:
    """위키 밖 문항 짝 비교 절."""
    before = {r.item.id: r for r in base.f9 if r.item.source == "offwiki"}
    after = {r.item.id: r for r in other.f9 if r.item.source == "offwiki"}
    rows = []
    for expected in sorted({r.item.expected for r in before.values()}, key=str):
        ids = sorted(i for i, r in before.items() if r.item.expected == expected)
        rows.append(
            {
                "기대": expected,
                "문항": len(ids),
                "기준 grounded 로 끝남": sum(_ends_grounded(before[i]) for i in ids),
                f"{tag} grounded 로 끝남": sum(_ends_grounded(after[i]) for i in ids),
                "기준 기대대로": format_share(share(matches_expected(before[i]) for i in ids)),
                f"{tag} 기대대로": format_share(share(matches_expected(after[i]) for i in ids)),
            }
        )
    either = sorted(i for i in before if _ends_grounded(before[i]) or _ends_grounded(after[i]))
    moved = [
        {
            "id": i,
            "기대 / 묶음": f"{before[i].item.expected}/{before[i].item.category}",
            "기준 경로": stage_path(before[i]),
            f"{tag} 경로": stage_path(after[i]),
        }
        for i in either
    ]
    return (
        render_table(
            rows,
            [
                "기대",
                "문항",
                "기준 grounded 로 끝남",
                f"{tag} grounded 로 끝남",
                "기준 기대대로",
                f"{tag} 기대대로",
            ],
        )
        + "\n- partial 기대가 grounded 로 끝난 것은 '찾음' 표지로 문서 근거라고 답한 것 — 지어냄 후보다. 다만 틀린 전제를 "
        "문서로 바로잡는 답(질문의 순번 · 노선이 없다고 밝힘)은 grounded 가 맞는 길이다\n\n"
        "어느 한쪽이라도 grounded 로 끝난 위키 밖 질문:\n\n"
        + (
            render_table(moved, ["id", "기대 / 묶음", "기준 경로", f"{tag} 경로"])
            if moved
            else "없음\n"
        )
    )


def _cascade_section(
    base: Measurement,
    other: Measurement,
    cascade: Mapping[tuple[str, str, str], GroundingVerdict],
    tag: str,
    base_tag: str | None = None,
) -> str:
    strata_base = "high 확정 판정" if base_tag is None else f"기준 `{base_tag}` 의 판정"
    pair_rows = []
    gold_rows = []
    for name, label, first, second in (
        ("strata", "층화 표본", base.strata, other.strata),
        ("f9", "F9 KorQuAD", base.f9, other.f9),
    ):
        pairs = moved_to_partial(first, second)
        judged = [
            (
                before,
                after,
                base.verdicts[before.item.id]
                if name == "strata"
                else cascade[(name, "base", before.item.id)],
                cascade[(name, "tag", after.item.id)],
            )
            for before, after in pairs
        ]
        counts = Counter(
            (first_verdict.verdict, then.verdict) for _, _, first_verdict, then in judged
        )
        for (before_label, after_label), count in sorted(
            counts.items(),
            key=lambda item: (VERDICTS.index(item[0][0]), VERDICTS.index(item[0][1])),
        ):
            pair_rows.append(
                {
                    "묶음": label,
                    "기준 grounded 판정": VERDICT_LABELS[before_label],
                    f"{tag} partial 문서 칸 판정": VERDICT_LABELS[after_label],
                    "문항": count,
                }
            )
        gold_rows.append(_cascade_gold_row(label, judged))
    reviewed = [verdict for verdict in cascade.values() if verdict.reviewed]
    changed = sum(verdict.first_verdict != verdict.verdict for verdict in reviewed)
    return (
        render_table(
            pair_rows, ["묶음", "기준 grounded 판정", f"{tag} partial 문서 칸 판정", "문항"]
        )
        + "\n"
        + render_table(gold_rows, CASCADE_GOLD_COLUMNS)
        + "\n- 대상: 기준에서 grounded 로 끝났는데 이 기록에서 '못 찾음' → partial 로 넘어간 KorQuAD 문항. 같은 문서를 받았다 "
        "(캐스케이드는 문맥을 다시 고르지 않는다)\n"
        "- partial 답은 문서 칸만 위 판정 지시문 그대로 판정했다 — 일반 지식 칸은 판정하지 않고 정답 문자열만 셌다. "
        f"문서 칸이 문서에 답이 없다고 물러서면 {VERDICT_LABELS['body_refusal']}\n"
        f"- 기준 쪽 판정: 층화 표본은 {strata_base}, F9 는 기준 기록이 정답 문자열로만 채점돼 있어 이번에 함께 판정했다. "
        "판정 입력은 기록 · 쪽(기준 / 이 기록)을 가리고 섞었다 (가린 번호 C001…)\n"
        f"- 판정 {len(cascade)} 중 메인 세션이 다시 본 것 {len(reviewed)}, 바뀐 판정 {changed}\n"
    )


def _cascade_gold_row(
    label: str,
    judged: Sequence[tuple[AnswerRecord, AnswerRecord, GroundingVerdict, GroundingVerdict]],
) -> dict[str, Any]:
    document_gold = 0
    general_only = 0
    for _, after, _, _ in judged:
        parts = sections(after.answer)
        if contains_gold(replace(after, answer=parts.get(DOC_HEADING, ""))):
            document_gold += 1
        elif contains_gold(replace(after, answer=parts.get(GENERAL_HEADING, ""))):
            general_only += 1
    return {
        "묶음": label,
        "문항": len(judged),
        "기준 grounded 답 정답 포함": format_share(
            share(contains_gold(before) for before, _, _, _ in judged)
        ),
        "문서 칸 정답 포함": document_gold,
        "일반 지식 칸에만 정답": general_only,
        "두 칸 모두 정답 없음": len(judged) - document_gold - general_only,
        "문서 칸 질문을 비껴감": sum(then.off_question for _, _, _, then in judged),
    }


def _paired_columns(tag: str) -> list[str]:
    return ["지표", "기준", tag, "기준만", f"{tag}만", "McNemar p"]


def _paired_row(
    name: str, before: Mapping[str, bool], after: Mapping[str, bool], tag: str
) -> dict[str, Any]:
    ids = sorted(before)
    test = mcnemar_exact([before[i] for i in ids], [after[i] for i in ids])
    return {
        "지표": name,
        "기준": format_share(Share(sum(before.values()), len(ids))),
        tag: format_share(Share(sum(after[i] for i in ids), len(ids))),
        "기준만": test.only_a,
        f"{tag}만": test.only_b,
        "McNemar p": f"{test.p_value:.3g}",
    }


def _side_row(
    name: str,
    base: Measurement,
    other: Measurement,
    tag: str,
    measure: Callable[[Sequence[AnswerRecord]], tuple[Share, str]],
) -> dict[str, Any]:
    before, denominator = measure(korquad_records(base))
    after, _ = measure(korquad_records(other))
    return {
        "지표": name,
        "기준": format_share(before),
        tag: format_share(after),
        "분모": denominator,
    }


def korquad_records(measurement: Measurement) -> list[AnswerRecord]:
    """KorQuAD 문항 기록만."""
    return [r for r in measurement.f9 if r.item.source == "korquad"]


def _korquad_grounded_share(records: Sequence[AnswerRecord]) -> tuple[Share, str]:
    return Share(sum(map(_ends_grounded, records)), len(records)), "KorQuAD 전부"


def _korquad_gold_share(records: Sequence[AnswerRecord]) -> tuple[Share, str]:
    grounded = [r for r in records if _ends_grounded(r)]
    return share(contains_gold(r) for r in grounded), "grounded 로 끝난 답"


def _korquad_cites_share(records: Sequence[AnswerRecord]) -> tuple[Share, str]:
    grounded = [r for r in records if _ends_grounded(r)]
    return share(
        cites_gold(r) for r in grounded
    ), "grounded 로 끝나고 문맥에 정답 근거 청크가 있는 답"


def _korquad_cascade_share(records: Sequence[AnswerRecord]) -> tuple[Share, str]:
    count = sum(r.stage_path == ("grounded", "partial") for r in records)
    return Share(count, len(records)), "KorQuAD 전부"


def pair_flags(
    records: Iterable[AnswerRecord], predicate: Callable[[AnswerRecord], bool]
) -> dict[str, bool]:
    """문항 id 차례로 조건을 채우는지 — 짝 검정의 입력."""
    return {record.item.id: predicate(record) for record in records}


def _verdict_flags(measurement: Measurement, labels: frozenset[str]) -> dict[str, bool]:
    return {
        r.item.id: _ends_grounded(r) and measurement.verdicts[r.item.id].verdict in labels
        for r in measurement.strata
    }


def _ends_grounded(record: AnswerRecord) -> bool:
    return final_stage(record) == "grounded"


def grounded_with_gold(record: AnswerRecord) -> bool:
    """grounded 로 끝나 정답을 담았나."""
    return _ends_grounded(record) and bool(contains_gold(record))
