"""반복 측정 집계 — 회차별 · 문항별로 답 표식과 단계가 흔들린 정도."""

import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence

from playground_evalkit import render_table

from playground_wiki.evaluation.answer_metrics import marker_outcome, stage_path
from playground_wiki.evaluation.answer_report import MARKER_LABELS
from playground_wiki.evaluation.answer_run import AnswerRecord

# 회차 기록의 꼬리표: 묶음-회차번호 (noise-v2-3 → 묶음 noise-v2 · 3회차)
ROUND_TAG = re.compile(r"(?P<series>.+)-(?P<round>\d+)")
TALLY_ROUND_COLUMNS = (
    "꼬리표",
    "grounded 프롬프트",
    "문항",
    "grounded 로 끝남",
    "grounded → partial",
    "그 밖 경로",
    "찾음",
    "못 찾음",
    "그 밖 표지",
)


def render_tally(rounds: Mapping[str, Sequence[AnswerRecord]], prompts: Mapping[str, str]) -> str:
    """회차 기록들의 단계 경로 · 첫 줄 표지 표 markdown — 문항 id 와 집계만 (질문 · 답 원문 없음).

    Args:
      rounds: 꼬리표 → 그 회차의 기록. 꼬리표 끝 -N 이 회차 번호, 그 앞이 묶음 (noise-v2-3 → noise-v2 · 3회차).
        끝에 번호가 없으면 꼬리표 전체가 묶음 · 1회차.
      prompts: 꼬리표 → 그 기록의 grounded 단계 블록 프롬프트 (meta 설정).
    """
    per_round = [
        _tally_round_row(tag, prompts[tag], rounds[tag]) for tag in sorted(rounds, key=_split_round)
    ]
    cells: dict[tuple[str, str], dict[int, AnswerRecord]] = defaultdict(dict)
    for tag, records in rounds.items():
        series, number = _split_round(tag)
        for record in records:
            cells[series, record.item.id][number] = record
    numbers = sorted({_split_round(tag)[1] for tag in rounds})
    per_item = [
        _tally_item_row(series, qid, by_round, numbers)
        for (series, qid), by_round in sorted(cells.items())
    ]
    return (
        render_table(per_round, TALLY_ROUND_COLUMNS)
        + "\n"
        + render_table(
            per_item,
            [
                "묶음",
                "id",
                "회차",
                "grounded 로 끝남",
                "grounded → partial",
                *(f"{number}회" for number in numbers),
            ],
        )
        + "\n- 같은 문항을 회차마다 따로 돌린 기록 — 같은 프롬프트로도 답이 갈린다 (Ollama GPU, temperature 0.1). "
        "표지는 grounded 첫 줄 (못 찾음이면 partial 로 넘어간다)\n"
    )


def _split_round(tag: str) -> tuple[str, int]:
    match = ROUND_TAG.fullmatch(tag)
    return (match["series"], int(match["round"])) if match else (tag, 1)


def _tally_round_row(
    tag: str, prompt: str, records: Sequence[AnswerRecord]
) -> dict[str, str | int | float]:
    paths = Counter(stage_path(record) for record in records)
    markers = Counter(_marker_label(record) for record in records)
    return {
        "꼬리표": tag,
        "grounded 프롬프트": f"`{prompt}`",
        "문항": len(records),
        "grounded 로 끝남": paths["grounded"],
        "grounded → partial": paths["grounded → partial"],
        "그 밖 경로": _others(paths, {"grounded", "grounded → partial"}),
        "찾음": markers[MARKER_LABELS["found"]],
        "못 찾음": markers[MARKER_LABELS["not_found"]],
        "그 밖 표지": _others(markers, {MARKER_LABELS["found"], MARKER_LABELS["not_found"]}),
    }


def _tally_item_row(
    series: str, qid: str, by_round: Mapping[int, AnswerRecord], numbers: Sequence[int]
) -> dict[str, str | int | float]:
    paths = Counter(stage_path(record) for record in by_round.values())
    return {
        "묶음": series,
        "id": qid,
        "회차": len(by_round),
        "grounded 로 끝남": paths["grounded"],
        "grounded → partial": paths["grounded → partial"],
        **{
            f"{number}회": (
                f"{stage_path(by_round[number])} · {_marker_label(by_round[number])}"
                if number in by_round
                else ""
            )
            for number in numbers
        },
    }


def _marker_label(record: AnswerRecord) -> str:
    outcome = marker_outcome(record)
    return MARKER_LABELS[outcome] if outcome is not None else "표지 없음"


def _others(counts: Counter[str], shown: set[str]) -> str:
    return " · ".join(
        f"{name} {count}" for name, count in sorted(counts.items()) if name not in shown
    )
