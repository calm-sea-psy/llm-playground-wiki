"""답변 평가 기록을 읽는 묶음 — 층화 표본 경로 · 한 프롬프트의 기록 · 판정 · 실행 설정 표시."""

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from playground_wiki.evaluation.answer_metrics import final_stage
from playground_wiki.evaluation.answer_run import AnswerRecord, read_records, tagged
from playground_wiki.evaluation.files import read_jsonl, read_run_meta
from playground_wiki.evaluation.grounding import GroundingVerdict
from playground_wiki.evaluation.paths import F9_ANSWERS_PATH as ANSWERS_PATH
from playground_wiki.evaluation.strata import PoolRow

POOL_PATH = Path("data/eval/f9-pool-scores-v1.jsonl")
STRATA_SAMPLE_PATH = Path("data/eval/f9-strata-sample-v1.jsonl")
STRATA_ANSWERS_PATH = Path("data/eval/f9-strata-answers-v1.jsonl")
JUDGE_INPUT_PATH = Path("data/eval/f9-strata-judge-input-v1.jsonl")
# 가린 번호 → 문항 id. 판정자에게는 주지 않는다
JUDGE_KEY_PATH = Path("data/eval/f9-strata-judge-key-v1.jsonl")
VERDICTS_PATH = Path("data/eval/f9-strata-verdicts-v1.jsonl")
# 기준에서 grounded 로 답하던 문항이 다시 잰 쪽에서 grounded → partial 로 넘어갔을 때, 그 partial 문서 칸이 문서로 제대로
# 답했는지 (꼬리표만 바뀐 것인지 실제 손실인지). 늘 꼬리표를 붙여 쓴다 (cascade-input --tag)
CASCADE_INPUT_PATH = Path("data/eval/f9-cascade-judge-input-v1.jsonl")
CASCADE_KEY_PATH = Path("data/eval/f9-cascade-judge-key-v1.jsonl")
CASCADE_VERDICTS_PATH = Path("data/eval/f9-cascade-verdicts-v1.jsonl")


@dataclass(frozen=True)
class Measurement:
    """한 프롬프트로 잰 기록 — F9 · 층화 표본 기록과 층화 표본 판정."""

    f9: tuple[AnswerRecord, ...]
    strata: tuple[AnswerRecord, ...]
    verdicts: Mapping[str, GroundingVerdict]
    """문항 id → 판정. 층화 표본에서 grounded 로 끝난 답마다 하나."""
    f9_runs: tuple[Mapping[str, Any], ...]
    """F9 기록의 meta runs (실행마다 설정)."""
    strata_runs: tuple[Mapping[str, Any], ...]
    """층화 표본 기록의 meta runs."""

    @classmethod
    def load(cls, tag: str | None) -> Self:
        """꼬리표 tag 의 기록 파일들에서. None 이면 꼬리표 없는 기록 (기준)."""
        f9_path = tagged(ANSWERS_PATH, tag)
        strata_path = tagged(STRATA_ANSWERS_PATH, tag)
        return cls(
            f9=tuple(read_records(f9_path)),
            strata=tuple(read_records(strata_path)),
            verdicts=read_verdicts(tag),
            f9_runs=tuple(read_runs(f9_path)),
            strata_runs=tuple(read_runs(strata_path)),
        )


@dataclass(frozen=True)
class SidePair:
    """같은 세트를 RAG(`/ask` 경로)와 closed-book 으로 잰 기록 한 쌍."""

    rag: tuple[AnswerRecord, ...]
    closed: tuple[AnswerRecord, ...]
    rag_runs: tuple[Mapping[str, Any], ...]
    """RAG 기록의 meta runs (실행마다 설정)."""
    closed_runs: tuple[Mapping[str, Any], ...]
    """closed-book 기록의 meta runs."""

    @classmethod
    def load(cls, path: Path, *, tag: str, rag_tag: str | None) -> Self:
        """기록 path 의 꼬리표 tag(closed-book) · rag_tag(RAG, None 이면 꼬리표 없는 기록) 파일에서."""
        rag_path = tagged(path, rag_tag)
        closed_path = tagged(path, tag)
        return cls(
            rag=tuple(read_records(rag_path)),
            closed=tuple(read_records(closed_path)),
            rag_runs=tuple(read_runs(rag_path)),
            closed_runs=tuple(read_runs(closed_path)),
        )


def read_pool() -> list[PoolRow]:
    """저장된 F9 후보 풀 점수를 읽는다."""
    return [PoolRow.from_json(row) for row in read_jsonl(POOL_PATH)]


def interval_text(low: float, high: float) -> str:
    """차이의 신뢰구간을 퍼센트포인트 `+1.0 ~ +2.0` 꼴 글로."""
    return f"{low * 100:+.1f} ~ {high * 100:+.1f}"


def read_verdicts(tag: str | None) -> dict[str, GroundingVerdict]:
    """문항 id → 판정 — 판정 열쇠로 가린 번호를 푼다."""
    key = {row["judge_id"]: row["id"] for row in read_jsonl(tagged(JUDGE_KEY_PATH, tag))}
    rows = read_jsonl(tagged(VERDICTS_PATH, tag))
    return {key[verdict.judge_id]: verdict for verdict in map(GroundingVerdict.from_json, rows)}


def read_runs(answers_path: Path) -> list[Mapping[str, Any]]:
    """답 기록 메타의 runs (실행마다 설정)."""
    return read_run_meta(answers_path)


def read_cascade_verdicts(
    tag: str, base_tag: str | None = None
) -> dict[tuple[str, str, str], GroundingVerdict] | None:
    """(기록, 쪽, 문항 id) → 넘어간 답의 판정. 판정 파일이 없으면 None.

    Raises:
      SystemExit: 판정 열쇠가 다른 기준(base_tag)으로 만든 것일 때 — 열쇠에 기준이 없던 기록은 꼬리표 없는 기준이다.
    """
    path = tagged(CASCADE_VERDICTS_PATH, tag)
    if not path.exists():
        return None
    key_rows = read_jsonl(tagged(CASCADE_KEY_PATH, tag))
    if key_rows and (bases := {row.get("base_tag") for row in key_rows}) != {base_tag}:
        raise SystemExit(
            f"{path} 는 기준 {sorted(map(str, bases))} 로 만든 판정이다 — compare 의 기준 {base_tag} 와 다르다. "
            "같은 --base-tag 로 cascade-input 부터 다시"
        )
    key = {row["judge_id"]: (row["record"], row["side"], row["id"]) for row in key_rows}
    verdicts = map(GroundingVerdict.from_json, read_jsonl(path))
    return {key[verdict.judge_id]: verdict for verdict in verdicts}


def flat_settings(settings: Mapping[str, Any], prefix: str = "") -> dict[str, str]:
    """중첩된 실행 설정을 `앞말.키` → JSON 글 값으로 편다."""
    flat: dict[str, str] = {}
    for name, value in settings.items():
        if isinstance(value, Mapping):
            flat |= flat_settings(value, f"{prefix}{name}.")
        else:
            flat[f"{prefix}{name}"] = json.dumps(value, ensure_ascii=False)
    return flat


def prompt_versions(measurement: Measurement) -> str:
    """Grounded 로 끝난 기록의 프롬프트 판과 수를 ` · ` 로 잇는다."""
    versions = Counter(
        r.prompt_version
        for r in (*measurement.f9, *measurement.strata)
        if final_stage(r) == "grounded"
    )
    return " · ".join(f"`{version}` {count}" for version, count in sorted(versions.items()))
