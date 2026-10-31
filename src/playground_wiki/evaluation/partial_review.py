"""partial 판정기 검증 검수표 — 검수표에 실을 표본 · 사람이 채운 사본 읽기 · 사람과 판정기의 일치 표 (scripts/judge_partial.py).

검수표 꼴(번호 · 문맥 · 답 · 빈 채점 칸)과 일치 계산은 `judge_review` 가, 여기는 partial 판정의 묶음 · 열쇠 파일을 맡는다.
열쇠 파일(빈 검수표 이름의 `.key.json`)의 entries 는 `{number, run, id}` — 검수표 번호와 원래 답(실행, 문항 id)을 잇는다.
"""

import json
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from playground_wiki.evaluation.answer_run import AnswerRecord
from playground_wiki.evaluation.judge_review import (
    Agreement,
    agreement,
    parse_sheet,
    stratified_sample,
)
from playground_wiki.evaluation.partial_judge import JUDGE_ITEMS

F9_GROUPS: Mapping[tuple[str, str | None], str] = MappingProxyType(
    {
        ("korquad", None): "KorQuAD",
        ("offwiki", "partial"): "위키 밖 · partial 기대",
        ("offwiki", "refuse"): "위키 밖 · refuse 기대",
    }
)
"""F9 partial 답의 묶음 — (문항 출처, 기대 단계) → 리포트 · 검수표 표본에 쓰는 이름."""
CHECKED_SUFFIX = "-checked"
"""사람이 채운 검수표는 빈 표 이름에 이것을 붙인 사본이다 (data/bench/paraphrase-v1-review-checked.md 와 같은 규칙)
— 빈 표는 틀로 남고, 일치율은 채운 사본만 읽는다."""


@dataclass(frozen=True)
class ReviewSheet:
    """사람이 채운 검수표 하나 — 리포트 · 일치율 · judge --review-only 가 읽는다."""

    path: Path
    """빈 검수표. 채운 사본은 `checked_path(path)`, 문항 번호 → 기록 열쇠는 `path.with_suffix(".key.json")`."""
    sampling: str
    """어떻게 뽑았나 — 리포트에 그대로 싣는다."""


def checked_path(sheet: Path) -> Path:
    """빈 검수표에 대응하는, 사람이 채운 사본의 경로."""
    return sheet.with_name(f"{sheet.stem}{CHECKED_SUFFIX}{sheet.suffix}")


def f9_group(record: AnswerRecord) -> str:
    """F9 partial 답의 묶음 이름 (F9_GROUPS)."""
    return F9_GROUPS[(record.item.source, record.item.expected)]


def f9_review_keys(records: Iterable[AnswerRecord], size: int, *, seed: int) -> list[str]:
    """F9 partial 답에서 묶음(F9_GROUPS) 크기에 비례해 고른 문항 id."""
    groups: dict[str, list[str]] = defaultdict(list)
    for record in records:
        groups[f9_group(record)].append(record.item.id)
    return stratified_sample(groups, size, seed=seed)


def harmful_review_keys(
    records: Iterable[AnswerRecord], pairs: Mapping[str, str], count: int, *, seed: int
) -> list[str]:
    """유해 · 대조 둘 다 partial 답이 있는 짝 count 개를 골라 두 문항 id 를 섞어서."""
    answered = {record.item.id: record for record in records}
    harmful = sorted(
        item_id
        for item_id, record in answered.items()
        if record.item.category == "harmful" and pairs.get(item_id) in answered
    )
    rng = random.Random(seed)
    chosen = rng.sample(harmful, min(count, len(harmful)))
    keys = [*chosen, *(pairs[item_id] for item_id in chosen)]
    rng.shuffle(keys)
    return keys


def review_keys(*, sheets: Mapping[str, ReviewSheet]) -> set[tuple[str, str]]:
    """검수표(sheets)에 실린 답의 (실행, 문항 id). 열쇠 파일이 없는 검수표는 건너뛴다."""
    keys: set[tuple[str, str]] = set()
    for sheet in sheets.values():
        key_path = sheet.path.with_suffix(".key.json")
        if not key_path.exists():
            continue
        key = json.loads(key_path.read_text(encoding="utf-8"))
        keys |= {(str(entry["run"]), str(entry["id"])) for entry in key["entries"]}
    return keys


def human_labels(
    root: Path, *, sheets: Mapping[str, ReviewSheet]
) -> dict[str, dict[tuple[str, str], dict[str, bool | None]]]:
    """채운 검수표마다 사람 채점 — 검수표 이름 → (실행, 문항 id) → 항목 → 예 · 아니오 · 빈 칸(None).

    Args:
      root: data/ 가 있는 폴더 (레포 루트).
      sheets: 읽을 검수표 — 이름 → 검수표.

    Returns:
      sheets 차례. 채운 사본이 없는 검수표 · 채점 칸이 없는 문항은 뺀다.
    """
    result: dict[str, dict[tuple[str, str], dict[str, bool | None]]] = {}
    for name, sheet in sheets.items():
        read = read_checked(root, sheet.path)
        if read is None:
            continue
        ratings, entries = read
        labels: dict[tuple[str, str], dict[str, bool | None]] = {}
        for entry in entries:
            row = ratings.get(int(str(entry["number"])))
            if row is not None:
                labels[(str(entry["run"]), str(entry["id"]))] = dict(row)
        result[name] = labels
    return result


def read_checked(
    root: Path, sheet: Path
) -> tuple[dict[int, dict[str, bool | None]], list[dict]] | None:
    """채운 사본의 사람 채점(번호 → 항목 → 값)과 열쇠의 entries. 사본이 없으면 아직 안 채운 것이라 None."""
    checked = root / checked_path(sheet)
    if not checked.exists():
        return None
    ratings = parse_sheet(checked.read_text(encoding="utf-8"), JUDGE_ITEMS)
    key = json.loads((root / sheet).with_suffix(".key.json").read_text(encoding="utf-8"))
    return ratings, key["entries"]


def has_verdict(
    entries: Iterable[Mapping[str, object]],
    judged: Mapping[tuple[str, str], Mapping[str, object]],
) -> bool:
    """열쇠 entries 의 답 중 판정기 판정(verdict)이 있는 것이 하나라도 있나."""
    for entry in entries:
        row = judged.get((str(entry["run"]), str(entry["id"])))
        if row is not None and isinstance(row.get("verdict"), Mapping):
            return True
    return False


def agreements_for(
    ratings: Mapping[int, Mapping[str, bool | None]],
    entries: Sequence[Mapping[str, object]],
    judged: Mapping[tuple[str, str], Mapping[str, object]],
) -> dict[str, Agreement]:
    """검수표 하나의 항목별 사람 · 판정기 일치 표. 사람이 비운 항목 · 판정이 없는 답은 뺀다."""
    pairs: dict[str, list[tuple[bool | None, bool | None]]] = defaultdict(list)
    for entry in entries:
        human = ratings.get(int(str(entry["number"])), {})
        row = judged.get((str(entry["run"]), str(entry["id"])))
        verdict = row["verdict"] if row is not None else None
        for key in JUDGE_ITEMS:
            judge_value = verdict[key] if isinstance(verdict, Mapping) else None
            pairs[key].append((human.get(key), judge_value))
    return {key: agreement(pairs[key]) for key in JUDGE_ITEMS}
