"""판정기 검증용 사람 채점: 표본 고르기 · 검수표 쓰기 · 채운 검수표 읽기 · 판정기와의 일치율.

검수표에는 판정기의 판정과 문항의 묶음(어느 세트 · 어느 종류)을 싣지 않는다 — 보이면 사람이 따라간다. 문항 번호와
원래 문항을 잇는 열쇠는 스크립트가 따로 저장한다. 사람은 항목마다 '예' · '아니오'를 적고, 비워 두면 그 항목은 일치율에서
빠진다. 표본은 판정을 돌리기 전에 seed 로 고정한다.
"""

import random
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from playground_wiki.evaluation.answer_run import ContextChunk

YES = frozenset({"예", "o", "y", "yes", "true"})
"""검수표에서 '예'로 읽는 값 (소문자로 바꿔 비교)."""
NO = frozenset({"아니오", "아니요", "x", "n", "no", "false"})
"""검수표에서 '아니오'로 읽는 값 (소문자로 바꿔 비교)."""
_ENTRY = re.compile(r"^## (\d+)\s*$")
_RATING = re.compile(r"^- (\w+) \([^)]*\):(.*)$")
_FENCE = re.compile(r"^(`{3,})")
_BACKTICKS = re.compile(r"`+")


@dataclass(frozen=True)
class ReviewEntry:
    """검수표 한 문항 — 사람이 보는 것."""

    number: int
    """검수표 안 번호 (1부터)."""
    question: str
    context: tuple[ContextChunk, ...]
    """답변 경로가 문맥으로 넘긴 청크 (출처 번호 n = context[n - 1])."""
    answer: str


@dataclass(frozen=True)
class Agreement:
    """한 항목에서 사람과 판정기를 짝지은 표 — 둘 다 값이 있는 문항만."""

    both: int
    """둘 다 예."""
    human_only: int
    """사람만 예 (판정기가 놓침)."""
    judge_only: int
    """판정기만 예."""
    neither: int
    """둘 다 아니오."""

    @property
    def total(self) -> int:
        """짝지은 문항 수."""
        return self.both + self.human_only + self.judge_only + self.neither

    @property
    def rate(self) -> float | None:
        """일치율. 문항이 없으면 None."""
        return (self.both + self.neither) / self.total if self.total else None

    @property
    def kappa(self) -> float | None:
        """Cohen's kappa — 우연히 맞을 몫을 뺀 일치도. 문항이 없거나 둘 다 한 값만 냈으면(우연 일치 1) None."""
        if not self.total:
            return None
        human = (self.both + self.human_only) / self.total
        judge = (self.both + self.judge_only) / self.total
        chance = human * judge + (1 - human) * (1 - judge)
        if chance == 1:
            return None
        rate = (self.both + self.neither) / self.total
        return (rate - chance) / (1 - chance)


def stratified_sample(
    keys_by_group: Mapping[str, Sequence[str]], size: int, *, seed: int
) -> list[str]:
    """묶음 크기에 비례해 size 개를 고른다 (최대 나머지 방식). 묶음 안은 seed 로 무작위, 돌려주는 순서도 섞는다.

    Args:
      keys_by_group: 묶음 이름 → 문항 키. 키는 묶음 사이에 겹치지 않는다.
      size: 고를 개수. 전체보다 많으면 전부.
      seed: 난수 시드. 같은 입력 · 시드면 같은 표본.
    """
    total = sum(len(keys) for keys in keys_by_group.values())
    size = min(size, total)
    if not total:
        return []
    groups = sorted(keys_by_group)
    quotas = {group: size * len(keys_by_group[group]) / total for group in groups}
    counts = {group: int(quota) for group, quota in quotas.items()}
    # 남은 자리는 소수 부분이 큰 묶음부터 (같으면 이름순)
    by_remainder = sorted(groups, key=lambda group: (counts[group] - quotas[group], group))
    for group in by_remainder[: size - sum(counts.values())]:
        counts[group] += 1
    rng = random.Random(seed)
    picked: list[str] = []
    for group in groups:
        picked += rng.sample(sorted(keys_by_group[group]), counts[group])
    rng.shuffle(picked)
    return picked


def render_sheet(
    title: str, entries: Sequence[ReviewEntry], items: Mapping[str, tuple[str, str]]
) -> str:
    """사람이 채울 검수표 markdown. 질문 · 문서 · 답은 코드 블록에 넣어 답 안의 제목 · 목록이 검수표 구조와 섞이지 않게 한다.

    Args:
      title: 검수표 제목.
      entries: 문항 (번호 순서대로 싣는다).
      items: 항목 키 → (사람이 읽는 이름, 판정 기준).
    """
    guide = "\n".join(
        f"- **{key}** ({name}): {criterion}" for key, (name, criterion) in items.items()
    )
    parts = [
        f"# {title}\n\n"
        "문항마다 항목 줄의 `:` 뒤에 **예** 또는 **아니오**를 적는다 (O · X 도 된다). 판단하기 어려우면 비워 두고 메모에 "
        "이유를 적는다 — 빈 항목은 일치율에서 빠진다. 아래 기준의 true 는 예, false 는 아니오다. 판정기의 판정과 "
        "문항의 출처는 일부러 싣지 않았다.\n\n"
        f"{guide}\n"
    ]
    parts += [_render_entry(entry, items) for entry in entries]
    return "\n".join(parts)


def parse_sheet(text: str, items: Sequence[str]) -> dict[int, dict[str, bool | None]]:
    """채운 검수표를 문항 번호 → 항목 → 사람 판정(비었으면 None)으로 읽는다. 코드 블록 안은 보지 않는다.

    Raises:
      ValueError: 항목 값을 예 · 아니오로 읽을 수 없거나, 항목 줄이 빠졌거나 두 번 나올 때 — 문항 번호와 항목을 알린다.
    """
    ratings: dict[int, dict[str, bool | None]] = {}
    current: dict[str, bool | None] | None = None
    number = 0
    fence: str | None = None
    for line in text.splitlines():
        if fence is not None:
            fence = None if line.strip() == fence else fence
            continue
        if (opened := _FENCE.match(line)) is not None:
            fence = opened.group(1)
            continue
        if (entry := _ENTRY.match(line)) is not None:
            number = int(entry.group(1))
            current = ratings.setdefault(number, {})
            continue
        rating = _RATING.match(line)
        if current is None or rating is None or rating.group(1) not in items:
            continue
        key = rating.group(1)
        if key in current:
            raise ValueError(f"entry {number}: {key} appears twice")
        current[key] = _read_value(rating.group(2), number=number, key=key)
    for entry_number, values in ratings.items():
        if missing := [key for key in items if key not in values]:
            raise ValueError(f"entry {entry_number}: rating lines missing for {missing}")
    return ratings


def agreement(pairs: Iterable[tuple[bool | None, bool | None]]) -> Agreement:
    """(사람, 판정기) 짝을 센다. 어느 한쪽이라도 None(비움 · 판정 실패)이면 뺀다."""
    both = human_only = judge_only = neither = 0
    for human, judge in pairs:
        if human is None or judge is None:
            continue
        both += human and judge
        human_only += human and not judge
        judge_only += judge and not human
        neither += not human and not judge
    return Agreement(both=both, human_only=human_only, judge_only=judge_only, neither=neither)


def _render_entry(entry: ReviewEntry, items: Mapping[str, tuple[str, str]]) -> str:
    documents = (
        "\n".join(
            f"[{number}] (리랭커 점수 {chunk.score:.3f})\n\n{_fenced(chunk.text)}"
            for number, chunk in enumerate(entry.context, start=1)
        )
        if entry.context
        else "(받은 문서 없음)\n"
    )
    ratings = "".join(f"- {key} ({name}):\n" for key, (name, _) in items.items())
    return (
        f"## {entry.number}\n\n"
        f"질문:\n\n{_fenced(entry.question)}\n"
        f"받은 문서 {len(entry.context)}개:\n\n{documents}\n"
        f"답:\n\n{_fenced(entry.answer)}\n"
        f"{ratings}- 메모:\n"
    )


def _fenced(text: str) -> str:
    # 본문에 있는 가장 긴 backtick 줄보다 길게 감싸야 본문의 ``` 가 블록을 닫지 않는다
    longest = max((len(run) for run in _BACKTICKS.findall(text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}text\n{text}\n{fence}\n"


def _read_value(raw: str, *, number: int, key: str) -> bool | None:
    value = raw.strip().lower()
    if not value:
        return None
    if value in YES:
        return True
    if value in NO:
        return False
    raise ValueError(f"entry {number}: {key} must be 예 or 아니오, got {raw.strip()!r}")
