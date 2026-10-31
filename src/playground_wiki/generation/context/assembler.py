"""컨텍스트 조립: 후보를 칸별로 골라(중복 제거 · 점수순 · 예산) 배치 순서대로 포장해 대화를 만든다.

근거는 요약하지 않고 고른다 — 요약하면 출처 번호와 원문의 연결이 끊기고, 요약하다 사실이 틀어지면 지어냄이 된다.
"""

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from playground_llm import Message, Role

from playground_wiki.generation.context.packing import TagScrubber, render_data_slot
from playground_wiki.generation.context.slots import Candidate, ContextLayout, Slot
from playground_wiki.generation.context.tokens import TokenCounter

SLOT_SEPARATOR = "\n\n"
"""같은 턴에 이어지는 칸 사이."""

DropReason = Literal["duplicate", "group_limit", "over_budget"]
"""후보를 뺀 이유. duplicate 는 같은 key 의 낮은 점수, group_limit 은 묶음 상한, over_budget 은 남은 예산 부족."""


class BudgetError(ValueError):
    """반드시 넣어야 하는 칸(지시 · 필수 data 칸)이 남은 예산에 들어가지 않는다."""


@dataclass(frozen=True)
class Placed:
    """칸에 넣은 후보 하나."""

    candidate: Candidate
    number: int | None
    """번호 칸이면 1부터의 번호 (답 속 `[n]`), 아니면 None."""


@dataclass(frozen=True)
class Dropped:
    """선별에서 뺀 후보 하나. 로그 · 평가에 쓴다."""

    candidate: Candidate
    reason: DropReason


@dataclass(frozen=True)
class SlotResult:
    """포장을 마친 칸 하나."""

    slot: Slot
    text: str
    """태그로 감싼 글 (지시 칸은 지시 그대로). 비었으면 빈 글 — 대화에 넣지 않는다."""
    tokens: int
    """text 를 조립기의 TokenCounter 로 센 값."""
    placed: tuple[Placed, ...]
    """넣은 후보, 놓인 순서대로. 지시 칸은 비어 있다."""


@dataclass(frozen=True)
class AssembledContext:
    """LLM 호출 한 번에 넘길 컨텍스트."""

    messages: tuple[Message, ...]
    """배치 순서대로 칸을 이은 대화. 같은 턴이 이어지는 칸은 한 메시지로 합친다."""
    slots: tuple[SlotResult, ...]
    """배치 순서대로. 빈 칸도 들어 있다."""
    dropped: tuple[Dropped, ...]
    """뺀 후보 — 칸 우선순위 순서, 칸 안에서는 고른 순서."""
    tokens: int
    """칸 토큰의 합. 턴 표지 · 칸 사이 빈 줄은 세지 않는다 (예산에 여유로 둔다)."""
    budget: int
    """레이아웃의 전체 상한."""

    def slot(self, name: str) -> SlotResult:
        """이름이 name 인 칸의 결과를 돌려준다. 프롬프트 틀에 칸 글을 따로 넣을 때 쓴다.

        Raises:
          KeyError: 그런 칸이 없을 때.
        """
        for result in self.slots:
            if result.slot.name == name:
                return result
        raise KeyError(name)


class ContextAssembler:
    """칸 배치대로 후보를 골라 예산에 맞추고, 태그로 포장해 대화를 만든다.

    칸은 우선순위 순서로 예산을 받는다: 칸마다 min(칸 예산, 남은 전체 예산) 안에서 고른다.
    data 칸은 중복 제거 → 묶음 상한 → 점수 높은 것부터 들어가는 만큼 (안 들어가는 항목은 건너뛰고 다음 항목을 본다).

    예)
        assembler = ContextAssembler(layout, CharTokenCounter())
        context = assembler.assemble({"system": system_text}, candidates)
        await llm.generate(context.messages, spec)
    """

    def __init__(self, layout: ContextLayout, counter: TokenCounter) -> None:
        self._layout = layout
        self._counter = counter
        self._scrubber = TagScrubber(layout.tags)

    def assemble(
        self, instructions: Mapping[str, str], candidates: Iterable[Candidate]
    ) -> AssembledContext:
        """지시와 후보로 컨텍스트를 만든다.

        Args:
          instructions: 지시 칸 이름 → 지시 글 (프롬프트 파일을 채운 것). 빠진 지시 칸은 빈 칸.
          candidates: data 칸에 넣을 후보. 같은 칸 안에서는 받은 순서가 `order="given"` 의 순서다.

        Raises:
          BudgetError: 지시 칸 · 필수 data 칸이 남은 예산에 들어가지 않을 때.
        """
        by_slot = self._group(instructions, candidates)
        remaining = self._layout.budget
        results: dict[str, SlotResult] = {}
        dropped: list[Dropped] = []
        # sorted 는 안정 정렬 — 우선순위가 같으면 배치 순서대로 예산을 받는다
        for slot in sorted(self._layout.slots, key=lambda slot: slot.priority):
            available = min(slot.budget, remaining)
            if slot.kind == "instruction":
                result = self._instruction(slot, instructions.get(slot.name, ""), available)
            else:
                result, slot_dropped = self._select(slot, by_slot.get(slot.name, []), available)
                dropped.extend(slot_dropped)
            remaining -= result.tokens
            results[slot.name] = result
        ordered = tuple(results[slot.name] for slot in self._layout.slots)
        return AssembledContext(
            messages=_messages(ordered),
            slots=ordered,
            dropped=tuple(dropped),
            tokens=sum(result.tokens for result in ordered),
            budget=self._layout.budget,
        )

    def _group(
        self, instructions: Mapping[str, str], candidates: Iterable[Candidate]
    ) -> dict[str, list[Candidate]]:
        kinds = {slot.name: slot.kind for slot in self._layout.slots}
        for name in instructions:
            if kinds.get(name) != "instruction":
                raise ValueError(f"{name!r} is not an instruction slot of this layout")
        by_slot: dict[str, list[Candidate]] = {}
        for candidate in candidates:
            # 바깥에서 온 글이 지시 칸에 들어가는 길을 막는다 — 후보는 태그 안(data 칸)으로만
            if kinds.get(candidate.slot) != "data":
                raise ValueError(f"{candidate.slot!r} is not a data slot of this layout")
            by_slot.setdefault(candidate.slot, []).append(candidate)
        return by_slot

    def _instruction(self, slot: Slot, text: str, available: int) -> SlotResult:
        tokens = self._counter.count(text)
        if tokens > available:
            raise BudgetError(f"{slot.name}: instruction needs {tokens} tokens, {available=}")
        return SlotResult(slot=slot, text=text, tokens=tokens, placed=())

    def _select(
        self, slot: Slot, candidates: Sequence[Candidate], available: int
    ) -> tuple[SlotResult, list[Dropped]]:
        unique, dropped = _dedupe(candidates)
        # (받은 순서, 후보) — given 배치는 받은 순서로 되돌린다. 점수 내림차순, 같으면 받은 순서
        ranked = sorted(enumerate(unique), key=lambda pair: -pair[1].score)
        chosen: list[tuple[int, Candidate]] = []
        groups: Counter[str] = Counter()
        text, tokens = "", 0
        for position, candidate in ranked:
            group = candidate.group
            limit = slot.max_per_group
            if limit is not None and group is not None and groups[group] >= limit:
                dropped.append(Dropped(candidate, "group_limit"))
                continue
            trial = [*chosen, (position, candidate)]
            trial_text = render_data_slot(slot, _arrange(slot, trial), self._scrubber)
            trial_tokens = self._counter.count(trial_text)
            if trial_tokens > available:
                if slot.required:
                    raise BudgetError(
                        f"{slot.name}: required item does not fit ({trial_tokens=}, {available=})"
                    )
                dropped.append(Dropped(candidate, "over_budget"))
                continue
            chosen, text, tokens = trial, trial_text, trial_tokens
            if group is not None:
                groups[group] += 1
        placed = tuple(Placed(candidate, number) for number, candidate in _arrange(slot, chosen))
        return SlotResult(slot=slot, text=text, tokens=tokens, placed=placed), dropped


def _dedupe(candidates: Sequence[Candidate]) -> tuple[list[Candidate], list[Dropped]]:
    """같은 key 는 점수가 가장 높은 하나만 — 자리는 처음 나온 자리. 점수가 같으면 먼저 나온 것."""
    unique: list[Candidate] = []
    index: dict[str, int] = {}
    dropped: list[Dropped] = []
    for candidate in candidates:
        if candidate.key is None:
            unique.append(candidate)
            continue
        seen = index.get(candidate.key)
        if seen is None:
            index[candidate.key] = len(unique)
            unique.append(candidate)
        elif candidate.score > unique[seen].score:
            dropped.append(Dropped(unique[seen], "duplicate"))
            unique[seen] = candidate
        else:
            dropped.append(Dropped(candidate, "duplicate"))
    return unique, dropped


def _arrange(
    slot: Slot, chosen: Sequence[tuple[int, Candidate]]
) -> list[tuple[int | None, Candidate]]:
    """고른 후보를 놓을 순서로 늘어놓고 번호를 붙인다 (chosen 은 점수 순서)."""
    ordered = sorted(chosen, key=lambda pair: pair[0]) if slot.order == "given" else list(chosen)
    if not slot.numbered:
        return [(None, candidate) for _, candidate in ordered]
    return [(number, candidate) for number, (_, candidate) in enumerate(ordered, start=1)]


def _messages(results: Sequence[SlotResult]) -> tuple[Message, ...]:
    """빈 칸은 빼고, 같은 턴이 이어지는 칸은 한 메시지로."""
    turns: list[tuple[Role, list[str]]] = []
    for result in results:
        if not result.text:
            continue
        if turns and turns[-1][0] == result.slot.role:
            turns[-1][1].append(result.text)
        else:
            turns.append((result.slot.role, [result.text]))
    return tuple(Message(role, SLOT_SEPARATOR.join(texts)) for role, texts in turns)
