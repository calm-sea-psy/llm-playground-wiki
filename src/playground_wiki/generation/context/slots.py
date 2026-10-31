"""컨텍스트 칸(slot) 정의: 칸 하나의 역할 · 예산 · 우선순위 · 태그, 칸 배치, 칸에 넣을 후보."""

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, get_args

from playground_llm import Role

SlotKind = Literal["instruction", "data"]
"""instruction 은 우리가 쓴 지시(그대로 넣고 태그로 감싸지 않음), data 는 바깥에서 온 글(태그 안에만)."""
SLOT_KINDS: tuple[str, ...] = get_args(SlotKind)
ItemOrder = Literal["score", "given"]
"""칸 안 항목의 배치 순서. score 는 점수 내림차순, given 은 후보를 받은 순서 (뒤에 붙여 나가 앞부분이 그대로)."""
ITEM_ORDERS: tuple[str, ...] = get_args(ItemOrder)
NUMBER_ATTRIBUTE = "id"
"""번호 칸(`Slot.numbered`)이 항목 태그에 붙이는 번호 속성. 후보가 직접 쓸 수 없다."""
# 태그 · 속성 이름. 소문자로 시작하는 영문 소문자 · 숫자 · 밑줄만 — 본문 속 흉내를 정규식 하나로 지울 수 있게
_NAME = re.compile(r"[a-z][a-z0-9_]*")


@dataclass(frozen=True)
class Slot:
    """칸 하나: 무엇이 · 어느 턴에 · 얼마나 들어가는지.

    예)
        Slot("documents", "data", budget=2400, priority=1, item_tag="document", numbered=True)
    """

    name: str
    kind: SlotKind
    budget: int
    """이 칸의 토큰 상한 (태그 포함)."""
    role: Role = "user"
    """이 칸이 들어갈 턴. data 칸은 user 만 — 데이터는 지시(system) 턴에 섞이지 않는다."""
    priority: int = 0
    """작을수록 먼저 전체 예산을 받는다. 같으면 배치 순서대로."""
    tag: str | None = None
    """data 칸 전체를 감싸는 태그 (예: `documents`). 없으면 항목 태그만."""
    item_tag: str | None = None
    """data 칸의 항목마다 감싸는 태그 (예: `document`). 없으면 항목을 빈 줄로만 가른다."""
    numbered: bool = False
    """항목 태그에 `id="1"` 부터 번호를 붙인다 — 답 속 출처 번호 `[n]` 과 같은 번호."""
    order: ItemOrder = "score"
    max_per_group: int | None = None
    """같은 묶음(`Candidate.group`, 예: 문서 id)에서 넣을 최대 항목 수. None 이면 제한 없음."""
    required: bool = False
    """data 칸: 후보를 하나라도 넣지 못하면 실패한다 (질문 칸). instruction 칸은 늘 필수."""

    def __post_init__(self) -> None:
        if not _NAME.fullmatch(self.name):
            raise ValueError(f"slot name must match {_NAME.pattern}, got {self.name=}")
        if self.kind not in SLOT_KINDS:
            raise ValueError(f"kind must be one of {SLOT_KINDS}, got {self.kind=}")
        if self.order not in ITEM_ORDERS:
            raise ValueError(f"order must be one of {ITEM_ORDERS}, got {self.order=}")
        if self.role == "assistant":
            raise ValueError(f"{self.name}: slots go to system or user turns, got {self.role=}")
        if self.budget <= 0:
            raise ValueError(f"{self.name}: budget must be > 0, got {self.budget=}")
        if self.max_per_group is not None and self.max_per_group < 1:
            raise ValueError(f"{self.name}: max_per_group must be >= 1, got {self.max_per_group=}")
        for tag in (self.tag, self.item_tag):
            if tag is not None and not _NAME.fullmatch(tag):
                raise ValueError(f"{self.name}: tag must match {_NAME.pattern}, got {tag=}")
        if self.kind == "data":
            self._check_data()
        else:
            self._check_instruction()

    def _check_data(self) -> None:
        if self.role != "user":
            raise ValueError(f"{self.name}: data slots must go to the user turn, got {self.role=}")
        if self.tag is None and self.item_tag is None:
            raise ValueError(f"{self.name}: data slots need tag or item_tag")
        if self.numbered and self.item_tag is None:
            raise ValueError(f"{self.name}: numbered slots need item_tag")

    def _check_instruction(self) -> None:
        packing = (self.tag, self.item_tag, self.max_per_group)
        if any(value is not None for value in packing) or self.numbered or self.required:
            raise ValueError(
                f"{self.name}: instruction slots take no tag, item_tag, numbered, "
                "max_per_group or required"
            )


@dataclass(frozen=True)
class ContextLayout:
    """LLM 호출 한 번의 칸 배치 · 전체 예산."""

    slots: tuple[Slot, ...]
    """배치 순서 — 앞은 거의 안 바뀌는 것, 뒤는 매번 바뀌는 것."""
    budget: int
    """모든 칸을 합친 토큰 상한. 모델 문맥 길이에서 답 길이 · 턴 표지 여유를 뺀 값."""
    reserved_tags: frozenset[str] = frozenset()
    """칸 밖(프롬프트 틀)이 쓰는 태그. 칸 태그와 함께 본문에서 흉내를 지운다 (예: 틀의 `<documents>`)."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "slots", tuple(self.slots))
        object.__setattr__(self, "reserved_tags", frozenset(self.reserved_tags))
        if not self.slots:
            raise ValueError("slots must not be empty")
        if self.budget <= 0:
            raise ValueError(f"budget must be > 0, got {self.budget=}")
        names = [slot.name for slot in self.slots]
        if len(set(names)) != len(names):
            raise ValueError(f"slot names must be unique, got {names}")
        for tag in self.reserved_tags:
            if not _NAME.fullmatch(tag):
                raise ValueError(f"reserved tag must match {_NAME.pattern}, got {tag=}")

    @property
    def tags(self) -> frozenset[str]:
        """본문에서 흉내를 지울 태그 이름 전부 (칸 · 항목 태그 + reserved_tags)."""
        tags = set(self.reserved_tags)
        for slot in self.slots:
            tags.update(tag for tag in (slot.tag, slot.item_tag) if tag is not None)
        return frozenset(tags)

    def slot(self, name: str) -> Slot:
        """이름이 name 인 칸을 돌려준다.

        Raises:
          KeyError: 그런 칸이 없을 때.
        """
        for slot in self.slots:
            if slot.name == name:
                return slot
        raise KeyError(name)


@dataclass(frozen=True)
class Candidate:
    """data 칸에 넣을 후보 하나 (검색 결과 한 건 · 도구 결과 · 기억 · 에이전트 중간 산출물)."""

    slot: str
    """넣을 칸 이름. data 칸만 — 후보는 지시 칸에 들어갈 수 없다."""
    text: str
    """본문. 요약하지 않은 원문 그대로 — 태그 흉내는 포장할 때 지운다."""
    score: float = 0.0
    """선별 점수 (예: 리랭커 점수). 높을수록 먼저 고른다."""
    key: str | None = None
    """중복 판단 키 (예: 청크 id). 같은 칸에서 key 가 같으면 점수가 높은 하나만 남긴다. None 이면 비교하지 않는다."""
    group: str | None = None
    """묶음 (예: 문서 id). 칸의 `max_per_group` 에 쓴다."""
    # 해시에서는 뺀다 (매핑은 해시가 안 된다). 같은지 비교에는 들어간다
    attributes: Mapping[str, str] = field(default_factory=dict, hash=False)
    """항목 태그에 붙일 속성 (예: `{"title": "경복궁"}`), 넣은 순서대로. `id` 는 번호 칸이 붙인다."""

    def __post_init__(self) -> None:
        for name in self.attributes:
            if not _NAME.fullmatch(name):
                raise ValueError(f"attribute name must match {_NAME.pattern}, got {name=}")
        if NUMBER_ATTRIBUTE in self.attributes:
            raise ValueError(f"attribute {NUMBER_ATTRIBUTE!r} is reserved for numbered slots")
        # 만든 쪽이 넘긴 dict 를 나중에 고쳐도 후보가 바뀌지 않게 읽기 전용 복사본으로
        object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))
