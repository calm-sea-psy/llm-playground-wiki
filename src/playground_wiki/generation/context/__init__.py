"""컨텍스트 엔진: 칸 정의 · 예산 · 선별 · 배치 · 포장 · 단기 메모리 · 후보 출처 약속 (docs/context-design.md).

위키 코드를 import 하지 않는다 (표준 라이브러리 · playground_llm · 이 패키지만, tests/test_layers.py 가 확인).
소설 프로젝트가 두 번째로 쓸 때 playground_llm 으로 옮긴다 — 위키의 칸 정의는 `generation/context_layouts.py`.
"""

from playground_wiki.generation.context.assembler import (
    SLOT_SEPARATOR,
    AssembledContext,
    BudgetError,
    ContextAssembler,
    Dropped,
    DropReason,
    Placed,
    SlotResult,
)
from playground_wiki.generation.context.memory import Evidence, WorkingMemory, memory_candidates
from playground_wiki.generation.context.packing import ITEM_SEPARATOR, TagScrubber
from playground_wiki.generation.context.slots import (
    ITEM_ORDERS,
    NUMBER_ATTRIBUTE,
    SLOT_KINDS,
    Candidate,
    ContextLayout,
    ItemOrder,
    Slot,
    SlotKind,
)
from playground_wiki.generation.context.sources import CandidateSource, collect_candidates
from playground_wiki.generation.context.tokens import (
    TOKENS_PER_CHAR,
    CharTokenCounter,
    TokenCounter,
)

__all__ = [
    "ITEM_ORDERS",
    "ITEM_SEPARATOR",
    "NUMBER_ATTRIBUTE",
    "SLOT_KINDS",
    "SLOT_SEPARATOR",
    "TOKENS_PER_CHAR",
    "AssembledContext",
    "BudgetError",
    "Candidate",
    "CandidateSource",
    "CharTokenCounter",
    "ContextAssembler",
    "ContextLayout",
    "DropReason",
    "Dropped",
    "Evidence",
    "ItemOrder",
    "Placed",
    "Slot",
    "SlotKind",
    "SlotResult",
    "TagScrubber",
    "TokenCounter",
    "WorkingMemory",
    "collect_candidates",
    "memory_candidates",
]
