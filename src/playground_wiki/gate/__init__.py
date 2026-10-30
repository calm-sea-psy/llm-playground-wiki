"""질문 관문: 잡담 · 실시간 · 자해 질문을 검색 전에 형태소 어휘 규칙으로 걸러 고정 답을 준다."""

from playground_wiki.gate.rules import (
    CRISIS_LINE,
    KINDS,
    REPLIES,
    SMALLTALK,
    GateMatch,
    Kind,
    QueryGate,
    Rule,
    crisis_match,
    fixed_reply,
    normalize,
    realtime_match,
    smalltalk_table,
)
from playground_wiki.gate.tokens import KiwiTokenizer, Token, Tokenizer

__all__ = [
    "CRISIS_LINE",
    "KINDS",
    "REPLIES",
    "SMALLTALK",
    "GateMatch",
    "Kind",
    "KiwiTokenizer",
    "QueryGate",
    "Rule",
    "Token",
    "Tokenizer",
    "crisis_match",
    "fixed_reply",
    "normalize",
    "realtime_match",
    "smalltalk_table",
]
