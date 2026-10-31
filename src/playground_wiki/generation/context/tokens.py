"""토큰 세기 계약과 기본 어림 (한국어 글자 수 기준)."""

import math
from typing import Protocol

TOKENS_PER_CHAR = 2 / 3
"""한국어 글자 수 어림 — 300자 ≈ 200토큰 (docs/context-design.md §6). 실제 값은 응답의 프롬프트 토큰 수로 확인한다."""


class TokenCounter(Protocol):
    """조립기가 칸 예산을 잴 때 부르는 계약. 실제 토크나이저는 쓰는 쪽이 넘긴다."""

    def count(self, text: str) -> int:
        """`text` 의 토큰 수를 돌려준다."""
        ...


class CharTokenCounter:
    """글자 수에 비율을 곱해 올림한 어림. 미리 세는 기능이 없는 서버(Ollama)용 기본값.

    예)
        CharTokenCounter().count("가" * 300)  # 200
    """

    def __init__(self, *, tokens_per_char: float = TOKENS_PER_CHAR) -> None:
        if tokens_per_char <= 0:
            raise ValueError(f"tokens_per_char must be > 0, got {tokens_per_char=}")
        self._tokens_per_char = tokens_per_char

    def count(self, text: str) -> int:
        """글자 수 × 비율을 올림한 토큰 수를 돌려준다. 빈 글은 0."""
        return math.ceil(len(text) * self._tokens_per_char)
