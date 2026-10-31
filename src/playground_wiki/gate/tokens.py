"""형태소 토큰: 관문 규칙은 문자열이 아니라 (형태, 품사) 토큰으로 맞춘다.

부분 문자열로 맞추면 '오늘날' · '지금까지' · '지수함수' · '몇 시간' · '출발점' 이 걸린다.
Kiwi 는 '오늘날' · '지금껏' · '출발점' 을 한 낱말로 읽고, '지금+까지' 는 조사를 떼어 준다.
검색(BM25)의 KiwiAnalyzer 는 품사를 버리고 playground_retrieval.search 를 통째로 불러와서, 여기서는 kiwipiepy 를 바로 쓴다.
"""

from dataclasses import dataclass
from typing import Protocol, Self

from kiwipiepy import Kiwi


@dataclass(frozen=True)
class Token:
    """형태소 하나."""

    form: str
    tag: str
    """세종 품사 태그 (NNG · MAG · JX …)."""


class Tokenizer(Protocol):
    """질문을 형태소 토큰으로 나눈다."""

    def tokenize(self, text: str) -> list[Token]:
        """형태소 단위의 형태와 품사, 문장 순서대로."""
        ...


class KiwiTokenizer:
    """kiwipiepy 형태소 분석기. 모델을 읽는 데 1~2초 · 메모리 약 500MB 라 프로세스에서 한 번 만들어 넘긴다.

    검색(BM25)의 KiwiAnalyzer 와 같은 설정의 Kiwi 를 쓴다 (지금은 둘 다 옵션 · 사용자 사전 없는 Kiwi()).
    사전(add_user_word 등)은 만들 때만 바꾼다 — 분석 중에 바꾸면 같은 인스턴스를 쓰는 다른 스레드가 깨질 수 있다.

    예)
        kiwi = Kiwi()  # 서버 · 평가 스크립트: BM25 의 KiwiAnalyzer 와 같은 인스턴스를 넘긴다
        tokenizer = KiwiTokenizer(kiwi)
        tokenizer = KiwiTokenizer.default()  # 혼자 쓸 때 (시험 · 관문 점검)
    """

    def __init__(self, kiwi: Kiwi) -> None:
        self._kiwi = kiwi

    @classmethod
    def default(cls) -> Self:
        """옵션 · 사용자 사전 없는 새 Kiwi() 로 만든다 (모델을 읽어 1~2초 걸린다)."""
        return cls(Kiwi())

    def tokenize(self, text: str) -> list[Token]:
        """형태소 단위의 형태와 품사 ('출발하는' → 출발/NNG · 하/XSV · 는/ETM)."""
        return [Token(token.form, token.tag) for token in self._kiwi.tokenize(text)]
