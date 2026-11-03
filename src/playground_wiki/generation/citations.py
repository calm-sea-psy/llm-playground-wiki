"""답의 출처 번호를 실제 문서로 잇는다 — 모델은 `[1]` 처럼 번호만 쓰고, 제목 · 링크는 서버가 붙인다 (ADR-026).

모델이 제목을 직접 쓰면 오타('랩톱' → '[랩력]')와 받지 않은 문서를 출처로 지어내는 일이 같은 경로로 생긴다.
번호만 받으면 둘 다 구조로 사라지고, 범위를 벗어난 번호는 떼어 내고 센다.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from urllib.parse import quote

from playground_retrieval.core import Hit

WIKI_URL = "https://ko.wikipedia.org/wiki/"
# 앞 공백 + 붙어 있는 번호 묶음 하나 ([1] · [1][3] · [1, 3]) — 숫자와 쉼표만 든 대괄호.
# 묶음째 다뤄야 [0][1] 에서 [0] 만 뗄 때 앞 공백이 남는다. [문서 제목] 같은 다른 대괄호는 건드리지 않는다
_CITATION = re.compile(r"(\s*)((?:\[\d+(?:\s*,\s*\d+)*\])+)")
_NUMBER = re.compile(r"\d+")


@dataclass(frozen=True)
class Source:
    """문맥으로 넘긴 문서 하나. number 가 답 속 `[n]` 과 `<document id>` 의 번호다."""

    number: int
    title: str
    section: str
    """문서 첫머리면 빈 문자열."""
    url: str
    score: float
    """리랭커 점수 (0~1)."""


@dataclass(frozen=True)
class CitedAnswer:
    """출처 번호를 정리한 답."""

    text: str
    """범위 밖 번호를 뗀 답. 남은 번호는 `[1][3]` 꼴로 맞춘다."""
    cited: tuple[int, ...]
    """답에 나온 번호 (처음 나온 순서, 중복 없이)."""
    dropped: tuple[int, ...]
    """범위를 벗어나 뗀 번호 (나온 순서 그대로). 로그 · 평가(범위 밖 번호 비율)에 쓴다."""


def wiki_url(title: str) -> str:
    """문서 제목의 한국어 위키백과 주소. 공백은 밑줄로 바꾸고 나머지는 퍼센트 인코딩."""
    return WIKI_URL + quote(title.replace(" ", "_"))


def sources(hits: Sequence[Hit]) -> tuple[Source, ...]:
    """문맥으로 넘긴 검색 결과를 번호 붙은 출처로 바꾼다 (`ask_documents` 와 같은 번호, 1부터)."""
    return tuple(
        Source(
            number=number,
            title=hit.chunk.title,
            section=hit.chunk.section,
            url=wiki_url(hit.chunk.title),
            score=hit.score,
        )
        for number, hit in enumerate(hits, start=1)
    )


def resolve_citations(text: str, count: int) -> CitedAnswer:
    """답 속 출처 번호를 1..count 로 거른다. 범위 밖 번호는 (앞 공백과 함께) 떼고, 남은 건 `[1][3]` 꼴로.

    Args:
      text: 모델이 쓴 답.
      count: 문맥으로 넘긴 문서 수 (`len(sources(hits))`). 0 이면 모든 번호를 뗀다.
    """
    cited: list[int] = []
    dropped: list[int] = []

    def replace(match: re.Match[str]) -> str:
        kept = []
        for part in _NUMBER.findall(match.group(2)):
            number = int(part)
            if 1 <= number <= count:
                kept.append(number)
                if number not in cited:
                    cited.append(number)
            else:
                dropped.append(number)
        if not kept:
            return ""
        return match.group(1) + "".join(f"[{number}]" for number in kept)

    resolved = _CITATION.sub(replace, text)
    return CitedAnswer(text=resolved, cited=tuple(cited), dropped=tuple(dropped))
