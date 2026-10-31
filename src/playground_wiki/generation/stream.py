"""답을 조각으로 흘리며 출처 번호를 거른다 — 조각 경계에서 잘린 번호 묶음은 잠깐 붙잡는다 (ADR-026).

붙잡는 길이에 상한을 둔다 — 상한이 없으면 끝내 닫히지 않은 `[` 가 생성이 끝날 때까지 화면에 나가지 않는다
(사실상 스트리밍이 아니게 된다). 첫 줄 표지를 붙잡는 쪽은 `generation.marker`.
"""

import re

from playground_wiki.generation.citations import resolve_citations

CITATION_HOLD_CHARS = 16
"""답 끝의 출처 번호 묶음을 붙잡는 글자 수 상한 (`[1][2][3]` 쯤). 넘으면 그냥 흘린다."""
# 답 끝에서 아직 완성되지 않았을 수 있는 출처 번호 꼬리: 공백 · 닫힌 묶음들 · 덜 닫힌 `[숫자`.
# 닫힌 `[1]` 도 붙잡는다 — 바로 뒤에 `[3]` 이 붙어 오면 묶음째 정리해야 범위 밖 번호를 앞 공백과 함께 뗄 수 있다
_CITATION_TAIL = re.compile(r"\s*(?:\[\d+(?:\s*,\s*\d+)*\])*\s*(?:\[[\d,\s]*)?$")


class CitationStream:
    """답 조각에서 출처 번호를 1..count 로 거르며 흘린다 — 끝에 한 번에 `resolve_citations` 한 것과 같은 글이 나온다.

    예)
        citations = CitationStream(count=len(context))
        for piece in pieces:
            send(citations.feed(piece))
        send(citations.finish())
        citations.cited, citations.dropped
    """

    def __init__(self, count: int, *, hold_chars: int = CITATION_HOLD_CHARS) -> None:
        self._count = count
        self._hold = hold_chars
        self._buffer = ""
        self._cited: list[int] = []
        self._dropped: list[int] = []

    @property
    def cited(self) -> tuple[int, ...]:
        """지금까지 답에 나온 번호 (처음 나온 순서, 중복 없이)."""
        return tuple(self._cited)

    @property
    def dropped(self) -> tuple[int, ...]:
        """지금까지 범위를 벗어나 뗀 번호 (나온 순서 그대로)."""
        return tuple(self._dropped)

    def feed(self, piece: str) -> str:
        """조각 하나를 받아 지금 내보내도 되는 글을 돌려준다 (빈 문자열일 수 있다)."""
        self._buffer += piece
        tail = self._buffer[_CITATION_TAIL.search(self._buffer).start() :]
        if len(tail) > self._hold:
            tail = ""  # 끝내 닫히지 않는 `[` 를 계속 붙잡지 않는다
        head = self._buffer[: len(self._buffer) - len(tail)]
        self._buffer = tail
        return self._resolve(head)

    def finish(self) -> str:
        """생성이 끝났을 때 — 붙잡아 둔 꼬리를 정리해 내보낸다."""
        head, self._buffer = self._buffer, ""
        return self._resolve(head)

    def _resolve(self, text: str) -> str:
        if not text:
            return ""
        answer = resolve_citations(text, self._count)
        self._cited += [number for number in answer.cited if number not in self._cited]
        self._dropped += answer.dropped
        return answer.text
