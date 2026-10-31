"""답을 조각으로 흘릴 때 잠깐 붙잡는 것: grounded@v2 첫 줄 표지 · 출처 번호 (ADR-024 · ADR-026).

둘 다 조각 경계에서 잘린 것을 붙잡아 판정하되 붙잡는 길이에 상한을 둔다 — 상한이 없으면 표지를 빼먹은 답이나
끝내 닫히지 않은 `[` 가 생성이 끝날 때까지 화면에 나가지 않는다 (사실상 스트리밍이 아니게 된다).
"""

import re
from dataclasses import dataclass
from typing import Literal

from playground_wiki.generation.citations import resolve_citations

MarkerStatus = Literal["found", "not_found", "missing"]
MARKER_HOLD_CHARS = 10
"""표지를 판정하려고 붙잡는 글자 수 상한. 넘으면 표지 누락으로 보고 흘린다."""
CITATION_HOLD_CHARS = 16
"""답 끝의 출처 번호 묶음을 붙잡는 글자 수 상한 (`[1][2][3]` 쯤). 넘으면 그냥 흘린다."""
# 첫머리 표지: "찾음" · "못 찾음" · "못찾음", 뒤에 마침표 · 쌍점 하나까지. 표지 뒤가 공백 · 줄바꿈이어야 표지다
# ("찾음직한" 은 표지가 아니다). 조각 중간에서는 뒤 글자가 와야 판정할 수 있다
_MARKER = re.compile(r"\s*(못\s?찾음|찾음)[.:]?(?=\s|$)")
_MARKER_FORMS = ("못 찾음", "못찾음", "찾음")
# 답 끝에서 아직 완성되지 않았을 수 있는 출처 번호 꼬리: 공백 · 닫힌 묶음들 · 덜 닫힌 `[숫자`.
# 닫힌 `[1]` 도 붙잡는다 — 바로 뒤에 `[3]` 이 붙어 오면 묶음째 정리해야 범위 밖 번호를 앞 공백과 함께 뗄 수 있다
_CITATION_TAIL = re.compile(r"\s*(?:\[\d+(?:\s*,\s*\d+)*\])*\s*(?:\[[\d,\s]*)?$")


@dataclass(frozen=True)
class MarkerResult:
    """첫 줄 표지를 읽은 결과."""

    status: MarkerStatus
    """found · not_found = 표지가 있었다, missing = 표지 없이 답이 시작됐다 (찾음으로 보고 흘린다 — ADR-024)."""
    rest: str
    """표지 뒤에 이어 붙어 온 답 (앞 공백 · 줄바꿈은 뗀다). missing 이면 붙잡아 둔 글 전부."""


class MarkerReader:
    """grounded@v2 출력 첫머리의 표지를 붙잡아 읽는다. 판정 뒤의 조각은 `body` 로 넘긴다.

    예)
        reader = MarkerReader()
        for piece in pieces:
            if reader.result is not None:
                send(reader.body(piece))
            elif (result := reader.feed(piece)) is not None:
                if result.status == "not_found":
                    break  # 생성을 끊고 partial 로
                send(result.rest)
        result = reader.finish()  # 판정 전에 생성이 끝났으면 여기서
    """

    def __init__(self, *, hold_chars: int = MARKER_HOLD_CHARS) -> None:
        self._hold = hold_chars
        self._buffer = ""
        self._leading = False
        self.result: MarkerResult | None = None

    def feed(self, piece: str) -> MarkerResult | None:
        """판정 전 조각 하나를 붙인다. 아직 판정할 수 없으면 None."""
        if self.result is not None:
            raise RuntimeError("marker already read — pass the following pieces to body()")
        self._buffer += piece
        text = self._buffer
        match = _MARKER.match(text)
        if match is None:
            if len(text) < self._hold and _could_be_marker(text):
                return None  # "못" · "못 찾" 처럼 아직 표지의 앞부분일 수 있다
            return self._set("missing", text)
        if match.end() == len(text) and len(text) < self._hold:
            return (
                None  # 표지 뒤에 무엇이 올지 아직 모른다 ("찾음" 다음에 "." · 줄바꿈 · 다른 글자)
            )
        return self._marked(match)

    def finish(self) -> MarkerResult:
        """생성이 끝났을 때 — 붙잡아 둔 것으로 판정한다."""
        if self.result is not None:
            return self.result
        match = _MARKER.match(self._buffer)
        return self._set("missing", self._buffer) if match is None else self._marked(match)

    def body(self, piece: str) -> str:
        """판정 뒤에 오는 조각. 표지 바로 뒤의 빈 줄 · 공백은 답 첫 글자가 나올 때까지 뗀다."""
        if self.result is None:
            raise RuntimeError("marker not read yet — feed() first")
        if self._leading:
            piece = piece.lstrip()
            self._leading = not piece
        return piece

    def _marked(self, match: re.Match[str]) -> MarkerResult:
        status: MarkerStatus = "not_found" if match.group(1).startswith("못") else "found"
        rest = self._buffer[match.end() :].lstrip()
        self._leading = not rest
        return self._set(status, rest)

    def _set(self, status: MarkerStatus, rest: str) -> MarkerResult:
        self.result = MarkerResult(status, rest)
        return self.result


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


def read_marker(text: str) -> MarkerResult:
    """스트리밍이 아닌 답(다 받은 글)의 첫 줄 표지 — 조각으로 받은 것과 같은 판정이다."""
    reader = MarkerReader(hold_chars=len(text) + 1)
    return reader.feed(text) or reader.finish()


def _could_be_marker(text: str) -> bool:
    """붙잡아 둔 글이 아직 표지의 앞부분일 수 있나 ("못" · "못 찾" · 공백만)."""
    head = text.lstrip()
    return any(form.startswith(head) for form in _MARKER_FORMS)
