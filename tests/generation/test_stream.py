import random

import pytest

from playground_wiki.generation import resolve_citations
from playground_wiki.generation.stream import CITATION_HOLD_CHARS, CitationStream

ANSWER = "서울은 수도다 [1][3]. 한양은 [9]옛 이름이다 [2, 7]. 끝 [1]"


def _splits(text: str) -> list[list[str]]:
    """두 조각으로 나누는 모든 경우 + 한 글자씩."""
    return [[text[:i], text[i:]] for i in range(1, len(text))] + [list(text)]


def _stream(pieces: list[str], count: int) -> tuple[str, CitationStream]:
    citations = CitationStream(count)
    out = "".join(citations.feed(piece) for piece in pieces) + citations.finish()
    return out, citations


@pytest.mark.parametrize("count", [0, 2, 3])
def test_citation_stream_matches_resolving_the_whole_answer(count: int):
    whole = resolve_citations(ANSWER, count)
    rng = random.Random(0)
    cuts = [sorted(rng.sample(range(1, len(ANSWER)), 4)) for _ in range(30)]
    splits = _splits(ANSWER) + [
        [ANSWER[a:b] for a, b in zip([0, *cut], [*cut, len(ANSWER)], strict=True)] for cut in cuts
    ]

    for pieces in splits:
        out, citations = _stream(pieces, count)
        assert out == whole.text, pieces
        assert citations.cited == whole.cited
        assert citations.dropped == whole.dropped


def test_citation_stream_releases_plain_text_right_away():
    citations = CitationStream(count=3)

    assert citations.feed("서울은 수도다") == "서울은 수도다"
    assert citations.feed(" [1") == ""  # 덜 닫힌 번호는 붙잡는다
    assert citations.feed("]") == ""  # 닫혀도 뒤에 [3] 이 붙어 올 수 있어 붙잡는다
    assert citations.feed(". 다음") == " [1]. 다음"


def test_citation_hold_has_a_limit():
    citations = CitationStream(count=3)
    unclosed = "[" + "1" * CITATION_HOLD_CHARS

    assert citations.feed(unclosed) == unclosed  # 끝내 닫히지 않는 [ 를 계속 붙잡지 않는다
