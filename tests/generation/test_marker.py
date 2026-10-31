import pytest

from playground_wiki.generation.marker import MARKER_HOLD_CHARS, MarkerReader, MarkerResult


def _read(pieces: list[str]) -> tuple[MarkerResult, int]:
    """조각을 차례로 넣어 판정 결과와 판정까지 넣은 조각 수. rest 에는 판정 뒤 조각을 body() 로 잇는다."""
    reader = MarkerReader()
    for index, piece in enumerate(pieces, start=1):
        if (result := reader.feed(piece)) is not None:
            rest = result.rest + "".join(reader.body(later) for later in pieces[index:])
            return MarkerResult(result.status, rest), index
    return reader.finish(), len(pieces)


def _splits(text: str) -> list[list[str]]:
    """두 조각으로 나누는 모든 경우 + 한 글자씩."""
    return [[text[:i], text[i:]] for i in range(1, len(text))] + [list(text)]


@pytest.mark.parametrize(
    ("text", "status", "rest"),
    [
        ("찾음\n흥선대원군이 중건했다 [1].", "found", "흥선대원군이 중건했다 [1]."),
        ("찾음.\n답", "found", "답"),
        ("  찾음:\n\n답", "found", "답"),
        ("못 찾음", "not_found", ""),
        ("못찾음\n", "not_found", ""),
        ("못 찾음.\n덧붙인 말", "not_found", "덧붙인 말"),
        ("흥선대원군이 중건했다.", "missing", "흥선대원군이 중건했다."),
        ("찾음직한 답이다", "missing", "찾음직한 답이다"),  # 표지 뒤가 공백이 아니면 표지가 아니다
        ("", "missing", ""),
    ],
)
def test_marker_is_read_the_same_however_the_text_is_split(text: str, status: str, rest: str):
    for pieces in _splits(text) or [[text]]:
        result, _ = _read(pieces)
        assert (result.status, result.rest) == (status, rest), pieces


def test_marker_without_newline_is_decided_as_soon_as_the_answer_starts():
    # 표지 없이 바로 답을 쓰면 첫 조각에서 판정한다 — 끝까지 붙잡지 않는다
    result, fed = _read(["서울", "입니다", "."])

    assert (result.status, fed) == ("missing", 1)


def test_marker_hold_has_a_limit():
    # 표지 앞부분처럼 보이는 공백만 길게 오면 상한에서 멈추고 누락으로 흘린다
    reader = MarkerReader()
    results = [reader.feed(" ") for _ in range(MARKER_HOLD_CHARS)]

    assert results[:-1] == [None] * (MARKER_HOLD_CHARS - 1)
    assert results[-1] == MarkerResult("missing", " " * MARKER_HOLD_CHARS)


def test_marker_reader_refuses_pieces_after_the_decision():
    reader = MarkerReader()
    reader.feed("찾음\n답")

    with pytest.raises(RuntimeError):
        reader.feed("더")


def test_blank_lines_after_the_marker_are_dropped_even_in_later_pieces():
    reader = MarkerReader()

    assert reader.feed("찾음:\n") == MarkerResult("found", "")
    assert [reader.body(piece) for piece in ("\n", "  ", "답 ", "이다")] == ["", "", "답 ", "이다"]
