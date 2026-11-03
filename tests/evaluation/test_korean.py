import pytest

from playground_wiki.evaluation.korean import RIEUL_FINAL, final_consonant


@pytest.mark.parametrize(
    ("char", "expected"),
    [("가", 0), ("각", 1), ("갈", RIEUL_FINAL), ("A", None), ("1", None)],
)
def test_final_consonant(char: str, expected: int | None):
    assert final_consonant(char) == expected
