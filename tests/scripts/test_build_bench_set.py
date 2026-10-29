"""scripts/build_bench_set.py 의 예비 원본 고르기. KorQuAD · 검색 없이 합성 입력으로 돈다."""

import importlib.util
from itertools import count, islice
from pathlib import Path

from playground_evalkit import KorQuADItem

from playground_wiki.evaluation.splits import split_of

_SCRIPT = Path(__file__).parents[2] / "scripts" / "build_bench_set.py"
_spec = importlib.util.spec_from_file_location("build_bench_set", _SCRIPT)
assert _spec is not None and _spec.loader is not None
bench_set = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench_set)


def _paragraphs(split: str, n: int) -> list[str]:
    """split_of 가 split 을 돌려주는 문단 키 n 개."""
    keys = (f"문서#{index}" for index in count())
    return list(islice((key for key in keys if split_of(key) == split), n))


def _item(item_id: str) -> KorQuADItem:
    return KorQuADItem(
        id=item_id, title="문서", question="질문?", answer="1976년", source_title="문서"
    )


def test_spare_items_skip_taken_paragraphs_and_other_splits():
    taken, free = _paragraphs("bench", 2)
    (eval_paragraph,) = _paragraphs("eval", 1)
    paragraphs = {"b": free, "a": free, "c": taken, "d": eval_paragraph}

    picked = bench_set.spare_items([_item(item_id) for item_id in "dcba"], paragraphs, {taken})

    assert [item.id for item in picked] == ["a", "b"]
