import json
from collections import Counter
from pathlib import Path
from typing import get_args

from playground_evalkit import KorQuADItem

from playground_wiki.evaluation import Split, korquad_paragraphs, sample_per_paragraph, split_of


def _item(qid: str) -> KorQuADItem:
    return KorQuADItem(id=qid, title="t", question=f"q{qid}", answer="a", source_title="t")


def test_paragraphs_come_from_file_structure(tmp_path: Path):
    data = {
        "data": [
            {
                "title": "서울",
                "paragraphs": [
                    {"qas": [{"id": "9-0-0"}, {"id": "7-0-1"}]},  # id 앞부분이 달라도 같은 문단
                    {"qas": [{"id": "9-1-0"}]},
                    {"qas": []},  # 질문 없는 문단은 키가 없다
                ],
            }
        ]
    }
    path = tmp_path / "dev.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    assert korquad_paragraphs(path) == {"9-0-0": "서울#0", "7-0-1": "서울#0", "9-1-0": "서울#1"}


def test_split_is_stable_and_roughly_20_40_40():
    keys = [f"문서{i}#{j}" for i in range(200) for j in range(5)]

    counts = Counter(split_of(key) for key in keys)

    assert [split_of(key) for key in keys] == [split_of(key) for key in keys]
    assert set(counts) == set(get_args(Split))
    assert 150 < counts["calibrate"] < 250
    assert 330 < counts["bench"] < 470
    assert 330 < counts["eval"] < 470


def test_sample_takes_one_question_per_paragraph_of_the_split():
    keys = [f"문서{i}#0" for i in range(60)]
    paragraphs = {f"{n}-{k}": key for n, key in enumerate(keys) for k in range(3)}
    items = [_item(qid) for qid in paragraphs]

    picked = sample_per_paragraph(items, paragraphs, "calibrate", seed=0)

    chosen = [paragraphs[item.id] for item in picked]
    assert len(chosen) == len(set(chosen))  # 문단마다 하나
    assert set(chosen) == {key for key in keys if split_of(key) == "calibrate"}
    assert chosen == sorted(chosen)


def test_sample_is_repeatable_with_the_same_seed():
    paragraphs = {f"{n}-{k}": f"문서{n}#0" for n in range(60) for k in range(3)}
    items = [_item(qid) for qid in paragraphs]

    first = sample_per_paragraph(items, paragraphs, "bench", seed=7)
    again = sample_per_paragraph(list(reversed(items)), paragraphs, "bench", seed=7)

    assert [item.id for item in first] == [item.id for item in again]  # 입력 순서와 상관없이
