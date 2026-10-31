import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from playground_wiki.evaluation import paths
from playground_wiki.evaluation.files import (
    append_jsonl,
    document_texts,
    read_jsonl,
    write_jsonl,
)

ROWS = [
    {"id": "a", "question": "경복궁을 중건한 사람은?", "hits": [{"score": 0.5, "title": "경복궁"}]},
    {"id": "b", "text": '줄바꿈\n과 따옴표 " · 역슬래시 \\ · 탭\t', "none": None, "flag": True},
    {"id": "c", "emoji": "😀", "nested": {"키": ["값", 1, 2.5]}},
]


def _old_write_jsonl(path: Path, rows: list[dict]) -> None:
    # 스크립트 8곳에 복사돼 있던 본문 그대로 (build_bench_set · eval_answer · judge_partial …)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _old_append_jsonl(path: Path, row: dict) -> None:
    # build_longtail_set · eval_answer · judge_partial 의 본문
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _old_append_jsonl_platform(path: Path, row: dict) -> None:
    # compare_prompts 의 본문 — newline 인자가 없다
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_write_jsonl_writes_the_same_bytes_as_the_old_script_copies(tmp_path: Path):
    write_jsonl(tmp_path / "new" / "rows.jsonl", ROWS)
    _old_write_jsonl(tmp_path / "old" / "rows.jsonl", ROWS)

    new = (tmp_path / "new" / "rows.jsonl").read_bytes()
    assert new == (tmp_path / "old" / "rows.jsonl").read_bytes()
    assert b"\r\n" not in new


def test_append_jsonl_writes_the_same_bytes_as_the_old_script_copies(tmp_path: Path):
    for row in ROWS:
        append_jsonl(tmp_path / "new.jsonl", row)
        _old_append_jsonl(tmp_path / "old.jsonl", row)
        append_jsonl(tmp_path / "new-platform.jsonl", row, newline=None)
        _old_append_jsonl_platform(tmp_path / "old-platform.jsonl", row)

    assert (tmp_path / "new.jsonl").read_bytes() == (tmp_path / "old.jsonl").read_bytes()
    assert (tmp_path / "new-platform.jsonl").read_bytes() == (
        tmp_path / "old-platform.jsonl"
    ).read_bytes()


def test_write_then_read_jsonl_round_trips_and_skips_blank_lines(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, ROWS)
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    assert read_jsonl(path) == ROWS


def test_document_texts_keeps_only_asked_titles_by_normalized_title(tmp_path: Path):
    table = pa.table({"title": ["경복궁", "Seoul_Station", "조선"], "text": ["궁", "역", "왕조"]})
    pq.write_table(table, tmp_path / "docs-0000.parquet")
    pq.write_table(pa.table({"title": ["조선"], "text": ["나중"]}), tmp_path / "docs-0001.parquet")

    assert document_texts(tmp_path, {"경복궁", "Seoul Station", "조선"}) == {
        "경복궁": "궁",
        "Seoul Station": "역",
        "조선": "나중",
    }


def test_paths_are_relative_to_the_repo_root():
    values = [value for name, value in vars(paths).items() if name.isupper()]

    assert values
    assert all(not Path(value).is_absolute() for value in values)
