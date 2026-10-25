"""scripts/build_bench_paraphrase.py 의 짝짓기 · 줄 바꾸기 · 검수표 읽기 · 파일 쓰기. KorQuAD 없이 합성 입력으로 돈다."""

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[2] / "scripts" / "build_bench_paraphrase.py"
_spec = importlib.util.spec_from_file_location("build_bench_paraphrase", _SCRIPT)
assert _spec is not None and _spec.loader is not None
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)

_REVIEW_HEAD = [
    "# 패러프레이즈 확인",
    "",
    "- 표본마다 25개를 seed 0 로 골랐다",
    "",
    "| 번호 | 표본 | id | 원문 | 패러프레이즈 | 정답 | 자동 점검 | 정답 같음 | 정답 안 샘 |",
    "|---|---|---|---|---|---|---|---|---|",
]


def _item(item_id, question, kind="answerable"):
    return {
        "id": f"korquad:{item_id}",
        "kind": kind,
        "question": question,
        "answers": ["정답"],
        "gold_title": "문서",
        "context": [
            {
                "rank": rank,
                "chunk_id": f"{item_id}-{rank}",
                "title": "문서",
                "section": "",
                "text": f"본문 {rank}",
                "score": 1.0 - rank / 10,
            }
            for rank in range(1, 6)
        ],
        "strata": {"backlinks": 1, "answer_shape": "other", "answer_rank": 1},
    }


def _line(item_id, question, paraphrase, *, sample="f4-answerable", problems=()):
    return {
        "sample": sample,
        "id": f"korquad:{item_id}" if sample == "f4-answerable" else item_id,
        "title": "문서",
        "question": question,
        "answers": ["정답"],
        "paraphrase": paraphrase,
        "model": "claude-opus-5-5",
        "prompt_version": "paraphrase-v1",
        "attempts": 1,
        "problems": list(problems),
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def _review_line(number, sample, item_id, paraphrase, same="O", leak="O"):
    return (
        f"| {number} | {sample} | {item_id} | 원문 | {paraphrase} | 정답 | 통과 | {same} | {leak} |"
    )


def _write_review(path, table, notes=()):
    lines = [*_REVIEW_HEAD, *table, "", "## 비고 (정답은 그대로지만 표현이 달라진 문항)", ""]
    lines += [f"- {note}" for note in notes]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def test_paraphrase_row_swaps_question_and_keeps_context():
    source = _item("1-0-0", "원문 질문은?")
    line = _line("1-0-0", "원문 질문은?", "고쳐 쓴 질문은?")

    row = bench.paraphrase_row(source, bench._parse_paraphrase(line))

    assert row["id"] == "paraphrase:1-0-0"
    assert row["kind"] == "paraphrase"
    assert row["question"] == "고쳐 쓴 질문은?"
    assert row["source_id"] == "korquad:1-0-0"
    assert row["original_question"] == "원문 질문은?"
    assert row["context"] == source["context"]  # 다시 검색하지 않고 원문 문항의 5 청크 그대로
    for key in ("answers", "gold_title", "strata"):
        assert row[key] == source[key]
    assert set(row) == set(source) | {
        "source_id",
        "original_question",
    }  # 비고 없으면 review_note 없음


def test_paraphrase_row_does_not_share_context_with_source():
    source = _item("1-0-0", "원문 질문은?")
    row = bench.paraphrase_row(
        source, bench._parse_paraphrase(_line("1-0-0", "원문 질문은?", "새"))
    )

    row["context"][0]["text"] = "바뀜"

    assert source["context"][0]["text"] == "본문 1"


def test_pair_excludes_each_reason():
    items = [
        _item("1-0-0", "질문 1"),
        _item("2-0-0", "질문 2"),
        _item("3-0-0", "질문 3", kind="no_doc"),
        _item("4-0-0", "세트를 다시 만들어 바뀐 질문"),
    ]
    lines = [
        _line("1-0-0", "질문 1", "새 질문 1"),
        _line("2-0-0", "질문 2", "새 질문 2", problems=["answer_leak"]),
        _line("9-0-0", "질문 9", "새 질문 9"),
        _line("3-0-0", "질문 3", "새 질문 3"),
        _line("4-0-0", "질문 4", "새 질문 4"),
    ]

    result = bench.pair([bench._parse_paraphrase(line) for line in lines], items, {})

    assert [row["id"] for row in result.rows] == ["paraphrase:1-0-0"]
    assert [paraphrase.id for paraphrase in result.used] == ["korquad:1-0-0"]
    assert [(exclusion.id, exclusion.reason) for exclusion in result.excluded] == [
        ("korquad:2-0-0", "problems"),
        ("korquad:9-0-0", "missing"),
        ("korquad:3-0-0", "not_answerable"),
        ("korquad:4-0-0", "question_mismatch"),
    ]


def test_pair_attaches_review_note_only_to_noted_items():
    items = [_item("1-0-0", "질문 1"), _item("2-0-0", "질문 2")]
    lines = [_line("1-0-0", "질문 1", "새 질문 1"), _line("2-0-0", "질문 2", "새 질문 2")]

    result = bench.pair(
        [bench._parse_paraphrase(line) for line in lines], items, {"korquad:2-0-0": "조건이 늘어남"}
    )

    assert "review_note" not in result.rows[0]
    assert result.rows[1]["review_note"] == "조건이 늘어남"


def test_pair_rejects_duplicate_paraphrase_ids():
    lines = [_line("1-0-0", "질문 1", "새 질문 1"), _line("1-0-0", "질문 1", "또 새 질문 1")]

    with pytest.raises(ValueError, match="duplicate paraphrase ids"):
        bench.pair(
            [bench._parse_paraphrase(line) for line in lines], [_item("1-0-0", "질문 1")], {}
        )


def test_read_review_parses_marks_and_maps_notes_to_ids(tmp_path):
    path = tmp_path / "review.md"
    table = [
        _review_line(1, "f4-answerable", "korquad:1-0-0", "새 질문 1"),
        _review_line(2, "f9-random", "7-0-0", "새 F9 질문", leak="X").removesuffix(
            " |"
        ),  # 끝 | 빠짐
    ]
    _write_review(
        path, table, ["2번: '처음'이 붙어 조건이 늘어남", "1·2번은 원문에도 같은 단어가 있음"]
    )

    review = bench.read_review(path)

    assert [(row.number, row.sample, row.id, row.paraphrase) for row in review.rows] == [
        (1, "f4-answerable", "korquad:1-0-0", "새 질문 1"),
        (2, "f9-random", "7-0-0", "새 F9 질문"),
    ]
    assert [(row.answer_same, row.no_leak) for row in review.rows] == [(True, True), (True, False)]
    assert dict(review.notes) == {
        "7-0-0": "'처음'이 붙어 조건이 늘어남"
    }  # "N번은 …" 줄은 비고 아님


def test_read_review_rejects_blank_mark(tmp_path):
    path = tmp_path / "review.md"
    _write_review(path, [_review_line(1, "f4-answerable", "korquad:1-0-0", "새 질문 1", same="")])

    with pytest.raises(ValueError, match="must be O or X"):
        bench.read_review(path)


def test_read_review_rejects_note_for_unknown_row(tmp_path):
    path = tmp_path / "review.md"
    _write_review(
        path, [_review_line(1, "f4-answerable", "korquad:1-0-0", "새 질문 1")], ["3번: 없음"]
    )

    with pytest.raises(ValueError, match="not in the table"):
        bench.read_review(path)


def _build_inputs(tmp_path, review_paraphrase="새 질문 1"):
    paraphrase_path = tmp_path / "paraphrase.jsonl"
    _write_jsonl(
        paraphrase_path,
        [
            _line("1-0-0", "질문 1", "새 질문 1"),
            _line("2-0-0", "질문 2", "새 질문 2", problems=["same_as_original"]),
            _line("3-0-0", "질문 3", "새 질문 3"),
            _line("7-0-0", "F9 질문", "새 F9 질문", sample="f9-random"),
        ],
    )
    review_path = tmp_path / "review.md"
    _write_review(
        review_path,
        [
            _review_line(1, "f4-answerable", "korquad:1-0-0", review_paraphrase),
            _review_line(2, "f9-random", "7-0-0", "새 F9 질문"),
        ],
        ["1번: 조건이 늘어남", "2번: 범위가 넓어짐"],
    )
    repo = tmp_path / "bench-data"
    _write_jsonl(
        repo / "source-set" / "items.jsonl",
        [_item("1-0-0", "질문 1"), _item("2-0-0", "질문 2"), _item("3-0-0", "질문 3")],
    )
    return paraphrase_path, review_path, repo


def test_build_writes_items_and_meta(tmp_path):
    paraphrase_path, review_path, repo = _build_inputs(tmp_path)

    bench.build(paraphrase_path, review_path, repo, "source-set", "out-set")

    out_lines = (repo / "out-set" / "items.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in out_lines]
    assert [row["id"] for row in rows] == ["paraphrase:1-0-0", "paraphrase:3-0-0"]
    assert [row.get("review_note") for row in rows] == ["조건이 늘어남", None]
    meta = json.loads((repo / "out-set" / "meta.json").read_text(encoding="utf-8"))
    assert meta["version"] == "out-set"
    assert meta["source_set"] == "source-set"
    assert meta["paraphrase_models"] == {"claude-opus-5-5": 2}
    assert meta["prompt_versions"] == {"paraphrase-v1": 2}
    assert meta["counts"] == {
        "paraphrase": 2,
        "f4_paraphrases": 3,
        "excluded": {"problems": 1, "missing": 0, "not_answerable": 0, "question_mismatch": 0},
        "other_samples": {"f9-random": 1},
    }
    assert list(meta["counts"]["excluded"]) == list(bench.EXCLUDE_REASONS)
    assert meta["excluded"] == [{"id": "korquad:2-0-0", "reason": "problems"}]
    assert meta["human_review"] == {
        "file": review_path.resolve().as_posix(),
        "checked": 2,
        "checked_in_set": 1,
        "answer_changed": 0,
        "answer_leaked": 0,
        "meaning_drift": 2,
        "meaning_drift_in_set": 1,
        "review_note_ids": ["paraphrase:1-0-0"],
    }


def test_build_rejects_review_of_other_paraphrase_text(tmp_path):
    paraphrase_path, review_path, repo = _build_inputs(tmp_path, review_paraphrase="옛 질문 1")

    with pytest.raises(ValueError, match="do not match the paraphrase file"):
        bench.build(paraphrase_path, review_path, repo, "source-set", "out-set")
