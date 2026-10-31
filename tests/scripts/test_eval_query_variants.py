"""scripts/eval_query_variants.py 의 인자 · 세트 읽기 · 리포트를 합성 입력으로 — 데이터 · 모델 · GPU 없이 돈다."""

import json
from datetime import date
from pathlib import Path

import pytest
from conftest import load_script

from playground_wiki.evaluation.files import write_jsonl
from playground_wiki.evaluation.query_rewrite import Rewrite
from playground_wiki.evaluation.query_variants import (
    VARIANTS,
    Variant,
    VariantRecord,
    VariantResult,
)

eval_query_variants = load_script("eval_query_variants")

CONFIG_DIR = Path(__file__).parents[2] / "configs"
QUESTION_MARK = "질문원문표지"
QUERY_MARK = "고친질의표지"


def test_rewrite_and_search_need_a_set():
    with pytest.raises(SystemExit):
        eval_query_variants._parse_args(["search"])

    args = eval_query_variants._parse_args(["rewrite", "--set", "offwiki", "--limit", "3"])

    assert (args.set, args.limit) == ("offwiki", 3)


@pytest.mark.parametrize(
    "argv",
    [
        ["report", "--set", "offwiki"],
        ["report", "--limit", "2"],
        ["search", "--set", "offwiki", "--report-date", "2026-11-01"],
        ["rewrite", "--set", "offwiki", "--limit", "0"],
        ["rewrite", "--set", "nope"],
    ],
)
def test_bad_arguments_are_refused(argv: list[str]):
    with pytest.raises(SystemExit):
        eval_query_variants._parse_args(argv)


def test_compound_set_is_read_only_when_it_exists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit, match="복합 질문 세트가 아직 없다"):
        eval_query_variants.load_items("compound", tmp_path)

    write_jsonl(
        eval_query_variants.COMPOUND_PATH,
        [
            {
                "id": "c1",
                "question": "q",
                "answer": "a",
                "aliases": [],
                "hops": [{"title": "가"}, {"title": "나"}],
            }
        ],
    )
    (item,) = eval_query_variants.load_items("compound", tmp_path)
    assert item.gold_titles == ("가", "나")


def test_korquad_and_paraphrase_sets_share_the_f9_random_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    _write_paraphrases()

    korquad = eval_query_variants.load_items("korquad", tmp_path)
    paraphrase = eval_query_variants.load_items("paraphrase", tmp_path)

    assert [item.id for item in korquad] == [item.id for item in paraphrase] == ["p1", "p2"]
    assert korquad[0].question != paraphrase[0].question


def test_search_stops_before_loading_models_when_rewrites_are_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    _write_offwiki()

    with pytest.raises(SystemExit, match="먼저 rewrite --set offwiki"):
        eval_query_variants.search("offwiki", limit=None, bench_data=tmp_path)


def test_report_has_aggregates_and_ids_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RAG_CONFIG_DIR", str(CONFIG_DIR))
    _write_paraphrases()
    _write_offwiki()
    _write_set(
        "korquad",
        [
            VariantRecord("p1", None, _results(base_rank=None, other_rank=1)),
            VariantRecord("p2", None, _results(base_rank=1, other_rank=1)),
        ],
    )
    _write_set(
        "offwiki",
        [
            VariantRecord("ow-1", None, _results(base_score=0.5, other_score=0.95)),
            VariantRecord("ow-2", "chitchat"),
        ],
    )
    write_jsonl(
        eval_query_variants.SENSITIVITY_PATH,
        [
            {"id": "p1", "original": {"answer_rank": None}, "paraphrase": {"answer_rank": 2}},
            {"id": "p2", "original": {"answer_rank": 1}, "paraphrase": {"answer_rank": 1}},
        ],
    )

    eval_query_variants.report(date(2026, 11, 2), tmp_path)

    text = (tmp_path / "reports" / "2026-11-02-query-variants.md").read_text(encoding="utf-8")
    assert text.startswith("# 질의 변형 실험 — 검색만 (2026-11-02)")
    assert (
        "| ③-a 여러 질의 + 원문 리랭커 | 2 | 2/2 (100.0%) | 2/2 (100.0%) | 1.000 | 0 / 1 (p 1) | 0 / 1 (p 1) |"
        in text
    )
    assert "| ① 원문 한 번 | 2 | 1/2 (50.0%) |" in text
    assert "- ② 고쳐 쓰기 + 원문 리랭커: 1 — ow-1" in text
    assert "단계 없음 (RRF 점수)" in text
    assert "| 위키 밖 질문 v1 | 2 | 2 | 1 |" in text
    assert "같음 2/2 (100.0%)" in text
    assert "고친 질의에서 숫자가 빠짐: 2 — p1, p2" in text
    assert QUESTION_MARK not in text
    assert QUERY_MARK not in text


def _write_paraphrases() -> None:
    write_jsonl(
        eval_query_variants.PARAPHRASE_PATH,
        [
            {
                "sample": "f9-random",
                "id": "p1",
                "title": "가",
                "question": f"{QUESTION_MARK} 1",
                "answers": ["a"],
                "paraphrase": "바꾼 말 1",
            },
            {
                "sample": "f9-random",
                "id": "p2",
                "title": "나",
                "question": f"{QUESTION_MARK} 2",
                "answers": ["b"],
                "paraphrase": "바꾼 말 2",
            },
            {
                "sample": "f4-answerable",
                "id": "p3",
                "title": "다",
                "question": "x",
                "answers": ["c"],
                "paraphrase": "y",
            },
        ],
    )


def _write_offwiki() -> None:
    write_jsonl(
        eval_query_variants.OFFWIKI_PATH,
        [
            {
                "id": "ow-1",
                "question": f"{QUESTION_MARK} 3",
                "expected": "partial",
                "category": "missing_fact",
                "trap": False,
                "related_title": None,
            },
            {
                "id": "ow-2",
                "question": "안녕",
                "expected": "refuse",
                "category": "smalltalk",
                "trap": False,
                "related_title": None,
            },
        ],
    )


def _write_set(set_name: str, records: list[VariantRecord]) -> None:
    path = eval_query_variants.search_path(set_name)
    write_jsonl(path, [record.to_json() for record in records])
    settings = {
        "collection": "c",
        "model": "m",
        "reranker_model": "r",
        "gate": True,
        "candidates": 50,
        "rrf_k": 60,
        "rerank_candidates": 20,
        "top_k": 5,
    }
    path.with_suffix(".meta.json").write_text(
        json.dumps({"runs": [{"settings": settings}]}), encoding="utf-8"
    )
    rewrites = [
        Rewrite(
            record.id,
            f"{QUESTION_MARK} 1997년",
            f"{QUERY_MARK} 고친",
            (f"{QUERY_MARK} 가",),
            100.0,
            150.0,
        )
        for record in records
    ]
    rewrites_path = eval_query_variants.rewrites_path(set_name)
    write_jsonl(rewrites_path, [rewrite.to_json() for rewrite in rewrites])
    meta = {
        "runs": [
            {
                "settings": {
                    "spec_id": "ollama/gemma4:12b?think=off",
                    "prompts": ["rewrite@v1:x", "multi_query@v1:y"],
                }
            }
        ]
    }
    rewrites_path.with_suffix(".meta.json").write_text(json.dumps(meta), encoding="utf-8")


def _results(
    *,
    base_rank: int | None = None,
    other_rank: int | None = None,
    base_score: float = 0.5,
    other_score: float = 0.5,
) -> tuple[VariantResult, ...]:
    return tuple(
        _result(
            variant,
            rank=base_rank if variant == "original" else other_rank,
            score=base_score if variant == "original" else other_score,
        )
        for variant in VARIANTS
    )


def _result(variant: Variant, *, rank: int | None, score: float) -> VariantResult:
    return VariantResult(
        variant=variant,
        chunk_ids=("c-0",),
        titles=("가",),
        scores=(score,),
        gold_rank=rank,
        hops_found=0,
        queries=1,
        pairs=20,
        search_ms=10.0,
        rerank_ms=5.0,
        llm_ms=0.0 if variant == "original" else 100.0,
    )
