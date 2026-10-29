"""F4 LLM 벤치의 KorQuAD 세트를 만든다 — 검색 결과를 고정해 LLM 레포 bench/ 는 읽기만 한다.

단계별로 돌린다 — 단계마다 결과를 data/bench/ 에 남겨, 뒤 단계를 다시 돌릴 때 앞 단계를 반복하지 않는다.
  sample  KorQuAD dev 의 bench 몫(문단 해시 40%)에서 문단마다 한 문항을 고르고, 정답 문서가 코퍼스에 있고
          정답 문장이 지금 문서에 남은 것만 → data/bench/f4-korquad-sample-v1.jsonl
  search  /ask 와 같은 검색(ApiConfig — Hybrid + 리랭커)으로 리랭커 후보 전부(SEARCH_K)를 받아
          → data/bench/f4-korquad-search-v1.jsonl (+ .meta.json). 상위 5 는 /ask 와 같고, 그 아래는 답 없음 A 를 채울 몫
  build   세트 → {out_dir}/f4-korquad-{검색 설정 해시}/items.jsonl · meta.json
          answerable: 상위 5 안에 정답 청크(정답 문서 + 정답 문자열)가 있는 문항에서 seed 로 ANSWERABLE 개
          no_doc: answerable 에서 seed 로 NO_DOC 개 — 정답 문서 청크와 정답 문자열이 든 청크를 빼고 다음 순위로 5 개를 채운다
          (기대 답은 null. 정답은 남겨 "기억으로 맞힘"을 가린다)
          문항마다 층: 정답 문서 역링크 수(scripts/count_backlinks.py) · 정답 모양 · 정답 청크 순위

예비 원본 — 1단계 세트로 모자란 파생 문항(build_bench_extra.py 반사실)을 채울 몫. 벤치로 돌리지 않는다.
  spare-sample  bench 몫에서 1단계 세트 문항이 하나도 없는 문단의 질문 전부 (정답이 지금 문서에 남은 것)
                → data/bench/f4-korquad-spare-sample-v1.jsonl. 1단계 세트와 문단(문맥)이 겹치지 않는다
  spare-search  search 와 같은 검색 → data/bench/f4-korquad-spare-search-v1.jsonl (+ .meta.json)
  spare-build   상위 5 안에 정답이 든 문항 전부 (뽑기 · no_doc 없음, 문단 키를 남긴다)
                → {out_dir}/f4-korquad-spare-{검색 설정 해시}/items.jsonl · meta.json

상위 5 필터를 거친 세트라 여기서 나온 수치는 사용자가 보는 end-to-end 정확도가 아니다 (리포트에 적는다).
data/ 와 LLM 레포 bench/data/ 는 git 제외 — KorQuAD 는 CC BY-ND 라 질문 · 문맥 원문은 로컬에만 둔다.
search 는 PG · Qdrant(docker compose up)와 저장한 BM25 색인(data/index/bm25/{컬렉션})이 필요하다.

사용법:
  uv run python scripts/build_bench_set.py sample
  uv run python scripts/build_bench_set.py search
  uv run python scripts/build_bench_set.py build
  uv run python scripts/build_bench_set.py spare-sample   (spare-search · spare-build 도 같은 꼴)
"""

import argparse
import asyncio
import hashlib
import json
import random
import re
import time
from collections.abc import Mapping, Sequence, Set
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
from playground_evalkit import (
    KorQuADItem,
    contains_answer,
    load_korquad,
    normalize_title,
    squash,
)
from playground_retrieval.search import SearchService
from playground_retrieval.store import PgStore, QdrantStore

from playground_wiki.api.config import ApiConfig
from playground_wiki.evaluation import korquad_paragraphs, sample_per_paragraph
from playground_wiki.evaluation.splits import split_of

SUBSET = Path("data/parsed/kowiki-subset-10000")  # subset.json: KorQuAD dev 경로 · 제목 보정표 (D4)
# 전체 덤프 파싱 결과 — 적재한 kowiki_kure-v1 컬렉션과 같은 문서
PARSED = Path("data/parsed/kowiki")
BACKLINKS = Path("data/bench/backlinks-20261001.parquet")
SAMPLE_PATH = Path("data/bench/f4-korquad-sample-v1.jsonl")
SEARCH_PATH = Path("data/bench/f4-korquad-search-v1.jsonl")
SPARE_SAMPLE_PATH = Path("data/bench/f4-korquad-spare-sample-v1.jsonl")
SPARE_SEARCH_PATH = Path("data/bench/f4-korquad-spare-search-v1.jsonl")
OUT_DIR = Path("../llm-playground-llm/bench/data")
SEED = 0
TOP_K = 5  # 모델에 넘기는 청크 수 — /ask 와 같다
SEARCH_K = (
    20  # 리랭커 후보 수(playground_retrieval rerank.CANDIDATES) — 이보다 깊게는 다시 매기지 않는다
)
ANSWERABLE = 300
NO_DOC = 50
# 정답 모양 층 (진단용). 인물은 질문 표현으로, 날짜 · 숫자는 정답에 숫자가 있는지로 가른다
PERSON_CUES = re.compile(r"누구|누가|인물|사람|선수|감독|작가|작곡가|배우|왕은|황제")


def main() -> None:
    args = _parse_args()
    if args.step == "sample":
        sample()
    elif args.step == "search":
        search(SAMPLE_PATH, SEARCH_PATH)
    elif args.step == "build":
        build(args.out_dir)
    elif args.step == "spare-sample":
        spare_sample(args.out_dir)
    elif args.step == "spare-search":
        search(SPARE_SAMPLE_PATH, SPARE_SEARCH_PATH)
    elif args.step == "spare-build":
        spare_build(args.out_dir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "step",
        choices=["sample", "search", "build", "spare-sample", "spare-search", "spare-build"],
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="build · spare-build: 세트를 쓸 곳, spare-sample: 1단계 세트를 읽을 곳",
    )
    return parser.parse_args()


def sample() -> None:
    summary = json.loads((SUBSET / "subset.json").read_text(encoding="utf-8"))
    korquad = Path(summary["korquad_dev"])
    items = load_korquad(korquad, title_aliases=summary["korquad_title_map"])
    answers = _all_answers(korquad)
    paragraphs = korquad_paragraphs(korquad)
    texts = _document_texts({item.title for item in items})
    in_corpus = [item for item in items if item.title in texts]
    picked = sample_per_paragraph(in_corpus, paragraphs, "bench", seed=SEED)
    # 문서는 있어도 정답 문장이 2018년 이후 고쳐졌으면 뺀다 — 검색이 아니라 문자열 대조라 선택 편향이 없다
    kept = [item for item in picked if contains_answer(item.title, texts[item.title], item)]
    rows = [
        {
            "id": item.id,
            "paragraph": paragraphs[item.id],
            "title": item.title,
            "question": item.question,
            "answer": item.answer,
            "answers": answers[item.id],
        }
        for item in kept
    ]
    _write_jsonl(SAMPLE_PATH, rows)
    print(
        f"KorQuAD dev 질문 {len(items):,} (정답 문서가 코퍼스에 있음 {len(in_corpus):,})\n"
        f"bench 문단마다 한 문항 {len(picked)} → 정답이 코퍼스에 남은 것 {len(kept)} "
        f"(빠짐 {len(picked) - len(kept)})\n저장: {SAMPLE_PATH}"
    )


def spare_sample(out_dir: Path) -> None:
    summary = json.loads((SUBSET / "subset.json").read_text(encoding="utf-8"))
    korquad = Path(summary["korquad_dev"])
    items = load_korquad(korquad, title_aliases=summary["korquad_title_map"])
    answers = _all_answers(korquad)
    paragraphs = korquad_paragraphs(korquad)
    stage1 = _read_jsonl(out_dir / _stage1_version() / "items.jsonl")
    taken = {paragraphs[row["id"].split(":", 1)[1]] for row in stage1}
    picked = spare_items(items, paragraphs, taken)
    texts = _document_texts({item.title for item in picked})
    kept = [
        item
        for item in picked
        if item.title in texts and contains_answer(item.title, texts[item.title], item)
    ]
    rows = [
        {
            "id": item.id,
            "paragraph": paragraphs[item.id],
            "title": item.title,
            "question": item.question,
            "answer": item.answer,
            "answers": answers[item.id],
        }
        for item in kept
    ]
    _write_jsonl(SPARE_SAMPLE_PATH, rows)
    print(
        f"bench 몫 중 1단계 세트 문항이 없는 문단의 질문 {len(picked)} "
        f"({len({paragraphs[item.id] for item in picked})} 문단) → 정답이 코퍼스에 남은 것 {len(kept)}\n"
        f"저장: {SPARE_SAMPLE_PATH}"
    )


def spare_items(
    items: Sequence[KorQuADItem], paragraphs: Mapping[str, str], taken: Set[str]
) -> list[KorQuADItem]:
    """bench 몫 문단 중 taken(1단계 세트가 쓴 문단)이 아닌 문단의 질문 전부를 id 순으로 돌려준다.

    1단계는 문단마다 한 문항만 골랐다 — 쓴 문단을 통째로 빼야 예비 문항이 1단계 문항과 문맥을 나눠 갖지 않는다.
    """
    return sorted(
        (
            item
            for item in items
            if split_of(paragraphs[item.id]) == "bench" and paragraphs[item.id] not in taken
        ),
        key=lambda item: item.id,
    )


def search(sample_path: Path, search_path: Path) -> None:
    rows = _read_jsonl(sample_path)
    config = ApiConfig.from_env()
    start = time.perf_counter()
    with PgStore.connect(config.pg) as pg:
        service = SearchService.build(config.search, pg, QdrantStore.from_env())
        print(f"검색 준비 {time.perf_counter() - start:.0f}s · {len(rows)}문항", flush=True)
        results = asyncio.run(_search_all(service, [row["question"] for row in rows]))
    out = [
        {
            "id": row["id"],
            "hits": [
                {
                    "chunk_id": hit.chunk.id,
                    "title": hit.chunk.title,
                    "section": hit.chunk.section,
                    "text": hit.chunk.text,
                    "score": round(hit.score, 6),
                }
                for hit in result.hits
            ],
            "took_ms": result.took_ms,
        }
        for row, result in zip(rows, results, strict=True)
    ]
    _write_jsonl(search_path, out)
    meta = {
        "created_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "search": _search_settings(config),
        "inputs": {"sample": str(sample_path)},
    }
    search_path.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"저장: {search_path} ({time.perf_counter() - start:.0f}s)")


def build(out_dir: Path) -> None:
    searched = _read_jsonl(SEARCH_PATH)
    search_meta = json.loads(SEARCH_PATH.with_suffix(".meta.json").read_text(encoding="utf-8"))
    backlinks = _backlinks()
    pool = _answer_pool(_read_jsonl(SAMPLE_PATH), searched)
    rng = random.Random(SEED)
    answerable = sorted(
        rng.sample(pool, min(ANSWERABLE, len(pool))), key=lambda entry: entry[0]["id"]
    )
    items = [
        _set_item("answerable", source, hits[:TOP_K], backlinks, answer_rank=rank)
        for source, hits, rank in answerable
    ]
    # 섞은 순서대로 보며 정답 없는 청크 5개를 채울 수 있는 것만 NO_DOC 개까지 (후보 20 개가 거의 정답 문서뿐이면 못 채운다)
    no_doc = []
    for source, hits, _ in rng.sample(answerable, len(answerable)):
        context = _without_answer(source, hits)
        if len(context) == TOP_K:
            no_doc.append(_set_item("no_doc", source, context, backlinks, answer_rank=None))
        if len(no_doc) == NO_DOC:
            break
    items += sorted(no_doc, key=lambda item: item["id"])

    version = f"f4-korquad-{_settings_hash(search_meta['search'])}"
    path = out_dir / version / "items.jsonl"
    _write_jsonl(path, items)
    meta = {
        "version": version,
        "created_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "search": search_meta["search"],
        "seed": SEED,
        "counts": {"answerable": len(items) - len(no_doc), "no_doc": len(no_doc)},
        "sample": len(searched),
        "recall_at_5": round(len(pool) / len(searched), 4),
        "note": "상위 5 안에 정답이 든 문항만 — end-to-end 정확도가 아니다",
    }
    (path.parent / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"표본 {len(searched)} 중 상위 5 에 정답 {len(pool)} (recall@5 {meta['recall_at_5']:.1%}) → "
        f"answerable {meta['counts']['answerable']} · no_doc {len(no_doc)}/{NO_DOC}\n저장: {path}"
    )


def spare_build(out_dir: Path) -> None:
    searched = _read_jsonl(SPARE_SEARCH_PATH)
    search_meta = _search_meta(SPARE_SEARCH_PATH)
    stage1 = _stage1_version()
    if search_meta["search"] != _search_meta(SEARCH_PATH)["search"]:
        # 검색 설정이 다르면 예비 문항의 문맥이 1단계 세트와 다른 검색에서 나온다 — 섞어 쓰면 안 된다
        raise ValueError(f"spare search settings differ from {stage1}: {search_meta['search']}")
    backlinks = _backlinks()
    items = [
        {
            **_set_item("answerable", source, hits[:TOP_K], backlinks, answer_rank=rank),
            "paragraph": source["paragraph"],  # 파생 세트가 문단마다 한 문항만 쓰도록
        }
        for source, hits, rank in _answer_pool(_read_jsonl(SPARE_SAMPLE_PATH), searched)
    ]
    version = f"f4-korquad-spare-{_settings_hash(search_meta['search'])}"
    path = out_dir / version / "items.jsonl"
    _write_jsonl(path, items)
    meta = {
        "version": version,
        "created_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "search": search_meta["search"],
        "stage1_set": stage1,
        "counts": {"answerable": len(items)},
        "paragraphs": len({item["paragraph"] for item in items}),
        "sample": len(searched),
        "recall_at_5": round(len(items) / len(searched), 4),
        "note": (
            f"예비 원본 — 벤치로 돌리지 않는다. bench 몫에서 {stage1} 문항이 없는 문단의 질문 중 "
            "상위 5 안에 정답이 든 것 전부. 파생 세트(build_bench_extra.py)가 모자란 문항을 채울 때 쓴다"
        ),
    }
    (path.parent / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"예비 표본 {len(searched)} 중 상위 5 에 정답 {len(items)} "
        f"(recall@5 {meta['recall_at_5']:.1%}, {meta['paragraphs']} 문단)\n저장: {path}"
    )


def _answer_pool(
    sample_rows: Sequence[dict], searched: Sequence[dict]
) -> list[tuple[dict, list[dict], int]]:
    """상위 5 안에 정답 청크가 든 문항마다 (표본 줄, 검색 결과 전부, 첫 정답 청크 순위). 검색 순서대로."""
    sources = {row["id"]: row for row in sample_rows}
    pool = []
    for row in searched:
        source = sources[row["id"]]
        item = _item(source)
        ranks = [
            rank
            for rank, hit in enumerate(row["hits"][:TOP_K], start=1)
            if contains_answer(hit["title"], hit["text"], item)
        ]
        if ranks:
            pool.append((source, row["hits"], ranks[0]))
    return pool


def _stage1_version() -> str:
    """build 가 만든 1단계 세트 이름 (검색 설정 해시를 붙인다)."""
    return f"f4-korquad-{_settings_hash(_search_meta(SEARCH_PATH)['search'])}"


def _search_meta(search_path: Path) -> dict:
    return json.loads(search_path.with_suffix(".meta.json").read_text(encoding="utf-8"))


def _set_item(
    kind: str, source: dict, hits: list[dict], backlinks: dict[str, int], *, answer_rank: int | None
) -> dict:
    prefix = "korquad" if kind == "answerable" else "korquad-nodoc"
    return {
        "id": f"{prefix}:{source['id']}",
        "kind": kind,
        "question": source["question"],
        "answers": source["answers"],
        "gold_title": source["title"],
        "context": [
            {
                "rank": rank,
                "chunk_id": hit["chunk_id"],
                "title": hit["title"],
                "section": hit["section"],
                "text": hit["text"],
                "score": hit["score"],
            }
            for rank, hit in enumerate(hits, start=1)
        ],
        "strata": {
            "backlinks": backlinks.get(source["title"]),
            "answer_shape": _answer_shape(source["question"], source["answer"]),
            "answer_rank": answer_rank,
        },
    }


def _without_answer(source: dict, hits: list[dict]) -> list[dict]:
    """정답 문서 청크와 정답 문자열(모든 정답 표기)이 든 청크를 뺀 상위 5."""
    answers = [squash(answer) for answer in source["answers"]]
    kept = [
        hit
        for hit in hits
        if normalize_title(hit["title"]) != source["title"]
        and not any(answer in squash(hit["text"]) for answer in answers)
    ]
    return kept[:TOP_K]


def _answer_shape(question: str, answer: str) -> str:
    if PERSON_CUES.search(question):
        return "person"
    if re.search(r"\d", answer):
        return "date_number"
    return "other"


def _item(row: dict) -> KorQuADItem:
    return KorQuADItem(
        id=row["id"],
        title=row["title"],
        question=row["question"],
        answer=row["answer"],
        source_title=row["title"],
    )


def _search_settings(config: ApiConfig) -> dict:
    settings = config.search
    return {
        "collection": settings.collection,
        "model": settings.model,
        "candidates": settings.candidates,
        "rrf_k": settings.rrf_k,
        "rerank_candidates": settings.rerank_candidates,
        "reranker_model": settings.reranker_model,
        "top_k": TOP_K,
        "search_k": SEARCH_K,
    }


def _settings_hash(settings: dict) -> str:
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:8]


async def _search_all(service: SearchService, questions: list[str]) -> list:
    # 서비스가 질의를 한 번에 하나씩 돌리므로(GPU 모델 · PG 접속 공유) 순서대로 기다린다
    results = []
    for index, question in enumerate(questions, start=1):
        results.append(await service.search(question, k=SEARCH_K))
        if index % 50 == 0:
            print(f"  {index}/{len(questions)}", flush=True)
    return results


def _all_answers(path: Path) -> dict[str, list[str]]:
    """질문 id → 정답 표기 전부 (중복 없이, 파일 순서). evalkit EM · F1 이 정답 목록의 최댓값을 쓴다."""
    data = json.loads(path.read_text(encoding="utf-8"))["data"]
    answers: dict[str, list[str]] = {}
    for article in data:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                answers[qa["id"]] = list(dict.fromkeys(a["text"] for a in qa["answers"]))
    return answers


def _document_texts(titles: set[str]) -> dict[str, str]:
    """정규화한 제목 → 본문. 필요한 제목만 담는다."""
    texts: dict[str, str] = {}
    for path in sorted(PARSED.glob("docs-*.parquet")):
        table = pq.read_table(path, columns=["title", "text"]).to_pydict()
        for title, text in zip(table["title"], table["text"], strict=True):
            normalized = normalize_title(title)
            if normalized in titles:
                texts[normalized] = text
    return texts


def _backlinks() -> dict[str, int]:
    """정규화한 제목 → 역링크 수."""
    table = pq.read_table(BACKLINKS, columns=["title", "backlinks"]).to_pydict()
    return {
        normalize_title(title): count
        for title, count in zip(table["title"], table["backlinks"], strict=True)
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
