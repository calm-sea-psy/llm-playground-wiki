"""KorQuAD dev 가 참조하는 문서를 모두 넣고 나머지를 랜덤으로 채운 서브셋을 만든다.

출력 ({parsed_dir}/{source.name}-subset-{size}/):
  docs-00000.parquet   서브셋 문서 (playground_corpus 형식, 원본 샤드와 같은 스키마)
  subset.json          구성 · KorQuAD 제목 매칭 결과 (korquad_title_map: KorQuAD 제목 → 코퍼스 제목.
                       CORPUS 벤치 · RAG 평가가 같은 매칭을 쓰도록 title_aliases 로 넘긴다)

서브셋 수치는 모델 · 설정 간 상대 비교 전용이다. 정답 문서가 다 들어 있어 방해 문서가 적으므로
hit@k 가 전체 코퍼스보다 높게 나온다 — 전체 수치와 비교하지 않는다.

사용법:
  uv run python scripts/make_subset.py
  uv run python scripts/make_subset.py --size 10000 --seed 20261022
"""

import argparse
import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml
from playground_evalkit import korquad_title, normalize_title

SUBSET_SIZE = 10_000  # 임베딩 3종 × 청크 크기 3종 비교를 한 번에 수십 분 안에 돌릴 수 있는 크기
SEED = 20261022
KORQUAD_DEV = Path("data/korquad/KorQuAD_v1.0_dev.json")

# KorQuAD(2018) 이후 넘겨주기 없이 이름이 바뀐 문서. 후보 문서 본문에 KorQuAD 정답이 들어 있는지로 확인 (2026-10-22)
#   김정남 15/15 · 윤정훈 14/14 · 김홍집 8/18 (문서가 크게 고쳐졌지만 같은 인물)
RENAMED_TITLES = {
    "김정남 (조선민주주의인민공화국)": "김정남 (김정일의 아들)",
    "김홍집 (1842년)": "김홍집",
    "윤정훈": "윤정훈 (목사)",
}


def main() -> None:
    args = _parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))["source"]
    parsed_dir = Path(config["parsed_dir"]) / config["name"]
    out_dir = Path(config["parsed_dir"]) / f"{config['name']}-subset-{args.size}"
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"{out_dir} 가 비어 있지 않다. 다시 만들려면 --overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)

    korquad_titles = _korquad_titles(args.korquad)
    aliases = _aliases(parsed_dir / "redirects.parquet")
    shards = sorted(parsed_dir.glob("docs-*.parquet"))
    titles = pa.concat_arrays(
        [pq.read_table(s, columns=["title"])["title"].combine_chunks() for s in shards]
    )
    title_to_index = {normalize_title(t): i for i, t in enumerate(titles.to_pylist())}

    matched: dict[str, str] = {}  # KorQuAD 제목 → 코퍼스 제목 (정규화)
    via_redirect: list[str] = []
    via_rename: list[str] = []
    missing: list[str] = []
    for original in korquad_titles:
        title = korquad_title(original)
        if title in title_to_index:
            matched[original] = title
        elif aliases.get(title) in title_to_index:
            matched[original] = aliases[title]
            via_redirect.append(original)
        elif RENAMED_TITLES.get(title) in title_to_index:
            matched[original] = RENAMED_TITLES[title]
            via_rename.append(original)
        else:
            missing.append(original)

    required = {title_to_index[title] for title in matched.values()}
    if len(required) > args.size:
        raise SystemExit(f"KorQuAD 문서 {len(required)}개가 size={args.size} 보다 많다")
    rng = random.Random(args.seed)
    pool = [i for i in range(len(titles)) if i not in required]
    chosen = sorted(required | set(rng.sample(pool, args.size - len(required))))

    table = _take(shards, chosen)
    pq.write_table(table, out_dir / "docs-00000.parquet")
    summary = {
        "size": table.num_rows,
        "seed": args.seed,
        "source_docs": len(titles),
        "korquad_dev": str(args.korquad),
        "korquad_titles": len(korquad_titles),
        "korquad_matched": len(matched),
        "korquad_matched_via_redirect": via_redirect,
        "korquad_matched_via_rename": via_rename,
        "korquad_missing": missing,
        "korquad_title_map": matched,
        "korquad_docs_in_subset": len(required),
        "random_docs": table.num_rows - len(required),
        "note": "상대 비교 전용 — 전체 코퍼스 수치와 비교하지 않는다",
    }
    (out_dir / "subset.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/ingest.yaml"))
    parser.add_argument("--korquad", type=Path, default=KORQUAD_DEV)
    parser.add_argument("--size", type=int, default=SUBSET_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _korquad_titles(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))["data"]
    return list(dict.fromkeys(article["title"] for article in data))


def _aliases(path: Path) -> dict[str, str]:
    table = pq.read_table(path)
    return {
        normalize_title(title): normalize_title(target)
        for title, target in zip(
            table["title"].to_pylist(), table["target"].to_pylist(), strict=True
        )
    }


def _take(shards: list[Path], indices: list[int]) -> pa.Table:
    """샤드를 이어 붙인 순서 기준 전역 인덱스로 행을 모은다."""
    parts = []
    offset = 0
    wanted = pa.array(indices, type=pa.int64())
    for shard in shards:
        table = pq.read_table(shard)
        local = pc.subtract(wanted, offset)
        mask = pc.and_(pc.greater_equal(local, 0), pc.less(local, table.num_rows))
        rows = pc.filter(local, mask)
        if len(rows):
            parts.append(table.take(rows))
        offset += table.num_rows
    return pa.concat_tables(parts)


if __name__ == "__main__":
    main()
