"""위키 덤프 전체를 파싱 · 정제해 parquet 샤드로 저장한다.

출력 ({parsed_dir}/{source.name}/):
  docs-00000.parquet ...   본문 문서 (playground_corpus 형식)
  redirects.parquet        넘겨주기 (title → target)
  stats.json               처리 통계

사용법:
  uv run python scripts/parse_wiki.py                 # configs/ingest.yaml 기준 전체
  uv run python scripts/parse_wiki.py --limit 5000    # 앞 5000 페이지만 (시험)
"""

import argparse
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from playground_corpus import Document, write_documents

from playground_wiki.api.config import CONFIG_DIR
from playground_wiki.sources import Redirect, Skipped, WikiSource

SHARD_SIZE = 50_000  # 문서 평균 4KB 기준 샤드 하나 약 200MB 메모리 · 수십 MB 파일
PROGRESS_EVERY = 50_000


def main() -> None:
    """커맨드라인 진입점."""
    args = _parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))["source"]
    out_dir = Path(config["parsed_dir"]) / config["name"]
    if out_dir.exists() and any(out_dir.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"{out_dir} 가 비어 있지 않다. 다시 만들려면 --overwrite")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source = WikiSource(
        Path(config["dump_path"]),
        dump_date=str(config["dump_date"]),
        min_chars=int(config["min_chars"]),
    )
    counts: Counter[str] = Counter()
    redirects: list[tuple[str, str]] = []
    shard: list[Document] = []
    shards: list[str] = []
    start = time.perf_counter()

    for index, page in enumerate(source.parsed(), start=1):
        if isinstance(page, Redirect):
            counts["redirect"] += 1
            redirects.append((page.title, page.target))
        elif isinstance(page, Skipped):
            counts[f"skipped_{page.reason}"] += 1
        else:
            counts["document"] += 1
            shard.append(page)
            if len(shard) == args.shard_size:
                shards.append(_write_shard(out_dir, len(shards), shard))
                shard = []
        if index % PROGRESS_EVERY == 0:
            print(
                f"{index:>9,} pages  {time.perf_counter() - start:7.0f}s  {dict(counts)}",
                flush=True,
            )
        if args.limit and index >= args.limit:
            break
    if shard:
        shards.append(_write_shard(out_dir, len(shards), shard))

    _write_redirects(out_dir / "redirects.parquet", redirects)
    elapsed = time.perf_counter() - start
    stats = {
        "source": config["name"],
        "dump_path": config["dump_path"],
        "dump_date": str(config["dump_date"]),
        "min_chars": int(config["min_chars"]),
        "limit": args.limit,
        "pages": sum(counts.values()),
        "counts": dict(counts),
        "shards": shards,
        "elapsed_sec": round(elapsed, 1),
    }
    (out_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "ingest.yaml")
    parser.add_argument(
        "--limit", type=int, default=0, help="앞에서부터 처리할 페이지 수 (0 = 전체)"
    )
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    parser.add_argument(
        "--overwrite", action="store_true", help="기존 출력 폴더를 지우고 다시 만든다"
    )
    return parser.parse_args()


def _write_shard(out_dir: Path, number: int, documents: list[Document]) -> str:
    name = f"docs-{number:05d}.parquet"
    write_documents(documents, out_dir / name)
    return name


def _write_redirects(path: Path, redirects: list[tuple[str, str]]) -> None:
    titles, targets = zip(*redirects, strict=True) if redirects else ((), ())
    table = pa.table({"title": list(titles), "target": list(targets)})
    pq.write_table(table, path)


if __name__ == "__main__":
    main()
