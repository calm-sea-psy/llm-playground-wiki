"""컬렉션의 PG 청크로 BM25 색인을 만들어 저장한다. /search 는 시작할 때 이 색인을 읽는다.

전체 규모(청크 약 240만)는 형태소 분석에 수십 분이 걸려 서버가 켜질 때마다 만들 수 없다.
청크는 PG 에서 흘려 받고(서버 쪽 커서), 배열은 .npy 로 써서 서버가 메모리 매핑으로 읽는다.

출력:
  data/index/bm25/{collection}/   offsets · rows · weights (.npy), terms · chunk_ids · doc_ids (.json), meta.json

사용법:
  uv run python scripts/build_bm25.py --collection kowiki_kure-v1
"""

import argparse
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

from playground_retrieval.search import Bm25Index, KiwiAnalyzer
from playground_retrieval.search.analyzer import KEEP_TAGS
from playground_retrieval.search.bm25 import K1, B
from playground_retrieval.store import PgConfig, PgStore

from playground_wiki.api.config import BM25_INDEX_DIR
from playground_wiki.sources.wiki import SOURCE_NAME


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = args.out or BM25_INDEX_DIR / args.collection
    start = time.perf_counter()
    with PgStore.connect(PgConfig.from_env()) as pg:
        index = Bm25Index.build(pg.iter_chunks(args.collection, SOURCE_NAME), KiwiAnalyzer())
    if not len(index):
        raise SystemExit(f"컬렉션에 청크가 없다: {args.collection}")
    build_sec = time.perf_counter() - start
    index.save(
        out,
        meta={
            "collection": args.collection,
            "source": SOURCE_NAME,
            "k1": K1,
            "b": B,
            "analyzer": f"kiwi {sorted(KEEP_TAGS)}",
            "build_sec": round(build_sec, 1),
            "built_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        },
    )
    print(f"done: {len(index):,} chunks → {out} ({build_sec:.0f}s)")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--collection", required=True)
    parser.add_argument("--out", type=Path, help="기본: data/index/bm25/{collection}")
    return parser.parse_args()


if __name__ == "__main__":
    main()
