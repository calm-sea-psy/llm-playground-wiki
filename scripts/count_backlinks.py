"""문서마다 역링크 수(본문 문서끼리의 [[링크]])를 세어 저장한다 — F4 롱테일 세트 · 인기도 층화용.

덤프를 정제 없이 한 번 훑는다. 넘겨주기 표는 parse_wiki.py 가 같은 덤프로 만든 redirects.parquet 를 쓴다.

출력: {out_dir}/backlinks-{dump_date}.parquet — page_id · title · backlinks (넘겨주기가 아닌 본문 이름공간 문서 전부.
stub · 동음이의 문서도 들어 있다 — 쓰는 쪽이 코퍼스 문서와 page_id 로 맞춘다)

사용법:
  uv run python scripts/count_backlinks.py
  uv run python scripts/count_backlinks.py --limit 20000    # 앞 2만 페이지만 (시험 — 파일 이름에 -limit20000)
"""

import argparse
import statistics
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from playground_wiki.sources import BacklinkCounter, WikiSource, redirect_target

PROGRESS_EVERY = 100_000


def main() -> None:
    args = _parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))["source"]
    redirects = pq.read_table(Path(config["parsed_dir"]) / config["name"] / "redirects.parquet")
    counter = BacklinkCounter(
        dict(zip(redirects["title"].to_pylist(), redirects["target"].to_pylist(), strict=True))
    )
    source = WikiSource(Path(config["dump_path"]), dump_date=str(config["dump_date"]))
    pages: list[tuple[str, str]] = []
    start = time.perf_counter()
    print(f"넘겨주기 {redirects.num_rows:,} · 덤프 {config['dump_path']}", flush=True)
    for index, page in enumerate(source.raw_pages(), start=1):
        if args.limit and index > args.limit:
            break
        if index % PROGRESS_EVERY == 0:
            print(f"  {index:,} 페이지 {time.perf_counter() - start:.0f}s", flush=True)
        if redirect_target(page.wikitext) is not None:
            continue
        pages.append((page.id, page.title))
        counter.add(page.title, page.wikitext)

    counts = counter.counts(title for _, title in pages)
    backlinks = [counts[title] for _, title in pages]
    suffix = f"-limit{args.limit}" if args.limit else ""
    out = args.out_dir / f"backlinks-{config['dump_date']}{suffix}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "page_id": [page_id for page_id, _ in pages],
                "title": [title for _, title in pages],
                "backlinks": backlinks,
            }
        ),
        out,
    )
    ranked = sorted(zip(backlinks, (title for _, title in pages), strict=True), reverse=True)
    print(
        f"끝 {time.perf_counter() - start:.0f}s · 문서 {len(pages):,} · 역링크 0 {backlinks.count(0):,} · "
        f"중앙값 {statistics.median(backlinks) if backlinks else 0} · {out}"
    )
    print("상위 10:", ", ".join(f"{title} {n:,}" for n, title in ranked[:10]))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/ingest.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/bench"))
    parser.add_argument("--limit", type=int, default=0, help="앞 n 페이지만 (0 이면 전체)")
    return parser.parse_args()


if __name__ == "__main__":
    main()
