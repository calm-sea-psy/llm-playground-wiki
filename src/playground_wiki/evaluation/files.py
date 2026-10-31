"""스크립트가 같이 쓰는 파일 읽기 · 쓰기 — JSONL 과 파싱된 문서 본문."""

import json
from collections.abc import Collection, Iterable, Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from playground_evalkit import normalize_title


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """JSONL 의 줄마다 JSON 하나 (빈 줄은 건너뛴다). 쓰다 만 줄이 있으면 실패한다 — 견디는 쪽은 `answer_run.read_records`."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """줄마다 JSON 하나로 덮어쓴다 — UTF-8 그대로(ensure_ascii 끔) · 줄 끝 LF. 폴더가 없으면 만든다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: Mapping[str, Any], *, newline: str | None = "\n") -> None:
    """한 줄을 덧붙인다. 줄마다 열고 닫아 실행이 끊겨도 끝난 줄은 남는다. 폴더가 없으면 만든다.

    Args:
      path: JSONL 파일.
      row: 덧붙일 값 — UTF-8 그대로(ensure_ascii 끔).
      newline: 줄 끝. None 이면 플랫폼 기본(Windows 는 CRLF) — scripts/compare_prompts.py 가 그렇게 써 왔다.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline=newline) as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def document_texts(parsed_dir: Path, titles: Collection[str]) -> dict[str, str]:
    """파싱된 덤프(docs-*.parquet)에서 정규화한 제목 → 본문. titles(정규화한 제목)에 든 것만 담는다.

    같은 정규화 제목이 여럿이면 파일 이름 순서로 나중 것이 남는다.
    """
    texts: dict[str, str] = {}
    for path in sorted(parsed_dir.glob("docs-*.parquet")):
        table = pq.read_table(path, columns=["title", "text"]).to_pydict()
        for title, text in zip(table["title"], table["text"], strict=True):
            normalized = normalize_title(title)
            if normalized in titles:
                texts[normalized] = text
    return texts
