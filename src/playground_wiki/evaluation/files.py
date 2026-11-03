"""스크립트가 같이 쓰는 파일 읽기 · 쓰기 — JSONL 과 파싱된 문서 본문."""

import json
import logging
from collections.abc import Collection, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from playground_evalkit import normalize_title

logger = logging.getLogger(__name__)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """JSONL 의 줄마다 JSON 하나 (빈 줄은 건너뛴다). 쓰다 만 줄이 있으면 실패한다 — 견디는 쪽은 `answer_run.read_records`."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def read_jsonl_tail(path: Path) -> tuple[list[dict[str, Any]], bool]:
    """이어 쓰는 기록의 줄들과, 쓰다 만 마지막 줄을 버렸는지. 파일이 없으면 빈 목록.

    Raises:
      json.JSONDecodeError: 마지막이 아닌 줄이 JSON 이 아닐 때 — 끊김이 아니라 파일이 깨졌다.
    """
    if not path.exists():
        return [], False
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    rows = []
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index < len(lines) - 1:
                raise
            logger.warning("dropping a partial last line in %s", path)
            return rows, True
    return rows, False


def rewrite_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """줄들을 임시 파일에 다 쓴 뒤 바꿔 끼운다 — 도중에 끊겨도 원래 기록이 남는다."""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


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
      newline: 줄 끝. None 이면 플랫폼 기본(Windows 는 CRLF) — scripts/judge_check.py 가 그렇게 써 왔다.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline=newline) as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def as_mapping(value: object) -> Mapping[str, Any]:
    """JSON 값을 매핑으로 좁힌다.

    Raises:
      TypeError: 매핑이 아닐 때.
    """
    if not isinstance(value, Mapping):
        raise TypeError(f"expected a mapping, got {value!r}")
    return value


def optional_str(value: object) -> str | None:
    """JSON 값을 문자열로 (None 은 그대로)."""
    return None if value is None else str(value)


def write_json(path: Path, value: object) -> None:
    """JSON 하나를 들여쓰기 2 로 덮어쓴다 — UTF-8 그대로 · 끝에 줄바꿈 · 폴더가 없으면 만든다.

    줄 끝은 플랫폼 기본(Windows 는 CRLF)이다 — 지금까지 쓴 meta · 열쇠 파일과 바이트를 맞춘다.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:  # noqa: ANN401  # JSON 경계 — 받는 쪽이 모양을 안다
    """JSON 파일 하나를 읽는다."""
    return json.loads(path.read_text(encoding="utf-8"))


def meta_path(path: Path) -> Path:
    """기록 옆 실행 meta 파일 (`이름.jsonl` → `이름.meta.json`)."""
    return path.with_suffix(".meta.json")


def read_run_meta(path: Path, *, missing_ok: bool = False) -> list[dict[str, Any]]:
    """기록 옆 meta 파일의 runs (실행마다 설정 한 줄, 실행 차례).

    Args:
      path: 기록 파일 (`meta_path` 로 meta 를 찾는다).
      missing_ok: meta 가 없으면 빈 목록을 돌려준다. False 면 FileNotFoundError.
    """
    meta = meta_path(path)
    if missing_ok and not meta.exists():
        return []
    return read_json(meta)["runs"]


def append_run_meta(path: Path, entry: Mapping[str, Any]) -> None:
    """기록 옆 meta 파일의 runs 에 실행 하나를 덧붙인다 (없으면 만든다)."""
    runs = read_run_meta(path, missing_ok=True)
    runs.append(entry)
    write_json(meta_path(path), {"runs": runs})


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
