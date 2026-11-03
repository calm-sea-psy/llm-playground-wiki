"""시험 파일 여럿이 같이 쓰는 보조 함수 · 가짜."""

import json
from pathlib import Path

from playground_wiki.evaluation import split_of
from playground_wiki.evaluation.answer_run import AnswerRecord
from playground_wiki.gate import Token


class NoTokens:
    """형태소가 없는 분석기 — 실시간 규칙은 걸리지 않고 잡담 사전만 본다 (Kiwi 를 올리지 않는다)."""

    def tokenize(self, text: str) -> list[Token]:
        return []


def paragraphs_in(split: str, count: int) -> list[str]:
    keys = (f"문서{i}#0" for i in range(1000))
    return [key for key in keys if split_of(key) == split][:count]


def char_splits(text: str) -> list[list[str]]:
    """두 조각으로 나누는 모든 경우 + 한 글자씩."""
    return [[text[:i], text[i:]] for i in range(1, len(text))] + [list(text)]


def write_records(path: Path, records: list[AnswerRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record.to_json(), ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
