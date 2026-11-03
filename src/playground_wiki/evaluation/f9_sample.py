"""F9 답변 평가 표본 — 문단마다 한 문항 고르기 · 되살린 문서 덧붙이기 · 롱테일 세트 문항 · 문항 지정 (scripts/eval_answer.py)."""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from playground_evalkit import KorQuADItem

from playground_wiki.evaluation.answer_run import EvalItem
from playground_wiki.evaluation.splits import answer_kept_sample

SEED = 0
"""F9 표본 · 층화 표본 · 판정 입력을 섞는 seed — 바꾸면 이미 뽑은 표본 · 기록과 짝이 깨진다."""
LONGTAIL_KIND = "longtail"
"""롱테일 세트에서 답 있음 문항의 kind — no_fact 는 문서에 그 사실이 없게 고른 문항이다."""


def sample_rows(
    items: Sequence[KorQuADItem],
    paragraphs: Mapping[str, str],
    texts: Mapping[str, str],
    offwiki: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, object]], int]:
    """표본 줄(KorQuAD 다음 위키 밖)과, 정답 문장을 대조하기 전에 문단마다 고른 KorQuAD 문항 수."""
    picked, kept = answer_kept_sample(items, paragraphs, texts, "eval", seed=SEED)
    rows: list[dict[str, object]] = [
        {
            "id": item.id,
            "source": "korquad",
            "question": item.question,
            "gold_title": item.title,
            "gold_answer": item.answer,
            "paragraph": paragraphs[item.id],
        }
        for item in kept
    ]
    rows += [
        {
            "id": row["id"],
            "source": "offwiki",
            "question": row["question"],
            "expected": row["expected"],
            "category": row["category"],
            "trap": row["trap"],
            "related_title": row["related_title"],
        }
        for row in offwiki
    ]
    return rows, len(picked)


def longtail_items(rows: Iterable[Mapping[str, Any]]) -> list[EvalItem]:
    """롱테일 세트 줄에서 답 있음(LONGTAIL_KIND) 문항만 (파일 순서). 정답 표기는 전부 — 첫째가 gold_answer.

    category 에는 Wikidata 속성 이름(출생지 · 감독 …)을 싣는다.
    """
    return [
        EvalItem(
            id=str(row["id"]),
            source="longtail",
            question=str(row["question"]),
            gold_title=str(row["gold_title"]),
            gold_answer=str(row["answers"][0]),
            category=str(row["strata"]["property"]),
            gold_aliases=tuple(str(answer) for answer in row["answers"][1:]),
        )
        for row in rows
        if row["kind"] == LONGTAIL_KIND
    ]


def select_items(items: Sequence[EvalItem], ids: Sequence[str]) -> list[EvalItem]:
    """표본에서 ids 의 문항만 (표본 순서).

    Raises:
      ValueError: 표본에 없는 id 가 있을 때 — 오타가 조용히 빈 실행이 되지 않게.
    """
    if missing := sorted(set(ids) - {item.id for item in items}):
        raise ValueError(f"ids not in the sample: {missing}")
    wanted = set(ids)
    return [item for item in items if item.id in wanted]
