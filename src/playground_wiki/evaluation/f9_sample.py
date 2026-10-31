"""F9 답변 평가 표본 — 문단마다 한 문항 고르기 · 되살린 문서 덧붙이기 · 롱테일 세트 문항 · 문항 지정 (scripts/eval_answer.py)."""

from collections.abc import Iterable, Mapping, Sequence

from playground_evalkit import KorQuADItem, contains_answer

from playground_wiki.evaluation.answer_run import EvalItem
from playground_wiki.evaluation.splits import sample_per_paragraph

SEED = 0
"""F9 표본 · 층화 표본 · 판정 입력을 섞는 seed — 바꾸면 이미 뽑은 표본 · 기록과 짝이 깨진다."""
LONGTAIL_KIND = "longtail"
"""롱테일 세트에서 답 있음 문항의 kind — no_fact 는 문서에 그 사실이 없게 고른 문항이다."""


def sample_rows(
    items: Sequence[KorQuADItem],
    paragraphs: Mapping[str, str],
    texts: Mapping[str, str],
    offwiki: Iterable[Mapping[str, object]],
) -> tuple[list[dict], int]:
    """표본 줄(KorQuAD 다음 위키 밖)과, 정답 문장을 대조하기 전에 문단마다 고른 KorQuAD 문항 수."""
    in_corpus = [item for item in items if item.title in texts]
    picked = sample_per_paragraph(in_corpus, paragraphs, "eval", seed=SEED)
    # 문서는 있어도 정답 문장이 2018년 이후 고쳐졌으면 뺀다 — 검색이 아니라 문자열 대조라 선택 편향이 없다
    kept = [item for item in picked if contains_answer(item.title, texts[item.title], item)]
    rows: list[dict] = [
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


def extend_rows(
    rows: Sequence[Mapping],
    items: Sequence[KorQuADItem],
    paragraphs: Mapping[str, str],
    texts: Mapping[str, str],
) -> tuple[list[dict], int]:
    """있는 표본 줄에 items 를 sample_rows 와 같은 규칙으로 골라 KorQuAD 줄 끝에 덧붙인 줄과 덧붙인 수.

    items 는 되살린 문서의 질문만 넘긴다 — 그 문서들의 문단만 따로 고르므로 있는 줄은 바뀌지 않는다. 이미 표본에
    있는 id · 문단은 다시 넣지 않는다 (두 번 돌려도 같다).
    """
    have_ids = {row["id"] for row in rows}
    have_paragraphs = {row["paragraph"] for row in rows if row["source"] == "korquad"}
    new, _ = sample_rows(
        [item for item in items if paragraphs[item.id] not in have_paragraphs],
        paragraphs,
        texts,
        (),
    )
    new = [row for row in new if row["id"] not in have_ids]
    korquad = [dict(row) for row in rows if row["source"] == "korquad"]
    rest = [dict(row) for row in rows if row["source"] != "korquad"]
    return [*korquad, *new, *rest], len(new)


def longtail_items(rows: Iterable[Mapping]) -> list[EvalItem]:
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
