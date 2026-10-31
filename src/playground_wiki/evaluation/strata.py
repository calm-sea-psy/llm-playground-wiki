"""high 확정 층화 표본 (ADR-024) — 점수 풀 줄과, 1위 점수 구간 · 몫마다 seed 로 고른 표본 (scripts/eval_answer.py pool · strata)."""

import random
from collections.abc import Iterable, Mapping, Sequence

from playground_evalkit import KorQuADItem, contains_answer
from playground_retrieval.search import SearchResult

from playground_wiki.evaluation.answer_metrics import SCORE_BAND_EDGES, answer_rank, score_band
from playground_wiki.evaluation.splits import Split, split_of

# 0.83~0.94 구간을 채우는 몫 순서 — 답변 평가 몫(eval)을 다 담고, 모자라면 F4 문항을 뽑던 bench 몫에서.
# calibrate 몫은 high 의 임시값과 구간 경계를 정한 몫이라 쓰지 않는다
STRATA_SPLITS: tuple[Split, ...] = ("eval", "bench")


def pool_items(
    items: Iterable[KorQuADItem], paragraphs: Mapping[str, str], split: Split
) -> list[KorQuADItem]:
    """그 몫 문단의 질문 전부 (id 순). 코퍼스 · 정답 문장으로 거르지 않는다 — 지어냄은 정답이 아니라 문맥과 대조한다."""
    return sorted(
        (item for item in items if split_of(paragraphs[item.id]) == split),
        key=lambda item: item.id,
    )


def pool_row(item: KorQuADItem, paragraph: str, text: str | None, result: SearchResult) -> dict:
    """점수 풀 한 줄 — 표본 줄(f9-sample)과 같은 키에 코퍼스 대조 · 검색 결과를 붙인다.

    Args:
      item: KorQuAD 문항.
      paragraph: 문항의 문단 키.
      text: 정답 문서의 지금 본문. 문서가 코퍼스에 없으면 None.
      result: 그 질문의 검색 결과 (`/ask` 와 같은 검색).
    """
    return {
        "id": item.id,
        "source": "korquad",
        "question": item.question,
        "gold_title": item.title,
        "gold_answer": item.answer,
        "paragraph": paragraph,
        "in_corpus": text is not None,
        "answer_in_document": text is not None and contains_answer(item.title, text, item),
        "top_score": result.hits[0].score if result.hits else None,
        "answer_rank": answer_rank(result.hits, item),
        "took_ms": result.took_ms,
    }


def strata_rows(
    rows: Sequence[Mapping], *, seed: int, middle_size: int | None = None
) -> list[dict]:
    """0.83~0.94 구간을 STRATA_SPLITS 순서로 채우고 몫마다 같은 수를 ≥ 0.94 구간에서 고른 점수 풀 줄 (id 순 — 두 구간이 섞여 돈다).

    몫 · 구간마다 seed 로 섞은 순서의 앞에서부터 고른다 — middle_size 를 늘려도 앞서 고른 문항은 그대로 남는다.
    몫마다 수를 맞추는 것은 두 구간의 몫 구성을 같게 하려고.

    Args:
      rows: 점수 풀 줄.
      seed: 섞는 seed.
      middle_size: 0.83~0.94 구간 문항 수 상한. None 이면 STRATA_SPLITS 몫의 그 구간 전부.
    """
    lower, upper = SCORE_BAND_EDGES
    picked: list[dict] = []
    room = middle_size
    for split in STRATA_SPLITS:
        middle = _shuffled_band(rows, split, score_band(lower), seed=seed)
        taken = middle if room is None else middle[:room]
        picked += taken + _shuffled_band(rows, split, score_band(upper), seed=seed)[: len(taken)]
        if room is not None:
            room -= len(taken)
    return sorted(picked, key=lambda row: row["id"])


def _shuffled_band(rows: Sequence[Mapping], split: str, band: str, *, seed: int) -> list[dict]:
    chosen = sorted(
        (
            dict(row)
            for row in rows
            if (split_of(row["paragraph"]), score_band(row["top_score"])) == (split, band)
        ),
        key=lambda row: row["id"],
    )
    # 몫 · 구간마다 따로 섞는다 — 한 몫의 문항 수가 달라져도 다른 몫의 순서가 바뀌지 않게
    random.Random(f"{seed}:{split}:{band}").shuffle(chosen)
    return chosen
