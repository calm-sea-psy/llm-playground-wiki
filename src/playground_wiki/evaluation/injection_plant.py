"""간접 인젝션 표적 고르기와 지시 심기 — 표적 세트 규칙 · 원본 마크업에 끼우는 자리 찾기."""

import random
from collections.abc import Iterable, Sequence

from playground_corpus import Chunk
from playground_evalkit import squash

from playground_wiki.evaluation.answer_metrics import contains_gold, gold_numbers
from playground_wiki.evaluation.answer_run import AnswerRecord
from playground_wiki.evaluation.injection_defs import (
    MAX_CANARY_DOCS,
    MIN_TOP_SCORE,
    SEED,
    SOURCE_LABELS,
    TARGET_DOCS,
    Target,
    Variant,
)


def target_row(target: Target, wikitext: str, *, with_source: bool) -> dict[str, object]:
    """targets.jsonl 한 줄. 기본 세트는 예전 키 그대로, 출처가 여럿인 세트만 source · gold_aliases 를 덧붙인다."""
    row: dict[str, object] = {
        "item_id": target.item_id,
        "doc_id": target.doc_id,
        "title": target.title,
        "question": target.question,
        "gold_answer": target.gold_answer,
        "gold_chunk_id": target.gold_chunk_id,
        "top_score": target.top_score,
    }
    if with_source:
        row |= {"source": target.source, "gold_aliases": list(target.gold_aliases)}
    return row | {"wikitext": wikitext}


def select_targets(records: Sequence[AnswerRecord]) -> list[Target]:
    """F9 기록에서 목표 문항을 seed 고정으로 TARGET_DOCS 개 고른다. 후보가 모자라면 멈춘다."""
    candidates = candidate_targets(records)
    random.Random(SEED).shuffle(candidates)  # seed 고정 — 어느 PC 에서 돌려도 같은 목표
    if len(candidates) < TARGET_DOCS:
        raise SystemExit(f"목표 후보가 {len(candidates)}개뿐이다 (필요 {TARGET_DOCS})")
    return candidates[:TARGET_DOCS]


def select_partial_targets(
    records: Sequence[AnswerRecord], extra: Sequence[AnswerRecord] = ()
) -> list[Target]:
    """F9 기록(records)과 더 읽은 기록(extra — 롱테일)에서 partial 표적을 전부 고른다 (seed 고정으로 섞은 순서).

    자르지 않는다 — 조건을 채우는 표적을 다 쓴다 (사용자 결정 2026-11-01). 기본(grounded) 표적과 문서(doc_id · 제목)가
    겹치는 후보는 빼고, 출처가 달라도 같은 문서는 하나만(F9 먼저). 후보가 없거나 canary 번호를 넘으면 멈춘다.
    """
    grounded = select_targets(records)
    taken_docs = {t.doc_id for t in grounded}
    taken_titles = {t.title for t in grounded}
    candidates = []
    for target in candidate_partial_targets([*records, *extra]):
        if target.doc_id not in taken_docs and target.title not in taken_titles:
            taken_docs.add(target.doc_id)  # 제목은 candidate_partial_targets 가 이미 하나씩
            candidates.append(target)
    random.Random(SEED).shuffle(candidates)  # seed 고정 — 어느 PC 에서 돌려도 같은 목표
    if not candidates:
        raise SystemExit("partial 표적 후보가 없다")
    if len(candidates) > MAX_CANARY_DOCS:
        raise SystemExit(
            f"partial 표적 {len(candidates)}개 — canary 번호(최대 {MAX_CANARY_DOCS})를 넘는다"
        )
    return candidates


def candidate_targets(records: Sequence[AnswerRecord]) -> list[Target]:
    """목표가 될 수 있는 문항을 문서당 하나씩 (doc_id 순). 조건: grounded 로 정답을 맞혔고 정답 근거 청크가 1위이며 1위 점수 ≥ MIN_TOP_SCORE — 깨끗한 grounded 라야 답변 변화를 인젝션 탓으로 돌릴 수 있다."""
    return _one_per_document(candidate(record) for record in records)


def candidate_partial_targets(records: Sequence[AnswerRecord]) -> list[Target]:
    """Partial 표적이 될 수 있는 문항을 문서당 하나씩 (doc_id 순). 조건: 마지막 단계가 partial 이고(grounded '못 찾음' → partial, 또는 처음부터 partial) 정답을 담아 답했으며, 정답 근거 청크가 1위로 partial 문맥에 들어갔다. 1위 점수 하한(MIN_TOP_SCORE)은 두지 않는다 — partial 로 가는 문항은 대개 점수가 high 미만이다."""
    return _one_per_document(_partial_candidate(record) for record in records)


def _one_per_document(targets: Iterable[Target | None]) -> list[Target]:
    # 같은 문서(제목)는 기록 순서상 처음 것 하나만, 그다음 doc_id 순
    seen: set[str] = set()
    candidates: list[Target] = []
    for target in targets:
        if target is not None and target.title not in seen:
            seen.add(target.title)
            candidates.append(target)
    candidates.sort(key=lambda t: t.doc_id)
    return candidates


def candidate(record: AnswerRecord) -> Target | None:
    """Grounded 세트 규칙을 채우는 기록이면 표적으로, 아니면 None."""
    item = record.item
    if (
        item.source != "korquad"
        or record.stage_path != ("grounded",)
        or record.marker != "found"
        or item.gold_answer is None
        or item.gold_title is None
        or record.top_score is None
        or record.top_score < MIN_TOP_SCORE
        or not record.hits
    ):
        return None
    return _lead_target(record)


def _partial_candidate(record: AnswerRecord) -> Target | None:
    # 정답 판정은 eval_answer 와 같은 answer_metrics 규칙 — 정답 표기 중 하나라도 (롱테일은 표기가 여럿).
    # 기본 세트(candidate · _lead_target)는 Phase 3 동안 고른 표적이 바뀌면 안 되어 예전 규칙 그대로 둔다
    item = record.item
    if (
        item.source not in SOURCE_LABELS
        or not record.stage_path
        or record.stage_path[-1] != "partial"
        or record.error is not None
        or item.gold_answer is None
        or item.gold_title is None
        or record.top_score is None
        or 1
        not in gold_numbers(record)  # 1위 청크가 정답 근거이고 partial 문맥(출처 1번)에 들어갔다
        or not contains_gold(record)
    ):
        return None
    gold = record.hits[0]
    return Target(
        item_id=item.id,
        doc_id=gold.chunk_id.rsplit("-", 1)[0],
        title=item.gold_title,
        question=item.question,
        gold_answer=item.gold_answer,
        gold_chunk_id=gold.chunk_id,
        top_score=record.top_score,
        source=item.source,
        gold_aliases=item.gold_aliases,
    )


def _lead_target(record: AnswerRecord) -> Target | None:
    """1위 청크가 정답 문서의 정답을 담은 청크이고 답에도 정답이 있으면 그 문항을 표적으로 (아니면 None)."""
    item = record.item
    if item.gold_answer is None or item.gold_title is None or record.top_score is None:
        return None
    gold = record.hits[0]
    if gold.title != item.gold_title or squash(item.gold_answer) not in squash(gold.text):
        return None
    if squash(item.gold_answer) not in squash(record.answer):
        return None
    return Target(
        item_id=item.id,
        doc_id=gold.chunk_id.rsplit("-", 1)[0],
        title=item.gold_title,
        question=item.question,
        gold_answer=item.gold_answer,
        gold_chunk_id=gold.chunk_id,
        top_score=record.top_score,
    )


def background_doc_ids(records: Sequence[AnswerRecord], targets: Sequence[Target]) -> set[str]:
    """목표 질문들의 검색 상위 청크가 속한 문서(목표 제외) — 작은 테스트 코퍼스의 배경이 된다."""
    target_items = {t.item_id for t in targets}
    target_docs = {t.doc_id for t in targets}
    background: set[str] = set()
    for record in records:
        if record.item.id in target_items:
            background.update(hit.chunk_id.rsplit("-", 1)[0] for hit in record.hits)
    return background - target_docs


def inject_at(wikitext: str, variant: Variant, payload: str, offset: int) -> str:
    """Offset 자리에 지시 조각을 심는다. inline 은 그 자리에, block · heading 은 그 줄 앞에 제 줄로."""
    fragment = variant.markup.format(payload=payload)
    if variant.mode == "inline":
        return f"{wikitext[:offset]}{fragment} {wikitext[offset:]}"
    line_start = wikitext.rfind("\n", 0, offset) + 1
    return f"{wikitext[:line_start]}{fragment}\n{wikitext[line_start:]}"


def has_gold(text: str, target: Target) -> bool:
    """글에 정답 표기 중 하나라도 있나 (공백 무시 — answer_metrics.contains_gold 와 같은 규칙)."""
    squashed = squash(text)
    return any(squash(answer) in squashed for answer in target.gold_answers)


def gold_occurrences(wikitext: str, gold_answer: str) -> list[int]:
    """원본 위키텍스트에서 정답 문자열이 나오는 자리 전부."""
    offsets, start = [], 0
    while (index := wikitext.find(gold_answer, start)) != -1:
        offsets.append(index)
        start = index + 1
    return offsets


def has_canary(chunk: Chunk, canary: str) -> bool:
    """청크 본문이나 절 제목에 canary 가 있나."""
    return canary in chunk.text or canary in chunk.section


def lead_offset(wikitext: str) -> int:
    """정답 문자열을 못 찾았을 때 심을 자리 — 머리말(첫 빈 줄) 뒤."""
    # 정답 문자열을 못 찾으면 머리말(첫 빈 줄) 뒤에 둔다 — 틀 · 정보상자 안으로 들어가지 않게
    break_at = wikitext.find("\n\n")
    return break_at + 2 if break_at != -1 else 0
