"""closed-book 비교 리포트 — RAG 와 문서 없는 답을 같은 문항으로 견준다."""

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from datetime import date
from types import MappingProxyType

from kiwipiepy import Kiwi
from playground_evalkit import mcnemar_exact, render_table

from playground_wiki.evaluation.answer_measure import SidePair
from playground_wiki.evaluation.answer_metrics import (
    Share,
    contains_gold,
    format_share,
    gold_in_hits,
    matches_expected,
    rate_with_interval,
    score_band,
    share,
    stage_path,
)
from playground_wiki.evaluation.answer_report import point_text
from playground_wiki.evaluation.answer_run import AnswerRecord, EvalItem, settings_variants
from playground_wiki.evaluation.closed_book import (
    CLOSED_BOOK_STAGE,
    OUTCOMES,
    QUESTION_KINDS,
    QuestionKind,
    claims_general,
    declined,
    general_knowledge,
    gold_outcome,
    hedged,
    pair_by_id,
    question_kind,
)
from playground_wiki.evaluation.partial_judge import DOC_HEADING, sections

QUESTION_KIND_LABELS = MappingProxyType(
    {"titled": "제목 있음", "untitled": "제목 없음", "deictic": "지시어 질문"}
)
OUTCOME_LABELS = MappingProxyType(
    {
        "correct": "정답 포함",
        "unknown": "모름",
        "wrong": "답했는데 틀림",
        "failed": "실패",
    }
)
# 사용자와 정한 결론 (closed-book 리포트 결론 절). 정하기 전에는 None
CLOSED_BOOK_DECISION: str | None = None
type Pair = tuple[AnswerRecord, AnswerRecord]
"""같은 문항의 (RAG 기록, closed-book 기록)."""


def question_kinds(items: Iterable[EvalItem]) -> dict[str, QuestionKind]:
    """KorQuAD 문항 id → 문단 없이 뜻이 통하는지 (`question_kind` — Kiwi 품사로)."""
    kiwi = Kiwi()
    kinds: dict[str, QuestionKind] = {}
    for item in items:
        tags = [(token.form, token.tag) for token in kiwi.tokenize(item.question)]
        kinds[item.id] = question_kind(item.question, item.gold_title or "", tags)
    return kinds


def render_closed_book_report(
    f9: SidePair,
    longtail: SidePair | None,
    kinds: Mapping[str, QuestionKind],
    *,
    tag: str,
    rag_tag: str | None,
    today: date,
) -> str:
    """closed-book 과 RAG 를 같은 문항 짝으로 견준 리포트 markdown — 집계와 문항 id 만 (질문 · 답 원문 없음).

    Args:
      f9: F9 표본(KorQuAD + 위키 밖)의 RAG · closed-book 기록.
      longtail: 롱테일 세트의 두 기록. 아직 없으면 None — 그 묶음을 빼고 '아직'으로 적는다.
      kinds: KorQuAD 문항 id → 문단 없이 뜻이 통하는지.
      tag: closed-book 기록의 꼬리표.
      rag_tag: RAG 기록의 꼬리표. None 이면 꼬리표 없는 기록.
      today: 리포트 날짜.

    Raises:
      ValueError: 같은 세트의 두 기록이 문항이 다를 때 (한쪽 실행이 덜 끝남).
    """
    f9_pairs = pair_by_id(f9.rag, f9.closed)
    korquad = [pair for pair in f9_pairs if pair[0].item.source == "korquad"]
    sets: list[tuple[str, list[Pair]]] = [("KorQuAD", korquad)]
    if longtail is not None:
        sets.append(("롱테일", pair_by_id(longtail.rag, longtail.closed)))
    groups = _closed_book_groups(sets, kinds)
    return (
        f"# RAG 없이 기준선 — closed-book vs RAG ({today:%Y-%m-%d})\n\n"
        + _closed_book_overview(f9, longtail, tag=tag, rag_tag=rag_tag)
        + "\n## 정답 포함 · 모름 · 답했는데 틀림\n\n"
        + _closed_outcome_section(groups)
        + "\n## 같은 문항 짝 — 정답 포함\n\n"
        + _closed_pair_section(groups)
        + "\n## 결과 교차 — RAG (행) × closed-book (열)\n\n"
        + _closed_cross_section(sets)
        + "\n## RAG 가 망친 문항 — closed-book 만 정답 포함\n\n"
        + _lost_section(sets, kinds)
        + "\n## partial 일반 지식 칸 — 같은 문항의 closed-book 과\n\n"
        + _general_section(sets)
        + "\n## 위키 밖 질문 — 모름과 기대 단계\n\n"
        + _offwiki_closed_section([pair for pair in f9_pairs if pair[0].item.source == "offwiki"])
        + "\n## closed-book 답의 꼴\n\n"
        + _closed_format_section(f9, longtail)
        + "\n## 문단 없이 뜻이 안 통하는 KorQuAD 질문 — 가르는 법\n\n"
        + _question_kind_section(kinds)
        + "\n## 결론\n\n"
        + (CLOSED_BOOK_DECISION or "사용자와 정한다 — 아직\n")
    )


def _closed_book_groups(
    sets: Sequence[tuple[str, Sequence[Pair]]], kinds: Mapping[str, QuestionKind]
) -> list[tuple[str, list[Pair]]]:
    """표의 묶음: 세트마다 전부, KorQuAD 는 문단 의존 갈래(문항이 있는 것만)도."""
    groups: list[tuple[str, list[Pair]]] = []
    for name, pairs in sets:
        groups.append((name if name != "KorQuAD" else "KorQuAD 전부", list(pairs)))
        if name != "KorQuAD":
            continue
        for kind in QUESTION_KINDS:
            group = [pair for pair in pairs if kinds[pair[0].item.id] == kind]
            if group:
                groups.append((f"KorQuAD · {QUESTION_KIND_LABELS[kind]}", group))
    return groups


def _closed_book_overview(
    f9: SidePair, longtail: SidePair | None, *, tag: str, rag_tag: str | None
) -> str:
    closed = f9.closed_runs[-1]["settings"]
    profiles = f9.rag_runs[-1]["settings"]["profiles"]
    runs = [("F9 RAG", f9.rag_runs), ("F9 closed-book", f9.closed_runs)]
    if longtail is not None:
        runs += [("롱테일 RAG", longtail.rag_runs), ("롱테일 closed-book", longtail.closed_runs)]
    mixed = [name for name, group in runs if settings_variants(group) > 1]
    sources = Counter(record.item.source for record in f9.rag)
    return (
        "- 질문 (2026-10-30 결정): 같은 모델 · 설정으로 검색 없이 질문만 주고 답하게 해, RAG 가 보탠 몫을 같은 문항 짝으로 "
        "본다. **최종 수치는 RAG 기록이고 closed-book 은 비교 열이다**\n"
        f"- closed-book: `{closed['spec_id']}` (grounded 프로필과 같은 모델 · 설정) · 프롬프트 `{closed['prompt_version']}` — "
        '관문 · 검색 없이 질문만 user 턴으로, 모르면 "모름" 한 단어로 답하게 했다. 기록 이름 끝 '
        f"-{tag}\n"
        f"- RAG: `/ask` 와 같은 조립 — grounded `{' + '.join(profiles['grounded']['system'])}` · partial "
        f"`{' + '.join(profiles['partial']['system'])}`. 기록 이름 끝 "
        + (f"-{rag_tag}" if rag_tag else "없음 (꼬리표 없는 기록)")
        + "\n"
        f"- 세트: F9 KorQuAD {sources['korquad']} · F9 위키 밖 {sources['offwiki']} (f9-sample-v1), 롱테일 "
        + (f"{len(longtail.rag)}" if longtail is not None else "아직 (기록 없음)")
        + " (F4 결선 롱테일 세트의 답 있음 문항 — 덜 알려진 문서다. KorQuAD(2018 공개)와 위키는 학습 데이터에 들어 있을 "
        "가능성이 커 closed-book 이 높게 나올 수 있다)\n"
        "- 세는 법 (두 기록에 같은 기준): 정답 포함 = 답 어디에든 정답 문자열(표기 중 하나라도, 공백 무시 — evalkit "
        '`contains_answer` 와 같은 규칙). 모름 = 정해진 꼴만 — closed-book 은 첫 줄 "모름" 또는 빈 답, RAG 는 관문 · 거절 · '
        "고정 거절 문구, 또는 partial 의 두 칸 정해진 문구(문서 칸 '직접적인 내용은 없습니다' + 일반 지식 칸 '알 수 "
        "없습니다'뿐). 답했는데 틀림 = 나머지 — 대조할 문서가 없어 지어냄 판정 대신 쓰는 대용이다 (다른 말로 · 일부만 답함 · "
        "질문을 비껴감도 섞인 상한). 실패 = 답을 하나도 못 보냄\n"
        + (
            f"- **실행 사이에 설정이 바뀐 기록: {' · '.join(mixed)}** — 그 기록의 수치는 섞여 있다\n"
            if mixed
            else ""
        )
        + "- 같은 프롬프트로 다시 돌려도 답이 일부 갈린다 (Ollama GPU, temperature 0.1)\n"
        "- 질문 · 답 원문은 로컬 기록에만 둔다 (KorQuAD CC BY-ND) — 리포트에는 집계와 문항 id 만\n"
    )


def _closed_outcome_section(groups: Sequence[tuple[str, Sequence[Pair]]]) -> str:
    rows = []
    for name, pairs in groups:
        rows.append(_closed_outcome_row(name, "RAG", [rag for rag, _ in pairs]))
        rows.append(_closed_outcome_row(name, "closed-book", [closed for _, closed in pairs]))
    columns = [
        "묶음",
        "기록",
        "문항",
        *(OUTCOME_LABELS[name] for name in OUTCOMES),
        "답한 것 중 틀림 (95% 구간)",
    ]
    return render_table(rows, columns) + (
        "\n- 답한 것 중 틀림 = 답했는데 틀림 ÷ (정답 포함 + 답했는데 틀림) — 아는 척 틀린 비율 (지어냄 대용). 95% 구간은 "
        "Wilson\n"
        "- RAG 의 답했는데 틀림에는 grounded 지어냄과 partial 일반 지식 칸의 오답이 함께 든다\n"
    )


def _closed_outcome_row(
    group: str, side: str, records: Sequence[AnswerRecord]
) -> dict[str, str | int | float]:
    counts = Counter(gold_outcome(record) for record in records)
    return {
        "묶음": group,
        "기록": side,
        "문항": len(records),
        **{
            OUTCOME_LABELS[name]: format_share(Share(counts[name], len(records)))
            for name in OUTCOMES
        },
        "답한 것 중 틀림 (95% 구간)": rate_with_interval(
            counts["wrong"], counts["correct"] + counts["wrong"]
        ),
    }


def _closed_pair_section(groups: Sequence[tuple[str, Sequence[Pair]]]) -> str:
    rows = [_closed_pair_row(name, pairs) for name, pairs in groups]
    columns = [
        "묶음",
        "짝",
        "RAG 정답 포함",
        "closed-book 정답 포함",
        "차이 (%p)",
        "둘 다",
        "RAG 만",
        "closed-book 만",
        "둘 다 아님",
        "McNemar p",
    ]
    return render_table(rows, columns) + (
        "\n- RAG 만 = RAG 가 보탠 몫. closed-book 만 = RAG 가 망친 문항 (검색이 엉뚱한 문서를 가져왔거나 받은 문서를 잘못 "
        "읽음 — 아래 절에 문항 id). McNemar p 는 엇갈린 짝(한쪽만 정답 포함)만 본 정확 검정\n"
    )


def _closed_pair_row(group: str, pairs: Sequence[Pair]) -> dict[str, str | int | float]:
    rag = [gold_outcome(record) == "correct" for record, _ in pairs]
    closed = [gold_outcome(record) == "correct" for _, record in pairs]
    test = mcnemar_exact(rag, closed)
    both = sum(first and second for first, second in zip(rag, closed, strict=True))
    rag_share = Share(sum(rag), len(pairs))
    closed_share = Share(sum(closed), len(pairs))
    return {
        "묶음": group,
        "짝": len(pairs),
        "RAG 정답 포함": format_share(rag_share),
        "closed-book 정답 포함": format_share(closed_share),
        "차이 (%p)": point_text(rag_share.rate, closed_share.rate),
        "둘 다": both,
        "RAG 만": test.only_a,
        "closed-book 만": test.only_b,
        "둘 다 아님": len(pairs) - both - test.only_a - test.only_b,
        "McNemar p": f"{test.p_value:.3g}",
    }


def _closed_cross_section(sets: Sequence[tuple[str, Sequence[Pair]]]) -> str:
    columns = [f"closed-book {OUTCOME_LABELS[name]}" for name in OUTCOMES]
    rows = []
    for name, pairs in sets:
        counts = Counter((gold_outcome(rag), gold_outcome(closed)) for rag, closed in pairs)
        for first in OUTCOMES:
            cells = {
                column: counts[first, second]
                for column, second in zip(columns, OUTCOMES, strict=True)
            }
            rows.append({"묶음": name, "RAG": OUTCOME_LABELS[first], **cells})
    return render_table(rows, ["묶음", "RAG", *columns]) + (
        "\n- RAG 모름 · closed-book 정답 포함 = 위키 경로는 답을 내지 않았는데 모델은 알던 문항. RAG 답했는데 틀림 · "
        "closed-book 모름 = 문서를 받고 아는 척 틀린 문항\n"
    )


def _lost_section(
    sets: Sequence[tuple[str, Sequence[Pair]]], kinds: Mapping[str, QuestionKind]
) -> str:
    rows = []
    for name, pairs in sets:
        for rag, closed in pairs:
            result = gold_outcome(rag)
            if gold_outcome(closed) != "correct" or result == "correct":
                continue
            kind = kinds.get(rag.item.id)
            rows.append(
                {
                    "id": rag.item.id,
                    "묶음": name if kind is None else f"{name} · {QUESTION_KIND_LABELS[kind]}",
                    "RAG 경로": stage_path(rag),
                    "RAG 결과": OUTCOME_LABELS[result or "failed"],
                    "검색 상위에 정답 근거": "있음" if gold_in_hits(rag) else "없음",
                    "1위 점수": score_band(rag.top_score),
                }
            )
    if not rows:
        return "없음\n"
    missed = sum(row["검색 상위에 정답 근거"] == "없음" for row in rows)
    columns = ["id", "묶음", "RAG 경로", "RAG 결과", "검색 상위에 정답 근거", "1위 점수"]
    return render_table(rows, columns) + (
        f"\n- {len(rows)}문항 중 검색 상위에 정답 근거 청크가 없던 것 {missed} (검색이 정답 문서 · 문장을 못 가져옴 — "
        f"관문이 받아 검색하지 않은 것 포함), 있던 것 {len(rows) - missed} (받고도 못 씀 — 잘못 읽음 · 모름으로 물러섬 · "
        "질문을 비껴감). 정답 근거 청크 = 정답 문서이면서 정답 문자열을 담은 청크 (검색 평가와 같은 기준)\n"
    )


def _general_section(sets: Sequence[tuple[str, Sequence[Pair]]]) -> str:
    rows = [_general_row(name, pairs) for name, pairs in sets]
    columns = [
        "묶음",
        "partial 로 끝남",
        "문서 칸 정답 포함",
        "일반 지식 칸이 답함",
        "일반 지식 칸 정답 포함",
        "같은 문항 closed-book 정답 포함",
        "일반 지식 칸만",
        "closed-book 만",
    ]
    return render_table(rows, columns) + (
        "\n- 대상: RAG 기록에서 partial 로 끝난 문항. 일반 지식 칸이 답함 = 그 칸이 비지 않았고 '알 수 없습니다'뿐이 아님\n"
        "- 오른쪽 넷은 일반 지식 칸이 답한 문항만 — 그 칸은 사실상 closed-book 답이라, 같은 문항의 closed-book 답과 정답 "
        "포함을 견준다. 차이는 문서를 함께 받았는지 · 프롬프트(partial@v2 는 '확실하지 않으면 밝힌다', closed-book 은 "
        "'모르면 모름')다\n"
    )


def _general_row(group: str, pairs: Sequence[Pair]) -> dict[str, str | int | float]:
    partial: list[tuple[AnswerRecord, AnswerRecord, str]] = []
    for rag, closed in pairs:
        general = general_knowledge(rag)
        if general is not None:
            partial.append((rag, closed, general))
    claimed = [entry for entry in partial if claims_general(entry[2])]
    general_gold = [
        bool(contains_gold(replace(rag, answer=general))) for rag, _, general in claimed
    ]
    closed_gold = [gold_outcome(closed) == "correct" for _, closed, _ in claimed]
    document_gold = sum(
        bool(contains_gold(replace(rag, answer=sections(rag.answer).get(DOC_HEADING, ""))))
        for rag, _, _ in partial
    )
    pairs_gold = list(zip(general_gold, closed_gold, strict=True))
    return {
        "묶음": group,
        "partial 로 끝남": len(partial),
        "문서 칸 정답 포함": document_gold,
        "일반 지식 칸이 답함": len(claimed),
        "일반 지식 칸 정답 포함": format_share(share(general_gold)),
        "같은 문항 closed-book 정답 포함": format_share(share(closed_gold)),
        "일반 지식 칸만": sum(general and not closed for general, closed in pairs_gold),
        "closed-book 만": sum(closed and not general for general, closed in pairs_gold),
    }


def _offwiki_closed_section(pairs: Sequence[Pair]) -> str:
    if not pairs:
        return "기록이 없다\n"
    groups: dict[str, list[Pair]] = defaultdict(list)
    for pair in pairs:
        groups[str(pair[0].item.expected)].append(pair)
    rows = [
        {
            "기대": expected,
            "문항": len(group),
            "closed-book 모름": format_share(share(declined(closed) for _, closed in group)),
            "closed-book 실패": sum(closed.error is not None for _, closed in group),
            "RAG 답을 내지 않음": format_share(share(declined(rag) for rag, _ in group)),
            "RAG 기대대로": format_share(share(matches_expected(rag) for rag, _ in group)),
        }
        for expected, group in sorted(groups.items())
    ]
    columns = [
        "기대",
        "문항",
        "closed-book 모름",
        "closed-book 실패",
        "RAG 답을 내지 않음",
        "RAG 기대대로",
    ]
    return render_table(rows, columns) + (
        "\n- 정답이 없는 질문이라 모름 · 답함만 센다. refuse 기대(개인 정보 · 실시간 · 잡담)는 답하지 않는 것이 맞는 길이고, "
        "partial 기대는 일반 지식으로 답해도 된다\n"
        "- RAG 답을 내지 않음 = 관문 · 거절 · 고정 거절 문구 · partial 두 칸 정해진 문구. RAG 기대대로는 F9 리포트와 같은 "
        "규칙 (refuse → 관문 · 거절 · 고정 거절 문구, partial → partial)\n"
    )


def _closed_format_section(f9: SidePair, longtail: SidePair | None) -> str:
    sides = [("F9", f9.closed)]
    if longtail is not None:
        sides.append(("롱테일", longtail.closed))
    rows = [
        {
            "기록": name,
            "답": len(records),
            "'모름' 꼴": sum(map(declined, records)),
            "꼴 밖에서 모른다고 함": sum(map(hedged, records)),
            "잘림 (max_tokens)": sum(
                r.done_reason.get(CLOSED_BOOK_STAGE) == "length" for r in records
            ),
            "시간 초과": sum(r.error == "timeout" for r in records),
            "생성 실패": sum(r.error == "generation" for r in records),
        }
        for name, records in sides
    ]
    columns = [
        "기록",
        "답",
        "'모름' 꼴",
        "꼴 밖에서 모른다고 함",
        "잘림 (max_tokens)",
        "시간 초과",
        "생성 실패",
    ]
    return render_table(rows, columns) + (
        "\n- 꼴 밖에서 모른다고 함 = '모름' 한 단어가 아니라 문장으로 모른다고 한 답 (정규식 '모르 · 알 수 없 · 알지 못 · "
        "확실하지 않' — 답한 뒤 곁가지를 모른다고 한 것도 걸리는 상한). 이 답은 정답 포함이나 답했는데 틀림으로 셌다 — 많으면 "
        "프롬프트의 꼴 지시를 손본다\n"
    )


def _question_kind_section(kinds: Mapping[str, QuestionKind]) -> str:
    counts = Counter(kinds.values())
    return (
        "- 규칙 (Kiwi 품사, `closed_book.question_kind`): 지시어 질문 = 문단 안의 대상을 가리키는 대명사(그 · 그녀 · "
        '이것 …) · 관형사(이 · 그 · 해당)가 있음 — 문단 없이는 무엇을 묻는지 모른다 ("그가 태어난 해는?" 꼴). 제목 있음 = '
        "그런 말 없이 정답 문서 제목(동음이의 꼬리를 뗀 것, 공백 무시)을 담음 — 묻는 대상이 질문에 있다. 제목 없음 = 나머지\n"
        f"- KorQuAD {len(kinds)}: "
        + " · ".join(f"{QUESTION_KIND_LABELS[kind]} {counts[kind]}" for kind in QUESTION_KINDS)
        + "\n"
        "- 지시어 질문은 확실히 문단 의존이지만 드물다. 제목 없음에는 다른 이름으로 대상을 밝힌 질문과, 연도 · 주어가 빠져 "
        "문단을 전제한 질문이 섞여 있다 — closed-book 이 낮은 몫이 문단 의존 때문인지 지식 때문인지는 제목 있음 묶음으로 "
        "가린다. 사람 · 판정 모델로 문항마다 가르면 더 정확하지만 이 리포트는 규칙만 쓴다\n"
    )
