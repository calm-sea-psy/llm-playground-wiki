"""RAG 없이 기준선(closed-book): 검색 없이 질문만 주고 답한 기록을 만들고, RAG 기록과 같은 기준으로 센다.

closed-book 답은 대조할 문서가 없어 지어냄 판정 대신 정답 포함 · '모름' · 답했는데 틀림으로 센다 (2026-10-30 결정).
비교하는 RAG 기록(`/ask` 경로)도 같은 기준으로 다시 센다. '모름'은 문장 매칭이 아니라 정해진 꼴로만 본다 —
closed-book 은 "모름" 한 단어(closed_book 프롬프트가 시킨다), partial 은 두 칸의 정해진 문구(partial@v2), 관문 ·
거절 · 고정 거절 문구는 답을 내지 않은 것.

evaluation 은 generation · api 를 import 하지 않는다 — 시스템 프롬프트와 모델 설정은 scripts/ 가 골라 넘긴다.
"""

import asyncio
import re
import time
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, get_args

from playground_evalkit import squash
from playground_llm import LLM, GenerationError, Message, ModelNotFoundError, ModelSpec
from playground_retrieval.search import SearchResult

from playground_wiki.evaluation.answer_metrics import FIXED_STAGES, contains_gold, final_stage
from playground_wiki.evaluation.answer_run import AnswerRecord
from playground_wiki.evaluation.partial_judge import (
    DOC_HEADING,
    GENERAL_HEADING,
    NO_DIRECT,
    UNKNOWN,
    sections,
)
from playground_wiki.evaluation.wikidata import strip_qualifier

CLOSED_BOOK_STAGE = "closed_book"
"""closed-book 기록의 stage_path 값 — 답변 경로에는 없는 단계 이름."""
UNKNOWN_REPLY = "모름"
"""모르면 이 한 단어만 쓰라고 closed_book 프롬프트가 시킨다."""
Outcome = Literal["correct", "unknown", "wrong", "failed"]
"""정답이 있는 문항 하나의 결과: 정답 포함 · 모름 · 답했는데 틀림 · 실패(답을 하나도 못 보냄)."""
OUTCOMES: tuple[str, ...] = get_args(Outcome)
QuestionKind = Literal["titled", "untitled", "deictic"]
"""KorQuAD 질문이 문단 없이 뜻이 통하는지 (`question_kind`)."""
QUESTION_KINDS: tuple[str, ...] = get_args(QuestionKind)
# 문단 안의 대상을 가리키는 말 (Kiwi 품사 NP 대명사 · MM 관형사). "그가 태어난 해는?" · "이 영화의 감독은?" 꼴.
# 의문 대명사(무엇 · 누구) · 수 관형사(몇 · 두)는 넣지 않는다
DEICTIC_PRONOUNS = frozenset(
    {"그", "그녀", "그들", "이", "이들", "저", "저들"}  # 사람
    | {"이것", "그것", "저것", "이곳", "그곳", "거기", "여기"}  # 사물 · 곳
)
DEICTIC_DETERMINERS = frozenset({"이", "그", "저", "해당"})
TIMEOUT_MESSAGE = "답을 만드는 데 시간이 너무 오래 걸렸습니다."
_TRAILING = re.compile(r"[\s.!。]+$")
# '모름' 꼴 밖에서 모른다고 말한 답 (정규식이라 상한 — 답한 뒤 곁가지를 모른다고 한 것도 걸린다)
_HEDGED = re.compile(r"모르|모릅|알 수 없|알지 못|확실하지 않")


@dataclass(frozen=True)
class ClosedBookEvent:
    """closed-book 답 하나의 SSE 꼴 이벤트 (`AnswerRun` 이 기록 한 줄로 모은다)."""

    name: str
    data: Mapping[str, object]

    def __post_init__(self) -> None:
        # frozen 이라도 dict 안쪽은 바뀐다 — 만들 때 읽기 전용으로 감싼다
        object.__setattr__(self, "data", MappingProxyType(dict(self.data)))


class NoSearch:
    """closed-book 실행의 검색기 자리 — `AnswerRun` 이 검색기를 받지만 closed-book 은 검색하지 않는다."""

    async def search(self, query: str) -> SearchResult:
        """부르면 실패한다.

        Raises:
          RuntimeError: 늘.
        """
        del query  # 검색기 계약 맞춤. 부르는 것 자체가 잘못이다
        raise RuntimeError("closed-book answers never search")


class ClosedBookAnswerer:
    """질문만 LLM 에 넘겨(검색 · 문서 없이) 답 하나를 답변 경로와 같은 꼴의 이벤트로 흘린다.

    이벤트: stage(closed_book) → delta(답 전체, 빈 답이면 없음) → done, 실패면 error (generation · timeout).
    done 진단에 spec_id · prompt_version · done_reason · 시간을 싣는다 — 출처 · 표지 · 1위 점수는 없다.

    예)
        answerer = ClosedBookAnswerer(llm, profile.spec, system=prompt.text,
                                      prompt_version=prompt.version, timeout_sec=60)
        run = AnswerRun(NoSearch(), lambda search: answerer)
    """

    def __init__(
        self,
        llm: LLM,
        spec: ModelSpec,
        *,
        system: str,
        prompt_version: str,
        timeout_sec: float,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._llm = llm
        self._spec = spec
        self._system = system
        self._prompt_version = prompt_version
        self._timeout_sec = timeout_sec
        self._clock = clock

    async def answer(self, question: str) -> AsyncIterator[ClosedBookEvent]:
        """질문 하나의 이벤트. user 턴은 질문 원문 그대로다."""
        start = self._clock()
        yield ClosedBookEvent("stage", {"stage": CLOSED_BOOK_STAGE})
        messages = (Message("system", self._system), Message("user", question))
        try:
            # generate 는 동기라 작업 스레드에서. 시간이 넘으면 기다리기만 그만둔다 (생성은 max_tokens 에서 끝난다).
            # GenerationTimeoutError 도 TimeoutError 다
            async with asyncio.timeout(self._timeout_sec):
                generation = await asyncio.to_thread(self._llm.generate, self._spec, messages)
        except TimeoutError:
            yield ClosedBookEvent("error", {"code": "timeout", "message": TIMEOUT_MESSAGE})
            return
        except (GenerationError, ModelNotFoundError) as e:
            yield ClosedBookEvent("error", {"code": "generation", "message": str(e)})
            return
        total_ms = round((self._clock() - start) * 1000, 1)
        text = generation.text.strip()
        if text:
            yield ClosedBookEvent("delta", {"text": text})
        diagnostics = {
            "stage_path": [CLOSED_BOOK_STAGE],
            "gate": None,
            "top_score": None,
            "marker": None,
            "done_reason": {CLOSED_BOOK_STAGE: generation.done_reason},
            "spec_id": self._spec.spec_id,
            "prompt_version": self._prompt_version,
            # 스트리밍하지 않아 첫 글자 시간은 전체 시간이다
            "timings_ms": {
                "search": None,
                "first_delta": total_ms if text else None,
                "total": total_ms,
            },
        }
        yield ClosedBookEvent(
            "done",
            {
                "stage_path": [CLOSED_BOOK_STAGE],
                "cited": [],
                "dropped": [],
                "truncated": generation.done_reason == "length",
                "timed_out": False,
                "diagnostics": diagnostics,
            },
        )


def declined(record: AnswerRecord) -> bool:
    """답을 내지 않았나 — 프롬프트 · 답변 경로가 정한 꼴로만 본다 (문장 매칭이 아니다).

    closed-book: 첫 줄이 "모름"(끝 마침표 무시)이거나 빈 답. RAG: 관문 · 거절 · 고정 거절 문구로 끝남, 또는 partial 이
    문서 칸을 NO_DIRECT 로 시작하고 일반 지식 칸이 UNKNOWN 뿐. grounded 로 끝난 답은 답을 낸 것이다.
    실패(error)로 끝난 기록은 False — `gold_outcome` 이 실패로 따로 센다.
    """
    if record.error is not None:
        return False
    stage = final_stage(record)
    if stage in FIXED_STAGES:
        return True
    if stage == CLOSED_BOOK_STAGE:
        lines = record.answer.strip().splitlines()
        return not lines or _bare(lines[0]) == UNKNOWN_REPLY
    if stage == "partial":
        parts = sections(record.answer)
        return parts.get(DOC_HEADING, "").startswith(NO_DIRECT) and _bare(
            parts.get(GENERAL_HEADING, "")
        ) == _bare(UNKNOWN)
    return False


def gold_outcome(record: AnswerRecord) -> Outcome | None:
    """정답이 있는 문항의 결과. 정답이 없는 문항(위키 밖)이면 None — `declined` 로 모름 · 답함만 센다.

    정답 문자열(표기 중 하나라도)이 답 어디에든 있으면 정답 포함이 먼저다 — F9 의 '어느 단계든 정답 포함'과 같은 규칙.
    """
    gold = contains_gold(record)
    if gold is None:
        return None
    if record.error is not None:
        return "failed"
    if gold:
        return "correct"
    return "unknown" if declined(record) else "wrong"


def hedged(record: AnswerRecord) -> bool:
    """답을 낸 것으로 센 답이 '모름' 꼴 밖에서 모른다고 말하나 (정규식 — 꼴을 지켰는지 보는 상한)."""
    return (
        record.error is None and not declined(record) and _HEDGED.search(record.answer) is not None
    )


def general_knowledge(record: AnswerRecord) -> str | None:
    """일반 지식 칸 본문 — partial 로 끝난 답만 (칸이 없으면 빈 문자열). partial 로 끝나지 않았으면 None."""
    if record.error is not None or final_stage(record) != "partial":
        return None
    return sections(record.answer).get(GENERAL_HEADING, "")


def claims_general(text: str) -> bool:
    """일반 지식 칸이 무언가 답했나 — 비지 않았고 UNKNOWN 한 문구뿐이 아니다."""
    return bool(text.strip()) and _bare(text) != _bare(UNKNOWN)


def question_kind(question: str, gold_title: str, tags: Sequence[tuple[str, str]]) -> QuestionKind:
    """KorQuAD 질문이 문단 없이 뜻이 통하는지 가르는 규칙.

    deictic: 문단 안의 대상을 가리키는 대명사 · 관형사가 있다 ("그가 태어난 해는?") — 문단 없이는 무엇을 묻는지 모른다.
    titled: 그런 말이 없고 정답 문서 제목(동음이의 꼬리를 뗀 것, 공백 무시)을 담았다 — 묻는 대상이 질문에 있다.
    untitled: 나머지 — 다른 이름으로 대상을 밝힌 질문과, 연도 · 주어가 빠져 문단을 전제한 질문이 섞여 있다.

    Args:
      question: 질문 원문.
      gold_title: 정답 문서 제목.
      tags: 질문의 형태소 (형태, 품사) — Kiwi 품사 표지 (NP 대명사 · MM 관형사).
    """
    if any(
        (tag == "NP" and form in DEICTIC_PRONOUNS) or (tag == "MM" and form in DEICTIC_DETERMINERS)
        for form, tag in tags
    ):
        return "deictic"
    title = strip_qualifier(gold_title)
    return "titled" if squash(title) in squash(question) else "untitled"


def pair_by_id(
    first: Sequence[AnswerRecord], second: Sequence[AnswerRecord]
) -> list[tuple[AnswerRecord, AnswerRecord]]:
    """같은 문항끼리 (first, second) 짝 (id 순).

    Raises:
      ValueError: 두 기록의 문항이 다를 때 (한쪽 실행이 덜 끝남).
    """
    after = {record.item.id: record for record in second}
    if {record.item.id for record in first} != after.keys():
        raise ValueError(f"records cover different items: {len(first)} vs {len(second)}")
    return [(record, after[record.item.id]) for record in sorted(first, key=lambda r: r.item.id)]


def _bare(text: str) -> str:
    return _TRAILING.sub("", text.strip())
