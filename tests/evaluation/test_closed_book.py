import asyncio
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import pytest
from playground_llm import (
    LLM,
    FakeLLM,
    Generation,
    GenerationError,
    Message,
    ModelSpec,
    PromptLibrary,
)

from playground_wiki.evaluation.answer_run import AnswerRecord, AnswerRun, EvalItem, read_records
from playground_wiki.evaluation.closed_book import (
    CLOSED_BOOK_STAGE,
    UNKNOWN_REPLY,
    ClosedBookAnswerer,
    NoSearch,
    claims_general,
    declined,
    general_knowledge,
    gold_outcome,
    hedged,
    pair_by_id,
    question_kind,
)
from playground_wiki.evaluation.partial_judge import (
    DOC_HEADING,
    GENERAL_HEADING,
    NO_DIRECT,
    UNKNOWN,
)
from playground_wiki.generation import PROMPT_DIR

SPEC = ModelSpec("ollama", "fake", {"think": "off", "max_tokens": 64})
PALACE = EvalItem(
    id="k1",
    source="korquad",
    question="경복궁을 중건한 사람은?",
    gold_title="경복궁",
    gold_answer="흥선대원군",
)
PERSONAL = EvalItem(
    id="o1", source="offwiki", question="내 월급날은?", expected="refuse", category="personal"
)


def _answerer(llm: LLM, *, timeout_sec: float = 5) -> ClosedBookAnswerer:
    return ClosedBookAnswerer(
        llm,
        SPEC,
        system="지시",
        prompt_version="closed_book@v1:abc",
        timeout_sec=timeout_sec,
    )


def _run(llm: LLM, items: Sequence[EvalItem], path: Path, **options: float) -> list[AnswerRecord]:
    answerer = _answerer(llm, **options)
    run = AnswerRun(NoSearch(), lambda search: answerer)
    asyncio.run(run.run(items, path))
    return read_records(path)


def test_closed_book_answer_is_recorded_without_search_or_sources(tmp_path: Path):
    llm = FakeLLM(["흥선대원군이 중건했다.", UNKNOWN_REPLY])

    first, second = _run(llm, [PALACE, PERSONAL], tmp_path / "a.jsonl")

    assert first.stage_path == (CLOSED_BOOK_STAGE,)
    assert first.answer == "흥선대원군이 중건했다."
    assert (first.hits, first.context_count, first.cited, first.marker) == ((), 0, (), None)
    assert (first.spec_id, first.prompt_version) == (SPEC.spec_id, "closed_book@v1:abc")
    assert first.done_reason == {CLOSED_BOOK_STAGE: "stop"}
    assert first.top_score is None
    assert second.answer == UNKNOWN_REPLY
    # user 턴은 질문 원문 그대로, system 턴은 넘긴 지시
    assert llm.calls[0].messages == (
        Message("system", "지시"),
        Message("user", "경복궁을 중건한 사람은?"),
    )


class _Failing(FakeLLM):
    def __init__(self) -> None:
        super().__init__(["안 쓰임"])

    def generate(
        self,
        spec: ModelSpec,
        messages: Sequence[Message],
        *,
        json_schema: Mapping[str, object] | None = None,
    ) -> Generation:
        raise GenerationError("server down")


class _Slow(FakeLLM):
    def __init__(self) -> None:
        super().__init__(["늦은 답"])

    def generate(
        self,
        spec: ModelSpec,
        messages: Sequence[Message],
        *,
        json_schema: Mapping[str, object] | None = None,
    ) -> Generation:
        time.sleep(0.3)
        return super().generate(spec, messages, json_schema=json_schema)


def test_generation_failure_and_timeout_are_recorded_as_errors(tmp_path: Path):
    (failed,) = _run(_Failing(), [PALACE], tmp_path / "failed.jsonl")
    (slow,) = _run(_Slow(), [PALACE], tmp_path / "slow.jsonl", timeout_sec=0.05)

    assert (failed.error, failed.answer) == ("generation", "")
    assert (slow.error, slow.answer) == ("timeout", "")
    assert gold_outcome(failed) == "failed"


def test_no_search_refuses_to_search():
    with pytest.raises(RuntimeError, match="never search"):
        asyncio.run(NoSearch().search("q"))


def test_closed_book_prompt_asks_for_the_unknown_reply_and_takes_only_today():
    template = PromptLibrary.load(PROMPT_DIR).get("closed_book@v1")

    assert template.variables == frozenset({"today"})
    assert f'"{UNKNOWN_REPLY}"' in template.text


def _closed(answer: str, item: EvalItem = PALACE) -> AnswerRecord:
    return AnswerRecord(
        item=item,
        stage_path=(CLOSED_BOOK_STAGE,),
        answer=answer,
        hits=(),
        context_count=0,
        marker=None,
        cited=(),
        dropped=(),
        truncated=False,
        timed_out=False,
        error=None,
        top_score=None,
        gate_rule=None,
        spec_id="ollama:fake",
        prompt_version="closed_book@v1:abc",
        search_ms=None,
        first_delta_ms=10.0,
        total_ms=10.0,
        wall_ms=11.0,
        done_reason={CLOSED_BOOK_STAGE: "stop"},
    )


def _partial(doc: str, general: str) -> AnswerRecord:
    answer = f"{DOC_HEADING}\n{doc}\n\n{GENERAL_HEADING}\n{general}"
    return replace(_closed(answer), stage_path=("grounded", "partial"), done_reason={})


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        (_closed("흥선 대원군입니다."), "correct"),
        (_closed("모름"), "unknown"),
        (_closed("모름."), "unknown"),
        (_closed(""), "unknown"),
        (_closed("고종입니다."), "wrong"),
        (_closed("잘 모르겠지만 고종입니다."), "wrong"),  # 꼴 밖의 모른다는 말은 답한 것
        (_partial(f"{NO_DIRECT}. [2]", f"{UNKNOWN}."), "unknown"),
        (_partial(f"{NO_DIRECT}.", "고종이 중건했다."), "wrong"),
        (_partial(f"{NO_DIRECT}.", "흥선대원군이 중건했다."), "correct"),
        (replace(_closed("x"), stage_path=("refuse",)), "unknown"),
        (replace(_closed("x"), stage_path=("gate",)), "unknown"),
        (replace(_closed("x"), stage_path=("partial",), first_delta_ms=None), "unknown"),
        (replace(_closed("x"), stage_path=("grounded",)), "wrong"),
        (replace(_closed(""), error="timeout"), "failed"),
    ],
)
def test_outcome_counts_unknown_only_in_the_fixed_forms(record: AnswerRecord, expected: str):
    assert gold_outcome(record) == expected


def test_offwiki_items_have_no_outcome_but_declined_still_applies():
    assert gold_outcome(_closed("모름", PERSONAL)) is None
    assert declined(_closed("모름", PERSONAL)) is True
    assert declined(_closed("25일입니다.", PERSONAL)) is False


def test_hedged_catches_free_form_unknown_answers_only():
    assert hedged(_closed("정확히는 알 수 없지만 고종입니다."))
    assert not hedged(_closed("모름"))
    assert not hedged(_closed("고종입니다."))


def test_general_knowledge_is_the_general_part_of_a_partial_answer():
    record = _partial(f"{NO_DIRECT}.", "흥선대원군이다.")

    assert general_knowledge(record) == "흥선대원군이다."
    assert general_knowledge(_closed("고종")) is None
    assert claims_general("흥선대원군이다.")
    assert not claims_general(f"{UNKNOWN}.")
    assert not claims_general("  ")


def _tags(pairs: Mapping[str, str]) -> list[tuple[str, str]]:
    return list(pairs.items())


@pytest.mark.parametrize(
    ("question", "title", "tags", "kind"),
    [
        ("그가 태어난 해는?", "홍길동", {"그": "NP", "가": "JKS", "해": "NNG"}, "deictic"),
        ("이 영화의 감독은?", "괴물 (영화)", {"이": "MM", "영화": "NNG"}, "deictic"),
        ("괴물의 감독은?", "괴물 (영화)", {"괴물": "NNG", "감독": "NNG"}, "titled"),
        # 의문 대명사는 지시어가 아니다. 제목은 공백을 무시하고 찾는다
        ("경복궁 을 중건한 사람은?", "경복궁", {"누구": "NP"}, "titled"),
        ("대부분의 현장에서 목격된 차량은?", "테드 번디", {"대부분": "NNG"}, "untitled"),
    ],
)
def test_question_kind(question: str, title: str, tags: Mapping[str, str], kind: str):
    assert question_kind(question, title, _tags(tags)) == kind


def test_paired_needs_the_same_items():
    first = [_closed("a"), _closed("b", PERSONAL)]
    second = [_closed("c", PERSONAL), _closed("d")]

    assert [(a.answer, b.answer) for a, b in pair_by_id(first, second)] == [("a", "d"), ("b", "c")]
    with pytest.raises(ValueError, match="different items"):
        pair_by_id(first, second[:1])
