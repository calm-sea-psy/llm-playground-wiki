from typing import get_args

from playground_wiki.agent import STEPS
from playground_wiki.api.answer import (
    AGENT_STAGE_MESSAGES,
    AGENT_STAGE_PREFIX,
    STAGE_MESSAGES,
    AnswerEvent,
)
from playground_wiki.api.schemas import (
    ASK_EVENTS,
    AgentStageName,
    DeltaEvent,
    EventName,
    StageName,
)


def test_every_event_name_has_a_data_model():
    assert tuple(ASK_EVENTS) == get_args(EventName)


def test_stage_names_are_the_messages_keys():
    assert set(STAGE_MESSAGES) | set(AGENT_STAGE_MESSAGES) == set(get_args(StageName))
    # 답변 근거 고르기(compose)는 바로 끝나 stage 이벤트가 없다
    nodes = {f"{AGENT_STAGE_PREFIX}{step}" for step in STEPS if step != "compose"}
    assert set(get_args(AgentStageName)) == nodes


def test_event_name_follows_the_model():
    event = AnswerEvent.of(DeltaEvent(text="답"))

    assert event == AnswerEvent("delta", {"text": "답"})
