import math
from pathlib import Path

import pytest
import yaml
from playground_corpus import Chunk
from playground_llm import Message, PromptLibrary, Think
from playground_llm.ollama import chat_request
from playground_retrieval.core import Hit

from playground_wiki.api.config import (
    ANSWER_CONFIG,
    load_answer_policy,
    load_dump_date,
    load_query_gate,
    log_questions_from_env,
    parse_profiles,
)
from playground_wiki.gate import KiwiTokenizer, Token, fixed_reply
from playground_wiki.generation import (
    PROMPT_DIR,
    AnswerPolicy,
    Thresholds,
    build_prompt,
    check_profiles,
)

CONFIG_DIR = Path(__file__).parents[2] / "configs"
ROW = {
    "provider": "ollama",
    "model": "gemma4:12b",
    "params": {"think": "off", "max_tokens": 512, "stop": ["\n\n"]},
    "system": ["base@v1", "grounded@v1"],
    "user": "question@v1",
    "timeout_sec": 60,
}


def _write(directory: Path, thresholds: dict, profiles: dict) -> Path:
    text = yaml.safe_dump({"thresholds": thresholds, "profiles": profiles}, allow_unicode=True)
    (directory / ANSWER_CONFIG).write_text(text, encoding="utf-8")
    return directory


def _repo_profiles() -> dict:
    raw = yaml.safe_load((CONFIG_DIR / ANSWER_CONFIG).read_text(encoding="utf-8"))
    return parse_profiles(raw["profiles"])


def test_repo_answer_yaml_profiles_build_a_policy():
    profiles = _repo_profiles()

    # partial 도 think off — think on · 2048 은 위키 밖 질문 대부분에 빈 답 (ADR-025)
    assert profiles["grounded"].spec.think is Think.OFF
    assert profiles["partial"].spec.think is Think.OFF
    assert all(profile.system[0].startswith("base@") for profile in profiles.values())
    AnswerPolicy(Thresholds(high=1.0, low=0.0), profiles)  # 단계 구성이 맞는지


def test_repo_answer_yaml_has_calibrated_thresholds():
    policy = load_answer_policy(CONFIG_DIR)

    assert 0 < policy.thresholds.low < policy.thresholds.high < 1


def test_repo_profiles_point_to_repo_prompts():
    profiles = _repo_profiles()
    library = PromptLibrary.load(PROMPT_DIR)
    hit = Hit(
        chunk=Chunk(
            id="1-0", doc_id="1", title="서울", section="", text="서울\n한양", seq=0, token_count=3
        ),
        score=0.95,
    )

    check_profiles(profiles.values(), library)
    for profile in profiles.values():
        prompt = build_prompt(
            profile,
            library,
            question="조선의 수도는?",
            context=[hit],
            today="2026년 10월 24일",
            dump_date="2026년 10월 1일",
        )
        assert "조선의 수도는?" in prompt.messages[1].content
    grounded = build_prompt(
        profiles["grounded"], library, question="q", context=[hit], today="t", dump_date="d"
    )
    assert '"못 찾음"' in grounded.messages[0].content  # grounded@v2 — 첫 줄 표지 (ADR-024)


def test_load_builds_policy_and_turns_yaml_lists_into_tuples(tmp_path: Path):
    config_dir = _write(tmp_path, {"high": 0.99, "low": 0.5}, {"grounded": ROW, "partial": ROW})

    policy = load_answer_policy(config_dir)

    assert policy.thresholds == Thresholds(high=0.99, low=0.5)
    grounded = policy.profiles["grounded"]
    assert grounded.spec.params["stop"] == ("\n\n",)
    assert (grounded.system, grounded.user) == (("base@v1", "grounded@v1"), "question@v1")


def test_load_refuses_uncalibrated_thresholds(tmp_path: Path):
    config_dir = _write(tmp_path, {"high": None, "low": None}, {"grounded": ROW, "partial": ROW})

    with pytest.raises(ValueError, match="not calibrated"):
        load_answer_policy(config_dir)


def test_unquoted_off_in_yaml_is_rejected(tmp_path: Path):
    # YAML 은 따옴표 없는 off 를 false 로 읽는다 — ModelSpec 이 bool 을 거부해 바로 드러난다
    row = (
        "{provider: ollama, model: m, params: {think: off, max_tokens: 512}, system: [base@v1], "
        "user: question@v1, timeout_sec: 60}"
    )
    text = f"thresholds: {{high: 0.9, low: 0.5}}\nprofiles:\n  grounded: {row}\n  partial: {row}\n"
    (tmp_path / ANSWER_CONFIG).write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="think must be one of"):
        load_answer_policy(tmp_path)


def test_misspelled_profile_key_is_rejected(tmp_path: Path):
    row = {key: value for key, value in ROW.items() if key != "timeout_sec"} | {"timeout": 60}
    config_dir = _write(tmp_path, {"high": 0.9, "low": 0.5}, {"grounded": row, "partial": ROW})

    with pytest.raises(ValueError, match="profile 'grounded' keys must be"):
        load_answer_policy(config_dir)


def test_repo_profiles_cap_tokens_in_the_ollama_request():
    # max_tokens 는 서버(Ollama)가 num_predict 로 끊는다 — 요청 본문에 실리는지 본다 (보내지 않고 본문만 만든다)
    for stage, profile in _repo_profiles().items():
        request = chat_request(profile.spec, [Message("user", "질문")], stream=True)

        max_tokens = profile.spec.params.get("max_tokens")
        assert isinstance(max_tokens, int), stage
        assert max_tokens > 0, stage
        assert request["options"]["num_predict"] == max_tokens, stage
        assert 0 < profile.timeout_sec < math.inf, stage


@pytest.mark.parametrize(
    "params",
    [
        {"think": "off"},
        {"think": "off", "max_tokens": -1},
        {"think": "off", "max_tokens": 0},
        {"think": "off", "max_tokens": "512"},
        {"think": "off", "max_tokens": True},
    ],
    ids=["missing", "unlimited", "zero", "string", "bool"],
)
def test_profile_without_a_positive_max_tokens_is_rejected(params: dict[str, object]):
    with pytest.raises(ValueError, match="max_tokens"):
        parse_profiles({"grounded": ROW | {"params": params}})


@pytest.mark.parametrize(
    ("value", "expected"), [(None, False), ("1", True), ("0", False), ("true", False)]
)
def test_question_logging_turns_on_only_with_1(
    monkeypatch: pytest.MonkeyPatch, value: str | None, expected: bool
):
    # 켜는 값은 1 하나 — 기본은 원문을 남기지 않는 쪽이라 true · yes 로는 켜지지 않는다 (ADR-027)
    monkeypatch.delenv("RAG_LOG_QUESTIONS", raising=False)
    if value is not None:
        monkeypatch.setenv("RAG_LOG_QUESTIONS", value)

    assert log_questions_from_env() is expected


class _UnusedTokenizer:
    """관문이 꺼져 있거나 설정이 틀려 쓰이지 않는 분석기."""

    def tokenize(self, text: str) -> list[Token]:
        raise AssertionError("tokenizer must not be used")


def test_repo_answer_yaml_enables_the_gate_and_reply_uses_the_dump_date():
    gate = load_query_gate(CONFIG_DIR, KiwiTokenizer())

    assert gate is not None
    match = gate.check("오늘 서울 날씨는?")
    assert match is not None
    assert "2026년 10월 1일 기준" in fixed_reply(match, dump_date=load_dump_date(CONFIG_DIR))


def test_gate_can_be_switched_off(tmp_path: Path):
    (tmp_path / ANSWER_CONFIG).write_text("gate:\n  enabled: false\n", encoding="utf-8")

    assert load_query_gate(tmp_path, _UnusedTokenizer()) is None


def test_gate_switch_must_be_a_bool(tmp_path: Path):
    (tmp_path / ANSWER_CONFIG).write_text('gate:\n  enabled: "yes"\n', encoding="utf-8")

    with pytest.raises(TypeError, match="must be true or false"):
        load_query_gate(tmp_path, _UnusedTokenizer())
