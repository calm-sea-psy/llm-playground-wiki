"""evaluation.defense_arms — 대조 프롬프트가 v4 에서 규칙 줄 하나만 뺐는지, 운영 프롬프트로 고를 수 없는지."""

import shutil
from dataclasses import replace
from pathlib import Path

import pytest
from playground_llm import PromptError, PromptLibrary

from playground_wiki.api.config import AnswerSetup, ApiConfig
from playground_wiki.evaluation.defense_arms import (
    ARM_NAMES,
    ARMS,
    CONTROL_PROMPT_DIR,
    SERVICE_BASE,
    swap_base,
)
from playground_wiki.evaluation.prompt_variants import DEFENSE_LINE
from playground_wiki.gate import Token
from playground_wiki.generation import PROMPT_DIR, check_profiles

CONFIG_DIR = Path(__file__).parents[2] / "configs"


class _UnusedTokenizer:
    """관문을 끈 구성이라 쓰이지 않는 분석기."""

    def tokenize(self, text: str) -> list[Token]:
        raise AssertionError("tokenizer must not be used")


def _text(directory: Path, ref: str) -> str:
    name, _, version = ref.partition("@")
    return (directory / name / f"{version}.md").read_text(encoding="utf-8")


def _setup(config_dir: Path, monkeypatch: pytest.MonkeyPatch) -> AnswerSetup:
    monkeypatch.setenv("RAG_CONFIG_DIR", str(config_dir))
    return AnswerSetup.from_config(ApiConfig.from_env(), tokenizer=_UnusedTokenizer(), gate=False)


def test_arms_follow_the_declared_names():
    assert tuple(ARMS) == ARM_NAMES
    assert all(ARMS[name].name == name for name in ARM_NAMES)
    assert ARMS["base-v5"].side_tag == "base-v5-1"


def test_control_prompt_is_v4_without_the_document_rule_line():
    v4 = _text(PROMPT_DIR, SERVICE_BASE).splitlines()
    control = _text(CONTROL_PROMPT_DIR, ARMS["base-norule"].base).splitlines()

    # 지운 줄 = prompt_variants 의 방어 문구 줄 (no-defense 변형이 지우는 줄과 같다)
    assert DEFENSE_LINE in v4
    assert control == [line for line in v4 if line != DEFENSE_LINE]


def test_control_prompt_lives_outside_the_service_prompts():
    service = PromptLibrary.load(PROMPT_DIR)
    control = PromptLibrary.load(CONTROL_PROMPT_DIR)

    assert ARMS["base-norule"].base in control.refs
    assert ARMS["base-norule"].base not in service.refs
    assert ARMS["base-v5"].base in service.refs  # 후보는 채택하면 answer.yaml 한 줄로 고른다
    assert not set(control.refs) & set(service.refs)


def test_both_arm_bases_pass_the_profile_check_with_the_stage_blocks(
    monkeypatch: pytest.MonkeyPatch,
):
    templates = []
    for directory in (PROMPT_DIR, CONTROL_PROMPT_DIR):
        loaded = PromptLibrary.load(directory)
        templates.extend(loaded.get(ref) for ref in loaded.refs)
    library = PromptLibrary(templates)
    setup = _setup(CONFIG_DIR, monkeypatch)

    for arm in ARMS.values():
        profiles = [
            replace(p, system=swap_base(p.system, arm.base)) for p in setup.policy.profiles.values()
        ]
        check_profiles(profiles, library)  # 없는 프롬프트 · 모르는 자리표시자를 잡는다


def test_server_setup_refuses_a_config_that_picks_the_control_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    configs = tmp_path / "configs"
    shutil.copytree(CONFIG_DIR, configs)
    answer = configs / "answer.yaml"
    text = answer.read_text(encoding="utf-8")
    answer.write_text(text.replace(SERVICE_BASE, ARMS["base-norule"].base), encoding="utf-8")

    with pytest.raises(PromptError, match="base_norule@v4"):
        _setup(configs, monkeypatch)


def test_swap_base_replaces_only_the_service_base():
    assert swap_base(["base@v4", "grounded@v5"], "base@v5") == ("base@v5", "grounded@v5")
    with pytest.raises(ValueError, match="exactly once"):
        swap_base(["base@v3", "grounded@v5"], "base@v5")
