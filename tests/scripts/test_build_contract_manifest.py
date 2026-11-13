import copy
import json

import pytest
from conftest import load_script

from playground_wiki.api.app import create_app
from playground_wiki.api.contract import openapi_text
from playground_wiki.api.version import CONTRACT_VERSION

manifest = load_script("build_contract_manifest")
OPENAPI = json.loads(openapi_text(create_app()))


def test_contract_hash_ignores_the_version_but_not_descriptions():
    bumped = copy.deepcopy(OPENAPI)
    bumped["info"]["version"] = "9.9.9"
    reworded = copy.deepcopy(OPENAPI)
    reworded["components"]["schemas"]["DeltaEvent"]["description"] = "다른 설명"

    assert manifest.contract_hash(bumped) == manifest.contract_hash(OPENAPI)
    assert manifest.contract_hash(reworded) != manifest.contract_hash(OPENAPI)


def test_components_list_endpoints_events_and_values():
    parts = manifest.components(OPENAPI)

    assert parts["endpoints"] == ["GET /health", "POST /ask", "POST /search"]
    assert parts["events"] == [
        f"POST /ask {name}" for name in ("stage", "sources", "delta", "done", "error")
    ]
    assert "agent" in parts["enums"]["POST /ask?mode"]
    assert "gate" in parts["enums"]["StageEvent.stage"]
    assert "text" in parts["schemas"]["DeltaEvent"]


def test_removed_items_finds_deletions_and_renames():
    before = manifest.components(OPENAPI)
    after = copy.deepcopy(before)
    after["schemas"]["DeltaEvent"] = ["content"]  # text → content
    after["enums"]["StageEvent.stage"].remove("gate")
    after["events"].remove("POST /ask error")
    del after["schemas"]["Timings"]

    removed = manifest.removed_items(before, after)

    assert "field DeltaEvent.text" in removed
    assert "value StageEvent.stage='gate'" in removed
    assert "event POST /ask error" in removed
    assert "schema Timings" in removed
    assert manifest.removed_items(before, before) == []


@pytest.mark.parametrize(
    ("base", "current", "changed", "removed", "ok"),
    [
        ("0.1.0", "0.1.0", False, [], True),
        ("0.1.0", "0.1.0", True, [], False),  # 바뀌었는데 그대로
        ("0.1.0", "0.1.1", True, [], True),  # 호환되는 추가 → patch
        ("0.1.0", "0.1.1", True, ["field X.y"], False),  # 삭제인데 patch
        ("0.1.0", "0.2.0", True, ["field X.y"], True),  # 0.x 의 breaking → minor
        ("1.2.0", "1.3.0", True, ["field X.y"], False),  # 1.x 의 breaking 은 major
        ("1.2.0", "2.0.0", True, ["field X.y"], True),
    ],
)
def test_bump_rules(base: str, current: str, changed: bool, removed: list[str], ok: bool):
    problems = manifest.bump_problems(base, current, changed=changed, removed=removed)

    assert (problems == []) is ok


def test_bump_against_a_base_manifest_with_the_same_version():
    base = {
        "contract_version": CONTRACT_VERSION,
        "contract_hash": manifest.contract_hash(OPENAPI),
        "components": manifest.components(OPENAPI),
    }
    renamed = copy.deepcopy(OPENAPI)
    schema = renamed["components"]["schemas"]["DeltaEvent"]
    schema["properties"]["content"] = schema["properties"].pop("text")

    assert manifest.bump_problems_against(base, OPENAPI) == []
    # 계약이 바뀌었는데 버전이 그대로다 — 몇을 올려야 하는지는 test_bump_rules 가 본다
    assert manifest.bump_problems_against(base, renamed)


def test_version_must_be_semver():
    with pytest.raises(ValueError, match=r"x\.y\.z"):
        manifest.parse_version("v0.1")


def test_changelog_needs_the_exact_version_heading():
    text = "# 기록\n\n## [0.1.0]\n\n최초.\n"

    assert manifest.changelog_has(text, "0.1.0")
    assert not manifest.changelog_has(text, "0.1.1")
    assert not manifest.changelog_has("[0.1.0] 은 최초", "0.1.0")


def test_committed_manifest_matches_the_app():
    # 계약을 고치면 CONTRACT_VERSION · CHANGELOG 를 올리고 scripts/build_contract_manifest.py 를 다시 돌린다
    committed = json.loads(manifest.MANIFEST_PATH.read_text(encoding="utf-8"))

    assert committed["contract_version"] == CONTRACT_VERSION
    assert committed["contract_hash"] == manifest.contract_hash(OPENAPI)
    assert committed["artifacts"] == manifest.artifact_hashes()
    assert manifest.changelog_has(
        manifest.CHANGELOG_PATH.read_text(encoding="utf-8"), CONTRACT_VERSION
    )
