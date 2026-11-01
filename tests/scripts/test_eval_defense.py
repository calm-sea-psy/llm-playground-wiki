"""scripts/eval_defense.py 의 칸 설정 폴더 · 유해 세트 검수표 — GPU · DB 없이."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from conftest import load_script

from playground_wiki.api.config import ApiConfig, load_answer_policy
from playground_wiki.evaluation.answer_run import AnswerRecord, ContextChunk, EvalItem
from playground_wiki.evaluation.defense_arms import ARMS, SERVICE_BASE

eval_defense = load_script("eval_defense")

CONFIG_DIR = Path(__file__).parents[2] / "configs"


def test_config_dir_swaps_only_the_service_base_of_each_profile(tmp_path: Path):
    out = tmp_path / "base-v5"

    eval_defense.make_config_dir(CONFIG_DIR, out, ARMS["base-v5"])

    assert sorted(p.name for p in out.iterdir()) == sorted(p.name for p in CONFIG_DIR.iterdir())
    for name in ("ingest.yaml", "retrieval.yaml"):
        assert (out / name).read_bytes() == (CONFIG_DIR / name).read_bytes()
    old = yaml.safe_load((CONFIG_DIR / "answer.yaml").read_text(encoding="utf-8"))
    new = yaml.safe_load((out / "answer.yaml").read_text(encoding="utf-8"))
    for stage in old["profiles"]:
        assert new["profiles"][stage]["system"][0] == "base@v5"
        assert new["profiles"][stage]["system"][1:] == old["profiles"][stage]["system"][1:]
        new["profiles"][stage]["system"] = old["profiles"][stage]["system"]
    assert new == old  # 바뀐 것은 system 의 base 하나뿐
    assert (out / "answer.yaml").read_text(encoding="utf-8").startswith("# 방어 비교 칸 사본")


def test_config_dir_is_what_the_answer_path_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    out = tmp_path / "base-norule"
    eval_defense.make_config_dir(CONFIG_DIR, out, ARMS["base-norule"])
    monkeypatch.setenv("RAG_CONFIG_DIR", str(out))

    policy = load_answer_policy(ApiConfig.from_env().config_dir)

    assert {profile.system[0] for profile in policy.profiles.values()} == {"base_norule@v4"}


def test_config_dir_keeps_a_same_folder_and_stops_on_a_different_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    out = tmp_path / "base-v5"
    eval_defense.make_config_dir(CONFIG_DIR, out, ARMS["base-v5"])

    eval_defense.make_config_dir(CONFIG_DIR, out, ARMS["base-v5"])
    assert "내용이 같다" in capsys.readouterr().out
    with pytest.raises(SystemExit, match=r"answer\.yaml"):
        eval_defense.make_config_dir(CONFIG_DIR, out, ARMS["base-norule"])


def test_arm_answer_yaml_stops_when_a_profile_lacks_the_service_base():
    text = (CONFIG_DIR / "answer.yaml").read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="exactly once"):
        eval_defense.arm_answer_yaml(text.replace(SERVICE_BASE, "base@v3"), ARMS["base-v5"])


def _record(qid: str, category: str, stage_path: tuple[str, ...]) -> AnswerRecord:
    hit = ContextChunk(chunk_id=f"{qid}-0", title="t", section="", text="문맥원문", score=0.5)
    return AnswerRecord(
        item=EvalItem(id=qid, source="harmful", question=f"질문원문 {qid}", category=category),
        stage_path=stage_path,
        answer=f"답원문 {qid}",
        hits=(hit,),
        context_count=1,
        marker=None,
        cited=(),
        dropped=(),
        truncated=False,
        timed_out=False,
        error=None,
        top_score=0.5,
        gate_rule=None,
        spec_id="ollama:fake",
        prompt_version="base@v5:x+partial@v4:y+question@v1:z",
        search_ms=1.0,
        first_delta_ms=1.0,
        total_ms=1.0,
        wall_ms=1.0,
    )


def _harmful_records() -> list[AnswerRecord]:
    return [
        _record("hz-01", "harmful", ("grounded", "partial")),
        _record("hz-02", "harmful", ("grounded",)),  # grounded 로 끝난 유해 질문 — 싣는다
        _record("hc-02", "contrast", ("grounded",)),  # grounded 로 끝난 대조 질문 — 싣지 않는다
        _record("hc-01", "contrast", ("partial",)),
        replace(_record("hz-03", "harmful", ()), gate_rule="self_harm"),  # 관문 — 싣지 않는다
        _record("hz-04", "harmful", ("refuse",)),
    ]


def test_review_picks_partial_answers_and_grounded_harmful_ones():
    picks = eval_defense.review_picks(_harmful_records())

    assert sorted(r.item.id for r in picks) == ["hc-01", "hz-01", "hz-02"]
    assert [r.item.id for r in picks] == [
        r.item.id for r in eval_defense.review_picks(list(reversed(_harmful_records())))
    ]  # seed 고정 — 기록 차례와 상관없이 같은 차례


def test_review_key_has_the_gate_v5_sheet_shape():
    picks = eval_defense.review_picks(_harmful_records())

    key = eval_defense.review_key(picks, "harmful-route-base-v5-1")

    assert sorted(key) == ["created_at", "entries", "part", "seed"]
    assert key["part"] == "harmful-route-base-v5-1"
    assert [sorted(entry) for entry in key["entries"]] == [["id", "number", "run", "stage"]] * 3
    assert {entry["run"] for entry in key["entries"]} == {"harmful-route-base-v5-1"}
    assert [entry["number"] for entry in key["entries"]] == [1, 2, 3]


def test_review_sheet_writes_the_sheet_and_key_without_overwriting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "data/eval/harmful-answers-v1-route-base-v5-1.jsonl"
    source.parent.mkdir(parents=True)
    rows = [json.dumps(_as_row(r), ensure_ascii=False) for r in _harmful_records()]
    source.write_text("\n".join(rows) + "\n", encoding="utf-8")

    eval_defense.review_sheet("base-v5-1")

    sheet = tmp_path / "data/eval/partial-judge-review-v1-harmful-route-base-v5-1.md"
    key = json.loads(sheet.with_suffix(".key.json").read_text(encoding="utf-8"))
    assert sheet.read_text(encoding="utf-8").startswith(
        "# partial 판정 검수표 v1 — harmful-route-base-v5-1 3문항"
    )
    assert len(key["entries"]) == 3
    with pytest.raises(SystemExit, match="이미 있다"):
        eval_defense.review_sheet("base-v5-1")


def test_parse_args_ties_each_option_to_its_step():
    assert eval_defense._parse_args(["config-dir"]).arm is None
    assert eval_defense._parse_args(["review-sheet", "--tag", "base-v5-1"]).tag == "base-v5-1"
    for argv in (
        ["review-sheet"],
        ["config-dir", "--tag", "x"],
        ["review-sheet", "--arm", "base-v5", "--tag", "x"],
        ["review-sheet", "--tag", "Bad_Tag"],
    ):
        with pytest.raises(SystemExit):
            eval_defense._parse_args(argv)


def _as_row(record: AnswerRecord) -> dict[str, object]:
    """read_records 가 읽는 기록 한 줄 (eval_answer run 의 꼴)."""
    return record.to_json()
