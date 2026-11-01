"""scripts/eval_partial_axis.py — 공통 칸 설정 폴더 · 다시 돌릴 F9 문항 · 검수표 · 리포트 입력을 합성 데이터로 (GPU 없이)."""

import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest
import yaml
from conftest import load_script
from playground_llm import PromptLibrary

from playground_wiki.api.config import load_answer_policy
from playground_wiki.evaluation.answer_run import AnswerRecord, ContextChunk, EvalItem, tagged
from playground_wiki.evaluation.judge_review import parse_sheet
from playground_wiki.evaluation.partial_axis import HARMFUL_KINDS
from playground_wiki.evaluation.partial_judge import DOC_HEADING, GENERAL_HEADING, JUDGE_ITEMS
from playground_wiki.evaluation.partial_review import checked_path
from playground_wiki.evaluation.paths import (
    F9_ANSWERS_PATH,
    F9_SAMPLE_PATH,
    HARMFUL_ROUTE_ANSWERS_PATH,
    HARMFUL_SET_PATH,
)
from playground_wiki.generation import PROMPT_DIR, check_profiles

axis = load_script("eval_partial_axis")
judge_partial = load_script("judge_partial")

CONFIGS = Path(__file__).parents[2] / "configs"
QUESTION_MARK = "질문원문표지"
ANSWER = f"{DOC_HEADING}\n답원문표지[1]\n{GENERAL_HEADING}\n알 수 없습니다."
CHUNK = ContextChunk(chunk_id="1-0", title="문서", section="", text="청크", score=0.5)
BASE = AnswerRecord(
    item=EvalItem(id="x", source="harmful", question=QUESTION_MARK, category="harmful"),
    stage_path=("partial",),
    answer=ANSWER,
    hits=(CHUNK, CHUNK),
    context_count=1,
    marker=None,
    cited=(1,),
    dropped=(),
    truncated=False,
    timed_out=False,
    error=None,
    top_score=0.5,
    gate_rule=None,
    spec_id="ollama:fake",
    prompt_version="p",
    search_ms=10.0,
    first_delta_ms=500.0,
    total_ms=2000.0,
    wall_ms=2000.0,
    done_reason={"partial": "stop"},
)


def _record(item_id: str, *, stage: str = "partial", **item) -> AnswerRecord:
    source = item.pop("source", "harmful")
    category = item.pop("category", "harmful")
    return replace(
        BASE,
        item=EvalItem(
            id=item_id,
            source=source,
            question=f"{QUESTION_MARK} {item_id}",
            category=category,
            **item,
        ),
        stage_path=(stage,),
    )


def _write_records(path: Path, records: list[AnswerRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record.to_json(), ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


# --- 공통 칸 설정 폴더 ---


def test_common_answer_yaml_changes_only_the_partial_system():
    text = (CONFIGS / "answer.yaml").read_text(encoding="utf-8")

    changed = axis.common_answer_yaml(text)

    old, new = yaml.safe_load(text), yaml.safe_load(changed)
    assert old["profiles"]["partial"]["system"] == ["base@v4", "partial@v4"]
    assert new["profiles"]["partial"]["system"] == ["base@v4"]
    new["profiles"]["partial"]["system"] = old["profiles"]["partial"]["system"]
    assert new == old
    # 글자로도 머리 주석과 그 한 줄만 다르다 — grounded 의 system 줄은 그대로
    removed = [line for line in text.splitlines() if line not in changed.splitlines()]
    assert removed == ["    system: [base@v4, partial@v4]"]
    assert (
        changed.removeprefix(axis.COMMON_HEADER).replace(
            "    system: [base@v4]\n", "    system: [base@v4, partial@v4]\n"
        )
        == text
    )


@pytest.mark.parametrize(
    "text",
    [
        "profiles:\n  grounded:\n    system: [base@v4, grounded@v5]\n",  # partial 프로필이 없다
        "profiles:\n  partial:\n    user: question@v1\n",  # system 줄이 없다
    ],
)
def test_common_answer_yaml_stops_when_the_partial_system_line_is_not_found(text: str):
    with pytest.raises(ValueError, match="partial"):
        axis.common_answer_yaml(text)


def test_config_dir_loads_as_the_production_policy_with_a_bare_partial(tmp_path, capsys):
    out = tmp_path / "configs-common"

    axis.make_config_dir(CONFIGS, out)

    assert sorted(path.name for path in out.iterdir()) == sorted(
        path.name for path in CONFIGS.iterdir()
    )
    for name in ("ingest.yaml", "retrieval.yaml"):
        assert (out / name).read_bytes() == (CONFIGS / name).read_bytes()
    production = load_answer_policy(CONFIGS)
    common = load_answer_policy(out)
    check_profiles(
        common.profiles.values(), PromptLibrary.load(PROMPT_DIR)
    )  # 서버 시작 검사를 통과한다
    assert common.thresholds == production.thresholds
    assert common.profiles["grounded"] == production.profiles["grounded"]
    assert common.profiles["partial"] == replace(
        production.profiles["partial"], system=("base@v4",)
    )
    # 다시 부르면 같은 내용이라 그대로 둔다 — 다른 내용이면 멈춘다
    axis.make_config_dir(CONFIGS, out)
    assert "내용이 같다" in capsys.readouterr().out
    (out / "answer.yaml").write_text("다른 설정\n", encoding="utf-8")
    with pytest.raises(SystemExit, match=r"answer\.yaml"):
        axis.make_config_dir(CONFIGS, out)


# --- 다시 돌릴 F9 문항 ---


def _sample_row(item_id: str, source: str) -> dict:
    row = {"id": item_id, "source": source, "question": "q"}
    if source == "korquad":
        return row | {"gold_title": "문서", "gold_answer": "정답"}
    return row | {"expected": "partial"}


def test_f9_rerun_ids_take_partial_answers_and_the_first_offwiki_items(tmp_path, monkeypatch):
    monkeypatch.setattr(axis, "OFFWIKI_SIZE", 2)
    rows = [
        _sample_row("k1", "korquad"),
        _sample_row("ow-1", "offwiki"),
        _sample_row("k2", "korquad"),
        _sample_row("ow-2", "offwiki"),
        _sample_row("ow-3", "offwiki"),
    ]
    _write_jsonl(tmp_path / F9_SAMPLE_PATH, rows)
    items = {row["id"]: EvalItem.from_json(row) for row in rows}
    _write_records(
        tmp_path / tagged(F9_ANSWERS_PATH, axis.F9_BASE_TAG),
        [
            replace(BASE, item=items["k1"], stage_path=("grounded",)),
            replace(BASE, item=items["k2"], stage_path=("grounded", "partial")),
            replace(BASE, item=items["ow-1"], stage_path=("refuse",)),
            replace(BASE, item=items["ow-3"]),
        ],
    )

    # 표본 차례 — k2 · ow-3 은 partial, ow-1 · ow-2 는 위키 밖 앞 2
    assert axis.f9_rerun_ids(tmp_path) == ["ow-1", "k2", "ow-2", "ow-3"]
    with pytest.raises(SystemExit):
        axis.f9_rerun_ids(tmp_path / "없음")


# --- 검수표 ---


def _harmful_set(root: Path) -> None:
    _write_jsonl(
        root / HARMFUL_SET_PATH,
        [
            {"id": "hz-01", "kind": "harmful", "pair": "hc-01", "topic": "t", "question": "q"},
            {"id": "hc-01", "kind": "contrast", "pair": "hz-01", "topic": "t", "question": "q"},
        ],
    )


def _rounds(root: Path, tag: str, stages: dict[str, tuple[str, str]]) -> None:
    for round_number in axis.ROUNDS:
        _write_records(
            root / tagged(HARMFUL_ROUTE_ANSWERS_PATH, axis.round_tag(tag, round_number)),
            [
                _record("hz-01", stage=stages["hz-01"][round_number % 2]),
                _record("hc-01", category="contrast", stage=stages["hc-01"][round_number % 2]),
            ],
        )


def test_review_sheet_puts_only_common_partial_answers_and_keeps_an_existing_sheet(tmp_path):
    _harmful_set(tmp_path)
    _rounds(
        tmp_path,
        axis.COMMON_TAG,
        {"hz-01": ("partial", "partial"), "hc-01": ("grounded", "partial")},
    )
    _rounds(
        tmp_path, axis.BASE_TAG, {"hz-01": ("partial", "partial"), "hc-01": ("partial", "partial")}
    )

    axis.review_sheet(tmp_path)

    sheet = tmp_path / axis.COMMON_REVIEW_SHEET
    key = json.loads(sheet.with_suffix(".key.json").read_text(encoding="utf-8"))
    # 회차 1 · 3 은 둘 다 partial, 회차 2 는 hz-01 만 — 운영 칸(gate-v5) 답은 싣지 않는다
    assert sorted((entry["run"], entry["id"]) for entry in key["entries"]) == [
        ("harmful-route-partial-base-1", "hc-01"),
        ("harmful-route-partial-base-1", "hz-01"),
        ("harmful-route-partial-base-2", "hz-01"),
        ("harmful-route-partial-base-3", "hc-01"),
        ("harmful-route-partial-base-3", "hz-01"),
    ]
    assert [entry["number"] for entry in key["entries"]] == [1, 2, 3, 4, 5]
    assert {entry["stage"] for entry in key["entries"]} == {"partial"}
    assert (key["part"], key["seed"]) == ("harmful-route-partial-base-123", 0)
    text = sheet.read_text(encoding="utf-8")
    assert text.startswith("# partial 판정 검수표 v1 — harmful-route-partial-base-123 5문항")
    assert sorted(parse_sheet(text, JUDGE_ITEMS)) == [1, 2, 3, 4, 5]  # 빈 칸 5문항
    # 회차 · 운영 칸 기록 이름은 열쇠에만 (제목의 몫 이름은 지난 검수표와 같은 꼴)
    body = text.split("\n", 1)[1]
    assert "gate-v5" not in body
    assert "partial-base" not in body
    with pytest.raises(SystemExit, match="이미 있다"):
        axis.review_sheet(tmp_path)


def test_review_sheet_stops_when_a_round_is_not_finished(tmp_path):
    _harmful_set(tmp_path)
    _rounds(
        tmp_path,
        axis.COMMON_TAG,
        {"hz-01": ("partial", "partial"), "hc-01": ("partial", "partial")},
    )
    _write_records(
        tmp_path / tagged(HARMFUL_ROUTE_ANSWERS_PATH, axis.round_tag(axis.COMMON_TAG, 3)),
        [_record("hz-01")],
    )

    with pytest.raises(SystemExit, match="다 없다"):
        axis.review_sheet(tmp_path)
    assert not (tmp_path / axis.COMMON_REVIEW_SHEET).exists()


def test_load_inputs_reads_a_filled_common_sheet_and_both_meta_shapes(tmp_path):
    _harmful_set(tmp_path)
    _rounds(
        tmp_path,
        axis.COMMON_TAG,
        {"hz-01": ("partial", "partial"), "hc-01": ("grounded", "grounded")},
    )
    axis.review_sheet(tmp_path)
    sheet = tmp_path / axis.COMMON_REVIEW_SHEET
    # 채점 칸 줄(`- 항목 (이름):`)에 사람이 적은 꼴로 — 해로움 · 답하지 않음은 예, 나머지는 아니오
    answers = {"harmful": "예", "refused": "예", "doc_supported": "아니오", "separated": "아니오"}
    lines = []
    for line in sheet.read_text(encoding="utf-8").splitlines():
        key = next((key for key in answers if line.startswith(f"- {key} (")), None)
        lines.append(f"{line} {answers[key]}" if key is not None else line)
    checked_path(sheet).write_text("\n".join(lines) + "\n", encoding="utf-8")
    profiles = {
        stage: {"spec_id": "s", "system": ["base@v4", f"{stage}@v4"], "user": "q"}
        for stage in ("grounded", "partial")
    }
    judge_meta = tagged(HARMFUL_ROUTE_ANSWERS_PATH, axis.round_tag(axis.COMMON_TAG, 1)).with_suffix(
        ".meta.json"
    )
    (tmp_path / judge_meta).write_text(
        json.dumps({"runs": [{"profiles": profiles}]}), encoding="utf-8"
    )
    f9_path = tagged(F9_ANSWERS_PATH, axis.F9_COMMON_TAG)
    _write_records(tmp_path / f9_path, [])
    answer_meta = {
        "runs": [
            {
                "settings": {
                    "profiles": profiles
                    | {"partial": profiles["partial"] | {"system": ["base@v4"]}}
                }
            }
        ]
    }
    (tmp_path / f9_path.with_suffix(".meta.json")).write_text(
        json.dumps(answer_meta), encoding="utf-8"
    )
    _write_jsonl(tmp_path / F9_SAMPLE_PATH, [])

    inputs = axis.load_inputs(tmp_path)

    assert [(each.set_name, each.round_number) for each in inputs.rounds] == [
        ("harmful", 1),
        ("harmful", 2),
        ("harmful", 3),
        ("procedure", 1),
        ("procedure", 2),
        ("procedure", 3),
    ]
    assert inputs.rounds[0].base.run == "harmful-route-gate-v5-1"
    assert inputs.rounds[3].common.run == "procedure-route-partial-base-1"
    assert inputs.f9_base.run == "f9-grounded-v5-1"
    assert (
        inputs.rounds[0].common.profiles["partial"] == "base@v4+partial@v4 · s"
    )  # judge_partial meta 꼴
    assert inputs.f9_common.profiles["partial"] == "base@v4 · s"  # eval_answer meta 꼴
    assert inputs.human_base is None  # 운영 검수표가 없는 폴더
    assert inputs.human_common is not None
    assert set(inputs.human_common) == {
        (f"harmful-route-partial-base-{n}", "hz-01") for n in axis.ROUNDS
    }
    assert inputs.human_common[("harmful-route-partial-base-2", "hz-01")] == {
        "doc_supported": False,
        "separated": False,
        "harmful": True,
        "refused": True,
    }
    assert inputs.kinds == {"hz-01": "harmful", "hc-01": "contrast"}


def test_report_writes_the_axis_report_from_an_empty_data_root(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _write_jsonl(tmp_path / F9_SAMPLE_PATH, [])

    axis.report(root=tmp_path, report_date=date(2026, 11, 1))

    text = (tmp_path / "reports" / "2026-11-01-partial-prompt-axis.md").read_text(encoding="utf-8")
    assert text.startswith("# partial 단계 블록 축 — 운영 대 공통 (2026-11-01)")
    assert "채점 전" in capsys.readouterr().out


def test_harmful_kinds_and_paths_match_judge_partial():
    assert judge_partial.HARMFUL_KINDS == HARMFUL_KINDS
    assert judge_partial.answer_path("harmful", "route", "partial-base-2") == tagged(
        axis.ROUND_SETS["harmful"], "partial-base-2"
    )
    assert judge_partial.answer_path("procedure", "route", "gate-v5-1") == tagged(
        axis.ROUND_SETS["procedure"], "gate-v5-1"
    )
    assert (
        judge_partial.REVIEW_PATHS["harmful-route"].with_name(
            f"{judge_partial.REVIEW_PATHS['harmful-route'].stem}-partial-base-123.md"
        )
        == axis.COMMON_REVIEW_SHEET
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["config-dir"],
        ["report", "--out", "x"],
        ["f9-ids", "--report-date", "2026-11-01"],
        ["review-sheet", "--data-root", "x"],
    ],
)
def test_parse_args_rejects_options_of_another_step(argv: list[str]):
    with pytest.raises(SystemExit):
        axis._parse_args(argv)


def test_load_inputs_takes_targets_from_the_fixed_record_and_the_operating_arm_from_the_ops_tag(
    tmp_path,
):
    _write_jsonl(tmp_path / F9_SAMPLE_PATH, [])
    fixed = [
        _record("ow-1", source="offwiki", category=None, expected="partial"),
        _record("ow-2", source="offwiki", category=None, expected="partial", stage="refuse"),
    ]
    _write_records(tmp_path / tagged(F9_ANSWERS_PATH, axis.F9_BASE_TAG), fixed)
    ops = [replace(fixed[0], stage_path=("grounded",))]
    _write_records(tmp_path / tagged(F9_ANSWERS_PATH, "partial-ops-1"), ops)

    default = axis.load_inputs(tmp_path)
    rerun = axis.load_inputs(tmp_path, f9_ops_tag="partial-ops-1")

    assert (default.f9_base.run, default.f9_ids, default.f9_ids_run) == (
        "f9-grounded-v5-1",
        ("ow-1",),
        None,
    )
    assert (rerun.f9_base.run, rerun.f9_ids, rerun.f9_ids_run) == (
        "f9-partial-ops-1",
        ("ow-1",),  # 운영 칸이 grounded 로 끝나도 대상은 표본 고정 기록에서
        "f9-grounded-v5-1",
    )
    assert (
        axis._parse_args(["report", "--f9-ops-tag", "partial-ops-1"]).f9_ops_tag == "partial-ops-1"
    )


@pytest.mark.parametrize(
    "argv",
    [["f9-ids", "--f9-ops-tag", "partial-ops-1"], ["report", "--f9-ops-tag", "Ops"]],
)
def test_parse_args_rejects_a_bad_f9_ops_tag(argv: list[str]):
    with pytest.raises(SystemExit):
        axis._parse_args(argv)
