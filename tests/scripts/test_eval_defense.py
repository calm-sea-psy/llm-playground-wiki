"""scripts/eval_defense.py 의 칸 설정 폴더 · 유해 세트 검수표 — GPU · DB 없이."""

import json
from dataclasses import replace
from datetime import date
from pathlib import Path
from types import MappingProxyType

import pytest
import yaml
from conftest import SCRIPTS, load_script

from playground_wiki.api.config import ApiConfig, load_answer_policy
from playground_wiki.evaluation.answer_run import AnswerRecord, ContextChunk, EvalItem
from playground_wiki.evaluation.defense_arms import ARMS, DEFENSE_BASE
from playground_wiki.evaluation.defense_side import InjectionVerdict, PairCount, RoundCount
from playground_wiki.evaluation.f4_prose import F4Item
from playground_wiki.evaluation.prose_labels import ProseLabels, RefusalVerdict

eval_defense = load_script("eval_defense")

CONFIG_DIR = Path(__file__).parents[2] / "configs"


def test_config_dir_swaps_only_the_service_base_of_each_profile(tmp_path: Path, v4_configs: Path):
    out = tmp_path / "base-v5"

    eval_defense.make_config_dir(v4_configs, out, ARMS["base-v5"])

    assert sorted(p.name for p in out.iterdir()) == sorted(p.name for p in v4_configs.iterdir())
    for name in ("ingest.yaml", "retrieval.yaml"):
        assert (out / name).read_bytes() == (v4_configs / name).read_bytes()
    old = yaml.safe_load((v4_configs / "answer.yaml").read_text(encoding="utf-8"))
    new = yaml.safe_load((out / "answer.yaml").read_text(encoding="utf-8"))
    for stage in old["profiles"]:
        assert new["profiles"][stage]["system"][0] == "base@v5"
        assert new["profiles"][stage]["system"][1:] == old["profiles"][stage]["system"][1:]
        new["profiles"][stage]["system"] = old["profiles"][stage]["system"]
    assert new == old  # 바뀐 것은 system 의 base 하나뿐
    assert (out / "answer.yaml").read_text(encoding="utf-8").startswith("# 방어 비교 칸 사본")


def test_config_dir_is_what_the_answer_path_reads(
    tmp_path: Path, v4_configs: Path, monkeypatch: pytest.MonkeyPatch
):
    out = tmp_path / "base-norule"
    eval_defense.make_config_dir(v4_configs, out, ARMS["base-norule"])
    monkeypatch.setenv("RAG_CONFIG_DIR", str(out))

    policy = load_answer_policy(ApiConfig.from_env().config_dir)

    assert {profile.system[0] for profile in policy.profiles.values()} == {"base_norule@v4"}


def test_config_dir_keeps_a_same_folder_and_stops_on_a_different_one(
    tmp_path: Path, v4_configs: Path, capsys: pytest.CaptureFixture[str]
):
    out = tmp_path / "base-v5"
    eval_defense.make_config_dir(v4_configs, out, ARMS["base-v5"])

    eval_defense.make_config_dir(v4_configs, out, ARMS["base-v5"])
    assert "내용이 같다" in capsys.readouterr().out
    with pytest.raises(SystemExit, match=r"answer\.yaml"):
        eval_defense.make_config_dir(v4_configs, out, ARMS["base-norule"])


def test_arm_answer_yaml_stops_when_a_profile_lacks_the_service_base():
    text = (CONFIG_DIR / "answer.yaml").read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="exactly once"):
        eval_defense.arm_answer_yaml(text.replace(DEFENSE_BASE, "base@v3"), ARMS["base-v5"])


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


def _write_records(path: Path, records: list[AnswerRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.dumps(_as_row(r), ensure_ascii=False) for r in records]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


# --- base-v6 · 새 단계 ---


def test_config_dir_stops_on_the_switched_service_config(tmp_path: Path):
    with pytest.raises(ValueError, match="base@v4"):
        eval_defense.make_config_dir(CONFIG_DIR, tmp_path / "base-v5", ARMS["base-v5"])


def test_config_dir_makes_the_v6_arm_folder(tmp_path: Path, v4_configs: Path):
    out = tmp_path / "base-v6"

    eval_defense.make_config_dir(v4_configs, out, ARMS["base-v6"])

    policy = yaml.safe_load((out / "answer.yaml").read_text(encoding="utf-8"))
    assert {p["system"][0] for p in policy["profiles"].values()} == {"base@v6"}
    assert "base-v6" in eval_defense.SIDE_ARMS
    assert "base-norule" not in eval_defense.SIDE_ARMS  # 측정 전용 대조는 부작용을 재지 않는다


def test_parse_args_for_the_new_steps():
    args = eval_defense._parse_args(
        ["item-sheet", "--id", "hz-20", "--tags", "base-v6-1,base-v6-2"]
    )
    assert (args.id, args.tags) == ("hz-20", "base-v6-1,base-v6-2")
    assert eval_defense._parse_args(["refusal-input", "--tag", "base-v6-1"]).tag == "base-v6-1"
    side = eval_defense._parse_args(["side-report", "--arm", "base-v6"])
    assert (side.arm, side.bench_data) == ("base-v6", eval_defense.BENCH_DATA)
    for argv in (
        ["item-sheet", "--id", "hz-20"],
        ["item-sheet", "--id", "hz-20", "--tags", "base-v6-1,base-v5-2"],
        ["refusal-input"],
        ["side-report"],
        ["side-report", "--arm", "base-norule"],
        ["side-report", "--arm", "base-v6", "--tag", "x"],
        ["review-sheet", "--tag", "x", "--bench-data", "d"],
        ["config-dir", "--id", "hz-20"],
    ):
        with pytest.raises(SystemExit):
            eval_defense._parse_args(argv)


def test_bundle_name_is_the_shared_name_of_numbered_tags():
    tags = [f"base-v6-{n}" for n in range(1, 11)]

    assert eval_defense.bundle_name(tags) == "base-v6"
    for bad in (["base-v6"], ["base-v6-1", "base-v5-2"], ["base-v6-1", "base-v6-1"], ["B-1"]):
        with pytest.raises(ValueError):
            eval_defense.bundle_name(bad)


def test_item_sheet_gathers_one_item_from_each_tag_without_overwriting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    tags = [f"base-v6-{n}" for n in range(1, 4)]
    for tag in tags:
        _write_records(
            tmp_path / f"data/eval/harmful-answers-v1-route-{tag}.jsonl",
            [
                _record("hz-20", "harmful", ("grounded", "partial")),
                _record("hz-08", "harmful", ("partial",)),
            ],
        )

    eval_defense.item_sheet("hz-20", tags)

    sheet = tmp_path / "data/eval/partial-judge-review-v1-harmful-route-hz-20-base-v6.md"
    key = json.loads(sheet.with_suffix(".key.json").read_text(encoding="utf-8"))
    assert sheet.read_text(encoding="utf-8").startswith(
        "# partial 판정 검수표 v1 — harmful-route-hz-20-base-v6 3문항"
    )
    assert (key["part"], key["seed"]) == ("harmful-route-hz-20-base-v6", 20261101)
    assert {e["id"] for e in key["entries"]} == {"hz-20"}
    assert sorted(e["run"] for e in key["entries"]) == [f"harmful-route-{t}" for t in tags]
    assert [sorted(e) for e in key["entries"]] == [["id", "number", "run", "stage"]] * 3
    assert "hz-08" not in sheet.read_text(encoding="utf-8")
    with pytest.raises(SystemExit, match="이미 있다"):
        eval_defense.item_sheet("hz-20", tags)


def test_item_picks_stop_when_a_tag_lacks_the_item(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    _write_records(
        tmp_path / "data/eval/harmful-answers-v1-route-base-v6-1.jsonl",
        [_record("hz-08", "harmful", ("partial",))],
    )

    with pytest.raises(SystemExit, match="0개"):
        eval_defense.item_picks("hz-20", ["base-v6-1"])
    with pytest.raises(SystemExit, match="없다"):
        eval_defense.item_picks("hz-20", ["base-v6-2"])


def _f4_item(qid: str, kind: str) -> F4Item:
    chunk = ContextChunk(chunk_id=f"{qid}-0", title="t", section="", text="문맥원문", score=0.9)
    return F4Item(
        id=qid,
        set_name="s",
        kind=kind,  # type: ignore[arg-type]  # 시험 문항
        question=f"질문원문 {qid}",
        answers=("정답",),
        gold_title="t",
        context=(chunk,),
    )


def _f4_record(item: F4Item, answer: str, stage_path: tuple[str, ...]) -> AnswerRecord:
    record = _record(item.id, item.kind, stage_path)
    return replace(record, item=item.eval_item(), answer=answer, marker="found")


def test_refusal_rows_mask_only_sensitive_answers_that_need_judgment():
    items = {
        "benign_sensitive:1": _f4_item("benign_sensitive:1", "benign_sensitive"),
        "benign_sensitive:2": _f4_item("benign_sensitive:2", "benign_sensitive"),
        "benign_sensitive:3": _f4_item("benign_sensitive:3", "benign_sensitive"),
        "no_fact:1": _f4_item("no_fact:1", "no_fact"),
    }
    records = [
        _f4_record(items["benign_sensitive:1"], "답원문 정답", ("grounded",)),
        _f4_record(items["benign_sensitive:2"], "답원문", ("grounded",)),
        _f4_record(items["benign_sensitive:3"], "고정", ("refuse",)),  # 고정 문구 — 규칙으로 끝난다
        _f4_record(items["no_fact:1"], "답원문", ("grounded",)),
    ]

    rows, key = eval_defense.refusal_rows(records, items)

    assert [r["judge_id"] for r in rows] == ["R001", "R002"]
    assert [sorted(r) for r in rows] == [["answer", "judge_id", "question"]] * 2
    assert {k["id"] for k in key} == {"benign_sensitive:1", "benign_sensitive:2"}
    assert [k["judge_id"] for k in key] == ["R001", "R002"]
    assert rows == eval_defense.refusal_rows(list(reversed(records)), items)[0]  # 시드 고정


def test_read_judged_unmasks_verdicts_and_stops_on_a_partial_file(tmp_path: Path):
    key = tmp_path / "refusal-judge-key-v1-t.jsonl"
    key.write_text(
        '{"judge_id": "R001", "id": "b:1"}\n{"judge_id": "R002", "id": "b:2"}\n', encoding="utf-8"
    )
    judged = tmp_path / "refusal-judged-v1-t.jsonl"

    assert eval_defense.read_judged(tmp_path, "t") is None
    judged.write_text('{"judge_id": "R002", "refused": true, "reason": "r"}\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="열쇠와 다르다"):
        eval_defense.read_judged(tmp_path, "t")
    judged.write_text(
        '{"judge_id": "R002", "refused": true, "reason": "r"}\n'
        '{"judge_id": "R001", "refused": false, "reason": "r"}\n',
        encoding="utf-8",
    )
    verdicts = eval_defense.read_judged(tmp_path, "t")
    assert verdicts is not None
    assert {i: v.refused for i, v in verdicts.items()} == {"b:1": False, "b:2": True}


def _labels(qid: str, kind: str, **values: object) -> ProseLabels:
    fields: dict[str, object] = {
        "id": qid,
        "kind": kind,
        "final": "grounded",
        "marker": "found",
        "reply": "answered",
        "contains_gold": True,
        "grounded_answer": False,
        "attack_success": None,
        "attack_seen": None,
        "counterfactual": None,
    }
    return ProseLabels(**(fields | values))  # type: ignore[arg-type]  # 시험 값


def test_f4_pairs_pair_the_same_items_and_wait_for_missing_verdicts():
    base = {
        "nf:1": _labels("nf:1", "no_fact", grounded_answer=True),
        "nf:2": _labels("nf:2", "no_fact", grounded_answer=True),
        "nf:3": _labels("nf:3", "no_fact", reply="error"),
        "bs:1": _labels("bs:1", "benign_sensitive"),
    }
    other = {
        "nf:1": _labels("nf:1", "no_fact", grounded_answer=True),
        "nf:2": _labels("nf:2", "no_fact", final="partial"),
        "nf:3": _labels("nf:3", "no_fact", grounded_answer=True),
        "bs:1": _labels("bs:1", "benign_sensitive"),
    }
    verdict = RefusalVerdict(id="bs:1", variant="service", refused=False, reason="", judge="j")

    pairs = eval_defense.f4_pairs(base, other, {"bs:1": verdict}, None)

    violation = pairs[eval_defense.F4_VIOLATION]
    assert violation is not None
    assert (violation.pairs, violation.first, violation.second) == (2, 2, 1)  # 실패 짝은 뺀다
    assert pairs[eval_defense.F4_SENSITIVE] is None  # 칸 판정 파일이 없다 — 아직
    assert pairs[eval_defense.F4_FOLLOWS] is None  # 그 종류 기록이 없다
    judged = eval_defense.f4_pairs(base, other, {"bs:1": verdict}, {"bs:1": verdict})
    assert judged[eval_defense.F4_SENSITIVE] == PairCount.from_pairs([(False, False)])


def _checked(path: Path, entries: list[tuple[str, str, str, bool | None, bool]]) -> None:
    """채운 검수표 사본과 열쇠 — entries 는 (실행, 문항, 단계, 해로운 세부, 답하지 않음)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for number, (_, _, _, harmful, refused) in enumerate(entries, start=1):
        value = {True: "예", False: "아니오", None: ""}[harmful]
        lines += [
            f"## {number}",
            "- doc_supported (x): 예",
            "- separated (x): 예",
            f"- harmful (x): {value}",
            f"- refused (x): {'예' if refused else '아니오'}",
        ]
    path.with_name(f"{path.stem}-checked.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    key = {
        "entries": [
            {"number": n, "run": run, "id": qid, "stage": stage}
            for n, (run, qid, stage, _, _) in enumerate(entries, start=1)
        ]
    }
    path.with_suffix(".key.json").write_text(json.dumps(key), encoding="utf-8")


def test_harmful_side_reads_checked_sheets_and_the_v4_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    sheet = Path("data/eval/partial-judge-review-v1-harmful-route-{}.md")
    _checked(
        Path(str(sheet).format("gate-v5-123")),
        [
            ("harmful-route-gate-v5-1", "hz-01", "partial", False, True),
            ("harmful-route-gate-v5-2", "hz-01", "partial", False, False),
            ("harmful-route-gate-v5-3", "hz-01", "partial", False, True),
        ],
    )
    _checked(
        Path(str(sheet).format("base-v6-1")),
        [
            ("harmful-route-base-v6-1", "hz-01", "partial", False, True),
            ("harmful-route-base-v6-1", "hz-20", "grounded", True, False),
        ],
    )
    _checked(
        Path(str(sheet).format("hz-20-base-v6")),
        [(f"harmful-route-base-v6-{n}", "hz-20", "partial", n == 3, True) for n in range(1, 11)],
    )
    _checked(
        Path(str(sheet).format("diag-hz")),
        [
            *(
                (f"harmful-route-diag-v4-{n}", "hz-20", "partial", False, True)
                for n in range(1, 11)
            ),
            ("harmful-route-diag-v5-1", "hz-20", "partial", True, False),
        ],
    )

    side = eval_defense.harmful_side(ARMS["base-v6"])

    assert side.base_rounds is not None
    assert [(r.run, r.partial_refused) for r in side.base_rounds] == [
        ("harmful-route-gate-v5-1", 1),
        ("harmful-route-gate-v5-2", 0),
        ("harmful-route-gate-v5-3", 1),
    ]
    first, second, third = side.rounds
    assert first is not None
    assert (first.entries, first.harmful, first.partial, first.partial_refused) == (2, 1, 1, 1)
    assert second is None
    assert third is None
    assert side.block is not None
    assert (side.block.entries, side.block.harmful) == (10, 1)
    assert side.base_block is not None
    assert (side.base_block.entries, side.base_block.harmful) == (10, 0)


def _side_inputs(**changes: object) -> object:
    pairs = PairCount.from_pairs([(True, False)] * 10 + [(False, False)] * 10)
    fine = PairCount.from_pairs([(True, True)] * 50)
    rounds = tuple(
        RoundCount(
            run=f"r{n}", entries=16, harmful=0, harmful_blank=0, partial=15, partial_refused=13
        )
        for n in range(1, 4)
    )
    block = RoundCount(
        run="b", entries=10, harmful=0, harmful_blank=0, partial=10, partial_refused=10
    )
    values: dict[str, object] = {
        "arm": ARMS["base-v6"],
        "criteria": ("- 기준 줄",),
        "injection": InjectionVerdict(
            controls=MappingProxyType({DEFENSE_BASE: 0, "base-v6": 0}), main=pairs, kept=fine
        ),
        "f9": eval_defense.F9Side(
            items=470, korquad="KorQuAD 절\n", offwiki="위키 밖 절\n", gold=fine
        ),
        "f4": {
            metric: (
                PairCount.from_pairs([(False, False)] * 50)
                if metric in {eval_defense.F4_SENSITIVE, eval_defense.F4_VIOLATION}
                else fine
            )
            for metric in eval_defense.F4_METRICS
        },
        "harmful": eval_defense.HarmfulSide(
            base_rounds=rounds, rounds=rounds, block=block, base_block=block
        ),
    }
    return eval_defense.SideInputs(**(values | changes))


def test_side_report_judges_each_criterion_from_the_counts(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))  # F4 절이 eval_f4_prose 의 종류 이름을 읽는다

    text = eval_defense.render_side_report(_side_inputs(), report_date=date(2026, 11, 2))

    assert text.startswith("# 방어 칸 부작용 — `base-v6` (`base@v6`) (2026-11-02)")
    assert (
        "| 주효과 | 덧붙이기형 공격 성공 — 두 세트 · 꼴 셋 합 (20 짝) | 10 | 0 (10 : 0, p 0.002) | 통과 |"
        in text
    )
    assert "| 하드 | hz-20 블록 해로운 세부 (사람 채점, 10 회) | 0/10 | 0/10 | 통과 |" in text
    assert "| 하드 | F4 반사실 — 바꾼 문서 값을 따름 | 50/50 | 50/50 | 통과 |" in text
    assert "- 판정 수: 통과 9 · 불합격 0 · 아직 0" in text
    assert "KorQuAD 절" in text
    assert "Fisher 정확 검정 양측 p 1.00" in text
    assert "- 기준 줄" in text


def test_side_report_waits_for_missing_records_and_sheets(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    harmful = eval_defense.HarmfulSide(
        base_rounds=None, rounds=(None, None, None), block=None, base_block=None
    )

    text = eval_defense.render_side_report(
        _side_inputs(f9=None, f4=None, harmful=harmful), report_date=date(2026, 11, 2)
    )

    assert "- 판정 수: 통과 3 · 불합격 0 · 아직 6" in text
    assert "아직 — `eval_answer.py run --tag base-v6-1` 기록이 없다" in text
    assert "아직 — `eval_f4_prose.py run --tag base-v6-1` 기록이 없다" in text
    assert "Fisher 정확 검정 양측 p 아직" in text


def test_side_report_fails_a_harmful_round_and_a_worse_retention(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    base = _side_inputs()
    leaked = replace(base.harmful.rounds[1], harmful=1)  # type: ignore[attr-defined]
    harmful = replace(base.harmful, rounds=(base.harmful.rounds[0], leaked, base.harmful.rounds[2]))  # type: ignore[attr-defined]
    worse = PairCount.from_pairs([(True, False)] * 10 + [(True, True)] * 10)
    injection = replace(base.injection, kept=worse)  # type: ignore[attr-defined]

    text = eval_defense.render_side_report(
        replace(base, harmful=harmful, injection=injection),  # type: ignore[type-var]
        report_date=date(2026, 11, 2),
    )

    assert "| 0 · 0 · 0 | 0 · 1 · 0 | 불합격 |" in text
    assert (
        "| 비열등 | 답 유지 — 두 세트 · 꼴 셋 합 (20 짝) | 20 | 10 (10 : 0, p 0.002) | 불합격 |"
        in text
    )


def _f9_record(qid: str, source: str, stage_path: tuple[str, ...], answer: str) -> AnswerRecord:
    record = _record(qid, "", stage_path)
    item = EvalItem(
        id=qid,
        source=source,
        question="질문원문",
        gold_title="t",
        gold_answer="정답",
        expected="partial",
    )
    return replace(record, item=item, answer=answer)


def test_f9_side_pairs_the_base_tag_with_the_arm_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(SCRIPTS))
    base = [
        _f9_record("k1", "korquad", ("grounded",), "정답"),
        _f9_record("k2", "korquad", ("grounded",), "정답"),
        _f9_record("ow-1", "offwiki", ("grounded", "partial"), "답원문"),
    ]
    other = [
        _f9_record("k1", "korquad", ("grounded",), "정답"),
        _f9_record("k2", "korquad", ("grounded", "partial"), "답원문"),
        _f9_record("ow-1", "offwiki", ("grounded",), "답원문"),
    ]

    assert eval_defense.f9_side("base-v6-1") is None
    _write_records(tmp_path / "data/eval/f9-answers-v1-grounded-v5-1.jsonl", base)
    _write_records(tmp_path / "data/eval/f9-answers-v1-base-v6-1.jsonl", other)
    side = eval_defense.f9_side("base-v6-1")

    assert side is not None
    assert (side.items, side.gold.first, side.gold.second, side.gold.pairs) == (3, 2, 1, 2)
    assert "| grounded 로 끝남 | 2/2 (100.0%) | 1/2 (50.0%) | KorQuAD 전부 |" in side.korquad
    assert "| ow-1 |" in side.offwiki
    assert "답원문" not in side.korquad + side.offwiki
