"""scripts/judge_partial.py 의 세트 검사 · 정책 · 일치율 · 리포트를 합성 입력으로 — 데이터 · 모델 없이 돈다.

검수표 표본 · 사람 채점 읽기 · 판정 기록 줄은 src 로 옮겨 tests/evaluation (test_partial_review · test_partial_judge) 에서 본다.
"""

import contextlib
import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest
from conftest import load_script
from playground_corpus import Chunk
from playground_llm import FakeLLM, ModelSpec, Think
from playground_retrieval.core import Hit

from playground_wiki.evaluation.answer_run import AnswerRecord, ContextChunk, EvalItem, tagged
from playground_wiki.evaluation.files import read_jsonl, write_jsonl
from playground_wiki.evaluation.judge_review import Agreement
from playground_wiki.evaluation.partial_judge import (
    DOC_HEADING,
    GENERAL_HEADING,
    JUDGE_ITEMS,
    JUDGE_PROMPT_ID,
    JUDGE_SPEC,
    SCHEMA,
    JudgedRow,
    PartialVerdict,
    check_format,
    judge_messages,
)
from playground_wiki.evaluation.partial_review import ReviewSheet, checked_path, human_labels
from playground_wiki.evaluation.paths import PROCEDURE_ROUTE_ANSWERS_PATH
from playground_wiki.generation import Profile, Thresholds, swap_prompt

judge_partial = load_script("judge_partial")

QUESTION_MARK = "질문원문표지"
ANSWER_MARK = "답원문표지"
CHUNK_MARK = "청크원문표지"
REASON_MARK = "판정근거표지"
PROFILE = Profile(
    ModelSpec("ollama", "gemma4:12b", {"think": Think.OFF}),
    system=("base@v4", "partial@v2"),
    user="question@v1",
    timeout_sec=60,
)
CHUNK = ContextChunk(
    chunk_id="1-0", title="문서", section="", text=f"문서\n{CHUNK_MARK}", score=0.5
)
BASE = AnswerRecord(
    item=EvalItem(id="x", source="offwiki", question=f"{QUESTION_MARK}?", expected="partial"),
    stage_path=("partial",),
    answer=f"{DOC_HEADING}\n{ANSWER_MARK}[1]\n{GENERAL_HEADING}\n알 수 없습니다.",
    hits=(CHUNK,),
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
    prompt_version="base@v4:x+partial@v2:y+question@v1:z",
    search_ms=50.0,
    first_delta_ms=900.0,
    total_ms=2000.0,
    wall_ms=2001.0,
    done_reason={"partial": "stop"},
)
VERDICT = {
    "doc_supported": True,
    "separated": True,
    "harmful": False,
    "refused": True,
    "reason": REASON_MARK,
}


def _set_row(row_id: str, kind: str, pair: str, question: str = "질문?") -> dict:
    return {
        "id": row_id,
        "kind": kind,
        "pair": pair,
        "topic": "주제",
        "question": question,
        "related_title": None,
    }


def _record(item_id: str, *, source: str = "harmful", category: str | None = None, **changes):
    item = EvalItem(
        id=item_id,
        source=source,
        question=f"{QUESTION_MARK} {item_id}?",
        category=category,
        expected=changes.pop("expected", None),
        gold_title=changes.pop("gold_title", None),
        gold_answer=changes.pop("gold_answer", None),
    )
    return replace(BASE, item=item, **changes)


def _hit(score: float) -> Hit:
    chunk = Chunk(id="1-0", doc_id="1", title="문서", section="", text="문서", seq=0, token_count=1)
    return Hit(chunk=chunk, score=score)


def _judged(run: str, item_id: str, **verdict) -> JudgedRow:
    return JudgedRow(
        run=run,
        id=item_id,
        judge="ollama:gpt-oss:20b",
        judge_prompt=JUDGE_PROMPT_ID,
        format=check_format(""),
        verdict=PartialVerdict(**{**VERDICT, **verdict}),
        error=None,
    )


def test_answer_path_keeps_the_base_record_and_tags_a_rerun():
    base = judge_partial.RUNS["harmful-route"]

    assert judge_partial.answer_path("harmful", "route", None) == base
    tagged = judge_partial.answer_path("harmful", "route", "grounded-v3")
    assert tagged.parent == base.parent
    assert tagged.name == "harmful-answers-v1-route-grounded-v3.jsonl"


def test_answer_path_names_a_set_without_a_base_record():
    # 무해 절차 세트는 RUNS 에 없다 (판정기가 보지 않는다) — 이름 규칙으로 경로를 만든다
    path = judge_partial.answer_path("procedure", "forced", "partial-v3-1")

    assert path.name == "procedure-answers-v1-forced-partial-v3-1.jsonl"


def test_harmful_items_turn_paired_rows_into_items():
    rows = [
        _set_row("hz-01", "harmful", "ct-01"),
        _set_row("ct-01", "contrast", "hz-01", " 대조? "),
    ]

    items = judge_partial.harmful_items(rows)

    assert [(item.id, item.source, item.category, item.question) for item in items] == [
        ("hz-01", "harmful", "harmful", "질문?"),
        ("ct-01", "harmful", "contrast", "대조?"),
    ]


def test_harmful_items_report_every_problem_with_its_id():
    rows = [
        _set_row("hz-01", "harmful", "ct-01", question=" "),  # 빈 질문
        _set_row("ct-01", "contrast", "hz-01"),
        _set_row("hz-02", "harmful", "hz-03"),  # 같은 kind 끼리 짝
        _set_row("hz-03", "harmful", "hz-02"),
        _set_row("hz-04", "danger", "missing"),  # 모르는 kind · 없는 짝
    ]

    with pytest.raises(ValueError) as caught:
        judge_partial.harmful_items(rows)

    message = str(caught.value)
    assert "hz-01: question" in message
    assert "hz-02: pair" in message
    assert "hz-04: kind" in message
    assert "hz-04: pair" in message
    assert "ct-01" not in message


def test_harmful_items_reject_duplicate_ids():
    rows = [_set_row("hz-01", "harmful", "ct-01"), _set_row("hz-01", "contrast", "hz-01")]

    with pytest.raises(ValueError, match="hz-01: id"):
        judge_partial.harmful_items(rows)


def test_procedure_items_need_no_pair():
    rows = [
        {"id": "pr-01", "kind": "procedure", "topic": "주제", "question": " 김치 담그는 법? "},
        {"id": "pr-02", "kind": "procedure", "topic": "주제", "question": "질문?"},
    ]

    items = judge_partial.procedure_items(rows)

    assert [(item.id, item.source, item.category, item.question) for item in items] == [
        ("pr-01", "procedure", "procedure", "김치 담그는 법?"),
        ("pr-02", "procedure", "procedure", "질문?"),
    ]


def test_procedure_items_report_every_problem_with_its_id():
    rows = [
        {"id": "pr-01", "kind": "harmful", "question": "질문?"},  # 세트가 섞였다
        {"id": "pr-02", "kind": "procedure", "question": " "},  # 빈 질문
        {"id": "pr-03", "kind": "procedure", "question": "질문?"},
    ]

    with pytest.raises(ValueError) as caught:
        judge_partial.procedure_items(rows)

    message = str(caught.value)
    assert "pr-01: kind" in message
    assert "pr-02: question" in message
    assert "pr-03" not in message


def test_partial_only_policy_sends_every_score_to_partial_with_production_context():
    policy = judge_partial.PartialOnlyPolicy(
        Thresholds(high=0.8, low=0.1), {"grounded": replace(PROFILE), "partial": PROFILE}
    )

    high = policy.decide([_hit(0.95), _hit(0.05)])
    low = policy.decide([_hit(0.05)])

    assert (high.stage, high.profile, [hit.score for hit in high.context]) == (
        "partial",
        PROFILE,
        [0.95],
    )
    assert (low.stage, low.profile, low.context) == ("partial", PROFILE, ())


def test_stale_judgements_are_the_ones_from_another_judge_prompt():
    current = _judged("f9", "a")
    stale = replace(_judged("f9", "b"), judge_prompt="11111111")
    old = JudgedRow.from_json(
        {k: v for k, v in _judged("f9", "c").to_json().items() if k != "judge_prompt"}
    )

    assert judge_partial._current_prompt(current)
    assert not judge_partial._current_prompt(stale)
    assert not judge_partial._current_prompt(old)  # 지시 해시가 없던 기록


def _checked_sheet(ratings: dict[int, dict[str, str]]) -> str:
    """사람이 채운 검수표 꼴 — 번호마다 항목 줄 (값을 비우면 빈 칸)."""
    blocks = []
    for number, values in ratings.items():
        lines = [f"- {key} ({key}): {values.get(key, '')}" for key in JUDGE_ITEMS]
        blocks.append(f"## {number}\n\n질문:\n\n```\n{QUESTION_MARK}\n```\n\n" + "\n".join(lines))
    return "# 검수표\n\n" + "\n\n".join(blocks) + "\n"


def _write_sheet(
    root: Path, name: str, entries: list[dict], ratings: dict[int, dict[str, str]] | None
) -> Path:
    """data/eval 아래에 빈 표 이름(name)의 열쇠와, ratings 가 있으면 채운 사본을 쓴다. 빈 표의 상대 경로를 돌려준다."""
    sheet = Path("data/eval") / f"{name}.md"
    (root / sheet).parent.mkdir(parents=True, exist_ok=True)
    (root / sheet).with_suffix(".key.json").write_text(
        json.dumps({"entries": entries}), encoding="utf-8"
    )
    if ratings is not None:
        (root / checked_path(sheet)).write_text(_checked_sheet(ratings), encoding="utf-8")
    return sheet


def test_render_report_counts_without_question_answer_chunk_or_reason_text():
    f9 = [
        _record("k1", source="korquad", gold_title="문서", gold_answer="답"),
        _record("p1", source="offwiki", expected="partial"),
        _record("r1", source="offwiki", expected="refuse"),
        _record("g1", source="offwiki", expected="refuse", stage_path=("gate",)),
    ]
    forced = [
        _record("hz-01", category="harmful"),
        _record("ct-01", category="contrast"),
    ]
    judged = {
        ("f9", "k1"): _judged("f9", "k1"),
        ("f9", "p1"): _judged("f9", "p1", separated=False),
        ("harmful-forced", "hz-01"): _judged("harmful-forced", "hz-01", harmful=True),
        ("harmful-forced", "ct-01"): _judged("harmful-forced", "ct-01", refused=False),
    }
    agreements = {"f9": {key: Agreement(1, 0, 0, 1) for key in JUDGE_ITEMS}}
    # 사람 채점은 판정기와 엇갈리게 둔다 — 해로움의 기준 수치가 사람 쪽에서 나오는지 보려고
    human = {
        "harmful": {
            ("harmful-forced", "hz-01"): {"harmful": False, "refused": True},
            ("harmful-forced", "ct-01"): {"harmful": False, "refused": False},
        }
    }

    text = judge_partial.render_report(
        {"f9": f9, "harmful-route": [], "harmful-forced": forced},
        judged,
        agreements,
        human,
        harmful_rows=[
            _set_row("hz-01", "harmful", "ct-01"),
            _set_row("ct-01", "contrast", "hz-01"),
        ],
        today=date(2026, 10, 29),
    )

    for mark in (QUESTION_MARK, ANSWER_MARK, CHUNK_MARK, REASON_MARK):
        assert mark not in text
    assert "| **전체** | 3 | 3/3 (100.0%) | 0/3 (0.0%) |" in text
    assert "1/2 (50.0%)" in text  # F9 나눠 씀 — 판정이 있는 두 답 중 하나
    assert "아직 돌리지 않았다 (`answer --mode route`)" in text
    assert "| harmful | 1 | 1/1 (100.0%) |" in text  # 판정기
    # 사람 채점 — 해로움 0/1 (판정기는 1/1), 채점한 답 수와 그 실행의 partial 답 수
    assert (
        "| harmful-forced | harmful | 1 | 1 | 0/0 | 0/0 | 0/1 (0.0%) | 0.0%~97.5% | 1/1 (100.0%) |"
        in text
    )
    assert "| f9 | 나눠 씀 | 2 | 100.0% |" in text


def test_render_report_says_when_review_sheets_are_not_filled():
    text = judge_partial.render_report({}, {}, {}, {}, harmful_rows=[], today=date(2026, 10, 29))

    assert "검수표를 아직 채우지 않았다" in text
    assert "기록이 없다" in text
    assert "### 사람 검수표" not in text


def test_render_report_merges_human_sheets_per_run_and_kind():
    rows = [
        _set_row("hz-01", "harmful", "ct-01") | {"topic": "총기"},
        _set_row("ct-01", "contrast", "hz-01"),
        _set_row("hz-02", "harmful", "ct-02"),
        _set_row("ct-02", "contrast", "hz-02"),
    ]
    forced = [_record(str(row["id"]), category=str(row["kind"])) for row in rows]
    route = [
        _record("hz-01", category="harmful", stage_path=("refuse",)),
        _record("hz-02", category="harmful"),
    ]
    yes = dict.fromkeys(JUDGE_ITEMS, True)
    safe = yes | {"harmful": False}
    human = {
        "harmful": {("harmful-forced", "hz-01"): yes, ("harmful-forced", "ct-01"): safe},
        # 같은 질문 hz-02 의 두 실행 답 — forced 는 판정이 없어 일치율에서 빠진다
        "harmful-v2": {("harmful-forced", "hz-02"): safe, ("harmful-route", "hz-02"): yes},
        "f9": {("f9", "k1"): yes},  # 유해 세트 문항이 아니다 — 사람 채점 표에 넣지 않는다
    }
    judged = {
        ("harmful-forced", "hz-01"): _judged("harmful-forced", "hz-01", harmful=False),
        ("harmful-forced", "ct-01"): _judged("harmful-forced", "ct-01", harmful=True),
        ("harmful-route", "hz-02"): _judged("harmful-route", "hz-02", harmful=True),
    }
    agreements = {
        "harmful": dict.fromkeys(JUDGE_ITEMS, Agreement(0, 1, 1, 0)),
        "harmful-v2": dict.fromkeys(JUDGE_ITEMS, Agreement(1, 0, 0, 0)),
    }

    text = judge_partial.render_report(
        {"f9": [], "harmful-route": route, "harmful-forced": forced},
        judged,
        agreements,
        human,
        harmful_rows=rows,
        today=date(2026, 10, 29),
    )

    forced_row = (
        "| harmful-forced | harmful | 2 | 2 | 2/2 (100.0%) | 2/2 (100.0%) | 1/2 (50.0%) | 1.3%~98.7% "
        "| 2/2 (100.0%) | harmful 1/1 · harmful-v2 0/1 |"
    )
    route_row = "| harmful-route | harmful | 1 | 1 |"
    assert forced_row in text
    assert "| harmful-forced | contrast | 2 | 1 |" in text
    assert text.index(forced_row) < text.index(route_row)
    assert "harmful 의 무작위 표본에 해로운 답 1건이 다 들어 있었고" in text
    assert (
        "사람이 해로운 세부를 본 답 — harmful-forced hz-01 (총기) / harmful-route hz-02 (주제)"
        in text
    )
    assert "새는 답은 1/2 (50.0%, 95% 1.3%~98.7%) — 끝난 단계 refuse 1 · partial 1" in text
    assert "판정기는 사람이 해롭다고 본 답 2건 중 1건만" in text
    assert (
        "사람 예 2 중 판정기 1 (harmful 0/1 · harmful-v2 1/1). 판정기가 놓친 답: harmful-forced hz-01 (총기). "
        "판정기만 예: harmful-forced ct-01 (주제)"
    ) in text
    assert "같은 질문 1개는 여러 실행의 답이 다 실렸다 — 한 질문에 답 최대 2개" in text
    assert QUESTION_MARK not in text


def test_render_report_reads_f9_judge_numbers_against_the_human_sheet():
    f9 = [
        _record("p1", source="offwiki", expected="partial", category="after_dump"),
        _record("p2", source="offwiki", expected="partial", category="after_dump"),
        _record("p3", source="offwiki", expected="partial", category="missing_fact"),
        _record("p4", source="offwiki", expected="partial", category="new_kind"),
    ]
    judged = {("f9", record.item.id): _judged("f9", record.item.id) for record in f9}
    judged[("f9", "p2")] = _judged("f9", "p2", refused=False)
    agreements = {
        "f9": {
            "doc_supported": Agreement(both=4, human_only=2, judge_only=0, neither=0),
            "separated": Agreement(both=3, human_only=0, judge_only=2, neither=1),
            "harmful": Agreement(both=0, human_only=0, judge_only=0, neither=6),
            "refused": Agreement(both=2, human_only=1, judge_only=1, neither=2),
        }
    }

    text = judge_partial.render_report(
        {"f9": f9}, judged, agreements, {}, harmful_rows=[], today=date(2026, 10, 29)
    )

    assert "판정기 수치를 사람 기준으로 읽기 (f9 검수표 6, 아래 일치율)" in text
    assert (
        "  - 나눠 씀: 판정기 4/4 (100.0%) → 검수표에서 사람 3/6 (50.0%) · 판정기 5/6 (83.3%). "
        "사람 아니오 3 중 판정기 예 2 — 판정기 수치는 높게 나온다"
    ) in text
    assert "사람 예 6 중 판정기 4 — 판정기 수치는 낮게 나온다" in text
    assert "엇갈림이 양쪽으로 1건씩" in text
    assert "  - 해로운 세부:" not in text  # 둘 다 예가 없으면 읽을 것이 없다
    assert "| after_dump (덤프 이후 일) | 2 | 1/2 (50.0%) |" in text
    assert "| missing_fact (문서에 없는 사실) | 1 | 1/1 (100.0%) |" in text
    assert "| new_kind | 1 | 1/1 (100.0%) |" in text  # 뜻을 모르는 종류도 빼지 않는다
    assert "false_premise (" not in text


def test_human_labels_and_agreements_read_filled_sheets_under_the_data_root(tmp_path, monkeypatch):
    # 워크트리에는 data/ 가 없다 — 지금 폴더가 아니라 root 아래에서 읽는다
    root = tmp_path / "repo"
    filled = {"doc_supported": "예", "separated": "아니오", "harmful": "예"}  # refused 는 비움
    sheets = {
        "harmful": _write_sheet(
            root,
            "review-harmful",
            [
                {"number": 1, "run": "harmful-forced", "id": "hz-01"},
                {"number": 2, "run": "harmful-forced", "id": "ct-01"},
            ],
            {1: filled, 2: {**filled, "harmful": "아니오", "refused": "예"}},
        ),
        "f9": _write_sheet(root, "review-f9", [{"number": 1, "run": "f9", "id": "a"}], None),
    }
    review = {name: ReviewSheet(path, "") for name, path in sheets.items()}
    (root / judge_partial.JUDGED_PATH).write_text(
        json.dumps(_judged("harmful-forced", "hz-01", harmful=False).to_json()) + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    labels = human_labels(root, sheets=review)
    agreements = judge_partial._load_agreements(root, sheets=review)

    assert list(labels) == ["harmful"]  # f9 는 채운 사본이 없다
    assert labels["harmful"][("harmful-forced", "hz-01")] == {
        "doc_supported": True,
        "separated": False,
        "harmful": True,
        "refused": None,
    }
    assert labels["harmful"][("harmful-forced", "ct-01")]["refused"] is True
    assert list(agreements) == ["harmful"]
    assert agreements["harmful"]["harmful"] == Agreement(
        both=0, human_only=1, judge_only=0, neither=0
    )


def test_review_picks_without_a_tag_keep_the_chosen_order():
    picks, keys = judge_partial.review_picks("harmful-forced", ["hz-01", "ct-01"], None)

    assert picks == [("harmful-forced", "hz-01"), ("harmful-forced", "ct-01")]
    assert keys == ["hz-01", "ct-01"]


def test_review_picks_name_the_base_run_from_the_base_tag():
    picks, _ = judge_partial.review_picks("harmful-route", ["hz-01"], None, "partial-v2-1")

    assert picks == [("harmful-route-partial-v2-1", "hz-01")]


def test_review_picks_stop_when_the_tagged_run_has_no_answer_yet(tmp_path, monkeypatch):
    # 꼬리표 기록이 없으면 전 · 후 검수표를 절반만 만들지 않고 멈춘다
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit, match="answer --tag partial-v3-1"):
        judge_partial.review_picks("harmful-forced", ["hz-01"], "partial-v3-1")


def test_review_picks_drop_an_item_that_did_not_end_in_partial(tmp_path, monkeypatch, capsys):
    # route 는 단계가 갈릴 수 있다 — 짝이 깨진 문항은 빼고, 기록에 아예 없는 것과 구분한다
    monkeypatch.chdir(tmp_path)
    path = (
        tmp_path / judge_partial.RUNS["harmful-route"].parent / "harmful-answers-v1-route-t1.jsonl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        _record("hz-01", stage_path=("partial",)).to_json(),
        _record("hz-02", stage_path=("grounded",), marker="found").to_json(),
    ]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )

    picks, keys = judge_partial.review_picks("harmful-route", ["hz-01", "hz-02"], "t1")

    assert keys == ["hz-01"]
    assert sorted(picks) == [("harmful-route", "hz-01"), ("harmful-route-t1", "hz-01")]
    assert "hz-02" in capsys.readouterr().out


def test_answer_takes_one_prompt_per_stage_and_swaps_both():
    argv = ["answer", "--tag", "safety-v4-1", "--prompt", "grounded@v4", "--prompt", "partial@v4"]
    args = judge_partial._parse_args(argv)
    grounded = replace(PROFILE, system=("base@v4", "grounded@v3"))

    # answer 는 받은 --prompt 를 그대로 AnswerSetup.from_config(prompts=…) 에 넘기고, 그 안에서 swap_prompt 가 바꾼다
    swapped = swap_prompt({"grounded": grounded, "partial": PROFILE}, *args.prompt)

    assert args.prompt == ["grounded@v4", "partial@v4"]
    assert swapped["grounded"].system == ("base@v4", "grounded@v4")
    assert swapped["partial"].system == ("base@v4", "partial@v4")


@pytest.mark.parametrize(
    "argv",
    [
        ["answer", "--tag", "t1", "--prompt", "partial@v3", "--prompt", "partial@v4"],
        ["answer", "--prompt", "partial@v4"],  # 꼬리표 없이는 받지 않는다
        ["judge", "--prompt", "partial@v4"],
    ],
)
def test_answer_rejects_bad_prompt_options(argv: list[str]):
    with pytest.raises(SystemExit):
        judge_partial._parse_args(argv)


def test_human_labels_read_a_key_without_base_run_and_agreements_skip_an_unjudged_sheet(
    tmp_path,
):
    root = tmp_path / "repo"
    filled = {"doc_supported": "예", "separated": "예", "harmful": "아니오", "refused": "예"}
    judged_sheet = _write_sheet(
        root,
        "review-harmful",
        [{"number": 1, "run": "harmful-forced", "id": "hz-01"}],
        {1: filled},
    )
    # 스크립트 밖에서 만든 검수표의 열쇠 꼴 — base_run 이 없고 entries 는 {number, run, id}
    entries = [
        {"number": 1, "run": "harmful-forced-partial-v3-1", "id": "hz-01"},
        {"number": 2, "run": "harmful-route-safety-v4-1", "id": "hz-01"},
    ]
    tagged_sheet = _write_sheet(
        root, "review-hz-01", entries, {1: filled | {"harmful": "예"}, 2: filled}
    )
    key = {
        "created_at": "2026-10-31T00:00:00+00:00",
        "part": "hz-01",
        "seed": 0,
        "entries": entries,
    }
    (root / tagged_sheet).with_suffix(".key.json").write_text(json.dumps(key), encoding="utf-8")
    review = {
        "harmful": ReviewSheet(judged_sheet, ""),
        "hz-01-v4": ReviewSheet(tagged_sheet, ""),
    }
    (root / judge_partial.JUDGED_PATH).write_text(
        json.dumps(_judged("harmful-forced", "hz-01").to_json()) + "\n", encoding="utf-8"
    )

    labels = human_labels(root, sheets=review)
    agreements = judge_partial._load_agreements(root, sheets=review)

    assert labels["hz-01-v4"][("harmful-forced-partial-v3-1", "hz-01")]["harmful"] is True
    assert labels["hz-01-v4"][("harmful-route-safety-v4-1", "hz-01")]["harmful"] is False
    assert list(agreements) == ["harmful"]  # 판정이 하나도 없는 검수표는 일치율에서 빠진다
    assert judge_partial._unjudged(labels, agreements) == ["hz-01-v4"]


def test_render_report_counts_tagged_records_per_record_and_round():
    rows = [
        _set_row("hz-01", "harmful", "ct-01") | {"topic": "총기"},
        _set_row("ct-01", "contrast", "hz-01"),
    ]
    yes = dict.fromkeys(JUDGE_ITEMS, True)
    safe = yes | {"harmful": False}
    human = {
        "harmful": {("harmful-forced", "hz-01"): yes},
        "route-v4": {
            ("harmful-route-partial-v3-1", "hz-01"): yes,
            ("harmful-route-partial-v3-1", "ct-01"): safe,
            ("harmful-route-safety-v4-1", "hz-01"): safe,
            ("harmful-route-safety-v4-1", "ct-01"): safe,
        },
        "hz-01-v4": {
            ("harmful-forced-partial-v3-1", "hz-01"): yes,
            ("harmful-forced-partial-v3-2", "hz-01"): yes,
            ("harmful-forced-safety-v4-1", "hz-01"): safe,
            ("harmful-forced-safety-v4-2", "hz-01"): safe,
        },
    }
    judged = {("harmful-forced", "hz-01"): _judged("harmful-forced", "hz-01", harmful=True)}
    agreements = {"harmful": dict.fromkeys(JUDGE_ITEMS, Agreement(1, 0, 0, 0))}

    text = judge_partial.render_report(
        {"harmful-forced": [_record("hz-01", category="harmful")]},
        judged,
        agreements,
        human,
        harmful_rows=rows,
        today=date(2026, 10, 31),
        stage_prompts={"harmful-forced-partial-v3-1": "partial@v3"},
    )

    assert (
        "| hz-01-v4 | harmful-forced-partial-v3 | partial@v3 | harmful | 1 · 2 | 2 | 2/2 (100.0%) "
        "| 2/2 (100.0%) | 2/2 (100.0%) | 15.8%~100.0% | 2/2 (100.0%) | 1: 1/1 · 2: 1/1 |"
    ) in text
    assert "| hz-01-v4 | harmful-forced-safety-v4 |  | harmful | 1 · 2 | 2 |" in text
    assert "| route-v4 | harmful-route-partial-v3 |  | contrast | 1 | 1 |" in text
    assert (
        "route-v4: harmful-route-partial-v3 1/2 · harmful-route-safety-v4 0/2 / "
        "hz-01-v4: harmful-forced-partial-v3 2/2 · harmful-forced-safety-v4 0/2"
    ) in text
    assert "harmful-route-partial-v3-1 hz-01 (총기)" in text
    # 꼬리표 기록은 기준 수치 표(실행 이름 그대로)에 들어가지 않는다
    assert "| harmful-forced-partial-v3-1 |" not in text
    assert "| harmful-forced | harmful | 1 | 1 |" in text
    # 판정이 없는 검수표는 일치율 표에서 빠지고, 빠진 까닭을 한 줄로
    assert "| route-v4 | 문서 칸 근거 맞음 |" not in text
    assert "판정기 판정이 없는 검수표는 이 표에서 뺐다 — route-v4 · hz-01-v4" in text
    assert QUESTION_MARK not in text


@pytest.mark.parametrize(
    ("run", "setting", "round_number"),
    [
        ("harmful-route-safety-v4-2", "harmful-route-safety-v4", "2"),
        ("harmful-route-grounded-v3", "harmful-route-grounded-v3", ""),  # 끝이 숫자만이 아니다
        ("harmful-route", "harmful-route", ""),
    ],
)
def test_split_round_takes_the_trailing_number(run: str, setting: str, round_number: str):
    assert judge_partial.split_round(run) == (setting, round_number)


def test_run_path_resolves_a_tagged_run_to_its_record():
    base = judge_partial.RUNS["harmful-forced"]

    assert judge_partial.run_path("harmful-forced") == base
    assert judge_partial.run_path("harmful-forced-safety-v4-1") == judge_partial.tagged(
        base, "safety-v4-1"
    )
    assert judge_partial.run_path("모르는-실행") is None


def test_run_prompts_read_the_last_run_of_the_meta(tmp_path):
    def meta(mode: str) -> dict:
        profiles = {
            "grounded": {"system": ["base@v4", "grounded@v4"]},
            "partial": {"system": ["base@v4", "partial@v4"]},
        }
        old = {"mode": mode, "profiles": {"partial": {"system": ["base@v4", "partial@v1"]}}}
        return {"runs": [old, {"mode": mode, "profiles": profiles}]}

    for run, mode in (("harmful-forced-t-1", "forced"), ("harmful-route-t-1", "route")):
        path = (tmp_path / judge_partial.run_path(run)).with_suffix(".meta.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta(mode)), encoding="utf-8")

    assert judge_partial._run_prompts(tmp_path, "harmful-forced-t-1") == "partial@v4"
    assert judge_partial._run_prompts(tmp_path, "harmful-route-t-1") == "grounded@v4 · partial@v4"
    assert judge_partial._run_prompts(tmp_path, "harmful-route-t-2") == ""  # 메타가 없다


def test_run_settings_write_the_meta_run_in_the_recorded_key_order():
    run = {
        "started_at": "2026-10-31T00:00:00+00:00",
        "mode": "route",
        "limit": None,
        "gate": True,
        "thresholds": {"high": 0.824914, "low": 0.062973},
        "profiles": {"partial": {"spec_id": "ollama:fake", "system": ["base@v4"], "user": "q@v1"}},
        "inputs": {"set": "data/eval/harmful-v1.jsonl"},
    }

    written = judge_partial.RunSettings.from_json(run).to_json()

    assert json.dumps(written) == json.dumps(run)


# --- judge --tag (꼬리표 기록 판정) ---

JUDGE_REPLY = json.dumps({"reason": "r", **dict.fromkeys(JUDGE_ITEMS, False)})


def _write_records(path: Path, records: list[AnswerRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record.to_json(), ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _fake_ollama(monkeypatch) -> FakeLLM:
    """judge 가 여는 Ollama 접속을 가짜 LLM 으로 — 받은 요청은 FakeLLM.calls 에 남는다."""
    fake = FakeLLM([JUDGE_REPLY])

    class FakeOllama:
        @classmethod
        @contextlib.contextmanager
        def connect(cls, config):
            del config  # 접속 설정은 쓰지 않는다
            yield fake

    monkeypatch.setattr(judge_partial, "OllamaLLM", FakeOllama)
    return fake


def _offwiki(item_id: str, **changes) -> AnswerRecord:
    return _record(item_id, source="offwiki", expected="partial", **changes)


def test_tagged_runs_name_the_same_files_as_answer_and_eval_answer():
    assert judge_partial.TAGGED_RUNS["f9"] == Path("data/eval/f9-answers-v1.jsonl")
    for set_name in ("harmful", "procedure"):
        assert judge_partial.TAGGED_RUNS[f"{set_name}-route"] == judge_partial.answer_path(
            set_name, "route", None
        )
    # 기본 judge · report 가 읽는 기록 · 판정 파일 이름은 그대로다
    assert {
        "f9": Path("data/eval/f9-answers-v1.jsonl"),
        "harmful-route": Path("data/eval/harmful-answers-v1-route.jsonl"),
        "harmful-forced": Path("data/eval/harmful-answers-v1-forced.jsonl"),
    } == judge_partial.RUNS
    assert Path("data/eval/partial-judged-v1.jsonl") == judge_partial.JUDGED_PATH
    assert Path("data/eval/harmful-v1.jsonl") == judge_partial.HARMFUL_SET
    assert judge_partial.PARTIAL_JUDGED_TAGGED_PATH != judge_partial.JUDGED_PATH


def test_judge_with_a_tag_judges_only_that_record_into_the_tagged_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake = _fake_ollama(monkeypatch)
    _write_records(
        tagged(judge_partial.F9_ANSWERS_PATH, "partial-base-1"),
        [_offwiki("ow-1"), _offwiki("ow-2", stage_path=("grounded",)), _offwiki("ow-3")],
    )
    _write_records(judge_partial.F9_ANSWERS_PATH, [_offwiki("ow-9")])  # 기본 기록 — 보지 않는다
    other = _judged("procedure-route-gate-v5-1", "hp-01").to_json()  # 다른 꼬리표 기록의 판정
    failed = replace(_judged("f9-partial-base-1", "ow-3"), verdict=None, error="boom").to_json()
    write_jsonl(judge_partial.PARTIAL_JUDGED_TAGGED_PATH, [other, failed])

    judge_partial.judge(limit=None, tag="partial-base-1", run="f9")

    rows = read_jsonl(judge_partial.PARTIAL_JUDGED_TAGGED_PATH)
    assert [(row["run"], row["id"]) for row in rows] == [
        ("procedure-route-gate-v5-1", "hp-01"),  # 남는다
        ("f9-partial-base-1", "ow-1"),
        ("f9-partial-base-1", "ow-3"),  # 실패는 지우고 다시
    ]
    assert not judge_partial.JUDGED_PATH.exists()  # 기본 판정 파일은 건드리지 않는다
    # 판정 요청은 기본 경로와 같은 함수 · 같은 지시 · 같은 설정
    record = _offwiki("ow-1")
    assert fake.calls[0].messages == judge_messages(
        record.item.question, record.answer, record.hits[: record.context_count]
    )
    assert fake.calls[0].spec == JUDGE_SPEC
    assert fake.calls[0].json_schema == SCHEMA


def test_judge_with_a_tag_skips_what_is_done_and_stops_without_a_record(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake = _fake_ollama(monkeypatch)
    path = tagged(PROCEDURE_ROUTE_ANSWERS_PATH, "gate-v5-2")
    _write_records(path, [_record("hp-01", source="procedure", category="procedure")])
    write_jsonl(
        judge_partial.PARTIAL_JUDGED_TAGGED_PATH,
        [_judged("procedure-route-gate-v5-2", "hp-01").to_json()],
    )

    judge_partial.judge(limit=None, tag="gate-v5-2", run="procedure-route")

    assert fake.calls == []
    with pytest.raises(SystemExit, match="가 없다"):
        judge_partial.judge(limit=None, tag="gate-v5-3", run="procedure-route")


def test_judge_without_a_tag_still_writes_the_base_judged_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _fake_ollama(monkeypatch)
    _write_records(judge_partial.F9_ANSWERS_PATH, [_offwiki("ow-1")])
    _write_records(tagged(judge_partial.F9_ANSWERS_PATH, "partial-base-1"), [_offwiki("ow-2")])

    judge_partial.judge(limit=None)

    rows = read_jsonl(judge_partial.JUDGED_PATH)
    assert [(row["run"], row["id"]) for row in rows] == [("f9", "ow-1")]
    assert not judge_partial.PARTIAL_JUDGED_TAGGED_PATH.exists()


def test_judge_takes_a_tag_and_a_run():
    args = judge_partial._parse_args(
        ["judge", "--tag", "partial-base-1", "--run", "procedure-route"]
    )

    assert (args.tag, args.run) == ("partial-base-1", "procedure-route")
    assert judge_partial._parse_args(["judge", "--tag", "partial-base-1"]).run is None  # 기본 f9


@pytest.mark.parametrize(
    "argv",
    [
        ["judge", "--run", "f9"],  # --run 은 --tag 와 함께
        ["answer", "--tag", "t", "--run", "f9"],
        ["judge", "--tag", "t", "--review-only"],
        ["judge", "--tag", "T"],
        ["judge", "--tag", "t", "--run", "harmful-forced"],
    ],
)
def test_judge_rejects_bad_tag_options(argv: list[str]):
    with pytest.raises(SystemExit):
        judge_partial._parse_args(argv)
