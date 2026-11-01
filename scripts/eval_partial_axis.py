"""partial 단계 블록 축 — 운영 partial(`[base@v4, partial@v4]`) 대 공통(`[base@v4]` 만) 준비 · 검수표 · 리포트.

think 는 축에서 뺐다 (끔 고정, 운영 설정 그대로 — reports/2026-10-24-partial-probe.md). 답 · 판정은 기존 스크립트가 한다
(judge_partial.py answer · judge, eval_answer.py run) — 공통 칸은 설정 폴더 사본을 RAG_CONFIG_DIR 로 가리켜 돌린다.
이 스크립트는 GPU 를 쓰지 않는다.

  config-dir --out DIR
            configs/ 의 파일을 DIR 에 복사하고 answer.yaml 의 partial system 만 [base@v4] 로 바꾼다 (나머지는 글자 그대로 —
            YAML 로 읽어 그 한 값 말고는 같은지 확인한다). DIR 이 이미 있으면 같은 내용인지만 보고 다르면 멈춘다
  f9-ids    F9 운영 기록(f9-answers-v1-grounded-v5-1)에서 partial 로 끝난 문항 + 표본의 위키 밖 앞 50 의 id 를 표본
            차례 쉼표 목록으로 찍는다 — eval_answer.py run --ids 에 준다
  review-sheet
            공통 칸 유해 세트 3회차(harmful-answers-v1-route-partial-base-1~3)에서 partial 로 끝난 답 전부를 섞은 사람
            검수표 → data/eval/partial-judge-review-v1-harmful-route-partial-base-123.md (+ .key.json — number · run · id ·
            stage). 운영 칸 답은 섞지 않는다 (채점 끝 — partial-judge-review-v1-harmful-route-gate-v5-123). 이미 있으면
            덮어쓰지 않는다. 채울 때는 이름에 -checked 를 붙인 사본에 적는다
  report    reports/{날짜}-partial-prompt-axis.md — 칸마다 짝 · McNemar, 끝난 단계, 지연, 두 칸 꼴. 공통 칸 검수표의
            채운 사본이 없으면 사람 채점 칸은 '채점 전'. --report-date YYYY-MM-DD · --data-root DIR 은 judge_partial
            report 와 같다 (data/ 를 DIR 아래에서 읽고 리포트는 지금 폴더의 reports/ 에). --f9-ops-tag T 는 F9
            운영 칸 기록을 f9-answers-v1-T.jsonl 로 (기본 grounded-v5-1) — 대상 문항은 그대로 grounded-v5-1 에서
            partial 로 끝난 것이고, 같은 문항을 운영 설정으로 공통 칸과 같은 날 · 같은 관문으로 다시 돌린 기록을 줄 때 쓴다

사용법:
  uv run python scripts/eval_partial_axis.py config-dir --out data/eval/partial-axis-configs
  uv run python scripts/eval_partial_axis.py f9-ids
  uv run python scripts/eval_partial_axis.py review-sheet
  uv run python scripts/eval_partial_axis.py report
  uv run python scripts/eval_partial_axis.py report --report-date 2026-11-01 --data-root ../../..
  uv run python scripts/eval_partial_axis.py report --f9-ops-tag partial-ops-1
"""

import argparse
import json
import random
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from types import MappingProxyType

import yaml

from playground_wiki.api.config import ANSWER_CONFIG, CONFIG_DIR
from playground_wiki.evaluation.answer_run import (
    TAG_PATTERN,
    AnswerRecord,
    EvalItem,
    read_records,
    tagged,
)
from playground_wiki.evaluation.files import read_jsonl
from playground_wiki.evaluation.judge_review import ReviewEntry, render_sheet
from playground_wiki.evaluation.partial_axis import (
    ARMS,
    ArmRun,
    AxisInputs,
    HumanLabels,
    RoundPair,
    is_partial,
    render_report,
)
from playground_wiki.evaluation.partial_judge import ITEM_GUIDE, JudgedRow
from playground_wiki.evaluation.partial_review import KeyEntry, checked_path, read_checked
from playground_wiki.evaluation.paths import (
    F9_ANSWERS_PATH,
    F9_SAMPLE_PATH,
    HARMFUL_ROUTE_ANSWERS_PATH,
    HARMFUL_SET_PATH,
    PARTIAL_JUDGED_TAGGED_PATH,
    PROCEDURE_ROUTE_ANSWERS_PATH,
    REPORT_DIR,
)

COMMON_TAG = "partial-base"
"""공통 칸 기록의 꼬리표 앞부분 — 회차 n 은 `partial-base-n` (judge_partial answer --tag · eval_answer run --tag)."""
BASE_TAG = "gate-v5"
"""운영 칸 유해 · 절차 기록의 꼬리표 앞부분 — 관문 뒤로 다시 잰 3회차 (reports/2026-10-31-baseline.md)."""
ROUNDS = (1, 2, 3)
F9_BASE_TAG = "grounded-v5-1"
"""F9 표본 고정 기록 — 지금 운영 구성(grounded@v5 · partial@v4)으로 잰 F9 (reports/2026-10-31-answer-eval.md).
다시 돌릴 대상(partial 로 끝난 문항)은 늘 여기서 뽑고, report 의 운영 칸 기본값도 이것이다 (--f9-ops-tag 로 바꾼다)."""
F9_COMMON_TAG = f"{COMMON_TAG}-1"
OFFWIKI_SIZE = 50
"""위키 밖 질문은 표본 차례로 앞 50 — 단발 대 에이전트 비교(eval_agent offwiki --limit 50)와 같은 문항."""
COMMON_PARTIAL_SYSTEM = ("base@v4",)
"""공통 칸의 partial system — 단계 블록 없이 공통 베이스만."""
ROUND_SETS: Mapping[str, Path] = MappingProxyType(
    {"harmful": HARMFUL_ROUTE_ANSWERS_PATH, "procedure": PROCEDURE_ROUTE_ANSWERS_PATH}
)
"""회차로 돌린 세트 → 꼬리표 붙이기 전 기록 경로 (judge_partial answer --mode route 와 같은 이름)."""
ROUND_RUN_NAMES: Mapping[str, str] = MappingProxyType(
    {"harmful": "harmful-route", "procedure": "procedure-route"}
)
"""세트 → 실행 이름 앞부분 — 판정 · 검수표 열쇠의 run 은 `{앞부분}-{꼬리표}` (judge_partial judge --tag 와 같다)."""
BASE_REVIEW_SHEET = Path("data/eval/partial-judge-review-v1-harmful-route-gate-v5-123.md")
"""운영 칸 유해 답의 사람 검수표 (채점 끝 — 채운 사본은 이름에 -checked)."""
COMMON_REVIEW_SHEET = Path(
    f"data/eval/partial-judge-review-v1-harmful-route-{COMMON_TAG}-{''.join(map(str, ROUNDS))}.md"
)
"""공통 칸 유해 답의 사람 검수표 (review-sheet 가 만든다)."""
SEED = 0
_PARTIAL_HEADER = re.compile(r"^  partial:\s*(#.*)?$")
_PROFILE_HEADER = re.compile(r"^ {0,2}\S")
"""partial 프로필 블록이 끝나는 줄 — 다음 프로필 · 그 앞 주석(들여쓰기 2) 또는 맨 위 키."""
_SYSTEM_LINE = re.compile(r"^(?P<indent>\s+)system:\s*\[.*\]\s*$")
COMMON_HEADER = (
    "# 공통 칸 사본 — partial system 만 [base@v4] (단계 블록 없음). scripts/eval_partial_axis.py config-dir 가 만들었다.\n"
    "# 나머지는 configs/answer.yaml 그대로다\n"
)


def main() -> None:
    args = _parse_args()
    if args.step == "config-dir":
        make_config_dir(CONFIG_DIR, args.out)
    elif args.step == "f9-ids":
        print(",".join(f9_rerun_ids(Path())))
    elif args.step == "review-sheet":
        review_sheet()
    elif args.step == "report":
        report(
            root=args.data_root or Path(),
            report_date=args.report_date,
            f9_ops_tag=args.f9_ops_tag or F9_BASE_TAG,
        )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["config-dir", "f9-ids", "review-sheet", "report"])
    # config-dir 만: 공통 칸 설정 폴더 (RAG_CONFIG_DIR 로 가리킨다)
    parser.add_argument("--out", type=Path)
    # report 만: 리포트 날짜(파일 이름 · 제목). 없으면 오늘
    parser.add_argument("--report-date", type=date.fromisoformat)
    # report 만: data/ 를 이 폴더 아래에서 읽는다 — 워크트리에서 기본 폴더의 기록으로 그릴 때
    parser.add_argument("--data-root", type=Path)
    # report 만: F9 운영 칸 기록의 꼬리표. 없으면 F9_BASE_TAG (같은 문항을 운영 설정으로 다시 돌렸으면 그 꼬리표)
    parser.add_argument("--f9-ops-tag")
    args = parser.parse_args(argv)
    if args.f9_ops_tag is not None and args.step != "report":
        parser.error("--f9-ops-tag 는 report 에만 쓴다")
    if args.f9_ops_tag is not None and not TAG_PATTERN.fullmatch(args.f9_ops_tag):
        parser.error(f"--f9-ops-tag 는 영문 소문자 · 숫자 · '-' 만 쓴다, got {args.f9_ops_tag!r}")
    if args.step == "config-dir" and args.out is None:
        parser.error("config-dir 는 --out 으로 만들 폴더를 준다")
    if args.step != "config-dir" and args.out is not None:
        parser.error("--out 은 config-dir 에만 쓴다")
    if args.step != "report" and (args.report_date is not None or args.data_root is not None):
        parser.error("--report-date · --data-root 는 report 에만 쓴다")
    return args


# --- 공통 칸 설정 폴더 ---


def common_answer_yaml(text: str) -> str:
    """answer.yaml 글에서 partial 프로필의 system 줄만 COMMON_PARTIAL_SYSTEM 으로 바꾸고 머리 주석을 단다.

    Raises:
      ValueError: partial 프로필의 system 줄을 하나로 찾지 못했거나, 바꾼 뒤 YAML 값이 partial system 말고도 달라졌을 때.
    """
    lines = text.splitlines(keepends=True)
    start = next((i for i, line in enumerate(lines) if _PARTIAL_HEADER.match(line)), None)
    if start is None:
        raise ValueError(f"{ANSWER_CONFIG}: profiles 아래 'partial:' 줄이 없다")
    end = next(
        (i for i in range(start + 1, len(lines)) if _PROFILE_HEADER.match(lines[i])), len(lines)
    )
    found = [
        (i, matched)
        for i in range(start + 1, end)
        if (matched := _SYSTEM_LINE.match(lines[i])) is not None
    ]
    if len(found) != 1:
        raise ValueError(
            f"{ANSWER_CONFIG}: partial 프로필의 system 줄이 {len(found)}개다 — 하나여야 한다"
        )
    index, matched = found[0]
    newline = "\n" if lines[index].endswith("\n") else ""
    lines[index] = f"{matched.group('indent')}system: [{', '.join(COMMON_PARTIAL_SYSTEM)}]{newline}"
    changed = COMMON_HEADER + "".join(lines)
    _check_only_partial_system(text, changed)
    return changed


def _check_only_partial_system(before: str, after: str) -> None:
    old = yaml.safe_load(before)
    new = yaml.safe_load(after)
    if new["profiles"]["partial"]["system"] != list(COMMON_PARTIAL_SYSTEM):
        raise ValueError(
            f"{ANSWER_CONFIG}: 바꾼 partial system 이 {COMMON_PARTIAL_SYSTEM} 가 아니다"
        )
    new["profiles"]["partial"]["system"] = old["profiles"]["partial"]["system"]
    if new != old:
        raise ValueError(f"{ANSWER_CONFIG}: partial system 말고도 값이 달라졌다")


def config_files(source: Path) -> dict[str, str]:
    """공통 칸 설정 폴더에 쓸 파일 이름 → 내용 — answer.yaml 만 바꾸고 나머지는 그대로."""
    files = {}
    for path in sorted(source.iterdir()):
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            files[path.name] = common_answer_yaml(text) if path.name == ANSWER_CONFIG else text
    if ANSWER_CONFIG not in files:
        raise ValueError(f"{source} 에 {ANSWER_CONFIG} 가 없다")
    return files


def make_config_dir(source: Path, out: Path) -> None:
    """공통 칸 설정 폴더를 만든다. 이미 있으면 같은 내용인지만 본다.

    Raises:
      SystemExit: 이미 있는 폴더의 파일이 만들 내용과 다를 때 (다른 설정일 수 있어 덮어쓰지 않는다).
    """
    files = config_files(source)
    if out.exists():
        differ = [
            name
            for name, text in files.items()
            if not (out / name).exists() or (out / name).read_text(encoding="utf-8") != text
        ]
        if differ:
            raise SystemExit(f"{out} 가 이미 있고 {differ} 가 다르다 — 지우고 다시 만든다")
        print(f"{out} 가 이미 있고 내용이 같다 ({len(files)} 파일)")
        return
    out.mkdir(parents=True)
    for name, text in files.items():
        (out / name).write_text(text, encoding="utf-8", newline="\n")
    print(
        f"공통 칸 설정 폴더: {out} ({len(files)} 파일) — partial system {list(COMMON_PARTIAL_SYSTEM)}\n"
        f"답을 받을 때: RAG_CONFIG_DIR={out} 을 붙여 judge_partial.py answer · eval_answer.py run 을 돌린다"
    )


# --- F9 다시 돌릴 문항 ---


def f9_rerun_ids(root: Path) -> list[str]:
    """F9 운영 기록에서 partial 로 끝난 문항 + 위키 밖 앞 OFFWIKI_SIZE — 표본 차례.

    Raises:
      SystemExit: 운영 기록이 없을 때.
    """
    base_path = root / tagged(F9_ANSWERS_PATH, F9_BASE_TAG)
    if not base_path.exists():
        raise SystemExit(f"{base_path} 가 없다 — F9 운영 기록을 찾지 못했다")
    partial = {record.item.id for record in read_records(base_path) if is_partial(record)}
    items = _sample_items(root)
    targets = partial | set(offwiki_ids(items))
    return [item.id for item in items if item.id in targets]


def offwiki_ids(items: Sequence[EvalItem]) -> list[str]:
    """표본 차례로 위키 밖 문항 앞 OFFWIKI_SIZE 개의 id."""
    return [item.id for item in items if item.source == "offwiki"][:OFFWIKI_SIZE]


def _sample_items(root: Path) -> list[EvalItem]:
    return [EvalItem.from_json(row) for row in read_jsonl(root / F9_SAMPLE_PATH)]


# --- 검수표 ---


def round_tag(prefix: str, round_number: int) -> str:
    """회차 기록의 꼬리표 (`partial-base` · 2 → `partial-base-2`)."""
    return f"{prefix}-{round_number}"


def review_sheet(root: Path = Path()) -> None:
    """공통 칸 유해 세트 3회차에서 partial 로 끝난 답 전부를 섞은 검수표와 열쇠를 쓴다.

    Raises:
      SystemExit: 검수표가 이미 있을 때 (채우던 것일 수 있다), 회차 기록에 세트 문항이 다 없을 때, partial 답이 없을 때.
    """
    path = root / COMMON_REVIEW_SHEET
    if path.exists():
        raise SystemExit(
            f"{path} 가 이미 있다 — 채우던 검수표일 수 있어 덮어쓰지 않는다. 다시 만들려면 직접 지운다"
        )
    expected = {str(row["id"]) for row in read_jsonl(root / HARMFUL_SET_PATH)}
    picks: list[tuple[str, AnswerRecord]] = []
    for round_number in ROUNDS:
        tag = round_tag(COMMON_TAG, round_number)
        records_path = root / tagged(HARMFUL_ROUTE_ANSWERS_PATH, tag)
        records = read_records(records_path) if records_path.exists() else []
        if {record.item.id for record in records} != expected:
            raise SystemExit(f"{records_path} 에 세트 문항이 다 없다 — answer 를 먼저 다 돌린다")
        run = f"{ROUND_RUN_NAMES['harmful']}-{tag}"
        picks += [(run, record) for record in records if is_partial(record)]
    if not picks:
        raise SystemExit("공통 칸 회차 기록에 partial 답이 없다")
    random.Random(SEED).shuffle(picks)
    entries = [
        ReviewEntry(
            number=number,
            question=record.item.question,
            context=record.hits[: record.context_count],
            answer=record.answer,
        )
        for number, (_, record) in enumerate(picks, start=1)
    ]
    part = COMMON_REVIEW_SHEET.stem.removeprefix("partial-judge-review-v1-")
    text = render_sheet(f"partial 판정 검수표 v1 — {part} {len(entries)}문항", entries, ITEM_GUIDE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    key = {
        "created_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "part": part,
        "seed": SEED,
        "entries": [
            KeyEntry(number, run, record.item.id).to_json() | {"stage": "partial"}
            for number, (run, record) in enumerate(picks, start=1)
        ],
    }
    key_path = path.with_suffix(".key.json")
    key_path.write_text(json.dumps(key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    per_run = " · ".join(
        f"{run} {count}" for run, count in sorted(Counter(run for run, _ in picks).items())
    )
    print(
        f"검수표 {len(entries)}문항 ({per_run}): {path}\n열쇠(문항 번호 → 기록): {key_path}\n"
        f"채울 때는 {checked_path(path).name} 으로 복사해 그 사본에 적는다 — 이 파일은 틀로 남는다"
    )


# --- 리포트 ---


def report(*, root: Path, report_date: date | None = None, f9_ops_tag: str = F9_BASE_TAG) -> None:
    """기록 · 판정 · 검수표를 읽어 리포트를 쓴다 (리포트는 지금 폴더의 reports/)."""
    today = report_date or datetime.now(tz=UTC).date()
    inputs = load_inputs(root, f9_ops_tag=f9_ops_tag)
    text = render_report(inputs, today=today)
    REPORT_DIR.mkdir(exist_ok=True)
    path = REPORT_DIR / f"{today:%Y-%m-%d}-partial-prompt-axis.md"
    path.write_text(text, encoding="utf-8", newline="\n")  # .gitattributes 가 LF
    common_rounds = sum(bool(each.common.records) for each in inputs.rounds)
    print(
        f"회차 {len(inputs.rounds)} (공통 칸 기록 있음 {common_rounds}) · F9 공통 {len(inputs.f9_common.records)} · "
        f"판정 {len(inputs.judged)} · 사람 채점 운영 {'있음' if inputs.human_base else '없음'} · 공통 "
        f"{'있음' if inputs.human_common else '채점 전'}\n리포트: {path}"
    )


def load_inputs(root: Path, *, f9_ops_tag: str = F9_BASE_TAG) -> AxisInputs:
    """data/ (root 아래)에서 리포트 입력을 읽는다. 없는 기록은 빈 기록으로 둔다.

    F9 대상 문항은 늘 F9_BASE_TAG 기록에서 partial 로 끝난 것이다 (표본 고정). f9_ops_tag 는 운영 칸 기록만 고른다.
    """
    rounds: list[RoundPair] = []
    for set_name in ROUND_SETS:
        rounds.extend(
            RoundPair(
                set_name=set_name,
                round_number=round_number,
                base=_round_run(root, set_name, round_tag(BASE_TAG, round_number)),
                common=_round_run(root, set_name, round_tag(COMMON_TAG, round_number)),
            )
            for round_number in ROUNDS
        )
    items = _sample_items(root)
    fixed = _arm_run(root, tagged(F9_ANSWERS_PATH, F9_BASE_TAG), f"f9-{F9_BASE_TAG}")
    f9_base = (
        fixed
        if f9_ops_tag == F9_BASE_TAG
        else _arm_run(root, tagged(F9_ANSWERS_PATH, f9_ops_tag), f"f9-{f9_ops_tag}")
    )
    harmful_rows = read_jsonl(root / HARMFUL_SET_PATH) if (root / HARMFUL_SET_PATH).exists() else []
    return AxisInputs(
        rounds=tuple(rounds),
        f9_base=f9_base,
        f9_common=_arm_run(root, tagged(F9_ANSWERS_PATH, F9_COMMON_TAG), f"f9-{F9_COMMON_TAG}"),
        f9_ids=tuple(record.item.id for record in fixed.records if is_partial(record)),
        f9_ids_run=None if f9_ops_tag == F9_BASE_TAG else fixed.run,
        offwiki_ids=tuple(offwiki_ids(items)),
        judged=MappingProxyType(_tagged_judged(root)),
        human_base=_human(root, BASE_REVIEW_SHEET),
        human_common=_human(root, COMMON_REVIEW_SHEET),
        kinds=MappingProxyType({str(row["id"]): str(row["kind"]) for row in harmful_rows}),
        sheets=MappingProxyType(
            {ARMS[0]: BASE_REVIEW_SHEET.name, ARMS[1]: COMMON_REVIEW_SHEET.name}
        ),
    )


def _round_run(root: Path, set_name: str, tag: str) -> ArmRun:
    return _arm_run(root, tagged(ROUND_SETS[set_name], tag), f"{ROUND_RUN_NAMES[set_name]}-{tag}")


def _arm_run(root: Path, path: Path, run: str) -> ArmRun:
    full = root / path
    records = tuple(read_records(full)) if full.exists() else ()
    return ArmRun(run=run, records=records, profiles=MappingProxyType(_meta_profiles(full)))


def _meta_profiles(path: Path) -> dict[str, str]:
    """기록 meta 마지막 실행의 단계 → `system 목록 · spec_id`. judge_partial(runs[-1].profiles) · eval_answer
    (runs[-1].settings.profiles) 두 꼴을 다 읽는다. meta 가 없으면 빈 dict."""
    meta_path = path.with_suffix(".meta.json")
    if not meta_path.exists():
        return {}
    last = json.loads(meta_path.read_text(encoding="utf-8"))["runs"][-1]
    profiles = last.get("settings", last)["profiles"]
    return {
        stage: f"{'+'.join(profile['system'])} · {profile['spec_id']}"
        for stage, profile in sorted(profiles.items())
    }


def _tagged_judged(root: Path) -> dict[tuple[str, str], JudgedRow]:
    path = root / PARTIAL_JUDGED_TAGGED_PATH
    rows = [JudgedRow.from_json(row) for row in read_jsonl(path)] if path.exists() else []
    return {(row.run, row.id): row for row in rows}


def _human(root: Path, sheet: Path) -> HumanLabels | None:
    """채운 검수표의 사람 채점 — (실행, 문항 id) → 항목 → 값. 채운 사본이 없으면 None."""
    read = read_checked(root, sheet)
    if read is None:
        return None
    ratings, entries = read
    return MappingProxyType(
        {entry.answer: ratings[entry.number] for entry in entries if entry.number in ratings}
    )


if __name__ == "__main__":
    main()
