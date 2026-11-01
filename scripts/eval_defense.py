"""간접 인젝션 방어 칸 준비 — 운영 base 만 바꾼 설정 폴더 사본 · 부작용 측정의 유해 세트 검수표.

칸(evaluation.defense_arms)은 base-v5(문서 규칙을 넓힌 후보 base@v5)와 base-norule(규칙 줄을 지운 측정 전용
base_norule@v4) 둘이다. 답은 기존 스크립트가 받는다 — 칸의 설정 폴더를 RAG_CONFIG_DIR 로 가리키고 꼬리표를 준다
(eval_injection.py run --tag · eval_answer.py run --tag · eval_f4_prose.py run --tag · judge_partial.py answer --tag).
비교 리포트는 eval_injection.py compare. 이 스크립트는 GPU · DB 를 쓰지 않는다.

  config-dir [--arm A]
            configs/ 의 파일을 data/eval/defense-configs/{칸}/ 에 복사하고 answer.yaml 의 grounded · partial system 에서
            base@v4 만 칸의 조각으로 바꾼다 (나머지는 글자 그대로 — YAML 로 읽어 그 값 말고는 같은지 확인한다).
            --arm 이 없으면 칸 둘 다. 폴더가 이미 있으면 같은 내용인지만 보고 다르면 멈춘다
  review-sheet --tag T
            유해 세트 실제 경로(route) 꼬리표 기록(data/eval/harmful-answers-v1-route-T.jsonl, judge_partial.py answer
            --mode route --tag T)에서 partial 로 끝난 답 전부 + grounded 로 끝난 유해 질문의 답을 섞은 사람 검수표 →
            data/eval/partial-judge-review-v1-harmful-route-T.md (+ .key.json — number · run · id · stage). 지난 관문 뒤
            3회차 검수표(gate-v5-123)와 같은 꼴이고, 기준 칸의 답은 섞지 않는다 (그 칸은 채점이 끝났다). 이미 있으면
            덮어쓰지 않는다. 채울 때는 이름에 -checked 를 붙인 사본에 적는다

사용법:
  uv run python scripts/eval_defense.py config-dir
  uv run python scripts/eval_defense.py config-dir --arm base-v5
  uv run python scripts/eval_defense.py review-sheet --tag base-v5-1
"""

import argparse
import json
import random
import re
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import yaml

from playground_wiki.api.config import ANSWER_CONFIG, CONFIG_DIR
from playground_wiki.evaluation.answer_metrics import final_stage
from playground_wiki.evaluation.answer_run import TAG_PATTERN, AnswerRecord, read_records, tagged
from playground_wiki.evaluation.defense_arms import (
    ARM_NAMES,
    ARMS,
    DEFENSE_CONFIG_ROOT,
    SERVICE_BASE,
    DefenseArm,
    swap_base,
)
from playground_wiki.evaluation.judge_review import ReviewEntry, render_sheet
from playground_wiki.evaluation.partial_judge import ITEM_GUIDE
from playground_wiki.evaluation.partial_review import checked_path

HARMFUL_ROUTE_ANSWERS_PATH = Path("data/eval/harmful-answers-v1-route.jsonl")
"""유해 세트 실제 경로 기록 — judge_partial.py answer --mode route 와 같은 이름 (꼬리표는 끝에 -T)."""
HARMFUL_ROUTE_REVIEW_PATH = Path("data/eval/partial-judge-review-v1-harmful-route.md")
"""유해 세트 실제 경로 검수표 — judge_partial.py review-sheet --part harmful-route 와 같은 이름 (꼬리표는 끝에 -T)."""
REVIEW_RUN = "harmful-route"
"""검수표 열쇠의 실행 이름 앞부분 — `{앞부분}-{꼬리표}` (judge_partial 의 실행 이름과 같다)."""
HARMFUL_KIND = "harmful"
"""유해 세트 문항의 category 중 유해 질문 (짝은 contrast)."""
SEED = 0
_SYSTEM_LINE = re.compile(r"^(?P<indent>\s+)system:\s*\[(?P<refs>[^\]]*)\]\s*$")
_HEADER = (
    "# 방어 비교 칸 사본 — grounded · partial system 의 {service} 만 {base} 로 ({arm}: {summary}).\n"
    "# scripts/eval_defense.py config-dir 가 만들었다. 나머지는 configs/answer.yaml 그대로다\n"
)


def main() -> None:
    args = _parse_args()
    if args.step == "config-dir":
        for name in [args.arm] if args.arm else ARM_NAMES:
            make_config_dir(CONFIG_DIR, DEFENSE_CONFIG_ROOT / name, ARMS[name])
    elif args.step == "review-sheet":
        review_sheet(args.tag)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["config-dir", "review-sheet"])
    # config-dir 만: 만들 칸 하나. 없으면 칸 전부
    parser.add_argument("--arm", choices=ARM_NAMES)
    # review-sheet 만: 유해 세트 실제 경로 꼬리표 기록 (예 base-v5-1)
    parser.add_argument("--tag")
    args = parser.parse_args(argv)
    if args.arm is not None and args.step != "config-dir":
        parser.error("--arm 은 config-dir 에만 쓴다")
    if args.step == "review-sheet" and args.tag is None:
        parser.error("review-sheet 는 --tag 로 검수할 기록을 준다")
    if args.tag is not None and args.step != "review-sheet":
        parser.error("--tag 는 review-sheet 에만 쓴다")
    if args.tag is not None and not TAG_PATTERN.fullmatch(args.tag):
        parser.error(f"--tag 는 영문 소문자 · 숫자 · '-' 만 쓴다, got {args.tag!r}")
    return args


# --- 칸 설정 폴더 ---


def arm_answer_yaml(text: str, arm: DefenseArm) -> str:
    """answer.yaml 글에서 프로필마다 system 줄의 운영 base 만 칸의 조각으로 바꾸고 머리 주석을 단다.

    Raises:
      ValueError: system 줄 수가 프로필 수와 다르거나, 어느 system 에 운영 base 가 하나가 아니거나,
        바꾼 뒤 YAML 값이 system 의 base 말고도 달라졌을 때.
    """
    lines = text.splitlines(keepends=True)
    found = 0
    for index, line in enumerate(lines):
        matched = _SYSTEM_LINE.match(line)
        if matched is None:
            continue
        refs = [ref.strip() for ref in matched.group("refs").split(",") if ref.strip()]
        newline = "\n" if line.endswith("\n") else ""
        swapped = ", ".join(swap_base(refs, arm.base))
        lines[index] = f"{matched.group('indent')}system: [{swapped}]{newline}"
        found += 1
    profiles = yaml.safe_load(text)["profiles"]
    if found != len(profiles):
        raise ValueError(
            f"{ANSWER_CONFIG}: system 줄 {found}개 · 프로필 {len(profiles)}개 — 프로필마다 한 줄이어야 한다"
        )
    header = _HEADER.format(service=SERVICE_BASE, base=arm.base, arm=arm.name, summary=arm.summary)
    changed = header + "".join(lines)
    _check_only_base(text, changed, arm)
    return changed


def _check_only_base(before: str, after: str, arm: DefenseArm) -> None:
    old = yaml.safe_load(before)
    new = yaml.safe_load(after)
    for stage, profile in old["profiles"].items():
        if tuple(new["profiles"][stage]["system"]) != swap_base(profile["system"], arm.base):
            raise ValueError(f"{ANSWER_CONFIG}: {stage} system 이 base 만 바뀐 꼴이 아니다")
        new["profiles"][stage]["system"] = profile["system"]
    if new != old:
        raise ValueError(f"{ANSWER_CONFIG}: system 의 base 말고도 값이 달라졌다")


def config_files(source: Path, arm: DefenseArm) -> dict[str, str]:
    """칸 설정 폴더에 쓸 파일 이름 → 내용 — answer.yaml 만 바꾸고 나머지는 그대로.

    Raises:
      ValueError: source 에 answer.yaml 이 없거나 바꿀 수 없을 때 (`arm_answer_yaml`).
    """
    files = {}
    for path in sorted(source.iterdir()):
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            files[path.name] = arm_answer_yaml(text, arm) if path.name == ANSWER_CONFIG else text
    if ANSWER_CONFIG not in files:
        raise ValueError(f"{source} 에 {ANSWER_CONFIG} 가 없다")
    return files


def make_config_dir(source: Path, out: Path, arm: DefenseArm) -> None:
    """칸 설정 폴더를 만든다. 이미 있으면 같은 내용인지만 본다.

    Raises:
      SystemExit: 이미 있는 폴더의 파일이 만들 내용과 다를 때 (다른 설정일 수 있어 덮어쓰지 않는다).
    """
    files = config_files(source, arm)
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
        f"칸 {arm.name} 설정 폴더: {out} ({len(files)} 파일) — system 의 {SERVICE_BASE} → {arm.base}\n"
        f"답을 받을 때: RAG_CONFIG_DIR={out.as_posix()} 를 붙이고 꼬리표를 준다"
    )


# --- 유해 세트 검수표 ---


def review_picks(records: Sequence[AnswerRecord]) -> list[AnswerRecord]:
    """검수표에 실을 답 — partial 로 끝난 답 전부 + grounded 로 끝난 유해 질문의 답, seed 로 섞은 차례.

    지난 관문 뒤 검수표(회차마다 partial 로 끝난 답 + grounded 로 끝난 유해 질문)와 같은 규칙이다. 관문 · refuse
    로 끝난 답은 LLM 글이 아니라 싣지 않는다.
    """
    picked = [record for record in sorted(records, key=lambda r: r.item.id) if _reviewed(record)]
    random.Random(SEED).shuffle(picked)
    return picked


def _reviewed(record: AnswerRecord) -> bool:
    stage = final_stage(record)
    return stage == "partial" or (stage == "grounded" and record.item.category == HARMFUL_KIND)


def review_sheet(tag: str) -> None:
    source = tagged(HARMFUL_ROUTE_ANSWERS_PATH, tag)
    path = tagged(HARMFUL_ROUTE_REVIEW_PATH, tag)
    if path.exists():
        raise SystemExit(
            f"{path} 가 이미 있다 — 채우던 검수표일 수 있어 덮어쓰지 않는다. 다시 만들려면 직접 지운다"
        )
    if not source.exists():
        raise SystemExit(
            f"{source} 가 없다 — judge_partial.py answer --mode route --tag {tag} 먼저"
        )
    picks = review_picks(read_records(source))
    if not picks:
        raise SystemExit(f"{source} 에 partial · grounded(유해 질문)로 끝난 답이 없다")
    run = f"{REVIEW_RUN}-{tag}"
    entries = [
        ReviewEntry(
            number=number,
            question=record.item.question,
            context=record.hits[: record.context_count],
            answer=record.answer,
        )
        for number, record in enumerate(picks, start=1)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    text = render_sheet(f"partial 판정 검수표 v1 — {run} {len(entries)}문항", entries, ITEM_GUIDE)
    path.write_text(text, encoding="utf-8", newline="\n")
    key_path = path.with_suffix(".key.json")
    key_path.write_text(
        json.dumps(review_key(picks, run), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"검수표 {len(entries)}문항 (단계 {_stage_counts(picks)}): {path}\n열쇠: {key_path}\n"
        f"채울 때는 {checked_path(path).name} 으로 복사해 그 사본에 적는다 — 이 파일은 틀로 남는다"
    )


def review_key(picks: Sequence[AnswerRecord], run: str) -> dict[str, object]:
    """검수표 열쇠 — 지난 관문 뒤 검수표와 같은 꼴 (part · seed · entries 의 number · run · id · stage)."""
    return {
        "created_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "part": run,
        "seed": SEED,
        "entries": [
            {"number": number, "run": run, "id": record.item.id, "stage": final_stage(record)}
            for number, record in enumerate(picks, start=1)
        ],
    }


def _stage_counts(picks: Sequence[AnswerRecord]) -> str:
    counts = Counter(final_stage(record) for record in picks)
    return " · ".join(f"{stage} {count}" for stage, count in sorted(counts.items()))


if __name__ == "__main__":
    main()
