"""판정 모델 확인: 사람이 확인한 148답에 판정 모델을 돌려 양성을 여전히 다 잡는지 센다.

판정 프롬프트(playground_wiki.evaluation.answer_judge) · think 를 바꿀 때마다 돌린다.
- 답: tests/evaluation/judge-gold-v1.jsonl (2026-10-24 프롬프트 v2 · v3 변형의 답, 라벨은 사람이 확인)
- 답마다 판정 모델(gpt-oss:20b)에 물음 셋을 묻고 data/eval/judge-check-v1.jsonl 에 이어 쓴다.
  끝난 답은 건너뛰므로 판정 프롬프트를 바꿨으면 그 파일을 지우고 돌린다
- 라벨(playground_wiki.evaluation.answer_labels)을 계산해 라벨마다 '사람 양성 중 잡은 수 · 판정만 양성'을 출력한다
Ollama 가 필요하다 (PG · Qdrant 는 쓰지 않는다).

사용법:
  uv run python scripts/judge_check.py
"""

import argparse
import re
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

from playground_llm import LLM, GenerationError
from playground_llm.ollama import OllamaConfig, OllamaLLM

from playground_wiki.api.config import CONFIG_DIR, load_dump_date
from playground_wiki.evaluation import JudgeAnswers, Traits, judge_answer, label_answer
from playground_wiki.evaluation.files import append_jsonl, read_jsonl
from playground_wiki.generation import korean_date

GOLD_PATH = Path("tests/evaluation/judge-gold-v1.jsonl")
CHECK_PATH = Path("data/eval/judge-check-v1.jsonl")
GOLD_DAY = date(2026, 10, 24)  # 사람이 라벨을 확인한 날 — 날짜 오류 규칙의 '오늘'
TIMEOUT_SEC = 300  # 판정 모델 한 번 호출 (think medium)
APPEND_NEWLINE = None  # 덧붙이는 줄 끝을 플랫폼 기본(Windows CRLF)으로 써 왔다 — 기록 바이트를 그대로 두려고 유지
LABEL_NAMES = {
    "date_error": "날짜 오류",
    "stale_as_current": "옛 값을 지금 값처럼",
    "no_current_disclaimer": "지금 모름 미표시",
    "over_refusal": "과잉 거절",
    "extra_caution": "불필요한 모름",
}


def main() -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    dump_date = load_dump_date(CONFIG_DIR)
    config = OllamaConfig(url=OllamaConfig.from_env().url, timeout_sec=TIMEOUT_SEC)
    with OllamaLLM.connect(config) as llm:
        judge_gold(llm, GOLD_PATH, CHECK_PATH, dump_date=dump_date)
    tally = tally_labels(
        read_jsonl(GOLD_PATH), read_jsonl(CHECK_PATH), today=GOLD_DAY, dump_day=_day(dump_date)
    )
    for name, title in LABEL_NAMES.items():
        positives = tally[(name, "caught")] + tally[(name, "missed")]
        print(
            f"{title}: 사람 양성 {positives} 중 잡음 {tally[(name, 'caught')]} · 판정만 양성 {tally[(name, 'extra')]}"
        )
    print(f"판정 오류 {tally['error']}")


def judge_gold(llm: LLM, source: Path, out: Path, *, dump_date: str) -> None:
    """source 의 답마다 판정 모델에 물어 out 에 한 줄씩 덧붙인다. out 에 이미 있는 (id, variant) 는 건너뛴다.

    판정에 실패한 답은 error 칸만 남기고 다음 답으로 간다.
    """
    rows = [r for r in read_jsonl(source) if r.get("answer")]
    done = {_key(r) for r in read_jsonl(out)} if out.exists() else set()
    print(f"판정 {len(rows) - len(done)}답 남음", flush=True)
    start = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        if _key(row) in done:
            continue
        today = row.get("today") or korean_date(date.today())
        head = {"id": row["id"], "variant": row["variant"], "run": row.get("run", 0)}
        try:
            verdict, reason = judge_answer(
                llm, row["question"], row["answer"], today=today, dump_date=dump_date
            )
        except (ValueError, GenerationError) as e:
            append_jsonl(out, {**head, "error": str(e)}, newline=APPEND_NEWLINE)
            continue
        append_jsonl(out, {**head, **asdict(verdict), "reason": reason}, newline=APPEND_NEWLINE)
        if index % 20 == 0:
            print(f"  {index}/{len(rows)} {time.perf_counter() - start:.0f}s", flush=True)
    print(f"끝 {time.perf_counter() - start:.0f}s · {out}")


def tally_labels(
    gold: Sequence[Mapping[str, Any]],
    verdicts: Sequence[Mapping[str, Any]],
    *,
    today: date,
    dump_day: date,
) -> Counter[object]:
    """판정마다 라벨을 계산해 사람 라벨과 맞춰 센다.

    Returns:
      (라벨, caught · missed · extra · ok) 별 답 수와 "error"(판정 실패) 수. 사람 라벨이 None 인 칸은 세지 않는다.
    """
    by_key = {(r["id"], r["variant"]): r for r in gold}
    tally: Counter[object] = Counter()
    for verdict in verdicts:
        if "error" in verdict:
            tally["error"] += 1
            continue
        row = by_key[(verdict["id"], verdict["variant"])]
        labels = asdict(
            label_answer(
                row["answer"],
                Traits(**row["traits"]),
                JudgeAnswers(
                    verdict["gives_value"], verdict["undated_value"], verdict["says_unknown"]
                ),
                today=today,
                dump_date=dump_day,
            )
        )
        for name in LABEL_NAMES:
            want, got = row["gold"][name], labels[name]
            if want is None:
                continue
            tally[(name, _outcome(want=want, got=got))] += 1
    return tally


def _outcome(*, want: bool, got: bool | None) -> str:
    if want:
        return "caught" if got else "missed"
    return "extra" if got else "ok"


def _key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    return (row["id"], row["variant"], row.get("run", 0))


def _day(label: str) -> date:
    """'2026년 10월 1일' 꼴을 date 로."""
    year, month, day = (int(x) for x in re.findall(r"\d+", label))
    return date(year, month, day)


if __name__ == "__main__":
    main()
