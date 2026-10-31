"""프롬프트 변형(base@v3 · base@v4)을 시간이 걸린 질문으로 비교한다 — 라벨로, 변형마다 여러 번, 문항 단위로.

같은 프롬프트도 temperature 0 · seed 고정으로 답 문장이 갈려서(2026-10-24, 37문항 중 21) 문장이 아니라 라벨을 비교한다.
단계마다 결과를 data/eval/ 에 이어 쓴다 — 멈췄다 다시 돌리면 끝난 답은 건너뛴다.
  generate     문항마다 검색은 한 번, 변형 × 회차마다 운영 설정(answer.yaml)으로 생성 (gemma4:12b 만 올린다)
               grounded 가 못 찾음이면 partial 로 (캐스케이드 흉내). 잘림은 completion_tokens 가 max_tokens 에 닿은 것
  judge        답마다 판정 모델(gpt-oss:20b 만 올린다 — 16GB 에 둘이 같이 못 올라간다)에 물음 셋
  report       라벨(playground_wiki.evaluation.answer_labels)을 문항별 k/회차 로 모아 v3 · v4 를 나란히,
               "v4 가 줄인 문항 · 늘린 문항 · 같은 문항" 수로 → reports/{날짜}-prompt-v4.md
  judge-check  판정 모델을 사람이 확인한 148답(tests/evaluation/judge-gold-v1.jsonl)에 돌려 양성을 여전히 다 잡는지
               — 판정 프롬프트 · think 를 바꿀 때마다

문항: 규칙 모르는 세트에서 관문이 놓치고 리랭커가 거절하지 않은 실시간 32 (tests/gate/gate-blind-v1.jsonl) +
이더리움 가격(덤프에 지금 값 없음) + 문서에 답이 있는 현재 상태 질문 6 (과잉 거절) + 시간과 무관한 KorQuAD 대조 5.
PG · Qdrant(docker compose up) · Ollama 가 필요하다.

사용법:
  uv run python scripts/compare_prompts.py generate --runs 5
  uv run python scripts/compare_prompts.py judge
  uv run python scripts/compare_prompts.py report
  uv run python scripts/compare_prompts.py judge-check
"""

import argparse
import asyncio
import re
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path

from playground_evalkit import render_table
from playground_llm import LLM, GenerationError, PromptLibrary
from playground_llm.ollama import OllamaConfig, OllamaLLM
from playground_retrieval.search import SearchService
from playground_retrieval.store import PgStore, QdrantStore

from playground_wiki.api.config import CONFIG_DIR, ApiConfig, load_answer_policy, load_dump_date
from playground_wiki.evaluation import JudgeAnswers, Traits, judge_answer, label_answer
from playground_wiki.evaluation.files import append_jsonl, read_jsonl
from playground_wiki.evaluation.paths import REPORT_DIR
from playground_wiki.generation import (
    NOT_FOUND,
    PROMPT_DIR,
    Decision,
    Profile,
    build_prompt,
    check_profiles,
    korean_date,
    read_marker,
)

VARIANTS = {"v3": "base@v3", "v4": "base@v4"}
ANSWERS_PATH = Path("data/eval/prompt-compare-v1-answers.jsonl")
JUDGED_PATH = Path("data/eval/prompt-compare-v1-judged.jsonl")
CHECK_PATH = Path("data/eval/judge-check-v1.jsonl")
BLIND_SET = Path("tests/gate/gate-blind-v1.jsonl")
GOLD_PATH = Path("tests/evaluation/judge-gold-v1.jsonl")
APPEND_NEWLINE = None  # 덧붙이는 줄 끝을 플랫폼 기본(Windows CRLF)으로 써 왔다 — 기록 바이트를 그대로 두려고 유지
LABELS = (
    "date_error",
    "stale_as_current",
    "no_current_disclaimer",
    "over_refusal",
    "extra_caution",
)
LABEL_NAMES = {
    "date_error": "날짜 오류",
    "stale_as_current": "옛 값을 지금 값처럼",
    "no_current_disclaimer": "지금 모름 미표시",
    "over_refusal": "과잉 거절",
    "extra_caution": "불필요한 모름",
}
REALTIME = Traits(should_say_unknown=True, answer_in_context=False)
ANSWERABLE = Traits(should_say_unknown=False, answer_in_context=True)
# 규칙 모르는 세트에서 2026-10-24 관문이 놓치고 리랭커가 거절하지 않은 실시간 질문 (A004 '비트코인 시세'는 그 뒤 관문이 잡지만 LLM 경로를 보려고 남긴다)
REALTIME_IDS = (
    "A004", "A006", "A008", "A011", "A012", "A015", "A016", "A017", "A018", "A019", "A021", "A022",
    "A024", "A025", "A026", "A028", "A029", "A030", "A031", "A032", "A033", "A034", "A035", "A036",
    "A037", "A039", "A041", "A042", "A043", "A044", "A049", "A050",
)  # fmt: skip
# (id, 질문, 문항 속성, 대조 문항의 정답). 문서에 답이 있는지는 검색 문서를 읽어 정했다
OTHER_QUESTIONS = (
    (
        "X02",
        "이더리움 가격은?",
        REALTIME,
        None,
    ),  # 문서엔 2017~2021년 값만 — '비트코인 시세'와 같은 꼴, 관문 밖
    ("X01", "비트코인 가격 추이", ANSWERABLE, None),
    ("X03", "서울 인구는 얼마야?", ANSWERABLE, None),
    ("X04", "대한민국 대통령은 누구야?", ANSWERABLE, None),
    ("C02", "유엔 사무총장은 누구야?", ANSWERABLE, None),
    ("C03", "대한민국 최저임금은 얼마야?", ANSWERABLE, None),
    (
        "C04",
        "일본 총리는 누구야?",
        ANSWERABLE,
        None,
    ),  # 다른 문서에 "이시바 시게루, 현직 총리 (2024년–현재)" — '현재' 복사의 탐침
    # KorQuAD calibrate 표본에서 때 표현 없는 문항을 시드 0 으로 5개
    ("K1", "암호화폐 코인에 유효성을 부여하는 것은?", ANSWERABLE, "블록체인"),
    (
        "K2",
        "유아인이 소년성의 특질로 캐릭터들을 자기만의 것으로 만들었다고 말한 씨네21의 편집장은?",
        ANSWERABLE,
        "주성철",
    ),
    ("K3", "가장 오래된 노르드어 문헌은 어느 시기의 물건인가?", ANSWERABLE, "12세기 중반"),
    ("K4", "재즈 기타리스트로 일렉트릭 기타를 재즈에 도입한 사람은?", ANSWERABLE, "찰리 크리스천"),
    ("K5", "엽기적인 그녀는 박스오피스 1위를 몇 주 동안 유지했는가?", ANSWERABLE, "6주"),
)
# v4 문구를 연기 시험으로 다듬는 데 쓴 문항 — 결과를 나머지와 따로 센다
TUNED = frozenset({"X01", "X03", "X04", "C02", "C03", "C04", "A044", "A004", "X02"})
PREFIX = re.compile(r"기준 문서에 따르면")
PM_NAMES = ("다카이치", "이시바")


def main() -> None:
    args = _parse_args()
    if args.step == "generate":
        generate(runs=args.runs)
    elif args.step == "judge":
        judge(ANSWERS_PATH, JUDGED_PATH)
    elif args.step == "report":
        report()
    elif args.step == "judge-check":
        judge_check()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["generate", "judge", "report", "judge-check"])
    parser.add_argument("--runs", type=int, default=5, help="generate 만: 변형마다 몇 번 돌릴지")
    return parser.parse_args()


def questions() -> list[dict]:
    """비교 문항: id · 질문 · 문항 속성 · 대조 정답."""
    blind = {r["id"]: r for r in read_jsonl(BLIND_SET)}
    rows = [
        {"id": qid, "question": blind[qid]["question"], "traits": REALTIME, "gold": None}
        for qid in REALTIME_IDS
    ]
    rows += [
        {"id": qid, "question": q, "traits": traits, "gold": gold}
        for qid, q, traits, gold in OTHER_QUESTIONS
    ]
    return rows


def generate(*, runs: int) -> None:
    items = questions()
    done = (
        {(r["id"], r["variant"], r["run"]) for r in read_jsonl(ANSWERS_PATH)}
        if ANSWERS_PATH.exists()
        else set()
    )
    config = ApiConfig.from_env()
    policy = load_answer_policy(CONFIG_DIR)
    variants = {
        name: {
            stage: replace(p, system=(base, *p.system[1:])) for stage, p in policy.profiles.items()
        }
        for name, base in VARIANTS.items()
    }
    library = PromptLibrary.load(PROMPT_DIR)
    for profiles in variants.values():
        check_profiles(profiles.values(), library)
    today, dump_date = korean_date(date.today()), load_dump_date(CONFIG_DIR)
    todo = len(items) * len(variants) * runs - len(done)
    print(
        f"{len(items)}문항 × {len(variants)}변형 × {runs}회 — 남은 {todo} · 오늘 {today} · 덤프 {dump_date}",
        flush=True,
    )
    start = time.perf_counter()
    with (
        PgStore.connect(config.pg) as pg,
        OllamaLLM.connect(OllamaConfig(url=OllamaConfig.from_env().url, timeout_sec=180)) as llm,
    ):
        service = SearchService.build(config.search, pg, QdrantStore.from_env())
        searched = asyncio.run(_search_all(service, [item["question"] for item in items]))
        for index, (item, result) in enumerate(zip(items, searched, strict=True), start=1):
            decision = policy.decide(result.hits)
            for name, profiles in variants.items():
                for run in range(runs):
                    if (item["id"], name, run) in done:
                        continue
                    record = {
                        "id": item["id"],
                        "variant": name,
                        "run": run,
                        "today": today,
                        "stage": decision.stage,
                        "top_score": decision.top_score,
                    }
                    record |= _answer(
                        llm,
                        library,
                        profiles,
                        decision,
                        item["question"],
                        today=today,
                        dump_date=dump_date,
                    )
                    append_jsonl(ANSWERS_PATH, record, newline=APPEND_NEWLINE)
            print(
                f"  {index}/{len(items)} {item['id']} {decision.stage} {time.perf_counter() - start:.0f}s",
                flush=True,
            )
    print(f"끝 {time.perf_counter() - start:.0f}s · {ANSWERS_PATH}")


def _answer(
    llm: LLM,
    library: PromptLibrary,
    profiles: Mapping[str, Profile],
    decision: Decision,
    question: str,
    *,
    today: str,
    dump_date: str,
) -> dict:
    """한 변형의 답. refuse 면 LLM 을 부르지 않고, grounded 가 못 찾으면 partial 로 다시."""
    path, answer, truncated, tokens = [decision.stage], "", False, 0
    stages = [] if decision.stage == "refuse" else [decision.stage]
    while stages:
        stage = stages.pop(0)
        prompt = build_prompt(
            profiles[stage],
            library,
            question=question,
            context=decision.context,
            today=today,
            dump_date=dump_date,
        )
        try:
            generation = llm.generate(profiles[stage].spec, prompt.messages)
        except GenerationError as e:  # 한 답 실패로 전체를 멈추지 않는다
            return {"path": path, "answer": None, "error": str(e)}
        answer, tokens = generation.text, generation.usage.completion_tokens
        truncated = tokens >= int(profiles[stage].spec.params["max_tokens"])
        if stage != "grounded":
            continue
        # grounded@v2 는 첫 줄 표지 — /ask 와 같은 판정. v1 은 고정 문장이라 그 문장으로
        marked = read_marker(answer)
        answer = marked.rest if marked.status != "missing" else answer
        if marked.status == "not_found" or NOT_FOUND in answer or not answer.strip():
            path.append("partial")
            stages.append("partial")
    return {"path": path, "answer": answer, "truncated": truncated, "completion_tokens": tokens}


def judge(source: Path, out: Path) -> None:
    rows = [r for r in read_jsonl(source) if r.get("answer")]
    by_id = {item["id"]: item["question"] for item in questions()}
    key = lambda r: (r["id"], r["variant"], r.get("run", 0))  # noqa: E731
    done = {key(r) for r in read_jsonl(out)} if out.exists() else set()
    dump_date = load_dump_date(CONFIG_DIR)
    print(f"판정 {len(rows) - len(done)}답 남음", flush=True)
    start = time.perf_counter()
    with OllamaLLM.connect(OllamaConfig(url=OllamaConfig.from_env().url, timeout_sec=300)) as llm:
        for index, row in enumerate(rows, start=1):
            if key(row) in done:
                continue
            question = row.get("question") or by_id[row["id"]]
            today = row.get("today") or korean_date(date.today())
            try:
                verdict, reason = judge_answer(
                    llm, question, row["answer"], today=today, dump_date=dump_date
                )
            except (ValueError, GenerationError) as e:
                append_jsonl(
                    out,
                    {
                        "id": row["id"],
                        "variant": row["variant"],
                        "run": row.get("run", 0),
                        "error": str(e),
                    },
                    newline=APPEND_NEWLINE,
                )
                continue
            append_jsonl(
                out,
                {
                    "id": row["id"],
                    "variant": row["variant"],
                    "run": row.get("run", 0),
                    **verdict.__dict__,
                    "reason": reason,
                },
                newline=APPEND_NEWLINE,
            )
            if index % 20 == 0:
                print(f"  {index}/{len(rows)} {time.perf_counter() - start:.0f}s", flush=True)
    print(f"끝 {time.perf_counter() - start:.0f}s · {out}")


def judge_check() -> None:
    """판정 모델을 사람이 확인한 148답에 돌려 라벨별로 '사람 양성 중 잡은 수 · 판정만 양성'을 센다."""
    judge(GOLD_PATH, CHECK_PATH)
    gold = {(r["id"], r["variant"]): r for r in read_jsonl(GOLD_PATH)}
    today, dump = date(2026, 10, 24), _dump_day()
    tally = Counter()
    for verdict in read_jsonl(CHECK_PATH):
        if "error" in verdict:
            tally["error"] += 1
            continue
        row = gold[(verdict["id"], verdict["variant"])]
        labels = label_answer(
            row["answer"],
            Traits(**row["traits"]),
            _judge_answers(verdict),
            today=today,
            dump_date=dump,
        )
        for name in LABELS:
            want, got = row["gold"][name], getattr(labels, name)
            if want is None:
                continue
            tally[
                (name, "caught" if want and got else "missed" if want else "extra" if got else "ok")
            ] += 1
    for name in LABELS:
        positives = tally[(name, "caught")] + tally[(name, "missed")]
        print(
            f"{LABEL_NAMES[name]}: 사람 양성 {positives} 중 잡음 {tally[(name, 'caught')]} · 판정만 양성 {tally[(name, 'extra')]}"
        )
    print(f"판정 오류 {tally['error']}")


def report() -> None:
    items = {item["id"]: item for item in questions()}
    answers = {(r["id"], r["variant"], r["run"]): r for r in read_jsonl(ANSWERS_PATH)}
    verdicts = {
        (r["id"], r["variant"], r["run"]): r for r in read_jsonl(JUDGED_PATH) if "error" not in r
    }
    today, dump = _today_of(answers.values()), _dump_day()
    # (문항, 변형) → 라벨별 [회차마다 True/False], 그 밖의 열
    per: dict[tuple[str, str], dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for k, row in answers.items():
        qid, variant, _ = k
        item = items[qid]
        cell = per[(qid, variant)]
        cell["truncated"].append(bool(row.get("truncated")))
        if row.get("answer") is None:
            cell["missing"].append(True)
            continue
        if k in verdicts:
            labels = label_answer(
                row["answer"],
                item["traits"],
                _judge_answers(verdicts[k]),
                today=today,
                dump_date=dump,
            )
            for name in LABELS:
                if getattr(labels, name) is not None:
                    cell[name].append(getattr(labels, name))
        elif row["stage"] == "refuse":
            pass  # LLM 을 부르지 않은 고정 거절
        if item["gold"]:
            cell["correct"].append(
                re.sub(r"\s", "", item["gold"]) in re.sub(r"\s", "", row["answer"])
            )
            cell["prefix"].append(bool(PREFIX.search(row["answer"])))
        if qid == "C04":
            cell["pm"].append("+".join(n for n in PM_NAMES if n in row["answer"]) or "없음")
    lines = _report_lines(items, per)
    path = REPORT_DIR / f"{datetime.now().strftime('%Y-%m-%d')}-prompt-v4.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:40]))
    print(f"… 저장: {path}")


def _report_lines(items: dict, per: dict) -> list[str]:
    runs = max((len(c["truncated"]) for c in per.values()), default=0)
    lines = [
        "# 프롬프트 base@v3 · base@v4 비교 — 시간이 걸린 질문",
        "",
        f"문항 {len(items)} × 변형 2 × {runs}회, 운영 설정(answer.yaml). 라벨은 문항 속성(사람) · 판정 물음(gpt-oss:20b) · "
        "날짜 규칙에서 계산 (scripts/compare_prompts.py · playground_wiki.evaluation.answer_labels).",
        "칸은 '그 라벨이 나온 회차 수 / 회차'. 줄인 문항 = v4 가 v3 보다 적게 나온 문항 (나쁜 라벨이라 적을수록 좋다).",
        "",
        "## 라벨별 요약",
        "",
    ]
    summary = []
    for name in LABELS:
        for part, keep in (
            ("다듬기 9", lambda q: q in TUNED),
            ("나머지", lambda q: q not in TUNED),
        ):
            better = worse = same = total3 = total4 = n = 0
            for qid in items:
                if not keep(qid):
                    continue
                v3, v4 = per[(qid, "v3")].get(name, []), per[(qid, "v4")].get(name, [])
                if not v3 and not v4:
                    continue
                n += 1
                a, b = sum(v3), sum(v4)
                total3, total4 = total3 + a, total4 + b
                better, worse, same = better + (b < a), worse + (b > a), same + (b == a)
            if n:
                summary.append(
                    {
                        "라벨": LABEL_NAMES[name],
                        "문항": f"{part} {n}",
                        "v3 합": total3,
                        "v4 합": total4,
                        "v4 가 줄인 문항": better,
                        "늘린 문항": worse,
                        "같은 문항": same,
                    }
                )
    lines += [render_table(summary, _columns(summary)), "", "## 문항별", ""]
    rows = []
    for qid, item in items.items():
        row = {"id": qid + (" *" if qid in TUNED else ""), "질문": item["question"][:28]}
        for name in LABELS:
            cells = [per[(qid, v)].get(name) for v in VARIANTS]
            if any(cells):
                row[LABEL_NAMES[name]] = " → ".join(
                    f"{sum(c)}/{len(c)}" if c else "-" for c in cells
                )
        row["잘림"] = " → ".join(f"{sum(per[(qid, v)]['truncated'])}" for v in VARIANTS)
        if item["gold"]:
            row["정답 · 접두"] = " → ".join(
                f"{sum(per[(qid, v)]['correct'])}·{sum(per[(qid, v)]['prefix'])}" for v in VARIANTS
            )
        if qid == "C04":
            row["총리 이름"] = " → ".join(
                ",".join(f"{k}{c}" for k, c in Counter(per[(qid, v)]["pm"]).items())
                for v in VARIANTS
            )
        rows.append(row)
    lines += [
        render_table(rows, _columns(rows)),
        "",
        "\\* 다듬기 세트 (v4 문구를 연기 시험으로 다듬은 문항).",
    ]
    return lines


def _columns(rows: list[dict]) -> list[str]:
    """행마다 칸이 달라(문항마다 해당하는 라벨이 다르다) 처음 나온 순서대로 모은다."""
    return list(dict.fromkeys(key for row in rows for key in row))


def _judge_answers(verdict: dict) -> JudgeAnswers:
    return JudgeAnswers(verdict["gives_value"], verdict["undated_value"], verdict["says_unknown"])


def _today_of(rows: Iterable[dict]) -> date:
    label = next(iter(rows))["today"]  # "2026년 10월 24일"
    year, month, day = (int(x) for x in re.findall(r"\d+", label))
    return date(year, month, day)


def _dump_day() -> date:
    year, month, day = (int(x) for x in re.findall(r"\d+", load_dump_date(CONFIG_DIR)))
    return date(year, month, day)


async def _search_all(service: SearchService, questions_: list[str]) -> list:
    return [await service.search(question) for question in questions_]


if __name__ == "__main__":
    main()
