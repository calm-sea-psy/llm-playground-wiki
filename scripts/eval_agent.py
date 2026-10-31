"""단발 대 에이전트 비교 측정: 같은 세트를 /ask 단발 · 에이전트 두 모드로 돌려 기록하고 리포트를 쓴다.

  # 기록 (GPU — Ollama · 임베딩 · 리랭커). 이어 쓰기: 이미 기록한 문항은 건너뛴다
  uv run python scripts/eval_agent.py run --set compound --mode single
  uv run python scripts/eval_agent.py run --set compound --mode agent
  uv run python scripts/eval_agent.py run --set f9 --mode agent --limit 100
  # 리포트 (GPU 없이) → reports/{날짜}-agent.md
  uv run python scripts/eval_agent.py report
  # 에이전트만 꼬리표로 다시 잰 것 — 단발(꼬리표 없음) 대 에이전트 v2 → reports/{날짜}-agent-v2.md
  uv run python scripts/eval_agent.py run --set compound --mode agent --tag v2
  uv run python scripts/eval_agent.py report --tag v2

안전 회귀 (Phase 3) — 유해 · 절차는 에이전트만 회차 꼬리표로 3 회차, 단발은 judge_partial 의 사람 채점된 기록
(data/eval/{세트}-answers-v1-route-gate-v5-{1,2,3}.jsonl)을 같은 회차 번호로 읽는다. F4 는 두 모드를 저장 문맥으로 한 회차.
  uv run python scripts/eval_agent.py run --set harmful --mode agent --tag safety-1      (-2 · -3 도)
  uv run python scripts/eval_agent.py run --set procedure --mode agent --tag safety-1    (-2 · -3 도)
  uv run python scripts/eval_agent.py run --set f4 --mode single --tag safety-1 --bench-data C:/Work/llm-playground-llm/bench/data
  uv run python scripts/eval_agent.py run --set f4 --mode agent --tag safety-1 --bench-data C:/Work/llm-playground-llm/bench/data
  # 에이전트 유해 답 사람 검수표 (GPU 없이) → data/eval/partial-judge-review-v1-harmful-agent-safety-123.md (+ .key.json)
  uv run python scripts/eval_agent.py review-sheet --set harmful --tag safety
  # F4 '답하지 않음' 판정 입력 (GPU 없이) → data/eval/agent/f4-refusal-judge-{input,key}-safety-1.jsonl
  uv run python scripts/eval_agent.py judge-input --tag safety --bench-data …
  # 판정자 출력(judge_id · refused · reason JSONL)을 열쇠로 풀어 → data/eval/agent/f4-refusal-verdicts-safety-1.jsonl
  uv run python scripts/eval_agent.py judge-verdicts --tag safety --judged 판정자-출력.jsonl
  # 안전 회귀 리포트 (GPU 없이) → reports/{날짜}-agent-safety.md
  uv run python scripts/eval_agent.py safety-report --tag safety --bench-data …

세트: compound(복합 50, tests/evaluation/compound-v1.jsonl) · f9(F9 표본의 KorQuAD 370) · offwiki(F9 표본의 위키 밖 100) ·
harmful(유해 정보 20 + 대조 20) · procedure(무해 절차 20) · f4(F4 저장 문맥 780 — 답 있음은 민감 주제 쌍둥이를 뺀 대조군 250).
기록은 data/eval/agent/{세트}-{모드}[-{꼬리표}].jsonl 과 옆 .meta.json (실행마다 설정). 리포트에는 집계 · 문항 id 만.
"""

import argparse
import asyncio
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path

from kiwipiepy import Kiwi
from playground_llm import PromptLibrary
from playground_llm.ollama import OllamaConfig, OllamaLLM
from playground_retrieval.search import KiwiAnalyzer, SearchService
from playground_retrieval.store import PgStore, QdrantStore

from playground_wiki.agent import AGENT_PROMPT_DIR
from playground_wiki.api.answer import AnswerService
from playground_wiki.api.config import AnswerSetup, ApiConfig
from playground_wiki.evaluation.agent_eval import (
    COMPARE_SETS,
    MODES,
    SAFETY_SETS,
    SET_NAMES,
    AgentEvalRun,
    AgentItem,
    AgentRecord,
    Mode,
    Pairing,
    RecordingSearch,
    SetName,
    compound_items,
    f9_items,
    read_records,
    render_report,
)
from playground_wiki.evaluation.agent_safety import (
    F4_KINDS_BY_SET,
    F4_ROUND,
    REFUSAL_JUDGE_KINDS,
    REVIEW_SEED,
    ROUNDS,
    SAFETY_SET_PATHS,
    SINGLE_REVIEW_SHEET,
    StoredContextSearch,
    agent_review_sheet,
    agent_tag,
    f4_agent_item,
    f4_labels,
    f4_safety_items,
    from_answer_record,
    refusal_judge_rows,
    refusal_verdicts,
    review_entries,
    review_key_entries,
    review_picks,
    safety_items,
    single_run,
    single_safety_path,
)
from playground_wiki.evaluation.agent_safety_report import (
    HumanLabels,
    RoundPairing,
    SafetyInputs,
    render_safety_report,
)
from playground_wiki.evaluation.answer_run import TAG_PATTERN, read_records as read_answer_records
from playground_wiki.evaluation.f4_prose import F4_KINDS, F4Item, F4Set, load_f4_items
from playground_wiki.evaluation.files import read_jsonl, write_jsonl
from playground_wiki.evaluation.judge_review import render_sheet
from playground_wiki.evaluation.partial_judge import ITEM_GUIDE
from playground_wiki.evaluation.partial_review import checked_path, read_checked
from playground_wiki.evaluation.paths import (
    AGENT_RUNS_DIR,
    BENCH_DATA,
    COMPOUND_PATH,
    F9_SAMPLE_PATH,
    REPORT_DIR,
)
from playground_wiki.evaluation.prompt_variants import CANARY_SENTENCE, VARIANTS, RewritingLLM
from playground_wiki.evaluation.prose_labels import RefusalVerdict
from playground_wiki.gate import KiwiTokenizer

STEPS = ("run", "report", "review-sheet", "judge-input", "judge-verdicts", "safety-report")
TAGGED_STEPS = frozenset({"review-sheet", "judge-input", "judge-verdicts", "safety-report"})
"""회차를 뺀 꼬리표(--tag safety)가 꼭 있어야 하는 단계."""
SINGLE_FROM_RECORDS = frozenset({"harmful", "procedure"})
"""단발을 새로 돌리지 않고 judge_partial 의 gate-v5 기록을 읽는 세트."""


def main() -> None:
    args = _parse_args()
    if args.step == "run" and args.set == "f4":
        kinds = _kinds(args.kinds)
        run_f4(args.mode, bench_data=args.bench_data, limit=args.limit, tag=args.tag, kinds=kinds)
    elif args.step == "run":
        run(args.set, args.mode, limit=args.limit, tag=args.tag)
    elif args.step == "report":
        report(args.tag, args.report_date, args.single_tag)
    elif args.step == "review-sheet":
        review_sheet(args.tag)
    elif args.step == "judge-input":
        judge_input(args.tag, bench_data=args.bench_data, kinds=_kinds(args.kinds))
    elif args.step == "judge-verdicts":
        judge_verdicts(args.tag, args.judged)
    else:
        safety_report(args.tag, args.report_date, root=args.data_root, bench_data=args.bench_data)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=STEPS)
    parser.add_argument("--set", choices=SET_NAMES, help="run · review-sheet: 세트")
    parser.add_argument("--mode", choices=MODES, help="run: 답변 모드")
    parser.add_argument("--limit", type=int, help="run: 이번에 돌릴 최대 문항 수 (점검용)")
    parser.add_argument(
        "--tag",
        help="기록 · 리포트 이름 끝에 붙일 꼬리표 (예 v2). report 에서는 에이전트 기록의 꼬리표, "
        "review-sheet · judge-input · judge-verdicts · safety-report 에서는 회차를 뺀 꼬리표 (safety → safety-1 …)",
    )
    parser.add_argument(
        "--single-tag", help="report: 짝지을 단발 기록의 꼬리표 (기본: 꼬리표 없는 단발 기록)"
    )
    parser.add_argument(
        "--report-date", type=date.fromisoformat, help="report · safety-report: 리포트 날짜"
    )
    parser.add_argument(
        "--bench-data",
        type=Path,
        default=BENCH_DATA,
        help="f4 세트: LLM 레포 bench/data (worktree 에서는 절대 경로)",
    )
    parser.add_argument(
        "--kinds", help=f"run --set f4 · judge-input: 이 F4 종류만 (쉼표로, {', '.join(F4_KINDS)})"
    )
    parser.add_argument("--judged", type=Path, help="judge-verdicts: 판정자 출력 JSONL")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(),
        help="safety-report: data/ 를 이 폴더 아래에서 읽는다 (리포트는 지금 폴더의 reports/ 에)",
    )
    args = parser.parse_args(argv)
    if args.step == "run" and (args.set is None or args.mode is None):
        parser.error("run needs --set and --mode")
    if args.step == "run" and args.set in SINGLE_FROM_RECORDS and args.mode != "agent":
        parser.error(
            f"--set {args.set} 의 단발은 새로 돌리지 않는다 — judge_partial 의 gate-v5 기록을 읽는다"
        )
    if args.step == "run" and args.set in SAFETY_SETS and args.tag is None:
        parser.error(f"--set {args.set} 는 회차 꼬리표(--tag safety-1 …)와 함께 돌린다")
    if args.step == "review-sheet" and args.set != "harmful":
        parser.error("review-sheet 는 --set harmful 만")
    if args.step in TAGGED_STEPS and args.tag is None:
        parser.error(f"{args.step} needs --tag")
    if args.step == "judge-verdicts" and args.judged is None:
        parser.error("judge-verdicts needs --judged")
    if args.kinds and (unknown := set(args.kinds.split(",")) - set(F4_KINDS)):
        parser.error(f"--kinds 에 모르는 종류: {sorted(unknown)}")
    for tag in (args.tag, args.single_tag):
        if tag is not None and not TAG_PATTERN.fullmatch(tag):
            parser.error(f"tags must match {TAG_PATTERN.pattern}")
    return args


def _kinds(text: str | None) -> frozenset[str] | None:
    return frozenset(text.split(",")) if text else None


def records_path(set_name: str, mode: str, tag: str | None = None) -> Path:
    """기록 경로 — data/eval/agent/{세트}-{모드}[-{꼬리표}].jsonl."""
    suffix = "" if tag is None else f"-{tag}"
    return AGENT_RUNS_DIR / f"{set_name}-{mode}{suffix}.jsonl"


def load_items(set_name: SetName) -> tuple[list[AgentItem], Path]:
    """세트의 문항과 그 입력 파일 (f4 는 `run_f4` 가 따로 읽는다)."""
    if set_name == "compound":
        return compound_items(read_jsonl(COMPOUND_PATH)), COMPOUND_PATH
    if set_name in SAFETY_SET_PATHS:
        path = SAFETY_SET_PATHS[set_name]
        return safety_items(read_jsonl(path), set_name), path
    return f9_items(read_jsonl(F9_SAMPLE_PATH), set_name), F9_SAMPLE_PATH


def run(set_name: SetName, mode: Mode, *, limit: int | None, tag: str | None) -> None:
    items, source = load_items(set_name)
    path = records_path(set_name, mode, tag)
    config = ApiConfig.from_env()
    # /ask 와 같이 Kiwi 하나(약 500MB)를 질문 관문과 BM25 가 나눠 쓴다
    kiwi = Kiwi()
    setup = AnswerSetup.from_config(config, tokenizer=KiwiTokenizer(kiwi), log_questions=False)
    _append_meta(path, run_settings(setup, config, mode=mode, source=source), limit=limit)
    print(
        f"{set_name} {len(items)} 문항 · {mode} · collection={config.search.collection}", flush=True
    )
    start = time.perf_counter()
    with PgStore.connect(config.pg) as pg, OllamaLLM.connect(OllamaConfig.from_env()) as llm:
        service = SearchService.build(
            config.search, pg, QdrantStore.from_env(), analyzer=KiwiAnalyzer(kiwi)
        )
        recording = RecordingSearch(service)
        runner = AgentEvalRun(
            recording, AnswerService.from_config(setup, search=recording, llm=llm)
        )
        summary = asyncio.run(runner.run(items, mode, path, limit=limit, on_record=_progress))
    print(
        f"이번 실행 {summary.recorded} · 건너뜀 {summary.skipped} · 다시 돈 생성 실패 {summary.retried} "
        f"/ 문항 {summary.total} · {time.perf_counter() - start:.0f}s\n저장: {path}"
    )


def run_f4(
    mode: Mode,
    *,
    bench_data: Path,
    limit: int | None,
    tag: str | None,
    kinds: frozenset[str] | None = None,
) -> None:
    """F4 저장 문맥 문항을 한 모드로 — 모든 검색 호출이 그 문항의 저장 문맥을 받는다. PG · Qdrant · 리랭커 없이 Ollama 만.

    직접 인젝션 문항은 eval_f4_prose 와 같이 지시문 첫 줄에 표식 한 문장을 심는다 (에이전트 노드 지시문에도 — 같은
    LLM 클라이언트를 쓴다).
    """
    items, sets = f4_run_items(bench_data, kinds)
    by_id = {item.id: item for item in items}
    path = records_path("f4", mode, tag)
    config = ApiConfig.from_env()
    # 검색은 저장 문맥이라 Kiwi 는 관문만 쓴다 (eval_f4_prose 와 같은 조립)
    setup = AnswerSetup.from_config(config, tokenizer=KiwiTokenizer.default(), log_questions=False)
    settings = f4_settings(setup, mode=mode, sets=sets, bench_data=bench_data)
    _append_meta(path, settings, limit=limit)
    print(f"f4 {len(items)} 문항 · {mode} · 관문 {'켬' if setup.gate else '끔'}", flush=True)
    start = time.perf_counter()
    stored = StoredContextSearch()
    recording = RecordingSearch(stored)
    with OllamaLLM.connect(OllamaConfig.from_env()) as ollama:
        rewriting = RewritingLLM(ollama, VARIANTS["service"])

        def prepare(item: AgentItem) -> None:
            f4_item = by_id[item.id]
            stored.use(f4_item)
            rewriting.set_canary(f4_item.canary if f4_item.kind == "injection_direct" else None)

        runner = AgentEvalRun(
            recording,
            AnswerService.from_config(setup, search=recording, llm=rewriting),
            before_answer=prepare,
        )
        agent_items = [f4_agent_item(item) for item in items]
        summary = asyncio.run(runner.run(agent_items, mode, path, limit=limit, on_record=_progress))
    print(
        f"이번 실행 {summary.recorded} · 건너뜀 {summary.skipped} · 다시 돈 생성 실패 {summary.retried} "
        f"/ 문항 {summary.total} · {time.perf_counter() - start:.0f}s\n저장: {path}"
    )


def f4_run_items(
    bench_data: Path, kinds: frozenset[str] | None = None
) -> tuple[list[F4Item], list[F4Set]]:
    """F4 안전 회귀 문항 (답 있음 쌍둥이 뺌, kinds 가 있으면 그 종류만)과 읽은 세트."""
    items, sets = load_f4_items(bench_data, F4_KINDS_BY_SET)
    picked = [item for item in f4_safety_items(items) if kinds is None or item.kind in kinds]
    return picked, sets


def run_settings(
    setup: AnswerSetup, config: ApiConfig, *, mode: str, source: Path
) -> dict[str, object]:
    """실행 설정 — 모델 · 프롬프트 판본 · 임계값 · 세트 파일 해시 (멈췄다 다시 돌린 사이에 바뀌었는지 리포트가 본다)."""
    return {
        "mode": mode,
        "collection": config.search.collection,
        **_answer_settings(setup),
        "set": str(source),
        "set_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


def f4_settings(
    setup: AnswerSetup, *, mode: str, sets: Sequence[F4Set], bench_data: Path
) -> dict[str, object]:
    """F4 실행 설정 — 검색 대신 저장 문맥 (collection 없음), 세트마다 해시 · 종류별 문항 수, 직접 인젝션 표식 문장."""
    return {
        "mode": mode,
        "search": "stored-context",
        **_answer_settings(setup),
        "direct_canary_sentence": CANARY_SENTENCE,
        "sets": [f4_set.to_json() for f4_set in sets],
        "bench_data": str(bench_data),
    }


def _answer_settings(setup: AnswerSetup) -> dict[str, object]:
    library = PromptLibrary.load(AGENT_PROMPT_DIR)
    return {
        "profiles": {
            stage: {"spec_id": p.spec.spec_id, "system": list(p.system), "user": p.user}
            for stage, p in setup.policy.profiles.items()
        },
        "thresholds": {"high": setup.policy.thresholds.high, "low": setup.policy.thresholds.low},
        "gate": setup.gate is not None,
        "agent_prompts": [f"{ref}:{library.get(ref).digest}" for ref in library.refs],
    }


def _append_meta(path: Path, settings: Mapping[str, object], *, limit: int | None) -> None:
    meta = path.with_suffix(".meta.json")
    runs = json.loads(meta.read_text(encoding="utf-8"))["runs"] if meta.exists() else []
    runs.append(
        {
            "started_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
            "limit": limit,
            "settings": dict(settings),
        }
    )
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(
        json.dumps({"runs": runs}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _progress(index: int, total: int, record: AgentRecord) -> None:
    print(
        f"  {index}/{total} {record.item.id} {record.final_stage} "
        f"검색 {len(record.searches)} · {record.wall_ms / 1000:.1f}s",
        flush=True,
    )


def report(tag: str | None, report_date: date | None, single_tag: str | None = None) -> None:
    pairings = []
    settings: dict[str, str] = {}
    for set_name in COMPARE_SETS:
        # --tag 는 에이전트 기록에만 — 단발 기록은 다시 돌리지 않고 꼬리표 없는 것(--single-tag 로 바꿈)과 짝짓는다
        single = read_records(records_path(set_name, "single", single_tag))
        agent = read_records(records_path(set_name, "agent", tag))
        pairings.append(Pairing.match(set_name, single, agent))
        meta = records_path(set_name, "agent", tag).with_suffix(".meta.json")
        if meta.exists() and not settings:
            settings = _meta_settings(meta)
    today = report_date or datetime.now(tz=UTC).date()
    text = render_report(pairings, today=today, settings=settings)
    REPORT_DIR.mkdir(exist_ok=True)
    suffix = "" if tag is None else f"-{tag}"
    path = REPORT_DIR / f"{today:%Y-%m-%d}-agent{suffix}.md"
    path.write_text(text, encoding="utf-8", newline="\n")  # .gitattributes 가 LF
    counts = " · ".join(f"{p.set_name} {len(p.single)}" for p in pairings)
    print(f"짝 {counts} · 리포트: {path}")


def _meta_settings(meta: Path) -> dict[str, str]:
    last = json.loads(meta.read_text(encoding="utf-8"))["runs"][-1]["settings"]
    settings = {
        "grounded": last["profiles"]["grounded"]["spec_id"],
        "agent_prompts": " ".join(last["agent_prompts"]),
    }
    if "collection" in last:
        settings["collection"] = last["collection"]
    return settings


def review_sheet(tag: str) -> None:
    """에이전트 유해 답 검수표 — 회차 셋의 partial 답 전부 + grounded 로 끝난 유해 질문 답을 섞어 가린 전문."""
    path = agent_review_sheet(tag)
    if path.exists():
        raise SystemExit(
            f"{path} 가 이미 있다 — 채우던 검수표일 수 있어 덮어쓰지 않는다. 다시 만들려면 직접 지운다"
        )
    items, _ = load_items("harmful")
    expected = {item.id for item in items}
    records_by_run: dict[str, list[AgentRecord]] = {}
    for round_number in ROUNDS:
        records_file = records_path("harmful", "agent", agent_tag(tag, round_number))
        records = read_records(records_file)
        if {record.item.id for record in records} != expected:
            raise SystemExit(f"{records_file} 에 세트 문항이 다 없다 — run 을 먼저 다 돌린다")
        records_by_run[records_file.stem] = records
    picks = review_picks(records_by_run)
    entries = review_entries(picks)
    part = f"harmful-agent-{tag}-123"
    text = render_sheet(f"partial 판정 검수표 v1 — {part} {len(entries)}문항", entries, ITEM_GUIDE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    key = {
        "created_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "part": part,
        "seed": REVIEW_SEED,
        "entries": review_key_entries(picks),
    }
    key_path = path.with_suffix(".key.json")
    key_path.write_text(json.dumps(key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    per_run = " · ".join(
        f"{run} {sum(pick.run == run for pick in picks)}" for run in records_by_run
    )
    print(
        f"검수표 {len(entries)}문항 ({per_run}): {path}\n열쇠(문항 번호 → 기록): {key_path}\n"
        f"채울 때는 {checked_path(path).name} 으로 복사해 그 사본에 적는다 — 이 파일은 틀로 남는다"
    )


def refusal_paths(tag: str) -> tuple[Path, Path, Path]:
    """F4 '답하지 않음' 판정의 (입력, 열쇠, 판정 결과) 경로 — 회차 꼬리표 이름."""
    round_tag = agent_tag(tag, F4_ROUND)
    return (
        AGENT_RUNS_DIR / f"f4-refusal-judge-input-{round_tag}.jsonl",
        AGENT_RUNS_DIR / f"f4-refusal-judge-key-{round_tag}.jsonl",
        AGENT_RUNS_DIR / f"f4-refusal-verdicts-{round_tag}.jsonl",
    )


def judge_input(tag: str, *, bench_data: Path, kinds: frozenset[str] | None = None) -> None:
    """F4 단발 · 에이전트 답 중 '답하지 않음'을 판정할 답의 입력 · 열쇠 (판정은 메인이 Claude 하위 에이전트로)."""
    input_path, key_path, verdict_path = refusal_paths(tag)
    if input_path.exists():
        raise SystemExit(f"{input_path} 가 이미 있다 — 판정 중일 수 있어 덮어쓰지 않는다")
    items, _ = f4_run_items(bench_data)
    by_id = {item.id: item for item in items}
    labeled = []
    for mode in MODES:
        records_file = records_path("f4", mode, agent_tag(tag, F4_ROUND))
        records = read_records(records_file)
        if not records:
            raise SystemExit(f"{records_file} 가 없다 — run --set f4 를 먼저 돌린다")
        labels = f4_labels(records, by_id)
        labeled += [
            (records_file.stem, record, labels[record.item.id])
            for record in records
            if record.item.id in labels
        ]
    inputs, keys = refusal_judge_rows(labeled, kinds=kinds or REFUSAL_JUDGE_KINDS)
    write_jsonl(input_path, inputs)
    write_jsonl(key_path, keys)
    print(
        f"판정할 답 {len(inputs)}: {input_path}\n열쇠: {key_path}\n"
        "판정 기준은 reports/2026-10-31-refusal-rubric.md — 판정자 출력(JSONL: judge_id · refused · reason)을 "
        f"judge-verdicts --judged 로 넘기면 {verdict_path} 에 쓴다"
    )


def judge_verdicts(tag: str, judged: Path) -> None:
    """판정자 출력을 열쇠로 풀어 판정 결과 파일로 (prose_labels.RefusalVerdict 한 줄씩)."""
    _, key_path, verdict_path = refusal_paths(tag)
    if verdict_path.exists():
        raise SystemExit(f"{verdict_path} 가 이미 있다 — 덮어쓰지 않는다")
    verdicts = refusal_verdicts(read_jsonl(judged), read_jsonl(key_path))
    write_jsonl(verdict_path, [verdict.to_json() for verdict in verdicts])
    refused = sum(verdict.refused for verdict in verdicts)
    print(f"판정 {len(verdicts)} (답하지 않음 {refused}): {verdict_path}")


def safety_report(
    tag: str, report_date: date | None, *, root: Path = Path(), bench_data: Path = BENCH_DATA
) -> None:
    """안전 회귀 리포트 — 유해 · 절차 세 회차 짝, F4 짝, 하드 기준 표 → reports/{날짜}-agent-safety.md."""
    harmful = _round_pairings("harmful", tag, root)
    procedure = _round_pairings("procedure", tag, root)
    f4_tag = agent_tag(tag, F4_ROUND)
    f4_single_path, f4_agent_path = (root / records_path("f4", m, f4_tag) for m in MODES)
    f4_single, f4_agent = read_records(f4_single_path), read_records(f4_agent_path)
    f4 = Pairing.match("f4", f4_single, f4_agent) if f4_single and f4_agent else None
    f4_items = {item.id: item for item in f4_run_items(bench_data)[0]} if f4 else {}
    verdict_path = root / refusal_paths(tag)[2]
    verdicts: dict[tuple[str, str], RefusalVerdict] = {}
    if verdict_path.exists():
        for row in read_jsonl(verdict_path):
            verdict = RefusalVerdict.from_json(row)
            verdicts[verdict.variant, verdict.id] = verdict
    inputs = SafetyInputs(
        harmful=tuple(harmful),
        procedure=tuple(procedure),
        f4=f4,
        f4_runs=(f4_single_path.stem, f4_agent_path.stem),
        f4_items=f4_items,
        verdicts=verdicts,
        human_single=_human(root, SINGLE_REVIEW_SHEET),
        human_agent=_human(root, agent_review_sheet(tag)),
        settings=_safety_settings(root, tag),
    )
    today = report_date or datetime.now(tz=UTC).date()
    text = render_safety_report(inputs, today=today)
    REPORT_DIR.mkdir(exist_ok=True)
    path = REPORT_DIR / f"{today:%Y-%m-%d}-agent-safety.md"
    path.write_text(text, encoding="utf-8", newline="\n")  # .gitattributes 가 LF
    print(
        f"유해 {len(harmful)} 회차 · 절차 {len(procedure)} 회차 · F4 {len(f4.single) if f4 else 0} 짝 · "
        f"판정 {len(verdicts)} · 사람 채점 단발 {'있음' if inputs.human_single else '없음'} · "
        f"에이전트 {'있음' if inputs.human_agent else '없음'}\n리포트: {path}"
    )


def _round_pairings(set_name: str, tag: str, root: Path) -> list[RoundPairing]:
    """회차마다 단발(gate-v5 기록) · 에이전트(tag-회차 기록) 짝 — 두 기록이 다 있는 회차만."""
    rounds = []
    for round_number in ROUNDS:
        single_path = single_safety_path(set_name, round_number, root=root)
        agent_path = root / records_path(set_name, "agent", agent_tag(tag, round_number))
        if not single_path.exists() or not agent_path.exists():
            continue
        single = [
            from_answer_record(record, set_name) for record in read_answer_records(single_path)
        ]
        pairing = Pairing.match(set_name, single, read_records(agent_path))
        rounds.append(
            RoundPairing(round_number, single_run(set_name, round_number), agent_path.stem, pairing)
        )
    return rounds


def _human(root: Path, sheet: Path) -> HumanLabels | None:
    """채운 검수표의 사람 채점 — (실행, 문항 id) → 항목 → 값. 채운 사본이 없으면 None."""
    read = read_checked(root, sheet)
    if read is None:
        return None
    ratings, entries = read
    return {entry.answer: ratings[entry.number] for entry in entries if entry.number in ratings}


def _safety_settings(root: Path, tag: str) -> dict[str, str]:
    for set_name, round_tag in (
        ("harmful", agent_tag(tag, ROUNDS[0])),
        ("f4", agent_tag(tag, F4_ROUND)),
    ):
        meta = (root / records_path(set_name, "agent", round_tag)).with_suffix(".meta.json")
        if meta.exists():
            return _meta_settings(meta)
    return {}


if __name__ == "__main__":
    main()
