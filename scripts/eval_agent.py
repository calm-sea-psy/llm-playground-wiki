"""단발 대 에이전트 비교 측정: 같은 세트를 /ask 단발 · 에이전트 두 모드로 돌려 기록하고 리포트를 쓴다.

  # 기록 (GPU — Ollama · 임베딩 · 리랭커). 이어 쓰기: 이미 기록한 문항은 건너뛴다
  uv run python scripts/eval_agent.py run --set compound --mode single
  uv run python scripts/eval_agent.py run --set compound --mode agent
  uv run python scripts/eval_agent.py run --set f9 --mode agent --limit 100
  # 리포트 (GPU 없이) → reports/{날짜}-agent.md
  uv run python scripts/eval_agent.py report

세트: compound(복합 50, tests/evaluation/compound-v1.jsonl) · f9(F9 표본의 KorQuAD 370) · offwiki(F9 표본의 위키 밖 100).
기록은 data/eval/agent/{세트}-{모드}[-{꼬리표}].jsonl 과 옆 .meta.json (실행마다 설정). 리포트에는 집계 · 문항 id 만.
"""

import argparse
import asyncio
import hashlib
import json
import time
from collections.abc import Sequence
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
    MODES,
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
from playground_wiki.evaluation.answer_run import TAG_PATTERN
from playground_wiki.evaluation.files import read_jsonl
from playground_wiki.evaluation.paths import (
    AGENT_RUNS_DIR,
    COMPOUND_PATH,
    F9_SAMPLE_PATH,
    REPORT_DIR,
)
from playground_wiki.gate import KiwiTokenizer


def main() -> None:
    args = _parse_args()
    if args.step == "run":
        run(args.set, args.mode, limit=args.limit, tag=args.tag)
    else:
        report(args.tag, args.report_date)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=["run", "report"])
    parser.add_argument("--set", choices=SET_NAMES, help="run: 세트")
    parser.add_argument("--mode", choices=MODES, help="run: 답변 모드")
    parser.add_argument("--limit", type=int, help="run: 이번에 돌릴 최대 문항 수 (점검용)")
    parser.add_argument("--tag", help="기록 · 리포트 이름 끝에 붙일 꼬리표 (예 judge-v2)")
    parser.add_argument("--report-date", type=date.fromisoformat, help="report: 리포트 날짜")
    args = parser.parse_args(argv)
    if args.step == "run" and (args.set is None or args.mode is None):
        parser.error("run needs --set and --mode")
    if args.tag is not None and not TAG_PATTERN.fullmatch(args.tag):
        parser.error(f"--tag must match {TAG_PATTERN.pattern}")
    return args


def records_path(set_name: str, mode: str, tag: str | None = None) -> Path:
    """기록 경로 — data/eval/agent/{세트}-{모드}[-{꼬리표}].jsonl."""
    suffix = "" if tag is None else f"-{tag}"
    return AGENT_RUNS_DIR / f"{set_name}-{mode}{suffix}.jsonl"


def load_items(set_name: SetName) -> tuple[list[AgentItem], Path]:
    """세트의 문항과 그 입력 파일."""
    if set_name == "compound":
        return compound_items(read_jsonl(COMPOUND_PATH)), COMPOUND_PATH
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


def run_settings(
    setup: AnswerSetup, config: ApiConfig, *, mode: str, source: Path
) -> dict[str, object]:
    """실행 설정 — 모델 · 프롬프트 판본 · 임계값 · 세트 파일 해시 (멈췄다 다시 돌린 사이에 바뀌었는지 리포트가 본다)."""
    library = PromptLibrary.load(AGENT_PROMPT_DIR)
    return {
        "mode": mode,
        "collection": config.search.collection,
        "profiles": {
            stage: {"spec_id": p.spec.spec_id, "system": list(p.system), "user": p.user}
            for stage, p in setup.policy.profiles.items()
        },
        "thresholds": {"high": setup.policy.thresholds.high, "low": setup.policy.thresholds.low},
        "gate": setup.gate is not None,
        "agent_prompts": [f"{ref}:{library.get(ref).digest}" for ref in library.refs],
        "set": str(source),
        "set_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


def _append_meta(path: Path, settings: dict[str, object], *, limit: int | None) -> None:
    meta = path.with_suffix(".meta.json")
    runs = json.loads(meta.read_text(encoding="utf-8"))["runs"] if meta.exists() else []
    runs.append(
        {
            "started_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
            "limit": limit,
            "settings": settings,
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


def report(tag: str | None, report_date: date | None) -> None:
    pairings = []
    settings: dict[str, str] = {}
    for set_name in SET_NAMES:
        single = read_records(records_path(set_name, "single", tag))
        agent = read_records(records_path(set_name, "agent", tag))
        pairings.append(Pairing.match(set_name, single, agent))
        meta = records_path(set_name, "agent", tag).with_suffix(".meta.json")
        if meta.exists() and not settings:
            last = json.loads(meta.read_text(encoding="utf-8"))["runs"][-1]["settings"]
            settings = {
                "grounded": last["profiles"]["grounded"]["spec_id"],
                "agent_prompts": " ".join(last["agent_prompts"]),
                "collection": last["collection"],
            }
    today = report_date or datetime.now(tz=UTC).date()
    text = render_report(pairings, today=today, settings=settings)
    REPORT_DIR.mkdir(exist_ok=True)
    suffix = "" if tag is None else f"-{tag}"
    path = REPORT_DIR / f"{today:%Y-%m-%d}-agent{suffix}.md"
    path.write_text(text, encoding="utf-8", newline="\n")  # .gitattributes 가 LF
    counts = " · ".join(f"{p.set_name} {len(p.single)}" for p in pairings)
    print(f"짝 {counts} · 리포트: {path}")


if __name__ == "__main__":
    main()
