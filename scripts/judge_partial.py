"""F9 partial(분리 답변) 판정 · 유해 정보 세트 · 판정기 검증 — 단계별로 돌린다.

  check     세트를 검사한다 (--set harmful | procedure) — 유해 세트는 유해 질문마다 같은 주제의 대조 질문(과잉 거절
            확인용)이 짝으로 있는지, 무해 절차 세트는 id · kind · 질문이 찼는지. 질문 원문은 출력하지 않는다
  answer    세트를 답변 경로로 돌려 기록 → data/eval/{set}-answers-v1-{route|forced}.jsonl (+ .meta.json).
            --mode route 는 /ask 와 같은 조립 그대로(관문 · 점수로 단계 · 캐스케이드), --mode forced 는 관문을 끄고 점수와
            상관없이 partial 로 (문맥은 운영과 같이 low 이상 청크) — partial 답을 문항마다 확보하는 최악 조건.
            기록 형식 · 이어 돌리기는 eval_answer.py run 과 같다. PG · Qdrant · BM25 색인 · Ollama(gemma4:12b) 필요.
            --tag T 는 프롬프트를 바꿔 다시 잴 때 — 기록을 {set}-answers-v1-{route|forced}-T.jsonl 에 따로 쓴다.
            --prompt partial@v3 은 그 단계 블록만 바꿔 돌린다 (configs/answer.yaml 은 그대로 — 채택은 측정 뒤 사람이).
            여러 번 주면 단계마다 바꾼다 (--prompt grounded@v4 --prompt partial@v4). 한 단계에 둘은 받지 않는다
  review-sheet
            판정기 검증용 사람 채점 검수표 → data/eval/partial-judge-review-v1-{f9|harmful}.md (+ .key.json — 검수표 번호와
            원래 문항을 잇는 열쇠). 판정 결과 · 문항 묶음은 싣지 않는다 (blind). 표본은 판정과 상관없이 seed 0 으로:
            --part f9 는 F9 partial 답에서 묶음(KorQuAD · 위키 밖 partial 기대 · 위키 밖 refuse 기대) 비례 30,
            --part harmful 은 forced 기록에서 짝 10개(유해 10 + 대조 10), --part harmful-route 는 운영 경로에서
            partial 로 끝난 답 전부. 이미 있는 검수표는 덮어쓰지 않는다.
            --tag T 는 프롬프트를 바꾼 뒤 전 · 후를 견줄 때 — 같은 문항의 기준 답과 T 답을 둘 다 싣고 섞는다
            (어느 쪽이 새 프롬프트인지 채점자가 모르게. 열쇠 파일의 run 에만 남는다).
            --base-tag B 는 기준 답도 꼬리표 기록에서 뽑는다 — route 는 grounded 가 바뀌면 어느 질문이 partial 로
            가는지가 달라져 옛 기준 기록과 견줄 수 없다. forced 는 점수와 무관하니 옛 기준 기록을 써도 된다
  judge     partial 로 끝난 답 전부(F9 + 유해 정보 세트 두 실행)를 판정 → data/eval/partial-judged-v1.jsonl.
            형식(정규식)과 판정 모델(gpt-oss:20b) 판정. 끝난 답은 건너뛰고, 판정 실패와 **판정 지시가 바뀐 뒤의
            예전 기록**은 다시 돌린다. --review-only 는 검수표에 실린 답만 (지시를 고친 뒤 일치율만 싸게 다시 잴 때).
            --tag T 는 꼬리표 기록 하나(--run f9 · harmful-route · procedure-route, 기본 f9 — 예 f9-answers-v1-T.jsonl)의
            partial 답만 같은 판정 지시로 판정해 data/eval/partial-judged-v1-tagged.jsonl 에 쓴다 (줄의 run 은 `실행-T`).
            기본 판정 파일 · report 는 건드리지 않는다. Ollama 필요
  agreement 채운 검수표(빈 표 이름에 `-checked` 를 붙인 사본 — 빈 표는 틀로 남긴다)를 판정과 맞춰
            항목별 일치율 · Cohen's kappa 를 출력한다. 읽는 검수표는 REVIEW_SHEETS 표 — 새 검수표는 거기에 한 줄 더한다
  report    reports/{날짜}-partial-judge.md 에 집계 · 문항 id · 주제 이름만 (질문 · 답 원문 없음). 사람 채점은
            REVIEW_SHEETS 의 검수표를 모두 합친다 — 꼬리표 기록(프롬프트를 바꿔 다시 잰 것)의 채점은 기록 · 회차별 표로
            따로 싣고, 판정기 판정이 없는 검수표는 일치율 표에서 뺀다. --report-date YYYY-MM-DD 는 리포트 날짜 (다시 그릴 때 처음 날짜를
            지킨다). agreement · report 의 --data-root DIR 은 data/ 를 DIR 아래에서 읽는다 — 워크트리에서 기본 폴더의
            기록으로 그릴 때 (리포트는 지금 폴더의 reports/ 에 쓴다)

유해 정보 세트 한 줄: {"id", "kind": "harmful" | "contrast", "pair": 짝 id, "topic", "question", "related_title": 문서 제목 | null}.
무해 절차 세트 한 줄: {"id", "kind": "procedure", "topic", "question"} — 짝이 없다.
세트 · 기록 · 검수표는 data/ 에만 둔다 (git 제외 — KorQuAD 문맥 · 위험 질문). 레포 루트에서 돈다.
GPU 하나를 나눠 쓴다 — answer(gemma4:12b)와 judge(gpt-oss:20b)는 같이 올라가지 않으니 차례로 돌린다.

사용법:
  uv run python scripts/judge_partial.py check
  uv run python scripts/judge_partial.py answer --mode forced --limit 3
  uv run python scripts/judge_partial.py answer --mode route
  uv run python scripts/judge_partial.py answer --mode route --tag grounded-v3
  uv run python scripts/judge_partial.py check --set procedure
  uv run python scripts/judge_partial.py answer --set procedure --mode route --tag partial-v3-1 --prompt partial@v3
  uv run python scripts/judge_partial.py answer --mode route --tag safety-v4-1 --prompt grounded@v4 --prompt partial@v4
  uv run python scripts/judge_partial.py review-sheet --part f9
  uv run python scripts/judge_partial.py review-sheet --part harmful --tag partial-v3-1
  uv run python scripts/judge_partial.py review-sheet --part harmful-route --tag partial-v3-1 --base-tag partial-v2-1
  uv run python scripts/judge_partial.py judge --limit 5
  uv run python scripts/judge_partial.py judge --review-only
  uv run python scripts/judge_partial.py judge --tag partial-base-1
  uv run python scripts/judge_partial.py judge --tag gate-v5-1 --run procedure-route
  uv run python scripts/judge_partial.py agreement
  uv run python scripts/judge_partial.py report
  uv run python scripts/judge_partial.py report --report-date 2026-10-29 --data-root ../../..
"""

import argparse
import asyncio
import random
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Self

from kiwipiepy import Kiwi
from playground_llm import LLM
from playground_llm.ollama import OllamaConfig, OllamaLLM
from playground_retrieval.core import Hit
from playground_retrieval.search import KiwiAnalyzer, SearchService
from playground_retrieval.store import PgStore, QdrantStore

from playground_wiki.api.answer import AnswerService
from playground_wiki.api.config import AnswerSetup, ApiConfig
from playground_wiki.evaluation.answer_judge import JUDGE_TIMEOUT_SEC
from playground_wiki.evaluation.answer_metrics import (
    final_stage,
)
from playground_wiki.evaluation.answer_run import (
    MAX_CONSECUTIVE_ERRORS,
    TAG_PATTERN,
    AnswerRecord,
    AnswerRun,
    refuse_prompt_change,
    tagged,
)
from playground_wiki.evaluation.files import (
    append_jsonl,
    append_run_meta,
    read_jsonl,
    write_jsonl,
)
from playground_wiki.evaluation.judge_review import (
    record_entry,
    refuse_existing_sheet,
    write_review_sheet,
)
from playground_wiki.evaluation.partial_items import (
    JUDGED_PATH,
    MODES,
    REVIEW_F9_SIZE,
    REVIEW_HARMFUL_PAIRS,
    REVIEW_PATHS,
    REVIEW_SHEETS,
    REVIEW_SOURCE_RUN,
    RUNS,
    SEED,
    SETS,
    TAGGED_RUNS,
    all_records,
    answer_path,
    judged_by_key,
    load_agreements,
    partial_records,
    read_judged,
    set_items,
    tagged_targets,
)
from playground_wiki.evaluation.partial_judge import (
    ITEM_GUIDE,
    JUDGE_PROMPT_ID,
    JUDGE_SPEC,
    JudgedRow,
    judged_row,
)
from playground_wiki.evaluation.partial_report import (
    agreement_section,
    render_report,
    run_prompts,
    unjudged_count,
)
from playground_wiki.evaluation.partial_review import (
    KeyEntry,
    f9_review_keys,
    fill_hint,
    harmful_review_keys,
    human_labels,
    review_keys,
)
from playground_wiki.evaluation.paths import (
    HARMFUL_SET_PATH as HARMFUL_SET,
    PARTIAL_JUDGED_TAGGED_PATH,
    REPORT_DIR,
)
from playground_wiki.gate import KiwiTokenizer
from playground_wiki.generation import AnswerPolicy, Decision


@dataclass(frozen=True)
class ProfileSettings:
    """answer 실행 meta 의 단계 하나 — 생성 설정과 단계 프롬프트 블록."""

    spec_id: str
    system: tuple[str, ...]
    user: str

    @classmethod
    def from_json(cls, row: Mapping[str, Any]) -> Self:
        """meta runs 의 profiles 값 하나에서."""
        return cls(
            spec_id=str(row["spec_id"]),
            system=tuple(str(block) for block in row["system"]),
            user=str(row["user"]),
        )

    def to_json(self) -> dict[str, Any]:
        """meta runs 의 profiles 값 하나 — 키 차례는 필드 차례."""
        return {"spec_id": self.spec_id, "system": list(self.system), "user": self.user}


@dataclass(frozen=True)
class RunSettings:
    """answer 실행 한 번의 meta (runs 한 줄) — 리포트는 마지막 실행의 단계 프롬프트를 읽는다 (run_prompts).

    쓰기만 이 꼴로 한다. 리포트는 meta 를 dict 로 읽는다 (옛 기록의 꼴도 함께 읽으려고).
    """

    started_at: str
    mode: str
    """route · forced."""
    limit: int | None
    gate: bool
    """질문 관문을 켰나 (forced 는 끈다)."""
    high: float
    low: float
    profiles: Mapping[str, ProfileSettings]
    """단계 이름 → 설정 (이름 차례)."""
    set_path: str
    """돌린 세트 파일 경로."""

    @classmethod
    def from_json(cls, row: Mapping[str, Any]) -> Self:
        """meta runs 한 줄에서."""
        return cls(
            started_at=str(row["started_at"]),
            mode=str(row["mode"]),
            limit=row["limit"],
            gate=bool(row["gate"]),
            high=float(row["thresholds"]["high"]),
            low=float(row["thresholds"]["low"]),
            profiles=MappingProxyType(
                {
                    str(stage): ProfileSettings.from_json(profile)
                    for stage, profile in row["profiles"].items()
                }
            ),
            set_path=str(row["inputs"]["set"]),
        )

    def to_json(self) -> dict[str, Any]:
        """meta runs 한 줄 — 키 차례는 기록 꼴 그대로."""
        return {
            "started_at": self.started_at,
            "mode": self.mode,
            "limit": self.limit,
            "gate": self.gate,
            "thresholds": {"high": self.high, "low": self.low},
            "profiles": {stage: profile.to_json() for stage, profile in self.profiles.items()},
            "inputs": {"set": self.set_path},
        }


class PartialOnlyPolicy(AnswerPolicy):
    """점수와 상관없이 partial — 유해 정보 세트의 최악 조건. 문맥은 운영과 같이 low 이상 청크 (없으면 빈 문맥)."""

    def decide(self, hits: Sequence[Hit]) -> Decision:
        """운영 정책의 판단에서 단계 · 생성 설정만 partial 로 바꾼다."""
        decision = super().decide(hits)
        return replace(decision, stage="partial", profile=self.profiles["partial"])


def main() -> None:
    """커맨드라인 진입점."""
    args = _parse_args()
    if args.step == "check":
        check(set_name=args.set)
    elif args.step == "answer":
        answer(
            set_name=args.set,
            mode=args.mode,
            limit=args.limit,
            tag=args.tag,
            prompts=args.prompt or (),
        )
    elif args.step == "review-sheet":
        review_sheet(part=args.part, tag=args.tag, base_tag=args.base_tag)
    elif args.step == "judge":
        judge(
            limit=args.limit,
            review_only=args.review_only,
            tag=args.tag,
            run=args.run or "f9",
        )
    elif args.step == "agreement":
        root = args.data_root or Path()
        agreements = load_agreements(root)
        human = human_labels(root, sheets=REVIEW_SHEETS)
        print(agreement_section(agreements, unjudged_count(human, agreements)))
    elif args.step == "report":
        report(root=args.data_root or Path(), report_date=args.report_date)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "step", choices=["check", "answer", "review-sheet", "judge", "agreement", "report"]
    )
    parser.add_argument("--mode", choices=MODES, default="forced")  # answer 만
    parser.add_argument("--set", choices=sorted(SETS), default="harmful")  # check · answer 만
    parser.add_argument("--part", choices=sorted(REVIEW_PATHS), default="f9")  # review-sheet 만
    # answer · judge: 이번에 돌릴 최대 문항 수 (점검용 — 끝난 것은 다음 실행이 건너뛴다)
    parser.add_argument("--limit", type=int)
    # judge 만: 검수표에 실린 답만 판정한다 — 판정 지시를 고친 뒤 일치율만 싸게 다시 재려고
    parser.add_argument("--review-only", action="store_true")
    # answer 만: 프롬프트 · 모델을 바꿔 다시 잰 기록의 꼬리표 (예 grounded-v3) — 기록 파일 이름 끝에 -{tag}.
    # 꼬리표 없는 기록(judge · report 가 읽는 기준)은 덮어쓰지 않는다.
    # judge 에서는 그 꼬리표 기록(--run)의 partial 답만 판정해 꼬리표 판정 파일에 쓴다
    parser.add_argument("--tag")
    # judge --tag 만: 판정할 꼬리표 기록의 실행 (TAGGED_RUNS). 없으면 f9
    parser.add_argument("--run", choices=sorted(TAGGED_RUNS))
    # answer 만: 단계 블록 프롬프트를 바꿔 돌린다 (예 partial@v3) — configs/answer.yaml 은 건드리지 않는다.
    # 여러 번 주면 단계마다 (grounded@v4 · partial@v4 를 함께). 기준 기록을 덮어쓰지 않게 --tag 와 함께 쓴다
    parser.add_argument("--prompt", action="append")
    # review-sheet 만: 견줄 기준 답을 꼬리표 없는 기록이 아니라 이 꼬리표 기록에서 뽑는다.
    # route 는 grounded 가 바뀌면 단계가 갈려 10-29 기준 기록과 견줄 수 없다 — 기준선도 같은 설정으로 다시 돌린다
    parser.add_argument("--base-tag")
    # report 만: 리포트 날짜(파일 이름 · 제목). 없으면 오늘 — 검수표를 더해 다시 그릴 때 처음 리포트 날짜를 지키려고
    parser.add_argument("--report-date", type=date.fromisoformat)
    # agreement · report 만: data/ 를 이 폴더 아래에서 읽는다 — 워크트리에는 data/ 가 없어 기본 폴더의 기록을 읽을 때
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args(argv)
    if args.report_date is not None and args.step != "report":
        parser.error("--report-date 는 report 에만 쓴다")
    if args.data_root is not None and args.step not in ("agreement", "report"):
        parser.error("--data-root 는 agreement · report 에만 쓴다 — 기록만 읽는 단계")
    if args.base_tag is not None and args.step != "review-sheet":
        parser.error("--base-tag 는 review-sheet 에만 쓴다")
    if args.base_tag is not None and args.tag is None:
        parser.error("--base-tag 는 --tag 와 함께 쓴다 — 견줄 두 기록을 다 준다")
    if args.base_tag is not None and not TAG_PATTERN.fullmatch(args.base_tag):
        parser.error(f"--base-tag 는 영문 소문자 · 숫자 · '-' 만 쓴다, got {args.base_tag!r}")
    if args.tag is not None and args.step not in ("answer", "review-sheet", "judge"):
        parser.error("--tag 는 answer · review-sheet · judge 에만 쓴다")
    if args.run is not None and (args.step != "judge" or args.tag is None):
        parser.error("--run 은 judge --tag 에만 쓴다 — 판정할 꼬리표 기록의 실행")
    if args.step == "judge" and args.tag is not None and args.review_only:
        parser.error("--review-only 는 꼬리표 없는 기록의 judge 에만 쓴다")
    if args.tag is not None and not TAG_PATTERN.fullmatch(args.tag):
        parser.error(f"--tag 는 영문 소문자 · 숫자 · '-' 만 쓴다, got {args.tag!r}")
    if args.prompt is not None and args.step != "answer":
        parser.error("--prompt 는 answer 에만 쓴다")
    if args.prompt is not None and args.tag is None:
        parser.error("--prompt 는 --tag 와 함께 쓴다 — 기준 기록을 덮어쓰지 않게")
    if args.prompt is not None and _repeated_stages(args.prompt):
        parser.error(f"--prompt 는 단계마다 하나만 준다, got {args.prompt}")
    return args


def _repeated_stages(refs: Sequence[str]) -> list[str]:
    """`단계@버전` 목록에서 두 번 이상 나온 단계 이름 (차례대로)."""
    counts = Counter(ref.partition("@")[0] for ref in refs)
    return [stage for stage, n in counts.items() if n > 1]


def check(*, set_name: str) -> None:
    """세트 형식을 확인한다."""
    path = SETS[set_name]
    rows = read_jsonl(path)
    try:
        items = set_items(set_name, rows)
    except ValueError as e:  # 세트를 채우는 사람이 읽을 목록이라 traceback 없이
        raise SystemExit(str(e)) from e
    kinds = Counter(item.category for item in items)
    topics = Counter(str(row.get("topic") or "") for row in rows if row["kind"] != "contrast")
    counts = " · ".join(f"{kind} {n}" for kind, n in sorted(kinds.items()))
    print(
        f"{path}: 문항 {len(items)} — {counts}\n주제 {len(topics)}가지: "
        + " · ".join(f"{t or '(빈 주제)'} {n}" for t, n in topics.most_common())
    )


def answer(
    *,
    set_name: str,
    mode: str,
    limit: int | None,
    tag: str | None = None,
    prompts: Sequence[str] = (),
) -> None:
    """세트를 답변 경로로 돌려 기록을 이어 쓴다."""
    items = set_items(set_name, read_jsonl(SETS[set_name]))
    path = answer_path(set_name, mode, tag)
    config = ApiConfig.from_env()
    # /ask 와 같이 Kiwi 하나를 질문 관문과 BM25 가 나눠 쓴다. forced 는 관문을 끄고 점수와 상관없이 partial 로
    # 보낸다 (문항마다 partial 답을 받으려고). 프롬프트를 바꿔 다시 잴 때도 configs/answer.yaml 은 그대로 둔다
    kiwi = Kiwi()
    setup = AnswerSetup.from_config(
        config,
        tokenizer=KiwiTokenizer(kiwi),
        prompts=prompts,
        gate=mode == "route",
        policy_type=PartialOnlyPolicy if mode == "forced" else AnswerPolicy,
        log_questions=False,
    )
    refuse_prompt_change(path, setup.policy.profiles)
    _append_run_meta(
        path,
        set_path=SETS[set_name],
        mode=mode,
        limit=limit,
        policy=setup.policy,
        gate_enabled=setup.gate is not None,
    )
    print(f"문항 {len(items)} · {mode} · 관문 {'켬' if setup.gate else '끔'}", flush=True)
    start = time.perf_counter()
    with PgStore.connect(config.pg) as pg, OllamaLLM.connect(OllamaConfig.from_env()) as llm:
        service = SearchService.build(
            config.search, pg, QdrantStore.from_env(), analyzer=KiwiAnalyzer(kiwi)
        )
        print(f"검색 서비스 준비 {time.perf_counter() - start:.0f}s", flush=True)
        runner = AnswerRun(
            service, lambda search: AnswerService.from_config(setup, search=search, llm=llm)
        )
        summary = asyncio.run(runner.run(items, path, limit=limit, on_record=_progress))
    print(
        f"이번 실행 {summary.recorded} · 건너뜀 {summary.skipped} · 다시 돈 생성 실패 {summary.retried} "
        f"/ 문항 {summary.total} · {time.perf_counter() - start:.0f}s\n저장: {path}"
    )


def _progress(index: int, total: int, record: AnswerRecord) -> None:
    print(f"  {index}/{total} {record.item.id} {final_stage(record)} {record.wall_ms / 1000:.1f}s")


def _append_run_meta(
    path: Path,
    *,
    set_path: Path,
    mode: str,
    limit: int | None,
    policy: AnswerPolicy,
    gate_enabled: bool,
) -> None:
    settings = RunSettings(
        started_at=datetime.now(tz=UTC).isoformat(timespec="seconds"),
        mode=mode,
        limit=limit,
        gate=gate_enabled,
        high=policy.thresholds.high,
        low=policy.thresholds.low,
        profiles=MappingProxyType(
            {
                stage: ProfileSettings(
                    spec_id=profile.spec.spec_id, system=tuple(profile.system), user=profile.user
                )
                for stage, profile in sorted(policy.profiles.items())
            }
        ),
        set_path=str(set_path),
    )
    append_run_meta(path, settings.to_json())


def review_sheet(*, part: str, tag: str | None = None, base_tag: str | None = None) -> None:
    """사람 채점 검수표와 열쇠를 쓴다."""
    path = tagged(REVIEW_PATHS[part], tag)
    refuse_existing_sheet(path)
    run = REVIEW_SOURCE_RUN[part]
    base_run = run if base_tag is None else f"{run}-{base_tag}"
    records = {record.item.id: record for record in partial_records(run, base_tag)}
    if not records:
        raise SystemExit(f"{base_run} 에 partial 답이 없다 — answer 를 먼저 돌린다")
    if part == "f9":
        keys = f9_review_keys(records.values(), REVIEW_F9_SIZE, seed=SEED)
    elif part == "harmful-route":
        keys = sorted(records)  # 운영 경로에서 partial 로 끝난 답 전부 — 짝이 맞는 문항이 적다
    else:
        pairs = {str(row["id"]): str(row["pair"]) for row in read_jsonl(HARMFUL_SET)}
        keys = harmful_review_keys(records.values(), pairs, REVIEW_HARMFUL_PAIRS, seed=SEED)
    picks, keys = review_picks(run, keys, tag, base_tag)
    by_run = {base_run: records}
    if tag is not None:
        by_run[f"{run}-{tag}"] = {r.item.id: r for r in partial_records(run, tag)}
    entries = [
        record_entry(number, by_run[pick_run][item_id])
        for number, (pick_run, item_id) in enumerate(picks, start=1)
    ]
    title = f"partial 판정 검수표 v1 — {part} {len(entries)}문항"
    key = {
        "created_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "part": part if tag is None else f"{part}-{tag}",
        "seed": SEED,
        "base_run": base_run,
        "entries": [
            KeyEntry(number, pick_run, item_id).to_json()
            for number, (pick_run, item_id) in enumerate(picks, start=1)
        ],
    }
    key_path = write_review_sheet(path, title, entries, ITEM_GUIDE, key)
    print(
        f"검수표 {len(entries)}문항: {path}\n열쇠(문항 번호 → 기록): {key_path}\n{fill_hint(path)}"
    )


def review_picks(
    run: str, keys: Sequence[str], tag: str | None, base_tag: str | None = None
) -> tuple[list[tuple[str, str]], list[str]]:
    """검수표에 실을 (실행 이름, 문항 id) 차례와, 실제로 쓴 문항 id.

    꼬리표가 없으면 고른 문항을 그 차례대로 한 번씩. 꼬리표가 있으면 같은 문항의 기준 답과 꼬리표 답을 둘 다
    싣고 섞는다 — 어느 쪽이 새 프롬프트인지 채점자가 모르게 (열쇠 파일에만 남는다).

    꼬리표 실행에서 그 문항이 partial 로 끝나지 않았으면(route 에서는 단계가 갈릴 수 있다) 짝이 깨지므로 뺀다.
    기록 자체가 없는 것은 실행을 덜 돌린 것이라 멈춘다 — 둘을 가려서 본다.

    Returns:
      (차례, 쓴 문항 id). 꼬리표가 없으면 차례는 keys 그대로다.

    Raises:
      SystemExit: 꼬리표 기록에 아예 없는 문항이 있을 때 (answer 를 다 돌리지 않았다).
    """
    if tag is None:
        base_run = run if base_tag is None else f"{run}-{base_tag}"
        return [(base_run, key) for key in keys], list(keys)
    base_run = run if base_tag is None else f"{run}-{base_tag}"
    tagged_run = f"{run}-{tag}"
    recorded = {record.item.id for record in all_records(run, tag)}
    if missing := [key for key in keys if key not in recorded]:
        raise SystemExit(
            f"{tagged_run} 기록에 없는 문항 {missing} — answer --tag {tag} 를 먼저 다 돌린다"
        )
    answered = {record.item.id for record in partial_records(run, tag)}
    paired = [key for key in keys if key in answered]
    if dropped := [key for key in keys if key not in answered]:
        print(f"{tagged_run} 에서 partial 로 끝나지 않아 뺀 문항 {len(dropped)}: {dropped}")
    picks = [(base_run, key) for key in paired] + [(tagged_run, key) for key in paired]
    random.Random(SEED).shuffle(picks)
    return picks, paired


def judge(
    *,
    limit: int | None,
    review_only: bool = False,
    tag: str | None = None,
    run: str = "f9",
    llm: LLM | None = None,
) -> None:
    """partial 답을 판정해 판정 파일에 덧붙인다. llm 을 주지 않으면 Ollama 에 접속한다.

    tag 가 없으면 RUNS 의 꼬리표 없는 기록 전부 → JUDGED_PATH. tag 가 있으면 꼬리표 기록 `run-tag` 하나 →
    PARTIAL_JUDGED_TAGGED_PATH (그 파일의 다른 기록 줄은 그대로 둔다). 판정 지시 · 요청은 둘이 같다 (judged_row).
    """
    targets: list[tuple[str, AnswerRecord]] = []
    judged_path = JUDGED_PATH
    if tag is None:
        for name in RUNS:
            targets += [(name, record) for record in partial_records(name)]
    else:
        targets = tagged_targets(run, tag)
        judged_path = PARTIAL_JUDGED_TAGGED_PATH
    if review_only:
        keys = review_keys(sheets=REVIEW_SHEETS)
        targets = [(name, record) for name, record in targets if (name, record.item.id) in keys]
    rows = read_judged(Path(), judged_path)
    others: list[JudgedRow] = []
    if tag is not None:
        # 꼬리표 판정 파일에는 다른 기록의 줄도 있다 — 이번 기록 줄만 다시 볼 대상으로 센다
        others = [row for row in rows if row.run != f"{run}-{tag}"]
        rows = [row for row in rows if row.run == f"{run}-{tag}"]
    kept = [row for row in rows if row.error is None and _current_prompt(row)]
    if len(kept) != len(rows):
        write_jsonl(judged_path, [row.to_json() for row in [*others, *kept]])
    done = {(row.run, row.id) for row in kept}
    todo = [(name, record) for name, record in targets if (name, record.item.id) not in done][
        :limit
    ]
    failed = sum(row.error is not None for row in rows)
    print(
        f"partial 답 {len(targets)}{' (검수표 몫만)' if review_only else ''} · 판정 끝남 {len(done)} · "
        f"이번 {len(todo)} (다시 도는 실패 {failed} · 지시가 바뀐 기록 {len(rows) - len(kept) - failed}) · "
        f"{JUDGE_SPEC.spec_id} 지시 {JUDGE_PROMPT_ID}",
        flush=True,
    )
    start = time.perf_counter()
    if llm is not None:
        _judge_rows(llm, todo, judged_path, start)
    else:
        config = OllamaConfig(url=OllamaConfig.from_env().url, timeout_sec=JUDGE_TIMEOUT_SEC)
        with OllamaLLM.connect(config) as ollama:
            _judge_rows(ollama, todo, judged_path, start)
    print(f"끝 {time.perf_counter() - start:.0f}s · {judged_path}")


def _judge_rows(
    llm: LLM, todo: Sequence[tuple[str, AnswerRecord]], judged_path: Path, start: float
) -> None:
    """답을 차례로 판정해 판정 파일에 덧붙인다. 호출 실패가 이어지면 남은 답을 실패로 채우지 않고 멈춘다."""
    streak = 0
    for index, (name, record) in enumerate(todo, start=1):
        row = judged_row(name, record, llm)
        append_jsonl(judged_path, row.to_json())
        streak = streak + 1 if row.error is not None else 0
        print(
            f"  {index}/{len(todo)} {name}:{record.item.id} "
            f"{'실패' if row.error else '판정'} {time.perf_counter() - start:.0f}s",
            flush=True,
        )
        if streak >= MAX_CONSECUTIVE_ERRORS:
            raise SystemExit(f"판정이 {streak}번 이어서 실패했다 — Ollama 를 확인하고 다시 돌린다")


def _current_prompt(row: JudgedRow) -> bool:
    # 판정 지시를 고치면 예전 판정은 기준이 다르다 — 실패처럼 지우고 다시 돌린다
    return row.judge_prompt == JUDGE_PROMPT_ID


def report(*, root: Path, report_date: date | None = None) -> None:
    """partial 판정 · 유해 정보 리포트를 쓴다."""
    today = report_date or datetime.now(tz=UTC).date()
    judged = judged_by_key(root)
    records = {run: all_records(run, root=root) for run in RUNS}
    harmful_set = root / HARMFUL_SET
    human = human_labels(root, sheets=REVIEW_SHEETS)
    tagged_runs: set[str] = set()
    for labels in human.values():
        tagged_runs |= {run for run, _ in labels if run not in RUNS}
    text = render_report(
        records,
        judged,
        load_agreements(root),
        human,
        harmful_rows=read_jsonl(harmful_set) if harmful_set.exists() else [],
        today=today,
        stage_prompts={run: run_prompts(root, run) for run in sorted(tagged_runs)},
    )
    REPORT_DIR.mkdir(exist_ok=True)
    path = REPORT_DIR / f"{today:%Y-%m-%d}-partial-judge.md"
    # Windows 에서도 LF 로 (리포트는 커밋하고 .gitattributes 가 LF)
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"판정 {len(judged)} · 리포트: {path}")


if __name__ == "__main__":
    main()
