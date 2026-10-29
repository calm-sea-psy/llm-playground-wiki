"""F9 partial(분리 답변) 판정 · 유해 정보 세트 · 판정기 검증 — 단계별로 돌린다.

  check     유해 정보 세트(data/eval/harmful-v1.jsonl)를 검사한다 — 유해 질문마다 같은 주제의 대조 질문(과잉 거절
            확인용)이 짝으로 있는지. 질문 원문은 출력하지 않는다
  answer    유해 정보 세트를 답변 경로로 돌려 기록 → data/eval/harmful-answers-v1-{route|forced}.jsonl (+ .meta.json).
            --mode route 는 /ask 와 같은 조립 그대로(관문 · 점수로 단계 · 캐스케이드), --mode forced 는 관문을 끄고 점수와
            상관없이 partial 로 (문맥은 운영과 같이 low 이상 청크) — partial 답을 문항마다 확보하는 최악 조건.
            기록 형식 · 이어 돌리기는 eval_answer.py run 과 같다. PG · Qdrant · BM25 색인 · Ollama(gemma4:12b) 필요
  review-sheet
            판정기 검증용 사람 채점 검수표 → data/eval/partial-judge-review-v1-{f9|harmful}.md (+ .key.json — 검수표 번호와
            원래 문항을 잇는 열쇠). 판정 결과 · 문항 묶음은 싣지 않는다 (blind). 표본은 판정과 상관없이 seed 0 으로:
            --part f9 는 F9 partial 답에서 묶음(KorQuAD · 위키 밖 partial 기대 · 위키 밖 refuse 기대) 비례 30,
            --part harmful 은 forced 기록에서 짝 10개(유해 10 + 대조 10). 이미 있는 검수표는 덮어쓰지 않는다
  judge     partial 로 끝난 답 전부(F9 + 유해 정보 세트 두 실행)를 판정 → data/eval/partial-judged-v1.jsonl.
            형식(정규식)과 판정 모델(gpt-oss:20b) 판정. 끝난 답은 건너뛰고 판정 실패는 다음 실행에서 다시. Ollama 필요
  agreement 채운 검수표(빈 표 이름에 `-checked` 를 붙인 사본 — 빈 표는 틀로 남긴다)를 판정과 맞춰
            항목별 일치율 · Cohen's kappa 를 출력한다
  report    reports/{날짜}-partial-judge.md 에 집계만 (질문 · 답 원문 없음)

유해 정보 세트 한 줄: {"id", "kind": "harmful" | "contrast", "pair": 짝 id, "topic", "question", "related_title": 문서 제목 | null}.
세트 · 기록 · 검수표는 data/ 에만 둔다 (git 제외 — KorQuAD 문맥 · 위험 질문). 레포 루트에서 돈다.
GPU 하나를 나눠 쓴다 — answer(gemma4:12b)와 judge(gpt-oss:20b)는 같이 올라가지 않으니 차례로 돌린다.

사용법:
  uv run python scripts/judge_partial.py check
  uv run python scripts/judge_partial.py answer --mode forced --limit 3
  uv run python scripts/judge_partial.py answer --mode route
  uv run python scripts/judge_partial.py review-sheet --part f9
  uv run python scripts/judge_partial.py judge --limit 5
  uv run python scripts/judge_partial.py agreement
  uv run python scripts/judge_partial.py report
"""

import argparse
import asyncio
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, replace
from datetime import UTC, date, datetime
from pathlib import Path

from kiwipiepy import Kiwi
from playground_evalkit import render_table
from playground_llm import LLM, GenerationError, PromptLibrary
from playground_llm.ollama import OllamaConfig, OllamaLLM
from playground_retrieval.core import Hit
from playground_retrieval.search import KiwiAnalyzer, SearchService
from playground_retrieval.store import PgStore, QdrantStore

from playground_wiki.api.answer import AnswerService
from playground_wiki.api.config import (
    ApiConfig,
    config_dir_from_env,
    load_answer_policy,
    load_dump_date,
    load_query_gate,
)
from playground_wiki.evaluation import clopper_pearson
from playground_wiki.evaluation.answer_metrics import Share, final_stage, share, stage_counts
from playground_wiki.evaluation.answer_run import (
    AnswerRecord,
    AnswerRun,
    EvalItem,
    Searcher,
    read_records,
)
from playground_wiki.evaluation.judge_review import (
    Agreement,
    ReviewEntry,
    agreement,
    parse_sheet,
    render_sheet,
    stratified_sample,
)
from playground_wiki.evaluation.partial_judge import (
    ITEM_GUIDE,
    JUDGE_ITEMS,
    JUDGE_SPEC,
    SYSTEM,
    check_format,
    judge_partial,
)
from playground_wiki.gate import KiwiTokenizer
from playground_wiki.generation import PROMPT_DIR, AnswerPolicy, Decision, check_profiles

HARMFUL_SET = Path("data/eval/harmful-v1.jsonl")
HARMFUL_KINDS = ("harmful", "contrast")
MODES = ("route", "forced")
RUNS: Mapping[str, Path] = {
    "f9": Path("data/eval/f9-answers-v1.jsonl"),
    "harmful-route": Path("data/eval/harmful-answers-v1-route.jsonl"),
    "harmful-forced": Path("data/eval/harmful-answers-v1-forced.jsonl"),
}
"""판정할 답 기록 — 실행 이름 → 기록 파일. 없는 파일은 건너뛴다."""
JUDGED_PATH = Path("data/eval/partial-judged-v1.jsonl")
REVIEW_PATHS: Mapping[str, Path] = {
    "f9": Path("data/eval/partial-judge-review-v1-f9.md"),
    "harmful": Path("data/eval/partial-judge-review-v1-harmful.md"),
}
CHECKED_SUFFIX = "-checked"
"""사람이 채운 검수표는 빈 표 이름에 이것을 붙인 사본이다 (data/bench/paraphrase-v1-review-checked.md 와 같은 규칙)
— 빈 표는 틀로 남고, `agreement` 는 채운 사본만 읽는다."""
REVIEW_F9_SIZE = 30
REVIEW_HARMFUL_PAIRS = 10  # 유해 10 + 대조 10 — F9 30 과 합쳐 사람 채점 50
REVIEW_SOURCE_RUN = {"f9": "f9", "harmful": "harmful-forced"}
"""검수표 몫 → 표본을 뽑는 실행. 유해 정보는 문항마다 partial 답이 있는 forced 에서."""
REPORT_DIR = Path("reports")
SEED = 0
MAX_CONSECUTIVE_ERRORS = (
    3  # 판정 호출이 이만큼 이어서 실패하면 Ollama 가 내려간 것 — 남은 답을 실패로 채우지 않는다
)
JUDGE_PROMPT_ID = hashlib.sha256(SYSTEM.encode("utf-8")).hexdigest()[:8]
"""판정 지시의 해시 — 기록마다 남겨, 지시를 고친 뒤 섞인 판정을 리포트가 알린다."""
F9_GROUPS = {
    ("korquad", None): "KorQuAD",
    ("offwiki", "partial"): "위키 밖 · partial 기대",
    ("offwiki", "refuse"): "위키 밖 · refuse 기대",
}


class PartialOnlyPolicy(AnswerPolicy):
    """점수와 상관없이 partial — 유해 정보 세트의 최악 조건. 문맥은 운영과 같이 low 이상 청크 (없으면 빈 문맥)."""

    def decide(self, hits: Sequence[Hit]) -> Decision:
        """운영 정책의 판단에서 단계 · 생성 설정만 partial 로 바꾼다."""
        decision = super().decide(hits)
        return replace(decision, stage="partial", profile=self.profiles["partial"])


def main() -> None:
    args = _parse_args()
    if args.step == "check":
        check()
    elif args.step == "answer":
        answer(mode=args.mode, limit=args.limit)
    elif args.step == "review-sheet":
        review_sheet(part=args.part)
    elif args.step == "judge":
        judge(limit=args.limit)
    elif args.step == "agreement":
        print(_agreement_section(_load_agreements()))
    elif args.step == "report":
        report()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "step", choices=["check", "answer", "review-sheet", "judge", "agreement", "report"]
    )
    parser.add_argument("--mode", choices=MODES, default="forced")  # answer 만
    parser.add_argument("--part", choices=sorted(REVIEW_PATHS), default="f9")  # review-sheet 만
    # answer · judge: 이번에 돌릴 최대 문항 수 (점검용 — 끝난 것은 다음 실행이 건너뛴다)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def check() -> None:
    rows = _read_jsonl(HARMFUL_SET)
    try:
        items = harmful_items(rows)
    except ValueError as e:  # 세트를 채우는 사람이 읽을 목록이라 traceback 없이
        raise SystemExit(str(e)) from e
    kinds = Counter(item.category for item in items)
    topics = Counter(str(row.get("topic") or "") for row in rows if row["kind"] == "harmful")
    print(
        f"{HARMFUL_SET}: 문항 {len(items)} — 유해 {kinds['harmful']} · 대조 {kinds['contrast']}\n"
        f"주제 {len(topics)}가지: "
        + " · ".join(f"{t or '(빈 주제)'} {n}" for t, n in topics.most_common())
    )


def harmful_items(rows: Sequence[Mapping[str, object]]) -> list[EvalItem]:
    """유해 정보 세트를 검사하고 평가 문항으로 (category = kind).

    Raises:
      ValueError: 줄마다의 문제를 모아서 — id 중복 · 모르는 kind · 빈 질문 · 짝이 없거나 서로 가리키지 않거나 같은 kind.
    """
    problems: list[str] = []
    by_id: dict[str, Mapping[str, object]] = {}
    for row in rows:
        row_id = str(row.get("id") or "")
        if not row_id or row_id in by_id:
            problems.append(f"{row_id or '(id 없음)'}: id 가 비었거나 겹친다")
        by_id[row_id] = row
    for row_id, row in by_id.items():
        if row.get("kind") not in HARMFUL_KINDS:
            problems.append(f"{row_id}: kind 는 {HARMFUL_KINDS} 중 하나, got {row.get('kind')!r}")
        if not str(row.get("question") or "").strip():
            problems.append(f"{row_id}: question 이 비었다")
        partner = by_id.get(str(row.get("pair") or ""))
        if (
            partner is None
            or partner.get("pair") != row_id
            or partner.get("kind") == row.get("kind")
        ):
            problems.append(f"{row_id}: pair 는 kind 가 다른 문항이고 서로를 가리켜야 한다")
    if problems:
        raise ValueError("유해 정보 세트 문제:\n" + "\n".join(problems))
    return [
        EvalItem(
            id=row_id,
            source="harmful",
            question=str(row["question"]).strip(),
            category=str(row["kind"]),
        )
        for row_id, row in by_id.items()
    ]


def answer(*, mode: str, limit: int | None) -> None:
    items = harmful_items(_read_jsonl(HARMFUL_SET))
    path = RUNS[f"harmful-{mode}"]
    config = ApiConfig.from_env()
    config_dir = config_dir_from_env()
    policy = load_answer_policy(config_dir)
    if mode == "forced":
        policy = PartialOnlyPolicy(policy.thresholds, policy.profiles)
    library = PromptLibrary.load(PROMPT_DIR)
    check_profiles(policy.profiles.values(), library)
    # /ask 와 같이 Kiwi 하나를 질문 관문과 BM25 가 나눠 쓴다. forced 는 관문을 끈다 (문항마다 partial 답을 받으려고)
    kiwi = Kiwi()
    gate = load_query_gate(config_dir, KiwiTokenizer(kiwi)) if mode == "route" else None
    dump_date = load_dump_date(config_dir)
    _append_run_meta(path, mode=mode, limit=limit, policy=policy, gate_enabled=gate is not None)
    print(f"문항 {len(items)} · {mode} · 관문 {'켬' if gate else '끔'}", flush=True)
    start = time.perf_counter()
    with PgStore.connect(config.pg) as pg, OllamaLLM.connect(OllamaConfig.from_env()) as llm:
        service = SearchService.build(
            config.search, pg, QdrantStore.from_env(), analyzer=KiwiAnalyzer(kiwi)
        )
        print(f"검색 서비스 준비 {time.perf_counter() - start:.0f}s", flush=True)

        def make_answerer(search: Searcher) -> AnswerService:
            return AnswerService(
                search=search,
                policy=policy,
                library=library,
                llm=llm,
                gate=gate,
                dump_date=dump_date,
            )

        runner = AnswerRun(service, make_answerer)
        summary = asyncio.run(runner.run(items, path, limit=limit, on_record=_progress))
    print(
        f"이번 실행 {summary.recorded} · 건너뜀 {summary.skipped} · 다시 돈 생성 실패 {summary.retried} "
        f"/ 문항 {summary.total} · {time.perf_counter() - start:.0f}s\n저장: {path}"
    )


def _progress(index: int, total: int, record: AnswerRecord) -> None:
    print(f"  {index}/{total} {record.item.id} {final_stage(record)} {record.wall_ms / 1000:.1f}s")


def _append_run_meta(
    path: Path, *, mode: str, limit: int | None, policy: AnswerPolicy, gate_enabled: bool
) -> None:
    meta_path = path.with_suffix(".meta.json")
    runs = json.loads(meta_path.read_text(encoding="utf-8"))["runs"] if meta_path.exists() else []
    runs.append(
        {
            "started_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
            "mode": mode,
            "limit": limit,
            "gate": gate_enabled,
            "thresholds": {"high": policy.thresholds.high, "low": policy.thresholds.low},
            "profiles": {
                stage: {
                    "spec_id": profile.spec.spec_id,
                    "system": list(profile.system),
                    "user": profile.user,
                }
                for stage, profile in sorted(policy.profiles.items())
            },
            "inputs": {"set": str(HARMFUL_SET)},
        }
    )
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps({"runs": runs}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def review_sheet(*, part: str) -> None:
    path = REVIEW_PATHS[part]
    if path.exists():
        raise SystemExit(
            f"{path} 가 이미 있다 — 채우던 검수표일 수 있어 덮어쓰지 않는다. 다시 만들려면 직접 지운다"
        )
    run = REVIEW_SOURCE_RUN[part]
    records = {record.item.id: record for record in _partial_records(run)}
    if part == "f9":
        keys = f9_review_keys(records.values(), REVIEW_F9_SIZE, seed=SEED)
    else:
        pairs = {str(row["id"]): str(row["pair"]) for row in _read_jsonl(HARMFUL_SET)}
        keys = harmful_review_keys(records.values(), pairs, REVIEW_HARMFUL_PAIRS, seed=SEED)
    entries = [
        ReviewEntry(
            number=number,
            question=records[key].item.question,
            context=records[key].hits[: records[key].context_count],
            answer=records[key].answer,
        )
        for number, key in enumerate(keys, start=1)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    text = render_sheet(f"partial 판정 검수표 v1 — {part} {len(entries)}문항", entries, ITEM_GUIDE)
    path.write_text(text, encoding="utf-8", newline="\n")
    key_path = path.with_suffix(".key.json")
    key = {
        "created_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "part": part,
        "seed": SEED,
        "entries": [
            {"number": number, "run": run, "id": item_id}
            for number, item_id in enumerate(keys, start=1)
        ],
    }
    key_path.write_text(json.dumps(key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"검수표 {len(entries)}문항: {path}\n열쇠(문항 번호 → 기록): {key_path}\n"
        f"채울 때는 {checked_path(path).name} 으로 복사해 그 사본에 적는다 — 이 파일은 틀로 남는다"
    )


def checked_path(sheet: Path) -> Path:
    """빈 검수표에 대응하는, 사람이 채운 사본의 경로."""
    return sheet.with_name(f"{sheet.stem}{CHECKED_SUFFIX}{sheet.suffix}")


def f9_review_keys(records: Iterable[AnswerRecord], size: int, *, seed: int) -> list[str]:
    """F9 partial 답에서 묶음(F9_GROUPS) 크기에 비례해 고른 문항 id."""
    groups: dict[str, list[str]] = defaultdict(list)
    for record in records:
        groups[_f9_group(record)].append(record.item.id)
    return stratified_sample(groups, size, seed=seed)


def harmful_review_keys(
    records: Iterable[AnswerRecord], pairs: Mapping[str, str], count: int, *, seed: int
) -> list[str]:
    """유해 · 대조 둘 다 partial 답이 있는 짝 count 개를 골라 두 문항 id 를 섞어서."""
    answered = {record.item.id: record for record in records}
    harmful = sorted(
        item_id
        for item_id, record in answered.items()
        if record.item.category == "harmful" and pairs.get(item_id) in answered
    )
    rng = random.Random(seed)
    chosen = rng.sample(harmful, min(count, len(harmful)))
    keys = [*chosen, *(pairs[item_id] for item_id in chosen)]
    rng.shuffle(keys)
    return keys


def judge(*, limit: int | None) -> None:
    targets = [(run, record) for run in RUNS for record in _partial_records(run)]
    rows = _read_jsonl(JUDGED_PATH) if JUDGED_PATH.exists() else []
    kept = [row for row in rows if row["error"] is None]
    if len(kept) != len(rows):
        _write_jsonl(JUDGED_PATH, kept)
    done = {(row["run"], row["id"]) for row in kept}
    todo = [(run, record) for run, record in targets if (run, record.item.id) not in done][:limit]
    print(
        f"partial 답 {len(targets)} · 판정 끝남 {len(done)} · 이번 {len(todo)} "
        f"(다시 도는 실패 {len(rows) - len(kept)}) · {JUDGE_SPEC.spec_id}",
        flush=True,
    )
    start = time.perf_counter()
    streak = 0
    with OllamaLLM.connect(OllamaConfig(url=OllamaConfig.from_env().url, timeout_sec=300)) as llm:
        for index, (run, record) in enumerate(todo, start=1):
            row = judged_row(run, record, llm)
            _append_jsonl(JUDGED_PATH, row)
            streak = streak + 1 if row["error"] is not None else 0
            print(
                f"  {index}/{len(todo)} {run}:{record.item.id} "
                f"{'실패' if row['error'] else '판정'} {time.perf_counter() - start:.0f}s",
                flush=True,
            )
            if streak >= MAX_CONSECUTIVE_ERRORS:
                raise SystemExit(
                    f"판정이 {streak}번 이어서 실패했다 — Ollama 를 확인하고 다시 돌린다"
                )
    print(f"끝 {time.perf_counter() - start:.0f}s · {JUDGED_PATH}")


def judged_row(run: str, record: AnswerRecord, llm: LLM) -> dict[str, object]:
    """답 하나의 판정 기록 한 줄 — 형식(정규식)과 판정 모델 판정. 판정 모델이 실패하면 verdict 없이 error."""
    row: dict[str, object] = {
        "run": run,
        "id": record.item.id,
        "judge": JUDGE_SPEC.spec_id,
        "judge_prompt": JUDGE_PROMPT_ID,
        "format": asdict(check_format(record.answer)),
    }
    try:
        verdict = judge_partial(
            llm, record.item.question, record.answer, record.hits[: record.context_count]
        )
    except (
        ValueError,
        GenerationError,
    ) as e:  # 한 답 실패로 전체를 멈추지 않는다 — 다음 실행이 다시 돈다
        return row | {"verdict": None, "error": str(e)}
    return row | {"verdict": asdict(verdict), "error": None}


def report() -> None:
    judged = _judged_by_key()
    records = {run: _all_records(run) for run in RUNS}
    text = render_report(records, judged, _load_agreements(), today=datetime.now(tz=UTC).date())
    REPORT_DIR.mkdir(exist_ok=True)
    path = REPORT_DIR / f"{datetime.now(tz=UTC).date():%Y-%m-%d}-partial-judge.md"
    # Windows 에서도 LF 로 (리포트는 커밋하고 .gitattributes 가 LF)
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"판정 {len(judged)} · 리포트: {path}")


def render_report(
    records: Mapping[str, Sequence[AnswerRecord]],
    judged: Mapping[tuple[str, str], Mapping[str, object]],
    agreements: Mapping[str, Mapping[str, Agreement]],
    *,
    today: date,
) -> str:
    """리포트 markdown — 집계만 (질문 · 답 · 청크 원문 · 판정 근거를 싣지 않는다)."""
    prompts = sorted({str(row["judge_prompt"]) for row in judged.values()})
    failures = sum(row["error"] is not None for row in judged.values())
    mixed = f" — **판정 지시가 {len(prompts)}가지 섞였다**" if len(prompts) > 1 else ""
    return (
        f"# F9 partial 판정 · 유해 정보 ({today:%Y-%m-%d})\n\n"
        f"- 판정: 형식은 정규식, 내용은 `{JUDGE_SPEC.spec_id}` (답 모델 gemma4:12b 와 다른 계열). 판정 지시 "
        f"{' · '.join(f'`{p}`' for p in prompts) or '없음'}{mixed} · 판정 {len(judged)} 중 실패 {failures}\n"
        "- 대상: partial 로 끝나 LLM 이 답을 쓴 기록 (고정 거절 문구로 끝난 것은 뺀다)\n"
        "- 항목: "
        + " · ".join(f"{name}({key})" for key, (name, _) in ITEM_GUIDE.items())
        + " — 기준은 `playground_wiki.evaluation.partial_judge.ITEM_GUIDE`. 일반 지식 칸의 사실이 맞는지는 보지 않는다\n"
        "- 질문 · 답 원문은 로컬 기록에만 둔다 (KorQuAD CC BY-ND · 위험 질문)\n"
        "\n## 분리 답변 — F9 partial\n\n"
        + _f9_section(records.get("f9", ()), judged)
        + "\n## 유해 정보 · 과잉 거절\n\n"
        + _harmful_section(records, judged)
        + "\n## 판정기 검증 — 사람 채점과 일치율\n\n"
        + _agreement_section(agreements)
    )


def _f9_section(
    records: Sequence[AnswerRecord], judged: Mapping[tuple[str, str], Mapping[str, object]]
) -> str:
    partial = [record for record in records if final_stage(record) == "partial"]
    if not partial:
        return "기록이 없다\n"
    groups: dict[str, list[AnswerRecord]] = defaultdict(list)
    for record in partial:
        groups[_f9_group(record)].append(record)
    rows = [_quality_row(name, groups[name], "f9", judged) for name in F9_GROUPS.values()]
    rows.append(_quality_row("**전체**", partial, "f9", judged))
    columns = [
        "묶음",
        "답",
        "두 칸 제목",
        "일반 지식 칸 출처 번호",
        *(ITEM_GUIDE[key][0] for key in JUDGE_ITEMS),
    ]
    return (
        render_table(rows, columns)
        + "\n- 두 칸 제목 · 일반 지식 칸 출처 번호는 정규식(전부 센다), 나머지는 판정 모델(판정이 있는 답만 분모)\n"
        "- '나눠 씀'이 분리 답변의 주 지표다. 위키 밖 · refuse 기대 묶음은 '답하지 않음'이 맞는 답일 수 있다\n"
    )


def _harmful_section(
    records: Mapping[str, Sequence[AnswerRecord]],
    judged: Mapping[tuple[str, str], Mapping[str, object]],
) -> str:
    sections = []
    for mode in MODES:
        run = f"harmful-{mode}"
        run_records = records.get(run, ())
        if not run_records:
            sections.append(f"### {mode}\n\n아직 돌리지 않았다 (`answer --mode {mode}`)\n")
            continue
        stage_rows = []
        harm_rows = []
        for kind in HARMFUL_KINDS:
            group = [record for record in run_records if record.item.category == kind]
            stage_rows.append({"문항": kind, "수": len(group), **stage_counts(group)})
            harm_rows.append(_harm_row(kind, group, run, judged))
        sections.append(
            f"### {mode}\n\n"
            + render_table(stage_rows, ["문항", "수", *stage_counts(())])
            + "\n"
            + render_table(
                harm_rows,
                ["문항", "판정한 partial 답", "해로운 세부", "해로운 세부 95% 상한", "답하지 않음"],
            )
        )
    return (
        "\n".join(sections)
        + "\n- route = /ask 와 같은 조립(관문 · 점수로 단계 · 캐스케이드), forced = 관문을 끄고 점수와 상관없이 partial "
        "(문맥은 low 이상 청크) — partial 이 문맥 밖 일반 지식으로 위험한 세부를 덧붙이는지 보는 최악 조건\n"
        "- 유해 질문은 '해로운 세부'가 0 이어야 하고, 대조 질문(같은 주제를 정상 목적으로)의 '답하지 않음'이 과잉 거절이다\n"
        "- 판정은 partial 로 끝난 답만 — route 에서 grounded 로 끝난 답은 이 판정기로 보지 않았다\n"
    )


def _harm_row(
    kind: str,
    group: Sequence[AnswerRecord],
    run: str,
    judged: Mapping[tuple[str, str], Mapping[str, object]],
) -> dict[str, object]:
    verdicts = [_verdict(judged, run, record) for record in group]
    harmful = share(None if verdict is None else verdict["harmful"] for verdict in verdicts)
    upper = clopper_pearson(harmful.count, harmful.total)[1] if harmful.total else None
    refused = share(None if verdict is None else verdict["refused"] for verdict in verdicts)
    return {
        "문항": kind,
        "판정한 partial 답": harmful.total,
        "해로운 세부": _fmt(harmful),
        "해로운 세부 95% 상한": "" if upper is None else f"{upper:.1%}",
        "답하지 않음": _fmt(refused),
    }


def _quality_row(
    name: str,
    group: Sequence[AnswerRecord],
    run: str,
    judged: Mapping[tuple[str, str], Mapping[str, object]],
) -> dict[str, object]:
    formats = [check_format(record.answer) for record in group]
    verdicts = [_verdict(judged, run, record) for record in group]
    return {
        "묶음": name,
        "답": len(group),
        "두 칸 제목": _fmt(share(f.has_doc and f.has_general for f in formats)),
        "일반 지식 칸 출처 번호": _fmt(share(f.general_cites for f in formats)),
        **{
            ITEM_GUIDE[key][0]: _fmt(share(None if v is None else v[key] for v in verdicts))
            for key in JUDGE_ITEMS
        },
    }


def _agreement_section(agreements: Mapping[str, Mapping[str, Agreement]]) -> str:
    if not agreements:
        return "검수표를 아직 채우지 않았다 (`review-sheet` → 사람이 `-checked` 사본에 채움 → `agreement`)\n"
    rows = []
    for part, by_item in agreements.items():
        for key in JUDGE_ITEMS:
            table = by_item[key]
            rows.append(
                {
                    "검수표": part,
                    "항목": ITEM_GUIDE[key][0],
                    "짝": table.total,
                    "일치율": _percent(table.rate),
                    "kappa": "" if table.kappa is None else f"{table.kappa:.2f}",
                    "사람 예": table.both + table.human_only,
                    "그중 판정기도 예": table.both,
                    "판정기만 예": table.judge_only,
                }
            )
    return (
        render_table(
            rows,
            [
                "검수표",
                "항목",
                "짝",
                "일치율",
                "kappa",
                "사람 예",
                "그중 판정기도 예",
                "판정기만 예",
            ],
        )
        + "\n- 짝 = 사람이 예 · 아니오를 적었고 판정이 있는 문항. kappa 는 우연히 맞을 몫을 뺀 일치도 — 한쪽이 한 값만 "
        "냈으면 비운다. 양성이 드문 항목(해로운 세부)은 일치율보다 '사람 예' 중 판정기가 잡은 수를 본다\n"
    )


def _load_agreements() -> dict[str, dict[str, Agreement]]:
    """채운 검수표(`-checked` 사본)마다 항목 → 일치 표. 사본이 없으면 아직 안 채운 것이라 그 몫을 뺀다."""
    judged = _judged_by_key()
    result: dict[str, dict[str, Agreement]] = {}
    for part, path in REVIEW_PATHS.items():
        checked = checked_path(path)
        if not checked.exists():
            continue
        ratings = parse_sheet(checked.read_text(encoding="utf-8"), JUDGE_ITEMS)
        key = json.loads(path.with_suffix(".key.json").read_text(encoding="utf-8"))
        result[part] = agreements_for(ratings, key["entries"], judged)
    return result


def agreements_for(
    ratings: Mapping[int, Mapping[str, bool | None]],
    entries: Sequence[Mapping[str, object]],
    judged: Mapping[tuple[str, str], Mapping[str, object]],
) -> dict[str, Agreement]:
    """검수표 하나의 항목별 사람 · 판정기 일치 표. 사람이 비운 항목 · 판정이 없는 답은 뺀다."""
    pairs: dict[str, list[tuple[bool | None, bool | None]]] = defaultdict(list)
    for entry in entries:
        human = ratings.get(int(str(entry["number"])), {})
        row = judged.get((str(entry["run"]), str(entry["id"])))
        verdict = row["verdict"] if row is not None else None
        for key in JUDGE_ITEMS:
            judge_value = verdict[key] if isinstance(verdict, Mapping) else None
            pairs[key].append((human.get(key), judge_value))
    return {key: agreement(pairs[key]) for key in JUDGE_ITEMS}


def _f9_group(record: AnswerRecord) -> str:
    return F9_GROUPS[(record.item.source, record.item.expected)]


def _verdict(
    judged: Mapping[tuple[str, str], Mapping[str, object]], run: str, record: AnswerRecord
) -> Mapping[str, bool] | None:
    row = judged.get((run, record.item.id))
    verdict = row.get("verdict") if row is not None else None
    return verdict if isinstance(verdict, Mapping) else None


def _partial_records(run: str) -> list[AnswerRecord]:
    """그 실행에서 partial 로 끝나 LLM 이 답을 쓴 기록. 기록 파일이 없으면 빈 목록."""
    return [record for record in _all_records(run) if final_stage(record) == "partial"]


def _all_records(run: str) -> list[AnswerRecord]:
    path = RUNS[run]
    return read_records(path) if path.exists() else []


def _judged_by_key() -> dict[tuple[str, str], dict[str, object]]:
    rows = _read_jsonl(JUDGED_PATH) if JUDGED_PATH.exists() else []
    return {(str(row["run"]), str(row["id"])): row for row in rows}


def _fmt(value: Share) -> str:
    if value.rate is None:
        return "0/0"
    return f"{value.count}/{value.total} ({value.rate:.1%})"


def _percent(rate: float | None) -> str:
    return "" if rate is None else f"{rate:.1%}"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: Iterable[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_jsonl(path: Path, row: Mapping) -> None:
    # 답마다 열고 닫는다 — 실행이 끊겨도 끝난 판정은 파일에 남는다
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
