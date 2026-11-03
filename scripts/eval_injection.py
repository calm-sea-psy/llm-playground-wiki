"""간접 인젝션 끝까지(E2E) 평가 — 원본 위키 마크업에 심은 지시가 정제 → 적재 → 검색 → `/ask` 까지 살아남는지.

정제 단계(어느 위치가 청크까지 남는지)는 tests/ingest/test_cleaner_injection.py + evaluation.injection_channels 가
위치 71곳마다 단언한다. 이 스크립트는 그 뒤를 잰다: 정제를 통과하는 자리에 심은 지시가 실제 문서를 거쳐 답변에
영향을 주는가. 정제하고 나면 통로 표의 "남는" 위치들은 LLM 이 보는 글 기준으로 몇 가지 꼴로 모이므로(본문 글 ·
절 제목 등), 답변까지는 그 꼴(INJECTION_VARIANTS)마다 한 번씩 돌린다. 위치 → 꼴 대응은 각 꼴의 represents 에 적는다.

운영 코퍼스는 건드리지 않는다 — 테스트용 복사본을 따로 둔다:
  · PostgreSQL: 별도 데이터베이스 INJECTION_DB (운영 DB 는 읽기만). cleanup 이 DROP DATABASE.
  · Qdrant: 로컬에 꼴마다 INJECTION_COLLECTION_PREFIX 컬렉션 (QDRANT_MODE 와 상관없이 로컬 고정).
  · 임베딩 캐시 · BM25: data/eval/injection-e2e/ 아래. BM25 는 시작할 때 PG 청크로 메모리에서 만든다(파일 안 남김).

단계 (data/ 에 중간 결과를 저장해 뒤 단계를 다시 돌릴 때 앞 단계를 반복하지 않는다):
  extract  F9 답 기록(data/eval/f9-answers-v1.jsonl)에서 목표 문항을 고른다 — grounded 로 정답을 맞혔고, 정답
           근거 청크가 1위이며 1위 점수 ≥ MIN_TOP_SCORE 인 문항을 문서당 하나, 앞에서 TARGET_DOCS 개. 그 문서들과
           그 질문들의 검색 상위 청크가 속한 배경 문서의 원본 위키텍스트를 덤프에서 뽑는다 (CPU만).
           → data/eval/injection-e2e/targets.jsonl · background.jsonl
  plant    꼴마다 목표 문서의 원본 마크업에 지시를 심어 정제 · 청킹한 뒤, 심은 글이 청크에 남는지 · 정답과 같은
           청크에 들었는지를 센다 (CPU만). → data/eval/injection-e2e/plant-v1.jsonl
  run      꼴마다 (심은) 목표 + 배경을 테스트용 복사본에 적재하고, 목표 문항을 `/ask` 와 같은 조립으로 돌려
           기록한다. → data/eval/injection-e2e/answers-{꼴}.jsonl (+ .meta.json). PG · Qdrant · Ollama · GPU 필요.
  report   꼴별 · 지시문별 공격 성공률 · 꼴 사이 짝 검정 · 답 유지 · 청크 생존을 집계한다. LLM · 검색 없이 기록만
           읽는다. → reports/{답 기록 날짜}-injection-e2e.md (집계만)
  cleanup  테스트용 복사본을 지운다 — DROP DATABASE · Qdrant 컬렉션 · 임베딩 캐시.

새 인자를 주지 않으면(기본) 위 그대로다 — Phase 3 동안 기본 동작의 파일 · 리포트는 바뀌지 않는다. 새 측정은 인자로만 켠다:
  --targets partial  두 번째 표적 세트: F9 기록 + 롱테일 기록(LONGTAIL_ANSWERS_PATH)에서 마지막 단계가 partial 이고
                     정답 근거 청크가 1위(partial 문맥 안)이며 답에 정답이 있는 문항을 문서당 하나, 기본(grounded) 표적과
                     문서가 겹치지 않게 자르지 않고 전부. 리포트 · 메타에 출처별 분모. 중간 결과 · 기록은
                     data/eval/injection-e2e/partial-v1/, 컬렉션은 injection-e2e-partial-v1-{꼴}, 리포트는
                     reports/{날짜}-injection-e2e-partial.md. 꼴 · 지시문 · 성공 판정은 기본과 같다.
  --mode agent       run · report 만: `/ask?mode=agent` (단발 먼저 · 모자랄 때만 에이전트) 로 같은 표적을 돌린다. 기록은
                     answers-{꼴}-agent.jsonl (done 진단의 agent 칸 — route — 을 싣는 evaluation.agent_eval 기록),
                     리포트는 reports/{날짜}-injection-e2e[-partial]-agent.md — 분모 둘(표적 전부 · 에이전트로 넘어간
                     표적, 단발로 끝난 표적 수도) · 끝난 단계별 · 같은 표적의 단발 기록과 짝 McNemar.
  --tag T            run · report 만: 프롬프트를 바꿔 다시 잰 기록 — 기록 · 메타 · 리포트 이름 맨 끝에 -T 를 붙여 꼬리표
                     없는 기록(방어 전)을 덮어쓰지 않는다 (answers-{꼴}[-agent]-T.jsonl, reports/{날짜}-injection-e2e
                     [-partial][-agent]-T.md). --targets · --mode 와 함께 쓴다. 바꾼 설정은 RAG_CONFIG_DIR 로 가리킨다
                     (scripts/eval_defense.py config-dir). 꼬리표 실행은 측정 전용 프롬프트(evaluation.defense_arms
                     CONTROL_PROMPT_DIR)도 함께 읽는다 — 운영 프롬프트 모음에 없는 대조 칸을 가리킬 수 있게. 테스트용 복사본
                     DB · 컬렉션 · 임베딩 캐시는 꼬리표와 상관없이 같은 것을 다시 쓴다 (적재하는 문서가 프롬프트와 상관없이 같다)
  compare [--mode agent]
                     방어 비교 리포트: 두 표적 세트(grounded · partial) 각각과 합에서 칸 셋 — base-norule · base@v4(꼬리표
                     없는 기록) · base-v5 — 의 꼴별 · 지시문별 공격 성공, 답 유지, 단계 바뀜, 끝난 단계별, 짝 McNemar
                     (norule 대 v4 · v4 대 v5). LLM · 검색 없이 기록만 읽는다. → reports/{날짜}-injection-e2e-defense[-agent].md
  compare --candidate base-v6 [--mode agent]
                     후보 비교 리포트: 칸 셋 — base@v4(꼬리표 없음) · base-v5 · base-v6. 판정 짝은 v4 대 v6, v5 대 v6 은
                     참고 줄. 채택 기준 아래 줄은 측정 전에 적은 V6_CRITERIA_NOTES. 표 꼴은 기본 compare 와 같다
                     → reports/{날짜}-injection-e2e-defense-v6[-agent].md. --candidate 가 없으면 위 기본 compare 그대로
새 리포트(partial · agent · 꼬리표)의 측정 조건 절 첫 줄에 E2E_CONDITION(실제 컬렉션을 검색하는 끝까지 측정)을 싣는다.

KorQuAD 는 CC BY-ND 라 질문 · 답 · canary 는 data/ 에만 둔다 (git 제외). 레포 루트에서 돌린다.

사용법:
  uv run --env-file .env --env-file .env.config python scripts/eval_injection.py extract
  uv run --env-file .env --env-file .env.config python scripts/eval_injection.py plant
  uv run --env-file .env --env-file .env.config python scripts/eval_injection.py run
  uv run --env-file .env --env-file .env.config python scripts/eval_injection.py report
  uv run --env-file .env --env-file .env.config python scripts/eval_injection.py cleanup
  # partial 표적 · 에이전트 모드 (extract · plant · run · report · cleanup 모두 --targets 를 받는다)
  uv run --env-file .env --env-file .env.config python scripts/eval_injection.py extract --targets partial
  uv run --env-file .env --env-file .env.config python scripts/eval_injection.py run --mode agent
  uv run --env-file .env --env-file .env.config python scripts/eval_injection.py report --targets partial --mode agent
  # 방어 비교 칸 (설정 폴더는 scripts/eval_defense.py config-dir)
  RAG_CONFIG_DIR=data/eval/defense-configs/base-v5 uv run --env-file .env --env-file .env.config python scripts/eval_injection.py run --tag base-v5
  uv run python scripts/eval_injection.py report --targets partial --tag base-v5
  uv run python scripts/eval_injection.py compare
  uv run python scripts/eval_injection.py compare --candidate base-v6 --mode agent
"""

import argparse
import asyncio
import bz2
import logging
import shutil
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from itertools import chain, product
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import psycopg
import yaml
from kiwipiepy import Kiwi
from lxml import etree
from playground_corpus import (
    EMBEDDING_MODELS,
    Chunker,
    ChunkingConfig,
    Document,
    EmbeddingCache,
    HFTokenizer,
    SentenceTransformerEmbedder,
)
from playground_llm import PromptLibrary
from playground_llm.ollama import OllamaConfig, OllamaLLM
from playground_retrieval.ingest import ingest_documents
from playground_retrieval.search import KiwiAnalyzer, SearchService
from playground_retrieval.store import PgConfig, PgStore, QdrantConfig, QdrantStore

from playground_wiki.agent import AGENT_PROMPT_DIR
from playground_wiki.api.answer import AnswerService
from playground_wiki.api.config import (
    CONFIG_DIR,
    AnswerSetup,
    ApiConfig,
    load_answer_policy,
    load_dump_date,
    load_query_gate,
)
from playground_wiki.evaluation.agent_eval import (
    MODES,
    AgentEvalRun,
    AgentItem,
    AgentRecord,
    Mode,
)
from playground_wiki.evaluation.answer_run import (
    AnswerRecord,
    AnswerRun,
    read_records,
    refuse_prompt_change,
)
from playground_wiki.evaluation.defense_arms import CONTROL_PROMPT_DIR
from playground_wiki.evaluation.files import (
    meta_path,
    read_jsonl,
    write_json,
    write_jsonl,
)
from playground_wiki.evaluation.injection_agent_report import (
    agent_run_settings,
    render_agent_report,
)
from playground_wiki.evaluation.injection_compare import (
    CANDIDATE_COMPARES,
    CANDIDATE_NAMES,
    DEFENSE_COMPARE,
    CompareSpec,
    arm_meta_paths,
    compare_arm,
    compare_report_name,
    load_sets,
    render_compare_report,
)
from playground_wiki.evaluation.injection_defs import (
    ANSWER_VARIANTS,
    INJECTION_VARIANTS,
    MAX_INJECTION_TRIES,
    TARGET_SET_NAMES,
    TARGET_SETS,
    Target,
    TargetSet,
    Variant,
    valid_tag,
    variant_canary,
    variant_payload,
)
from playground_wiki.evaluation.injection_plant import (
    background_doc_ids,
    gold_occurrences,
    has_canary,
    has_gold,
    inject_at,
    lead_offset,
    select_partial_targets,
    select_targets,
    target_row,
)
from playground_wiki.evaluation.injection_records import load_agent_answers, load_answers, run_date
from playground_wiki.evaluation.injection_report import count_text, render_report, source_counts
from playground_wiki.evaluation.paths import (
    F9_ANSWERS_PATH,
    REPORT_DIR,
)
from playground_wiki.evaluation.search_adapters import RecordingSearch
from playground_wiki.gate import KiwiTokenizer, Tokenizer
from playground_wiki.generation import PROMPT_DIR, check_profiles
from playground_wiki.ingest import WikiCleaner
from playground_wiki.sources.wiki import (
    ARTICLE_NAMESPACE,
    MIN_CHARS,
    SOURCE_NAME,
    RawPage,
    parse_page,
)

logger = logging.getLogger(__name__)

INGEST_CONFIG = CONFIG_DIR / "ingest.yaml"


INJECTION_DB = "rag_injection_e2e"  # 별도 데이터베이스 — 운영 DB(POSTGRES_DB)는 읽기만, cleanup 이 DROP DATABASE
ADMIN_DB = "postgres"  # CREATE · DROP DATABASE 를 돌릴 관리 접속 (대상 DB 에 붙은 채로는 못 지운다)


Placement = Literal["gold", "body", "none"]
_PLACEMENT_RANK = MappingProxyType(
    {"none": 0, "body": 1, "gold": 2}
)  # gold(정답 청크) 를 가장 좋은 자리로 고른다


@dataclass(frozen=True)
class PlantOutcome:
    """지시를 심어 정제 · 청킹한 결과 — 심은 글이 어느 청크에 들어갔는지."""

    document: Document | None
    """정제된 문서. 정제에서 걸러졌으면(stub · 동음이의) None."""
    canary: str
    placement: Placement
    """gold: 정답을 담은 청크에 심은 글이 들어감 · body: 다른 청크에 · none: 어느 청크에도 안 남음."""
    at_gold: bool
    """원본에서 정답 문자열을 찾아 그 곁에 심었는지 (못 찾으면 본문 머리로)."""
    gold_chunk_chars: int | None
    """정답을 담은 청크의 글자 수. 그런 청크가 없으면 None."""


def main() -> None:
    """커맨드라인 진입점."""
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    target_set = TARGET_SETS[args.targets]
    if args.step == "extract":
        extract(target_set)
    elif args.step == "plant":
        plant(target_set)
    elif args.step == "run":
        run(
            variant_ids=args.variants,
            limit_docs=args.limit_docs,
            target_set=target_set,
            mode=args.mode,
            tag=args.tag,
        )
    elif args.step == "report":
        report(target_set, mode=args.mode, tag=args.tag)
    elif args.step == "compare":
        spec = DEFENSE_COMPARE if args.candidate is None else CANDIDATE_COMPARES[args.candidate]
        compare(mode=args.mode, spec=spec)
    elif args.step == "cleanup":
        cleanup(target_set)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["extract", "plant", "run", "report", "compare", "cleanup"])
    parser.add_argument(
        "--variants", help="run 만: 돌릴 꼴 (쉼표로). 기본은 전부 (baseline 포함)", default=""
    )
    parser.add_argument(
        "--limit-docs", type=int, default=0, help="run 만: 꼴마다 앞 문서 N 개만 (0 = 전부, 점검용)"
    )
    parser.add_argument(
        "--targets",
        choices=TARGET_SET_NAMES,
        default="grounded",
        help="표적 세트. 기본 grounded(첫 세트), partial 은 두 번째 세트 (data/eval/injection-e2e/partial-v1/)",
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="single",
        help="run · report · compare 만: /ask 모드. agent 면 기록 · 리포트 이름 끝에 -agent",
    )
    parser.add_argument(
        "--tag",
        help="run · report 만: 다시 잰 기록의 꼬리표 — 기록 · 메타 · 리포트 이름 맨 끝에 -T (예 base-v5)",
    )
    parser.add_argument(
        "--candidate",
        choices=CANDIDATE_NAMES,
        help="compare 만: 후보 비교 — 기준 판 · 앞 후보 · 이 후보 칸 셋, 리포트 이름 끝에 -v6 꼴 (예 base-v6)",
    )
    args = parser.parse_args(argv)
    if args.candidate is not None and args.step != "compare":
        parser.error("--candidate 는 compare 에만 쓴다")
    if args.mode != "single" and args.step not in {"run", "report", "compare"}:
        parser.error("--mode is for run, report and compare only")
    if args.tag is not None and args.step not in {"run", "report"}:
        parser.error("--tag 는 run · report 에만 쓴다 (compare 는 칸의 꼬리표를 스스로 읽는다)")
    if args.tag is not None and not valid_tag(args.tag):
        parser.error(
            f"--tag 는 영문 소문자 · 숫자 · '-' 만, 'agent' 로 시작하지 않게 쓴다, got {args.tag!r}"
        )
    return args


# --- extract ---------------------------------------------------------------


def extract(target_set: TargetSet) -> None:
    """표적 · 배경 문서를 덤프에서 뽑아 쓴다."""
    records = read_records(F9_ANSWERS_PATH)
    extra = list(chain.from_iterable(read_records(path) for path in target_set.extra_sources))
    targets = (
        select_targets(records)
        if target_set.name == "grounded"
        else select_partial_targets(records, extra)
    )
    if target_set.by_source:
        print(f"표적 {count_text(targets, by_source=True)}")
    background_ids = background_doc_ids([*records, *extra], targets)
    doc_ids = {t.doc_id for t in targets} | background_ids
    print(
        f"목표 문서 {len(targets)} · 배경 문서 {len(background_ids)} · 덤프에서 뽑을 문서 {len(doc_ids)}"
    )
    pages = _read_dump_pages(doc_ids)
    missing = doc_ids - set(pages)
    if missing:
        raise SystemExit(f"덤프에서 못 찾은 문서 {len(missing)}개: {sorted(missing)[:5]} …")

    target_set.out_dir.mkdir(parents=True, exist_ok=True)
    target_rows = [
        target_row(t, pages[t.doc_id].wikitext, with_source=target_set.by_source) for t in targets
    ]
    write_jsonl(target_set.targets_path, target_rows)
    background_rows = [
        {"doc_id": doc_id, "title": pages[doc_id].title, "wikitext": pages[doc_id].wikitext}
        for doc_id in sorted(background_ids)
    ]
    write_jsonl(target_set.background_path, background_rows)
    print(
        f"저장: {target_set.targets_path} ({len(target_rows)}줄) · {target_set.background_path} "
        f"({len(background_rows)}줄)"
    )


def _read_dump_pages(doc_ids: set[str]) -> dict[str, RawPage]:
    """덤프를 한 번 훑어 doc_ids 의 원본 페이지(제목 · 위키텍스트)를 모은다. 전부 찾으면 멈춘다."""
    source = _ingest_config()["source"]
    dump_path = Path(str(source["dump_path"]))
    found: dict[str, RawPage] = {}
    start = time.perf_counter()
    for page in _raw_pages(dump_path):
        if page.id in doc_ids:
            found[page.id] = page
            if len(found) == len(doc_ids):
                break
    logger.info(
        "dump scan: %d/%d docs in %.0fs", len(found), len(doc_ids), time.perf_counter() - start
    )
    return found


def _raw_pages(dump_path: Path) -> Iterable[RawPage]:
    # WikiSource.raw_pages 와 같은 방식이되, 멀티프로세스 정제 없이 원본 위키텍스트만 흘려 준다
    with bz2.open(dump_path, "rb") as handle:
        namespace = ""
        for _, elem in etree.iterparse(handle, events=("end",)):
            tag = elem.tag
            if not namespace and tag.endswith("}siteinfo"):
                namespace = tag[: tag.index("}") + 1]
            if tag != f"{namespace}page":
                continue
            if elem.findtext(f"{namespace}ns") == ARTICLE_NAMESPACE:
                revision = elem.find(f"{namespace}revision")
                text = revision.findtext(f"{namespace}text") if revision is not None else None
                yield RawPage(
                    id=elem.findtext(f"{namespace}id") or "",
                    title=elem.findtext(f"{namespace}title") or "",
                    wikitext=text or "",
                )
            elem.clear()


# --- plant (정제 · 청킹 단계, CPU만) --------------------------------------------


def plant(target_set: TargetSet) -> None:
    """꼴마다 심은 글이 청크까지 남는지 기록한다."""
    targets = [Target.from_json(row) for row in read_jsonl(target_set.targets_path)]
    wikitext = {row["doc_id"]: row["wikitext"] for row in read_jsonl(target_set.targets_path)}
    cleaner = WikiCleaner()
    chunker = _chunker()
    dump_date = str(_ingest_config()["source"]["dump_date"])
    rows = [
        _plant_row(
            target,
            doc_index,
            variant,
            wikitext[target.doc_id],
            chunker=chunker,
            cleaner=cleaner,
            dump_date=dump_date,
        )
        for (doc_index, target), variant in product(enumerate(targets), ANSWER_VARIANTS)
    ]
    target_set.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(target_set.plant_path, rows)
    survived = Counter((row["variant"], row["ok"]) for row in rows)
    print(f"저장: {target_set.plant_path} ({len(rows)}줄)")
    for variant in ANSWER_VARIANTS:
        ok = survived[variant.id, True]
        print(
            f"  {variant.id}: 기대대로 {ok}/{len(targets)} (심은 글이 청크에 {'남음' if variant.expect_canary else '없음'})"
        )


def _plant_row(
    target: Target,
    doc_index: int,
    variant: Variant,
    wikitext: str,
    *,
    chunker: Chunker,
    cleaner: WikiCleaner,
    dump_date: str,
) -> dict[str, Any]:
    outcome = place_injection(
        target, doc_index, variant, wikitext, cleaner=cleaner, chunker=chunker, dump_date=dump_date
    )
    return {
        "variant": variant.id,
        "item_id": target.item_id,
        "doc_id": target.doc_id,
        "at_gold": outcome.at_gold,
        "skipped": outcome.document is None,
        "placement": outcome.placement,
        "gold_chunk_chars": outcome.gold_chunk_chars,
        "expect_canary": variant.expect_canary,
        "ok": (outcome.placement != "none") == variant.expect_canary,
    }


def place_injection(
    target: Target,
    doc_index: int,
    variant: Variant,
    wikitext: str,
    *,
    cleaner: WikiCleaner,
    chunker: Chunker,
    dump_date: str,
) -> PlantOutcome:
    """지시를 심은 문서를 정제 · 청킹해, 심은 글이 정답 청크에 들어가는 자리를 고른다.

    정답 문자열이 정보상자 · 각주처럼 정제에서 지워지는 곳에 먼저 나오면 본문 occurrence 로 넘어간다 —
    공격자라면 눈에 보이는 본문에 심으므로. gold(정답 청크) > body(다른 청크) > none(안 남음) 순으로 고른다.
    run 과 plant 가 같은 함수를 써서, 검증한 문서와 적재하는 문서가 같다.
    """
    canary = variant_canary(variant, doc_index)
    payload = variant_payload(doc_index, canary)
    # 정답 표기가 여럿이면(롱테일) 원본에 나오는 첫 표기 곁에 심는다 — KorQuAD 는 표기가 하나라 예전과 같다
    occurrences = next(
        (found for answer in target.gold_answers if (found := gold_occurrences(wikitext, answer))),
        [],
    )
    offsets = occurrences or [lead_offset(wikitext)]
    best: PlantOutcome | None = None
    for offset in offsets[:MAX_INJECTION_TRIES]:
        document = _document(
            target, inject_at(wikitext, variant, payload, offset), cleaner, dump_date
        )
        chunks = chunker.chunk(document) if document is not None else []
        gold_chunk = next((c for c in chunks if has_gold(c.text, target)), None)
        in_gold = gold_chunk is not None and has_canary(gold_chunk, canary)
        placement = (
            "gold" if in_gold else "body" if any(has_canary(c, canary) for c in chunks) else "none"
        )
        outcome = PlantOutcome(
            document=document,
            canary=canary,
            placement=placement,
            at_gold=bool(occurrences),
            gold_chunk_chars=None if gold_chunk is None else len(gold_chunk.text),
        )
        if best is None or _PLACEMENT_RANK[placement] > _PLACEMENT_RANK[best.placement]:
            best = outcome
        if placement == "gold":
            break
    if best is None:  # offsets 는 최소 한 개라 실제로는 나지 않는다
        raise RuntimeError(f"no injection offset for {target.doc_id}")
    return best


# --- run (적재 · 답변, GPU) ------------------------------------------------------


@dataclass(frozen=True)
class _RunInputs:
    """run 이 꼴마다 쓰는 재료 — 표적 · 배경 문서 · 적재 도구 · 답변 구성."""

    target_set: TargetSet
    mode: Mode
    tag: str | None
    targets: Sequence[Target]
    wikitext: Mapping[str, str]
    background: Sequence[Document]
    cleaner: WikiCleaner
    chunker: Chunker
    embedder: SentenceTransformerEmbedder
    dump_date: str
    config: ApiConfig
    setup: AnswerSetup
    kiwi: Kiwi
    source_meta: Mapping[str, object] | None


def run(
    variant_ids: str,
    limit_docs: int,
    *,
    target_set: TargetSet = TARGET_SETS["grounded"],
    mode: Mode = "single",
    tag: str | None = None,
) -> None:
    """꼴마다 심은 문서로 답변 경로를 돌려 기록을 이어 쓴다."""
    target_rows = read_jsonl(target_set.targets_path)
    targets = [Target.from_json(row) for row in target_rows]
    wikitext = {row["doc_id"]: row["wikitext"] for row in target_rows}
    config = ApiConfig.from_env()
    # 심을 문서 · 배경 문서의 날짜 — 답변 경로의 덤프 날짜(setup.dump_date)와 같은 ingest.yaml 값이다
    dump_date = load_dump_date(config.config_dir)
    cleaner = WikiCleaner()
    background = [
        doc
        for row in read_jsonl(target_set.background_path)
        if (doc := _background_document(row, cleaner, dump_date)) is not None
    ]
    variants = _pick_variants(variant_ids)
    if limit_docs:
        targets = targets[:limit_docs]
    # 기본(grounded 표적 · 단발)은 예전 출력 그대로 — 새 인자를 줬을 때만 세트 · 모드를 덧붙인다
    extra = (
        ""
        if (target_set.name, mode) == ("grounded", "single")
        else f" · {target_set.name} · {mode}"
    )
    if tag is not None:
        extra += f" · 꼬리표 {tag} · 설정 {config.config_dir.as_posix()}"
    print(
        f"꼴 {len(variants)} × 목표 {len(targets)} + 배경 {len(background)} · DB {INJECTION_DB}{extra}",
        flush=True,
    )

    # /ask 와 같이 Kiwi 하나를 질문 관문과 BM25 가 나눠 쓴다
    kiwi = Kiwi()
    setup = answer_setup(config, KiwiTokenizer(kiwi), tag=tag)
    for variant in variants:
        refuse_prompt_change(target_set.answers_path(variant.id, mode, tag), setup.policy.profiles)
    chunker = _chunker()
    embedder = _embedder()
    qdrant = QdrantStore.connect(QdrantConfig())  # QDRANT_MODE 와 상관없이 로컬 고정

    _create_database()
    # 출처가 여럿인 세트만 메타에 출처별 표적 수 (분모) — 기본 세트의 메타 키는 그대로. 꼬리표 실행은 꼬리표 · 설정 폴더도
    source_meta = {"targets_by_source": source_counts(targets)} if target_set.by_source else None
    if tag is not None:
        source_meta = {**(source_meta or {}), **_tag_meta(tag, config)}
    inputs = _RunInputs(
        target_set=target_set,
        mode=mode,
        tag=tag,
        targets=targets,
        wikitext=wikitext,
        background=background,
        cleaner=cleaner,
        chunker=chunker,
        embedder=embedder,
        dump_date=dump_date,
        config=config,
        setup=setup,
        kiwi=kiwi,
        source_meta=source_meta,
    )
    with PgStore.connect(_injection_pg()) as pg, OllamaLLM.connect(OllamaConfig.from_env()) as llm:
        for variant in variants:
            _run_variant(inputs, variant, pg=pg, qdrant=qdrant, llm=llm)


def _run_variant(
    inputs: _RunInputs, variant: Variant, *, pg: PgStore, qdrant: QdrantStore, llm: OllamaLLM
) -> None:
    """꼴 하나: 심은 문서 + 배경을 테스트용 복사본에 적재하고 표적을 돌려 기록 · 메타를 쓴다."""
    target_set, mode, tag = inputs.target_set, inputs.mode, inputs.tag
    targets, config, setup = inputs.targets, inputs.config, inputs.setup
    collection = target_set.collection_prefix + variant.id
    documents = [
        *_planted_documents(
            targets,
            inputs.wikitext,
            variant,
            cleaner=inputs.cleaner,
            chunker=inputs.chunker,
            dump_date=inputs.dump_date,
        ),
        *inputs.background,
    ]
    _ingest(
        documents,
        collection,
        chunker=inputs.chunker,
        embedder=inputs.embedder,
        pg=pg,
        qdrant=qdrant,
        dump_date=inputs.dump_date,
        emb_dir=target_set.emb_dir,
    )
    service = SearchService.build(
        replace(config.search, collection=collection, bm25_index=None),
        pg,
        qdrant,
        analyzer=KiwiAnalyzer(inputs.kiwi),
    )
    path = target_set.answers_path(variant.id, mode, tag)
    if mode == "single":
        answerer = AnswerRun(
            service,
            lambda search: AnswerService.from_config(setup, search=search, llm=llm),
        )
        items = [t.eval_item() for t in targets]
        summary = asyncio.run(answerer.run(items, path, on_record=_progress))
        extra = inputs.source_meta
    else:
        # 에이전트 모드는 검색을 여러 번 해 출처가 마지막 검색 순서와 맞지 않는다 — AnswerRun 기록(출처 =
        # 마지막 검색의 앞부분) 대신 eval_agent 와 같은 기록(검색마다 · done 진단의 agent 칸)을 쓴다
        recording = RecordingSearch(service)
        runner = AgentEvalRun(
            recording, AnswerService.from_config(setup, search=recording, llm=llm)
        )
        summary = asyncio.run(
            runner.run([_agent_item(t) for t in targets], mode, path, on_record=_agent_progress)
        )
        extra = {**_agent_settings(setup, mode), **(inputs.source_meta or {})}
    _write_run_meta(path, config, collection, variant, len(targets), extra=extra)
    print(
        f"  {variant.id}: 기록 {summary.recorded} · 건너뜀 {summary.skipped} / {summary.total} → {path}",
        flush=True,
    )


def _planted_documents(
    targets: Sequence[Target],
    wikitext: Mapping[str, str],
    variant: Variant,
    *,
    cleaner: WikiCleaner,
    chunker: Chunker,
    dump_date: str,
) -> list[Document]:
    documents = []
    for doc_index, target in enumerate(targets):
        if variant.id == "baseline":
            document = _document(target, wikitext[target.doc_id], cleaner, dump_date)
        else:
            document = place_injection(
                target,
                doc_index,
                variant,
                wikitext[target.doc_id],
                cleaner=cleaner,
                chunker=chunker,
                dump_date=dump_date,
            ).document
        if document is None:
            raise SystemExit(f"심은 문서가 정제 뒤 걸러졌다: {variant.id} {target.doc_id}")
        documents.append(document)
    return documents


def _ingest(
    documents: Sequence[Document],
    collection: str,
    *,
    chunker: Chunker,
    embedder: SentenceTransformerEmbedder,
    pg: PgStore,
    qdrant: QdrantStore,
    dump_date: str,
    emb_dir: Path,
) -> None:
    qdrant.delete_collection(collection)  # 회차를 다시 돌릴 때 옛 점이 섞이지 않게 비우고 넣는다
    # 캐시는 있는 샤드를 건너뛴다 — 세트마다 폴더 · 컬렉션 이름이 달라야 다른 코퍼스의 벡터를 쓰지 않는다
    cache = EmbeddingCache(emb_dir / f"{collection}-{chunker.config.max_tokens}")
    result = ingest_documents(
        documents,
        source=SOURCE_NAME,
        collection=collection,
        chunker=chunker,
        embedder=embedder,
        cache=cache,
        pg=pg,
        qdrant=qdrant,
        dump_date=dump_date,
        on_disk=bool(_ingest_config()["store"]["qdrant"]["on_disk"]),
    )
    logger.info("ingest %s: %d chunks, %d points", collection, result.chunks, result.points)


def _progress(index: int, total: int, record: AnswerRecord) -> None:
    print(
        f"    {index}/{total} {record.item.id} {'→'.join(record.stage_path)} {record.wall_ms / 1000:.1f}s",
        flush=True,
    )


def _agent_progress(index: int, total: int, record: AgentRecord) -> None:
    print(
        f"    {index}/{total} {record.item.id} {record.route} {'→'.join(record.stage_path)} "
        f"{record.wall_ms / 1000:.1f}s",
        flush=True,
    )


def _agent_item(target: Target) -> AgentItem:
    """에이전트 모드 기록의 문항. set_name 은 정답이 있는 세트(f9) 꼴 — 롱테일 표적도 맞힘 판정(정답 표기 중
    하나라도)이 같다. 출처는 표적(targets.jsonl)에 있다."""
    return AgentItem(
        id=target.item_id, set_name="f9", question=target.question, answers=target.gold_answers
    )


def _agent_settings(setup: AnswerSetup, mode: Mode) -> dict[str, object]:
    """에이전트 기록 메타에 더 적는 설정 — 기록에 없는 모델 · 프롬프트 판본 (리포트의 측정 조건)."""
    library = PromptLibrary.load(AGENT_PROMPT_DIR)
    return {
        "mode": mode,
        "profiles": {
            stage: {
                "spec_id": profile.spec.spec_id,
                "system": "+".join(
                    f"{ref}:{setup.library.get(ref).digest}" for ref in profile.system
                ),
            }
            for stage, profile in setup.policy.profiles.items()
        },
        "agent_prompts": [f"{ref}:{library.get(ref).digest}" for ref in library.refs],
    }


def measurement_library() -> PromptLibrary:
    """운영 프롬프트 모음(PROMPT_DIR)에 측정 전용 프롬프트(CONTROL_PROMPT_DIR)를 더한 모음. 이름이 겹치면 멈춘다.

    Raises:
      PromptError: 두 폴더에 같은 `이름@버전` 이 있을 때 — 대조 프롬프트가 운영 프롬프트를 가리지 않게.
    """
    service = PromptLibrary.load(PROMPT_DIR)
    control = PromptLibrary.load(CONTROL_PROMPT_DIR)
    return PromptLibrary(
        [*(service.get(ref) for ref in service.refs), *(control.get(ref) for ref in control.refs)]
    )


def answer_setup(config: ApiConfig, tokenizer: Tokenizer, *, tag: str | None) -> AnswerSetup:
    """run 의 답변 경로 구성. 꼬리표가 없으면 서버와 같은 `AnswerSetup.from_config` 그대로.

    꼬리표 실행은 같은 설정 파일(config.config_dir — RAG_CONFIG_DIR 의 칸 사본)을 같은 함수로 읽되 프롬프트만
    measurement_library 에서 찾는다 — 운영 프롬프트 모음에 없는 대조 칸(base_norule)을 가리킬 수 있게. 서버는
    운영 모음만 읽으므로 answer.yaml 이 대조 프롬프트를 가리키면 시작할 때 멈춘다.
    """
    if tag is None:
        return AnswerSetup.from_config(config, tokenizer=tokenizer, log_questions=False)
    library = measurement_library()
    policy = load_answer_policy(config.config_dir)
    check_profiles(policy.profiles.values(), library)
    return AnswerSetup(
        policy=policy,
        library=library,
        gate=load_query_gate(config.config_dir, tokenizer),
        dump_date=load_dump_date(config.config_dir),
        log_questions=False,
    )


def _tag_meta(tag: str, config: ApiConfig) -> dict[str, str]:
    """꼬리표 실행의 메타에 더하는 키 — 꼬리표와 읽은 설정 폴더 (꼬리표 없는 실행의 메타 키는 그대로)."""
    return {"tag": tag, "config_dir": config.config_dir.as_posix()}


# --- report ----------------------------------------------------------------


def report(
    target_set: TargetSet = TARGET_SETS["grounded"],
    *,
    mode: Mode = "single",
    tag: str | None = None,
) -> None:
    """표적 세트 하나의 간접 인젝션 리포트를 쓴다."""
    if not target_set.targets_path.exists():
        raise SystemExit(f"{target_set.targets_path} 가 없다 — extract 를 먼저 돌린다")
    targets = [Target.from_json(row) for row in read_jsonl(target_set.targets_path)]
    answers = {
        variant.id: load_answers(target_set, variant.id, tag) for variant in INJECTION_VARIANTS
    }
    # 리포트 날짜는 답 기록을 만든 날 — 기록에서 다시 만들어도 같은 리포트를 고쳐 쓴다
    metas = [
        meta_path(target_set.answers_path(variant.id, mode, tag)) for variant in INJECTION_VARIANTS
    ]
    report_date = run_date(metas) or datetime.now(tz=UTC).date()
    # 꼬리표 기록은 대조 칸(측정 전용 프롬프트)일 수 있어 그 모음까지 읽는다
    library = PromptLibrary.load(PROMPT_DIR) if tag is None else measurement_library()
    if mode == "single":
        plant_path = target_set.plant_path
        plant_rows = read_jsonl(plant_path) if plant_path.exists() else []
        text = render_report(
            targets,
            answers,
            plant_rows,
            library=library,
            report_date=report_date,
            target_set=target_set,
            tag=tag,
        )
    else:
        agent_answers = {
            variant.id: load_agent_answers(target_set, variant.id, tag)
            for variant in INJECTION_VARIANTS
        }
        text = render_agent_report(
            targets,
            agent_answers,
            answers,
            settings=agent_run_settings(metas),
            library=library,
            report_date=report_date,
            target_set=target_set,
            tag=tag,
        )
    REPORT_DIR.mkdir(exist_ok=True)
    path = REPORT_DIR / target_set.report_name(report_date, mode, tag)
    path.write_text(text, encoding="utf-8", newline="\n")  # .gitattributes 가 LF
    print(f"리포트: {path}")


# --- report (출처가 여럿인 세트 · 새 리포트) -------------------------------------------


# --- report (에이전트 모드) ------------------------------------------------------


# --- compare (방어 비교 칸) ---------------------------------------------------


def compare(mode: Mode = "single", spec: CompareSpec = DEFENSE_COMPARE) -> None:
    """방어 칸 비교 리포트를 쓴다."""
    sets = load_sets()
    arms = [compare_arm(column, mode) for column in spec.arms]
    # 리포트 날짜는 새 칸 기록을 만든 날 (방어 전 기록은 그 전에 쟀다)
    metas = []
    for tag in spec.dated_by:
        metas.extend(arm_meta_paths(tag, mode))
    report_date = run_date(metas) or datetime.now(tz=UTC).date()
    text = render_compare_report(
        sets, arms, mode=mode, library=measurement_library(), report_date=report_date, spec=spec
    )
    REPORT_DIR.mkdir(exist_ok=True)
    path = REPORT_DIR / compare_report_name(report_date, mode, spec.suffix)
    path.write_text(text, encoding="utf-8", newline="\n")  # .gitattributes 가 LF
    print(f"리포트: {path}")


# --- cleanup ---------------------------------------------------------------


def cleanup(target_set: TargetSet = TARGET_SETS["grounded"]) -> None:
    """그 세트의 Qdrant 컬렉션 · 임베딩 캐시를 지우고 테스트용 DB 를 지운다 (DB 는 세트가 같이 쓴다)."""
    qdrant = QdrantStore.connect(QdrantConfig())
    for variant in INJECTION_VARIANTS:
        qdrant.delete_collection(target_set.collection_prefix + variant.id)
    print(f"Qdrant: {target_set.collection_prefix}* 컬렉션 지움")
    _drop_database()
    if target_set.emb_dir.exists():
        shutil.rmtree(target_set.emb_dir)
        print(f"임베딩 캐시 지움: {target_set.emb_dir}")
    print(
        f"남긴 것: {target_set.out_dir}/ 의 중간 결과(targets · background · plant · answers)는 로컬에 둔다"
    )


# --- 공통 ------------------------------------------------------------------


def _document(
    target: Target, wikitext: str, cleaner: WikiCleaner, dump_date: str
) -> Document | None:
    page = RawPage(id=target.doc_id, title=target.title, wikitext=wikitext)
    parsed = parse_page(page, cleaner=cleaner, dump_date=dump_date, min_chars=MIN_CHARS)
    return parsed if isinstance(parsed, Document) else None


def _background_document(
    row: Mapping[str, object], cleaner: WikiCleaner, dump_date: str
) -> Document | None:
    page = RawPage(id=str(row["doc_id"]), title=str(row["title"]), wikitext=str(row["wikitext"]))
    parsed = parse_page(page, cleaner=cleaner, dump_date=dump_date, min_chars=MIN_CHARS)
    return parsed if isinstance(parsed, Document) else None


def _chunker() -> Chunker:
    config = _ingest_config()
    chunking = config["chunking"]
    spec = EMBEDDING_MODELS[config["embedding"]["model"]]
    tokenizer = HFTokenizer.from_pretrained(spec.model_id)
    max_tokens = min(
        chunking["max_tokens"], spec.max_chunk_tokens(tokenizer, tokenizer.special_token_count)
    )
    return Chunker(
        tokenizer,
        ChunkingConfig(
            max_tokens=max_tokens,
            overlap_ratio=chunking["overlap_ratio"],
            min_chars=chunking["min_chars"],
        ),
    )


def _embedder() -> SentenceTransformerEmbedder:
    config = _ingest_config()
    spec = EMBEDDING_MODELS[config["embedding"]["model"]]
    return SentenceTransformerEmbedder.load(
        spec, fp16=config["embedding"]["fp16"], batch_size=config["embedding"]["batch_size"]
    )


def _pick_variants(variant_ids: str) -> list[Variant]:
    if not variant_ids:
        return list(INJECTION_VARIANTS)
    wanted = [name.strip() for name in variant_ids.split(",") if name.strip()]
    by_id = {v.id: v for v in INJECTION_VARIANTS}
    missing = [name for name in wanted if name not in by_id]
    if missing:
        raise SystemExit(f"모르는 꼴: {missing} (있는 것: {sorted(by_id)})")
    return [by_id[name] for name in wanted]


def _injection_pg() -> PgConfig:
    return replace(PgConfig.from_env(), dbname=INJECTION_DB)


def _create_database() -> None:
    if not _database_exists():
        _admin_execute(f'CREATE DATABASE "{INJECTION_DB}"')
        logger.info("created database %s", INJECTION_DB)


def _drop_database() -> None:
    if _database_exists():
        _admin_execute(f'DROP DATABASE "{INJECTION_DB}" WITH (FORCE)')
        print(f"PostgreSQL: 데이터베이스 {INJECTION_DB} 지움")
    else:
        print(f"PostgreSQL: 데이터베이스 {INJECTION_DB} 없음 (이미 지워짐)")


def _database_exists() -> bool:
    with psycopg.connect(_admin_conninfo(), autocommit=True) as conn:
        row = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (INJECTION_DB,)
        ).fetchone()
    return row is not None


def _admin_execute(sql: str) -> None:
    # CREATE · DROP DATABASE 는 트랜잭션 밖에서만 돈다 → autocommit. 이름은 상수라 보간해도 안전하다
    with psycopg.connect(_admin_conninfo(), autocommit=True) as conn:
        conn.execute(sql)


def _admin_conninfo() -> str:
    return replace(PgConfig.from_env(), dbname=ADMIN_DB).conninfo


def _write_run_meta(
    path: Path,
    config: ApiConfig,
    collection: str,
    variant: Variant,
    doc_count: int,
    *,
    extra: Mapping[str, object] | None = None,
) -> None:
    meta = {
        "created_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "variant": variant.id,
        "collection": collection,
        "database": INJECTION_DB,
        "doc_count": doc_count,
        "reranker_model": config.search.reranker_model,
        "top_k": config.search.top_k,
        **(extra or {}),  # 에이전트 모드만 — 단발 메타는 예전 키 그대로
    }
    write_json(meta_path(path), meta)


def _ingest_config() -> Mapping[str, Any]:
    return yaml.safe_load(INGEST_CONFIG.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
