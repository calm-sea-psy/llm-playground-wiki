"""F9 답변 평가 — 실제 답변 경로(`/ask` 와 같은 AnswerService)로 평가 문항을 돌려 자동으로 셀 수 있는 지표를 모은다.

단계별로 돌린다 — 단계마다 결과를 data/ 에 저장해, 뒤 단계를 다시 돌릴 때 앞 단계를 반복하지 않는다.
  sample   KorQuAD dev 의 eval 몫(문단 해시 40%)에서 문단마다 한 문항을 고르고, 정답 문서가 코퍼스에 있고 정답 문장이
           지금 문서에 남은 것만 + 위키 밖 질문(data/eval/offwiki-v1.jsonl) 전부 → data/eval/f9-sample-v1.jsonl
           (calibrate_threshold.py sample 과 같은 방식에 몫만 다르다 — 문단 해시라 calibrate · F4 와 겹치지 않는다).
           지금 표본 파일은 이 단계 뒤 제목이 어긋났던 두 문서(아가씨 · 조현아)의 문항을 KorQuAD 줄 끝에 덧붙인 것이다
           (2026-10-31, 덧붙인 단계는 지웠다 — git 이력). 다시 돌리면 기록과 짝이 깨지므로 돌리지 않는다
  run      표본을 서버 없이 /ask 와 같은 조립(관문 · 검색 · 답변 정책 · grounded → partial 캐스케이드)으로 돌려
           문항마다 SSE 이벤트를 기록 한 줄로 → data/eval/f9-answers-v1.jsonl (+ .meta.json 에 실행마다 설정).
           단계 경로 · 표지 · 답 원문 · 검색 청크 · 출처 번호(cited · dropped) · 잘림 · 시간 초과 · 진단 시간 · 1위 점수.
           끝난 문항은 건너뛰어 멈췄다 다시 돌려도 된다 (생성 실패 기록은 지우고 다시 돈다). --limit N 은 N 문항만 (점검용).
           --ids a,b,c 는 그 문항만 (표본 순서 — 같은 문항을 여러 번 돌려 노이즈를 볼 때 --tag 로 회차를 나눈다).
           PG · Qdrant(docker compose up) · BM25 색인(data/index/bm25/{컬렉션}) · Ollama 가 필요하다
  retrieval-sensitivity
           패러프레이즈 v1(data/bench/paraphrase-v1.jsonl)의 f9-random 300 을 원문 · 패러프레이즈로 같은 검색
           (Hybrid + 리랭커, 상위 5)에 돌려 정답 근거 청크 순위를 → data/eval/f9-retrieval-sensitivity-v1.jsonl (+ .meta.json).
           LLM 은 쓰지 않는다. PG · Qdrant · BM25 색인이 필요하다
  report   reports/{날짜}-answer-eval.md 에 집계만 (질문 · 답 원문 없음). 민감도 결과가 없으면 그 절은 "아직"으로 둔다.
           민감도 절에는 패러프레이즈 사람 확인 요약과, 뜻이 조금 달라진 문항(PARAPHRASE_SHIFTED)을 뺀 hit 도 싣는다.
           --tag T 는 꼬리표 기록(f9-answers-v1-T.jsonl)으로 그린다 — 채택한 프롬프트로 다시 잰 기록을 최종 리포트로

high 확정 (ADR-024) — 1위 점수 0.83~0.94 구간은 표본의 약 5% 뿐이라 점수로 층화해 그 구간을 일부러 많이 담는다:
  pool     eval 몫 문단의 KorQuAD 질문 전부(문단마다 한 문항 · 정답 문장 필터 없이)를 /ask 와 같은 검색에만 돌려
           1위 점수를 → data/eval/f9-pool-scores-v1.jsonl (+ .meta.json). LLM 은 쓰지 않는다. 끝난 문항은 건너뛴다.
           지금 풀 파일에는 중간 구간이 모자라 덧붙인 bench 몫이 함께 있다 (2026-10-29, 덧붙인 옵션은 지웠다 — git 이력)
  strata   점수 풀에서 0.83~0.94 구간을 eval 몫부터(모자라면 bench 몫에서 seed 순서로, --middle-size 까지) 담고,
           몫마다 같은 수를 ≥ 0.94 구간에서 seed 로 → data/eval/f9-strata-sample-v1.jsonl
  run --strata
           층화 표본을 run 과 같이 → data/eval/f9-strata-answers-v1.jsonl (+ .meta.json)
  judge-input
           grounded 로 끝난 답을 점수 · 구간 · 정답을 가리고 섞어 판정 입력으로 → data/eval/f9-strata-judge-input-v1.jsonl
           (+ 가린 번호 → 문항 id 열쇠 f9-strata-judge-key-v1.jsonl). 판정(grounding 라벨)은 이 파일만 보고
           data/eval/f9-strata-verdicts-v1.jsonl 에 쓴다
  strata-report
           reports/{날짜}-high-threshold.md 에 구간별 지어냄 비율 · 차이 구간 · 지어냄 × 검색 상위 5 정답 근거 · 판정 검수
           (집계만). --report-date 는 리포트 날짜 (다시 그릴 때 처음 날짜를 지킨다)

프롬프트 · 모델을 바꿔 다시 잴 때 (ADR-024 결론 — 점수 풀 · 층화 표본은 그대로, 같은 단계로):
  run --tag T · run --strata --tag T · judge-input --tag T
           기록 · 판정 입력 · 열쇠 파일 이름 끝에 -T 를 붙여 따로 쓴다 (f9-answers-v1-T.jsonl …) — 꼬리표 없는 기록은
           덮어쓰지 않는다. 판정은 data/eval/f9-strata-verdicts-v1-T.jsonl 에.
           run --prompt 단계@버전 은 그 단계 블록 프롬프트만 바꿔 돌린다 (예 partial@v3) — configs/answer.yaml 은
           그대로다. 채택은 측정 뒤 사람이 그 파일에 적는다. 여러 번 주면 단계마다 바꾼다
           (--prompt grounded@v4 --prompt partial@v4). 한 단계에 둘은 받지 않는다
  cascade-input --tag T
           기준에서 grounded 로 끝났는데 T 에서 grounded → partial 로 넘어간 KorQuAD 문항(F9 · 층화)의 판정 입력 →
           data/eval/f9-cascade-judge-input-v1-T.jsonl (+ 열쇠). T 의 partial 답은 문서 칸만, F9 문항은 기준 grounded 답도
           (층화 문항의 기준 답은 high 확정 판정이 있다). 섞어서 가린다 — 판정은 f9-cascade-verdicts-v1-T.jsonl 에.
           --base-tag B 는 기준을 꼬리표 없는 기록 대신 B 기록으로 (B 의 층화 판정 f9-strata-verdicts-v1-B.jsonl 까지).
           compare 에도 같은 B 를 준다 — 열쇠에 남은 기준과 다르면 멈춘다
  compare --tag T
           reports/{날짜}-T.md 에 꼬리표 없는 기록(기준)과 T 기록을 같은 문항으로 비교 — 층화 표본 지어냄 · F9 KorQuAD ·
           위키 밖 질문 (집계만). cascade 판정이 있으면 넘어간 답의 짝 표 절을 더한다. --report-date YYYY-MM-DD 는
           리포트 날짜 (점검을 더해 다시 그릴 때 처음 날짜를 지킨다). --base-tag B 는 기준을 B 기록 · B 판정으로
           (리포트 머리에 기준 이름이 나온다)
  tally --tag P
           꼬리표가 P-…-N (N 은 회차 번호) 인 F9 기록들을 회차마다 · 문항마다 단계 경로 · 첫 줄 표지로 세어 표로 찍는다
           (같은 문항을 같은 프롬프트로 여러 번 돌린 노이즈 점검 — 파일을 쓰지 않는다). --ids 로 문항을 좁힌다

RAG 없이 기준선 (closed-book, 2026-10-30 결정 — 최종 수치는 RAG 기록, closed-book 은 비교 열):
  run --closed-book --tag T
           같은 표본을 관문 · 검색 없이 grounded 프로필의 모델 · 설정으로, 질문만 주고 답하게 한다 (프롬프트 closed_book@v1
           — 모르면 "모름" 한 단어) → f9-answers-v1-T.jsonl. Ollama 만 있으면 된다 (PG · Qdrant · 검색 모델을 올리지 않는다)
  run --longtail [--closed-book] --tag T
           F4 결선 롱테일 세트(LLM 레포 bench/data, 읽기만)의 답 있음 300 을 `/ask` 경로로(또는 closed-book 으로) →
           data/eval/longtail-answers-v1-T.jsonl. RAG 쪽도 이 경로로 따로 돌려야 짝이 된다 (F4 는 저장된 문맥으로 쟀다)
  closed-book-report --tag T [--rag-tag R]
           reports/{날짜}-closed-book.md 에 closed-book 기록(T)과 RAG 기록(R, 없으면 꼬리표 없는 기록)을 같은 문항 짝으로 —
           정답 포함 · 모름 · 답했는데 틀림, RAG 만 · closed-book 만 정답, RAG 가 망친 문항 id, partial 일반 지식 칸, 위키 밖
           (집계와 문항 id 만). 롱테일 기록이 둘 다 있으면 그 묶음도. KorQuAD 는 문단 없이 뜻이 통하는지로 갈라 센다
           (Kiwi 품사 규칙 — closed_book.question_kind). --report-date 는 리포트 날짜

지표 규칙은 playground_wiki.evaluation.answer_metrics, 실행 기록은 playground_wiki.evaluation.answer_run.
KorQuAD 는 CC BY-ND 라 질문 · 답 원문 · 패러프레이즈는 data/ 에만 둔다 (git 제외). 레포 루트에서 돌린다.

사용법:
  uv run python scripts/eval_answer.py sample
  uv run python scripts/eval_answer.py run --limit 5
  uv run python scripts/eval_answer.py run
  uv run python scripts/eval_answer.py retrieval-sensitivity
  uv run python scripts/eval_answer.py report
  uv run python scripts/eval_answer.py report --tag safety-v4-1
  uv run python scripts/eval_answer.py pool
  uv run python scripts/eval_answer.py strata
  uv run python scripts/eval_answer.py run --strata
  uv run python scripts/eval_answer.py judge-input
  uv run python scripts/eval_answer.py strata-report
  uv run python scripts/eval_answer.py run --tag grounded-v3
  uv run python scripts/eval_answer.py compare --tag grounded-v3
  uv run python scripts/eval_answer.py compare --tag safety-v4-1 --base-tag grounded-v3
  uv run python scripts/eval_answer.py run --tag partial-v3-1 --prompt partial@v3
  uv run python scripts/eval_answer.py run --tag safety-v4-1 --prompt grounded@v4 --prompt partial@v4
  uv run python scripts/eval_answer.py run --ids ow-049,ow-064,ow-086 --tag noise-v2-1 --prompt grounded@v2
  uv run python scripts/eval_answer.py tally --tag noise
  uv run python scripts/eval_answer.py run --closed-book --tag closed-book
  uv run python scripts/eval_answer.py run --longtail --tag grounded-v3 --prompt grounded@v3
  uv run python scripts/eval_answer.py run --longtail --closed-book --tag closed-book
  uv run python scripts/eval_answer.py closed-book-report --tag closed-book --rag-tag grounded-v3
"""

import argparse
import asyncio
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Self

from kiwipiepy import Kiwi
from playground_evalkit import (
    KorQuADItem,
    normalize_title,
)
from playground_llm import PromptLibrary, compose
from playground_llm.ollama import OllamaConfig, OllamaLLM
from playground_retrieval.search import KiwiAnalyzer, SearchResult, SearchService
from playground_retrieval.store import PgStore, QdrantStore

from playground_wiki.api.answer import AnswerService
from playground_wiki.api.config import AnswerSetup, ApiConfig, load_answer_policy
from playground_wiki.evaluation import (
    korquad_paragraphs,
    split_of,
)
from playground_wiki.evaluation.answer_compare_report import render_compare_report
from playground_wiki.evaluation.answer_measure import (
    CASCADE_INPUT_PATH,
    CASCADE_KEY_PATH,
    JUDGE_INPUT_PATH,
    JUDGE_KEY_PATH,
    POOL_PATH,
    STRATA_ANSWERS_PATH,
    STRATA_SAMPLE_PATH,
    Measurement,
    SidePair,
    read_cascade_verdicts,
    read_pool,
    read_runs,
    read_verdicts,
)
from playground_wiki.evaluation.answer_metrics import (
    answer_rank,
    format_percent,
    paired_hits,
    score_band,
    score_bands,
    stage_path,
)
from playground_wiki.evaluation.answer_report import hit_pairs, render_report
from playground_wiki.evaluation.answer_run import (
    TAG_PATTERN,
    AnswerRecord,
    AnswerRun,
    EvalItem,
    read_records,
    refuse_prompt_change,
    repeated_stages,
    tagged,
)
from playground_wiki.evaluation.answer_strata_report import render_strata_report
from playground_wiki.evaluation.answer_tally import render_tally
from playground_wiki.evaluation.bench_search import search_all
from playground_wiki.evaluation.closed_book import (
    CLOSED_BOOK_STAGE,
    ClosedBookAnswerer,
)
from playground_wiki.evaluation.closed_book_report import question_kinds, render_closed_book_report
from playground_wiki.evaluation.f9_sample import (
    SEED,
    longtail_items,
    sample_rows,
    select_items,
)
from playground_wiki.evaluation.files import (
    append_jsonl,
    append_run_meta,
    document_texts,
    meta_path,
    read_json,
    read_jsonl,
    read_run_meta,
    write_json,
    write_jsonl,
)
from playground_wiki.evaluation.grounding import (
    cascade_rows,
    judge_rows,
)
from playground_wiki.evaluation.paths import (
    BENCH_DATA,
    F9_ANSWERS_PATH as ANSWERS_PATH,
    F9_SAMPLE_PATH as SAMPLE_PATH,
    F9_SENSITIVITY_PATH as SENSITIVITY_PATH,
    LONGTAIL_ANSWERS_PATH,
    LONGTAIL_STAGE2_SET,
    OFFWIKI_PATH,
    PARAPHRASE_PATH,
    PARAPHRASE_SAMPLE,
    PARSED,
    REPORT_DIR,
    SUBSET,
)
from playground_wiki.evaluation.search_adapters import NoSearch
from playground_wiki.evaluation.splits import subset_korquad
from playground_wiki.evaluation.strata import pool_items, pool_row, strata_rows
from playground_wiki.gate import KiwiTokenizer
from playground_wiki.generation import PROMPT_DIR, AnswerPolicy, korean_date

TAG_STEPS = frozenset(
    {"run", "report", "judge-input", "cascade-input", "compare", "tally", "closed-book-report"}
)
# 꼬리표가 꼭 있어야 하는 단계 — 기준 기록과 비교하거나 (tally 는) 꼬리표 앞부분으로 회차 기록들을 고른다
TAG_REQUIRED_STEPS = frozenset({"cascade-input", "compare", "tally", "closed-book-report"})
# 기준 기록을 꼬리표 기록으로 고를(--base-tag) 수 있는 단계
BASE_TAG_STEPS = frozenset({"cascade-input", "compare"})
# 리포트 날짜(--report-date)를 고를 수 있는 단계
REPORT_DATE_STEPS = frozenset({"compare", "closed-book-report", "strata-report"})
# 문항 지정(--ids)을 받는 단계
IDS_STEPS = frozenset({"run", "tally"})
# RAG 없이 기준선 (closed-book, 2026-10-30 결정): 같은 문항을 검색 없이 grounded 프로필의 모델 · 설정으로 — 프롬프트만
# 이것으로 바꾼다 (base 는 문서 칸 · 출처 규칙이 있어 그대로 쓰지 않는다). 최종 수치는 RAG, closed-book 은 비교 열
CLOSED_BOOK_PROMPT = "closed_book@v1"
# F4 결선 롱테일 세트 (LLM 레포, 읽기만) — 덜 알려진 문서라 KorQuAD(2018 공개)보다 학습 데이터 오염을 덜 받는다.
# 답 있음(kind longtail) 300 만 쓴다 (f9_sample.longtail_items)
LONGTAIL_SET = BENCH_DATA / LONGTAIL_STAGE2_SET / "items.jsonl"


@dataclass(frozen=True)
class ProfileSettings:
    """실행 meta settings 의 단계 하나 — 생성 설정 · 단계 프롬프트 블록 · 시간 제한."""

    spec_id: str
    system: tuple[str, ...]
    user: str
    timeout_sec: float

    @classmethod
    def from_json(cls, row: Mapping[str, Any]) -> Self:
        """meta settings 의 profiles 값 하나에서."""
        return cls(
            spec_id=str(row["spec_id"]),
            system=tuple(str(block) for block in row["system"]),
            user=str(row["user"]),
            timeout_sec=float(row["timeout_sec"]),
        )

    def to_json(self) -> dict[str, Any]:
        """meta settings 의 profiles 값 하나 — 키 차례는 필드 차례."""
        return {
            "spec_id": self.spec_id,
            "system": list(self.system),
            "user": self.user,
            "timeout_sec": self.timeout_sec,
        }


@dataclass(frozen=True)
class RunSettings:
    """`/ask` 경로로 돈 run 한 번의 설정 (meta runs 의 settings) — 리포트가 실행 사이 · 기록 사이에 견준다.

    쓰기만 이 꼴로 한다. 리포트는 옛 기록(키가 덜 있는 settings)도 읽어 견주므로 meta 를 dict 로 읽는다.
    """

    collection: str
    model: str
    candidates: int
    rerank_candidates: int
    reranker_model: str
    top_k: int
    high: float
    low: float
    gate: bool
    """질문 관문을 켰나."""
    dump_date: str
    profiles: Mapping[str, ProfileSettings]
    """단계 이름 → 설정 (이름 차례)."""
    sample: str
    """표본 파일 경로."""

    @classmethod
    def from_config(
        cls,
        config: ApiConfig,
        policy: AnswerPolicy,
        *,
        gate_enabled: bool,
        dump_date: str,
        sample_path: Path,
    ) -> Self:
        """이번 실행의 검색 설정 · 답변 정책에서."""
        return cls(
            collection=config.search.collection,
            model=config.search.model,
            candidates=config.search.candidates,
            rerank_candidates=config.search.rerank_candidates,
            reranker_model=config.search.reranker_model,
            top_k=config.search.top_k,
            high=policy.thresholds.high,
            low=policy.thresholds.low,
            gate=gate_enabled,
            dump_date=dump_date,
            profiles=MappingProxyType(
                {
                    stage: ProfileSettings(
                        spec_id=profile.spec.spec_id,
                        system=tuple(profile.system),
                        user=profile.user,
                        timeout_sec=profile.timeout_sec,
                    )
                    for stage, profile in sorted(policy.profiles.items())
                }
            ),
            sample=str(sample_path),
        )

    @classmethod
    def from_json(cls, settings: Mapping[str, Any]) -> Self:
        """meta runs 의 settings 하나에서 (지금 꼴만 — 옛 꼴이면 KeyError)."""
        return cls(
            collection=str(settings["collection"]),
            model=str(settings["model"]),
            candidates=int(settings["candidates"]),
            rerank_candidates=int(settings["rerank_candidates"]),
            reranker_model=str(settings["reranker_model"]),
            top_k=int(settings["top_k"]),
            high=float(settings["thresholds"]["high"]),
            low=float(settings["thresholds"]["low"]),
            gate=bool(settings["gate"]),
            dump_date=str(settings["dump_date"]),
            profiles=MappingProxyType(
                {
                    str(stage): ProfileSettings.from_json(profile)
                    for stage, profile in settings["profiles"].items()
                }
            ),
            sample=str(settings["inputs"]["sample"]),
        )

    def to_json(self) -> dict[str, Any]:
        """meta runs 의 settings — 키 차례는 기록 꼴 그대로."""
        return {
            "collection": self.collection,
            "model": self.model,
            "candidates": self.candidates,
            "rerank_candidates": self.rerank_candidates,
            "reranker_model": self.reranker_model,
            "top_k": self.top_k,
            "thresholds": {"high": self.high, "low": self.low},
            "gate": self.gate,
            "dump_date": self.dump_date,
            "profiles": {stage: profile.to_json() for stage, profile in self.profiles.items()},
            "inputs": {"sample": self.sample},
        }


@dataclass(frozen=True)
class ClosedBookSettings:
    """closed-book run 한 번의 설정 (meta runs 의 settings) — grounded 프로필의 모델 · 설정에 프롬프트만 바꾼다."""

    spec_id: str
    profile: str
    """모델 · 생성 설정을 빌린 프로필 이름."""
    system: tuple[str, ...]
    prompt_version: str
    timeout_sec: float
    sample: str
    """표본 파일 경로."""

    @classmethod
    def from_json(cls, settings: Mapping[str, Any]) -> Self:
        """meta runs 의 settings 하나에서."""
        return cls(
            spec_id=str(settings["spec_id"]),
            profile=str(settings["profile"]),
            system=tuple(str(block) for block in settings["system"]),
            prompt_version=str(settings["prompt_version"]),
            timeout_sec=float(settings["timeout_sec"]),
            sample=str(settings["inputs"]["sample"]),
        )

    def to_json(self) -> dict[str, Any]:
        """meta runs 의 settings — 키 차례는 기록 꼴 그대로 (mode 가 먼저)."""
        return {
            "mode": CLOSED_BOOK_STAGE,
            "spec_id": self.spec_id,
            "profile": self.profile,
            "system": list(self.system),
            "prompt_version": self.prompt_version,
            "timeout_sec": self.timeout_sec,
            "inputs": {"sample": self.sample},
        }


def main() -> None:
    """커맨드라인 진입점."""
    args = _parse_args()
    if args.step == "sample":
        sample()
    elif args.step == "run":
        sample_path, answers_path = _run_paths(strata=args.strata, longtail=args.longtail)
        if args.closed_book:
            run_closed_book(
                args.limit,
                sample_path=sample_path,
                answers_path=tagged(answers_path, args.tag),
                ids=args.ids,
            )
        else:
            run(
                limit=args.limit,
                sample_path=sample_path,
                answers_path=tagged(answers_path, args.tag),
                prompts=args.prompt or (),
                ids=args.ids,
            )
    elif args.step == "retrieval-sensitivity":
        retrieval_sensitivity()
    elif args.step == "report":
        report(args.tag)
    elif args.step == "pool":
        pool()
    elif args.step == "strata":
        strata_sample(args.middle_size)
    elif args.step == "judge-input":
        judge_input(args.tag)
    elif args.step == "strata-report":
        strata_report(args.report_date)
    elif args.step == "cascade-input":
        cascade_input(args.tag, args.base_tag)
    elif args.step == "compare":
        compare(args.tag, args.report_date, args.base_tag)
    elif args.step == "tally":
        tally(args.tag, args.ids)
    elif args.step == "closed-book-report":
        closed_book_report(args.tag, args.rag_tag, args.report_date)


def _run_paths(*, strata: bool, longtail: bool) -> tuple[Path, Path]:
    """run 의 (표본, 꼬리표 붙이기 전 기록) 경로 — F9 표본 · 층화 표본 · 롱테일 세트."""
    if strata:
        return STRATA_SAMPLE_PATH, STRATA_ANSWERS_PATH
    if longtail:
        return LONGTAIL_SET, LONGTAIL_ANSWERS_PATH
    return SAMPLE_PATH, ANSWERS_PATH


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "step",
        choices=[
            "sample",
            "run",
            "retrieval-sensitivity",
            "report",
            "pool",
            "strata",
            "judge-input",
            "strata-report",
            "cascade-input",
            "compare",
            "tally",
            "closed-book-report",
        ],
    )
    # run 만: 이번에 돌릴 최대 문항 수 (전체를 돌리기 전 점검용 — 끝난 문항은 다음 실행이 건너뛴다)
    parser.add_argument("--limit", type=int)
    # run · tally 만: 이 문항만 (쉼표 목록, 예 ow-049,ow-064,ow-086). 표본에 없는 id 가 있으면 멈춘다
    parser.add_argument("--ids", type=_id_list)
    # run 만: F9 표본 대신 층화 표본 (high 확정, ADR-024)
    parser.add_argument("--strata", action="store_true")
    # run 만: F9 표본 대신 롱테일 세트의 답 있음 300 (LONGTAIL_SET → longtail-answers-v1.jsonl)
    parser.add_argument("--longtail", action="store_true")
    # run 만: 검색 · 관문 없이 질문만 주고 답한다 (RAG 없이 기준선). 기준 기록을 덮어쓰지 않게 --tag 와 함께
    parser.add_argument("--closed-book", action="store_true")
    # closed-book-report 만: 견줄 RAG 기록의 꼬리표. 없으면 꼬리표 없는 기록 (--tag 는 closed-book 기록의 꼬리표)
    parser.add_argument("--rag-tag")
    # cascade-input · compare 만: 기준 기록의 꼬리표 (예 grounded-v3 — 지금 운영 구성). 없으면 꼬리표 없는 기록.
    # 두 단계에 같은 값을 준다 — cascade 열쇠에 남아 compare 가 어긋남을 잡는다
    parser.add_argument("--base-tag")
    # strata 만: 0.83~0.94 구간 문항 수 상한 (eval 몫부터 다 담고 bench 몫에서 채운다). 없으면 전부
    parser.add_argument("--middle-size", type=int)
    # run · judge-input · cascade-input · compare 만: 프롬프트 · 모델을 바꿔 다시 잰 기록의 꼬리표 (예 grounded-v3).
    # cascade-input · compare 는 꼭 준다. tally 는 회차 기록 꼬리표의 앞부분 (noise → noise-v2-1 · noise-v3-2 …)
    parser.add_argument("--tag")
    # run 만: 단계 블록 프롬프트를 바꿔 돌린다 (예 partial@v3) — configs/answer.yaml 은 건드리지 않는다.
    # 여러 번 주면 단계마다 (grounded@v4 · partial@v4 를 함께). 기준 기록을 덮어쓰지 않게 --tag 와 함께 쓴다
    parser.add_argument("--prompt", action="append")
    # compare · closed-book-report · strata-report 만: 리포트 날짜(파일 이름 · 제목). 없으면 오늘 — 점검을 더해 다시
    # 그릴 때 처음 리포트 날짜를 지키려고
    parser.add_argument("--report-date", type=date.fromisoformat)
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.report_date is not None and args.step not in REPORT_DATE_STEPS:
        parser.error(f"--report-date 는 {' · '.join(sorted(REPORT_DATE_STEPS))} 에만 쓴다")
    if (args.closed_book or args.longtail) and args.step != "run":
        parser.error("--closed-book · --longtail 은 run 에만 쓴다")
    if args.longtail and args.strata:
        parser.error("--longtail 과 --strata 는 함께 쓰지 않는다")
    if args.closed_book and args.tag is None:
        parser.error("--closed-book 은 --tag 와 함께 쓴다 — 기준 기록을 덮어쓰지 않게")
    if args.closed_book and args.prompt is not None:
        parser.error(
            f"--closed-book 은 프롬프트가 {CLOSED_BOOK_PROMPT} 로 정해져 있어 --prompt 를 받지 않는다"
        )
    if args.rag_tag is not None and args.step != "closed-book-report":
        parser.error("--rag-tag 는 closed-book-report 에만 쓴다")
    if args.rag_tag is not None and not TAG_PATTERN.fullmatch(args.rag_tag):
        parser.error(f"--rag-tag 는 영문 소문자 · 숫자 · '-' 만 쓴다, got {args.rag_tag!r}")
    if args.base_tag is not None and args.step not in BASE_TAG_STEPS:
        parser.error(f"--base-tag 는 {' · '.join(sorted(BASE_TAG_STEPS))} 에만 쓴다")
    if args.base_tag is not None and not TAG_PATTERN.fullmatch(args.base_tag):
        parser.error(f"--base-tag 는 영문 소문자 · 숫자 · '-' 만 쓴다, got {args.base_tag!r}")
    if args.base_tag is not None and args.base_tag == args.tag:
        parser.error("--base-tag 는 --tag 와 다른 기록을 준다")
    if args.tag is not None and args.step not in TAG_STEPS:
        parser.error(f"--tag 는 {' · '.join(sorted(TAG_STEPS))} 에만 쓴다")
    if args.tag is not None and not TAG_PATTERN.fullmatch(args.tag):
        parser.error(f"--tag 는 영문 소문자 · 숫자 · '-' 만 쓴다, got {args.tag!r}")
    if args.step in TAG_REQUIRED_STEPS and args.tag is None:
        parser.error(f"{args.step} 는 --tag 로 비교할 기록을 준다")
    if args.prompt is not None and args.step != "run":
        parser.error("--prompt 는 run 에만 쓴다")
    if args.prompt is not None and args.tag is None:
        parser.error("--prompt 는 --tag 와 함께 쓴다 — 기준 기록을 덮어쓰지 않게")
    if args.prompt is not None and repeated_stages(args.prompt):
        parser.error(f"--prompt 는 단계마다 하나만 준다, got {args.prompt}")
    if args.ids is not None and args.step not in IDS_STEPS:
        parser.error(f"--ids 는 {' · '.join(sorted(IDS_STEPS))} 에만 쓴다")
    return args


def _id_list(value: str) -> tuple[str, ...]:
    """--ids 값(쉼표 목록)의 문항 id. 빈 칸 · 겹친 id 는 받지 않는다."""
    ids = tuple(part.strip() for part in value.split(","))
    if not all(ids) or len(set(ids)) != len(ids):
        raise argparse.ArgumentTypeError(f"쉼표로 나눈 서로 다른 문항 id 를 준다, got {value!r}")
    return ids


def sample() -> None:
    """F9 표본(KorQuAD + 위키 밖)을 쓴다."""
    korquad, items = subset_korquad(SUBSET)
    paragraphs = korquad_paragraphs(korquad)
    texts = document_texts(PARSED, {item.title for item in items})
    offwiki = read_jsonl(OFFWIKI_PATH)
    rows, picked = sample_rows(items, paragraphs, texts, offwiki)
    write_jsonl(SAMPLE_PATH, rows)

    kept = sum(row["source"] == "korquad" for row in rows)
    print(
        f"KorQuAD dev 질문 {len(items):,} → eval 몫 문단마다 한 문항 {picked} "
        f"→ 정답이 코퍼스에 남은 것 {kept} (빠짐 {picked - kept}, {1 - kept / picked:.1%})\n"
        f"위키 밖 질문 {len(offwiki)}\n"
        f"저장: {SAMPLE_PATH} ({len(rows)}줄)"
    )


def run(
    limit: int | None,
    *,
    sample_path: Path,
    answers_path: Path,
    prompts: Sequence[str] = (),
    ids: Sequence[str] | None = None,
) -> None:
    """표본을 답변 경로로 돌려 문항마다 기록을 이어 쓴다."""
    items = load_items(sample_path)
    if ids is not None:
        items = select_items(items, ids)
    config = ApiConfig.from_env()
    # /ask 와 같이 Kiwi 하나(약 500MB)를 질문 관문과 BM25 가 나눠 쓴다
    kiwi = Kiwi()
    # 프롬프트를 바꿔 다시 잴 때도 configs/answer.yaml 은 그대로 둔다 — 채택은 측정 뒤 사람이 정한다
    setup = AnswerSetup.from_config(
        config, tokenizer=KiwiTokenizer(kiwi), prompts=prompts, log_questions=False
    )
    refuse_prompt_change(answers_path, setup.policy.profiles)
    settings = RunSettings.from_config(
        config,
        setup.policy,
        gate_enabled=setup.gate is not None,
        dump_date=setup.dump_date,
        sample_path=sample_path,
    )
    _append_run_meta(answers_path, settings, limit=limit, ids=ids)
    print(
        f"문항 {len(items)} · collection={config.search.collection} · 관문 {'켬' if setup.gate else '끔'}",
        flush=True,
    )
    start = time.perf_counter()
    with PgStore.connect(config.pg) as pg, OllamaLLM.connect(OllamaConfig.from_env()) as llm:
        service = SearchService.build(
            config.search, pg, QdrantStore.from_env(), analyzer=KiwiAnalyzer(kiwi)
        )
        print(f"검색 서비스 준비 {time.perf_counter() - start:.0f}s", flush=True)
        runner = AnswerRun(
            service, lambda search: AnswerService.from_config(setup, search=search, llm=llm)
        )
        summary = asyncio.run(runner.run(items, answers_path, limit=limit, on_record=_progress))
    print(
        f"이번 실행 {summary.recorded} · 건너뜀 {summary.skipped} · 다시 돈 생성 실패 {summary.retried} "
        f"/ 문항 {summary.total} · {time.perf_counter() - start:.0f}s\n저장: {answers_path}"
    )


def run_closed_book(
    limit: int | None,
    *,
    sample_path: Path,
    answers_path: Path,
    ids: Sequence[str] | None = None,
) -> None:
    """표본을 검색 없이 질문만으로 돌려 기록을 이어 쓴다."""
    items = load_items(sample_path)
    if ids is not None:
        items = select_items(items, ids)
    policy = load_answer_policy(ApiConfig.from_env().config_dir)
    # 모델 · 생성 설정(gemma4:12b · think off · max_tokens · temperature) · 시간 제한은 grounded 프로필 그대로
    profile = policy.profiles["grounded"]
    library = PromptLibrary.load(PROMPT_DIR)
    today = korean_date(datetime.now(tz=UTC).astimezone().date())
    prompt = compose([library.get(CLOSED_BOOK_PROMPT)], {"today": today})
    settings = ClosedBookSettings(
        spec_id=profile.spec.spec_id,
        profile="grounded",
        system=(CLOSED_BOOK_PROMPT,),
        prompt_version=prompt.version,
        timeout_sec=profile.timeout_sec,
        sample=str(sample_path),
    )
    _append_run_meta(answers_path, settings, limit=limit, ids=ids)
    print(f"문항 {len(items)} · closed-book {profile.spec.spec_id} · {prompt.version}", flush=True)
    start = time.perf_counter()
    with OllamaLLM.connect(OllamaConfig.from_env()) as llm:
        answerer = ClosedBookAnswerer(
            llm,
            profile.spec,
            system=prompt.text,
            prompt_version=prompt.version,
            timeout_sec=profile.timeout_sec,
        )
        runner = AnswerRun(NoSearch(), lambda search: answerer)
        summary = asyncio.run(runner.run(items, answers_path, limit=limit, on_record=_progress))
    print(
        f"이번 실행 {summary.recorded} · 건너뜀 {summary.skipped} · 다시 돈 생성 실패 {summary.retried} "
        f"/ 문항 {summary.total} · {time.perf_counter() - start:.0f}s\n저장: {answers_path}"
    )


def load_items(sample_path: Path) -> list[EvalItem]:
    """표본 파일의 평가 문항. 롱테일 세트(LONGTAIL_SET)는 답 있음 문항만 평가 문항 꼴로 바꾼다."""
    rows = read_jsonl(sample_path)
    if sample_path == LONGTAIL_SET:
        return longtail_items(rows)
    return [EvalItem.from_json(row) for row in rows]


def _progress(index: int, total: int, record: AnswerRecord) -> None:
    print(
        f"  {index}/{total} {record.item.id} {stage_path(record)} {record.wall_ms / 1000:.1f}s",
        flush=True,
    )


def _append_run_meta(
    answers_path: Path,
    settings: RunSettings | ClosedBookSettings,
    *,
    limit: int | None,
    ids: Sequence[str] | None,
) -> None:
    """실행마다 설정을 meta 의 runs 에 덧붙인다 — 멈췄다 다시 돌린 사이에 설정이 바뀌었는지 리포트가 본다."""
    _append_meta(
        answers_path,
        {
            "started_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
            "limit": limit,
            # 문항을 지정해 돌렸으면 그 id (limit 처럼 실행 범위라 settings 밖에 둔다 — 비교 리포트가 settings 만 견준다)
            "ids": list(ids) if ids is not None else None,
            "settings": settings.to_json(),
        },
    )


def _append_meta(answers_path: Path, entry: Mapping[str, Any]) -> None:
    """기록 옆 meta 파일의 runs 에 실행 하나(entry)를 덧붙인다 (없으면 만든다)."""
    append_run_meta(answers_path, entry)


def retrieval_sensitivity() -> None:
    """패러프레이즈 표본을 원문 · 패러프레이즈로 검색해 정답 근거 순위를 쓴다."""
    rows = [row for row in read_jsonl(PARAPHRASE_PATH) if row["sample"] == PARAPHRASE_SAMPLE]
    config = ApiConfig.from_env()
    print(
        f"collection={config.search.collection} · 문항 {len(rows)} × 원문 · 패러프레이즈",
        flush=True,
    )
    start = time.perf_counter()
    with PgStore.connect(config.pg) as pg:
        service = SearchService.build(config.search, pg, QdrantStore.from_env())
        print(f"검색 서비스 준비 {time.perf_counter() - start:.0f}s", flush=True)
        originals = asyncio.run(
            search_all(
                service.search, [row["question"] for row in rows], on_progress=_search_progress
            )
        )
        paraphrases = asyncio.run(
            search_all(
                service.search, [row["paraphrase"] for row in rows], on_progress=_search_progress
            )
        )

    out = [
        sensitivity_row(row, original, paraphrase)
        for row, original, paraphrase in zip(rows, originals, paraphrases, strict=True)
    ]
    write_jsonl(SENSITIVITY_PATH, out)
    meta = {
        "created_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "collection": config.search.collection,
        "model": config.search.model,
        "candidates": config.search.candidates,
        "rerank_candidates": config.search.rerank_candidates,
        "reranker_model": config.search.reranker_model,
        "top_k": config.search.top_k,
        "inputs": {"paraphrase": str(PARAPHRASE_PATH), "sample": PARAPHRASE_SAMPLE},
    }
    write_json(meta_path(SENSITIVITY_PATH), meta)
    table = paired_hits(hit_pairs(out, config.search.top_k))
    print(
        f"hit@{config.search.top_k} 원문 {format_percent(table.original_rate)} · "
        f"패러프레이즈 {format_percent(table.paraphrase_rate)} "
        f"(원문만 {table.original_only} · 패러프레이즈만 {table.paraphrase_only}, "
        f"McNemar p={table.mcnemar_p:.3g})\n저장: {SENSITIVITY_PATH} · {meta_path(SENSITIVITY_PATH)}"
    )


def _search_progress(done: int, total: int) -> None:
    print(f"  {done}/{total}", flush=True)


def sensitivity_row(
    row: Mapping[str, Any], original: SearchResult, paraphrase: SearchResult
) -> dict[str, Any]:
    """패러프레이즈 한 줄의 원문 · 패러프레이즈 검색 결과 (정답 근거 청크 순위 — 질문 원문은 싣지 않는다)."""
    item = KorQuADItem(
        id=row["id"],
        title=normalize_title(row["title"]),
        question=row["question"],
        answer=row["answers"][0],
        source_title=row["title"],
    )
    return {
        "id": row["id"],
        "title": item.title,
        "original": _search_fields(original, item),
        "paraphrase": _search_fields(paraphrase, item),
    }


def _search_fields(result: SearchResult, item: KorQuADItem) -> dict[str, Any]:
    return {
        "answer_rank": answer_rank(result.hits, item),
        "top_score": result.hits[0].score if result.hits else None,
        "chunk_ids": [hit.chunk.id for hit in result.hits],
        "took_ms": result.took_ms,
    }


def pool() -> None:
    """eval 몫 문항의 1위 점수 풀을 이어 쓴다."""
    korquad, items = subset_korquad(SUBSET)
    paragraphs = korquad_paragraphs(korquad)
    split_items = pool_items(items, paragraphs, "eval")
    done = {row["id"] for row in read_jsonl(POOL_PATH)} if POOL_PATH.exists() else set()
    pending = [item for item in split_items if item.id not in done]
    texts = document_texts(PARSED, {item.title for item in pending})
    config = ApiConfig.from_env()
    print(
        f"eval 몫 질문 {len(split_items)} · 남은 것 {len(pending)} · "
        f"collection={config.search.collection}",
        flush=True,
    )
    start = time.perf_counter()
    with PgStore.connect(config.pg) as pg:
        service = SearchService.build(config.search, pg, QdrantStore.from_env())
        print(f"검색 서비스 준비 {time.perf_counter() - start:.0f}s", flush=True)
        asyncio.run(_score_pool(service, pending, paragraphs, texts))

    meta = {
        "created_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "collection": config.search.collection,
        "model": config.search.model,
        "candidates": config.search.candidates,
        "rerank_candidates": config.search.rerank_candidates,
        "reranker_model": config.search.reranker_model,
        "top_k": config.search.top_k,
    }
    rows = read_pool()
    meta["inputs"] = {
        "korquad_dev": str(korquad),
        "splits": sorted({split_of(row.paragraph) for row in rows}),
    }
    write_json(meta_path(POOL_PATH), meta)
    counts = Counter(score_band(row.top_score) for row in rows)
    print(
        " · ".join(f"{band} {counts[band]}" for band in score_bands() if counts[band])
        + f" · {time.perf_counter() - start:.0f}s\n저장: {POOL_PATH} · {meta_path(POOL_PATH)}"
    )


async def _score_pool(
    service: SearchService,
    items: Sequence[KorQuADItem],
    paragraphs: Mapping[str, str],
    texts: Mapping[str, str],
) -> None:
    # 서비스가 질의를 한 번에 하나씩 돌리므로 순서대로 기다린다. 문항마다 덧붙여 끊겨도 끝난 문항은 남는다
    for index, item in enumerate(items, start=1):
        result = await service.search(item.question)
        row = pool_row(item, paragraphs[item.id], texts.get(item.title), result)
        append_jsonl(POOL_PATH, row.to_json())
        if index % 100 == 0:
            print(f"  {index}/{len(items)}", flush=True)


def strata_sample(middle_size: int | None) -> None:
    """점수 풀에서 층화 표본을 쓴다."""
    rows = read_pool()
    picked = strata_rows(rows, seed=SEED, middle_size=middle_size)
    write_jsonl(STRATA_SAMPLE_PATH, [row.to_json() for row in picked])
    counts = Counter((split_of(row.paragraph), score_band(row.top_score)) for row in picked)
    paragraphs = {
        band: len({row.paragraph for row in picked if score_band(row.top_score) == band})
        for band in {band for _, band in counts}
    }
    print(
        f"점수 풀 {len(rows)} → "
        + " · ".join(f"{split} {band} {count}" for (split, band), count in sorted(counts.items()))
        + " (문단 "
        + " · ".join(f"{band} {count}" for band, count in sorted(paragraphs.items()))
        + f")\n저장: {STRATA_SAMPLE_PATH} ({len(picked)}줄)"
    )


def judge_input(tag: str | None) -> None:
    """grounded 답의 판정 입력과 열쇠를 쓴다."""
    records = read_records(tagged(STRATA_ANSWERS_PATH, tag))
    rows, key = judge_rows(records, seed=SEED)
    input_path = tagged(JUDGE_INPUT_PATH, tag)
    key_path = tagged(JUDGE_KEY_PATH, tag)
    write_jsonl(input_path, rows)
    write_jsonl(key_path, key)
    print(f"기록 {len(records)} → grounded 로 끝난 답 {len(rows)}\n저장: {input_path} · {key_path}")


def strata_report(report_date: date | None = None) -> None:
    """high 확정 리포트를 쓴다."""
    records = read_records(STRATA_ANSWERS_PATH)
    runs = read_run_meta(STRATA_ANSWERS_PATH)
    pool_rows = read_pool()
    verdicts = read_verdicts(None)
    today = report_date or datetime.now(tz=UTC).date()
    text = render_strata_report(records, runs, pool_rows, verdicts, today=today)
    REPORT_DIR.mkdir(exist_ok=True)
    path = REPORT_DIR / f"{today:%Y-%m-%d}-high-threshold.md"
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"기록 {len(records)} · 판정 {len(verdicts)} · 리포트: {path}")


def cascade_input(tag: str, base_tag: str | None = None) -> None:
    """grounded → partial 로 넘어간 답의 판정 입력과 열쇠를 쓴다."""
    base = Measurement.load(base_tag)
    other = Measurement.load(tag)
    rows, key = cascade_rows(base, other, seed=SEED)
    # 어느 기준과 견준 판정인지 열쇠에 남긴다 — compare 가 다른 기준으로 그리면 짝이 어긋난다
    key = [{**row, "base_tag": base_tag} for row in key]
    input_path = tagged(CASCADE_INPUT_PATH, tag)
    key_path = tagged(CASCADE_KEY_PATH, tag)
    write_jsonl(input_path, rows)
    write_jsonl(key_path, key)
    counts = Counter((row["record"], row["side"]) for row in key)
    print(
        "판정 입력 "
        + " · ".join(f"{record} {side} {count}" for (record, side), count in sorted(counts.items()))
        + f"\n저장: {input_path} · {key_path}"
    )


def compare(tag: str, report_date: date | None, base_tag: str | None = None) -> None:
    """기준 기록과 꼬리표 기록의 비교 리포트를 쓴다."""
    base = Measurement.load(base_tag)
    other = Measurement.load(tag)
    cascade = read_cascade_verdicts(tag, base_tag)
    today = report_date or datetime.now(tz=UTC).date()
    text = render_compare_report(
        base, other, tag=tag, today=today, cascade=cascade, base_tag=base_tag
    )
    REPORT_DIR.mkdir(exist_ok=True)
    path = REPORT_DIR / f"{today:%Y-%m-%d}-{tag}.md"
    path.write_text(text, encoding="utf-8", newline="\n")
    print(
        f"F9 {len(other.f9)} · 층화 {len(other.strata)} · 판정 {len(other.verdicts)} · "
        f"넘어간 답 판정 {len(cascade) if cascade else '없음'} · 리포트: {path}"
    )


def closed_book_report(tag: str, rag_tag: str | None, report_date: date | None) -> None:
    """closed-book 대 RAG 리포트를 쓴다."""
    f9 = SidePair.load(ANSWERS_PATH, tag=tag, rag_tag=rag_tag)
    longtail = None
    if all(tagged(LONGTAIL_ANSWERS_PATH, name).exists() for name in (tag, rag_tag)):
        longtail = SidePair.load(LONGTAIL_ANSWERS_PATH, tag=tag, rag_tag=rag_tag)
    kinds = question_kinds(r.item for r in f9.rag if r.item.source == "korquad")
    today = report_date or datetime.now(tz=UTC).date()
    text = render_closed_book_report(f9, longtail, kinds, tag=tag, rag_tag=rag_tag, today=today)
    REPORT_DIR.mkdir(exist_ok=True)
    path = REPORT_DIR / f"{today:%Y-%m-%d}-closed-book.md"
    path.write_text(text, encoding="utf-8", newline="\n")
    print(
        f"F9 {len(f9.closed)} · 롱테일 {len(longtail.closed) if longtail else '아직'} · 리포트: {path}"
    )


def tally(prefix: str, ids: Sequence[str] | None) -> None:
    """회차 기록들의 단계 경로 · 표지를 세어 찍는다."""
    pattern = tagged(ANSWERS_PATH, f"{prefix}-*").name
    rounds: dict[str, list[AnswerRecord]] = {}
    prompts: dict[str, str] = {}
    for path in sorted(ANSWERS_PATH.parent.glob(pattern)):
        tag = path.stem.removeprefix(f"{ANSWERS_PATH.stem}-")
        rounds[tag] = [r for r in read_records(path) if ids is None or r.item.id in ids]
        # 멈췄다 다시 돌린 사이에 프롬프트가 바뀌었으면 둘 다 보인다
        prompts[tag] = " / ".join(sorted({_grounded_prompt(entry) for entry in read_runs(path)}))
    if not rounds:
        raise SystemExit(
            f"{ANSWERS_PATH.parent / pattern} 가 없다 — run --tag {prefix}-…-N 을 먼저 돌린다"
        )
    print(render_tally(rounds, prompts))


def _grounded_prompt(entry: Mapping[str, Any]) -> str:
    # entry = meta runs 의 한 줄 (실행 하나의 설정)
    return entry["settings"]["profiles"]["grounded"]["system"][-1]


def report(tag: str | None) -> None:
    """F9 자동 지표 리포트를 쓴다."""
    answers_path = tagged(ANSWERS_PATH, tag)
    if not answers_path.exists():
        raise SystemExit(f"{answers_path} 가 없다 — run 을 먼저 돌린다")
    records = read_records(answers_path)
    runs = read_run_meta(answers_path)
    sample_count = len(read_jsonl(SAMPLE_PATH))
    sensitivity: tuple[list[dict[str, Any]], dict[str, Any]] | None = None
    if SENSITIVITY_PATH.exists():
        sensitivity = (read_jsonl(SENSITIVITY_PATH), read_json(meta_path(SENSITIVITY_PATH)))
    today = datetime.now(tz=UTC).date()
    text = render_report(
        records, runs, sample_count, sensitivity, today=today, answers_path=answers_path
    )
    REPORT_DIR.mkdir(exist_ok=True)
    path = REPORT_DIR / f"{today:%Y-%m-%d}-answer-eval.md"
    # Windows 에서도 LF 로 (리포트는 커밋하고 .gitattributes 가 LF)
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"기록 {len(records)} / 표본 {sample_count} · 리포트: {path}")


if __name__ == "__main__":
    main()
