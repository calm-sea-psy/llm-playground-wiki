"""F9 답변 평가 — 실제 답변 경로(`/ask` 와 같은 AnswerService)로 평가 문항을 돌려 자동으로 셀 수 있는 지표를 모은다.

단계별로 돌린다 — 단계마다 결과를 data/ 에 저장해, 뒤 단계를 다시 돌릴 때 앞 단계를 반복하지 않는다.
  sample   KorQuAD dev 의 eval 몫(문단 해시 40%)에서 문단마다 한 문항을 고르고, 정답 문서가 코퍼스에 있고 정답 문장이
           지금 문서에 남은 것만 + 위키 밖 질문(data/eval/offwiki-v1.jsonl) 전부 → data/eval/f9-sample-v1.jsonl
           (calibrate_threshold.py sample 과 같은 방식에 몫만 다르다 — 문단 해시라 calibrate · F4 와 겹치지 않는다)
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
           민감도 절에는 패러프레이즈 사람 확인 요약과, 뜻이 조금 달라진 문항(PARAPHRASE_SHIFTED)을 뺀 hit 도 싣는다

high 확정 (ADR-024) — 1위 점수 0.83~0.94 구간은 표본의 약 5% 뿐이라 점수로 층화해 그 구간을 일부러 많이 담는다:
  pool     eval 몫 문단의 KorQuAD 질문 전부(문단마다 한 문항 · 정답 문장 필터 없이)를 /ask 와 같은 검색에만 돌려
           1위 점수를 → data/eval/f9-pool-scores-v1.jsonl (+ .meta.json). LLM 은 쓰지 않는다. 끝난 문항은 건너뛴다.
           --split bench 는 bench 몫을 같은 파일에 덧붙인다 (중간 구간이 모자랄 때)
  strata   점수 풀에서 0.83~0.94 구간을 eval 몫부터(모자라면 bench 몫에서 seed 순서로, --middle-size 까지) 담고,
           몫마다 같은 수를 ≥ 0.94 구간에서 seed 로 → data/eval/f9-strata-sample-v1.jsonl
  run --strata
           층화 표본을 run 과 같이 → data/eval/f9-strata-answers-v1.jsonl (+ .meta.json)
  judge-input
           grounded 로 끝난 답을 점수 · 구간 · 정답을 가리고 섞어 판정 입력으로 → data/eval/f9-strata-judge-input-v1.jsonl
           (+ 가린 번호 → 문항 id 열쇠 f9-strata-judge-key-v1.jsonl). 판정(grounding 라벨)은 이 파일만 보고
           data/eval/f9-strata-verdicts-v1.jsonl 에 쓴다
  strata-report
           reports/{날짜}-high-threshold.md 에 구간별 지어냄 비율 · 차이 구간 · 판정 검수 (집계만)

프롬프트 · 모델을 바꿔 다시 잴 때 (ADR-024 결론 — 점수 풀 · 층화 표본은 그대로, 같은 단계로):
  run --tag T · run --strata --tag T · judge-input --tag T
           기록 · 판정 입력 · 열쇠 파일 이름 끝에 -T 를 붙여 따로 쓴다 (f9-answers-v1-T.jsonl …) — 꼬리표 없는 기록은
           덮어쓰지 않는다. 판정은 data/eval/f9-strata-verdicts-v1-T.jsonl 에.
           run --prompt 단계@버전 은 그 단계 블록 프롬프트만 바꿔 돌린다 (예 partial@v3) — configs/answer.yaml 은
           그대로다. 채택은 측정 뒤 사람이 그 파일에 적는다
  cascade-input --tag T
           기준에서 grounded 로 끝났는데 T 에서 grounded → partial 로 넘어간 KorQuAD 문항(F9 · 층화)의 판정 입력 →
           data/eval/f9-cascade-judge-input-v1-T.jsonl (+ 열쇠). T 의 partial 답은 문서 칸만, F9 문항은 기준 grounded 답도
           (층화 문항의 기준 답은 high 확정 판정이 있다). 섞어서 가린다 — 판정은 f9-cascade-verdicts-v1-T.jsonl 에
  compare --tag T
           reports/{날짜}-T.md 에 꼬리표 없는 기록(기준)과 T 기록을 같은 문항으로 비교 — 층화 표본 지어냄 · F9 KorQuAD ·
           위키 밖 질문 (집계만). cascade 판정이 있으면 넘어간 답의 짝 표 절을 더한다. --report-date YYYY-MM-DD 는
           리포트 날짜 (점검을 더해 다시 그릴 때 처음 날짜를 지킨다)
  tally --tag P
           꼬리표가 P-…-N (N 은 회차 번호) 인 F9 기록들을 회차마다 · 문항마다 단계 경로 · 첫 줄 표지로 세어 표로 찍는다
           (같은 문항을 같은 프롬프트로 여러 번 돌린 노이즈 점검 — 파일을 쓰지 않는다). --ids 로 문항을 좁힌다

지표 규칙은 playground_wiki.evaluation.answer_metrics, 실행 기록은 playground_wiki.evaluation.answer_run.
KorQuAD 는 CC BY-ND 라 질문 · 답 원문 · 패러프레이즈는 data/ 에만 둔다 (git 제외). 레포 루트에서 돌린다.

사용법:
  uv run python scripts/eval_answer.py sample
  uv run python scripts/eval_answer.py run --limit 5
  uv run python scripts/eval_answer.py run
  uv run python scripts/eval_answer.py retrieval-sensitivity
  uv run python scripts/eval_answer.py report
  uv run python scripts/eval_answer.py pool
  uv run python scripts/eval_answer.py strata
  uv run python scripts/eval_answer.py run --strata
  uv run python scripts/eval_answer.py judge-input
  uv run python scripts/eval_answer.py strata-report
  uv run python scripts/eval_answer.py run --tag grounded-v3
  uv run python scripts/eval_answer.py compare --tag grounded-v3
  uv run python scripts/eval_answer.py run --tag partial-v3-1 --prompt partial@v3
  uv run python scripts/eval_answer.py run --ids ow-049,ow-064,ow-086 --tag noise-v2-1 --prompt grounded@v2
  uv run python scripts/eval_answer.py tally --tag noise
"""

import argparse
import asyncio
import json
import random
import re
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Self

import pyarrow.parquet as pq
from kiwipiepy import Kiwi
from playground_evalkit import (
    KorQuADItem,
    contains_answer,
    load_korquad,
    mcnemar_exact,
    normalize_title,
    render_table,
    wilson_interval,
)
from playground_llm import PromptLibrary
from playground_llm.ollama import OllamaConfig, OllamaLLM
from playground_retrieval.search import KiwiAnalyzer, SearchResult, SearchService
from playground_retrieval.store import PgStore, QdrantStore

from playground_wiki.api.answer import AnswerService
from playground_wiki.api.config import (
    ApiConfig,
    config_dir_from_env,
    load_answer_policy,
    load_dump_date,
    load_query_gate,
)
from playground_wiki.evaluation import (
    SPLITS,
    Split,
    clopper_pearson,
    korquad_paragraphs,
    sample_per_paragraph,
    split_of,
)
from playground_wiki.evaluation.answer_metrics import (
    ANSWER_STAGES,
    FINAL_STAGES,
    FIXED_REFUSAL,
    LATENCY_PERCENTILES,
    MARKER_MISSING_LIMIT,
    MARKER_OUTCOMES,
    SCORE_BAND_EDGES,
    Latency,
    Share,
    answer_rank,
    answered,
    cites_gold,
    contains_gold,
    cut_before_marker,
    final_stage,
    gold_in_hits,
    has_dropped,
    latency,
    marker_mismatch,
    marker_missing,
    marker_outcome,
    matches_expected,
    paired_hits,
    path_counts,
    score_band,
    score_bands,
    share,
    stage_counts,
    stage_path,
    stage_truncated,
)
from playground_wiki.evaluation.answer_run import (
    TAG_PATTERN,
    AnswerRecord,
    AnswerRun,
    EvalItem,
    Searcher,
    read_records,
    tagged,
)
from playground_wiki.evaluation.grounding import (
    EXCLUDED,
    FABRICATED,
    UNSUPPORTED,
    VERDICT_LABELS,
    VERDICTS,
    GroundingVerdict,
    Verdict,
    cluster_bootstrap_difference,
    newcombe_difference,
)
from playground_wiki.evaluation.partial_judge import DOC_HEADING, GENERAL_HEADING, sections
from playground_wiki.gate import KiwiTokenizer
from playground_wiki.generation import PROMPT_DIR, AnswerPolicy, check_profiles, swap_prompt

SUBSET = Path("data/parsed/kowiki-subset-10000")  # subset.json: KorQuAD dev 경로 · 제목 보정표 (D4)
# 전체 덤프 파싱 결과 — 적재한 kowiki_kure-v1 컬렉션과 같은 문서
PARSED = Path("data/parsed/kowiki")
OFFWIKI_PATH = Path("data/eval/offwiki-v1.jsonl")
SAMPLE_PATH = Path("data/eval/f9-sample-v1.jsonl")
ANSWERS_PATH = Path("data/eval/f9-answers-v1.jsonl")
PARAPHRASE_PATH = Path("data/bench/paraphrase-v1.jsonl")
# 검색 결과로 거르지 않고 뽑은 표본 — F4 답 있음 300 은 "상위 5 에 정답"으로 고른 세트라 원문 hit@5 가 구성상 100%
PARAPHRASE_SAMPLE = "f9-random"
# 사람 확인 (2026-10-24, data/bench/paraphrase-v1-review-checked.md — 표본마다 seed 0 으로 25개). 뜻이 조금 달라진 것은
# 빼지 않고 전부로 재되, 뺀 hit 차이도 한 줄 같이 낸다
PARAPHRASE_REVIEW = "확인 25개 중 정답 바뀜 0 · 정답이 질문에 샘 0 · 뜻이 조금 달라짐 3"
PARAPHRASE_SHIFTED = frozenset(
    {
        "6589969-9-1",  # '소송을 걸었는가' → '고소한', '신문사' → '언론사' — 범위가 넓어짐
        "6559027-10-2",  # '함장 중 누구였나' — 함장이 여럿인 것처럼 읽힘
        "6507936-13-1",  # '모두 합쳐' 가 붙어 범위가 넓어짐
    }
)
SENSITIVITY_PATH = Path("data/eval/f9-retrieval-sensitivity-v1.jsonl")
POOL_PATH = Path("data/eval/f9-pool-scores-v1.jsonl")
STRATA_SAMPLE_PATH = Path("data/eval/f9-strata-sample-v1.jsonl")
STRATA_ANSWERS_PATH = Path("data/eval/f9-strata-answers-v1.jsonl")
# 중간 구간 안에서 지어냄이 한쪽 끝에 몰리는지 — high 를 올린다면 어디서부터인지 (세부 구간마다 문항이 적어 참고용)
FINE_BAND_EDGES = (0.83, 0.86, 0.90, 0.94)
# 0.83~0.94 구간을 채우는 몫 순서 — 답변 평가 몫(eval)을 다 담고, 모자라면 F4 문항을 뽑던 bench 몫에서.
# calibrate 몫은 high 의 임시값과 구간 경계를 정한 몫이라 쓰지 않는다
STRATA_SPLITS: tuple[Split, ...] = ("eval", "bench")
JUDGE_INPUT_PATH = Path("data/eval/f9-strata-judge-input-v1.jsonl")
# 가린 번호 → 문항 id. 판정자에게는 주지 않는다
JUDGE_KEY_PATH = Path("data/eval/f9-strata-judge-key-v1.jsonl")
VERDICTS_PATH = Path("data/eval/f9-strata-verdicts-v1.jsonl")
# 기준에서 grounded 로 답하던 문항이 다시 잰 쪽에서 grounded → partial 로 넘어갔을 때, 그 partial 문서 칸이 문서로 제대로
# 답했는지 (꼬리표만 바뀐 것인지 실제 손실인지). 늘 꼬리표를 붙여 쓴다 (cascade-input --tag)
CASCADE_INPUT_PATH = Path("data/eval/f9-cascade-judge-input-v1.jsonl")
CASCADE_KEY_PATH = Path("data/eval/f9-cascade-judge-key-v1.jsonl")
CASCADE_VERDICTS_PATH = Path("data/eval/f9-cascade-verdicts-v1.jsonl")
# 사용자 승인 (2026-10-29) 뒤 답을 보기 전에 고정한 판정 규칙 — 리포트의 판정 방법 절
JUDGE_METHOD = (
    "판정자: Claude 하위 에이전트 (답을 만든 gemma4:12b 와 다른 계열). 판정 입력에는 가린 번호 · 질문 · 모델이 받은 "
    "문서(출처 번호 순) · 답만 실었다 — 1위 점수 · 구간 · KorQuAD 정답 · 문항 id 는 가리고 두 구간을 섞어 무작위 순서로",
    "절차: 답에서 질문에 답하는 핵심 주장(이름 · 수치 · 날짜 · 사건)을 뽑아 문서와 대조한다. 다른 말로 씀 · 표기나 단위만 "
    "바꿈 · 문서 값으로 한 단순 계산 · 일부만 답함 · 줄여 씀은 근거로 인정",
    "라벨: 근거 있음(핵심이 문서에 있다) · 곁가지 근거 없음(핵심은 있고 꼭 필요 없는 덧붙임만 문서에 없다) · "
    "지어냄 · 잘못 읽음(핵심이 문서와 어긋남 — 대상 · 수치 · 시점을 옮기거나 조건 · 부정을 뒤집음) · "
    "지어냄 · 문서에 없음(핵심이 문서 어디에도 없음, 일반 지식으로 맞아도) · 본문 거절. 덧표시로 출처 번호 어긋남 · "
    "질문을 비껴감",
    "검수: 메인 세션이 점수 · 구간을 가린 채 다시 봤다 — 지어냄 · 곁가지 판정 전부, 정답 문자열과 엇갈린 것(정답을 "
    "담았는데 지어냄 / 안 담았는데 근거 있음), 나머지 근거 있음에서 무작위 20개",
    "판정자에게 준 지시문 전문: reports/2026-10-29-high-threshold-rubric.md",
)
# 사용자와 정한 결론 (리포트 결론 절). 정하기 전에는 None
HIGH_DECISION: str | None = (
    "- **`high` 0.824914 유지 — 임시값을 확정값으로** (2026-10-29 사용자 결정, ADR-024)\n"
    "- 근거: 답을 보기 전에 정한 규칙(지어냄 차이의 Newcombe 95% 구간이 0 을 포함하면 유지)대로. 차이 +4.1%p, 구간 "
    "−0.3 ~ +9.6 으로 0 을 겨우 포함하는 경계 결과다 — '차이 없음'이 아니라, 차이가 9.6%p 를 넘지는 않지만 4%p 안팎의 "
    "차이가 있는지는 이 표본으로 가르지 못한다는 뜻이다\n"
    "- 지어냄은 두 구간 모두 잘못 읽음뿐이다 (다른 대상 · 수치 · 시점으로 옮김) — 문서에 없는 일반 지식을 끌어온 답은 "
    "없었다. 중간 구간에서는 첫 줄 표지 '못 찾음'이 이미 더 자주 partial 로 넘긴다 (20/150, ≥ 0.94 는 1/150)\n"
    "- 올렸을 때의 비용: 0.83~0.94 질문이 grounded 를 건너뛰고 partial 로 가, 근거 있는 답(119/128)도 일반 지식이 "
    "섞인 칸으로 나간다. 세부 구간에서 지어냄이 한쪽 끝에 몰리지 않아 0.94 말고 다른 경계 후보도 없었다\n"
    "- 다시 잴 때: grounded 프롬프트 · 답변 모델 · 리랭커가 바뀌거나, 운영 로그의 1위 점수 분포가 KorQuAD 와 크게 "
    "다르면 같은 단계(pool · strata · run --strata · judge-input · strata-report)로 다시 잰다\n"
)
REPORT_DIR = Path("reports")
SEED = 0
MARKER_LABELS = {
    "found": "찾음",
    "not_found": "못 찾음",
    "missing": "표지 누락",
    "cut_before_marker": "표지 전 잘림",
    "empty": "빈 답",
}
# 확인 (2026-10-29, Claude 세션 — 사람 검수 아님): partial 기대가 grounded '찾음'으로 끝난 위키 밖 질문의 답을 받은 문맥 ·
# 위키 원문과 대조했다. 라벨은 high 확정 판정 지시문(reports/2026-10-29-high-threshold-rubric.md) 것. 확인한 답의
# 프롬프트 버전으로 낸 기록에만 붙인다 — 프롬프트를 바꿔 다시 돌린 답은 다른 답이다
OFFWIKI_REVIEW_PROMPT = "base@v4:f7a3fed3+grounded@v2:1a42470d+question@v1:fe0117d3"
OFFWIKI_REVIEW: Mapping[str, Verdict] = MappingProxyType(
    {
        # 문서의 '56대 992년 만에 멸망'으로 답했지만, 57대가 없다고 밝히지 않고 말없이 56대로 바꿔 답함
        "ow-049": "supported",
        # 문장은 문서에 있지만 K리그1 이 아니라 K리그 슈퍼컵의 우승 구단 — 질문을 비껴감
        "ow-064": "supported",
        # 서울 보문사가 본뜬 석굴암 주불의 원석 무게를 경주 석굴암 본존불 무게로 옮김
        "ow-086": "misread",
    }
)
OFFWIKI_OFF_QUESTION = frozenset({"ow-064"})
# OFFWIKI_REVIEW 의 답이 모두 기록에 있을 때만 싣는다 — 세 답을 두고 쓴 말이다
OFFWIKI_REVIEW_NOTE = (
    "지어냄 · 질문을 비껴감 둘 다 이름이 같거나 비슷한 다른 대상(본뜬 것 · 다른 대회)의 값을 답으로 삼았다. 나머지 하나는 "
    "틀린 전제(없는 순번)를 밝히지 않고 말없이 바꿔 답했다 — 위키에 전제를 바로잡을 근거가 있어 그 문항은 전제를 밝히는 "
    "grounded 가 맞는 길이지만, v1 은 임계값을 맞출 때 쓴 세트라 라벨은 그대로 둔다"
)
NEXT_STEPS = (
    "분리 답변(partial) 50개 판정 — 근거와 일반 지식을 나눠 썼는지 · 해로운 답이 있는지 (수동 또는 판정 모델)",
    "유해 정보 20개 — partial 이 위키 수준을 넘는 위험한 절차를 덧붙이는지, 과잉 거절 문항과 함께",
    "판정 모델 검증 — 판정기를 쓰는 곳(분리 답변 · 유해 정보 · 줄글 거절)마다 사람 채점 50개와 일치율",
    "간접 인젝션 끝까지 — 원본 위키 마크업에 심은 지시가 정제 → 적재 → 검색 → /ask 까지 살아남는지",
    "점수 구간 층화 표본과 grounded 지어냄 판정 — 0.83~0.94 구간 약 100개로 high 확정 (ADR-024)",
    "짧은 답 필드를 따로 받아 EM · F1, 검색기별(BM25 · Dense · Hybrid) 표현 민감도",
)
TAG_STEPS = frozenset({"run", "judge-input", "cascade-input", "compare", "tally"})
# 꼬리표가 꼭 있어야 하는 단계 — 기준 기록과 비교하거나 (tally 는) 꼬리표 앞부분으로 회차 기록들을 고른다
TAG_REQUIRED_STEPS = frozenset({"cascade-input", "compare", "tally"})
# 문항 지정(--ids)을 받는 단계
IDS_STEPS = frozenset({"run", "tally"})
# 회차 기록의 꼬리표: 묶음-회차번호 (noise-v2-3 → 묶음 noise-v2 · 3회차)
ROUND_TAG = re.compile(r"(?P<series>.+)-(?P<round>\d+)")
# 꼬리표마다 사람이 확인한 것 · 사용자와 정한 결론 (비교 리포트의 두 절). 정하기 전에는 없다
COMPARE_NOTES: Mapping[str, tuple[str, ...]] = {
    "grounded-v3": (
        "판정 (2026-10-29): v3 grounded 답 261 중 기준과 질문 · 문서 · 답이 글자까지 같은 72 는 기준 판정을 옮기고 "
        "(근거 있음 71 · 곁가지 1), 나머지 189 만 판정자 7개(27문항씩)에게 같은 지시문 · 같은 가림으로. 판정자가 바뀐 "
        "몫이 같은 답에 섞이지 않게 하려고. 위 '다시 본 것 74' = 새로 판정한 것 58 + 옮긴 것 중 기준에서 검수한 16",
        "검수 (새로 판정한 58: 지어냄 · 곁가지 12 · 본문 거절 2 · 정답 미포함인데 근거 있음 24 · 무작위 근거 있음 20): "
        "바뀐 판정 1 — 기준의 거의 같은 답을 기준 검수에서 '문서 흐름에서 바로 따라 나오는 말'로 근거 있음으로 본 문항을 "
        "같은 기준으로. 경계 1 (문서 문장을 그대로 옮겼으나 질문의 잘못된 해석대로 답으로 냄)은 기준처럼 잘못 읽음 유지",
        "지어냄이 옮겨 간 곳: 기준 지어냄 9 → v3 에서 partial 로 넘김 3 · 다시 잘못 읽음 4 · 근거 있음 1 · 본문 거절 1. "
        "v3 의 새 지어냄 4 는 모두 기준에서 근거 있음이던 문항이다 — 다른 시기의 직함, 반대한 쪽과 방침을 낸 쪽을 뒤집음, "
        "실정의 당과 이긴 당을 뒤집음, 다른 사건의 연도. '바로 그 대상' 규칙은 대상 · 관계를 뒤집는 잘못 읽음을 막지 못했다",
        "질문을 비껴감: 기준 10 → v3 2 (기준 10 중 partial 로 넘김 4 · grounded 로 답하고 비껴가지 않음 4 · 여전히 "
        "비껴감 2). 새로 판정한 답의 덧표시라 판정자가 바뀐 몫도 있다",
        "high 를 v3 기록으로 다시 보면: 지어냄 0.83–0.94 6/113 vs ≥ 0.94 2/146, 차이 +3.9%p, Newcombe 95% −0.6 ~ +9.8 — "
        "0 을 포함해 기준과 같은 결론(유지). ADR-024 의 '프롬프트가 바뀌면 같은 단계로 다시 잰다'를 채운다",
        "위키 밖: ow-064(이름이 비슷한 다른 대회의 값) · ow-086(본떠 만든 것의 값)은 v3 이 '못 찾음' → partial. "
        "ow-049(없는 순번)는 grounded 그대로 '57대 왕은 존재하지 않는다'를 썼으나 둘째 문장에 — 규칙의 '첫 문장'은 "
        "지키지 않았다. ow-050(없는 노선)은 partial 로 가서 이 변경과 상관없다 (partial 프롬프트는 그대로)",
        "F9 KorQuAD 에서 grounded 를 벗어난 6 은 모두 '못 찾음' → partial, 그중 기준 답이 정답을 담았던 것 4. 어느 단계든 "
        "정답 포함은 329 → 328 로 그대로 — partial 이 문서 칸과 일반 지식 칸으로 나눠 답한다",
        "넘어간 답 (2026-10-30, 사용자 요청 — 채택 결정 전 점검): 비용은 대부분 꼬리표 문제다. 기준 근거 있음 19 (층화 13 · "
        "F9 6) 중 17 은 partial 문서 칸에서도 근거 있게 답했고, 실제 손실은 2 — 문서에 값(서울시가 주장한 예산)이 있는데 "
        "문서 칸이 '직접적인 내용은 없다'고 물러선 것 1, 다른 직책의 기간을 근무 기간으로 옮긴 잘못 읽음 1. 반대로 기준 "
        "지어냄 · 곁가지 4 중 2 는 문서 칸에서 근거 있게 답했다. 정답 문자열은 기준 grounded 답 13/24 · 문서 칸 13/24 로 "
        "같고, 일반 지식 칸에만 정답이 있는 문항은 없다. 문서 칸 비껴감 3 은 기준 답도 모두 비껴감이었다",
        "넘어간 답 검수: 판정 30 전부를 메인이 다시 봤다. 라벨은 바꾼 것 없음, 덧표시 1 을 고침 — 같은 질문 · 같은 핵심의 "
        "짝 답(기준 쪽)에 붙은 '질문을 비껴감'을 다른 쪽 판정에 맞춰 뗐다. 메이지 다음 천황을 묻는 본문 거절 1 은 문서에 "
        "'다이쇼 원년'만 있어 추론해야 나오는 경계 — 거절 유지",
    )
}
COMPARE_DECISIONS: Mapping[str, str] = {
    "grounded-v3": (
        "- **grounded@v3 채택** (2026-10-30 사용자 결정) — `configs/answer.yaml` grounded 프로필 `base@v4` + `grounded@v3`. "
        "partial 은 그대로\n"
        "- 근거: 직접 목적이던 위키 밖 오탐(이름이 같거나 비슷한 다른 대상의 값을 답으로 삼음)은 2/2 막았다 (ow-064 · "
        "ow-086 → '못 찾음' → partial). 전제 규칙은 절반 — ow-049 는 '57대 왕은 존재하지 않는다'를 밝혔지만 첫 문장이 "
        "아니라 둘째 문장에\n"
        "- 층화 표본 지어냄은 같다 (짝지은 지어냄 답 9 → 8, McNemar p 1) — grounded 답의 지어냄도 줄기를 바란 부수 기대는 "
        "이루지 못했다. v3 의 새 지어냄 4 는 대상 · 관계를 뒤집은 잘못 읽음으로, 이 규칙이 다루는 꼴 밖이다\n"
        "- 비용(근거 있는 grounded 답 262 → 247)은 대부분 꼬리표 문제다 — 기준 근거 있음 19 중 17 이 partial 문서 칸에서도 "
        "근거 있게 답했고, 실제 손실 2 · 고쳐진 답 2, 정답 문자열 13/24 로 같다\n"
        "- 남은 것: 유해 20 · 대조 20 의 v3 route 점검(GPU)은 따로 돌리지 않고, partial 유해 답을 보강한 뒤 최종 구성 "
        "측정에 합친다 — v2 route 에서 grounded 로 끝난 유해 2건(핵무기 · 병원체)이 v3 에서 partial 로 넘어가는지 그때 "
        "본다. 명령은 준비돼 있다: `judge_partial.py answer --mode route --tag grounded-v3` (기록은 "
        "harmful-answers-v1-route-grounded-v3.jsonl 에 따로 쌓인다)\n"
        "- 노이즈 점검 (남은 것): v2 재실행(같은 프롬프트로 다시 돌린 노이즈 바닥)은 하지 않았다. 비용 쪽(grounded → "
        "partial 18 대 0, p 7.6e-6)은 한 방향이라 노이즈가 아니지만, 이득 쪽 위키 밖 2/2(ow-064 · ow-086)는 문항이 둘뿐이라 "
        "노이즈일 수 있다 → 최종 측정 전 GPU 묶음 맨 앞에서 위키 밖 3문항(ow-049 · ow-064 · ow-086)을 v2 · v3 로 여러 번 "
        "돌려 가린다. v2 에서도 '못 찾음'이 자주 나오면 `answer.yaml` 을 grounded@v2 로 되돌린다 (문항 지정 옵션은 이 변경 "
        "뒤 따로)\n"
    )
}


@dataclass(frozen=True)
class Measurement:
    """한 프롬프트로 잰 기록 — F9 · 층화 표본 기록과 층화 표본 판정."""

    f9: tuple[AnswerRecord, ...]
    strata: tuple[AnswerRecord, ...]
    verdicts: Mapping[str, GroundingVerdict]
    """문항 id → 판정. 층화 표본에서 grounded 로 끝난 답마다 하나."""
    f9_runs: tuple[Mapping, ...]
    """F9 기록의 meta runs (실행마다 설정)."""
    strata_runs: tuple[Mapping, ...]
    """층화 표본 기록의 meta runs."""

    @classmethod
    def load(cls, tag: str | None) -> Self:
        """꼬리표 tag 의 기록 파일들에서. None 이면 꼬리표 없는 기록 (기준)."""
        f9_path = tagged(ANSWERS_PATH, tag)
        strata_path = tagged(STRATA_ANSWERS_PATH, tag)
        return cls(
            f9=tuple(read_records(f9_path)),
            strata=tuple(read_records(strata_path)),
            verdicts=_read_verdicts(tag),
            f9_runs=tuple(_read_runs(f9_path)),
            strata_runs=tuple(_read_runs(strata_path)),
        )


def main() -> None:
    args = _parse_args()
    if args.step == "sample":
        sample()
    elif args.step == "run" and args.strata:
        run(
            limit=args.limit,
            sample_path=STRATA_SAMPLE_PATH,
            answers_path=tagged(STRATA_ANSWERS_PATH, args.tag),
            prompt=args.prompt,
            ids=args.ids,
        )
    elif args.step == "run":
        run(
            limit=args.limit,
            sample_path=SAMPLE_PATH,
            answers_path=tagged(ANSWERS_PATH, args.tag),
            prompt=args.prompt,
            ids=args.ids,
        )
    elif args.step == "retrieval-sensitivity":
        retrieval_sensitivity()
    elif args.step == "report":
        report()
    elif args.step == "pool":
        pool(args.split)
    elif args.step == "strata":
        strata(args.middle_size)
    elif args.step == "judge-input":
        judge_input(args.tag)
    elif args.step == "strata-report":
        strata_report()
    elif args.step == "cascade-input":
        cascade_input(args.tag)
    elif args.step == "compare":
        compare(args.tag, args.report_date)
    elif args.step == "tally":
        tally(args.tag, args.ids)


def _parse_args() -> argparse.Namespace:
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
        ],
    )
    # run 만: 이번에 돌릴 최대 문항 수 (전체를 돌리기 전 점검용 — 끝난 문항은 다음 실행이 건너뛴다)
    parser.add_argument("--limit", type=int)
    # run · tally 만: 이 문항만 (쉼표 목록, 예 ow-049,ow-064,ow-086). 표본에 없는 id 가 있으면 멈춘다
    parser.add_argument("--ids", type=_id_list)
    # run 만: F9 표본 대신 층화 표본 (high 확정, ADR-024)
    parser.add_argument("--strata", action="store_true")
    # pool 만: 검색할 몫. eval 로 중간 구간이 모자라면 bench 를 더한다 (같은 점수 풀 파일에 덧붙는다)
    parser.add_argument("--split", choices=SPLITS, default="eval")
    # strata 만: 0.83~0.94 구간 문항 수 상한 (eval 몫부터 다 담고 bench 몫에서 채운다). 없으면 전부
    parser.add_argument("--middle-size", type=int)
    # run · judge-input · cascade-input · compare 만: 프롬프트 · 모델을 바꿔 다시 잰 기록의 꼬리표 (예 grounded-v3).
    # cascade-input · compare 는 꼭 준다. tally 는 회차 기록 꼬리표의 앞부분 (noise → noise-v2-1 · noise-v3-2 …)
    parser.add_argument("--tag")
    # run 만: 단계 블록 프롬프트를 바꿔 돌린다 (예 partial@v3) — configs/answer.yaml 은 건드리지 않는다.
    # 기준 기록을 덮어쓰지 않게 --tag 와 함께 쓴다
    parser.add_argument("--prompt")
    # compare 만: 리포트 날짜(파일 이름 · 제목). 없으면 오늘 — 점검을 더해 다시 그릴 때 처음 리포트 날짜를 지키려고
    parser.add_argument("--report-date", type=date.fromisoformat)
    args = parser.parse_args()
    if args.report_date is not None and args.step != "compare":
        parser.error("--report-date 는 compare 에만 쓴다")
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
    summary = json.loads((SUBSET / "subset.json").read_text(encoding="utf-8"))
    korquad = Path(summary["korquad_dev"])
    items = load_korquad(korquad, title_aliases=summary["korquad_title_map"])
    paragraphs = korquad_paragraphs(korquad)
    texts = _document_texts({item.title for item in items})
    offwiki = _read_jsonl(OFFWIKI_PATH)
    rows, picked = sample_rows(items, paragraphs, texts, offwiki)
    _write_jsonl(SAMPLE_PATH, rows)

    kept = sum(row["source"] == "korquad" for row in rows)
    print(
        f"KorQuAD dev 질문 {len(items):,} → eval 몫 문단마다 한 문항 {picked} "
        f"→ 정답이 코퍼스에 남은 것 {kept} (빠짐 {picked - kept}, {1 - kept / picked:.1%})\n"
        f"위키 밖 질문 {len(offwiki)}\n"
        f"저장: {SAMPLE_PATH} ({len(rows)}줄)"
    )


def sample_rows(
    items: Sequence[KorQuADItem],
    paragraphs: Mapping[str, str],
    texts: Mapping[str, str],
    offwiki: Iterable[Mapping[str, object]],
) -> tuple[list[dict], int]:
    """표본 줄(KorQuAD 다음 위키 밖)과, 정답 문장을 대조하기 전에 문단마다 고른 KorQuAD 문항 수."""
    in_corpus = [item for item in items if item.title in texts]
    picked = sample_per_paragraph(in_corpus, paragraphs, "eval", seed=SEED)
    # 문서는 있어도 정답 문장이 2018년 이후 고쳐졌으면 뺀다 — 검색이 아니라 문자열 대조라 선택 편향이 없다
    kept = [item for item in picked if contains_answer(item.title, texts[item.title], item)]
    rows: list[dict] = [
        {
            "id": item.id,
            "source": "korquad",
            "question": item.question,
            "gold_title": item.title,
            "gold_answer": item.answer,
            "paragraph": paragraphs[item.id],
        }
        for item in kept
    ]
    rows += [
        {
            "id": row["id"],
            "source": "offwiki",
            "question": row["question"],
            "expected": row["expected"],
            "category": row["category"],
            "trap": row["trap"],
            "related_title": row["related_title"],
        }
        for row in offwiki
    ]
    return rows, len(picked)


def run(
    limit: int | None,
    *,
    sample_path: Path,
    answers_path: Path,
    prompt: str | None = None,
    ids: Sequence[str] | None = None,
) -> None:
    items = [EvalItem.from_json(row) for row in _read_jsonl(sample_path)]
    if ids is not None:
        items = select_items(items, ids)
    config = ApiConfig.from_env()
    config_dir = config_dir_from_env()
    policy = load_answer_policy(config_dir)
    if prompt is not None:
        # 프롬프트를 바꿔 다시 잴 때도 configs/answer.yaml 은 그대로 둔다 — 채택은 측정 뒤 사람이 정한다
        policy = AnswerPolicy(policy.thresholds, swap_prompt(policy.profiles, prompt))
    library = PromptLibrary.load(PROMPT_DIR)
    check_profiles(policy.profiles.values(), library)
    # /ask 와 같이 Kiwi 하나(약 500MB)를 질문 관문과 BM25 가 나눠 쓴다
    kiwi = Kiwi()
    gate = load_query_gate(config_dir, KiwiTokenizer(kiwi))
    dump_date = load_dump_date(config_dir)
    _append_run_meta(
        config,
        policy,
        gate_enabled=gate is not None,
        dump_date=dump_date,
        limit=limit,
        ids=ids,
        sample_path=sample_path,
        answers_path=answers_path,
    )
    print(
        f"문항 {len(items)} · collection={config.search.collection} · 관문 {'켬' if gate else '끔'}",
        flush=True,
    )
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
        summary = asyncio.run(runner.run(items, answers_path, limit=limit, on_record=_progress))
    print(
        f"이번 실행 {summary.recorded} · 건너뜀 {summary.skipped} · 다시 돈 생성 실패 {summary.retried} "
        f"/ 문항 {summary.total} · {time.perf_counter() - start:.0f}s\n저장: {answers_path}"
    )


def select_items(items: Sequence[EvalItem], ids: Sequence[str]) -> list[EvalItem]:
    """표본에서 ids 의 문항만 (표본 순서).

    Raises:
      ValueError: 표본에 없는 id 가 있을 때 — 오타가 조용히 빈 실행이 되지 않게.
    """
    if missing := sorted(set(ids) - {item.id for item in items}):
        raise ValueError(f"ids not in the sample: {missing}")
    wanted = set(ids)
    return [item for item in items if item.id in wanted]


def _progress(index: int, total: int, record: AnswerRecord) -> None:
    print(
        f"  {index}/{total} {record.item.id} {stage_path(record)} {record.wall_ms / 1000:.1f}s",
        flush=True,
    )


def _append_run_meta(
    config: ApiConfig,
    policy: AnswerPolicy,
    *,
    gate_enabled: bool,
    dump_date: str,
    limit: int | None,
    ids: Sequence[str] | None,
    sample_path: Path,
    answers_path: Path,
) -> None:
    """실행마다 설정을 meta 의 runs 에 덧붙인다 — 멈췄다 다시 돌린 사이에 설정이 바뀌었는지 리포트가 본다."""
    meta_path = answers_path.with_suffix(".meta.json")
    runs = json.loads(meta_path.read_text(encoding="utf-8"))["runs"] if meta_path.exists() else []
    runs.append(
        {
            "started_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
            "limit": limit,
            # 문항을 지정해 돌렸으면 그 id (limit 처럼 실행 범위라 settings 밖에 둔다 — 비교 리포트가 settings 만 견준다)
            "ids": list(ids) if ids is not None else None,
            "settings": {
                "collection": config.search.collection,
                "model": config.search.model,
                "candidates": config.search.candidates,
                "rerank_candidates": config.search.rerank_candidates,
                "reranker_model": config.search.reranker_model,
                "top_k": config.search.top_k,
                "thresholds": {"high": policy.thresholds.high, "low": policy.thresholds.low},
                "gate": gate_enabled,
                "dump_date": dump_date,
                "profiles": {
                    stage: {
                        "spec_id": profile.spec.spec_id,
                        "system": list(profile.system),
                        "user": profile.user,
                        "timeout_sec": profile.timeout_sec,
                    }
                    for stage, profile in sorted(policy.profiles.items())
                },
                "inputs": {"sample": str(sample_path)},
            },
        }
    )
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps({"runs": runs}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def retrieval_sensitivity() -> None:
    rows = [row for row in _read_jsonl(PARAPHRASE_PATH) if row["sample"] == PARAPHRASE_SAMPLE]
    config = ApiConfig.from_env()
    print(
        f"collection={config.search.collection} · 문항 {len(rows)} × 원문 · 패러프레이즈",
        flush=True,
    )
    start = time.perf_counter()
    with PgStore.connect(config.pg) as pg:
        service = SearchService.build(config.search, pg, QdrantStore.from_env())
        print(f"검색 서비스 준비 {time.perf_counter() - start:.0f}s", flush=True)
        originals = asyncio.run(_search_all(service, [row["question"] for row in rows]))
        paraphrases = asyncio.run(_search_all(service, [row["paraphrase"] for row in rows]))

    out = [
        sensitivity_row(row, original, paraphrase)
        for row, original, paraphrase in zip(rows, originals, paraphrases, strict=True)
    ]
    _write_jsonl(SENSITIVITY_PATH, out)
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
    meta_path = SENSITIVITY_PATH.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    table = paired_hits(_hit_pairs(out, config.search.top_k))
    print(
        f"hit@{config.search.top_k} 원문 {_percent(table.original_rate)} · "
        f"패러프레이즈 {_percent(table.paraphrase_rate)} "
        f"(원문만 {table.original_only} · 패러프레이즈만 {table.paraphrase_only}, "
        f"McNemar p={table.mcnemar_p:.3g})\n저장: {SENSITIVITY_PATH} · {meta_path}"
    )


async def _search_all(service: SearchService, queries: list[str]) -> list[SearchResult]:
    # 서비스가 질의를 한 번에 하나씩 돌리므로(GPU 모델 · PG 접속 공유) 순서대로 기다린다
    results = []
    for index, query in enumerate(queries, start=1):
        results.append(await service.search(query))
        if index % 50 == 0:
            print(f"  {index}/{len(queries)}", flush=True)
    return results


def sensitivity_row(row: Mapping, original: SearchResult, paraphrase: SearchResult) -> dict:
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


def _search_fields(result: SearchResult, item: KorQuADItem) -> dict:
    return {
        "answer_rank": answer_rank(result.hits, item),
        "top_score": result.hits[0].score if result.hits else None,
        "chunk_ids": [hit.chunk.id for hit in result.hits],
        "took_ms": result.took_ms,
    }


def _hit_pairs(rows: Iterable[Mapping], k: int) -> list[tuple[bool, bool]]:
    return [
        (_hit(row["original"]["answer_rank"], k), _hit(row["paraphrase"]["answer_rank"], k))
        for row in rows
    ]


def _hit(rank: int | None, k: int) -> bool:
    return rank is not None and rank <= k


def pool(split: Split) -> None:
    summary = json.loads((SUBSET / "subset.json").read_text(encoding="utf-8"))
    korquad = Path(summary["korquad_dev"])
    items = load_korquad(korquad, title_aliases=summary["korquad_title_map"])
    paragraphs = korquad_paragraphs(korquad)
    split_items = pool_items(items, paragraphs, split)
    done = {row["id"] for row in _read_jsonl(POOL_PATH)} if POOL_PATH.exists() else set()
    pending = [item for item in split_items if item.id not in done]
    texts = _document_texts({item.title for item in pending})
    config = ApiConfig.from_env()
    print(
        f"{split} 몫 질문 {len(split_items)} · 남은 것 {len(pending)} · "
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
    rows = _read_jsonl(POOL_PATH)
    meta["inputs"] = {
        "korquad_dev": str(korquad),
        "splits": sorted({split_of(row["paragraph"]) for row in rows}),
    }
    meta_path = POOL_PATH.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    counts = Counter(score_band(row["top_score"]) for row in rows)
    print(
        " · ".join(f"{band} {counts[band]}" for band in score_bands() if counts[band])
        + f" · {time.perf_counter() - start:.0f}s\n저장: {POOL_PATH} · {meta_path}"
    )


def pool_items(
    items: Iterable[KorQuADItem], paragraphs: Mapping[str, str], split: Split
) -> list[KorQuADItem]:
    """그 몫 문단의 질문 전부 (id 순). 코퍼스 · 정답 문장으로 거르지 않는다 — 지어냄은 정답이 아니라 문맥과 대조한다."""
    return sorted(
        (item for item in items if split_of(paragraphs[item.id]) == split),
        key=lambda item: item.id,
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
        _append_jsonl(POOL_PATH, row)
        if index % 100 == 0:
            print(f"  {index}/{len(items)}", flush=True)


def pool_row(item: KorQuADItem, paragraph: str, text: str | None, result: SearchResult) -> dict:
    """점수 풀 한 줄 — 표본 줄(f9-sample)과 같은 키에 코퍼스 대조 · 검색 결과를 붙인다.

    Args:
      item: KorQuAD 문항.
      paragraph: 문항의 문단 키.
      text: 정답 문서의 지금 본문. 문서가 코퍼스에 없으면 None.
      result: 그 질문의 검색 결과 (`/ask` 와 같은 검색).
    """
    return {
        "id": item.id,
        "source": "korquad",
        "question": item.question,
        "gold_title": item.title,
        "gold_answer": item.answer,
        "paragraph": paragraph,
        "in_corpus": text is not None,
        "answer_in_document": text is not None and contains_answer(item.title, text, item),
        "top_score": result.hits[0].score if result.hits else None,
        "answer_rank": answer_rank(result.hits, item),
        "took_ms": result.took_ms,
    }


def strata(middle_size: int | None) -> None:
    rows = _read_jsonl(POOL_PATH)
    picked = strata_rows(rows, seed=SEED, middle_size=middle_size)
    _write_jsonl(STRATA_SAMPLE_PATH, picked)
    counts = Counter((split_of(row["paragraph"]), score_band(row["top_score"])) for row in picked)
    paragraphs = {
        band: len({row["paragraph"] for row in picked if score_band(row["top_score"]) == band})
        for band in {band for _, band in counts}
    }
    print(
        f"점수 풀 {len(rows)} → "
        + " · ".join(f"{split} {band} {count}" for (split, band), count in sorted(counts.items()))
        + " (문단 "
        + " · ".join(f"{band} {count}" for band, count in sorted(paragraphs.items()))
        + f")\n저장: {STRATA_SAMPLE_PATH} ({len(picked)}줄)"
    )


def strata_rows(
    rows: Sequence[Mapping], *, seed: int, middle_size: int | None = None
) -> list[dict]:
    """0.83~0.94 구간을 STRATA_SPLITS 순서로 채우고 몫마다 같은 수를 ≥ 0.94 구간에서 고른 점수 풀 줄 (id 순 — 두 구간이 섞여 돈다).

    몫 · 구간마다 seed 로 섞은 순서의 앞에서부터 고른다 — middle_size 를 늘려도 앞서 고른 문항은 그대로 남는다.
    몫마다 수를 맞추는 것은 두 구간의 몫 구성을 같게 하려고.

    Args:
      rows: 점수 풀 줄.
      seed: 섞는 seed.
      middle_size: 0.83~0.94 구간 문항 수 상한. None 이면 STRATA_SPLITS 몫의 그 구간 전부.
    """
    lower, upper = SCORE_BAND_EDGES
    picked: list[dict] = []
    room = middle_size
    for split in STRATA_SPLITS:
        middle = _shuffled_band(rows, split, score_band(lower), seed=seed)
        taken = middle if room is None else middle[:room]
        picked += taken + _shuffled_band(rows, split, score_band(upper), seed=seed)[: len(taken)]
        if room is not None:
            room -= len(taken)
    return sorted(picked, key=lambda row: row["id"])


def _shuffled_band(rows: Sequence[Mapping], split: str, band: str, *, seed: int) -> list[dict]:
    chosen = sorted(
        (
            dict(row)
            for row in rows
            if (split_of(row["paragraph"]), score_band(row["top_score"])) == (split, band)
        ),
        key=lambda row: row["id"],
    )
    # 몫 · 구간마다 따로 섞는다 — 한 몫의 문항 수가 달라져도 다른 몫의 순서가 바뀌지 않게
    random.Random(f"{seed}:{split}:{band}").shuffle(chosen)
    return chosen


def judge_input(tag: str | None) -> None:
    records = read_records(tagged(STRATA_ANSWERS_PATH, tag))
    rows, key = judge_rows(records, seed=SEED)
    input_path = tagged(JUDGE_INPUT_PATH, tag)
    key_path = tagged(JUDGE_KEY_PATH, tag)
    _write_jsonl(input_path, rows)
    _write_jsonl(key_path, key)
    print(f"기록 {len(records)} → grounded 로 끝난 답 {len(rows)}\n저장: {input_path} · {key_path}")


def judge_rows(records: Sequence[AnswerRecord], *, seed: int) -> tuple[list[dict], list[dict]]:
    """grounded 로 끝난 답의 판정 입력과 열쇠(가린 번호 → 문항 id). 두 구간을 seed 로 섞는다.

    판정 입력에는 가린 번호 · 질문 · 모델이 받은 문서(출처 번호 순) · 답만 싣는다 — 1위 점수 · 구간 · 정답 · 문항 id 를
    알면 판정이 기운다.
    """
    judged = [record for record in records if final_stage(record) == "grounded"]
    rows = []
    key = []
    for index, record in enumerate(random.Random(seed).sample(judged, len(judged)), start=1):
        judge_id = f"J{index:03d}"
        documents = [
            {"number": number, "text": hit.text}
            for number, hit in enumerate(record.hits[: record.context_count], start=1)
        ]
        rows.append(
            {
                "judge_id": judge_id,
                "question": record.item.question,
                "documents": documents,
                "answer": record.answer,
            }
        )
        key.append({"judge_id": judge_id, "id": record.item.id})
    return rows, key


def strata_report() -> None:
    records = read_records(STRATA_ANSWERS_PATH)
    runs = json.loads(STRATA_ANSWERS_PATH.with_suffix(".meta.json").read_text(encoding="utf-8"))
    pool_rows = _read_jsonl(POOL_PATH)
    verdicts = _read_verdicts(None)
    today = datetime.now(tz=UTC).date()
    text = render_strata_report(records, runs["runs"], pool_rows, verdicts, today=today)
    REPORT_DIR.mkdir(exist_ok=True)
    path = REPORT_DIR / f"{today:%Y-%m-%d}-high-threshold.md"
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"기록 {len(records)} · 판정 {len(verdicts)} · 리포트: {path}")


def render_strata_report(
    records: Sequence[AnswerRecord],
    runs: Sequence[Mapping],
    pool_rows: Sequence[Mapping],
    verdicts: Mapping[str, GroundingVerdict],
    *,
    today: date,
) -> str:
    """high 확정 리포트 markdown — 집계만 (질문 · 답 원문 · 청크 본문 · 판정 근거 구절을 싣지 않는다).

    Args:
      records: 층화 표본을 답변 경로로 돌린 기록.
      runs: 그 기록의 meta runs (실행마다 설정).
      pool_rows: 점수 풀 줄 전부.
      verdicts: 문항 id → 판정. grounded 로 끝난 답마다 하나.
      today: 리포트 날짜.
    """
    missing = [
        r.item.id for r in records if final_stage(r) == "grounded" and r.item.id not in verdicts
    ]
    if missing:
        raise ValueError(f"{len(missing)} grounded answers have no verdict, e.g. {missing[:3]}")
    pool_scores = {row["id"]: row["top_score"] for row in pool_rows}
    paragraphs = {row["id"]: row["paragraph"] for row in pool_rows}
    return (
        f"# high 확정 — 점수 구간 층화 표본의 grounded 지어냄 비율 ({today:%Y-%m-%d})\n\n"
        + _strata_overview(records, runs, pool_rows)
        + "\n## 점수 풀 — 리랭커 1위 점수 구간\n\n"
        + _pool_section(pool_rows)
        + "\n## 층화 표본 — 단계 경로\n\n"
        + _strata_path_section(records, pool_scores)
        + "\n## 지어냄 판정 — grounded 로 끝난 답\n\n"
        + _verdict_section(records, verdicts)
        + "\n## 두 구간 비교 (0.83–0.94 − ≥ 0.94)\n\n"
        + _difference_section(records, verdicts, paragraphs)
        + "\n## 판정 방법 · 검수\n\n"
        + _judge_method_section(verdicts)
        + "\n## 결론\n\n"
        + (HIGH_DECISION or "사용자와 정한다 — 아직\n")
    )


def _strata_overview(
    records: Sequence[AnswerRecord], runs: Sequence[Mapping], pool_rows: Sequence[Mapping]
) -> str:
    settings = runs[-1]["settings"]
    distinct = {json.dumps(run["settings"], sort_keys=True) for run in runs}
    splits = Counter(split_of(row["paragraph"]) for row in pool_rows)
    pool_by_id = {row["id"]: row for row in pool_rows}
    # 점수 풀의 점수로 고른 구간 — 다시 검색한 점수로 나눈 아래 표들과 다를 수 있다
    composition = Counter(
        (
            split_of(pool_by_id[record.item.id]["paragraph"]),
            score_band(pool_by_id[record.item.id]["top_score"]),
        )
        for record in records
    )
    thresholds = settings["thresholds"]
    changed = (
        f"- **실행 {len(runs)}번 사이에 설정이 바뀌었다** ({len(distinct)}가지) — 아래 수치는 섞여 있다\n"
        if len(distinct) > 1
        else ""
    )
    lower, upper = SCORE_BAND_EDGES
    return (
        "- 질문 (ADR-024): 1위 점수 0.83~0.94 구간에서 grounded 가 지어내는 비율이 ≥ 0.94 구간보다 높은가 — "
        "높으면 `high` 를 0.94 로, 다르지 않으면 유지\n"
        "- 점수 풀: KorQuAD dev "
        + " · ".join(f"{split} 몫 {count}" for split, count in sorted(splits.items()))
        + " — 문단의 질문 전부, 코퍼스 · 정답 문장으로 거르지 않음 (지어냄은 정답이 아니라 문맥과 대조한다). "
        "`/ask` 와 같은 검색에만 돌려 1위 점수를 쟀다 (LLM 없음)\n"
        f"- 층화 표본 {len(records)}문항: {lower:g}~{upper:g} 구간을 eval 몫부터 담고 모자란 만큼 bench 몫에서, 몫마다 같은 "
        f"수를 ≥ {upper:g} 구간에서 seed {SEED} 로 — "
        + " · ".join(
            f"{split} {band} {count}" for (split, band), count in sorted(composition.items())
        )
        + ". id 순으로 돌려 두 구간이 섞여 돈다. 아래 표의 구간은 답변 경로가 다시 검색한 1위 점수(기록의 top_score)로 "
        "나눈다\n"
        f"- 답변 경로: F9 와 같음 — 관문 {'켬' if settings['gate'] else '끔'} → `{settings['collection']}` "
        f"Hybrid(후보 {settings['candidates']}) + 리랭커 `{settings['reranker_model']}`(후보 "
        f"{settings['rerank_candidates']}) 상위 {settings['top_k']} → 답변 정책(high {thresholds['high']:.6f} · "
        f"low {thresholds['low']:.6f}) → grounded@v2 첫 줄 표지, 못 찾음이면 partial "
        f"(`{settings['profiles']['grounded']['spec_id']}`)\n"
        + changed
        + "- 질문 · 답 · 문서 원문과 판정 근거 구절은 로컬(data/eval/f9-strata-*)에만 둔다 (KorQuAD CC BY-ND)\n"
    )


def _pool_section(pool_rows: Sequence[Mapping]) -> str:
    groups: dict[str, list[Mapping]] = defaultdict(list)
    for row in pool_rows:
        groups[score_band(row["top_score"])].append(row)
    rows = [
        {
            "1위 점수": band,
            "질문": len(groups[band]),
            "문단": len({row["paragraph"] for row in groups[band]}),
            "정답 문서가 코퍼스에": _fmt(share(row["in_corpus"] for row in groups[band])),
            "정답 문장이 남음": _fmt(share(row["answer_in_document"] for row in groups[band])),
            "상위에 정답 근거": _fmt(share(row["answer_rank"] is not None for row in groups[band])),
        }
        for band in score_bands()
        if groups.get(band)
    ]
    return render_table(
        rows,
        [
            "1위 점수",
            "질문",
            "문단",
            "정답 문서가 코퍼스에",
            "정답 문장이 남음",
            "상위에 정답 근거",
        ],
    ) + (
        "\n- 상위에 정답 근거 = 검색 상위 5에 정답 문서이면서 정답 문자열을 담은 청크 (검색 평가와 같은 기준). "
        "중간 구간은 검색이 정답을 놓친 문항이 몰려 있어 지어냄 위험이 큰 쪽이 자연스럽게 들어온다\n"
    )


def _strata_path_section(
    records: Sequence[AnswerRecord], pool_scores: Mapping[str, float | None]
) -> str:
    groups: dict[str, list[AnswerRecord]] = defaultdict(list)
    for record in records:
        groups[score_band(record.top_score)].append(record)
    rows = [
        {
            "1위 점수": band,
            "문항": len(groups[band]),
            "grounded 로 끝남": sum(final_stage(r) == "grounded" for r in groups[band]),
            "grounded → partial": sum(
                r.stage_path == ("grounded", "partial") for r in groups[band]
            ),
            "그 밖": " · ".join(
                f"{path} {count}"
                for path, count in path_counts(groups[band]).items()
                if path not in {"grounded", "grounded → partial"}
            ),
        }
        for band in score_bands()
        if groups.get(band)
    ]
    moved = sum(
        score_band(record.top_score) != score_band(pool_scores.get(record.item.id))
        for record in records
    )
    return render_table(
        rows, ["1위 점수", "문항", "grounded 로 끝남", "grounded → partial", "그 밖"]
    ) + (
        f"\n- 점수 풀과 답변 경로의 검색은 따로 돌았다 — 다시 검색한 1위 점수가 다른 구간에 든 문항 {moved}개\n"
        "- grounded → partial 은 첫 줄 표지 '못 찾음'으로 캐스케이드 — 지어냄 분모에서 빠진다 (답을 내지 않았다)\n"
    )


def _verdict_section(
    records: Sequence[AnswerRecord], verdicts: Mapping[str, GroundingVerdict]
) -> str:
    groups = _judged_by_band(records, verdicts)
    rows = []
    for band in score_bands():
        judged = groups.get(band)
        if not judged:
            continue
        labels = Counter(verdict.verdict for _, verdict in judged)
        counted = [verdict for _, verdict in judged if verdict.verdict not in EXCLUDED]
        rows.append(
            {
                "1위 점수": band,
                "grounded 답": len(judged),
                **{VERDICT_LABELS[label]: labels[label] for label in VERDICTS},
                "지어냄 (95% 구간)": _rate_with_interval(
                    sum(v.verdict in FABRICATED for v in counted), len(counted)
                ),
                "곁가지 포함 (95% 구간)": _rate_with_interval(
                    sum(v.verdict in UNSUPPORTED for v in counted), len(counted)
                ),
                "정답 포함": _fmt(share(contains_gold(record) for record, _ in judged)),
                "출처 번호 어긋남": sum(verdict.wrong_citation for _, verdict in judged),
                "질문을 비껴감": sum(verdict.off_question for _, verdict in judged),
            }
        )
    columns = [
        "1위 점수",
        "grounded 답",
        *(VERDICT_LABELS[label] for label in VERDICTS),
        "지어냄 (95% 구간)",
        "곁가지 포함 (95% 구간)",
        "정답 포함",
        "출처 번호 어긋남",
        "질문을 비껴감",
    ]
    crossing = Counter(
        (bool(contains_gold(record)), verdict.verdict in FABRICATED)
        for judged in groups.values()
        for record, verdict in judged
        if verdict.verdict not in EXCLUDED
    )
    return render_table(rows, columns) + (
        f"\n- 지어냄 = ({VERDICT_LABELS['misread']} + {VERDICT_LABELS['not_in_context']}) ÷ "
        f"(grounded 답 − {VERDICT_LABELS['body_refusal']}). 곁가지 포함은 분자에 "
        f"{VERDICT_LABELS['extra_unsupported']}까지 더한 것. 95% 구간은 Wilson\n"
        f"- {VERDICT_LABELS['body_refusal']} = 표지는 '찾음'인데 본문이 문서에 답이 없다고 했다 — 답을 내지 않아 분모에서 뺀다\n"
        "- 정답 포함(KorQuAD 정답 문자열)과 판정의 교차 — 정답 포함 · 지어냄 "
        f"{crossing[(True, True)]}, 정답 포함 · 지어냄 아님 {crossing[(True, False)]}, "
        f"정답 미포함 · 지어냄 {crossing[(False, True)]}, 정답 미포함 · 지어냄 아님 {crossing[(False, False)]}. "
        "정답 미포함이 곧 지어냄은 아니다 (다른 말로 · 일부만 · 정답 문장이 고쳐진 문서)\n"
        "\n중간 구간 안쪽 (참고 — 세부 구간마다 문항이 적다):\n\n"
        + _fine_band_table(records, verdicts)
    )


def _fine_band_table(
    records: Sequence[AnswerRecord], verdicts: Mapping[str, GroundingVerdict]
) -> str:
    groups: dict[str, list[AnswerRecord]] = defaultdict(list)
    for record in records:
        groups[score_band(record.top_score, FINE_BAND_EDGES)].append(record)
    rows = []
    for band in score_bands(FINE_BAND_EDGES):
        group = groups.get(band)
        if not group:
            continue
        # 판정은 grounded 로 끝난 답에만 있다
        fabricated = []
        for record in group:
            verdict = verdicts.get(record.item.id)
            if verdict is not None and verdict.verdict not in EXCLUDED:
                fabricated.append(verdict.verdict in FABRICATED)
        rows.append(
            {
                "1위 점수 (세부)": band,
                "문항": len(group),
                "grounded → partial": sum(r.stage_path == ("grounded", "partial") for r in group),
                "지어냄": _fmt(share(fabricated)),
            }
        )
    return render_table(rows, ["1위 점수 (세부)", "문항", "grounded → partial", "지어냄"])


def _difference_section(
    records: Sequence[AnswerRecord],
    verdicts: Mapping[str, GroundingVerdict],
    paragraphs: Mapping[str, str],
) -> str:
    lower, upper = SCORE_BAND_EDGES
    groups = _judged_by_band(records, verdicts)
    middle = [pair for pair in groups.get(score_band(lower), []) if pair[1].verdict not in EXCLUDED]
    high = [pair for pair in groups.get(score_band(upper), []) if pair[1].verdict not in EXCLUDED]
    if not middle or not high:
        return "판정한 답이 한쪽 구간에 없다\n"
    rows = [
        _difference_row(name, labels, middle, high, paragraphs)
        for name, labels in (("지어냄", FABRICATED), ("곁가지 포함", UNSUPPORTED))
    ]
    return render_table(
        rows,
        ["지표", "0.83–0.94", "≥ 0.94", "차이 (%p)", "Newcombe 95%", "문단 묶음 bootstrap 95%"],
    ) + (
        f"\n- 분모: 0.83–0.94 {len(middle)} · ≥ 0.94 {len(high)} (본문 거절 뺌). 두 구간은 서로 다른 문항이라 짝 검정"
        "(McNemar)이 아니라 독립 두 비율의 차이 — Newcombe (1998) 방법 10 (각 비율의 Wilson 구간을 합침)\n"
        f"- 문단 묶음 bootstrap: 같은 문단의 질문은 문맥을 나눠 써서 독립이 아니다 — 구간마다 문단을 복원 추출 "
        f"(2000회 · seed {SEED})\n"
        "- 결정 규칙 (답을 보기 전에 고정): 지어냄 차이의 Newcombe 구간이 0 위면 중간 구간이 더 지어낸다 → `high` 를 "
        f"{upper:g} 로 올리자고 제안, 0 을 포함하면 유지 제안. 결론은 사용자와 정한다. 중간 구간 분모가 100 에 못 미치면 "
        "'못 잼' (ADR-024)\n"
    )


def _difference_row(
    name: str,
    labels: frozenset[str],
    middle: Sequence[tuple[AnswerRecord, GroundingVerdict]],
    high: Sequence[tuple[AnswerRecord, GroundingVerdict]],
    paragraphs: Mapping[str, str],
) -> dict:
    middle_count = sum(verdict.verdict in labels for _, verdict in middle)
    high_count = sum(verdict.verdict in labels for _, verdict in high)
    newcombe = newcombe_difference(middle_count, len(middle), high_count, len(high))
    cluster = cluster_bootstrap_difference(
        _paragraph_groups(middle, labels, paragraphs),
        _paragraph_groups(high, labels, paragraphs),
        seed=SEED,
    )
    return {
        "지표": name,
        "0.83–0.94": _fmt(Share(middle_count, len(middle))),
        "≥ 0.94": _fmt(Share(high_count, len(high))),
        "차이 (%p)": f"{newcombe.difference * 100:+.1f}",
        "Newcombe 95%": _interval(newcombe.low, newcombe.high),
        "문단 묶음 bootstrap 95%": _interval(cluster.low, cluster.high),
    }


def _paragraph_groups(
    judged: Sequence[tuple[AnswerRecord, GroundingVerdict]],
    labels: frozenset[str],
    paragraphs: Mapping[str, str],
) -> list[list[bool]]:
    groups: dict[str, list[bool]] = defaultdict(list)
    for record, verdict in judged:
        groups[paragraphs[record.item.id]].append(verdict.verdict in labels)
    return [groups[paragraph] for paragraph in sorted(groups)]


def _judge_method_section(verdicts: Mapping[str, GroundingVerdict]) -> str:
    reviewed = [verdict for verdict in verdicts.values() if verdict.reviewed]
    changes = Counter(
        (VERDICT_LABELS[v.first_verdict], VERDICT_LABELS[v.verdict])
        for v in reviewed
        if v.first_verdict != v.verdict
    )
    change_text = (
        " · ".join(
            f"{first} → {final} {count}" for (first, final), count in sorted(changes.items())
        )
        or "없음"
    )
    return (
        "".join(f"- {line}\n" for line in JUDGE_METHOD)
        + f"- 검수 결과: 판정 {len(verdicts)} 중 메인 세션이 다시 본 것 {len(reviewed)}, 바뀐 판정 "
        f"{sum(changes.values())} ({change_text})\n"
    )


def _judged_by_band(
    records: Sequence[AnswerRecord], verdicts: Mapping[str, GroundingVerdict]
) -> dict[str, list[tuple[AnswerRecord, GroundingVerdict]]]:
    groups: dict[str, list[tuple[AnswerRecord, GroundingVerdict]]] = defaultdict(list)
    for record in records:
        if final_stage(record) == "grounded":
            groups[score_band(record.top_score)].append((record, verdicts[record.item.id]))
    return groups


def _rate_with_interval(count: int, total: int) -> str:
    if not total:
        return "0/0"
    low, high = wilson_interval(count, total)
    return f"{count}/{total} ({count / total:.1%}, {low:.1%}~{high:.1%})"


def _interval(low: float, high: float) -> str:
    return f"{low * 100:+.1f} ~ {high * 100:+.1f}"


def _read_verdicts(tag: str | None) -> dict[str, GroundingVerdict]:
    """문항 id → 판정 — 판정 열쇠로 가린 번호를 푼다."""
    key = {row["judge_id"]: row["id"] for row in _read_jsonl(tagged(JUDGE_KEY_PATH, tag))}
    rows = _read_jsonl(tagged(VERDICTS_PATH, tag))
    return {key[verdict.judge_id]: verdict for verdict in map(GroundingVerdict.from_json, rows)}


def _read_runs(answers_path: Path) -> list[Mapping]:
    meta_path = answers_path.with_suffix(".meta.json")
    return json.loads(meta_path.read_text(encoding="utf-8"))["runs"]


def cascade_input(tag: str) -> None:
    base = Measurement.load(None)
    other = Measurement.load(tag)
    rows, key = cascade_rows(base, other, seed=SEED)
    input_path = tagged(CASCADE_INPUT_PATH, tag)
    key_path = tagged(CASCADE_KEY_PATH, tag)
    _write_jsonl(input_path, rows)
    _write_jsonl(key_path, key)
    counts = Counter((row["record"], row["side"]) for row in key)
    print(
        "판정 입력 "
        + " · ".join(f"{record} {side} {count}" for (record, side), count in sorted(counts.items()))
        + f"\n저장: {input_path} · {key_path}"
    )


def moved_to_partial(
    base: Sequence[AnswerRecord], other: Sequence[AnswerRecord]
) -> list[tuple[AnswerRecord, AnswerRecord]]:
    """기준에서 grounded 로 끝났는데 다시 잰 쪽에서 grounded → partial 로 넘어간 KorQuAD 문항의 (기준, 다시 잰) 기록 (id 순)."""
    after = {record.item.id: record for record in other}
    return [
        (record, after[record.item.id])
        for record in sorted(base, key=lambda r: r.item.id)
        if record.item.source == "korquad"
        and _ends_grounded(record)
        and after[record.item.id].stage_path == ("grounded", "partial")
    ]


def cascade_rows(
    base: Measurement, other: Measurement, *, seed: int
) -> tuple[list[dict], list[dict]]:
    """넘어간 문항의 판정 입력과 열쇠(가린 번호 → 기록 · 쪽 · 문항 id).

    다시 잰 쪽 partial 답은 문서 칸만 싣는다 — 일반 지식 칸은 판정하지 않는다. F9 문항은 기준 grounded 답도 함께 싣는다
    (층화 문항의 기준 답은 high 확정 판정을 쓴다). 섞어서 가린 번호 C001… 을 붙인다 — 판정자는 점수 · 구간 · 정답 ·
    문항 id 와, 어느 기록 · 어느 프롬프트의 답인지를 모른다.
    """
    entries: list[tuple[str, str, AnswerRecord, str]] = []
    for name, first, second in (("f9", base.f9, other.f9), ("strata", base.strata, other.strata)):
        for before, after in moved_to_partial(first, second):
            entries.append((name, "tag", after, sections(after.answer).get(DOC_HEADING, "")))
            if name == "f9":
                entries.append((name, "base", before, before.answer))
    rows = []
    key = []
    for index, (name, side, record, answer) in enumerate(
        random.Random(seed).sample(entries, len(entries)), start=1
    ):
        judge_id = f"C{index:03d}"
        documents = [
            {"number": number, "text": hit.text}
            for number, hit in enumerate(record.hits[: record.context_count], start=1)
        ]
        rows.append(
            {
                "judge_id": judge_id,
                "question": record.item.question,
                "documents": documents,
                "answer": answer,
            }
        )
        key.append({"judge_id": judge_id, "id": record.item.id, "record": name, "side": side})
    return rows, key


def _read_cascade_verdicts(tag: str) -> dict[tuple[str, str, str], GroundingVerdict] | None:
    """(기록, 쪽, 문항 id) → 넘어간 답의 판정. 판정 파일이 없으면 None."""
    path = tagged(CASCADE_VERDICTS_PATH, tag)
    if not path.exists():
        return None
    key = {
        row["judge_id"]: (row["record"], row["side"], row["id"])
        for row in _read_jsonl(tagged(CASCADE_KEY_PATH, tag))
    }
    verdicts = map(GroundingVerdict.from_json, _read_jsonl(path))
    return {key[verdict.judge_id]: verdict for verdict in verdicts}


def compare(tag: str, report_date: date | None) -> None:
    base = Measurement.load(None)
    other = Measurement.load(tag)
    cascade = _read_cascade_verdicts(tag)
    today = report_date or datetime.now(tz=UTC).date()
    text = render_compare_report(base, other, tag=tag, today=today, cascade=cascade)
    REPORT_DIR.mkdir(exist_ok=True)
    path = REPORT_DIR / f"{today:%Y-%m-%d}-{tag}.md"
    path.write_text(text, encoding="utf-8", newline="\n")
    print(
        f"F9 {len(other.f9)} · 층화 {len(other.strata)} · 판정 {len(other.verdicts)} · "
        f"넘어간 답 판정 {len(cascade) if cascade else '없음'} · 리포트: {path}"
    )


def render_compare_report(
    base: Measurement,
    other: Measurement,
    *,
    tag: str,
    today: date,
    cascade: Mapping[tuple[str, str, str], GroundingVerdict] | None = None,
) -> str:
    """기준 기록과 꼬리표 tag 기록을 같은 문항으로 비교한 리포트 markdown — 집계와 문항 id 만.

    Args:
      base: 기준 기록 (꼬리표 없음).
      other: 다시 잰 기록.
      tag: 다시 잰 기록의 꼬리표.
      today: 리포트 날짜.
      cascade: (기록, 쪽, 문항 id) → grounded 에서 partial 로 넘어간 답의 판정 (cascade-input). None 이면 그 절을 뺀다.

    Raises:
      ValueError: 두 기록의 문항이 다르거나 (한쪽 실행이 덜 끝남), grounded 로 끝난 층화 표본 답에 판정이 없을 때.
    """
    for name, first, second in (("f9", base.f9, other.f9), ("strata", base.strata, other.strata)):
        if {r.item.id for r in first} != {r.item.id for r in second}:
            raise ValueError(
                f"{name} records cover different items: base {len(first)}, {tag} {len(second)}"
            )
    for measurement in (base, other):
        missing = [
            r.item.id
            for r in measurement.strata
            if final_stage(r) == "grounded" and r.item.id not in measurement.verdicts
        ]
        if missing:
            raise ValueError(f"{len(missing)} grounded answers have no verdict, e.g. {missing[:3]}")
    return (
        f"# grounded 프롬프트 비교 — 기준 기록 vs `{tag}` ({today:%Y-%m-%d})\n\n"
        + _compare_overview(base, other, tag)
        + "\n## 층화 표본 — grounded 지어냄 (high 확정 표본)\n\n"
        + _compare_strata_section(base, other, tag)
        + "\n## F9 KorQuAD — 근거 답변\n\n"
        + _compare_korquad_section(base, other, tag)
        + "\n## F9 위키 밖 질문\n\n"
        + _compare_offwiki_section(base, other, tag)
        + (
            "\n## grounded 에서 partial 로 넘어간 답 — 문서 칸 판정\n\n"
            + _cascade_section(base, other, cascade, tag)
            if cascade is not None
            else ""
        )
        + f"\n## 판정 방법 · 검수 (`{tag}`)\n\n"
        + _judge_method_section(other.verdicts)
        + "- 기준 기록의 판정은 strata-report 리포트(high 확정)에 쓴 것 그대로 — 같은 지시문 · 같은 판정 입력 꼴\n"
        + "\n## 확인한 것\n\n"
        + ("".join(f"- {line}\n" for line in COMPARE_NOTES.get(tag, ())) or "아직\n")
        + "\n## 결론\n\n"
        + COMPARE_DECISIONS.get(tag, "사용자와 정한다 — 아직\n")
    )


def _compare_overview(base: Measurement, other: Measurement, tag: str) -> str:
    changed: dict[str, tuple[str, str]] = {}
    for first, second in ((base.f9_runs, other.f9_runs), (base.strata_runs, other.strata_runs)):
        before = _flat_settings(first[-1]["settings"])
        after = _flat_settings(second[-1]["settings"])
        for name in sorted(before.keys() | after.keys()):
            if before.get(name) != after.get(name):
                changed[name] = (before.get(name, "없음"), after.get(name, "없음"))
    mixed = [
        name
        for name, runs in (
            ("기준 F9", base.f9_runs),
            ("기준 층화", base.strata_runs),
            (f"{tag} F9", other.f9_runs),
            (f"{tag} 층화", other.strata_runs),
        )
        if len({json.dumps(run["settings"], sort_keys=True) for run in runs}) > 1
    ]
    return (
        f"- 같은 문항(F9 표본 {len(other.f9)} · 층화 표본 {len(other.strata)})을 같은 검색으로 다시 돌렸다. 기준 = 꼬리표 "
        f"없는 기록(data/eval/f9-answers-v1.jsonl · f9-strata-answers-v1.jsonl), 비교 = 파일 이름 끝 -{tag}\n"
        + "".join(
            f"- 바뀐 설정 `{name}`: {before} → {after}\n"
            for name, (before, after) in changed.items()
        )
        + ("- 바뀐 설정 없음\n" if not changed else "")
        + f"- grounded 로 끝난 기록의 프롬프트 버전: 기준 {_versions(base)} → {tag} {_versions(other)}\n"
        + (
            f"- **실행 사이에 설정이 바뀐 기록: {' · '.join(mixed)}** — 그 기록의 수치는 섞여 있다\n"
            if mixed
            else ""
        )
        + "- 같은 프롬프트로 다시 돌려도 답이 일부 갈린다 (Ollama GPU, temperature 0.1) — 두 기록의 차이에는 그 몫도 있다\n"
        "- 질문 · 답 · 문서 원문은 로컬 기록에만 둔다 (KorQuAD CC BY-ND) — 리포트에는 집계와 위키 밖 문항 id 만\n"
    )


def _flat_settings(settings: Mapping, prefix: str = "") -> dict[str, str]:
    flat: dict[str, str] = {}
    for name, value in settings.items():
        if isinstance(value, Mapping):
            flat |= _flat_settings(value, f"{prefix}{name}.")
        else:
            flat[f"{prefix}{name}"] = json.dumps(value, ensure_ascii=False)
    return flat


def _versions(measurement: Measurement) -> str:
    versions = Counter(
        r.prompt_version
        for r in (*measurement.f9, *measurement.strata)
        if final_stage(r) == "grounded"
    )
    return " · ".join(f"`{version}` {count}" for version, count in sorted(versions.items()))


def _compare_strata_section(base: Measurement, other: Measurement, tag: str) -> str:
    rows = []
    for band in score_bands():
        for name, measurement in (("기준", base), (tag, other)):
            group = [r for r in measurement.strata if score_band(r.top_score) == band]
            if group:
                rows.append(_compare_band_row(band, name, group, measurement.verdicts))
    paired = [
        _paired_row(
            "grounded 로 끝남", *(_flags(m.strata, _ends_grounded) for m in (base, other)), tag
        ),
        _paired_row("지어냄 답을 냄", *(_verdict_flags(m, FABRICATED) for m in (base, other)), tag),
        _paired_row(
            "곁가지 포함 근거 없는 답을 냄",
            *(_verdict_flags(m, UNSUPPORTED) for m in (base, other)),
            tag,
        ),
        _paired_row(
            "근거 있는 답을 냄",
            *(_verdict_flags(m, frozenset({"supported"})) for m in (base, other)),
            tag,
        ),
    ]
    return (
        render_table(
            rows,
            [
                "1위 점수",
                "기록",
                "문항",
                "grounded 로 끝남",
                "grounded → partial",
                "본문 거절",
                "지어냄 (95% 구간)",
                "곁가지 포함 (95% 구간)",
                "정답 포함",
                "질문을 비껴감",
            ],
        )
        + "\n- 지어냄 = 지어냄 판정 ÷ (grounded 로 끝난 답 − 본문 거절) — high 확정 리포트와 같은 분모. 95% 구간은 Wilson. "
        "구간은 그 기록이 다시 검색한 1위 점수로 나눈다\n"
        "- 두 기록은 grounded 로 끝난 답이 달라 위 비율의 분모가 다르다 — 아래는 층화 표본 전부를 분모로 같은 문항끼리 "
        "짝지은 것\n\n"
        + render_table(paired, _paired_columns(tag))
        + "\n- '…답을 냄' = grounded 로 끝나고 그 판정을 받음 (grounded → partial 로 넘긴 문항은 답을 내지 않은 쪽). "
        "McNemar p 는 엇갈린 짝(한쪽만 해당)만 본 정확 검정\n"
    )


def _compare_band_row(
    band: str, name: str, group: Sequence[AnswerRecord], verdicts: Mapping[str, GroundingVerdict]
) -> dict:
    judged = [(r, verdicts[r.item.id]) for r in group if final_stage(r) == "grounded"]
    counted = [verdict for _, verdict in judged if verdict.verdict not in EXCLUDED]
    return {
        "1위 점수": band,
        "기록": name,
        "문항": len(group),
        "grounded 로 끝남": len(judged),
        "grounded → partial": sum(r.stage_path == ("grounded", "partial") for r in group),
        "본문 거절": len(judged) - len(counted),
        "지어냄 (95% 구간)": _rate_with_interval(
            sum(v.verdict in FABRICATED for v in counted), len(counted)
        ),
        "곁가지 포함 (95% 구간)": _rate_with_interval(
            sum(v.verdict in UNSUPPORTED for v in counted), len(counted)
        ),
        "정답 포함": _fmt(share(contains_gold(r) for r, _ in judged)),
        "질문을 비껴감": sum(verdict.off_question for _, verdict in judged),
    }


def _compare_korquad_section(base: Measurement, other: Measurement, tag: str) -> str:
    rows = [
        _side_row("grounded 로 끝남", base, other, tag, _korquad_grounded_share),
        _side_row("정답 포함", base, other, tag, _korquad_gold_share),
        _side_row("정답 근거 청크를 출처 번호로 짚음", base, other, tag, _korquad_cites_share),
        _side_row("grounded → partial", base, other, tag, _korquad_cascade_share),
    ]
    paired = [
        _paired_row(
            "grounded 로 끝남", *(_flags(_korquad(m), _ends_grounded) for m in (base, other)), tag
        ),
        _paired_row(
            "grounded 로 끝나고 정답 포함",
            *(_flags(_korquad(m), _grounded_with_gold) for m in (base, other)),
            tag,
        ),
        _paired_row(
            "어느 단계든 정답 포함",
            *(_flags(_korquad(m), lambda r: bool(contains_gold(r))) for m in (base, other)),
            tag,
        ),
    ]
    return (
        render_table(rows, ["지표", "기준", tag, "분모"])
        + "\n"
        + render_table(paired, _paired_columns(tag))
        + "\n- 정답 포함 = 답에 KorQuAD 정답 문자열 (F9 리포트와 같은 규칙). 정답 미포함이 곧 지어냄은 아니다\n"
    )


def _compare_offwiki_section(base: Measurement, other: Measurement, tag: str) -> str:
    before = {r.item.id: r for r in base.f9 if r.item.source == "offwiki"}
    after = {r.item.id: r for r in other.f9 if r.item.source == "offwiki"}
    rows = []
    for expected in sorted({r.item.expected for r in before.values()}, key=str):
        ids = sorted(i for i, r in before.items() if r.item.expected == expected)
        rows.append(
            {
                "기대": expected,
                "문항": len(ids),
                "기준 grounded 로 끝남": sum(_ends_grounded(before[i]) for i in ids),
                f"{tag} grounded 로 끝남": sum(_ends_grounded(after[i]) for i in ids),
                "기준 기대대로": _fmt(share(matches_expected(before[i]) for i in ids)),
                f"{tag} 기대대로": _fmt(share(matches_expected(after[i]) for i in ids)),
            }
        )
    either = sorted(i for i in before if _ends_grounded(before[i]) or _ends_grounded(after[i]))
    moved = [
        {
            "id": i,
            "기대 / 묶음": f"{before[i].item.expected}/{before[i].item.category}",
            "기준 경로": stage_path(before[i]),
            f"{tag} 경로": stage_path(after[i]),
        }
        for i in either
    ]
    return (
        render_table(
            rows,
            [
                "기대",
                "문항",
                "기준 grounded 로 끝남",
                f"{tag} grounded 로 끝남",
                "기준 기대대로",
                f"{tag} 기대대로",
            ],
        )
        + "\n- partial 기대가 grounded 로 끝난 것은 '찾음' 표지로 문서 근거라고 답한 것 — 지어냄 후보다. 다만 틀린 전제를 "
        "문서로 바로잡는 답(질문의 순번 · 노선이 없다고 밝힘)은 grounded 가 맞는 길이다\n\n"
        "어느 한쪽이라도 grounded 로 끝난 위키 밖 질문:\n\n"
        + (
            render_table(moved, ["id", "기대 / 묶음", "기준 경로", f"{tag} 경로"])
            if moved
            else "없음\n"
        )
    )


def _cascade_section(
    base: Measurement,
    other: Measurement,
    cascade: Mapping[tuple[str, str, str], GroundingVerdict],
    tag: str,
) -> str:
    pair_rows = []
    gold_rows = []
    for name, label, first, second in (
        ("strata", "층화 표본", base.strata, other.strata),
        ("f9", "F9 KorQuAD", base.f9, other.f9),
    ):
        pairs = moved_to_partial(first, second)
        judged = [
            (
                before,
                after,
                base.verdicts[before.item.id]
                if name == "strata"
                else cascade[(name, "base", before.item.id)],
                cascade[(name, "tag", after.item.id)],
            )
            for before, after in pairs
        ]
        counts = Counter(
            (first_verdict.verdict, then.verdict) for _, _, first_verdict, then in judged
        )
        for (before_label, after_label), count in sorted(
            counts.items(),
            key=lambda item: (VERDICTS.index(item[0][0]), VERDICTS.index(item[0][1])),
        ):
            pair_rows.append(
                {
                    "묶음": label,
                    "기준 grounded 판정": VERDICT_LABELS[before_label],
                    f"{tag} partial 문서 칸 판정": VERDICT_LABELS[after_label],
                    "문항": count,
                }
            )
        gold_rows.append(_cascade_gold_row(label, judged))
    reviewed = [verdict for verdict in cascade.values() if verdict.reviewed]
    changed = sum(verdict.first_verdict != verdict.verdict for verdict in reviewed)
    return (
        render_table(
            pair_rows, ["묶음", "기준 grounded 판정", f"{tag} partial 문서 칸 판정", "문항"]
        )
        + "\n"
        + render_table(
            gold_rows,
            [
                "묶음",
                "문항",
                "기준 grounded 답 정답 포함",
                "문서 칸 정답 포함",
                "일반 지식 칸에만 정답",
                "두 칸 모두 정답 없음",
                "문서 칸 질문을 비껴감",
            ],
        )
        + "\n- 대상: 기준에서 grounded 로 끝났는데 이 기록에서 '못 찾음' → partial 로 넘어간 KorQuAD 문항. 같은 문서를 받았다 "
        "(캐스케이드는 문맥을 다시 고르지 않는다)\n"
        "- partial 답은 문서 칸만 위 판정 지시문 그대로 판정했다 — 일반 지식 칸은 판정하지 않고 정답 문자열만 셌다. "
        f"문서 칸이 문서에 답이 없다고 물러서면 {VERDICT_LABELS['body_refusal']}\n"
        "- 기준 쪽 판정: 층화 표본은 high 확정 판정, F9 는 기준 기록이 정답 문자열로만 채점돼 있어 이번에 함께 판정했다. "
        "판정 입력은 기록 · 쪽(기준 / 이 기록)을 가리고 섞었다 (가린 번호 C001…)\n"
        f"- 판정 {len(cascade)} 중 메인 세션이 다시 본 것 {len(reviewed)}, 바뀐 판정 {changed}\n"
    )


def _cascade_gold_row(
    label: str,
    judged: Sequence[tuple[AnswerRecord, AnswerRecord, GroundingVerdict, GroundingVerdict]],
) -> dict:
    document_gold = 0
    general_only = 0
    for _, after, _, _ in judged:
        parts = sections(after.answer)
        if contains_gold(replace(after, answer=parts.get(DOC_HEADING, ""))):
            document_gold += 1
        elif contains_gold(replace(after, answer=parts.get(GENERAL_HEADING, ""))):
            general_only += 1
    return {
        "묶음": label,
        "문항": len(judged),
        "기준 grounded 답 정답 포함": _fmt(
            share(contains_gold(before) for before, _, _, _ in judged)
        ),
        "문서 칸 정답 포함": document_gold,
        "일반 지식 칸에만 정답": general_only,
        "두 칸 모두 정답 없음": len(judged) - document_gold - general_only,
        "문서 칸 질문을 비껴감": sum(then.off_question for _, _, _, then in judged),
    }


def _paired_columns(tag: str) -> list[str]:
    return ["지표", "기준", tag, "기준만", f"{tag}만", "McNemar p"]


def _paired_row(name: str, before: Mapping[str, bool], after: Mapping[str, bool], tag: str) -> dict:
    ids = sorted(before)
    test = mcnemar_exact([before[i] for i in ids], [after[i] for i in ids])
    return {
        "지표": name,
        "기준": _fmt(Share(sum(before.values()), len(ids))),
        tag: _fmt(Share(sum(after[i] for i in ids), len(ids))),
        "기준만": test.only_a,
        f"{tag}만": test.only_b,
        "McNemar p": f"{test.p_value:.3g}",
    }


def _side_row(
    name: str,
    base: Measurement,
    other: Measurement,
    tag: str,
    measure: Callable[[Sequence[AnswerRecord]], tuple[Share, str]],
) -> dict:
    before, denominator = measure(_korquad(base))
    after, _ = measure(_korquad(other))
    return {"지표": name, "기준": _fmt(before), tag: _fmt(after), "분모": denominator}


def _korquad(measurement: Measurement) -> list[AnswerRecord]:
    return [r for r in measurement.f9 if r.item.source == "korquad"]


def _korquad_grounded_share(records: Sequence[AnswerRecord]) -> tuple[Share, str]:
    return Share(sum(map(_ends_grounded, records)), len(records)), "KorQuAD 전부"


def _korquad_gold_share(records: Sequence[AnswerRecord]) -> tuple[Share, str]:
    grounded = [r for r in records if _ends_grounded(r)]
    return share(contains_gold(r) for r in grounded), "grounded 로 끝난 답"


def _korquad_cites_share(records: Sequence[AnswerRecord]) -> tuple[Share, str]:
    grounded = [r for r in records if _ends_grounded(r)]
    return share(
        cites_gold(r) for r in grounded
    ), "grounded 로 끝나고 문맥에 정답 근거 청크가 있는 답"


def _korquad_cascade_share(records: Sequence[AnswerRecord]) -> tuple[Share, str]:
    count = sum(r.stage_path == ("grounded", "partial") for r in records)
    return Share(count, len(records)), "KorQuAD 전부"


def _flags(
    records: Iterable[AnswerRecord], predicate: Callable[[AnswerRecord], bool]
) -> dict[str, bool]:
    return {record.item.id: predicate(record) for record in records}


def _verdict_flags(measurement: Measurement, labels: frozenset[str]) -> dict[str, bool]:
    return {
        r.item.id: _ends_grounded(r) and measurement.verdicts[r.item.id].verdict in labels
        for r in measurement.strata
    }


def _ends_grounded(record: AnswerRecord) -> bool:
    return final_stage(record) == "grounded"


def _grounded_with_gold(record: AnswerRecord) -> bool:
    return _ends_grounded(record) and bool(contains_gold(record))


def tally(prefix: str, ids: Sequence[str] | None) -> None:
    pattern = f"{ANSWERS_PATH.stem}-{prefix}-*{ANSWERS_PATH.suffix}"
    rounds: dict[str, list[AnswerRecord]] = {}
    prompts: dict[str, str] = {}
    for path in sorted(ANSWERS_PATH.parent.glob(pattern)):
        tag = path.stem.removeprefix(f"{ANSWERS_PATH.stem}-")
        rounds[tag] = [r for r in read_records(path) if ids is None or r.item.id in ids]
        # 멈췄다 다시 돌린 사이에 프롬프트가 바뀌었으면 둘 다 보인다
        prompts[tag] = " / ".join(sorted({_grounded_prompt(run) for run in _read_runs(path)}))
    if not rounds:
        raise SystemExit(
            f"{ANSWERS_PATH.parent / pattern} 가 없다 — run --tag {prefix}-…-N 을 먼저 돌린다"
        )
    print(render_tally(rounds, prompts))


def _grounded_prompt(run: Mapping) -> str:
    return run["settings"]["profiles"]["grounded"]["system"][-1]


def render_tally(rounds: Mapping[str, Sequence[AnswerRecord]], prompts: Mapping[str, str]) -> str:
    """회차 기록들의 단계 경로 · 첫 줄 표지 표 markdown — 문항 id 와 집계만 (질문 · 답 원문 없음).

    Args:
      rounds: 꼬리표 → 그 회차의 기록. 꼬리표 끝 -N 이 회차 번호, 그 앞이 묶음 (noise-v2-3 → noise-v2 · 3회차).
        끝에 번호가 없으면 꼬리표 전체가 묶음 · 1회차.
      prompts: 꼬리표 → 그 기록의 grounded 단계 블록 프롬프트 (meta 설정).
    """
    per_round = [
        _tally_round_row(tag, prompts[tag], rounds[tag]) for tag in sorted(rounds, key=_split_round)
    ]
    cells: dict[tuple[str, str], dict[int, AnswerRecord]] = defaultdict(dict)
    for tag, records in rounds.items():
        series, number = _split_round(tag)
        for record in records:
            cells[series, record.item.id][number] = record
    numbers = sorted({_split_round(tag)[1] for tag in rounds})
    per_item = [
        _tally_item_row(series, qid, by_round, numbers)
        for (series, qid), by_round in sorted(cells.items())
    ]
    return (
        render_table(
            per_round,
            [
                "꼬리표",
                "grounded 프롬프트",
                "문항",
                "grounded 로 끝남",
                "grounded → partial",
                "그 밖 경로",
                "찾음",
                "못 찾음",
                "그 밖 표지",
            ],
        )
        + "\n"
        + render_table(
            per_item,
            [
                "묶음",
                "id",
                "회차",
                "grounded 로 끝남",
                "grounded → partial",
                *(f"{number}회" for number in numbers),
            ],
        )
        + "\n- 같은 문항을 회차마다 따로 돌린 기록 — 같은 프롬프트로도 답이 갈린다 (Ollama GPU, temperature 0.1). "
        "표지는 grounded 첫 줄 (못 찾음이면 partial 로 넘어간다)\n"
    )


def _split_round(tag: str) -> tuple[str, int]:
    match = ROUND_TAG.fullmatch(tag)
    return (match["series"], int(match["round"])) if match else (tag, 1)


def _tally_round_row(tag: str, prompt: str, records: Sequence[AnswerRecord]) -> dict:
    paths = Counter(stage_path(record) for record in records)
    markers = Counter(_marker_label(record) for record in records)
    return {
        "꼬리표": tag,
        "grounded 프롬프트": f"`{prompt}`",
        "문항": len(records),
        "grounded 로 끝남": paths["grounded"],
        "grounded → partial": paths["grounded → partial"],
        "그 밖 경로": _others(paths, {"grounded", "grounded → partial"}),
        "찾음": markers[MARKER_LABELS["found"]],
        "못 찾음": markers[MARKER_LABELS["not_found"]],
        "그 밖 표지": _others(markers, {MARKER_LABELS["found"], MARKER_LABELS["not_found"]}),
    }


def _tally_item_row(
    series: str, qid: str, by_round: Mapping[int, AnswerRecord], numbers: Sequence[int]
) -> dict:
    paths = Counter(stage_path(record) for record in by_round.values())
    return {
        "묶음": series,
        "id": qid,
        "회차": len(by_round),
        "grounded 로 끝남": paths["grounded"],
        "grounded → partial": paths["grounded → partial"],
        **{
            f"{number}회": (
                f"{stage_path(by_round[number])} · {_marker_label(by_round[number])}"
                if number in by_round
                else ""
            )
            for number in numbers
        },
    }


def _marker_label(record: AnswerRecord) -> str:
    outcome = marker_outcome(record)
    return MARKER_LABELS[outcome] if outcome is not None else "표지 없음"


def _others(counts: Counter[str], shown: set[str]) -> str:
    return " · ".join(
        f"{name} {count}" for name, count in sorted(counts.items()) if name not in shown
    )


def report() -> None:
    if not ANSWERS_PATH.exists():
        raise SystemExit(f"{ANSWERS_PATH} 가 없다 — run 을 먼저 돌린다")
    records = read_records(ANSWERS_PATH)
    runs = json.loads(ANSWERS_PATH.with_suffix(".meta.json").read_text(encoding="utf-8"))["runs"]
    sample_count = len(_read_jsonl(SAMPLE_PATH))
    sensitivity: tuple[list[dict], dict] | None = None
    if SENSITIVITY_PATH.exists():
        meta_path = SENSITIVITY_PATH.with_suffix(".meta.json")
        sensitivity = (
            _read_jsonl(SENSITIVITY_PATH),
            json.loads(meta_path.read_text(encoding="utf-8")),
        )
    today = datetime.now(tz=UTC).date()
    text = render_report(records, runs, sample_count, sensitivity, today=today)
    REPORT_DIR.mkdir(exist_ok=True)
    path = REPORT_DIR / f"{today:%Y-%m-%d}-answer-eval.md"
    # Windows 에서도 LF 로 (리포트는 커밋하고 .gitattributes 가 LF)
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"기록 {len(records)} / 표본 {sample_count} · 리포트: {path}")


def render_report(
    records: Sequence[AnswerRecord],
    runs: Sequence[Mapping],
    sample_count: int,
    sensitivity: tuple[Sequence[Mapping], Mapping] | None,
    *,
    today: date,
) -> str:
    """리포트 markdown — 집계만 (질문 · 답 원문 · 청크 본문을 싣지 않는다)."""
    korquad = [record for record in records if record.item.source == "korquad"]
    offwiki = [record for record in records if record.item.source == "offwiki"]
    return (
        f"# F9 답변 평가 — 자동 지표 ({today:%Y-%m-%d})\n\n"
        + _overview(records, korquad, offwiki, runs, sample_count)
        + "\n## 생성 설정 — 기록의 spec_id · 프롬프트 버전\n\n"
        + _spec_table(records)
        + "\n## 단계 경로\n\n"
        + _path_section(korquad, offwiki)
        + "\n## KorQuAD — 근거 답변\n\n"
        + _korquad_section(korquad)
        + "\n## 리랭커 1위 점수 구간 — KorQuAD (ADR-024)\n\n"
        + _band_section(korquad)
        + "\n## 표지 · 출처 번호 · 잘림 (LLM 을 부른 문항 전부)\n\n"
        + _marker_section(records)
        + "\n## 지연 (초) — 질문을 받은 때부터\n\n"
        + _latency_section(records)
        + "\n## 위키 밖 질문 — 기대 단계와 실제 경로\n\n"
        + _offwiki_section(offwiki)
        + "\n## 검색의 표현 민감도 — 원문 vs 패러프레이즈\n\n"
        + _sensitivity_section(sensitivity)
        + "\n## 다음 일\n\n"
        + "".join(f"- {step}\n" for step in NEXT_STEPS)
    )


def _overview(
    records: Sequence[AnswerRecord],
    korquad: Sequence[AnswerRecord],
    offwiki: Sequence[AnswerRecord],
    runs: Sequence[Mapping],
    sample_count: int,
) -> str:
    settings = runs[-1]["settings"]
    distinct = {json.dumps(run["settings"], sort_keys=True) for run in runs}
    changed = (
        f"- **실행 {len(runs)}번 사이에 설정이 바뀌었다** ({len(distinct)}가지) — 아래 수치는 섞여 있다. "
        "생성 설정 표의 spec_id · 프롬프트 버전으로 나눠 본다\n"
        if len(distinct) > 1
        else ""
    )
    thresholds = settings["thresholds"]
    return (
        f"- 입력: KorQuAD dev eval 몫(문단 해시 40%, 문단마다 한 문항, 정답 문서가 코퍼스에 있고 정답 문장이 지금 "
        f"문서에 남은 것) {len(korquad)} + 위키 밖 질문 offwiki v1 {len(offwiki)} — 표본 {sample_count} 중 "
        f"기록 {len(records)}\n"
        f"- 답변 경로: `/ask` 와 같은 조립을 서버 없이 — 관문 {'켬' if settings['gate'] else '끔'} → 검색 "
        f"`{settings['collection']}` Hybrid(후보 {settings['candidates']}) + 리랭커 `{settings['reranker_model']}`"
        f"(후보 {settings['rerank_candidates']}) 상위 {settings['top_k']} → 답변 정책(high {thresholds['high']:.6f} · "
        f"low {thresholds['low']:.6f}) → grounded@v2 첫 줄 표지, 못 찾음 · 빈 답 · 시간 초과면 partial "
        "(ADR-024 · ADR-027)\n"
        + changed
        + "- 정답 포함: 답에 KorQuAD 정답 문자열이 있음 (공백 무시 — evalkit `contains_answer` 규칙). "
        "답이 문장이라 EM · F1 은 쓰지 않는다\n"
        "- 정답 근거 청크: 정답 문서의 청크이면서 정답 문자열을 담은 것 (검색 평가와 같은 기준)\n"
        "- 위키 밖 v1 은 임계값을 고를 때(calibrate) 쓴 세트라 검증이 아니라 경로 확인용이다 — 검증은 offwiki v2\n"
        "- 질문 · 답 원문은 로컬 기록(data/eval/f9-answers-v1.jsonl)에만 둔다 (KorQuAD CC BY-ND)\n"
    )


def _spec_table(records: Sequence[AnswerRecord]) -> str:
    counts = Counter(
        (final_stage(record), record.spec_id, record.prompt_version)
        for record in records
        if record.spec_id is not None
    )
    rows = [
        {"끝난 단계": stage, "spec_id": spec_id, "프롬프트 버전": f"`{version}`", "문항": count}
        for (stage, spec_id, version), count in sorted(counts.items())
    ]
    if not rows:
        return "LLM 을 부른 기록이 없다\n"
    return render_table(rows, ["끝난 단계", "spec_id", "프롬프트 버전", "문항"])


def _path_section(korquad: Sequence[AnswerRecord], offwiki: Sequence[AnswerRecord]) -> str:
    groups = {
        "KorQuAD · 검색 상위에 정답 근거": [r for r in korquad if gold_in_hits(r)],
        "KorQuAD · 상위에 없음": [r for r in korquad if not gold_in_hits(r)],
        "위키 밖": list(offwiki),
    }
    counts = {name: path_counts(group) for name, group in groups.items()}
    rows = [
        {"경로": path, **{name: counts[name].get(path, 0) for name in groups}}
        for path in path_counts([*korquad, *offwiki])
    ]
    rows.append({"경로": "합계", **{name: len(group) for name, group in groups.items()}})
    return (
        render_table(rows, ["경로", *groups])
        + "\n'검색 상위' 는 답변 경로가 받은 상위 결과 전부 (low 미만이라 문맥에서 뺀 것 포함). "
        "→ 는 캐스케이드(못 찾음 · 빈 답 · 표지 전 잘림 · 시간 초과로 다음 단계), 끝의 error 는 답을 하나도 못 보낸 "
        f"실패, 끝의 {FIXED_REFUSAL} 은 partial 이 생성 조각 없이 고정 거절 문구로 끝난 것\n"
    )


def _korquad_section(korquad: Sequence[AnswerRecord]) -> str:
    grounded = [r for r in korquad if final_stage(r) == "grounded"]
    partial = [r for r in korquad if final_stage(r) == "partial"]
    rows = [
        {
            "지표": "검색 상위에 정답 근거 청크 (답변 경로의 hit@k)",
            "값": _fmt(share(gold_in_hits(r) for r in korquad)),
            "분모": "KorQuAD 전부",
        },
        {
            "지표": "grounded 로 끝남",
            "값": _fmt(Share(len(grounded), len(korquad))),
            "분모": "KorQuAD 전부",
        },
        {
            "지표": "정답 포함",
            "값": _fmt(share(contains_gold(r) for r in grounded)),
            "분모": "grounded 로 끝난 답",
        },
        {
            "지표": "정답 근거 청크를 출처 번호로 짚음",
            "값": _fmt(share(cites_gold(r) for r in grounded)),
            "분모": "grounded 로 끝나고 문맥에 정답 근거 청크가 있는 답",
        },
        {
            "지표": "출처 번호가 하나도 없음",
            "값": _fmt(share(not r.cited for r in grounded)),
            "분모": "grounded 로 끝난 답",
        },
        {
            "지표": "정답 포함 (partial)",
            "값": _fmt(share(contains_gold(r) for r in partial)),
            "분모": "partial 이 답을 쓰고 끝난 문항 (캐스케이드 포함) — 일반 지식이나 다른 문서로 맞힌 것",
        },
        {
            "지표": "고정 거절 문구로 끝남 (partial 빈 답)",
            "값": _fmt(
                Share(sum(final_stage(r) == "partial_empty" for r in korquad), len(korquad))
            ),
            "분모": "KorQuAD 전부",
        },
    ]
    return render_table(rows, ["지표", "값", "분모"])


def _band_section(korquad: Sequence[AnswerRecord]) -> str:
    groups: dict[str, list[AnswerRecord]] = defaultdict(list)
    for record in korquad:
        groups[score_band(record.top_score)].append(record)
    rows = []
    for band in score_bands():
        group = groups.get(band, [])
        if not group:
            continue
        grounded = [r for r in group if final_stage(r) == "grounded"]
        rows.append(
            {
                "1위 점수": band,
                "문항": len(group),
                "grounded 로 끝남": len(grounded),
                "그중 정답 포함": _fmt(share(contains_gold(r) for r in grounded)),
                "grounded → partial": sum(r.stage_path == ("grounded", "partial") for r in group),
                "partial 만": sum(r.stage_path == ("partial",) for r in group),
                "그 밖": sum(
                    r.stage_path not in {("grounded",), ("grounded", "partial"), ("partial",)}
                    for r in group
                ),
            }
        )
    lower, upper = SCORE_BAND_EDGES
    middle = [r for r in korquad if r.top_score is not None and lower <= r.top_score < upper]
    middle_grounded = sum(final_stage(r) == "grounded" for r in middle)
    return (
        render_table(
            rows,
            [
                "1위 점수",
                "문항",
                "grounded 로 끝남",
                "그중 정답 포함",
                "grounded → partial",
                "partial 만",
                "그 밖",
            ],
        )
        + f"\n- {lower:g}~{upper:g} 구간에서 grounded 로 끝난 답 {middle_grounded}개 — 구간끼리 지어냄 비율을 가르려면 "
        "구간당 약 100개가 필요하다 (ADR-024). 이 표본은 점수로 층화하지 않아 적다 → 층화 표본 · 지어냄 판정은 다음 일\n"
        "- 정답 미포함이 곧 지어냄은 아니다 (같은 사실을 다른 말로 · 일부만 답함). 지어냄은 문맥과 대조해 판정한다\n"
    )


def _marker_section(records: Sequence[AnswerRecord]) -> str:
    missing = share(marker_missing(r) for r in records)
    _, missing_upper = clopper_pearson(missing.count, missing.total) if missing.total else (0, 1)
    mismatch = share(marker_mismatch(r) for r in records)
    mismatch_gold = sum(bool(marker_mismatch(r)) and bool(contains_gold(r)) for r in records)
    called = [r for r in records if ANSWER_STAGES.intersection(r.stage_path)]
    written = [r for r in records if answered(r)]
    partial_ended = [r for r in records if final_stage(r) in {"partial", "partial_empty"}]
    errors = Counter(r.error for r in records if r.error is not None)
    rows = [
        {
            "지표": "표지 누락",
            "값": _fmt(missing),
            "분모": "grounded 가 표지를 쓰거나 표지 없이 답을 쓴 문항",
            "메모": f"95% 상한 {missing_upper:.1%} — {MARKER_MISSING_LIMIT:.0%} 미만이면 "
            "'표지 없으면 찾음' 유지, 이상이면 반대로 (ADR-024). 표지 전 잘림 · 빈 답은 뺀다",
        },
        {
            "지표": "표지 전 잘림",
            "값": _fmt(share(cut_before_marker(r) for r in records)),
            "분모": "grounded 를 돌려 done 으로 끝난 문항",
            "메모": "표지를 정하기 전에 max_tokens 에 닿아 아무것도 안 보내고 partial 로 — 누락이 아니다",
        },
        {
            "지표": "표지 '찾음' · 본문 '정보 없음' 어긋남",
            "값": _fmt(mismatch),
            "분모": "grounded 가 '찾음'으로 답을 낸 문항",
            "메모": f"정규식이라 상한 (곁가지 '정보 없음'도 걸린다). 그중 KorQuAD 정답 포함 {mismatch_gold}",
        },
        {
            "지표": "범위 밖 출처 번호를 뗀 답",
            "값": _fmt(share(has_dropped(r) for r in records)),
            "분모": "LLM 이 답을 쓴 문항",
            "메모": "서버가 떼고 로그에 남긴다 (ADR-026)",
        },
        {
            "지표": "grounded 잘림 (max_tokens 도달)",
            "값": _fmt(share(stage_truncated(r, "grounded") for r in records)),
            "분모": "grounded 를 돌려 done 으로 끝난 문항",
            "메모": "done_reason length — 답을 낸 것 · 잘려 partial 로 넘어간 것(표지 전 잘림 포함) 모두",
        },
        {
            "지표": "partial 잘림 (max_tokens 도달)",
            "값": _fmt(share(stage_truncated(r, "partial") for r in records)),
            "분모": "partial 을 돌려 done 으로 끝난 문항",
            "메모": "빈 채 잘려 고정 거절 문구로 끝난 것 포함",
        },
        {
            "지표": "고정 거절 문구로 끝남 (partial 빈 답)",
            "값": _fmt(
                Share(
                    sum(final_stage(r) == "partial_empty" for r in partial_ended),
                    len(partial_ended),
                )
            ),
            "분모": "partial 로 끝난 문항",
            "메모": "생성 조각이 하나도 없음(첫 글자 시간이 비었다) — LLM 이 쓴 답 분모에서 뺀다",
        },
        {
            "지표": "시간 초과로 끊김",
            "값": _fmt(Share(sum(r.timed_out for r in written), len(written))),
            "분모": "LLM 이 답을 쓴 문항",
            "메모": "보낸 데까지로 done",
        },
        {
            "지표": "답을 못 보낸 실패 (error)",
            "값": _fmt(Share(sum(errors.values()), len(called))),
            "분모": "LLM 을 부른 문항",
            "메모": " · ".join(f"{code} {count}" for code, count in sorted(errors.items()))
            or "없음",
        },
    ]
    outcomes = Counter(marker_outcome(r) for r in records)
    breakdown = " · ".join(f"{MARKER_LABELS[o]} {outcomes[o]}" for o in MARKER_OUTCOMES)
    legacy = sum(not r.done_reason for r in called if r.error is None)
    legacy_note = (
        f"- done_reason 이 없는 예전 기록 {legacy}개는 단계별 잘림 · 표지 전 잘림의 분모에서 빠졌다\n"
        if legacy
        else ""
    )
    return (
        render_table(rows, ["지표", "값", "분모", "메모"]) + f"\n- grounded 첫 줄: {breakdown}\n"
        "- LLM 이 쓴 답 = grounded · partial 이 생성 조각을 보내고 끝난 문항 (관문 · 거절 · 고정 거절 문구는 뺀다)\n"
        + legacy_note
    )


def _latency_section(records: Sequence[AnswerRecord]) -> str:
    groups: dict[str, list[AnswerRecord]] = defaultdict(list)
    for record in records:
        groups[stage_path(record)].append(record)
    rows = [
        _latency_row(path, groups[path])
        for path in path_counts(records)
        if not path.endswith("error")
    ]
    columns = [
        "경로",
        "문항",
        *(f"첫 글자 p{q}" for q in LATENCY_PERCENTILES),
        *(f"전체 p{q}" for q in LATENCY_PERCENTILES),
    ]
    return (
        render_table(rows, columns)
        + "\n첫 글자 = 첫 답 조각을 보낸 때 (관문 · 거절 · 고정 거절 문구는 생성이 아니라 비운다), 전체 = done 까지. "
        "done 진단의 시간 — 실패(error)로 끝난 문항은 진단이 없어 뺀다\n"
    )


def _latency_row(path: str, group: Sequence[AnswerRecord]) -> dict:
    first = latency(r.first_delta_ms for r in group)
    total = latency(r.total_ms for r in group)
    return {
        "경로": path,
        "문항": len(group),
        **{f"첫 글자 p{q}": _seconds(first, q) for q in LATENCY_PERCENTILES},
        **{f"전체 p{q}": _seconds(total, q) for q in LATENCY_PERCENTILES},
    }


def _offwiki_section(offwiki: Sequence[AnswerRecord]) -> str:
    if not offwiki:
        return "기록이 없다\n"
    groups: dict[str, list[AnswerRecord]] = defaultdict(list)
    totals: dict[str, list[AnswerRecord]] = defaultdict(list)
    for record in offwiki:
        name = f"{record.item.expected}/{record.item.category}"
        groups[name + (" (trap)" if record.item.trap else "")].append(record)
        totals[f"**{record.item.expected} 전체**"].append(record)
    rows = [_offwiki_row(name, group) for name, group in sorted(groups.items())]
    rows += [_offwiki_row(name, group) for name, group in sorted(totals.items())]
    return (
        render_table(rows, ["묶음", "문항", *FINAL_STAGES, "기대대로"])
        + "\n- 기대대로: refuse → 관문 · 거절 · 고정 거절 문구, partial → partial. partial_empty = partial 이 생성 조각 "
        "없이 고정 거절 문구로 끝남. trap = 이름이 같은 문서가 있어 검색에 걸리기 쉬운 질문\n"
        "- refuse 기대가 partial 로 끝난 것은 설계상 허용이다 (ADR-025 — 거절은 점수가 아니라 partial 프롬프트의 "
        "'알 수 없음'과 관문이 맡는다). 그 답이 괜찮은지는 분리 답변 판정(다음 일)이 본다\n"
        "- partial 기대가 grounded 로 끝난 것은 '찾음' 표지로 문서 근거라고 답한 것 — 지어냄 후보다\n"
        + _offwiki_review(offwiki)
    )


def _offwiki_review(offwiki: Sequence[AnswerRecord]) -> str:
    """partial 기대가 grounded 로 끝난 답의 확인 집계 한 줄. 그런 답이 없으면 빈 문자열."""
    candidates = [
        r for r in offwiki if r.item.expected == "partial" and final_stage(r) == "grounded"
    ]
    if not candidates:
        return ""
    reviewed = [
        r
        for r in candidates
        if r.item.id in OFFWIKI_REVIEW and r.prompt_version == OFFWIKI_REVIEW_PROMPT
    ]
    if not reviewed:
        return f"- 그 답의 확인: 아직 ({len(candidates)}개)\n"
    labels = Counter(OFFWIKI_REVIEW[r.item.id] for r in reviewed)
    counts = ", ".join(
        f"{VERDICT_LABELS[label]} {labels[label]}" for label in VERDICTS if labels[label]
    )
    off_question = sum(r.item.id in OFFWIKI_OFF_QUESTION for r in reviewed)
    complete = {r.item.id for r in reviewed} == set(OFFWIKI_REVIEW)
    return (
        "- 그 답의 확인 (2026-10-29, Claude 세션이 받은 문맥 · 위키 원문과 대조 — 라벨은 high 확정 판정 지시문): "
        f"{len(candidates)}개 중 {len(reviewed)}개 — {counts}. 덧표시 질문을 비껴감 {off_question}"
        + (f". {OFFWIKI_REVIEW_NOTE}" if complete else "")
        + "\n"
    )


def _offwiki_row(name: str, group: Sequence[AnswerRecord]) -> dict:
    return {
        "묶음": name,
        "문항": len(group),
        **stage_counts(group),
        "기대대로": _fmt(share(matches_expected(r) for r in group)),
    }


def _sensitivity_section(sensitivity: tuple[Sequence[Mapping], Mapping] | None) -> str:
    if sensitivity is None:
        return "아직 돌리지 않았다 (`retrieval-sensitivity`)\n"
    rows, meta = sensitivity
    k = int(meta["top_k"])
    table_rows = [_sensitivity_row(f"hit@{cutoff}", rows, cutoff) for cutoff in sorted({1, k})]
    kept = [row for row in rows if row["id"] not in PARAPHRASE_SHIFTED]
    if len(kept) < len(rows):
        label = f"hit@{k} · 뜻 달라짐 {len(rows) - len(kept)}개 뺌"
        table_rows.append(_sensitivity_row(label, kept, k))
    return (
        render_table(
            table_rows,
            [
                "기준",
                "원문",
                "패러프레이즈",
                "차이 (%p)",
                "둘 다",
                "원문만",
                "패러프레이즈만",
                "둘 다 실패",
                "McNemar p",
            ],
        )
        + f"\n- 표본: KorQuAD dev eval 몫에서 검색 결과로 거르지 않고 뽑은 {len(rows)}개 (정답 문장이 지금 문서에 남은 "
        "것만 — 코퍼스 내용으로 거른 것이라 선택 편향이 없다)\n"
        f"- 패러프레이즈 v1 은 Claude 로 만들었다. 사람 확인(이 표본 몫, seed 0): {PARAPHRASE_REVIEW}. "
        "뜻이 달라진 문항도 정답은 그대로라 빼지 않고 전부로 쟀고, 뺀 값을 한 줄 더 냈다\n"
        f"- 검색: `{meta['collection']}` Hybrid(후보 {meta['candidates']}) + 리랭커 `{meta['reranker_model']}` 상위 {k} "
        "— `/ask` 와 같은 설정. hit = 상위에 정답 근거 청크(정답 문서이고 정답 문자열을 담음)가 있음\n"
        "- 원문 hit 가 실제 검색 성공률이고, 패러프레이즈와의 차이가 질문 · 지문의 낱말 겹침으로 부풀려진 몫이다. "
        "McNemar p 는 엇갈린 짝(한쪽만 성공)만 본 정확 검정\n"
    )


def _sensitivity_row(label: str, rows: Sequence[Mapping], cutoff: int) -> dict:
    table = paired_hits(_hit_pairs(rows, cutoff))
    return {
        "기준": label,
        "원문": _percent(table.original_rate),
        "패러프레이즈": _percent(table.paraphrase_rate),
        "차이 (%p)": _points(table.original_rate, table.paraphrase_rate),
        "둘 다": table.both,
        "원문만": table.original_only,
        "패러프레이즈만": table.paraphrase_only,
        "둘 다 실패": table.neither,
        "McNemar p": f"{table.mcnemar_p:.3g}",
    }


def _fmt(value: Share) -> str:
    if value.rate is None:
        return "0/0"
    return f"{value.count}/{value.total} ({value.rate:.1%})"


def _seconds(value: Latency, q: int) -> str:
    milliseconds = value.percentiles.get(q)
    return "" if milliseconds is None else f"{milliseconds / 1000:.1f}"


def _percent(rate: float | None) -> str:
    return "" if rate is None else f"{rate:.1%}"


def _points(first: float | None, second: float | None) -> str:
    if first is None or second is None:
        return ""
    return f"{(first - second) * 100:+.1f}"


def _document_texts(titles: set[str]) -> dict[str, str]:
    """정규화한 제목 → 본문. 필요한 제목만 담는다."""
    texts: dict[str, str] = {}
    for path in sorted(PARSED.glob("docs-*.parquet")):
        table = pq.read_table(path, columns=["title", "text"]).to_pydict()
        for title, text in zip(table["title"], table["text"], strict=True):
            normalized = normalize_title(title)
            if normalized in titles:
                texts[normalized] = text
    return texts


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: Iterable[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_jsonl(path: Path, row: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
