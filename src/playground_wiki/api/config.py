"""API 설정: configs/*.yaml 과 환경변수를 읽어 서비스 구성으로 바꾼다."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Self

import yaml
from playground_llm import ModelSpec
from playground_retrieval.search import SearchSettings
from playground_retrieval.store import PgConfig

from playground_wiki.gate import QueryGate, Tokenizer
from playground_wiki.generation import AnswerPolicy, Profile, Thresholds, korean_date

CONFIG_DIR = Path("configs")  # 레포 루트에서 실행할 때. RAG_CONFIG_DIR 로 바꾼다
BM25_INDEX_DIR = Path("data/index/bm25")  # scripts/build_bm25.py 가 {컬렉션} 폴더로 저장한다
ANSWER_CONFIG = "answer.yaml"
# answer.yaml 프로필 한 행의 키. 오타 난 키가 조용히 무시되지 않게 정확히 이 집합이어야 한다
PROFILE_KEYS = frozenset({"provider", "model", "params", "system", "user", "timeout_sec"})


@dataclass(frozen=True)
class ApiConfig:
    """API 프로세스 하나의 구성. 이 레포의 환경변수(RAG_*)는 여기(`from_env`)에서만 읽는다."""

    search: SearchSettings
    pg: PgConfig
    config_dir: Path
    """설정 파일 폴더 (answer.yaml · ingest.yaml · retrieval.yaml)."""
    log_questions: bool
    """/ask 질문 원문을 로그에 남길지 (ADR-027). 꺼져 있으면 길이와 해시만."""

    @classmethod
    def from_env(cls) -> Self:
        """RAG_CONFIG_DIR(기본 configs/) 의 ingest.yaml · retrieval.yaml 과 환경변수로 만든다.

        컬렉션은 RAG_COLLECTION, 없으면 ingest.yaml 의 store.collection 템플릿(`{source}_{model}`).
        서브셋을 띄울 때는 RAG_COLLECTION=kowiki-subset-10000_kure-v1 처럼 지정한다.
        BM25 색인은 RAG_BM25_INDEX_DIR(기본 data/index/bm25) 아래 컬렉션 이름 폴더.
        질문 원문 로그는 RAG_LOG_QUESTIONS=1 일 때만 켠다 — 기본은 원문을 남기지 않는 쪽이라 true · yes 로는 켜지지 않는다.
        """
        config_dir = Path(os.environ.get("RAG_CONFIG_DIR", CONFIG_DIR))
        ingest = yaml.safe_load((config_dir / "ingest.yaml").read_text(encoding="utf-8"))
        retrieval = yaml.safe_load((config_dir / "retrieval.yaml").read_text(encoding="utf-8"))
        source = ingest["source"]["name"]
        model = ingest["embedding"]["model"]
        collection = os.environ.get("RAG_COLLECTION") or ingest["store"]["collection"].format(
            source=source, model=model
        )
        # HybridRetriever 는 검색기마다 같은 후보 수를 쓴다
        if retrieval["dense"]["top_k"] != retrieval["bm25"]["top_k"]:
            raise ValueError("retrieval.yaml: dense.top_k and bm25.top_k must be equal")
        search = SearchSettings(
            collection=collection,
            source=source,
            model=model,
            candidates=retrieval["dense"]["top_k"],
            rrf_k=retrieval["fusion"]["rrf_k"],
            rerank_candidates=retrieval["fusion"]["top_k"],
            reranker_model=retrieval["reranker"]["model"],
            top_k=retrieval["reranker"]["top_k"],
            bm25_index=Path(os.environ.get("RAG_BM25_INDEX_DIR", BM25_INDEX_DIR)) / collection,
        )
        return cls(
            search=search,
            pg=PgConfig.from_env(),
            config_dir=config_dir,
            log_questions=os.environ.get("RAG_LOG_QUESTIONS") == "1",
        )


def load_dump_date(config_dir: Path) -> str:
    """ingest.yaml 의 덤프 날짜(YYYYMMDD)를 프롬프트 · 관문 고정 답에 넣을 꼴로 ("2026년 10월 1일")."""
    ingest = yaml.safe_load((config_dir / "ingest.yaml").read_text(encoding="utf-8"))
    return korean_date(datetime.strptime(str(ingest["source"]["dump_date"]), "%Y%m%d").date())


def load_query_gate(config_dir: Path, tokenizer: Tokenizer) -> QueryGate | None:
    """answer.yaml 의 gate.enabled 가 켜져 있으면 질문 관문을 만든다. 꺼져 있으면 None.

    분석기는 부르는 쪽이 넘긴다 — Kiwi 하나가 메모리를 약 500MB 써서, 검색(BM25)과 같은 인스턴스를 쓰게 한다.

    Raises:
      TypeError: gate.enabled 가 true · false 가 아닐 때 (따옴표 친 "yes" 같은 문자열).
    """
    raw = yaml.safe_load((config_dir / ANSWER_CONFIG).read_text(encoding="utf-8"))
    enabled = raw["gate"]["enabled"]
    if not isinstance(enabled, bool):
        raise TypeError(f"{ANSWER_CONFIG}: gate.enabled must be true or false, got {enabled!r}")
    return QueryGate(tokenizer) if enabled else None


def load_answer_policy(config_dir: Path) -> AnswerPolicy:
    """config_dir 의 answer.yaml(임계값 · 단계별 프로필)로 답변 정책을 만든다.

    Raises:
      ValueError: 프로필 값이 틀렸거나, 임계값을 아직 정하지 않았을 때(null).
    """
    raw = yaml.safe_load((config_dir / ANSWER_CONFIG).read_text(encoding="utf-8"))
    profiles = parse_profiles(raw["profiles"])
    high, low = raw["thresholds"]["high"], raw["thresholds"]["low"]
    if high is None or low is None:
        raise ValueError(
            f"{ANSWER_CONFIG}: thresholds are not calibrated yet — run scripts/calibrate_threshold.py"
        )
    return AnswerPolicy(Thresholds(high=float(high), low=float(low)), profiles)


def parse_profiles(raw: Mapping[str, Mapping[str, Any]]) -> dict[str, Profile]:
    """answer.yaml 의 profiles 표를 단계별 Profile 로 바꾼다. YAML 목록은 튜플로."""
    return {stage: _profile(stage, row) for stage, row in raw.items()}


def _profile(stage: str, row: Mapping[str, Any]) -> Profile:
    if set(row) != PROFILE_KEYS:
        raise ValueError(
            f"{ANSWER_CONFIG}: profile {stage!r} keys must be {sorted(PROFILE_KEYS)}, "
            f"got {sorted(row)}"
        )
    # 한도가 없거나 -1 이면 Ollama 가 끝없이 생성하고 timeout_sec 만 남는다 (보안 점검 2026-10-25)
    max_tokens = row["params"].get("max_tokens")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError(
            f"{ANSWER_CONFIG}: profile {stage!r} params.max_tokens must be a positive int, "
            f"got {max_tokens!r}"
        )
    params = {
        key: tuple(value) if isinstance(value, list) else value
        for key, value in row["params"].items()
    }
    return Profile(
        spec=ModelSpec(row["provider"], row["model"], params),
        system=tuple(row["system"]),
        user=row["user"],
        timeout_sec=float(row["timeout_sec"]),
    )
