"""API 설정: configs/*.yaml 과 환경변수를 읽어 서비스 구성으로 바꾼다."""

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Self, cast, get_args

import yaml
from playground_llm import ModelSpec, PromptLibrary
from playground_retrieval.search import SearchSettings
from playground_retrieval.store import PgConfig
from playground_retrieval.store.qdrant import MODE_ENV as QDRANT_MODE_ENV, QDRANT_MODES, QdrantMode

from playground_wiki.api.schemas import MAX_K
from playground_wiki.gate import QueryGate, Tokenizer
from playground_wiki.generation import (
    PROMPT_DIR,
    AnswerPolicy,
    Profile,
    Thresholds,
    check_profiles,
    korean_date,
    swap_prompt,
)

CONFIG_DIR = Path("configs")  # 레포 루트에서 실행할 때. RAG_CONFIG_DIR 로 바꾼다
BM25_INDEX_DIR = Path("data/index/bm25")  # scripts/build_bm25.py 가 {컬렉션} 폴더로 저장한다
ANSWER_CONFIG = "answer.yaml"
INGEST_CONFIG = "ingest.yaml"
# answer.yaml 프로필 한 행의 키. 오타 난 키가 조용히 무시되지 않게 정확히 이 집합이어야 한다
PROFILE_KEYS = frozenset({"provider", "model", "params", "system", "user", "timeout_sec"})
Impl = Literal["real", "mock"]
"""서버 구현 (RAG_IMPL). real 은 PG · Qdrant · 검색 모델 · Ollama, mock 은 예시 검색 결과 · FakeLLM (api.mock)."""
IMPLS: tuple[str, ...] = get_args(Impl)


@dataclass(frozen=True)
class ApiConfig:
    """API 프로세스 하나의 구성. 이 레포의 환경변수(RAG_*)는 여기(`from_env`)에서만 읽는다."""

    search: SearchSettings
    pg: PgConfig
    config_dir: Path
    """설정 파일 폴더 (answer.yaml · ingest.yaml · retrieval.yaml)."""
    log_questions: bool
    """/ask 질문 원문을 로그에 남길지 (ADR-027). 꺼져 있으면 길이와 해시만."""
    impl: Impl
    """서버 구현. mock 이면 저장소 · 모델 · Ollama 에 접속하지 않는다."""
    qdrant_mode: QdrantMode
    """벡터 저장소 Qdrant 의 위치 (local · cloud). 접속은 retrieval 이 같은 QDRANT_MODE 로 하고, 여기서는 /health 에 알린다."""

    @classmethod
    def from_env(cls) -> Self:
        """RAG_CONFIG_DIR(기본 configs/) 의 ingest.yaml · retrieval.yaml 과 환경변수로 만든다.

        컬렉션은 RAG_COLLECTION, 없으면 ingest.yaml 의 store.collection 템플릿(`{source}_{model}`).
        서브셋을 띄울 때는 RAG_COLLECTION=kowiki-subset-10000_kure-v1 처럼 지정한다.
        BM25 색인은 RAG_BM25_INDEX_DIR(기본 data/index/bm25) 아래 컬렉션 이름 폴더.
        질문 원문 로그는 RAG_LOG_QUESTIONS=1 일 때만 켠다 — 기본은 원문을 남기지 않는 쪽이라 true · yes 로는 켜지지 않는다.
        서버 구현은 RAG_IMPL(real · mock, 기본 real).
        Qdrant 위치는 retrieval 이 접속할 때 읽는 QDRANT_MODE(local · cloud, 기본 local)를 같은 규칙으로 읽는다.

        Raises:
          ValueError: RAG_IMPL 이 real · mock 이 아니거나, QDRANT_MODE 가 local · cloud 가 아니거나,
            retrieval.yaml 값이 맞지 않을 때.
        """
        impl = os.environ.get("RAG_IMPL", "real")
        if impl not in IMPLS:
            raise ValueError(f"RAG_IMPL must be one of {IMPLS}, got {impl!r}")
        qdrant_mode = os.environ.get(QDRANT_MODE_ENV) or "local"
        if qdrant_mode not in QDRANT_MODES:
            raise ValueError(
                f"{QDRANT_MODE_ENV} must be one of {QDRANT_MODES}, got {qdrant_mode!r}"
            )
        config_dir = Path(os.environ.get("RAG_CONFIG_DIR", CONFIG_DIR))
        ingest = _read_yaml(config_dir / INGEST_CONFIG)
        retrieval = _read_yaml(config_dir / "retrieval.yaml")
        source = ingest["source"]["name"]
        model = ingest["embedding"]["model"]
        collection = os.environ.get("RAG_COLLECTION") or ingest["store"]["collection"].format(
            source=source, model=model
        )
        # HybridRetriever 는 검색기마다 같은 후보 수를 쓴다
        if retrieval["dense"]["top_k"] != retrieval["bm25"]["top_k"]:
            raise ValueError("retrieval.yaml: dense.top_k and bm25.top_k must be equal")
        if retrieval["fusion"]["top_k"] < MAX_K:
            raise ValueError(f"retrieval.yaml: fusion.top_k must be >= /search k limit {MAX_K}")
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
            impl=cast("Impl", impl),
            qdrant_mode=cast("QdrantMode", qdrant_mode),
        )


@dataclass(frozen=True)
class AnswerSetup:
    """설정 폴더에서 읽고 검사까지 마친 답변 경로 구성. 검색기 · LLM 을 붙이면 `AnswerService.from_config` 가 된다.

    검색기 · LLM 은 접속(PG · Ollama)을 `with` 로 여닫는 쪽이 들고 있어서 여기 넣지 않는다. 서버 · 평가 스크립트는
    접속 · 모델 로딩 전에 이것을 만들어, 설정이 틀렸으면 먼저 실패하고 실행 기록(meta)에 정책 · 관문을 적는다.

    예)
        setup = AnswerSetup.from_config(ApiConfig.from_env(), tokenizer=KiwiTokenizer(kiwi))
        service = AnswerService.from_config(setup, search=search_service, llm=llm)
    """

    policy: AnswerPolicy
    library: PromptLibrary
    """프로필이 가리키는 프롬프트를 모두 담은 라이브러리 (check_profiles 를 거쳤다)."""
    gate: QueryGate | None
    """질문 관문. answer.yaml 의 gate.enabled 가 꺼져 있거나 만들 때 끄면 None."""
    dump_date: str
    """덤프 날짜 ("2026년 10월 1일") — 프롬프트 · 관문 고정 답에 넣는다."""
    log_questions: bool
    """/ask 질문 원문을 로그에 남길지 (ADR-027)."""

    @classmethod
    def from_config(
        cls,
        config: ApiConfig,
        *,
        tokenizer: Tokenizer,
        prompts: Sequence[str] = (),
        gate: bool = True,
        policy_type: type[AnswerPolicy] = AnswerPolicy,
        log_questions: bool | None = None,
    ) -> Self:
        """config.config_dir 의 answer.yaml · ingest.yaml 과 저장소 프롬프트로 만든다.

        Args:
          config: 설정 폴더 · 질문 기록 설정을 읽는다.
          tokenizer: 관문의 형태소 분석기 — 검색(BM25)과 같은 Kiwi 를 넘긴다 (하나가 약 500MB). 관문을 끄면 쓰지 않는다.
          prompts: 단계 블록 프롬프트를 바꿔 다시 잴 때 `단계@버전` (`swap_prompt`). answer.yaml 은 그대로 둔다.
          gate: False 면 answer.yaml 과 상관없이 관문을 끈다 (모든 질문이 검색으로 — 유해 세트 forced 모드).
          policy_type: 답변 정책 클래스. 점수와 상관없이 partial 로 보내는 평가용 정책을 끼울 때.
          log_questions: 질문 원문 로그. None 이면 config 를 따르고, 평가 스크립트는 False 로 끈다.

        Raises:
          ValueError: 프로필 · 임계값이 틀렸거나, prompts 가 없는 단계를 가리키거나, 프롬프트 자리표시자가 맞지 않을 때.
          PromptError: 프로필이 없는 프롬프트를 가리킬 때.
          TypeError: gate.enabled 가 true · false 가 아닐 때 (gate 가 False 면 보지 않는다).
        """
        raw = _read_yaml(config.config_dir / ANSWER_CONFIG)
        loaded = _answer_policy(raw)
        policy = policy_type(loaded.thresholds, swap_prompt(loaded.profiles, *prompts))
        library = PromptLibrary.load(PROMPT_DIR)
        check_profiles(policy.profiles.values(), library)
        return cls(
            policy=policy,
            library=library,
            gate=_query_gate(raw, tokenizer) if gate else None,
            dump_date=load_dump_date(config.config_dir),
            log_questions=config.log_questions if log_questions is None else log_questions,
        )


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_dump_date(config_dir: Path) -> str:
    """ingest.yaml 의 덤프 날짜(YYYYMMDD)를 프롬프트 · 관문 고정 답에 넣을 꼴로 ("2026년 10월 1일")."""
    ingest = _read_yaml(config_dir / INGEST_CONFIG)
    return korean_date(datetime.strptime(str(ingest["source"]["dump_date"]), "%Y%m%d").date())


def load_query_gate(config_dir: Path, tokenizer: Tokenizer) -> QueryGate | None:
    """answer.yaml 의 gate.enabled 가 켜져 있으면 질문 관문을 만든다. 꺼져 있으면 None.

    분석기는 부르는 쪽이 넘긴다 — Kiwi 하나가 메모리를 약 500MB 써서, 검색(BM25)과 같은 인스턴스를 쓰게 한다.

    Raises:
      TypeError: gate.enabled 가 true · false 가 아닐 때 (따옴표 친 "yes" 같은 문자열).
    """
    return _query_gate(_read_yaml(config_dir / ANSWER_CONFIG), tokenizer)


def _query_gate(raw: Mapping[str, Any], tokenizer: Tokenizer) -> QueryGate | None:
    enabled = raw["gate"]["enabled"]
    if not isinstance(enabled, bool):
        raise TypeError(f"{ANSWER_CONFIG}: gate.enabled must be true or false, got {enabled!r}")
    return QueryGate(tokenizer) if enabled else None


def load_answer_policy(config_dir: Path) -> AnswerPolicy:
    """config_dir 의 answer.yaml(임계값 · 단계별 프로필)로 답변 정책을 만든다.

    Raises:
      ValueError: 프로필 값이 틀렸거나, 임계값을 아직 정하지 않았을 때(null).
    """
    return _answer_policy(_read_yaml(config_dir / ANSWER_CONFIG))


def _answer_policy(raw: Mapping[str, Any]) -> AnswerPolicy:
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
