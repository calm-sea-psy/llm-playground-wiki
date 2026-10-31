"""단발 대 에이전트 안전 회귀의 입력 (scripts/eval_agent.py 의 harmful · procedure · f4 세트).

- 유해 정보 · 무해 절차 세트: 에이전트는 회차 꼬리표로 새로 돌리고, 단발은 새로 돌리지 않고 이미 사람이 채점한
  `judge_partial.py answer --mode route` 기록(gate-v5 1~3 회차, 2026-11-01 기준선과 같은 구성)을 읽어 짝짓는다 — 기록
  형식(`AnswerRecord`)은 읽을 때 `AgentRecord` 로 맞춘다.
- F4 저장 문맥: 단발 · 에이전트 모두 새로 돌린다. 모든 검색 호출(원문 · 하위 질문 · 새 검색어)에 그 문항의 저장 문맥을
  돌려준다 (`StoredContextSearch`). 채점은 `prose_labels` 규칙을 그대로 쓴다 — 기록을 `AnswerRecord` 로 바꿔 넘긴다.
- 사람 검수표: 에이전트 유해 답(partial 로 끝난 답 전부 + grounded 로 끝난 유해 질문 답)을 회차를 섞어 가린 검수표.
- '답하지 않음' 판정 입력: 판정은 메인 세션이 Claude 하위 에이전트로 한다 (기준 reports/2026-10-31-refusal-rubric.md).

평가 쪽은 api · gate 를 import 하지 않는다 — route 이름 · 단계 이름은 여기 적고, 시험이 api 의 정의와 맞는지 본다.
"""

import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from playground_retrieval.search import SearchResult

from playground_wiki.evaluation.agent_eval import AgentItem, AgentRecord, SafetySet, SearchCall
from playground_wiki.evaluation.answer_run import AnswerRecord, EvalItem
from playground_wiki.evaluation.f4_prose import F4Item
from playground_wiki.evaluation.judge_review import ReviewEntry
from playground_wiki.evaluation.paths import (
    EXTRA_SET,
    KORQUAD_SET,
    LONGTAIL_STAGE1_SET,
    LONGTAIL_STAGE2_SET,
)
from playground_wiki.evaluation.prose_labels import (
    ProseLabels,
    RefusalVerdict,
    label_record,
    needs_judgment,
)

# 유해 정보 · 무해 절차 세트 — judge_partial.py 의 HARMFUL_SET · PROCEDURE_SET 와 같은 파일. 그 스크립트는 Phase 3 동안
# 고정이라 import 하지 않고 여기 적는다 (스크립트는 패키지가 아니기도 하다)
SAFETY_SET_PATHS: Mapping[str, Path] = MappingProxyType(
    {
        "harmful": Path("data/eval/harmful-v1.jsonl"),
        "procedure": Path("data/eval/harmless-procedure-v1.jsonl"),
    }
)
"""세트 이름 → 세트 파일 (한 줄: id · kind · question …). f4 는 LLM 레포 bench/data 의 세트를 읽는다."""
ROUNDS = (1, 2, 3)
"""유해 · 절차 세트의 회차 — 같은 구성으로 세 번 돌린다 (생성이 실행마다 갈려서, 기준선 회귀 판정 규칙)."""
F4_ROUND = 1
"""F4 는 한 회차 (기준선 회귀 판정 규칙 — 유해 세트 밖은 1 회차)."""
SINGLE_SAFETY_TAG = "route-gate-v5"
"""단발로 읽는 사람 채점된 기록 — data/eval/{세트}-answers-v1-route-gate-v5-{회차}.jsonl. 실행 이름은
`{세트}-route-gate-v5-{회차}` (judge_partial 검수표 열쇠의 run 과 같은 꼴)."""
SINGLE_REVIEW_SHEET = Path("data/eval/partial-judge-review-v1-harmful-route-gate-v5-123.md")
"""단발 유해 답(gate-v5 세 회차)의 사람 검수표 — 채운 사본(-checked)을 읽어 하드 기준 표에 넣는다."""
REVIEW_SEED = 0
HARMFUL_KIND = "harmful"
"""유해 정보 세트에서 해로운 답을 볼 문항의 kind (대조 질문은 contrast)."""
NUCLEAR_ID = "hz-20"
"""핵무기 질문 — 기준선 하드 기준이 따로 본다 (grounded 가 해로운 답을 냈던 문항, 2026-10-31)."""
ESCALATED_ROUTES = frozenset({"escalated_not_found", "escalated_partial", "gate_sub"})
"""에이전트 그래프를 돈 route (api.answer.AGENT_ROUTES 중). 나머지(gate · single_*)는 단발과 같은 길로 끝났다."""
SUB_GATE_ROUTE = "gate_sub"
"""하위 질문 · 새 검색어가 관문에 걸린 route."""
# F4 세트 → 고를 종류. eval_f4_prose.py 의 KINDS_BY_SET 과 같은 표 (그 스크립트는 고정이라 여기 적는다)
F4_KINDS_BY_SET: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        KORQUAD_SET: frozenset({"answerable", "no_doc"}),
        LONGTAIL_STAGE1_SET: frozenset({"no_fact"}),
        LONGTAIL_STAGE2_SET: frozenset({"no_fact"}),
        EXTRA_SET: frozenset(
            {"counterfactual", "benign_sensitive", "injection_indirect", "injection_direct"}
        ),
    }
)
# 민감 주제 문항 id 와 같은 KorQuAD 문항의 답 있음 id 앞머리 — eval_f4_prose.py 의 BENIGN_PREFIX · ANSWERABLE_PREFIX
BENIGN_PREFIX = "benign_sensitive:"
ANSWERABLE_PREFIX = "korquad:"
REFUSAL_JUDGE_KINDS = frozenset({"answerable", "benign_sensitive"})
"""'답하지 않음' 판정 입력을 만드는 기본 종류 — 하드 기준(민감 주제 과잉 거절)과 그 대조군."""
REFUSAL_JUDGE_PREFIX = "A"
"""판정 입력의 가린 번호 앞말 (F4 줄글의 R 과 겹치지 않게)."""
REFUSAL_JUDGE = "claude-subagent"
"""판정 결과에 남기는 판정한 쪽 이름 (F4 줄글 판정과 같은 이름)."""


@dataclass(frozen=True)
class ReviewPick:
    """검수표에 실을 답 하나 — 실행 이름과 기록."""

    run: str
    """기록 파일 이름 (예 harmful-agent-safety-1)."""
    record: AgentRecord


class StoredContextSearch:
    """문항마다 F4 저장 문맥(상위 5 청크 · 리랭커 점수)을 어떤 검색어에도 돌려준다.

    `f4_prose.StoredSearch` 와 같은 뜻이다 — 다른 것은 검색어가 질문과 달라도 막지 않는 것뿐이다 (에이전트의 하위 질문 ·
    새 검색어는 질문과 다르다). 그래서 에이전트가 무엇을 찾든 그 문항의 고친 청크(반사실 · 인젝션)만 근거로 받는다.

    예)
        search = StoredContextSearch()
        run = AgentEvalRun(RecordingSearch(search), answerer, before_answer=lambda item: search.use(items[item.id]))
    """

    def __init__(self) -> None:
        self._item: F4Item | None = None

    def use(self, item: F4Item) -> None:
        """다음 검색들에 돌려줄 문항을 고른다."""
        self._item = item

    async def search(self, query: str) -> SearchResult:
        """use() 로 고른 문항의 저장 문맥.

        Raises:
          RuntimeError: use() 를 부르지 않았을 때.
        """
        del query  # 원문 · 하위 질문 · 새 검색어 모두 같은 저장 문맥
        if self._item is None:
            raise RuntimeError("call use() with the item to answer before search")
        return self._item.search_result()


def safety_items(rows: Iterable[Mapping[str, Any]], set_name: SafetySet) -> list[AgentItem]:
    """유해 정보 · 무해 절차 세트 줄에서 문항을 돌려준다 (kind 에 harmful · contrast · procedure).

    세트 검사(짝 · 빈 질문)는 `judge_partial.py check` 가 한다. 질문은 그 스크립트처럼 앞뒤 공백을 지운다.
    """
    return [
        AgentItem(
            id=str(row["id"]),
            set_name=set_name,
            question=str(row["question"]).strip(),
            kind=str(row["kind"]),
        )
        for row in rows
    ]


def single_safety_path(set_name: str, round_number: int, *, root: Path = Path()) -> Path:
    """단발로 읽는 사람 채점된 기록의 경로."""
    return root / f"data/eval/{set_name}-answers-v1-{SINGLE_SAFETY_TAG}-{round_number}.jsonl"


def single_run(set_name: str, round_number: int) -> str:
    """단발 기록의 실행 이름 (검수표 열쇠의 run)."""
    return f"{set_name}-{SINGLE_SAFETY_TAG}-{round_number}"


def agent_tag(tag: str, round_number: int) -> str:
    """에이전트 기록의 회차 꼬리표 — `{tag}-{회차}` (eval_agent.py run --tag)."""
    return f"{tag}-{round_number}"


def agent_review_sheet(tag: str) -> Path:
    """에이전트 유해 답 검수표 경로 — 단발 검수표(…-route-gate-v5-123)와 같은 꼴."""
    return Path(f"data/eval/partial-judge-review-v1-harmful-agent-{tag}-123.md")


def from_answer_record(record: AnswerRecord, set_name: SafetySet) -> AgentRecord:
    """judge_partial 의 답 기록(`AnswerRecord`)을 단발 기록(`AgentRecord`)으로 맞춘다.

    관문이 받지 않았으면 질문 그대로 검색 한 번이다 (단발 /ask). 답변 근거는 검색 결과의 앞쪽 context_count 개다.
    """
    context = record.hits[: record.context_count]
    searched = record.gate_rule is None and record.stage_path[:1] != ("gate",)
    searches = (
        (SearchCall(record.item.question, tuple(hit.title for hit in record.hits)),)
        if searched
        else ()
    )
    return AgentRecord(
        item=AgentItem(
            id=record.item.id,
            set_name=set_name,
            question=record.item.question,
            kind=record.item.category,
        ),
        mode="single",
        stage_path=record.stage_path,
        answer=record.answer,
        source_titles=tuple(chunk.title for chunk in context),
        searches=searches,
        error=record.error,
        first_delta_ms=record.first_delta_ms,
        total_ms=record.total_ms,
        wall_ms=record.wall_ms,
        gate_rule=record.gate_rule,
        top_score=record.top_score,
        marker=record.marker,
        context=context,
    )


def to_answer_record(record: AgentRecord, item: EvalItem) -> AnswerRecord:
    """F4 채점(`prose_labels.label_record`)에 넘길 꼴로 — 답변 단계 경로 · 답 · 답변 근거 · 표지만 옮긴다.

    에이전트 노드 단계(agent_*)는 빼고, 답변 근거(context)를 검색 결과 자리에 둔다 — 간접 인젝션의 '지시가 문맥에
    들었나'는 답을 쓴 단계가 받은 근거로 본다.
    """
    return AnswerRecord(
        item=item,
        stage_path=record.answer_path,
        answer=record.answer,
        hits=record.context,
        context_count=len(record.context),
        marker=record.marker,
        cited=(),
        dropped=(),
        truncated=False,
        timed_out=False,
        error=record.error,
        top_score=record.top_score,
        gate_rule=record.gate_rule,
        spec_id=None,
        prompt_version=None,
        search_ms=None,
        first_delta_ms=record.first_delta_ms,
        total_ms=record.total_ms,
        wall_ms=record.wall_ms,
    )


def f4_safety_items(items: Sequence[F4Item]) -> list[F4Item]:
    """F4 문항에서 답 있음 중 민감 주제와 같은 입력(쌍둥이 50)을 뺀 것 — 대조군 250 (eval_f4_prose 리포트와 같은 대조군)."""
    twins = {
        item.id.removeprefix(BENIGN_PREFIX) for item in items if item.kind == "benign_sensitive"
    }
    return [
        item
        for item in items
        if not (item.kind == "answerable" and item.id.removeprefix(ANSWERABLE_PREFIX) in twins)
    ]


def f4_agent_item(item: F4Item) -> AgentItem:
    """F4 문항을 측정 문항으로 (kind 에 F4 종류, answers 에 정답 표기)."""
    return AgentItem(
        id=item.id, set_name="f4", question=item.question, answers=item.answers, kind=item.kind
    )


def f4_labels(
    records: Iterable[AgentRecord], items: Mapping[str, F4Item]
) -> dict[str, ProseLabels]:
    """F4 기록마다 줄글 규칙 채점 — 문항 id → 채점. 세트에 없는 문항 기록은 뺀다."""
    labels = {}
    for record in records:
        item = items.get(record.item.id)
        if item is not None:
            labels[item.id] = label_record(to_answer_record(record, item.eval_item()), item)
    return labels


def escalated(record: AgentRecord) -> bool:
    """에이전트 그래프를 돈 기록인가 (route 가 ESCALATED_ROUTES)."""
    return record.route in ESCALATED_ROUTES


def review_picks(records_by_run: Mapping[str, Sequence[AgentRecord]]) -> list[ReviewPick]:
    """검수표에 실을 답 — 실행마다 partial 로 끝난 답 전부 + grounded 로 끝난 유해 질문 답. 실행을 섞어 seed 로 섞는다.

    Raises:
      ValueError: 실을 답의 답변 근거 본문(context)이 비었을 때 — context 를 남기기 전 기록이다.
    """
    picks = []
    for run, records in records_by_run.items():
        for record in records:
            if not review_target(record):
                continue
            if not record.context:
                raise ValueError(f"{run} {record.item.id}: no answer context in the record")
            picks.append(ReviewPick(run, record))
    random.Random(REVIEW_SEED).shuffle(picks)
    return picks


def review_target(record: AgentRecord) -> bool:
    """사람이 해로움을 채점할 답인가 — partial 로 끝난 답, grounded 로 끝난 유해 질문 답 (단발 gate-v5 검수표와 같은 몫)."""
    stage = record.final_stage
    return stage == "partial" or (stage == "grounded" and record.item.kind == HARMFUL_KIND)


def review_entries(picks: Sequence[ReviewPick]) -> list[ReviewEntry]:
    """검수표 문항 (번호 1부터) — 질문 · 답변 근거 · 답만. 실행 · route · 단계는 열쇠에만."""
    return [
        ReviewEntry(
            number=number,
            question=pick.record.item.question,
            context=pick.record.context,
            answer=pick.record.answer,
        )
        for number, pick in enumerate(picks, start=1)
    ]


def review_key_entries(picks: Sequence[ReviewPick]) -> list[dict[str, Any]]:
    """열쇠 entries — 검수표 번호 → 실행 · 문항 id · 끝난 단계 · route (`partial_review.KeyEntry` 가 number · run · id 를 읽는다)."""
    return [
        {
            "number": number,
            "run": pick.run,
            "id": pick.record.item.id,
            "stage": pick.record.final_stage,
            "route": pick.record.route,
        }
        for number, pick in enumerate(picks, start=1)
    ]


def refusal_judge_rows(
    labeled: Iterable[tuple[str, AgentRecord, ProseLabels]],
    *,
    kinds: frozenset[str] = REFUSAL_JUDGE_KINDS,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """'답하지 않음' 판정 입력과 열쇠 — 판정할 답(needs_judgment)만, 실행을 섞어 seed 로 섞고 가린 번호를 붙인다.

    Args:
      labeled: (실행 이름, 기록, 그 기록의 채점).
      kinds: 판정할 F4 종류.

    Returns:
      (입력 — judge_id · question · answer, 열쇠 — judge_id · run · id · kind). 판정 입력에는 실행 · 종류 · 정답 · 문서를
      싣지 않는다 (refusal-rubric 과 같은 가림).
    """
    targets = [
        (run, record, label)
        for run, record, label in labeled
        if label.kind in kinds and needs_judgment(label)
    ]
    random.Random(REVIEW_SEED).shuffle(targets)
    inputs, keys = [], []
    width = max(3, len(str(len(targets))))
    for number, (run, record, label) in enumerate(targets, start=1):
        judge_id = f"{REFUSAL_JUDGE_PREFIX}{number:0{width}d}"
        inputs.append(
            {"judge_id": judge_id, "question": record.item.question, "answer": record.answer}
        )
        keys.append({"judge_id": judge_id, "run": run, "id": label.id, "kind": label.kind})
    return inputs, keys


def refusal_verdicts(
    judged: Iterable[Mapping[str, Any]], keys: Iterable[Mapping[str, Any]]
) -> list[RefusalVerdict]:
    """판정자 출력(judge_id · refused · reason)을 열쇠로 풀어 판정 결과로 (variant 에 실행 이름).

    Raises:
      ValueError: 열쇠에 없는 judge_id 가 있거나, 같은 judge_id 가 두 번 나올 때.
    """
    by_id = {str(key["judge_id"]): key for key in keys}
    verdicts = []
    seen: set[str] = set()
    for row in judged:
        judge_id = str(row["judge_id"])
        if judge_id not in by_id or judge_id in seen:
            raise ValueError(f"judge_id {judge_id!r} is unknown or repeated")
        seen.add(judge_id)
        key = by_id[judge_id]
        verdicts.append(
            RefusalVerdict(
                id=str(key["id"]),
                variant=str(key["run"]),
                refused=bool(row["refused"]),
                reason=str(row.get("reason", "")),
                judge=REFUSAL_JUDGE,
            )
        )
    return verdicts
