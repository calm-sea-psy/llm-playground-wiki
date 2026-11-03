"""partial 판정의 입력 — 세트 문항 · 기록 경로 · 판정 행 읽기."""

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType

from playground_wiki.evaluation import clopper_pearson
from playground_wiki.evaluation.answer_metrics import Share, final_stage
from playground_wiki.evaluation.answer_run import (
    TAG_PATTERN,
    AnswerRecord,
    EvalItem,
    read_records,
    tagged,
)
from playground_wiki.evaluation.files import read_jsonl
from playground_wiki.evaluation.judge_review import Agreement
from playground_wiki.evaluation.partial_axis import HARMFUL_KINDS
from playground_wiki.evaluation.partial_judge import JudgedRow, PartialVerdict
from playground_wiki.evaluation.partial_review import (
    ReviewSheet,
    agreements_for,
    has_verdict,
    read_checked,
)
from playground_wiki.evaluation.paths import (
    F9_ANSWERS_PATH,
    HARMFUL_ROUTE_ANSWERS_PATH,
    HARMFUL_ROUTE_REVIEW_PATH,
    HARMFUL_SET_PATH as HARMFUL_SET,
    PROCEDURE_ROUTE_ANSWERS_PATH,
    PROCEDURE_SET_PATH as PROCEDURE_SET,
)

# 무해 절차 세트는 partial 안전 규칙이 '하는 법' 질문까지 막는지 본다 (유해 세트의 대조 질문은 유해 인접 주제뿐이라 이
# 회귀를 잡지 못한다). 짝이 없고 kind 는 procedure 하나다
SETS: Mapping[str, Path] = MappingProxyType({"harmful": HARMFUL_SET, "procedure": PROCEDURE_SET})
MODES = ("route", "forced")
RUNS: Mapping[str, Path] = MappingProxyType(
    {
        "f9": F9_ANSWERS_PATH,
        "harmful-route": HARMFUL_ROUTE_ANSWERS_PATH,
        "harmful-forced": Path("data/eval/harmful-answers-v1-forced.jsonl"),
    }
)
"""판정할 답 기록 — 실행 이름 → 기록 파일. 없는 파일은 건너뛴다."""
TAGGED_RUNS: Mapping[str, Path] = MappingProxyType(
    {
        "f9": F9_ANSWERS_PATH,
        "harmful-route": HARMFUL_ROUTE_ANSWERS_PATH,
        "procedure-route": PROCEDURE_ROUTE_ANSWERS_PATH,
    }
)
"""judge --tag 가 판정할 수 있는 기록 — 실행 이름 → 꼬리표 붙이기 전 기록 파일 (`answer_path` · eval_answer run 과 같은
이름). 꼬리표 기록의 판정은 PARTIAL_JUDGED_TAGGED_PATH 에 따로 쓴다 — 기본 judge · report 는 RUNS 만 본다."""
JUDGED_PATH = Path("data/eval/partial-judged-v1.jsonl")
REVIEW_PATHS: Mapping[str, Path] = MappingProxyType(
    {
        "f9": Path("data/eval/partial-judge-review-v1-f9.md"),
        "harmful": Path("data/eval/partial-judge-review-v1-harmful.md"),
        "harmful-route": HARMFUL_ROUTE_REVIEW_PATH,
    }
)
REVIEW_F9_SIZE = 30
REVIEW_HARMFUL_PAIRS = 10  # 유해 10 + 대조 10 — F9 30 과 합쳐 사람 채점 50
REVIEW_SOURCE_RUN = MappingProxyType(
    {"f9": "f9", "harmful": "harmful-forced", "harmful-route": "harmful-route"}
)
"""검수표 몫 → 표본을 뽑는 실행. 유해 정보는 문항마다 partial 답이 있는 forced 에서,
harmful-route 는 운영 경로에서 partial 로 끝난 답 전부에서 (결정 지표는 route 라 전 · 후를 여기서 견준다)."""
ROUND = re.compile(r"(?P<setting>.+)-(?P<round>\d+)")
"""꼬리표 기록 이름 끝의 회차 — `harmful-route-safety-v4-2` 는 설정 `harmful-route-safety-v4` 를 두 번째로 돌린 기록.
같은 설정을 여러 번 돌린 기록을 사람 채점 표에서 한 줄로 묶는다."""
SEED = 0
REVIEW_SHEETS: Mapping[str, ReviewSheet] = MappingProxyType(
    {
        "f9": ReviewSheet(
            REVIEW_PATHS["f9"],
            "F9 partial 답에서 묶음(KorQuAD · 위키 밖 partial 기대 · 위키 밖 refuse 기대) 크기에 비례해 "
            f"{REVIEW_F9_SIZE}",
        ),
        "harmful": ReviewSheet(
            REVIEW_PATHS["harmful"],
            f"forced 에서 유해 · 대조 둘 다 partial 답이 있는 짝 {REVIEW_HARMFUL_PAIRS}개 (유해 + 대조)",
        ),
        "harmful-v2": ReviewSheet(
            Path("data/eval/partial-judge-review-v2-harmful.md"),
            "forced 유해 답 중 harmful 검수표에 없던 것 전부 + route 에서 partial 로 끝난 유해 답 전부 — 두 실행을 "
            "섞어 어느 실행의 답인지 가렸다",
        ),
        "harmful-route-partial-v3-1": ReviewSheet(
            tagged(REVIEW_PATHS["harmful-route"], "partial-v3-1"),
            "route 에서 partial-v2-1 · partial-v3-1 두 기록 모두 partial 로 끝난 문항의 답 전부 — 두 기록을 섞어 "
            "어느 쪽 답인지 가렸다",
        ),
        "harmful-route-safety-v4-1": ReviewSheet(
            tagged(REVIEW_PATHS["harmful-route"], "safety-v4-1"),
            "route 에서 partial-v3-1 · safety-v4-1 두 기록 모두 partial 로 끝난 문항의 답 전부 — 두 기록을 섞어 "
            "어느 쪽 답인지 가렸다",
        ),
        "hz-20-safety-v4": ReviewSheet(
            Path("data/eval/partial-judge-review-v1-hz-20-safety-v4.md"),
            "핵무기 질문(hz-20) 하나의 partial 답 9개 — route safety-v4 · forced safety-v4 · forced partial-v3 를 "
            "회차 1~3 씩 섞어 가렸다 (review-sheet 밖 스크립트로 만들었다)",
        ),
    }
)
"""리포트 · 일치율 · judge --review-only 가 읽는 사람 검수표 — 이름(리포트의 '검수표' 칸) → 검수표. 표본은 모두 판정
결과와 상관없이 seed 로 뽑고 섞었다. 새 검수표(예: partial@v3 전 · 후의 `review-sheet --tag`)는 여기에 한 줄 더한다.
채운 사본이 없는 것은 아직 안 채운 것이라 건너뛴다. 열쇠 entries 는 `{number, run, id}` 만 읽는다 (base_run 은 없어도 된다)."""


def set_items(set_name: str, rows: Sequence[Mapping[str, object]]) -> list[EvalItem]:
    """세트 이름에 맞는 검사 · 변환. 유해 세트는 짝을, 무해 절차 세트는 kind 만 본다."""
    return harmful_items(rows) if set_name == "harmful" else procedure_items(rows)


def procedure_items(rows: Sequence[Mapping[str, object]]) -> list[EvalItem]:
    """무해 절차 세트를 검사하고 평가 문항으로. 짝이 없어 유해 세트보다 검사가 가볍다.

    Raises:
      ValueError: 줄마다의 문제를 모아서 — id 중복 · kind 가 procedure 가 아님 · 빈 질문.
    """
    problems: list[str] = []
    by_id: dict[str, Mapping[str, object]] = {}
    for row in rows:
        row_id = str(row.get("id") or "")
        if not row_id or row_id in by_id:
            problems.append(f"{row_id or '(id 없음)'}: id 가 비었거나 겹친다")
        if row.get("kind") != "procedure":
            problems.append(f"{row_id}: kind 는 'procedure', got {row.get('kind')!r}")
        if not str(row.get("question") or "").strip():
            problems.append(f"{row_id}: question 이 비었다")
        by_id[row_id] = row
    if problems:
        raise ValueError("무해 절차 세트 문제:\n" + "\n".join(problems))
    return [
        EvalItem(
            id=row_id,
            source="procedure",
            question=str(row["question"]).strip(),
            category="procedure",
        )
        for row_id, row in by_id.items()
    ]


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


def answer_path(set_name: str, mode: str, tag: str | None) -> Path:
    """Answer 의 기록 파일 — 꼬리표가 있으면 이름 끝에 -{tag} (기준 기록을 덮어쓰지 않는다)."""
    run = RUNS.get(f"{set_name}-{mode}")
    base = run if run is not None else Path(f"data/eval/{set_name}-answers-v1-{mode}.jsonl")
    return tagged(base, tag)


def tagged_targets(run: str, tag: str) -> list[tuple[str, AnswerRecord]]:
    """꼬리표 기록 `run-tag` 에서 partial 로 끝나 LLM 이 답을 쓴 기록 — (실행 이름, 기록).

    Raises:
      SystemExit: 기록 파일이 없을 때 (answer · eval_answer run --tag 를 먼저 돌린다).
    """
    path = tagged(TAGGED_RUNS[run], tag)
    if not path.exists():
        raise SystemExit(f"{path} 가 없다 — 그 꼬리표로 답 기록을 먼저 만든다")
    name = f"{run}-{tag}"
    return [(name, record) for record in read_records(path) if final_stage(record) == "partial"]


def load_agreements(
    root: Path = Path(), *, sheets: Mapping[str, ReviewSheet] = REVIEW_SHEETS
) -> dict[str, dict[str, Agreement]]:
    """채운 검수표(`-checked` 사본, sheets 기본 REVIEW_SHEETS)마다 항목 → 일치 표.

    사본이 없으면 아직 안 채운 것이라, 실린 답에 판정기 판정이 하나도 없으면(꼬리표 기록 — 판정기를 돌리지 않았다)
    견줄 것이 없어 그 몫을 뺀다. 판정이 없는 검수표도 사람 채점(human_labels)에는 들어간다.
    """
    judged = judged_by_key(root)
    result: dict[str, dict[str, Agreement]] = {}
    for name, sheet in sheets.items():
        read = read_checked(root, sheet.path)
        if read is not None and has_verdict(read[1], judged):
            result[name] = agreements_for(*read, judged)
    return result


def read_verdict(
    judged: Mapping[tuple[str, str], JudgedRow], run: str, record: AnswerRecord
) -> PartialVerdict | None:
    """기록의 판정기 판정. 판정 행이 없으면 None."""
    row = judged.get((run, record.item.id))
    return row.verdict if row is not None else None


def partial_records(run: str, tag: str | None = None) -> list[AnswerRecord]:
    """그 실행에서 partial 로 끝나 LLM 이 답을 쓴 기록. 기록 파일이 없으면 빈 목록."""
    return [record for record in all_records(run, tag) if final_stage(record) == "partial"]


def all_records(run: str, tag: str | None = None, *, root: Path = Path()) -> list[AnswerRecord]:
    """실행(꼬리표 포함)의 답 기록 전부."""
    path = root / tagged(RUNS[run], tag)
    return read_records(path) if path.exists() else []


def run_path(run: str) -> Path | None:
    """기록 이름 → 기록 파일. RUNS 이름이거나 `RUNS 이름-꼬리표`(answer --tag) 꼴이 아니면 None."""
    if run in RUNS:
        return RUNS[run]
    for base, path in RUNS.items():
        tag = run.removeprefix(f"{base}-")
        if tag != run and TAG_PATTERN.fullmatch(tag):
            return tagged(path, tag)
    return None


def read_judged(root: Path, judged_path: Path = JUDGED_PATH) -> list[JudgedRow]:
    """판정 행 파일을 읽는다. 파일이 없으면 빈 목록."""
    path = root / judged_path
    return [JudgedRow.from_json(row) for row in read_jsonl(path)] if path.exists() else []


def judged_by_key(root: Path = Path()) -> dict[tuple[str, str], JudgedRow]:
    """(실행 이름, 문항 id) → 판정 행."""
    return {(row.run, row.id): row for row in read_judged(root)}


def count_where(value: Share) -> str:
    """비율의 분자/분모 글."""
    return f"{value.count}/{value.total}"


def interval_text(value: Share) -> str:
    """비율의 신뢰구간 글."""
    if not value.total:
        return ""
    low, high = clopper_pearson(value.count, value.total)
    return f"{low:.1%}~{high:.1%}"


def named_runs_of(item_id: str, topics: Mapping[str, str]) -> str:
    """문항 id 에 주제 이름을 붙인 글 — 질문 원문은 싣지 않는다."""
    # 주제 이름만 붙인다 — 질문 원문은 싣지 않는다
    topic = topics.get(item_id)
    return f"{item_id} ({topic})" if topic else item_id
