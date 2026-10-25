"""F4 LLM 벤치 오염 판정에 쓸 패러프레이즈 세트를 만든다 — GPU · 검색 서버 · API 없이 저장된 파일만 읽는다.

입력은 세 파일이다.
  패러프레이즈  scripts/paraphrase_korquad.py 가 Claude 로 고쳐 쓴 질문 (위키 레포 data/bench/paraphrase-v1.jsonl)
  사람 검수표   그 결과에서 표본 50개를 사람이 확인한 markdown (data/bench/paraphrase-v1-review-checked.md)
  KorQuAD 세트  scripts/build_bench_set.py 가 만든 1단계 세트 (LLM 레포 bench/data/{세트}/items.jsonl)

F4 표본(sample "f4-answerable")의 패러프레이즈마다 같은 id 의 answerable 문항을 그대로 복사하고 question 만
패러프레이즈로 바꾼다. kind 는 "paraphrase", id 는 paraphrase:{KorQuAD id}, source_id 는 원래 문항 id(짝지어 비교용),
original_question 은 원문 질문이다. context 는 원문 문항의 상위 5 청크 그대로다 — 패러프레이즈로 다시 검색하지 않는다.
다시 검색하면 검색 손실이 섞여 "문맥이 같을 때 생성이 표현 변화에 얼마나 약한가"를 따로 잴 수 없다.
짝이 없거나 자동 점검(problems)에 걸린 패러프레이즈는 빼고 이유별 수를 meta.json 에 적는다.

검수표 비고에 "N번: …" 으로 적힌 문항(정답은 그대로지만 표현이 달라진 것)은 빼지 않고 review_note 로 비고 글을 단다 —
오염 판정 때 그 문항을 뺀 값도 같이 볼 수 있게. 검수 요약(확인 수 · 정답 바뀜 · 정답 샘 · 표현 달라짐)은 meta.json 에.
검수표의 패러프레이즈가 지금 패러프레이즈 파일과 다르면 옛 검수가 붙지 않게 멈춘다.

오염 판정은 원문 · 패러프레이즈의 문맥 없는(norag) 점수를 source_id 로 짝지어 비교하고, 차이가 크면 같은 문맥으로(rag)
패러프레이즈를 다시 돌려 외움과 표현 민감을 가른다. 채점 코드는 bench/f4_bench.py 가 붙인다.
세트 파일은 KorQuAD 가 CC BY-ND 라 git 제외 폴더(LLM 레포 bench/data/)에만 쓴다.

사용법 (위키 레포 루트에서):
  uv run python scripts/build_bench_paraphrase.py
worktree 등 루트가 아닌 곳에서는 세 입력의 경로를 절대 경로로 넘긴다 (한 줄로):
  uv run python scripts/build_bench_paraphrase.py
    --paraphrase C:/Work/llm-playground-wiki/data/bench/paraphrase-v1.jsonl
    --review C:/Work/llm-playground-wiki/data/bench/paraphrase-v1-review-checked.md
    --repo C:/Work/llm-playground-llm/bench/data
"""

import argparse
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, get_args

# 기본값은 위키 레포 루트에서 돌릴 때 기준 — worktree 등 다른 곳에서는 --paraphrase · --review · --repo 로 절대 경로를 넘긴다.
PARAPHRASE_FILE = Path("data/bench/paraphrase-v1.jsonl")  # scripts/paraphrase_korquad.py 결과
REVIEW_FILE = Path(
    "data/bench/paraphrase-v1-review-checked.md"
)  # 표본 50개(F4 25 · F9 25) 사람 검수
SOURCE_REPO = Path("../llm-playground-llm/bench/data")  # LLM 레포의 세트 폴더 (git 제외)
SOURCE_SET = "f4-korquad-9014dd9e"  # build_bench_set.py 가 만든 1단계 KorQuAD 세트
OUT_SET = "f4-paraphrase-v1"
SAMPLE = "f4-answerable"  # 패러프레이즈 파일에서 F4 세트로 만든 표본. 나머지(f9-random)는 F9 용이라 쓰지 않는다
SOURCE_KIND = "answerable"
KIND = "paraphrase"

Row = dict[str, Any]  # 세트 jsonl 한 줄 — 모양을 그대로 복사해 넘기므로 dataclass 로 바꾸지 않는다

# 세트에서 빼는 이유
#   problems          자동 점검(빈 답 · 원문과 같음 · 정답 샘)에 걸림
#   missing           같은 id 문항이 KorQuAD 세트에 없음
#   not_answerable    같은 id 문항이 answerable 이 아님
#   question_mismatch 세트의 질문이 패러프레이즈의 원문과 다름 (세트를 다시 만들어 짝이 어긋남)
ExcludeReason = Literal["problems", "missing", "not_answerable", "question_mismatch"]
EXCLUDE_REASONS: tuple[str, ...] = get_args(ExcludeReason)

# 검수표 모양: 머리 줄의 칸 이름으로 칸을 찾고, O / X 칸과 비고의 "- N번: …" 줄만 읽는다
_NUMBER_COLUMN = "번호"
_MARKS = {"O": True, "X": False}
_NOTE = re.compile(r"^- (\d+)번: (.+)$")


@dataclass(frozen=True)
class Paraphrase:
    """패러프레이즈 파일 한 줄 중 세트를 만드는 데 쓰는 필드."""

    sample: str
    id: str
    question: str
    """고쳐 쓰기 전 원문 질문."""
    paraphrase: str
    model: str
    prompt_version: str
    problems: tuple[str, ...]
    """자동 점검에 걸린 이유. 비어 있어야 세트에 넣는다."""


@dataclass(frozen=True)
class ReviewRow:
    """사람 검수표 한 줄."""

    number: int
    sample: str
    id: str
    paraphrase: str
    """검수한 패러프레이즈 글. 지금 패러프레이즈 파일의 글과 같아야 한다."""
    answer_same: bool
    """'정답 같음' 칸이 O 인가."""
    no_leak: bool
    """'정답 안 샘' 칸이 O 인가."""


@dataclass(frozen=True)
class Review:
    """사람 검수 결과 — 검수표의 줄과 비고."""

    rows: tuple[ReviewRow, ...]
    notes: Mapping[str, str]
    """비고에 적힌 문항의 패러프레이즈 id → 비고 글. 정답은 그대로지만 표현이 달라진 것."""


@dataclass(frozen=True)
class Exclusion:
    """세트에서 뺀 패러프레이즈 한 건."""

    id: str
    reason: ExcludeReason


@dataclass(frozen=True)
class PairedSet:
    """패러프레이즈와 원문 문항을 짝지은 결과."""

    rows: tuple[Row, ...]
    """세트에 들어갈 줄. 패러프레이즈 파일 순서."""
    used: tuple[Paraphrase, ...]
    """rows 를 만든 패러프레이즈. rows 와 같은 순서."""
    excluded: tuple[Exclusion, ...]


def main() -> None:
    args = _parse_args()
    build(args.paraphrase, args.review, args.repo, args.source, args.out)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--paraphrase",
        type=Path,
        default=PARAPHRASE_FILE,
        help="패러프레이즈 jsonl 경로 (위키 레포 루트가 아닌 곳에서 돌릴 때 절대 경로로)",
    )
    parser.add_argument(
        "--review",
        type=Path,
        default=REVIEW_FILE,
        help="사람 검수표 markdown 경로 (위키 레포 루트가 아닌 곳에서 돌릴 때 절대 경로로)",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=SOURCE_REPO,
        help="LLM 레포의 bench/data 경로 (위키 레포 루트가 아닌 곳에서 돌릴 때 절대 경로로)",
    )
    parser.add_argument("--source", default=SOURCE_SET, help="입력 KorQuAD 세트 폴더 이름")
    parser.add_argument("--out", default=OUT_SET, help="만들 패러프레이즈 세트 폴더 이름")
    return parser.parse_args()


def build(paraphrase_path: Path, review_path: Path, repo: Path, source: str, out: str) -> None:
    """패러프레이즈 세트를 repo/out 에 items.jsonl · meta.json 으로 쓰고 건수를 출력한다."""
    source_path = repo / source / "items.jsonl"
    paraphrases = [_parse_paraphrase(line) for line in _read_jsonl(paraphrase_path)]
    review = read_review(review_path)
    _check_review(review, paraphrases)
    f4 = [paraphrase for paraphrase in paraphrases if paraphrase.sample == SAMPLE]
    paired = pair(f4, _read_jsonl(source_path), review.notes)

    out_dir = repo / out
    _write_jsonl(out_dir / "items.jsonl", paired.rows)
    excluded = Counter(exclusion.reason for exclusion in paired.excluded)
    meta = {
        "version": out,
        "created_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "paraphrase_file": paraphrase_path.resolve().as_posix(),
        "source_set": source,
        "source_items": source_path.resolve().as_posix(),
        "sample": SAMPLE,
        "paraphrase_models": _count(paraphrase.model for paraphrase in paired.used),
        "prompt_versions": _count(paraphrase.prompt_version for paraphrase in paired.used),
        "counts": {
            KIND: len(paired.rows),
            "f4_paraphrases": len(f4),
            "excluded": {reason: excluded[reason] for reason in EXCLUDE_REASONS},
            "other_samples": _count(
                paraphrase.sample for paraphrase in paraphrases if paraphrase.sample != SAMPLE
            ),
        },
        "excluded": [{"id": item.id, "reason": item.reason} for item in paired.excluded],
        "human_review": _review_summary(review, review_path, paired),
        "note": (
            "context 는 원문 문항의 상위 5 청크 그대로 (패러프레이즈로 다시 검색하지 않음). "
            "source_id 로 원문 문항과 짝지어 비교한다. review_note 가 있는 문항은 정답은 그대로지만 "
            "사람 검수에서 표현이 달라졌다고 본 것 — 뺀 값도 같이 본다. 채점 코드는 bench/f4_bench.py 가 붙인다"
        ),
    }
    _write_json(out_dir / "meta.json", meta)
    _report(out_dir, meta)


def pair(
    paraphrases: Sequence[Paraphrase], items: Sequence[Row], review_notes: Mapping[str, str]
) -> PairedSet:
    """패러프레이즈마다 같은 id 의 answerable 문항을 찾아 패러프레이즈 문항으로 바꾼다. 짝이 안 맞으면 뺀다.

    Args:
      paraphrases: 세트로 만들 패러프레이즈. id 가 겹치면 안 된다.
      items: KorQuAD 세트의 문항 줄.
      review_notes: 패러프레이즈 id → 사람 검수 비고. 있는 문항에만 review_note 로 단다.
    """
    id_counts = Counter(paraphrase.id for paraphrase in paraphrases)
    duplicated = sorted(item_id for item_id, count in id_counts.items() if count > 1)
    if duplicated:
        raise ValueError(f"duplicate paraphrase ids: {duplicated[:5]} ({len(duplicated)} ids)")
    by_id = {item["id"]: item for item in items}
    rows: list[Row] = []
    used: list[Paraphrase] = []
    excluded: list[Exclusion] = []
    for paraphrase in paraphrases:
        source = by_id.get(paraphrase.id)
        reason = _exclude_reason(paraphrase, source)
        if reason is not None:
            excluded.append(Exclusion(paraphrase.id, reason))
            continue
        review_note = review_notes.get(paraphrase.id)
        rows.append(paraphrase_row(by_id[paraphrase.id], paraphrase, review_note=review_note))
        used.append(paraphrase)
    return PairedSet(tuple(rows), tuple(used), tuple(excluded))


def paraphrase_row(source: Row, paraphrase: Paraphrase, *, review_note: str | None = None) -> Row:
    """원문 문항을 그대로 복사하고 질문만 패러프레이즈로 바꾼 줄. context 는 원문의 상위 5 청크 그대로.

    Args:
      source: KorQuAD 세트의 answerable 문항 줄.
      paraphrase: 그 문항의 패러프레이즈.
      review_note: 사람 검수 비고. None 이면 review_note 필드를 넣지 않는다.
    """
    korquad_id = source["id"].split(":", 1)[1]
    row = {
        **source,
        "id": f"{KIND}:{korquad_id}",
        "kind": KIND,
        "question": paraphrase.paraphrase,
        "answers": list(source["answers"]),
        "context": [dict(chunk) for chunk in source["context"]],
        "source_id": source["id"],
        "original_question": source["question"],
    }
    if review_note is not None:
        row["review_note"] = review_note
    return row


def read_review(path: Path) -> Review:
    """사람 검수표(markdown)를 읽는다. 표의 O / X 칸과 비고의 "- N번: …" 줄만 본다.

    Raises:
      ValueError: 표 칸 수가 머리 줄과 다르거나, O / X 칸이 비었거나, 비고 번호가 표에 없을 때.
    """
    header: list[str] = []
    rows: list[ReviewRow] = []
    note_by_number: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("|"):
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if not header and cells[0] == _NUMBER_COLUMN:
                header = cells
            elif cells[0].isdigit():
                rows.append(_review_row(header, cells))
        elif match := _NOTE.match(line):
            note_by_number[int(match[1])] = match[2].strip()
    id_by_number = {row.number: row.id for row in rows}
    if unknown := sorted(set(note_by_number) - set(id_by_number)):
        raise ValueError(f"review notes refer to rows not in the table: {unknown}")
    notes = {id_by_number[number]: note for number, note in sorted(note_by_number.items())}
    return Review(tuple(rows), MappingProxyType(notes))


def _review_row(header: Sequence[str], cells: Sequence[str]) -> ReviewRow:
    if len(cells) != len(header):
        raise ValueError(f"review row {cells[0]} has {len(cells)} cells, header has {len(header)}")
    column = dict(zip(header, cells, strict=True))
    return ReviewRow(
        number=int(column[_NUMBER_COLUMN]),
        sample=column["표본"],
        id=column["id"],
        paraphrase=column["패러프레이즈"],
        answer_same=_mark(column, "정답 같음"),
        no_leak=_mark(column, "정답 안 샘"),
    )


def _mark(column: Mapping[str, str], name: str) -> bool:
    mark = column[name]
    if mark not in _MARKS:
        number = column[_NUMBER_COLUMN]
        raise ValueError(f"review row {number} {name!r} must be O or X, got {mark!r}")
    return _MARKS[mark]


def _check_review(review: Review, paraphrases: Sequence[Paraphrase]) -> None:
    """검수표의 패러프레이즈가 지금 파일의 글과 같은지 본다 — 다시 만든 파일에 옛 검수가 붙지 않게."""
    text_by_key = {
        (paraphrase.sample, paraphrase.id): paraphrase.paraphrase for paraphrase in paraphrases
    }
    stale = [
        row.number for row in review.rows if text_by_key.get((row.sample, row.id)) != row.paraphrase
    ]
    if stale:
        raise ValueError(
            f"review rows do not match the paraphrase file: {stale[:5]} ({len(stale)} rows)"
        )


def _review_summary(review: Review, path: Path, paired: PairedSet) -> dict[str, object]:
    used_ids = {paraphrase.id for paraphrase in paired.used}
    noted = [row["id"] for row in paired.rows if "review_note" in row]
    return {
        "file": path.resolve().as_posix(),
        "checked": len(review.rows),
        "checked_in_set": sum(
            1 for row in review.rows if row.sample == SAMPLE and row.id in used_ids
        ),
        "answer_changed": sum(1 for row in review.rows if not row.answer_same),
        "answer_leaked": sum(1 for row in review.rows if not row.no_leak),
        "meaning_drift": len(review.notes),
        "meaning_drift_in_set": len(noted),
        "review_note_ids": noted,
    }


def _exclude_reason(paraphrase: Paraphrase, source: Row | None) -> ExcludeReason | None:
    """세트에서 빼야 하면 그 이유, 넣어도 되면 None."""
    if paraphrase.problems:
        return "problems"
    if source is None:
        return "missing"
    if source["kind"] != SOURCE_KIND:
        return "not_answerable"
    if source["question"] != paraphrase.question:
        return "question_mismatch"
    return None


def _parse_paraphrase(line: Mapping[str, Any]) -> Paraphrase:
    return Paraphrase(
        sample=line["sample"],
        id=line["id"],
        question=line["question"],
        paraphrase=line["paraphrase"],
        model=line["model"],
        prompt_version=line["prompt_version"],
        problems=tuple(line["problems"]),
    )


def _count(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _report(out_dir: Path, meta: Mapping[str, Any]) -> None:
    counts = meta["counts"]
    excluded = counts["excluded"]
    review = meta["human_review"]
    print(f"저장: {out_dir / 'items.jsonl'} · meta.json")
    print(
        f"패러프레이즈 문항 {counts[KIND]} (F4 패러프레이즈 {counts['f4_paraphrases']} 중)"
        f" · 뺀 것 {sum(excluded.values())} {excluded}"
    )
    print(f"쓰지 않은 표본: {counts['other_samples']}")
    print(f"패러프레이즈 모델 {meta['paraphrase_models']} · 프롬프트 {meta['prompt_versions']}")
    print(
        f"사람 검수: 확인 {review['checked']} (이 세트 {review['checked_in_set']})"
        f" · 정답 바뀜 {review['answer_changed']} · 정답 샘 {review['answer_leaked']}"
        f" · 표현 달라짐 {review['meaning_drift']} (이 세트 {review['meaning_drift_in_set']}:"
        f" {review['review_note_ids']})"
    )
    for item in meta["excluded"]:
        print(f"  뺌 {item['id']}: {item['reason']}")


def _read_jsonl(path: Path) -> list[Row]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: Sequence[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
