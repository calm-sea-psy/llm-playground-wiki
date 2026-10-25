"""KorQuAD 질문 패러프레이즈를 Claude API 로 만든다 — F4 LLM 벤치 오염 판정 · F9 리트리버 민감도용.

문항마다 같은 문서 · 같은 사실 · 같은 정답을 묻되 표현(어순 · 낱말)만 바꾼 질문 하나를 만든다.
오염 판정은 후보 모델(로컬 LLM)이 원문 질문을 외웠는지 보는 것이라, 패러프레이즈는 후보 모델이 아닌 Claude 로 만든다.
모델에는 질문과 정답을 함께 준다 (정답이 바뀌지 않게). 출력은 구조화 출력(JSON) 한 필드.

대상 (두 표본은 문단 해시 분할이 달라 겹치지 않는다)
  f4-answerable  F4 벤치 세트의 답 있음 300 (../llm-playground-llm/bench/data/f4-korquad-9014dd9e/items.jsonl)
  f9-random      KorQuAD dev 의 eval 몫에서 문단마다 한 문항 → 정답 문서가 코퍼스에 있고 정답 문자열이 지금
                 문서에 남은 것 → seed 0 으로 300. 검색 결과로 거르지 않는다 (선택 편향 없는 표본)

단계
  run     패러프레이즈를 data/bench/paraphrase-v1.jsonl 에 한 줄씩 더한다. 끝난 문항은 건너뛰므로 멈췄다 다시 돌려도 된다.
          --per-sample N 이면 표본마다 seed 로 N 개만 (파일럿 — 전체를 돌릴 때 다시 하지 않는다).
          자동 점검(정답이 질문에 샘 · 원문과 같음)에 걸리면 이유를 알려 주고 한 번 더 쓰게 한다.
          끝에 토큰 수와 금액을 출력한다
  review  표본마다 25개를 seed 로 골라 원문 · 패러프레이즈 · 정답을 나란히 data/bench/paraphrase-v1-review.md 에 쓴다

KorQuAD 는 CC BY-ND 라 패러프레이즈(2차 저작물)는 data/ 에만 둔다 (git 제외).
인증은 anthropic SDK 기본 순서 (ANTHROPIC_API_KEY, 없으면 `ant auth login` 프로필).
레포 루트(data/ 가 있고 ../llm-playground-llm 이 나란히 있는 곳)에서 돌린다.

사용법:
  uv run --group claude python scripts/paraphrase_korquad.py run --per-sample 10
  uv run --group claude python scripts/paraphrase_korquad.py run
  uv run --group claude python scripts/paraphrase_korquad.py review
"""

import argparse
import asyncio
import json
import random
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TextIO, get_args

import anthropic
import pyarrow.parquet as pq
from anthropic import AsyncAnthropic
from anthropic.types.beta import BetaMessage
from playground_evalkit import (
    contains_answer,
    load_korquad,
    normalize_title,
    render_table,
    squash,
)

from playground_wiki.evaluation import korquad_paragraphs, sample_per_paragraph

SUBSET = Path("data/parsed/kowiki-subset-10000")  # subset.json: KorQuAD dev 경로 · 제목 보정표 (D4)
PARSED = Path("data/parsed/kowiki")  # 전체 덤프 파싱 결과 — 정답 문자열이 지금 문서에 남았는지 대조
F4_ITEMS = Path("../llm-playground-llm/bench/data/f4-korquad-9014dd9e/items.jsonl")
OUT_PATH = Path("data/bench/paraphrase-v1.jsonl")
REVIEW_PATH = Path("data/bench/paraphrase-v1-review.md")
SEED = 0
F9_SIZE = 300
REVIEW_PER_SAMPLE = 25

MODEL = "claude-opus-5-5"
# Opus 5.5 의 기본값을 명시. 질문 한 줄이라 더 올릴 이유가 없다 — 파일럿에서 정답이 바뀌는 문항이 보이면 올린다
EFFORT = "medium"
MAX_TOKENS = (
    16000  # 생각 + 질문 한 줄로는 닿지 않는 값. 닿으면 그 문항은 실패로 두고 다음에 다시 돈다
)
CONCURRENCY = 8  # 동시 요청 수. 속도 제한(429)은 SDK 가 기다렸다 재시도한다
MAX_RETRIES = 5  # SDK 재시도 (429 · 5xx · 연결 오류)
# 정책 거절이면 서버가 거절 종류에 맞는 권장 모델로 다시 돌린다 (fallbacks="default"). 답한 모델은 행의 model 에 남는다
FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_ATTEMPTS = (
    2  # 자동 점검에 걸리면 이유를 알려 주고 한 번 더. 그래도 걸리면 problems 를 남기고 사람이 본다
)
# 2026-10-06 가격표 기준 claude-opus-5-5 의 100만 토큰당 달러 (생각 토큰은 출력으로 센다)
INPUT_USD_PER_MTOK = 4.0
OUTPUT_USD_PER_MTOK = 20.0

PROMPT_VERSION = (
    "paraphrase-v1"  # 아래 프롬프트 · 출력 형식을 바꾸면 올리고 출력 파일 이름도 바꾼다
)
SYSTEM_PROMPT = """\
너는 한국어 독해 평가 세트의 질문을 고쳐 쓰는 편집자다. 받은 질문을 같은 뜻의 다른 표현으로 다시 쓴다.

다시 쓴 질문은 다음을 모두 지킨다.
1. 같은 문서의 같은 사실을 묻고, 정답이 원래 정답 그대로다. 묻는 대상(누구 · 무엇 · 언제 · 어디 · 몇 · 왜)을 바꾸지 않는다.
2. 원래 질문에 있는 단서(인물 · 시기 · 장소 · 조건)를 빼거나 새로 더하지 않는다. 정답이 하나로 정해지는 정도가 원래 질문과 같아야 한다.
3. 어순과 낱말을 바꾼다. 원래 질문의 구절을 길게 그대로 옮기지 않는다. 고유명사 · 숫자 · 날짜 · 작품 제목은 그대로 둔다 — 바꾸면 다른 대상이 된다.
4. 정답 문자열을 질문에 넣지 않는다. 원래 질문에 없던 정답의 낱말도 새로 넣지 않는다.
5. 사람이 실제로 물을 법한 자연스러운 한국어 의문문 한 문장이다.

정답은 질문이 무엇을 묻는지 확인하는 데만 쓴다. 정답을 짐작하게 하는 정보를 더하지 않는다."""
OUTPUT_FORMAT = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {"paraphrase": {"type": "string", "description": "다시 쓴 질문 한 문장"}},
        "required": ["paraphrase"],
        "additionalProperties": False,
    },
}

Sample = Literal["f4-answerable", "f9-random"]
SAMPLES: tuple[str, ...] = get_args(Sample)
Problem = Literal["empty", "same_as_original", "answer_leak"]
# 다시 쓰게 할 때 모델에 알려 주는 이유
PROBLEM_REASONS: dict[Problem, str] = {
    "empty": "질문이 비어 있다",
    "same_as_original": "원래 질문과 같다",
    "answer_leak": "정답 문자열이 질문에 들어 있다",
}
TRANSIENT_ERRORS = (  # SDK 재시도 뒤에도 남은 일시 오류 — 그 문항만 건너뛰고 다음 실행에서 다시 돈다
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    anthropic.ServiceUnavailableError,
    anthropic.OverloadedError,
    anthropic.DeadlineExceededError,
    anthropic.APIConnectionError,
)


class ParaphraseError(RuntimeError):
    """응답에서 패러프레이즈를 얻지 못함 (거절 · 한도 도달 · JSON 아님)."""


@dataclass(frozen=True)
class SourceItem:
    """패러프레이즈할 KorQuAD 질문 하나."""

    sample: Sample
    id: str
    """표본의 문항 id 그대로 (F4 는 `korquad:` 접두가 붙은 것, F9 는 KorQuAD id)."""
    title: str
    """정답 문서 제목."""
    question: str
    answers: tuple[str, ...]


@dataclass(frozen=True)
class Paraphrase:
    """문항 하나의 패러프레이즈."""

    text: str
    model: str
    """실제로 답한 모델. 거절 뒤 fallback 이 돌았으면 MODEL 과 다르다."""
    attempts: int
    problems: tuple[Problem, ...]
    """마지막 시도에 남은 자동 점검 문제. 비면 통과."""
    input_tokens: int
    """모든 시도의 합."""
    output_tokens: int
    """모든 시도의 합 (생각 토큰 포함)."""


class Paraphraser:
    """Claude 로 질문 하나를 패러프레이즈한다. 자동 점검에 걸리면 이유를 알려 주고 다시 쓰게 한다.

    예)
        async with AsyncAnthropic() as client:
            result = await Paraphraser(client).paraphrase(item)
    """

    def __init__(self, client: AsyncAnthropic) -> None:
        self._client = client

    async def paraphrase(self, item: SourceItem) -> Paraphrase:
        """item 의 패러프레이즈. 점검 문제가 남아도 MAX_ATTEMPTS 번째 결과를 돌려준다.

        Raises:
          ParaphraseError: 거절 · 한도 도달 · JSON 이 아닌 응답.
        """
        rejected: tuple[str, Problem] | None = None
        input_tokens = output_tokens = 0
        for attempt in range(1, MAX_ATTEMPTS + 1):
            response = await self._client.beta.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": _user_message(item, rejected)}],
                thinking={"type": "adaptive"},
                output_config={"effort": EFFORT, "format": OUTPUT_FORMAT},
                fallbacks="default",
                betas=[FALLBACK_BETA],
            )
            input_tokens += response.usage.input_tokens
            output_tokens += response.usage.output_tokens
            text = _parse(response)
            problems = check(item, text)
            if not problems or attempt == MAX_ATTEMPTS:
                break
            rejected = (text, problems[0])
        return Paraphrase(
            text=text,
            model=response.model,
            attempts=attempt,
            problems=problems,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


def check(item: SourceItem, text: str) -> tuple[Problem, ...]:
    """패러프레이즈의 자동 점검 문제. 원래 질문에 이미 들어 있던 정답 문자열은 샌 것으로 치지 않는다."""
    question, candidate = squash(item.question), squash(text)
    problems: list[Problem] = []
    if not candidate:
        problems.append("empty")
    elif candidate == question:
        problems.append("same_as_original")
    if any(squash(a) in candidate and squash(a) not in question for a in item.answers):
        problems.append("answer_leak")
    return tuple(problems)


def main() -> None:
    args = _parse_args()
    if args.step == "run":
        run(args.per_sample)
    elif args.step == "review":
        review()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["run", "review"])
    parser.add_argument("--per-sample", type=int, help="run 만: 표본마다 seed 로 이만큼만 (파일럿)")
    return parser.parse_args()


def run(per_sample: int | None) -> None:
    items = _f4_items() + _f9_items()
    _write_meta()
    done = {(row["sample"], row["id"]) for row in _read_rows()}
    todo = [item for item in _pick(items, per_sample) if (item.sample, item.id) not in done]
    print(f"문항 {len(items)} · 끝남 {len(done)} · 이번에 {len(todo)} ({MODEL}, effort {EFFORT})")
    if not todo:
        return
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # 끝나는 대로 한 줄씩 쓴다 — 중간에 멈춰도 끝난 문항은 남고, 다시 돌리면 건너뛴다
    with OUT_PATH.open("a", encoding="utf-8", newline="\n") as out:
        rows, failures = asyncio.run(_paraphrase_all(todo, out))
    _print_summary(rows, failures, total=len(items))


async def _paraphrase_all(items: list[SourceItem], out: TextIO) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    failures: list[str] = []
    semaphore = asyncio.Semaphore(CONCURRENCY)
    async with AsyncAnthropic(max_retries=MAX_RETRIES) as client:
        paraphraser = Paraphraser(client)

        async def one(item: SourceItem) -> None:
            async with semaphore:
                try:
                    result = await paraphraser.paraphrase(item)
                except (ParaphraseError, *TRANSIENT_ERRORS) as e:
                    failures.append(f"{item.sample} {item.id}: {e}")
                    print(f"  실패 {item.sample} {item.id}: {e}", flush=True)
                    return
            row = _row(item, result)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            rows.append(row)
            mark = f" [{', '.join(result.problems)}]" if result.problems else ""
            print(f"  {len(rows)}/{len(items)} {item.sample} {item.id}{mark}", flush=True)

        # 인증 · 잘못된 요청 같은 오류는 잡지 않는다 — TaskGroup 이 나머지를 멈춘다
        async with asyncio.TaskGroup() as group:
            for item in items:
                group.create_task(one(item))
    return rows, failures


def _user_message(item: SourceItem, rejected: tuple[str, Problem] | None) -> str:
    message = f"원래 질문: {item.question}\n정답: {' / '.join(item.answers)}"
    if rejected:
        text, problem = rejected
        message += (
            f"\n\n앞서 쓴 질문 「{text}」은 쓸 수 없다 — {PROBLEM_REASONS[problem]}. "
            "같은 조건으로 다시 쓴다."
        )
    return message


def _parse(response: BetaMessage) -> str:
    if response.stop_reason != "end_turn":
        details = response.stop_details.category if response.stop_details else None
        raise ParaphraseError(f"stop_reason={response.stop_reason} ({details=})")
    # output_config.format 이면 text 블록 하나에 스키마에 맞는 JSON 이 온다 (앞에 생각 블록)
    text = next((block.text for block in response.content if block.type == "text"), None)
    if text is None:
        raise ParaphraseError("text 블록이 없다")
    try:
        return json.loads(text)["paraphrase"].strip()
    except (json.JSONDecodeError, KeyError) as e:
        raise ParaphraseError(f"JSON 이 아니다: {text[:200]!r}") from e


def _row(item: SourceItem, result: Paraphrase) -> dict:
    return {
        "sample": item.sample,
        "id": item.id,
        "title": item.title,
        "question": item.question,
        "answers": list(item.answers),
        "paraphrase": result.text,
        "model": result.model,
        "prompt_version": PROMPT_VERSION,
        "attempts": result.attempts,
        "problems": list(result.problems),
        "usage": {"input_tokens": result.input_tokens, "output_tokens": result.output_tokens},
    }


def _f4_items() -> list[SourceItem]:
    rows = [json.loads(line) for line in F4_ITEMS.read_text(encoding="utf-8").splitlines() if line]
    items = [
        SourceItem(
            sample="f4-answerable",
            id=row["id"],
            title=row["gold_title"],
            question=row["question"],
            answers=tuple(row["answers"]),
        )
        for row in rows
        if row["kind"] == "answerable"
    ]
    print(f"f4-answerable: {F4_ITEMS} 의 답 있음 {len(items)}")
    return items


def _f9_items() -> list[SourceItem]:
    """calibrate_threshold.py 의 sample() 과 같은 거르기를 eval 몫에 — 그중 seed 로 F9_SIZE 개."""
    summary = json.loads((SUBSET / "subset.json").read_text(encoding="utf-8"))
    korquad = Path(summary["korquad_dev"])
    items = load_korquad(korquad, title_aliases=summary["korquad_title_map"])
    paragraphs = korquad_paragraphs(korquad)
    texts = _document_texts({item.title for item in items})
    in_corpus = [item for item in items if item.title in texts]
    picked = sample_per_paragraph(in_corpus, paragraphs, "eval", seed=SEED)
    # 문서는 있어도 정답 문장이 2018년 이후 고쳐졌으면 뺀다 — 검색이 아니라 문자열 대조라 선택 편향이 없다
    kept = [item for item in picked if contains_answer(item.title, texts[item.title], item)]
    chosen = random.Random(SEED).sample(kept, F9_SIZE)
    print(
        f"f9-random: eval 문단마다 한 문항 {len(picked)} → 정답이 코퍼스에 남은 것 {len(kept)} "
        f"→ seed {SEED} 로 {len(chosen)}"
    )
    return [
        SourceItem(
            sample="f9-random",
            id=item.id,
            title=item.title,
            question=item.question,
            answers=(item.answer,),
        )
        for item in chosen
    ]


def _pick(items: list[SourceItem], per_sample: int | None) -> list[SourceItem]:
    if per_sample is None:
        return items
    picked = []
    for sample in SAMPLES:
        group = [item for item in items if item.sample == sample]
        picked += random.Random(SEED).sample(group, min(per_sample, len(group)))
    return picked


def _write_meta() -> None:
    """프롬프트 · 모델 설정을 출력 옆에 남긴다. 이미 있는데 다르면 멈춘다 (한 파일에 두 프롬프트가 섞이지 않게)."""
    meta = {
        "prompt_version": PROMPT_VERSION,
        "model": MODEL,
        "effort": EFFORT,
        "max_attempts": MAX_ATTEMPTS,
        "system": SYSTEM_PROMPT,
        "output_format": OUTPUT_FORMAT,
        "sources": {
            "f4-answerable": str(F4_ITEMS),
            "f9-random": f"KorQuAD dev eval 몫, seed {SEED}",
        },
    }
    path = OUT_PATH.with_suffix(".meta.json")
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        saved.pop("created_at")
        if saved != meta:
            raise SystemExit(
                f"{path} 의 프롬프트 · 설정이 지금 코드와 다르다 — PROMPT_VERSION 을 올린다"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    meta["created_at"] = datetime.now(tz=UTC).isoformat(timespec="seconds")
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _print_summary(rows: list[dict], failures: list[str], *, total: int) -> None:
    problems = Counter(problem for row in rows for problem in row["problems"])
    retried = sum(row["attempts"] > 1 for row in rows)
    models = Counter(row["model"] for row in rows)
    kept = [_kept_words(row["question"], row["paraphrase"]) for row in rows]
    print(
        f"\n이번 실행: 성공 {len(rows)} · 실패 {len(failures)} · 다시 쓰게 함 {retried} · "
        f"남은 점검 문제 {dict(problems) or '없음'} · 답한 모델 {dict(models)}"
    )
    if kept:
        print(
            f"원문 어절이 그대로 남은 비율: 중앙값 {statistics.median(kept):.0%} · 최대 {max(kept):.0%}"
        )
    print(f"이번 실행 {_cost_line(rows)}")
    all_rows = _read_rows()
    print(f"파일 전체 {len(all_rows)}/{total} {_cost_line(all_rows)}")
    if all_rows and len(all_rows) < total:
        per_item = _usd(all_rows) / len(all_rows)
        print(f"남은 {total - len(all_rows)}문항 예상 ${per_item * (total - len(all_rows)):.2f}")


def _cost_line(rows: list[dict]) -> str:
    input_tokens = sum(row["usage"]["input_tokens"] for row in rows)
    output_tokens = sum(row["usage"]["output_tokens"] for row in rows)
    per_item = f" (문항당 ${_usd(rows) / len(rows):.4f})" if rows else ""
    return f"입력 {input_tokens:,} · 출력 {output_tokens:,} 토큰 → ${_usd(rows):.2f}{per_item}"


def _usd(rows: list[dict]) -> float:
    input_tokens = sum(row["usage"]["input_tokens"] for row in rows)
    output_tokens = sum(row["usage"]["output_tokens"] for row in rows)
    return (input_tokens * INPUT_USD_PER_MTOK + output_tokens * OUTPUT_USD_PER_MTOK) / 1_000_000


def _kept_words(question: str, paraphrase: str) -> float:
    """원문 어절 중 패러프레이즈에 그대로 있는 비율 (문장부호 무시). 표현을 얼마나 바꿨는지 보는 대략의 값."""
    original = {word.strip("?.,!\"'") for word in question.split()} - {""}
    rewritten = {word.strip("?.,!\"'") for word in paraphrase.split()}
    return len(original & rewritten) / len(original) if original else 0.0


def review() -> None:
    rows = _read_rows()
    rng = random.Random(SEED)
    picked = []
    for sample in SAMPLES:
        group = sorted((row for row in rows if row["sample"] == sample), key=lambda row: row["id"])
        picked += rng.sample(group, min(REVIEW_PER_SAMPLE, len(group)))
    table = [
        {
            "번호": index,
            "표본": row["sample"],
            "id": row["id"],
            "원문": _cell(row["question"]),
            "패러프레이즈": _cell(row["paraphrase"]),
            "정답": _cell(" / ".join(row["answers"])),
            "자동 점검": ", ".join(row["problems"]) or "통과",
            "정답 같음": "",
            "정답 안 샘": "",
        }
        for index, row in enumerate(picked, start=1)
    ]
    report = (
        f"# 패러프레이즈 확인 — {PROMPT_VERSION} ({datetime.now(tz=UTC):%Y-%m-%d})\n\n"
        f"- 표본마다 {REVIEW_PER_SAMPLE}개를 seed {SEED} 로 골랐다 (전체 {len(rows)}개 중)\n"
        "- 확인 기준: 정답 같음 — 패러프레이즈가 원문과 같은 대상을 물어 정답이 그대로인가. "
        "정답 안 샘 — 정답 문자열(이나 그걸 짐작하게 하는 말)이 질문에 들어가지 않았는가. 빈칸에 O / X\n"
        "- 자동 점검은 문자열 대조만 한다 (정답 포함 · 원문과 같음). 뜻이 바뀐 것은 사람이 본다\n\n"
        + render_table(table, list(table[0]))
    )
    REVIEW_PATH.write_text(report, encoding="utf-8", newline="\n")
    print(f"{len(picked)}개 · 저장: {REVIEW_PATH}")


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


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


def _read_rows() -> list[dict]:
    if not OUT_PATH.exists():
        return []
    rows = [json.loads(line) for line in OUT_PATH.read_text(encoding="utf-8").splitlines() if line]
    if stale := {row["prompt_version"] for row in rows} - {PROMPT_VERSION}:
        raise SystemExit(f"{OUT_PATH} 에 다른 프롬프트 버전 {sorted(stale)} 이 섞였다")
    return rows


if __name__ == "__main__":
    main()
