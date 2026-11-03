"""KorQuAD dev 문항을 용도별로 나눈다: 답변 정책 임계값(calibrate) · LLM 벤치(bench) · 답변 평가(eval).

나눔은 문단 단위다 — 같은 문단의 질문들은 문맥을 공유하므로 서로 다른 용도에 흩어지면 안 된다.
문단 키의 해시로 정하므로 어느 스크립트가 언제 뽑아도 같은 문단은 같은 용도에 들어간다.
"""

import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, get_args

from playground_evalkit import KorQuADItem, contains_answer, load_korquad

Split = Literal["calibrate", "bench", "eval"]
SPLITS: tuple[str, ...] = get_args(Split)
SALT = "korquad-dev-split-v1"  # 바꾸면 모든 문단의 용도가 바뀌어 이미 뽑은 표본과 겹친다 — 바꾸지 않는다
# 해시 % 100 이 이 값 미만이면 그 용도. dev 문단 964개 기준 calibrate 약 190 · bench 약 390 · eval 약 390
# (bench 는 F4 답 있음 300 이 상위 5개 필터 뒤에 남을 만큼, eval 은 F9 무작위 300 + 답변 평가)
_UPPER_BOUNDS: tuple[tuple[Split, int], ...] = (("calibrate", 20), ("bench", 60), ("eval", 100))


def subset_korquad(subset: Path) -> tuple[Path, list[KorQuADItem]]:
    """부분 코퍼스 subset.json 이 가리키는 KorQuAD dev 경로와, 그 제목 보정표로 읽은 문항."""
    summary = json.loads((subset / "subset.json").read_text(encoding="utf-8"))
    korquad = Path(summary["korquad_dev"])
    return korquad, load_korquad(korquad, title_aliases=summary["korquad_title_map"])


def korquad_paragraphs(path: Path) -> dict[str, str]:
    """KorQuAD JSON 의 질문 id → 문단 키(`원래 제목#문단 순번`). dev 는 문서 제목이 겹치지 않는다.

    KorQuAD id 앞부분은 문단과 맞지 않아(dev 문단 964개에 앞부분 2,416종) 파일 구조에서 읽는다.
    질문이 없는 문단은 키가 생기지 않는다 (dev 에 하나).
    """
    data = json.loads(path.read_text(encoding="utf-8"))["data"]
    paragraphs: dict[str, str] = {}
    for article in data:
        for index, paragraph in enumerate(article["paragraphs"]):
            for qa in paragraph["qas"]:
                paragraphs[qa["id"]] = f"{article['title']}#{index}"
    return paragraphs


def split_of(paragraph: str) -> Split:
    """문단 키의 용도. 해시로 정해 언제 불러도 같다."""
    bucket = int(hashlib.sha256(f"{SALT}:{paragraph}".encode()).hexdigest(), 16) % 100
    return next(split for split, upper in _UPPER_BOUNDS if bucket < upper)


def answer_kept_sample(
    items: Sequence[KorQuADItem],
    paragraphs: Mapping[str, str],
    texts: Mapping[str, str],
    split: Split,
    *,
    seed: int,
) -> tuple[list[KorQuADItem], list[KorQuADItem]]:
    """용도(split) 문단마다 고른 문항과, 그중 정답 문장이 지금 문서에 남은 것.

    문서는 있어도 정답 문장이 2018년 이후 고쳐졌으면 뺀다 — 검색이 아니라 문자열 대조라 선택 편향이 없다.

    Args:
      items: 고를 질문들.
      paragraphs: `korquad_paragraphs` 결과.
      texts: 정규화한 제목 → 지금 문서 본문. 없는 문서의 문항은 고르기 전에 뺀다.
      split: 고를 용도.
      seed: `sample_per_paragraph` 의 seed.

    Returns:
      (문단마다 고른 문항, 정답 문장이 남은 문항).
    """
    in_corpus = [item for item in items if item.title in texts]
    picked = sample_per_paragraph(in_corpus, paragraphs, split, seed=seed)
    return picked, [item for item in picked if contains_answer(item.title, texts[item.title], item)]


def sample_per_paragraph(
    items: Sequence[KorQuADItem], paragraphs: Mapping[str, str], split: Split, *, seed: int
) -> list[KorQuADItem]:
    """고를 용도(split)에 속한 문단마다 질문 하나를 seed 로 고른다. 문단 키 순으로 돌려준다.

    Args:
      items: 고를 질문들. 정답 문서가 코퍼스에 있는 것만 넘기는 등 거르기는 쓰는 쪽이 한다.
      paragraphs: `korquad_paragraphs` 결과. items 의 모든 id 가 있어야 한다.
      split: 고를 용도.
      seed: 같은 seed 면 같은 질문을 고른다.
    """
    by_paragraph: dict[str, list[KorQuADItem]] = defaultdict(list)
    for item in items:
        paragraph = paragraphs[item.id]
        if split_of(paragraph) == split:
            by_paragraph[paragraph].append(item)
    rng = random.Random(seed)
    return [
        rng.choice(sorted(group, key=lambda item: item.id))
        for _, group in sorted(by_paragraph.items())
    ]
