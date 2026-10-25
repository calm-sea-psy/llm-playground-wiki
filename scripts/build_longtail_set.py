"""F4 롱테일 · 답 없음 B 문항을 Wikidata 로 만든다 — 덜 알려진 문서의 (속성 · 참값)을 템플릿 질문으로.

KorQuAD dev 는 2018년부터 공개돼 후보 모델이 문항을 봤을 수 있다. 그래서 역링크가 적은 문서(롱테일)에서 문항을 따로 만든다.
단계별로 돌린다 — 단계마다 결과를 data/bench/ 에 남기고, Wikidata 응답은 항목마다 캐시해 다시 묻지 않는다.
  candidates  코퍼스 문서 중 역링크 LONGTAIL_MAX 이하 · KorQuAD 문서가 아닌 것에서 seed 로 CANDIDATES 개
              → data/bench/longtail-candidates-v1.jsonl (역링크: scripts/count_backlinks.py)
  fetch       후보 제목으로 wbgetentities(sites=kowiki) → 주제 항목, 템플릿 속성이 가리키는 값 항목의 한국어 표기 →
              data/bench/wikidata/entities.jsonl · values.jsonl (네트워크만 쓴다 — GPU 를 쓰는 벤치와 같이 돌려도 된다)
  questions   속성 템플릿(playground_wiki.evaluation.wikidata)으로 질문과 참값 표기 → data/bench/longtail-questions-v1.jsonl
              사람 항목 중 사망일이 없는 이(생존 인물로 본다)에게는 민감 질문 하나 (참값 없음 — null 만 정답)
  search      /ask 와 같은 검색 상위 5 → data/bench/longtail-search-v1.jsonl (GPU — 벤치가 끝난 뒤에)
  label       답 있음 · 답 없음 B · 버림(이유)을 붙여 → data/bench/longtail-labeled-v1.jsonl. 버린 것도 검색 결과와 함께
              남긴다 (선택 편향 없는 표본이 필요할 때)
  pilot       라벨마다 seed 로 몇 개씩 보여 준다 — 어색한 질문 · 너무 빡빡한 조건 · 빠진 표기를 눈으로 거른다
  build       1단계 세트 → {out_dir}/f4-longtail-{검색 설정 해시}/items.jsonl · meta.json (LLM 레포 bench/data)
              longtail(답 있음) LONGTAIL · no_fact(답 없음 B) NO_FACT · sensitive(사적 정보) SENSITIVE 개.
              속성(민감은 질문 꼴)마다 돌아가며 골라 생년월일 · 출생지 같은 많은 속성이 세트를 차지하지 않게 하고,
              한 주제(QID)에서는 문항 하나만. 남은 문항은 2단계 결선 몫이다

Wikidata 는 CC0. 요청 규칙: 연락처가 담긴 User-Agent, 요청 하나씩 REQUEST_INTERVAL 초 간격, 429 면 물러난다.
data/ 는 git 제외.

사용법:
  uv run python scripts/build_longtail_set.py candidates
  uv run python scripts/build_longtail_set.py fetch
  uv run python scripts/build_longtail_set.py questions
  uv run python scripts/build_longtail_set.py search
  uv run python scripts/build_longtail_set.py label
  uv run python scripts/build_longtail_set.py pilot
  uv run python scripts/build_longtail_set.py build
"""

import argparse
import asyncio
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pyarrow.parquet as pq
from playground_evalkit import load_korquad, normalize_title
from playground_retrieval.search import SearchService
from playground_retrieval.store import PgStore, QdrantStore

from playground_wiki.api.config import ApiConfig
from playground_wiki.evaluation.wikidata import (
    PROPERTIES,
    Fact,
    FactQuestion,
    fact_question,
    facts,
    is_human,
    kowiki_titles,
    label_fact,
    sensitive_question,
    value_ids,
    value_names,
)

SUBSET = Path("data/parsed/kowiki-subset-10000")  # subset.json: KorQuAD dev 경로 · 제목 보정표 (D4)
PARSED = Path("data/parsed/kowiki")
BACKLINKS = Path("data/bench/backlinks-20261001.parquet")
CANDIDATES_PATH = Path("data/bench/longtail-candidates-v1.jsonl")
WIKIDATA = Path("data/bench/wikidata")
ENTITIES_PATH = WIKIDATA / "entities.jsonl"
VALUES_PATH = WIKIDATA / "values.jsonl"
# v2: 사람 검수(2026-10-24, 50개) 뒤 질문 표현 · 판정 규칙을 고쳤다 — 질문 문장이 그대로인 문항은 v1 검색 결과를 옮겨 쓴다
QUESTIONS_PATH = Path("data/bench/longtail-questions-v2.jsonl")
SEARCH_PATH = Path("data/bench/longtail-search-v2.jsonl")
LABELED_PATH = Path("data/bench/longtail-labeled-v2.jsonl")
PREVIOUS_QUESTIONS = Path("data/bench/longtail-questions-v1.jsonl")
PREVIOUS_SEARCH = Path("data/bench/longtail-search-v1.jsonl")
SEED = 0
LONGTAIL_MAX = 2  # 코퍼스 문서 역링크 하위 약 25% (p25 = 2, 2026-10-24 집계)
CANDIDATES = 6000  # 1단계 롱테일 100 · B 100 + 2단계 +300 · +200 이 라벨 뒤에 남을 만큼 (수율은 파일럿으로 확인)
TOP_K = 5
API = "https://www.wikidata.org/w/api.php"
USER_AGENT = "llm-playground-wiki-bench/0.1 (https://github.com/calm-sea-psy; LLM benchmark question set) httpx"
BATCH = 50  # wbgetentities 가 한 번에 받는 제목 · id 수
REQUEST_INTERVAL = 1.0  # Wikimedia API 규칙 — 요청은 하나씩, 간격을 둔다
MAX_RETRIES = 5
PILOT_PER_LABEL = 10
OUT_DIR = Path("../llm-playground-llm/bench/data")
# 고르는 규칙 · 질문 · 판정 규칙을 바꾸면 올린다 — 세트 이름 해시에 들어간다. v2: 사람 검수 반영
SET_RULES = "v2"
# 판정 라벨 → (세트 kind, 1단계 문항 수). 롱테일 답 있음은 KorQuAD 300 과 합치지 않고 따로 보고한다
STAGE1 = {
    "answerable": ("longtail", 100),
    "no_fact": ("no_fact", 100),
    "sensitive": ("sensitive", 30),
}


def main() -> None:
    args = _parse_args()
    steps = {
        "candidates": candidates,
        "fetch": fetch,
        "questions": questions,
        "search": search,
        "label": label,
        "pilot": pilot,
    }
    if args.step == "build":
        build(args.out_dir)
    else:
        steps[args.step]()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "step", choices=["candidates", "fetch", "questions", "search", "label", "pilot", "build"]
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="build 만: 세트를 쓸 곳")
    return parser.parse_args()


def candidates() -> None:
    backlinks = pq.read_table(BACKLINKS).to_pydict()
    corpus = set()
    for path in sorted(PARSED.glob("docs-*.parquet")):
        corpus.update(pq.read_table(path, columns=["id"])["id"].to_pylist())
    summary = json.loads((SUBSET / "subset.json").read_text(encoding="utf-8"))
    korquad = {
        item.title
        for item in load_korquad(
            Path(summary["korquad_dev"]), title_aliases=summary["korquad_title_map"]
        )
    }
    pool = [
        {"page_id": page_id, "title": title, "backlinks": count}
        for page_id, title, count in zip(
            backlinks["page_id"], backlinks["title"], backlinks["backlinks"], strict=True
        )
        if page_id in corpus and count <= LONGTAIL_MAX and normalize_title(title) not in korquad
    ]
    picked = random.Random(SEED).sample(pool, min(CANDIDATES, len(pool)))
    _write_jsonl(CANDIDATES_PATH, sorted(picked, key=lambda row: int(row["page_id"])))
    print(
        f"코퍼스 문서 {len(corpus):,} 중 역링크 {LONGTAIL_MAX} 이하 · KorQuAD 아님 {len(pool):,} → {len(picked):,}\n"
        f"저장: {CANDIDATES_PATH}"
    )


def fetch() -> None:
    titles = [row["title"] for row in _read_jsonl(CANDIDATES_PATH)]
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60) as client:
        done = {row["title"] for row in _read_jsonl_if(ENTITIES_PATH)}
        todo = [title for title in titles if title not in done]
        print(f"주제 항목: 후보 {len(titles):,} · 남은 {len(todo):,}", flush=True)
        for index, batch in enumerate(_batches(todo), start=1):
            response = _get(
                client, sites="kowiki", titles="|".join(batch), props="claims|sitelinks"
            )
            found = {title: qid for qid, title in kowiki_titles(response).items()}
            entities = response.get("entities", {})
            for title in batch:
                qid = found.get(title)
                _append_jsonl(
                    ENTITIES_PATH,
                    {"title": title, "qid": qid, "entity": entities.get(qid) if qid else None},
                )
            if index % 10 == 0:
                print(f"  {index * BATCH:,}/{len(todo):,}", flush=True)

        needed = set()
        for row in _read_jsonl_if(ENTITIES_PATH):
            if row["entity"]:
                needed |= value_ids(row["entity"])
        have = {row["qid"] for row in _read_jsonl_if(VALUES_PATH)}
        todo_ids = sorted(needed - have)
        print(f"값 항목: {len(needed):,} · 남은 {len(todo_ids):,}", flush=True)
        for index, batch in enumerate(_batches(todo_ids), start=1):
            response = _get(
                client, ids="|".join(batch), props="labels|aliases|sitelinks", languages="ko"
            )
            for qid in batch:
                _append_jsonl(
                    VALUES_PATH, {"qid": qid, "entity": response.get("entities", {}).get(qid)}
                )
            if index % 20 == 0:
                print(f"  {index * BATCH:,}/{len(todo_ids):,}", flush=True)
    print(f"저장: {ENTITIES_PATH} · {VALUES_PATH}")


def questions() -> None:
    backlinks = {row["title"]: row["backlinks"] for row in _read_jsonl(CANDIDATES_PATH)}
    names = {
        row["qid"]: value_names(row["entity"]) for row in _read_jsonl(VALUES_PATH) if row["entity"]
    }
    rows, per_property, humans = [], Counter(), 0
    entities = [row for row in _read_jsonl(ENTITIES_PATH) if row["entity"]]
    for row in entities:
        qid, subject, entity = row["qid"], row["title"], row["entity"]
        human = is_human(entity)
        humans += human
        for fact in facts(qid, subject, entity, names):
            question = fact_question(fact)
            rows.append(_question_row(question, backlinks[subject]))
            per_property[question.template.name] += 1
        if human and "P570" not in entity.get("claims", {}):
            question_id, text = sensitive_question(qid, subject)
            rows.append(
                {
                    "id": question_id,
                    "kind": "sensitive",
                    "qid": qid,
                    "subject": subject,
                    "question": text,
                    "answers": [],
                    "backlinks": backlinks[subject],
                }
            )
    _write_jsonl(QUESTIONS_PATH, rows)
    print(
        f"후보 {len(backlinks):,} 중 Wikidata 항목 {len(entities):,} (사람 {humans:,}) → 질문 {len(rows):,} "
        f"(민감 {sum(row['kind'] == 'sensitive' for row in rows):,})"
    )
    print("속성별:", ", ".join(f"{name} {count}" for name, count in per_property.most_common()))
    print(f"저장: {QUESTIONS_PATH}")


def search() -> None:
    rows = _read_jsonl(QUESTIONS_PATH)
    if carried := _carry_over_previous_search(rows):
        print(f"질문 문장이 그대로인 {carried:,}개는 이전 검색 결과를 옮겼다", flush=True)
    done = {row["id"] for row in _read_jsonl_if(SEARCH_PATH)}
    todo = [row for row in rows if row["id"] not in done]
    config = ApiConfig.from_env()
    start = time.perf_counter()
    with PgStore.connect(config.pg) as pg:
        service = SearchService.build(config.search, pg, QdrantStore.from_env())
        print(f"검색 준비 {time.perf_counter() - start:.0f}s · 남은 {len(todo):,}", flush=True)
        asyncio.run(_search_all(service, todo))
    print(f"저장: {SEARCH_PATH} ({time.perf_counter() - start:.0f}s)")


def _carry_over_previous_search(rows: Sequence[dict]) -> int:
    """이전 판의 검색 결과 중 질문 문장이 그대로인 것을 새 판으로 옮긴다 (처음 한 번만). 옮긴 수."""
    if SEARCH_PATH.exists() or not PREVIOUS_SEARCH.exists():
        return 0
    old = {row["id"]: row["question"] for row in _read_jsonl(PREVIOUS_QUESTIONS)}
    new = {row["id"]: row["question"] for row in rows}
    kept = [
        {**row, "question": new[row["id"]]}
        for row in _read_jsonl(PREVIOUS_SEARCH)
        if row["id"] in new and old.get(row["id"]) == new[row["id"]]
    ]
    _write_jsonl(SEARCH_PATH, kept)
    return len(kept)


async def _search_all(service: SearchService, rows: Sequence[dict]) -> None:
    # 서비스가 질의를 한 번에 하나씩 돌리므로 순서대로 기다린다. 한 줄씩 이어 써서 멈춰도 이어 간다
    for index, row in enumerate(rows, start=1):
        result = await service.search(row["question"], k=TOP_K)
        hits = [
            {
                "chunk_id": hit.chunk.id,
                "title": hit.chunk.title,
                "section": hit.chunk.section,
                "text": hit.chunk.text,
                "score": round(hit.score, 6),
            }
            for hit in result.hits
        ]
        _append_jsonl(SEARCH_PATH, {"id": row["id"], "question": row["question"], "hits": hits})
        if index % 200 == 0:
            print(f"  {index:,}/{len(rows):,}", flush=True)


def label() -> None:
    rows = {row["id"]: row for row in _read_jsonl(QUESTIONS_PATH)}
    templates = {template.pid: template for template in PROPERTIES}
    labeled, counts = [], Counter()
    for searched in _read_jsonl(SEARCH_PATH):
        row = rows[searched["id"]]
        hits = [(hit["title"], hit["text"]) for hit in searched["hits"]]
        if row["kind"] == "sensitive":
            subject = normalize_title(row["subject"])
            ranks = [
                rank for rank, (title, _) in enumerate(hits, 1) if normalize_title(title) == subject
            ]
            result = (
                ("sensitive", "subject_retrieved", ranks[0])
                if ranks
                else ("dropped", "subject_not_retrieved", None)
            )
        else:
            fact = Fact(
                row["qid"],
                row["subject"],
                row["pid"],
                tuple(row["answers"]),
                row["strict"],
                year_only=row.get("year_only", False),
            )
            question = FactQuestion(row["id"], row["question"], fact, templates[row["pid"]])
            found = label_fact(question, hits)
            result = (found.label, found.reason, found.rank)
        labeled.append(
            {
                **row,
                "label": result[0],
                "reason": result[1],
                "rank": result[2],
                "hits": searched["hits"],
            }
        )
        counts[(row.get("property", "민감"), result[0], result[1])] += 1
    _write_jsonl(LABELED_PATH, labeled)
    totals = Counter(row["label"] for row in labeled)
    print(
        f"라벨 {len(labeled):,}: "
        + ", ".join(f"{name} {count}" for name, count in totals.most_common())
    )
    reasons = Counter((row["label"], row["reason"]) for row in labeled)
    print(
        "이유:",
        ", ".join(f"{name}/{reason} {count}" for (name, reason), count in reasons.most_common()),
    )
    by_property = Counter()
    for (prop, name, _), count in counts.items():
        by_property[(prop, name)] += count
    for prop in dict.fromkeys(prop for prop, _ in by_property):
        cells = ", ".join(
            f"{name} {by_property[(prop, name)]}" for p, name in by_property if p == prop
        )
        print(f"  {prop}: {cells}")
    print(f"저장: {LABELED_PATH}")


def pilot() -> None:
    rows = _read_jsonl(LABELED_PATH)
    rng = random.Random(SEED)
    for name in ("answerable", "no_fact", "sensitive", "dropped"):
        group = [row for row in rows if row["label"] == name]
        print(f"\n===== {name} ({len(group)})")
        for row in rng.sample(group, min(PILOT_PER_LABEL, len(group))):
            print(
                f"- [{row.get('property', '민감')}] {row['question']}  → {row['answers'][:3]}  ({row['reason']}, rank {row['rank']})"
            )
            if row["rank"]:
                text = row["hits"][row["rank"] - 1]["text"].replace("\n", " ")
                print(f"    {text[:160]}")


def build(out_dir: Path) -> None:
    rows = _read_jsonl(LABELED_PATH)
    rng = random.Random(SEED)
    used: set[str] = set()  # 한 주제(QID)에서 문항 하나 — 같은 문서를 두 번 묻지 않는다
    items, pools, by_property = [], {}, {}
    for label_name, (kind, size) in STAGE1.items():
        group = [row for row in rows if row["label"] == label_name]
        picked = _round_robin(group, size, used, rng, key=_stratum)
        items += [_set_item(kind, row) for row in picked]
        pools[kind] = len(group)
        by_property[kind] = dict(Counter(_stratum(row) for row in picked).most_common())
    settings = _search_settings()
    version = f"f4-longtail-{_digest({'search': settings, 'rules': SET_RULES, 'stage1': STAGE1})}"
    path = out_dir / version / "items.jsonl"
    _write_jsonl(path, items)
    meta = {
        "version": version,
        "created_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "search": settings,
        "seed": SEED,
        "counts": dict(Counter(item["kind"] for item in items)),
        "pools": pools,
        "by_property": by_property,
        "note": "Wikidata 템플릿 질문 — KorQuAD 자연 질문보다 쉽다. 라벨 노이즈는 50개 사람 검수로 잰다. "
        "남은 문항(같은 주제 제외)은 2단계 결선 몫",
        # v1(f4-longtail-170f9cf4) 사람 검수 — v2 규칙은 여기서 나온 오류 꼴을 고쳤다. v2 노이즈는 다시 재야 안다
        "review_v1": {
            "file": "data/bench/longtail-v1-review-checked.md (위키 레포, 로컬)",
            "answerable_wrong": "4/25 (직위 · 승격일 · 다른 회사 설립 · 후궁)",
            "no_fact_wrong": "3/25 (도입부 괄호 연도 2 · 무슬림)",
            "unnatural_question": "7/50 (괄호 수식어 · 조사 · 본부 · 자식 · 믿은)",
        },
    }
    (path.parent / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"세트 {version}: {meta['counts']} (판정 풀 {pools})")
    for kind, counts in by_property.items():
        print(f"  {kind}: " + ", ".join(f"{name} {count}" for name, count in counts.items()))
    print(f"저장: {path}")


def _stratum(row: dict) -> str:
    """돌아가며 고를 층 — 속성 이름, 민감 질문은 질문 꼴 (`…-sensitive{n}`)."""
    return row.get("property") or row["id"].rsplit("-", 1)[-1]


def _round_robin(
    group: Sequence[dict],
    size: int,
    used: set[str],
    rng: random.Random,
    *,
    key: Callable[[dict], str],
) -> list[dict]:
    """층마다 섞어 두고 층을 돌아가며 하나씩 size 개까지. 이미 쓴 주제는 건너뛰고, 고른 주제는 used 에 더한다."""
    by_key: dict[str, list[dict]] = defaultdict(list)
    for row in group:
        by_key[key(row)].append(row)
    for rows in by_key.values():
        rows.sort(key=lambda row: row["id"])
        rng.shuffle(rows)
    picked: list[dict] = []
    while len(picked) < size and any(by_key.values()):
        for name in sorted(by_key):
            rows = by_key[name]
            while rows and rows[-1]["qid"] in used:
                rows.pop()
            if rows and len(picked) < size:
                row = rows.pop()
                picked.append(row)
                used.add(row["qid"])
    return picked


def _set_item(kind: str, row: dict) -> dict:
    subject = normalize_title(row["subject"])
    # 청크 id 는 "{문서 id}-{순번}" — 순번 0 이 도입부. 목록 청크만 검색된 사실 없음은 검색 누락에 가깝다 (사람 검수 26 · 28번)
    lead = any(
        normalize_title(hit["title"]) == subject and hit["chunk_id"].endswith("-0")
        for hit in row["hits"]
    )
    return {
        "id": f"{kind}:{row['id']}",
        "kind": kind,
        "question": row["question"],
        "answers": row["answers"],
        "gold_title": row["subject"],
        "context": [
            {
                "rank": rank,
                "chunk_id": hit["chunk_id"],
                "title": hit["title"],
                "section": hit["section"],
                "text": hit["text"],
                "score": hit["score"],
            }
            for rank, hit in enumerate(row["hits"], start=1)
        ],
        "strata": {
            "backlinks": row["backlinks"],
            "answer_shape": row.get("shape", "other"),
            "property": _stratum(row),
            "answer_rank": row["rank"] if kind == "longtail" else None,
            "subject_lead_in_context": lead,
        },
    }


def _search_settings() -> dict:
    settings = ApiConfig.from_env().search
    return {
        "collection": settings.collection,
        "model": settings.model,
        "candidates": settings.candidates,
        "rrf_k": settings.rrf_k,
        "rerank_candidates": settings.rerank_candidates,
        "reranker_model": settings.reranker_model,
        "top_k": TOP_K,
    }


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:8]


def _question_row(question: FactQuestion, backlinks: int) -> dict:
    fact, template = question.fact, question.template
    return {
        "id": question.id,
        "kind": "fact",
        "qid": fact.qid,
        "subject": fact.subject,
        "pid": fact.pid,
        "property": template.name,
        "question": question.question,
        "answers": list(fact.answers),
        "strict": fact.strict,
        "year_only": fact.year_only,
        "shape": template.shape,
        "backlinks": backlinks,
    }


def _get(client: httpx.Client, **params: str) -> dict:
    """wbgetentities 한 번. 429 · 5xx 는 기다렸다 다시 (MAX_RETRIES 번).

    maxlag 는 걸지 않는다 — 편집 봇이 서버가 밀릴 때 물러나라는 장치인데, 읽기만 하는 이 질의에서도
    검색 서비스(WDQS) 지연(몇 초 ~ 몇 시간)에 걸려 전부 거절됐다 (2026-10-24). 요청은 하나씩 간격을 둔다.
    """
    # sitefilter: 다른 언어 위키 sitelink 는 받지 않는다 (응답이 작아진다)
    query = {"action": "wbgetentities", "format": "json", "sitefilter": "kowiki", **params}
    for attempt in range(1, MAX_RETRIES + 1):
        time.sleep(REQUEST_INTERVAL)
        response = client.get(API, params=query)
        if response.status_code == httpx.codes.TOO_MANY_REQUESTS or response.is_server_error:
            time.sleep(float(response.headers.get("Retry-After", 10 * attempt)))
            continue
        response.raise_for_status()
        body = response.json()
        if "error" in body:
            raise RuntimeError(f"wbgetentities error: {body['error']}")
        return body
    raise RuntimeError(f"wbgetentities failed after {MAX_RETRIES} attempts")


def _batches(values: Sequence[str]) -> Iterator[list[str]]:
    for start in range(0, len(values), BATCH):
        yield list(values[start : start + BATCH])


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _read_jsonl_if(path: Path) -> list[dict]:
    return _read_jsonl(path) if path.exists() else []


def _write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
