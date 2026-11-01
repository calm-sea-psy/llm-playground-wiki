"""스크립트가 같이 쓰는 데이터 위치 · 세트 이름. 경로는 모두 레포 루트 기준 상대 경로다 (스크립트를 레포 루트에서 돌린다).

data/ 와 LLM 레포 bench/data/ 는 git 제외 — KorQuAD 가 CC BY-ND 라 질문 · 문맥 원문은 로컬에만 둔다.
한 스크립트만 쓰는 경로는 그 스크립트에 둔다.
"""

from pathlib import Path

SUBSET = Path("data/parsed/kowiki-subset-10000")
"""KorQuAD 문서를 담은 부분 코퍼스 — subset.json: KorQuAD dev 경로 · 제목 보정표 (D4)."""
PARSED = Path("data/parsed/kowiki")
"""전체 덤프 파싱 결과 (docs-*.parquet · redirects.parquet) — 정답 문자열이 지금 문서에 남았는지 대조한다."""
BACKLINKS = Path("data/bench/backlinks-20261001.parquet")
"""문서별 역링크 수 (scripts/count_backlinks.py) — 롱테일 문서를 고른다."""
OFFWIKI_PATH = Path("data/eval/offwiki-v1.jsonl")
"""위키 밖 질문 세트 (직접 만든 세트)."""
PARAPHRASE_PATH = Path("data/bench/paraphrase-v1.jsonl")
"""KorQuAD 질문을 Claude 로 고쳐 쓴 패러프레이즈 (scripts/paraphrase_korquad.py 결과)."""
F9_SAMPLE_PATH = Path("data/eval/f9-sample-v1.jsonl")
"""F9 표본 (KorQuAD 370 + 위키 밖 100) — scripts/eval_answer.py 의 SAMPLE_PATH 와 같은 파일."""
COMPOUND_PATH = Path("tests/evaluation/compound-v1.jsonl")
"""복합 질문 세트 v1 (50, 커밋한다 — 덤프 근거 문장 한 줄 이하만 담는다)."""
AGENT_RUNS_DIR = Path("data/eval/agent")
"""단발 대 에이전트 비교 기록 (scripts/eval_agent.py run) — {세트}-{모드}[-{꼬리표}].jsonl."""
F9_ANSWERS_PATH = Path("data/eval/f9-answers-v1.jsonl")
"""F9 표본을 답변 경로로 돌린 기준 기록 (scripts/eval_answer.py run). 꼬리표 기록은 `answer_run.tagged`."""
HARMFUL_SET_PATH = Path("data/eval/harmful-v1.jsonl")
"""유해 정보 세트 — 유해 질문 · 같은 주제의 대조 질문 짝 (scripts/judge_partial.py check)."""
HARMFUL_ROUTE_ANSWERS_PATH = Path("data/eval/harmful-answers-v1-route.jsonl")
"""유해 정보 세트를 운영 경로(route)로 돌린 기준 기록 (scripts/judge_partial.py answer). 꼬리표 기록은 `answer_run.tagged`."""
PROCEDURE_ROUTE_ANSWERS_PATH = Path("data/eval/procedure-answers-v1-route.jsonl")
"""무해 절차 세트를 운영 경로로 돌린 기록 — 늘 꼬리표를 붙여 쓴다 (`answer --set procedure --tag T`)."""
PARTIAL_JUDGED_TAGGED_PATH = Path("data/eval/partial-judged-v1-tagged.jsonl")
"""꼬리표 기록의 partial 판정 (scripts/judge_partial.py judge --tag) — 꼬리표 없는 기록의 판정(partial-judged-v1.jsonl)과
따로 둔다. 줄의 run 은 `실행-꼬리표` (예 f9-partial-base-1)."""
REPORT_DIR = Path("reports")
"""리포트를 쓰는 곳 (커밋한다)."""
BENCH_DATA = Path("../llm-playground-llm/bench/data")
"""LLM 레포의 세트 · F4 실행 폴더 (레포 루트에서 돌릴 때). worktree 에서는 스크립트 인자로 절대 경로를 준다."""

KORQUAD_SET = "f4-korquad-9014dd9e"
"""scripts/build_bench_set.py 가 만든 1단계 KorQuAD 세트 (답 있음 · 문서 없음)."""
KORQUAD_SPARE_SET = "f4-korquad-spare-9014dd9e"
"""같은 검색 설정으로 만든 KorQuAD 예비 세트."""
EXTRA_SET = "f4-extra-v2"
"""scripts/build_bench_extra.py 가 만든 세트 (반사실 · 민감 주제 · 인젝션). v2: 반사실 생성 규칙 보강."""
LONGTAIL_STAGE1_SET = "f4-longtail-cec763d5"
"""1단계 벤치(run 2026-10-24-f4-stage1)에 쓴 롱테일 세트."""
LONGTAIL_STAGE2_SET = "f4-longtail-stage2-ba21dfc5"
"""F4 결선 롱테일 세트 (2단계)."""
