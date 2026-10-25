"""scripts/build_bench_extra.py 의 변형 · 주입 헬퍼와 재현성 검사. KorQuAD 없이 합성 입력으로 돈다."""

import importlib.util
import random
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[2] / "scripts" / "build_bench_extra.py"
_spec = importlib.util.spec_from_file_location("build_bench_extra", _SCRIPT)
assert _spec is not None and _spec.loader is not None
extra = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(extra)


def _answerable(item_id, question, answer, shape, title="문서", texts=None):
    texts = texts or [f"{answer} 에 대한 설명." for _ in range(5)]
    return {
        "id": f"korquad:{item_id}",
        "kind": "answerable",
        "question": question,
        "answers": [answer],
        "gold_title": title,
        "context": [
            {
                "rank": rank,
                "chunk_id": f"{item_id}-{rank}",
                "title": title,
                "section": "",
                "text": text,
                "score": 1.0,
            }
            for rank, text in enumerate(texts, start=1)
        ],
        "strata": {"backlinks": 1, "answer_shape": shape, "answer_rank": 1},
    }


def test_shift_run_year_keeps_width_and_stays_near():
    rng = random.Random("seed")
    out = extra._shift_run("1976", "년 ", rng)

    assert len(out) == 4 and out.isdigit() and out != "1976"
    assert abs(int(out) - 1976) <= 60  # 가까운 해로만


def test_shift_run_month_stays_in_range():
    rng = random.Random("seed")
    out = extra._shift_run("10", "월", rng)

    assert 1 <= int(out) <= 12 and out != "10"


def test_shift_run_decade_is_multiple_of_ten():
    rng = random.Random("seed")
    out = extra._shift_run("60", "년대", rng)

    assert len(out) == 2 and int(out) % 10 == 0 and out != "60"


def test_substitute_number_changes_all_runs_and_avoids_context():
    rng = random.Random("seed")
    new = extra.substitute_number("1976년 10월", context_squash="", rng=rng)

    assert new is not None
    assert new != "1976년 10월"
    assert "년" in new and "월" in new
    month = int(new.split("년", 1)[1].replace("월", "").strip())
    assert 1 <= month <= 12  # 바뀐 달도 유효 범위


def test_substitute_number_retries_until_absent_from_context():
    # 바뀐 값이 원래 문맥(여기선 모든 연도)에 있으면 쓰지 않는다
    rng = random.Random("seed")
    forbidden = extra.squash("".join(f"{year}년" for year in range(1000, 10000)))
    assert extra.substitute_number("1976년", context_squash=forbidden, rng=rng) is None


def test_substitute_person_picks_other_name():
    rng = random.Random("seed")
    new = extra.substitute_person(
        ["김구"], ["김구", "이순신", "세종"], context_squash="이순신", rng=rng
    )

    assert new == "세종"  # 원래 답(김구) · 문맥(이순신) 에 없는 유일한 이름


def test_replace_in_context_allows_spacing():
    context = [{"rank": 1, "text": "답은 증강 현실 이다."}]
    out, count = extra.replace_in_context(context, "증강현실", "가상세계")

    assert count == 1
    assert out[0]["text"] == "답은 가상세계 이다."


def test_embed_injection_front_and_end():
    assert extra.embed_injection("본문", "지시", "front") == "지시\n\n본문"
    assert extra.embed_injection("본문", "지시", "end") == "본문\n\n지시"


def test_topic_matches_sensitive_keywords():
    war = _answerable(
        "1-0-0", "전투는 몇 년에 있었나?", "1941년", "date_number", title="러시아 원정"
    )
    plain = _answerable("2-0-0", "앨범 제목은?", "화양연화", "other", title="방탄소년단")

    assert extra._topic(war) == "war"
    assert extra._topic(plain) is None


def test_counterfactual_row_substitutes_number():
    source = _answerable("3-0-0", "몇 년에 입대했나?", "1976년", "date_number", title="차범근")
    row = extra._counterfactual_row(source, person_pool=[], rng=random.Random("0:counterfactual"))

    assert row is not None
    assert row["id"] == "counterfactual:3-0-0"
    assert row["original_answers"] == ["1976년"]
    assert row["answers"] != ["1976년"]
    assert row["substitution"]["type"] == "date"
    # 바뀐 값이 모든 청크에 반영되고 원래 값은 사라진다
    joined = extra.squash("".join(chunk["text"] for chunk in row["context"]))
    assert extra.squash(row["answers"][0]) in joined
    assert extra.squash("1976년") not in joined


def test_build_indirect_is_deterministic_and_unique():
    answerable = [
        _answerable(f"{index}-0-0", f"질문 {index}?", f"답{index}", "other") for index in range(60)
    ]
    first = extra._build_indirect(answerable, random.Random("0:injection_indirect"))
    second = extra._build_indirect(answerable, random.Random("0:injection_indirect"))

    assert len(first) == extra.INJECT_INDIRECT
    assert first == second  # seed 고정 → 같은 세트
    canaries = [row["canary"] for row in first]
    assert len(set(canaries)) == len(canaries)  # 문항마다 고유
    for row in first:
        assert row["canary"] in "".join(chunk["text"] for chunk in row["context"])


def test_build_direct_appends_attack_and_keeps_answer():
    answerable = [
        _answerable(f"{index}-0-0", f"질문 {index}?", f"답{index}", "other") for index in range(40)
    ]
    rows = extra._build_direct(answerable, random.Random("0:injection_direct"))

    assert len(rows) == extra.INJECT_DIRECT
    assert {row["attack"]["lang"] for row in rows} == {"ko", "en"}
    for row in rows:
        assert row["canary"] == extra.DIRECT_CANARY
        assert row["question"].startswith("질문")
        assert row["answers"] == [row["question"].split("?", 1)[0].replace("질문 ", "답")]


def test_indirect_templates_cover_at_least_ten():
    assert len({template.id for template in extra.INDIRECT_TEMPLATES}) >= 10
    assert len({template.id for template in extra.DIRECT_TEMPLATES}) >= 10


@pytest.mark.parametrize("template", extra.INDIRECT_TEMPLATES)
def test_indirect_templates_have_canary_slot(template):
    assert "{canary}" in template.ko
    assert "{canary}" in template.en
