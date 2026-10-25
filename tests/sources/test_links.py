from playground_wiki.sources import BacklinkCounter, article_links, normalize_title, redirect_target


def test_normalize_title_matches_mediawiki_link_forms():
    assert normalize_title(" 서울_특별시  ") == "서울 특별시"
    assert normalize_title(":분류:도시") == "분류:도시"
    assert normalize_title("iPhone") == "IPhone"  # 첫 글자는 대소문자를 가리지 않는다


def test_article_links_takes_targets_once_without_label_or_section():
    text = "[[서울특별시|서울]]은 [[대한민국#역사|한국]]의 수도. [[서울특별시]] [[분류:도시]] [[|]]"

    assert article_links(text) == {"서울특별시", "대한민국", "분류:도시"}


def test_counter_follows_redirects_and_counts_each_page_once():
    counter = BacklinkCounter({"서울시": "서울특별시", "한양": "서울시"})

    counter.add("대한민국", "[[서울시]] · [[서울특별시]] · [[한양]]")  # 셋 다 같은 문서 → 1
    counter.add("조선", "도읍은 [[한양]]")  # 넘겨주기 사슬
    counter.add("서울특별시", "[[서울시]]는 자기 자신")  # 자기 링크는 뺀다
    counter.add("부산광역시", "[[분류:도시]] [[없는 문서]]")

    assert counter.counts(["서울특별시", "대한민국", "부산광역시"]) == {
        "서울특별시": 2,
        "대한민국": 0,
        "부산광역시": 0,
    }


def test_counter_stops_on_redirect_loop():
    counter = BacklinkCounter({"가": "나", "나": "가"})

    assert counter.resolve("가") in {"가", "나"}


def test_redirect_target_reads_both_markers_and_drops_section():
    assert redirect_target("#넘겨주기 [[서울특별시#역사]]") == "서울특별시"
    assert redirect_target("  #REDIRECT [[Seoul]]") == "Seoul"
    assert redirect_target("서울특별시는 [[대한민국]]의 수도") is None
