from playground_wiki.ingest import WikiCleaner
from playground_wiki.ingest.cleaner import clean_text


def _clean(wikitext: str):
    return WikiCleaner().clean(wikitext)


def test_clean_splits_sections_with_lead_at_level_zero():
    page = _clean("'''서울'''은 수도이다.\n== 역사 ==\n조선의 수도였다.\n=== 근대 ===\n개항했다.")

    assert [(s.level, s.title, s.text) for s in page.sections] == [
        (0, "", "서울은 수도이다."),
        (2, "역사", "조선의 수도였다."),
        (3, "근대", "개항했다."),
    ]
    assert page.text == "서울은 수도이다.\n\n조선의 수도였다.\n\n개항했다."


def test_clean_drops_tail_sections_with_their_subsections():
    page = _clean(
        "본문.\n== 각주 ==\n각주 내용\n=== 내용주 ===\n하위 내용\n== 같이 보기 ==\n* 링크\n"
    )

    assert [s.title for s in page.sections] == [""]


def test_clean_keeps_section_after_dropped_section():
    page = _clean("본문.\n== 참고 문헌 ==\n책\n== 생애 ==\n태어났다.")

    assert [s.title for s in page.sections] == ["", "생애"]


def test_clean_moves_categories_out_of_text():
    page = _clean("본문이다.\n[[분류:대한민국의 도시|서울]]\n[[Category:Capitals]]")

    assert page.categories == ("대한민국의 도시", "Capitals")
    assert page.text == "본문이다."


def test_clean_removes_media_and_interlanguage_links_but_keeps_link_text():
    page = _clean(
        "[[파일:Seoul.jpg|섬네일|250px|서울 전경]][[서울특별시|서울]]은 [[수도]]이다.[[en:Seoul]]"
    )

    assert page.text == "서울은 수도이다."


def test_clean_removes_refs_and_tables():
    page = _clean('수도이다.<ref name="a">출처 문장</ref>\n{| class="wikitable"\n|-\n| 셀\n|}\n끝.')

    assert page.text == "수도이다.\n\n끝."  # 표가 있던 자리는 문단 경계로 남는다


def test_clean_hides_rest_of_document_after_unclosed_comment():
    page = _clean("본문이다.<!-- 숨김\n\n숨김\n== 역사 ==\n숨김\n[[분류:숨김]]")

    assert [(s.title, s.text) for s in page.sections] == [("", "본문이다.")]
    assert page.categories == ()


def test_clean_lets_unclosed_ref_swallow_text_up_to_next_ref_close():
    page = _clean("앞<ref>숨김\n\n숨김<ref>출처</ref> 뒤.")

    assert page.text == "앞 뒤."


def test_clean_hides_unclosed_hidden_span_to_paragraph_end():
    page = _clean(
        '앞 <span style="display:none">숨김\n숨김\n\n다음 문단.\n'
        '줄 <span style="display:none">숨김\n* 목록\n== 역사 ==\n본문.'
    )

    assert page.text == "앞\n\n다음 문단.\n줄\n목록\n\n본문."


def test_clean_hides_rest_of_document_after_unclosed_hidden_div():
    page = _clean('앞.\n<div style="display:none">\n숨김 <div>숨김</div> 숨김\n== 역사 ==\n숨김')

    assert [(s.title, s.text) for s in page.sections] == [("", "앞.")]


def test_clean_keeps_text_after_unclosed_hidden_div_in_dropped_markup():
    # 위키는 표 칸 · 틀 인자 끝에서 div 를 닫는다. 표 · 틀은 통째로 지우니 뒤 본문을 자르면 안 된다
    page = _clean(
        '{|\n| <div style="display:none">칸\n|}\n'
        '{{정보상자 사람|이름=<div style="display:none">이름}}\n본문이다.'
    )

    assert page.text == "본문이다."


def test_clean_keeps_hidden_markup_shown_as_text_in_nowiki():
    page = _clean(
        '<nowiki><div style="display:none"></nowiki> 처럼 숨긴다.<nowiki><!--</nowiki> 주석.\n\n다음 문단.'
    )

    assert page.text == "처럼 숨긴다.<!-- 주석.\n\n다음 문단."


def test_clean_rewrites_text_bearing_templates():
    page = _clean(
        "카터({{lang|en|Jimmy Carter}}, {{출생일|1924|10|01}})는 {{USS|폼프렛|SS-391}}에서 "
        "복무했다.{{Sfn|Thomas|1978}}"
    )

    assert page.text == "카터(Jimmy Carter, 1924년 10월 1일)는 USS 폼프렛에서 복무했다."


def test_clean_keeps_lead_parenthesis_templates():
    page = _clean(
        "'''하지원'''({{본명|전해림}}, 1978년 ~ ). "
        "'''장국영'''({{zh|order=t|t=張國榮|s=张国荣|p=Zhāng Guóróng}}, "
        "{{llang|en|Leslie Cheung|레슬리 청}}). "
        "'''대한민국'''({{한자|大韓民國}}). '''보리'''({{학명|Hordeum vulgare}}). "
        "'''로봇'''({{문화어|로보트}}). '''비엣남'''({{vie|v=Việt Nam|hn=越南}}). "
        "'''동인지'''({{Ruby-ja|同人誌|どうじんし}})."
    )

    assert page.text == (
        "하지원(본명 전해림, 1978년 ~ ). 장국영(張國榮, Leslie Cheung). "
        "대한민국(大韓民國). 보리(Hordeum vulgare). 로봇(문화어 로보트). "
        "비엣남(Việt Nam). 동인지(同人誌)."
    )


def test_clean_keeps_more_lead_text_templates():
    page = _clean(
        "'''도시락'''({{중세 국어|도슭}}). "
        "'''푸런'''({{llang|mnc|{{만몽 유니코드|ᠠᡳᠰᡳᠨ}}|Aisin Gioro Pu Ren}}). "
        "'''이삭'''({{히브리어 이름|יִצְחָק|Yitskhak}}, {{transl|grc|''Isaak''}}). "
        "'''분화'''({{linktext|行星|分化}}). '''NSK'''({{langnf|de||뉴 슬로베니안 아트}}). "
        "'''멜니크'''({{llang|uk|{{임시링크|Андрій Мельник|uk|Мельник Андрій}}}}). "
        "'''바쿠'''({{lang|ja|{{ruby|井手上|いでがみ}}}})."
    )

    assert page.text == (
        "도시락(도슭). 푸런(Aisin Gioro Pu Ren). 이삭(יִצְחָק, Isaak). 분화(行星分化). "
        "NSK(뉴 슬로베니안 아트). 멜니크(Андрій Мельник). 바쿠(井手上)."
    )


def test_clean_removes_separators_left_by_dropped_templates():
    page = _clean(
        "'''Xfce'''({{IPA|[ɛks ɛf siː iː]}}, 엑스에프시이). "
        "'''NSK'''({{IPA|de|ˈnɔʏə|lang}}; '''NSK'''; {{IPA|de|x}}). "
        "'''옹프레''' ({{IPA|fr|miʃɛl|lang}}; 1959년 출생). '''빈칸'''({{IPA|x}}; {{IPA|y}})."
    )

    assert page.text == "Xfce(엑스에프시이). NSK(NSK). 옹프레 (1959년 출생). 빈칸."


def test_clean_keeps_latex_thin_space_before_parenthesis():
    page = _clean(r"각가속도는 알파 (<math>{\alpha}\,</math>)로 표시한다.")

    assert page.text == r"각가속도는 알파 ({\alpha}\,)로 표시한다."


def test_clean_rewrites_nested_templates_from_the_inside():
    page = _clean(
        "'''니시토쿄시'''({{llang|ja|{{ruby-ja|西東京市|にしとうきょうし}}}}, {{문화어|니시도꾜시}})는 시이다."
    )

    assert page.text == "니시토쿄시(西東京市, 문화어 니시도꾜시)는 시이다."


def test_clean_keeps_short_inline_math_as_latex():
    page = _clean(
        "어떤 모수 <math>\\theta</math> 로 결정되는 가능도 <math>L( \\theta )</math> 는 다음과 같다."
    )

    assert page.text == "어떤 모수 \\theta 로 결정되는 가능도 L( \\theta ) 는 다음과 같다."


def test_clean_keeps_absolute_value_bars_in_inline_math():
    page = _clean("거리는 <math>\\frac{1+|z|}{1-|z|}</math> 이다.")

    assert page.text == "거리는 \\frac{1+|z|}{1-|z|} 이다."


def test_clean_drops_long_and_block_math():
    long_formula = "\\prod_{i=1}^{n} f(x_i \\mid \\mu, \\sigma^2) = (2\\pi\\sigma^2)^{-n/2} e^{-1}"
    page = _clean(
        f'가능도는 <math>{long_formula}</math> 이다. 블록 <math display="block">x</math> 끝.'
    )

    assert page.text == "가능도는 이다. 블록 끝."


def test_clean_detects_disambiguation_by_template():
    assert _clean("'''소수'''는 다음을 가리킨다.\n{{동음이의}}").is_disambiguation


def test_clean_detects_disambiguation_by_category():
    assert _clean("목록.\n[[분류:동음이의 문서]]").is_disambiguation


def test_clean_text_keeps_letters_and_digits_when_collapsing_spaces():
    # 공백 정규식이 깨져 a · 0 · 2 · b 가 지워졌던 회귀
    nbsp, zero_width_space = chr(0xA0), chr(0x200B)
    text = f"James Earl Carter 1924년 10월 2일 ab{nbsp}b{zero_width_space}c"

    assert clean_text(text) == "James Earl Carter 1924년 10월 2일 ab b c"


def test_clean_text_removes_markup_the_parser_missed():
    text = (
        "__NOTOC__\n"
        "* 목록 항목\n"
        ": 정의 설명\n"
        "! 표 머리\n"
        "|- \n"
        "[[:en:Listen|들어라]] <sub>아래<sub> 문장 [1] .\n"
        '각주<ref name="깨진|저자 = 누구'
    )

    assert clean_text(text) == "목록 항목\n정의 설명\n\n들어라 아래 문장.\n각주"


def test_clean_text_keeps_angle_bracket_titles():
    assert clean_text("소설 <afternoon, a story>(1987)") == "소설 <afternoon, a story>(1987)"
