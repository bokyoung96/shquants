from __future__ import annotations

from collections.abc import Iterable

import pytest

from kind.models import Disclosure, ParsedPage
from kind.parser import KindSchemaError, parse_disclosure_page


PROVISIONAL_TITLE = "영업 (잠정) 실적 (공정공시)"


def _row(
    *,
    time: str = "08:05",
    company: str = "SK하이닉스",
    title: str = PROVISIONAL_TITLE,
    submitter: str = "SK하이닉스",
    issuer_onclick: str | None = "companysummary_open('00066'); return false;",
    disclosure_onclick: str | None = (
        "openDisclsViewer('20240425000004','')"
    ),
) -> str:
    company_html = company
    if issuer_onclick is not None:
        company_html = (
            f'<a href="#" onclick="{issuer_onclick}"> {company} </a>'
        )

    title_html = title
    if disclosure_onclick is not None:
        title_html = (
            f'<a href="#" onclick="{disclosure_onclick}"> {title} </a>'
        )

    return f"""
        <tr>
          <td> {time} </td>
          <td> {company_html} </td>
          <td> {title_html} </td>
          <td> {submitter} </td>
          <td>-</td>
        </tr>
    """


def _market_stat_row() -> str:
    return """
        <tr>
          <td>18:00</td>
          <td>유가증권시장</td>
          <td>투자자별 매매동향</td>
          <td>한국거래소</td>
          <td>-</td>
        </tr>
    """


def _page_html(
    rows: Iterable[str] = (),
    *,
    pagination: str = "",
) -> str:
    return f"""
        <html><body>
          <table class="list type-00 mt10"
                 summary="시간, 회사명, 공시제목, 제출인, 차트/주가">
            <tbody>{''.join(rows)}</tbody>
          </table>
          {pagination}
        </body></html>
    """


def test_parse_disclosure_page_extracts_valid_row_and_pagination() -> None:
    html = _page_html(
        [_row(), _market_stat_row()],
        pagination="""
            <a href="#" onclick="fnPageGo('2')">2</a>
            <a href="#" onclick="fnPageGo('5')">5</a>
        """,
    )

    actual = parse_disclosure_page(
        html,
        announcement_date="2024-04-25",
        page=1,
    )

    assert actual == ParsedPage(
        disclosures=(
            Disclosure(
                announcement_date="2024-04-25",
                time="08:05",
                company="SK하이닉스",
                title=PROVISIONAL_TITLE,
                submitter="SK하이닉스",
                issuer_id="00066",
                receipt_id="20240425000004",
                page=1,
                position=1,
            ),
        ),
        total_pages=5,
    )


def test_parse_disclosure_page_requires_disclosure_table() -> None:
    with pytest.raises(KindSchemaError, match="table"):
        parse_disclosure_page(
            "<html><body><p>no disclosures</p></body></html>",
            announcement_date="2024-04-25",
            page=1,
        )


@pytest.mark.parametrize(
    "cells",
    [
        "<td>08:05</td><td>company</td><td>title</td><td>submitter</td>",
        (
            "<td>08:05</td><td>company</td><td>title</td>"
            "<td>submitter</td><td>-</td><td>extra</td>"
        ),
        (
            "<td>08:05</td><td><table><tr><td>nested</td></tr></table></td>"
            "<td>title</td><td>submitter</td>"
        ),
    ],
)
def test_parse_disclosure_page_requires_exactly_five_direct_cells(
    cells: str,
) -> None:
    html = _page_html([f"<tr>{cells}</tr>"])

    with pytest.raises(KindSchemaError, match="5.*cells"):
        parse_disclosure_page(
            html,
            announcement_date="2024-04-25",
            page=1,
        )


@pytest.mark.parametrize("time", ["8:05", "0٨:0٥", "08：05", "24:00"])
def test_parse_disclosure_page_rejects_noncanonical_company_time(
    time: str,
) -> None:
    with pytest.raises(KindSchemaError, match="time"):
        parse_disclosure_page(
            _page_html([_row(time=time)]),
            announcement_date="2024-04-25",
            page=1,
        )


@pytest.mark.parametrize(
    ("issuer_onclick", "disclosure_onclick"),
    [
        (None, "openDisclsViewer('20240425000004','')"),
        ("companysummary_open('00066'); return false;", None),
    ],
)
def test_provisional_row_requires_both_links(
    issuer_onclick: str | None,
    disclosure_onclick: str | None,
) -> None:
    with pytest.raises(KindSchemaError, match="provisional.*link"):
        parse_disclosure_page(
            _page_html(
                [
                    _row(
                        issuer_onclick=issuer_onclick,
                        disclosure_onclick=disclosure_onclick,
                    )
                ]
            ),
            announcement_date="2024-04-25",
            page=1,
        )


@pytest.mark.parametrize(
    "title",
    [
        "연결재무제표기준영업(잠정)실적(공정공시)",
        "영업(잠정)실적(공정공시)(자회사의 주요경영사항)",
        (
            "연결재무제표기준 영업 (잠정) 실적 (공정공시)  "
            "(자회사의 주요경영사항)"
        ),
    ],
)
@pytest.mark.parametrize("missing_link", ["company", "disclosure"])
def test_provisional_title_family_requires_both_links(
    title: str,
    missing_link: str,
) -> None:
    issuer_onclick: str | None = (
        "companysummary_open('00066'); return false;"
    )
    disclosure_onclick: str | None = (
        "openDisclsViewer('20240425000004','')"
    )
    if missing_link == "company":
        issuer_onclick = None
    else:
        disclosure_onclick = None

    with pytest.raises(KindSchemaError, match="provisional.*link"):
        parse_disclosure_page(
            _page_html(
                [
                    _row(
                        title=title,
                        issuer_onclick=issuer_onclick,
                        disclosure_onclick=disclosure_onclick,
                    )
                ]
            ),
            announcement_date="2024-04-25",
            page=1,
        )


@pytest.mark.parametrize(
    "onclick",
    [
        "companysummary_open('0006'); return false;",
        "companysummary_open('000660'); return false;",
        "companysummary_open('00-66'); return false;",
        "companysummary_open('00066'",
        (
            "companysummary_open('00066'); "
            "companysummary_open('00067')"
        ),
    ],
)
def test_parse_disclosure_page_rejects_malformed_issuer_handler(
    onclick: str,
) -> None:
    with pytest.raises(KindSchemaError, match="issuer"):
        parse_disclosure_page(
            _page_html([_row(issuer_onclick=onclick)]),
            announcement_date="2024-04-25",
            page=1,
        )


@pytest.mark.parametrize(
    "onclick",
    [
        "openDisclsViewer('2024042500000','')",
        "openDisclsViewer('202404250000044','')",
        "openDisclsViewer('2024A425000004','')",
        "openDisclsViewer('20240425000004')",
        "openDisclsViewer('20240425000004','",
        (
            "openDisclsViewer('20240425000004',''); "
            "openDisclsViewer('20240425000005','')"
        ),
    ],
)
def test_parse_disclosure_page_rejects_malformed_receipt_handler(
    onclick: str,
) -> None:
    with pytest.raises(KindSchemaError, match="receipt"):
        parse_disclosure_page(
            _page_html([_row(disclosure_onclick=onclick)]),
            announcement_date="2024-04-25",
            page=1,
        )


@pytest.mark.parametrize(
    ("issuer_onclick", "disclosure_onclick"),
    [
        (
            "prefixcompanysummary_open('00066')suffix",
            "openDisclsViewer('20240425000004','')",
        ),
        (
            "companysummary_open('00066'); return false;",
            "prefixopenDisclsViewer('20240425000004','')suffix",
        ),
    ],
)
def test_parse_disclosure_page_rejects_handler_surrounding_junk(
    issuer_onclick: str,
    disclosure_onclick: str,
) -> None:
    with pytest.raises(KindSchemaError):
        parse_disclosure_page(
            _page_html(
                [
                    _row(
                        issuer_onclick=issuer_onclick,
                        disclosure_onclick=disclosure_onclick,
                    )
                ]
            ),
            announcement_date="2024-04-25",
            page=1,
        )


@pytest.mark.parametrize(
    ("issuer_onclick", "disclosure_onclick"),
    [
        (
            "companysummary_open('00066'); return false;",
            "openDisclsViewer('20240425000004','')",
        ),
        (
            "  companysummary_open('00066') ;  ",
            "  openDisclsViewer('20240425000004' , 'detail') ;  ",
        ),
    ],
)
def test_parse_disclosure_page_accepts_observed_complete_handlers(
    issuer_onclick: str,
    disclosure_onclick: str,
) -> None:
    actual = parse_disclosure_page(
        _page_html(
            [
                _row(
                    issuer_onclick=issuer_onclick,
                    disclosure_onclick=disclosure_onclick,
                )
            ]
        ),
        announcement_date="2024-04-25",
        page=1,
    )

    assert actual.disclosures[0].issuer_id == "00066"
    assert actual.disclosures[0].receipt_id == "20240425000004"


@pytest.mark.parametrize(
    "pagination",
    [
        '<a onclick="fnPageGo(\'0\')">0</a>',
        '<a onclick="fnPageGo(\'1000\')">1000</a>',
        '<a onclick="fnPageGo(\'１２\')">12</a>',
        '<script>fnPageGo(pageNumber)</script>',
    ],
)
def test_invalid_page_handler_does_not_create_false_total_page(
    pagination: str,
) -> None:
    with pytest.raises(KindSchemaError, match="pagination"):
        parse_disclosure_page(
            _page_html(pagination=pagination),
            announcement_date="2024-04-25",
            page=1,
        )


def test_parse_disclosure_page_rejects_total_below_current_page() -> None:
    with pytest.raises(KindSchemaError, match="current page"):
        parse_disclosure_page(
            _page_html(pagination='<a onclick="fnPageGo(\'2\')">2</a>'),
            announcement_date="2024-04-25",
            page=3,
        )


@pytest.mark.parametrize("page", [True, False, 0, 1000, -1, 1.0, "1"])
def test_parse_disclosure_page_rejects_invalid_caller_page(
    page: object,
) -> None:
    with pytest.raises(KindSchemaError, match="page"):
        parse_disclosure_page(
            _page_html(),
            announcement_date="2024-04-25",
            page=page,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("rows", [[], [_market_stat_row()]])
def test_empty_or_noncompany_table_is_valid(rows: list[str]) -> None:
    actual = parse_disclosure_page(
        _page_html(rows),
        announcement_date="2024-04-25",
        page=1,
    )

    assert actual == ParsedPage(disclosures=(), total_pages=1)


@pytest.mark.parametrize(
    "title",
    ["주주총회소집공고", "영업실적 등에 대한 전망"],
)
def test_nonprovisional_row_without_required_links_is_skipped(
    title: str,
) -> None:
    actual = parse_disclosure_page(
        _page_html(
            [
                _row(
                    title=title,
                    issuer_onclick=None,
                    disclosure_onclick=None,
                )
            ]
        ),
        announcement_date="2024-04-25",
        page=1,
    )

    assert actual.disclosures == ()


def test_multiple_valid_rows_preserve_dom_order_and_positions() -> None:
    html = _page_html(
        [
            _row(),
            _row(
                time="09:17",
                company="삼성전자",
                title="주주총회소집공고",
                submitter="삼성전자",
                issuer_onclick="companysummary_open('00593')",
                disclosure_onclick=(
                    "openDisclsViewer('20240425000005','detail')"
                ),
            ),
        ]
    )

    actual = parse_disclosure_page(
        html,
        announcement_date="2024-04-25",
        page=1,
    )

    assert [item.company for item in actual.disclosures] == [
        "SK하이닉스",
        "삼성전자",
    ]
    assert [item.receipt_id for item in actual.disclosures] == [
        "20240425000004",
        "20240425000005",
    ]
    assert [item.position for item in actual.disclosures] == [1, 2]
