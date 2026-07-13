from __future__ import annotations

from collections.abc import Iterable

import pytest

from kind.models import Disclosure, ParsedPage
from kind.parser import KindSchemaError, parse_disclosure_page


ANNOUNCEMENT_DATE = "2024-04-25"
PROVISIONAL_TITLE = "영업 (잠정) 실적 (공정공시)"
TABLE_SUMMARY = "시간, 회사명, 공시제목, 제출인, 차트/주가"


def _row(
    *,
    time: str = "08:05",
    company: str = "SK하이닉스",
    title: str = PROVISIONAL_TITLE,
    submitter: str = "SK하이닉스",
    issuer_onclick: str | None = "companysummary_open('00066');return false;",
    disclosure_onclick: str | None = (
        "openDisclsViewer('20240425000004','')"
    ),
    company_html: str | None = None,
    title_html: str | None = None,
) -> str:
    if company_html is None:
        company_html = company
        if issuer_onclick is not None:
            company_html = (
                f'<a href="#" onclick="{issuer_onclick}">{company}</a>'
            )

    if title_html is None:
        title_html = title
        if disclosure_onclick is not None:
            title_html = (
                f'<a href="#" onclick="{disclosure_onclick}">{title}</a>'
            )

    return f"""
        <tr>
          <td>{time}</td>
          <td>{company_html}</td>
          <td>{title_html}</td>
          <td>{submitter}</td>
          <td>-</td>
        </tr>
    """


def _market_stat_row(*, with_receipt: bool = True) -> str:
    title = "투자자별 매매동향"
    if with_receipt:
        title = (
            "<a href=\"#\" "
            "onclick=\"openDisclsViewer('20240425000099','')\">"
            f"{title}</a>"
        )
    return f"""
        <tr>
          <td>18:00</td>
          <td>   </td>
          <td>{title}</td>
          <td>한국거래소</td>
          <td>-</td>
        </tr>
    """


def _paging_html(
    *,
    current: int = 1,
    total: int = 1,
    links: Iterable[int] = (),
    active_html: str | None = None,
    nav_extra: str = "",
    info_html: str | None = None,
) -> str:
    if active_html is None:
        active_html = (
            f'<a class="active" href="#page_link_{current}" '
            f'onclick="return false;">{current}</a>'
        )
    clickable = "".join(
        f'<a href="#page_link_{number}" '
        f'onclick="fnPageGo(\'{number}\');return false;">{number}</a>'
        for number in links
    )
    if info_html is None:
        info_html = (
            '<div class="info type-00">전체 <em>442</em> 건 : '
            f'<strong>{current}</strong>/{total}&nbsp; 페이지</div>'
        )
    return f"""
        <section class="paging-group">
          <div class="paging type-00">{clickable}{active_html}{nav_extra}</div>
          {info_html}
        </section>
    """


def _table_html(
    rows: Iterable[str] = (),
    *,
    summary: str = TABLE_SUMMARY,
    tbody_html: str | None = None,
) -> str:
    if tbody_html is None:
        tbody_html = f"<tbody>{''.join(rows)}</tbody>"
    return f"""
        <section class="scrarea type-00">
          <table class="list type-00 mt10" summary="{summary}">
            {tbody_html}
          </table>
        </section>
    """


def _page_html(
    rows: Iterable[str] = (),
    *,
    current: int = 1,
    total: int = 1,
    links: Iterable[int] = (),
    pagination_html: str | None = None,
    table_html: str | None = None,
    extra_html: str = "",
) -> str:
    if table_html is None:
        table_html = _table_html(rows)
    if pagination_html is None:
        pagination_html = _paging_html(
            current=current,
            total=total,
            links=links,
        )
    return f"<html><body>{table_html}{pagination_html}{extra_html}</body></html>"


def _parse(html: str, *, page: int = 1, date: object = ANNOUNCEMENT_DATE) -> ParsedPage:
    return parse_disclosure_page(
        html,
        announcement_date=date,  # type: ignore[arg-type]
        page=page,
    )


def test_parse_page_one_uses_scoped_info_total_and_skips_market_row() -> None:
    html = _page_html(
        [_row(), _market_stat_row()],
        current=1,
        total=5,
        links=[2, 5],
    )

    actual = _parse(html)

    assert actual == ParsedPage(
        disclosures=(
            Disclosure(
                announcement_date=ANNOUNCEMENT_DATE,
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


def test_final_page_uses_info_total_when_handlers_only_link_backwards() -> None:
    actual = _parse(
        _page_html([_row()], current=5, total=5, links=[1, 2, 3, 4]),
        page=5,
    )

    assert actual.total_pages == 5
    assert actual.disclosures[0].page == 5


def test_global_page_handler_text_outside_pagination_is_ignored() -> None:
    actual = _parse(
        _page_html(
            current=1,
            total=5,
            links=[2, 5],
            extra_html="<script>fnPageGo('999')</script>",
        )
    )

    assert actual.total_pages == 5


@pytest.mark.parametrize(
    "pagination_html",
    [
        "",
        _paging_html() + _paging_html(),
        """
          <section class="paging-group">
            <div class="info type-00"><strong>1</strong>/1</div>
          </section>
        """,
        """
          <section class="paging-group">
            <div class="paging type-00">
              <a class="active" href="#page_link_1"
                 onclick="return false;">1</a>
            </div>
          </section>
        """,
        """
          <section class="paging-group">
            <div class="paging type-00"></div>
            <div class="paging type-00"></div>
            <div class="info type-00"><strong>1</strong>/1</div>
          </section>
        """,
        """
          <section class="paging-group">
            <div class="paging type-00"></div>
            <div class="info type-00"><strong>1</strong>/1</div>
            <div class="info type-00"><strong>1</strong>/1</div>
          </section>
        """,
    ],
)
def test_pagination_structure_must_be_unique_and_complete(
    pagination_html: str,
) -> None:
    with pytest.raises(KindSchemaError, match="pagination"):
        _parse(_page_html(pagination_html=pagination_html))


@pytest.mark.parametrize(
    "info_html",
    [
        '<div class="info type-00"><strong>0</strong>/5</div>',
        '<div class="info type-00"><strong>01</strong>/5</div>',
        '<div class="info type-00"><strong>１</strong>/5</div>',
        '<div class="info type-00"><strong>1</strong>/0</div>',
        '<div class="info type-00"><strong>1</strong>/05</div>',
        '<div class="info type-00"><strong>1</strong>/５</div>',
        '<div class="info type-00"><strong>5</strong>/4</div>',
        '<div class="info type-00"><strong>1</strong> /5</div>',
        (
            '<div class="info type-00"><strong>1</strong>'
            '<span>/5</span></div>'
        ),
    ],
)
def test_pagination_info_requires_canonical_current_and_total(
    info_html: str,
) -> None:
    pagination = _paging_html(info_html=info_html)

    with pytest.raises(KindSchemaError, match="pagination"):
        _parse(_page_html(pagination_html=pagination))


def test_pagination_current_must_equal_caller_page() -> None:
    with pytest.raises(KindSchemaError, match="current.*page 2"):
        _parse(_page_html(current=1, total=5), page=2)


@pytest.mark.parametrize(
    "active_html",
    [
        "",
        (
            '<a class="active" href="#page_link_1" '
            'onclick="return false;">1</a>'
            '<a class="active" href="#page_link_1" '
            'onclick="return false;">1</a>'
        ),
        '<a class="active" href="#page_link_1" onclick="return false;">2</a>',
        '<a class="active" href="#page_link_2" onclick="return false;">1</a>',
        '<a class="active" href="#page_link_1" onclick="return false">1</a>',
        (
            '<a class="active" href="#page_link_1" '
            'onclick="fnPageGo(\'1\');return false;">1</a>'
        ),
    ],
)
def test_active_page_link_is_exact_boundary_metadata(active_html: str) -> None:
    with pytest.raises(KindSchemaError, match="active"):
        _parse(
            _page_html(
                pagination_html=_paging_html(active_html=active_html)
            )
        )


@pytest.mark.parametrize(
    "handler",
    [
        "fnPageGo('2')",
        "fnPageGo('2'); return false;",
        " fnPageGo('2');return false;",
        "fnPageGo('02');return false;",
        "fnPageGo('6');return false;",
        "prefixfnPageGo('2');return false;",
    ],
)
def test_every_pagination_handler_has_exact_observed_grammar(
    handler: str,
) -> None:
    malformed = f'<a href="#page_link_2" onclick="{handler}">2</a>'
    pagination = _paging_html(current=1, total=5, nav_extra=malformed)

    with pytest.raises(KindSchemaError, match="pagination.*handler"):
        _parse(_page_html(pagination_html=pagination))


def test_disclosure_table_is_required() -> None:
    with pytest.raises(KindSchemaError, match="table"):
        _parse(_page_html(table_html="<p>no table</p>"))


def test_multiple_candidate_tables_are_rejected() -> None:
    tables = _table_html() + _table_html()

    with pytest.raises(KindSchemaError, match="table"):
        _parse(_page_html(table_html=tables))


def test_table_summary_must_match_exactly() -> None:
    with pytest.raises(KindSchemaError, match="summary"):
        _parse(
            _page_html(table_html=_table_html(summary="시간, 회사명, 제목"))
        )


@pytest.mark.parametrize(
    "tbody_html",
    [
        "",
        "<tbody></tbody><tbody></tbody>",
        "<div><tbody></tbody></div>",
    ],
)
def test_table_requires_exactly_one_direct_tbody(tbody_html: str) -> None:
    with pytest.raises(KindSchemaError, match="tbody"):
        _parse(_page_html(table_html=_table_html(tbody_html=tbody_html)))


@pytest.mark.parametrize(
    "cells",
    [
        "<td>08:05</td><td>company</td><td>title</td><td>submitter</td>",
        (
            "<td>08:05</td><td>company</td><td>title</td>"
            "<td>submitter</td><td>-</td><td>extra</td>"
        ),
    ],
)
def test_each_direct_row_requires_exactly_five_direct_cells(cells: str) -> None:
    with pytest.raises(KindSchemaError, match="5.*cells"):
        _parse(_page_html([f"<tr>{cells}</tr>"]))


def test_nested_table_rows_are_not_treated_as_disclosure_rows() -> None:
    row = _row().replace(
        "<td>-</td>",
        "<td><table><tbody><tr><td>nested</td></tr></tbody></table></td>",
    )

    actual = _parse(_page_html([row]))

    assert len(actual.disclosures) == 1


@pytest.mark.parametrize(
    "rows",
    [[], [_market_stat_row(with_receipt=False)], [_market_stat_row()]],
)
def test_empty_or_proven_noncompany_rows_are_valid(rows: list[str]) -> None:
    assert _parse(_page_html(rows)).disclosures == ()


def test_company_text_without_company_handler_is_schema_error() -> None:
    with pytest.raises(KindSchemaError, match="company.*handler"):
        _parse(
            _page_html(
                [
                    _row(
                        title="영업실적 등에 대한 전망",
                        issuer_onclick=None,
                        disclosure_onclick=None,
                    )
                ]
            )
        )


def test_company_handler_without_receipt_handler_is_schema_error() -> None:
    with pytest.raises(KindSchemaError, match="receipt.*handler"):
        _parse(
            _page_html(
                [
                    _row(
                        title="주주총회소집공고",
                        disclosure_onclick=None,
                    )
                ]
            )
        )


@pytest.mark.parametrize(
    "title",
    [
        PROVISIONAL_TITLE,
        "연결재무제표기준영업(잠정)실적(공정공시)",
        "영업(잠정)실적(공정공시)(자회사의 주요경영사항)",
    ],
)
@pytest.mark.parametrize("missing_link", ["company", "disclosure"])
def test_provisional_title_family_requires_both_links(
    title: str,
    missing_link: str,
) -> None:
    issuer: str | None = "companysummary_open('00066');return false;"
    receipt: str | None = "openDisclsViewer('20240425000004','')"
    if missing_link == "company":
        issuer = None
    else:
        receipt = None

    with pytest.raises(KindSchemaError, match="provisional.*link"):
        _parse(
            _page_html(
                [
                    _row(
                        title=title,
                        issuer_onclick=issuer,
                        disclosure_onclick=receipt,
                    )
                ]
            )
        )


def test_multiple_company_handler_anchors_are_rejected() -> None:
    company_html = (
        '<a onclick="companysummary_open(\'00066\')">SK하이닉스</a>'
        '<a onclick="companysummary_open(\'00067\')">다른회사</a>'
    )

    with pytest.raises(KindSchemaError, match="multiple.*company"):
        _parse(_page_html([_row(company_html=company_html)]))


def test_multiple_receipt_handler_anchors_are_rejected() -> None:
    title_html = (
        '<a onclick="openDisclsViewer(\'20240425000004\',\'\')">첫 공시</a>'
        '<a onclick="openDisclsViewer(\'20240425000005\',\'\')">둘째 공시</a>'
    )

    with pytest.raises(KindSchemaError, match="multiple.*receipt"):
        _parse(_page_html([_row(title_html=title_html)]))


def test_selected_anchor_text_excludes_plain_markers_and_unrelated_links() -> None:
    company_html = (
        '<span>유가증권</span>'
        '<a onclick="companysummary_open(\'00066\')"> SK하이닉스 </a>'
        '<a href="#profile">회사정보</a>'
    )
    title_html = (
        '정정 '
        '<a onclick="openDisclsViewer(\'20240425000004\',\'\')">'
        f" {PROVISIONAL_TITLE} </a>"
        '<a href="#note">첨부</a>'
    )

    disclosure = _parse(
        _page_html(
            [_row(company_html=company_html, title_html=title_html)]
        )
    ).disclosures[0]

    assert disclosure.company == "SK하이닉스"
    assert disclosure.title == PROVISIONAL_TITLE


@pytest.mark.parametrize("time", ["8:05", "0٨:0٥", "08：05", "24:00"])
def test_company_row_requires_ascii_canonical_time(time: str) -> None:
    with pytest.raises(KindSchemaError, match="time"):
        _parse(_page_html([_row(time=time)]))


@pytest.mark.parametrize(
    ("issuer", "receipt"),
    [
        ("companysummary_open('00066')", "openDisclsViewer('20240425000004','')"),
        ("companysummary_open('00066');", "openDisclsViewer('20240425000004','');"),
        (
            "companysummary_open('00066'); return false;",
            "openDisclsViewer('20240425000004' , 'detail');return false;",
        ),
        (
            "  companysummary_open('00066');return false;  ",
            "  openDisclsViewer('20240425000004',''); return false;  ",
        ),
    ],
)
def test_complete_observed_identity_handlers_are_accepted(
    issuer: str,
    receipt: str,
) -> None:
    disclosure = _parse(
        _page_html(
            [_row(issuer_onclick=issuer, disclosure_onclick=receipt)]
        )
    ).disclosures[0]

    assert disclosure.issuer_id == "00066"
    assert disclosure.receipt_id == "20240425000004"


@pytest.mark.parametrize(
    ("field", "handler"),
    [
        ("issuer", "companysummary_open('00066')return false;"),
        ("issuer", "prefixcompanysummary_open('00066')suffix"),
        (
            "issuer",
            "companysummary_open('00066');companysummary_open('00067')",
        ),
        ("issuer", "companysummary_open('0006');return false;"),
        ("receipt", "openDisclsViewer('20240425000004','')return false;"),
        ("receipt", "prefixopenDisclsViewer('20240425000004','')suffix"),
        (
            "receipt",
            "openDisclsViewer('20240425000004','');"
            "openDisclsViewer('20240425000005','')",
        ),
        ("receipt", "openDisclsViewer('20240425000004')"),
    ],
)
def test_identity_handlers_reject_junk_truncation_and_ambiguous_tails(
    field: str,
    handler: str,
) -> None:
    issuer = handler if field == "issuer" else "companysummary_open('00066')"
    receipt = (
        handler if field == "receipt" else "openDisclsViewer('20240425000004','')"
    )

    with pytest.raises(KindSchemaError, match=f"{field}.*handler"):
        _parse(
            _page_html(
                [_row(issuer_onclick=issuer, disclosure_onclick=receipt)]
            )
        )


@pytest.mark.parametrize(
    "date",
    [
        True,
        None,
        " 2024-04-25",
        "2024-04-25 ",
        "2024-4-25",
        "2024-04-25T00:00:00",
        "２０２４-０４-２５",
        "2024-02-30",
    ],
)
def test_announcement_date_requires_valid_ascii_iso_date(date: object) -> None:
    with pytest.raises(KindSchemaError, match="announcement date"):
        _parse(_page_html(), date=date)


def test_receipt_date_must_match_requested_announcement_date() -> None:
    with pytest.raises(
        KindSchemaError,
        match=r"2024-04-25.*page 1.*row 1.*20240424",
    ):
        _parse(
            _page_html(
                [
                    _row(
                        disclosure_onclick=(
                            "openDisclsViewer('20240424000004','')"
                        )
                    )
                ]
            )
        )


def test_nonprovisional_prior_day_receipt_remains_valid_live_evidence() -> None:
    disclosure = _parse(
        _page_html(
            [
                _row(
                    title="임원ㆍ주요주주특정증권등소유상황보고서",
                    disclosure_onclick=(
                        "openDisclsViewer('20240424000740','')"
                    ),
                )
            ]
        )
    ).disclosures[0]

    assert disclosure.announcement_date == "2024-04-25"
    assert disclosure.receipt_id == "20240424000740"


def test_duplicate_receipt_within_page_is_schema_corruption() -> None:
    with pytest.raises(
        KindSchemaError,
        match=r"2024-04-25.*page 1.*row 2.*20240425000004",
    ):
        _parse(
            _page_html(
                [
                    _row(),
                    _row(
                        company="삼성전자",
                        issuer_onclick="companysummary_open('00593')",
                    ),
                ]
            )
        )


@pytest.mark.parametrize("page", [True, False, 0, 1000, -1, 1.0, "1"])
def test_caller_page_requires_ascii_range_integer(page: object) -> None:
    with pytest.raises(KindSchemaError, match="page"):
        _parse(_page_html(), page=page)  # type: ignore[arg-type]


def test_valid_rows_preserve_dom_order_and_one_based_row_positions() -> None:
    actual = _parse(
        _page_html(
            [
                _market_stat_row(),
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
    )

    assert [item.company for item in actual.disclosures] == [
        "SK하이닉스",
        "삼성전자",
    ]
    assert [item.position for item in actual.disclosures] == [2, 3]
