from __future__ import annotations

import re

from bs4 import BeautifulSoup
from bs4.element import Tag

from kind.models import Disclosure, ParsedPage
from kind.selectors import (
    COMPANY_LINK_ONCLICK,
    DISCLOSURE_LINK_ONCLICK,
    EXPECTED_CELL_COUNT,
    ISSUER_PATTERN,
    PAGE_PATTERN,
    PROVISIONAL_TITLE_PATTERN,
    RECEIPT_PATTERN,
    ROW_SELECTOR,
    TABLE_SELECTOR,
    TIME_PATTERN,
)


class KindSchemaError(ValueError):
    """Raised when KIND HTML does not match the expected disclosure schema."""


def parse_disclosure_page(
    html: str,
    *,
    announcement_date: str,
    page: int,
) -> ParsedPage:
    if isinstance(page, bool) or not isinstance(page, int) or not 1 <= page <= 999:
        raise KindSchemaError("page must be an integer from 1 through 999")

    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one(TABLE_SELECTOR)
    if table is None:
        raise KindSchemaError(f"missing disclosure table matching {TABLE_SELECTOR!r}")

    page_matches = tuple(re.finditer(PAGE_PATTERN, html))
    if "fnPageGo(" in html and not page_matches:
        raise KindSchemaError("pagination handlers do not match the KIND schema")

    total_pages = max(
        (int(match.group(1)) for match in page_matches),
        default=1,
    )
    if total_pages < page:
        raise KindSchemaError(
            f"pagination total {total_pages} is below current page {page}"
        )

    disclosures: list[Disclosure] = []
    for position, row in enumerate(table.select(ROW_SELECTOR), start=1):
        cells = row.find_all("td", recursive=False)
        if len(cells) != EXPECTED_CELL_COUNT:
            raise KindSchemaError(
                f"row {position} must contain exactly "
                f"{EXPECTED_CELL_COUNT} direct td cells"
            )

        time_cell, company_cell, title_cell, submitter_cell, _ = cells
        title = _cell_text(title_cell)
        company_link = _handler_link(company_cell, COMPANY_LINK_ONCLICK)
        disclosure_link = _handler_link(
            title_cell,
            DISCLOSURE_LINK_ONCLICK,
        )

        if company_link is None or disclosure_link is None:
            if re.search(PROVISIONAL_TITLE_PATTERN, title):
                raise KindSchemaError(
                    f"provisional disclosure row {position} is missing a required link"
                )
            continue

        time = _cell_text(time_cell)
        if re.fullmatch(TIME_PATTERN, time) is None:
            raise KindSchemaError(f"row {position} has invalid disclosure time")

        issuer_id = _extract_identifier(
            company_link,
            ISSUER_PATTERN,
            "issuer",
            position,
            allow_return_false=True,
        )
        receipt_id = _extract_identifier(
            disclosure_link,
            RECEIPT_PATTERN,
            "receipt",
            position,
        )

        disclosures.append(
            Disclosure(
                announcement_date=announcement_date,
                time=time,
                company=_cell_text(company_cell),
                title=title,
                submitter=_cell_text(submitter_cell),
                issuer_id=issuer_id,
                receipt_id=receipt_id,
                page=page,
                position=position,
            )
        )

    return ParsedPage(disclosures=tuple(disclosures), total_pages=total_pages)


def _cell_text(cell: Tag) -> str:
    return cell.get_text(" ", strip=True)


def _handler_link(cell: Tag, handler_name: str) -> Tag | None:
    handler_start = re.compile(rf"\s*{re.escape(handler_name)}\s*\(")
    return cell.find(
        "a",
        onclick=lambda value: (
            isinstance(value, str) and handler_start.match(value) is not None
        ),
    )


def _extract_identifier(
    link: Tag,
    pattern: str,
    identifier_name: str,
    position: int,
    *,
    allow_return_false: bool = False,
) -> str:
    onclick = link.get("onclick")
    if not isinstance(onclick, str):
        raise KindSchemaError(
            f"row {position} has invalid {identifier_name} handler"
        )

    allowed_tail = r"\s*;?\s*"
    if allow_return_false:
        allowed_tail += r"(?:return false;\s*)?"
    match = re.fullmatch(rf"\s*(?:{pattern}){allowed_tail}", onclick)
    if match is None:
        raise KindSchemaError(
            f"row {position} has invalid {identifier_name} handler"
        )
    return match.group(1)
