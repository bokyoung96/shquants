from __future__ import annotations

from datetime import date as calendar_date
import re

from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag

from kind.models import Disclosure, ParsedPage
from kind.selectors import (
    ACTIVE_PAGE_SELECTOR,
    COMPANY_LINK_ONCLICK,
    DISCLOSURE_LINK_ONCLICK,
    EXPECTED_CELL_COUNT,
    EXPECTED_TABLE_SUMMARY,
    ISSUER_PATTERN,
    PAGE_PATTERN,
    PAGINATION_GROUP_SELECTOR,
    PAGINATION_INFO_SELECTOR,
    PAGINATION_NAV_SELECTOR,
    PROVISIONAL_TITLE_PATTERN,
    RECEIPT_PATTERN,
    ROW_SELECTOR,
    TABLE_SELECTOR,
    TIME_PATTERN,
)


_ASCII_ISO_DATE_PATTERN = r"[0-9]{4}-[0-9]{2}-[0-9]{2}"


class KindSchemaError(ValueError):
    """Raised when KIND HTML does not match the expected disclosure schema."""


def parse_disclosure_page(
    html: str,
    *,
    announcement_date: str,
    page: int,
) -> ParsedPage:
    _validate_page(page)
    _validate_announcement_date(announcement_date, page)
    context = f"announcement date {announcement_date!r}, page {page}"

    soup = BeautifulSoup(html, "html.parser")
    tbody = _disclosure_tbody(soup, context)
    total_pages = _pagination_total(soup, page, context)

    disclosures: list[Disclosure] = []
    receipt_ids: set[str] = set()
    for position, row in enumerate(tbody.select(ROW_SELECTOR), start=1):
        cells = row.find_all("td", recursive=False)
        if len(cells) != EXPECTED_CELL_COUNT:
            raise _row_error(
                context,
                position,
                f"expected {EXPECTED_CELL_COUNT} direct td cells, got {len(cells)}",
            )

        time_cell, company_cell, title_cell, submitter_cell, _ = cells
        company_links = _handler_links(company_cell, COMPANY_LINK_ONCLICK)
        receipt_links = _handler_links(title_cell, DISCLOSURE_LINK_ONCLICK)
        company_evidence = _cell_text(company_cell)
        title_evidence = _cell_text(title_cell)

        if not company_evidence and not company_links:
            continue

        provisional = re.search(PROVISIONAL_TITLE_PATTERN, title_evidence) is not None
        company_link = _require_identity_link(
            company_links,
            company_cell,
            COMPANY_LINK_ONCLICK,
            "company",
            provisional,
            context,
            position,
        )
        receipt_link = _require_identity_link(
            receipt_links,
            title_cell,
            DISCLOSURE_LINK_ONCLICK,
            "receipt",
            provisional,
            context,
            position,
        )

        time = _cell_text(time_cell)
        if re.fullmatch(TIME_PATTERN, time) is None:
            raise _row_error(
                context,
                position,
                f"invalid disclosure time {time!r}",
            )

        issuer_id = _extract_identifier(
            company_link,
            ISSUER_PATTERN,
            "issuer",
            context,
            position,
        )
        receipt_id = _extract_identifier(
            receipt_link,
            RECEIPT_PATTERN,
            "receipt",
            context,
            position,
        )
        expected_receipt_date = announcement_date.replace("-", "")
        # KIND can list prior-day receipts for non-provisional reports. Only
        # provisional-family disclosures are eligible for announcement-time
        # matching, so their date provenance remains strict.
        if provisional and not receipt_id.startswith(expected_receipt_date):
            raise _row_error(
                context,
                position,
                f"receipt {receipt_id!r} belongs to a different date",
            )
        if receipt_id in receipt_ids:
            raise _row_error(
                context,
                position,
                f"duplicate receipt {receipt_id!r}",
            )
        receipt_ids.add(receipt_id)

        company = _cell_text(company_link)
        title = _cell_text(receipt_link)
        if not company:
            raise _row_error(context, position, "company anchor text is empty")
        if not title:
            raise _row_error(context, position, "receipt anchor title is empty")

        disclosures.append(
            Disclosure(
                announcement_date=announcement_date,
                time=time,
                company=company,
                title=title,
                submitter=_cell_text(submitter_cell),
                issuer_id=issuer_id,
                receipt_id=receipt_id,
                page=page,
                position=position,
            )
        )

    return ParsedPage(disclosures=tuple(disclosures), total_pages=total_pages)


def _validate_page(page: int) -> None:
    if isinstance(page, bool) or not isinstance(page, int) or not 1 <= page <= 999:
        raise KindSchemaError("page must be an integer from 1 through 999")


def _validate_announcement_date(announcement_date: str, page: int) -> None:
    if not isinstance(announcement_date, str) or re.fullmatch(
        _ASCII_ISO_DATE_PATTERN,
        announcement_date,
    ) is None:
        raise KindSchemaError(
            f"announcement date {announcement_date!r}, page {page}: "
            "expected ASCII YYYY-MM-DD"
        )
    try:
        calendar_date.fromisoformat(announcement_date)
    except ValueError as error:
        raise KindSchemaError(
            f"announcement date {announcement_date!r}, page {page}: "
            "invalid calendar date"
        ) from error


def _disclosure_tbody(soup: BeautifulSoup, context: str) -> Tag:
    tables = soup.select(TABLE_SELECTOR)
    if len(tables) != 1:
        raise KindSchemaError(
            f"{context}: expected exactly one disclosure table matching "
            f"{TABLE_SELECTOR!r}, got {len(tables)}"
        )
    table = tables[0]
    if table.get("summary") != EXPECTED_TABLE_SUMMARY:
        raise KindSchemaError(
            f"{context}: table summary {table.get('summary')!r} does not match "
            f"{EXPECTED_TABLE_SUMMARY!r}"
        )
    tbodies = table.find_all("tbody", recursive=False)
    if len(tbodies) != 1:
        raise KindSchemaError(
            f"{context}: expected exactly one direct tbody, got {len(tbodies)}"
        )
    return tbodies[0]


def _pagination_total(
    soup: BeautifulSoup,
    caller_page: int,
    context: str,
) -> int:
    groups = soup.select(PAGINATION_GROUP_SELECTOR)
    if len(groups) != 1:
        raise KindSchemaError(
            f"{context}: pagination requires exactly one group, got {len(groups)}"
        )
    group = groups[0]
    navs = group.select(PAGINATION_NAV_SELECTOR)
    infos = group.select(PAGINATION_INFO_SELECTOR)
    if len(navs) != 1 or len(infos) != 1:
        raise KindSchemaError(
            f"{context}: pagination requires one direct nav and one direct info "
            f"block, got {len(navs)} nav and {len(infos)} info"
        )
    nav = navs[0]
    current, total = _pagination_info(infos[0], context)
    if current != caller_page:
        raise KindSchemaError(
            f"{context}: pagination current {current} does not match caller "
            f"page {caller_page}"
        )
    if total < current:
        raise KindSchemaError(
            f"{context}: pagination total {total} is below current {current}"
        )

    active_links = nav.select(ACTIVE_PAGE_SELECTOR)
    if len(active_links) != 1:
        raise KindSchemaError(
            f"{context}: expected exactly one active page link, "
            f"got {len(active_links)}"
        )
    active = active_links[0]
    if (
        active.get_text() != str(current)
        or active.get("href") != f"#page_link_{current}"
        or active.get("onclick") != "return false;"
    ):
        raise KindSchemaError(
            f"{context}: active page link does not match current page {current}"
        )

    handler_pattern = rf"(?:{PAGE_PATTERN});return false;"
    for link in nav.find_all("a"):
        if link is active:
            continue
        onclick = link.get("onclick")
        if not isinstance(onclick, str) or "fnPageGo" not in onclick:
            continue
        match = re.fullmatch(handler_pattern, onclick)
        if match is None:
            raise KindSchemaError(
                f"{context}: invalid pagination handler {onclick!r}"
            )
        target = int(match.group(1))
        if target > total or target == current:
            raise KindSchemaError(
                f"{context}: pagination handler {onclick!r} targets page "
                f"outside 1..{total} excluding current {current}"
            )
    return total


def _pagination_info(info: Tag, context: str) -> tuple[int, int]:
    current_nodes = info.find_all("strong", recursive=False)
    if len(current_nodes) != 1:
        raise KindSchemaError(
            f"{context}: pagination info requires one direct strong current page"
        )
    current_node = current_nodes[0]
    current_text = current_node.get_text()
    if re.fullmatch(r"[1-9][0-9]{0,2}", current_text) is None:
        raise KindSchemaError(
            f"{context}: invalid pagination current {current_text!r}"
        )
    following = current_node.next_sibling
    if not isinstance(following, NavigableString):
        raise KindSchemaError(
            f"{context}: pagination total must directly follow current page"
        )
    total_match = re.match(r"/([1-9][0-9]{0,2})(?=\s|$)", str(following))
    if total_match is None:
        raise KindSchemaError(
            f"{context}: invalid pagination total metadata {str(following)!r}"
        )
    return int(current_text), int(total_match.group(1))


def _cell_text(cell: Tag) -> str:
    return cell.get_text(" ", strip=True)


def _handler_links(cell: Tag, handler_name: str) -> list[Tag]:
    handler_start = re.compile(rf"\s*{re.escape(handler_name)}\s*\(")
    return [
        anchor
        for anchor in cell.find_all("a")
        if isinstance(anchor.get("onclick"), str)
        and handler_start.match(anchor["onclick"]) is not None
    ]


def _require_identity_link(
    links: list[Tag],
    cell: Tag,
    handler_name: str,
    identity_name: str,
    provisional: bool,
    context: str,
    position: int,
) -> Tag:
    if len(links) == 1:
        return links[0]
    if len(links) > 1:
        raise _row_error(
            context,
            position,
            f"multiple {identity_name} handler anchors ({len(links)})",
        )
    mentioned = any(
        isinstance(anchor.get("onclick"), str)
        and handler_name in anchor["onclick"]
        for anchor in cell.find_all("a")
    )
    if provisional and not mentioned:
        raise _row_error(
            context,
            position,
            f"provisional disclosure is missing required {identity_name} link",
        )
    handler_label = (
        "company/issuer" if identity_name == "company" else identity_name
    )
    raise _row_error(
        context,
        position,
        f"missing or malformed {handler_label} handler",
    )


def _extract_identifier(
    link: Tag,
    pattern: str,
    identifier_name: str,
    context: str,
    position: int,
) -> str:
    onclick = link.get("onclick")
    if not isinstance(onclick, str):
        raise _row_error(
            context,
            position,
            f"missing {identifier_name} handler",
        )
    complete_handler = rf"\s*(?:{pattern})\s*(?:;(?:\s*return false;)?)?\s*"
    match = re.fullmatch(complete_handler, onclick)
    if match is None:
        raise _row_error(
            context,
            position,
            f"invalid {identifier_name} handler {onclick!r}",
        )
    return match.group(1)


def _row_error(context: str, position: int, detail: str) -> KindSchemaError:
    return KindSchemaError(f"{context}, row {position}: {detail}")
