KIND_MAIN_URL = (
    "https://kind.krx.co.kr/disclosure/todaydisclosure.do"
    "?method=searchTodayDisclosureMain"
)
KIND_SUB_URL = "https://kind.krx.co.kr/disclosure/todaydisclosure.do"
PARSER_SCHEMA_VERSION = 1

FORM_DEFAULTS = {
    "method": "searchTodayDisclosureSub",
    "currentPageSize": "100",
    "orderMode": "0",
    "orderStat": "D",
    "marketType": "",
    "forward": "todaydisclosure_sub",
    "searchMode": "",
    "searchCodeType": "",
    "chose": "S",
    "todayFlag": "N",
    "repIsuSrtCd": "",
    "kosdaqSegment": "",
    "searchCorpName": "",
    "copyUrl": "",
}

TABLE_SELECTOR = "table.list"
ROW_SELECTOR = "tbody tr"
COMPANY_LINK_ONCLICK = "companysummary_open"
DISCLOSURE_LINK_ONCLICK = "openDisclsViewer"
EXPECTED_CELL_COUNT = 5
TIME_PATTERN = r"^(?:[01]\d|2[0-3]):[0-5]\d$"
PAGE_PATTERN = r"fnPageGo\('(\d+)'\)"
ISSUER_PATTERN = r"companysummary_open\('([^']+)'\)"
RECEIPT_PATTERN = r"openDisclsViewer\('([^']+)'"
PROVISIONAL_TITLE_PATTERN = r"영업\s*\(잠정\)\s*실적\s*\(공정공시\)"
FORECAST_TITLE_PATTERN = r"영업실적\s*등에\s*대한\s*전망"
