from pathlib import Path

from etfs.fnguide.methodology import (
    FnIndexEntry,
    MethodologyCandidate,
    MethodologyDownload,
    build_fnguide_queries,
    build_fnindex_catalog_candidates,
    extract_pdf_links,
    save_pdf,
    rank_candidates,
    write_manifest,
)
from etfs.research import EtfListing


def test_build_fnguide_queries_prefers_fnindex_and_methodology_terms() -> None:
    queries = build_fnguide_queries(EtfListing(code="395270", name="HANARO Fn K-반도체"))

    assert queries[0] == '"HANARO Fn K-반도체" FnGuide 지수 방법론'
    assert "site:fnindex.co.kr" in queries[1]
    assert "file.fnguide.com" in queries[2]


def test_rank_candidates_puts_fnindex_detail_and_fnguide_files_first() -> None:
    candidates = [
        MethodologyCandidate(title="blog", url="https://example.com/post", snippet="", query="q"),
        MethodologyCandidate(title="pdf", url="https://file.fnguide.com/fnindex/files/a.pdf", snippet="", query="q"),
        MethodologyCandidate(title="detail", url="https://www.fnindex.co.kr/overview/detail/I/FI00.X", snippet="", query="q"),
        MethodologyCandidate(title="krx", url="https://index.krx.co.kr/x.pdf", snippet="", query="q"),
    ]

    ranked = rank_candidates(candidates)

    assert [item.url for item in ranked[:3]] == [
        "https://www.fnindex.co.kr/overview/detail/I/FI00.X",
        "https://file.fnguide.com/fnindex/files/a.pdf",
        "https://index.krx.co.kr/x.pdf",
    ]


def test_build_fnindex_catalog_candidates_matches_cleaned_etf_name() -> None:
    catalog = [
        FnIndexEntry(code="FI00.WLT.NHS", name="FnGuide K-반도체 지수", detail_type="I"),
        FnIndexEntry(code="FI00.WLT.SBI", name="FnGuide 2차전지 산업 지수", detail_type="I"),
    ]

    candidates = build_fnindex_catalog_candidates(
        EtfListing(code="395270", name="HANARO Fn K-반도체"),
        catalog,
    )

    assert candidates[0].url == "https://www.fnindex.co.kr/overview/detail/I/FI00.WLT.NHS"
    assert candidates[0].query == "fnindex_catalog"


def test_build_fnindex_catalog_candidates_prefers_specific_token_overlap() -> None:
    catalog = [
        FnIndexEntry(code="FI00.45.30", name="MKF 반도체", detail_type="I"),
        FnIndexEntry(code="FI00.WLT.HBM", name="FnGuide AI 반도체 TOP3+ 지수", detail_type="I"),
    ]

    candidates = build_fnindex_catalog_candidates(
        EtfListing(code="469150", name="ACE AI반도체TOP3+"),
        catalog,
    )

    assert candidates[0].url == "https://www.fnindex.co.kr/overview/detail/I/FI00.WLT.HBM"


def test_build_fnindex_catalog_candidates_allows_partial_korean_token_match() -> None:
    catalog = [
        FnIndexEntry(code="FI00.WLT.NH2", name="FnGuide 2차전지 소재 주도주 지수", detail_type="I"),
        FnIndexEntry(code="FI00.WLT.HST", name="FnGuide 성장소비주도", detail_type="I"),
    ]

    candidates = build_fnindex_catalog_candidates(
        EtfListing(code="226380", name="ACE Fn성장소비주도주"),
        catalog,
    )

    assert candidates[0].url == "https://www.fnindex.co.kr/overview/detail/I/FI00.WLT.HST"


def test_build_fnindex_catalog_candidates_demotes_leverage_when_listing_is_not_leverage() -> None:
    catalog = [
        FnIndexEntry(code="FI00.WLT.L05", name="FnGuide 2차전지 산업 레버리지 지수", detail_type="I"),
        FnIndexEntry(code="FI00.WLT.SBI", name="FnGuide 2차전지 산업 지수", detail_type="I"),
    ]

    candidates = build_fnindex_catalog_candidates(
        EtfListing(code="385600", name="ACE 2차전지&친환경차액티브"),
        catalog,
    )

    assert candidates[0].url == "https://www.fnindex.co.kr/overview/detail/I/FI00.WLT.SBI"


def test_build_fnindex_catalog_candidates_splits_compound_sector_terms() -> None:
    catalog = [FnIndexEntry(code="FI00.WLT.GMS", name="FnGuide 게임 산업", detail_type="I")]

    candidates = build_fnindex_catalog_candidates(
        EtfListing(code="300950", name="KODEX 게임산업"),
        catalog,
    )

    assert candidates[0].url == "https://www.fnindex.co.kr/overview/detail/I/FI00.WLT.GMS"


def test_build_fnindex_catalog_candidates_splits_number_letter_terms() -> None:
    catalog = [FnIndexEntry(code="FI00.45", name="MKF IT", detail_type="I")]

    candidates = build_fnindex_catalog_candidates(
        EtfListing(code="363580", name="KODEX 200IT TR"),
        catalog,
    )

    assert candidates[0].url == "https://www.fnindex.co.kr/overview/detail/I/FI00.45"


def test_extract_pdf_links_returns_absolute_fnguide_links_first() -> None:
    html = """
    <a href="/local.pdf">local</a>
    <a href="https://file.fnguide.com/fnindex/files/FnGuide_Methodology.pdf">method</a>
    <a href="https://index.krx.co.kr/file.pdf">krx</a>
    """

    links = extract_pdf_links(html, base_url="https://www.fnindex.co.kr/overview/detail/I/FI00.X")

    assert links == [
        "https://file.fnguide.com/fnindex/files/FnGuide_Methodology.pdf",
        "https://www.fnindex.co.kr/local.pdf",
        "https://index.krx.co.kr/file.pdf",
    ]


def test_write_manifest_records_downloads(tmp_path: Path) -> None:
    downloads = [
        MethodologyDownload(
            code="395270",
            name="HANARO Fn K-반도체",
            status="downloaded",
            source_url="https://file.fnguide.com/fnindex/files/FnGuide_Methodology.pdf",
            page_url="https://www.fnindex.co.kr/overview/detail/I/FI00.X",
            file_path="etfs/raw/methodologies/395270.pdf",
            sha256="abc",
            bytes=12,
            query='"HANARO Fn K-반도체" FnGuide 지수 방법론',
            error="",
            provider="fnguide",
            index_name="FnGuide AI Semiconductor Index",
            source_type="methodology_pdf",
            confidence="high",
        )
    ]

    csv_path, json_path = write_manifest(downloads, tmp_path)

    assert csv_path.name == "pdfs.csv"
    assert json_path.name == "pdfs.json"
    assert "395270" in csv_path.read_text(encoding="utf-8-sig")
    assert "fnguide" in csv_path.read_text(encoding="utf-8-sig")
    assert '"status": "downloaded"' in json_path.read_text(encoding="utf-8")
    assert '"provider": "fnguide"' in json_path.read_text(encoding="utf-8")


def test_save_pdf_marks_download_as_fnguide_methodology(tmp_path: Path) -> None:
    class FakeResponse:
        content = b"%PDF-1.4 sample"
        headers = {"content-type": "application/pdf"}

        def raise_for_status(self) -> None:
            return None

    class FakeClient:
        def get(self, url, follow_redirects=True):
            return FakeResponse()

    download = save_pdf(
        FakeClient(),
        EtfListing(code="091160", name="KODEX 반도체"),
        pdf_url="https://file.fnguide.com/fnindex/files/sample.pdf",
        page_url="https://www.fnindex.co.kr/overview/detail/I/FI00.X",
        query="fnindex_catalog",
        raw_dir=tmp_path,
        index_name="FnGuide AI Semiconductor Index",
    )

    assert download.provider == "fnguide"
    assert download.index_name == "FnGuide AI Semiconductor Index"
    assert download.source_type == "methodology_pdf"
    assert download.confidence == "high"
