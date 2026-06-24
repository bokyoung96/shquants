from __future__ import annotations

import html
import math
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from PIL import Image as PILImage

from root import ROOT

from backtesting.data import ParquetStore
from backtesting.strategies.positivity import positivity_score


INPUT_DIR = ROOT.results_path / "pos_research" / "momentum_horizon_high_low_2020"
REPORT_DIR = ROOT.results_path / "pos_research" / "positivity_momentum_report"
REPORT_PATH = REPORT_DIR / "positivity_momentum_word_report.docx"
HORIZON_ORDER = ["1M", "3M", "6M", "12M", "3-1M", "6-1M", "12-1M"]
FACTOR_ORDER = ["positivity", "return_momentum", "high_sharpe", "high_low"]
FACTOR_LABELS = {
    "positivity": "Positivity",
    "return_momentum": "Return momentum",
    "high_sharpe": "High Sharpe",
    "high_low": "High-low channel",
}
LEG_LABELS = {"q5": "Q5 long-only", "q5_minus_q1": "Q5-Q1 spread"}
EMU_PER_INCH = 914400


@dataclass
class ImagePart:
    path: Path
    rid: str
    name: str
    width_emu: int
    height_emu: int


class DocxBuilder:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.parts: list[str] = []
        self.images: list[ImagePart] = []

    def heading(self, text: str, level: int = 1) -> None:
        style = "Title" if level == 0 else f"Heading{min(level, 3)}"
        self.parts.append(
            f'<w:p><w:pPr><w:pStyle w:val="{style}"/></w:pPr>{self._run(text, bold=level == 0)}</w:p>'
        )

    def paragraph(self, text: str, *, bold_lead: str | None = None) -> None:
        runs = ""
        if bold_lead:
            runs += self._run(bold_lead, bold=True)
            text = text.removeprefix(bold_lead)
        runs += self._run(text)
        self.parts.append(f"<w:p>{runs}</w:p>")

    def bullets(self, items: list[str]) -> None:
        for item in items:
            self.parts.append(
                '<w:p><w:pPr><w:pStyle w:val="ListBullet"/></w:pPr>'
                f"{self._run('- ' + item)}</w:p>"
            )

    def page_break(self) -> None:
        self.parts.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')

    def table(self, rows: list[list[object]], *, widths: list[int] | None = None) -> None:
        if not rows:
            return
        col_count = len(rows[0])
        widths = widths or [int(9000 / col_count)] * col_count
        grid = "".join(f'<w:gridCol w:w="{width}"/>' for width in widths[:col_count])
        body = [f'<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="0" w:type="auto"/></w:tblPr><w:tblGrid>{grid}</w:tblGrid>']
        for row_idx, row in enumerate(rows):
            cells = []
            for col_idx, value in enumerate(row):
                shading = '<w:shd w:fill="D9EAF7"/>' if row_idx == 0 else ""
                bold = row_idx == 0
                align = '<w:jc w:val="center"/>' if row_idx == 0 else ""
                cell_width = widths[col_idx] if col_idx < len(widths) else widths[-1]
                text = _format_cell(value)
                cells.append(
                    "<w:tc>"
                    f'<w:tcPr><w:tcW w:w="{cell_width}" w:type="dxa"/>{shading}</w:tcPr>'
                    f"<w:p><w:pPr>{align}</w:pPr>{self._run(text, bold=bold)}</w:p>"
                    "</w:tc>"
                )
            body.append(f"<w:tr>{''.join(cells)}</w:tr>")
        body.append("</w:tbl>")
        self.parts.append("".join(body))

    def image(self, path: Path, caption: str, *, max_width_inches: float = 6.5) -> None:
        path = path.resolve()
        with PILImage.open(path) as image:
            width_px, height_px = image.size
        ratio = height_px / width_px
        width_emu = int(max_width_inches * EMU_PER_INCH)
        height_emu = int(width_emu * ratio)
        rid = f"rId{len(self.images) + 1}"
        name = f"image{len(self.images) + 1}{path.suffix.lower()}"
        image_id = len(self.images) + 1
        self.images.append(ImagePart(path=path, rid=rid, name=name, width_emu=width_emu, height_emu=height_emu))
        self.parts.append(
            "<w:p><w:pPr><w:jc w:val=\"center\"/></w:pPr><w:r><w:drawing>"
            "<wp:inline distT=\"0\" distB=\"0\" distL=\"0\" distR=\"0\" "
            'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">'
            f'<wp:extent cx="{width_emu}" cy="{height_emu}"/>'
            f'<wp:docPr id="{image_id}" name="Picture {image_id}"/>'
            '<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
            '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            '<pic:nvPicPr><pic:cNvPr id="0" name="Picture"/><pic:cNvPicPr/></pic:nvPicPr>'
            '<pic:blipFill>'
            f'<a:blip r:embed="{rid}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/>'
            '<a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            '<pic:spPr><a:xfrm><a:off x="0" y="0"/>'
            f'<a:ext cx="{width_emu}" cy="{height_emu}"/></a:xfrm>'
            '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
            "</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>"
        )
        self.parts.append(
            '<w:p><w:pPr><w:jc w:val="center"/></w:pPr>'
            f'{self._run(caption, italic=True, size=18)}</w:p>'
        )

    def save(self) -> None:
        body = "".join(self.parts)
        document_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas" '
            'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
            'xmlns:o="urn:schemas-microsoft-com:office:office" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
            'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
            'xmlns:v="urn:schemas-microsoft-com:vml" '
            'xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing" '
            'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
            'xmlns:w10="urn:schemas-microsoft-com:office:word" '
            'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
            'xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" '
            'xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup" '
            'xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk" '
            'xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml" '
            'xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape" '
            'mc:Ignorable="w14 wp14"><w:body>'
            f"{body}"
            '<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="720" w:right="720" '
            'w:bottom="720" w:left="720" w:header="450" w:footer="450" w:gutter="0"/></w:sectPr>'
            "</w:body></w:document>"
        )
        with zipfile.ZipFile(self.path, "w", zipfile.ZIP_DEFLATED) as docx:
            docx.writestr("[Content_Types].xml", self._content_types())
            docx.writestr("_rels/.rels", self._root_rels())
            docx.writestr("word/document.xml", document_xml)
            docx.writestr("word/styles.xml", self._styles())
            docx.writestr("word/_rels/document.xml.rels", self._document_rels())
            for image in self.images:
                docx.write(image.path, f"word/media/{image.name}")

    def _run(self, text: str, *, bold: bool = False, italic: bool = False, size: int = 20) -> str:
        props = ['<w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:eastAsia="Malgun Gothic"/>', f'<w:sz w:val="{size}"/>']
        if bold:
            props.append("<w:b/>")
        if italic:
            props.append("<w:i/>")
        safe = html.escape(str(text), quote=False)
        return f"<w:r><w:rPr>{''.join(props)}</w:rPr><w:t>{safe}</w:t></w:r>"

    def _content_types(self) -> str:
        image_defaults = "".join(
            {
                '<Default Extension="png" ContentType="image/png"/>'
                for image in self.images
                if image.name.lower().endswith(".png")
            }
        )
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            f"{image_defaults}"
            '<Override PartName="/word/document.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            '<Override PartName="/word/styles.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
            "</Types>"
        )

    @staticmethod
    def _root_rels() -> str:
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="word/document.xml"/></Relationships>'
        )

    def _document_rels(self) -> str:
        rels = [
            '<Relationship Id="rStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
        ]
        for image in self.images:
            rels.append(
                f'<Relationship Id="{image.rid}" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
                f'Target="media/{image.name}"/>'
            )
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f"{''.join(rels)}</Relationships>"
        )

    @staticmethod
    def _styles() -> str:
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/>'
            '<w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:eastAsia="Malgun Gothic"/><w:sz w:val="20"/></w:rPr></w:style>'
            '<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/>'
            '<w:rPr><w:b/><w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:eastAsia="Malgun Gothic"/><w:sz w:val="34"/></w:rPr></w:style>'
            '<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/>'
            '<w:rPr><w:b/><w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:eastAsia="Malgun Gothic"/><w:sz w:val="28"/></w:rPr></w:style>'
            '<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/>'
            '<w:rPr><w:b/><w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:eastAsia="Malgun Gothic"/><w:sz w:val="24"/></w:rPr></w:style>'
            '<w:style w:type="paragraph" w:styleId="Heading3"><w:name w:val="heading 3"/>'
            '<w:rPr><w:b/><w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:eastAsia="Malgun Gothic"/><w:sz w:val="22"/></w:rPr></w:style>'
            '<w:style w:type="paragraph" w:styleId="ListBullet"><w:name w:val="List Bullet"/>'
            '<w:pPr><w:ind w:left="360" w:hanging="180"/></w:pPr><w:rPr><w:sz w:val="20"/></w:rPr></w:style>'
            '<w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/>'
            '<w:tblPr><w:tblBorders><w:top w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/>'
            '<w:left w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/>'
            '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/>'
            '<w:right w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/>'
            '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/>'
            '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="BFBFBF"/></w:tblBorders></w:tblPr></w:style>'
            "</w:styles>"
        )


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(INPUT_DIR / "summary.csv")
    portfolio = _latest_2026_positivity_12m_portfolio()
    sector_summary = _sector_summary(portfolio)

    doc = DocxBuilder(REPORT_PATH)
    _write_report(doc, summary, portfolio, sector_summary)
    doc.save()
    print(REPORT_PATH)


def _write_report(doc: DocxBuilder, summary: pd.DataFrame, portfolio: pd.DataFrame, sector_summary: pd.DataFrame) -> None:
    doc.heading("Positivity Momentum 보고서", level=0)
    doc.paragraph(f"작성일: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    doc.paragraph(
        "주 논거: positivity는 단순히 많이 오른 종목을 사는 전략이 아니라, 관측 기간 동안 하락하지 않은 날의 비율로 상승 경로의 꾸준함을 측정한다. "
        "한국 KOSPI200 테스트에서는 같은 lookback 조건에서 특히 Q5-Q1 spread의 위험조정 성과가 가장 일관적으로 좋았다."
    )
    doc.bullets(
        [
            "테스트 구간은 2020-01-02부터 2026-06-02까지이며, 신호 계산에는 2020년 이전 warm-up 데이터를 사용했다.",
            "성과는 비용 미반영 gross 기준이고, 신호일 종가부터 다음 거래일 종가까지의 close-to-close 수익률로 측정했다.",
            "비교 팩터는 positivity, return momentum, high Sharpe, high-low channel이며 lookback은 1M, 3M, 6M, 12M, 3-1M, 6-1M, 12-1M이다.",
        ]
    )

    doc.heading("1. 핵심 결론", level=1)
    doc.bullets(
        [
            "1M을 제외한 같은 horizon 비교에서 positivity는 Q5 long-only와 Q5-Q1 spread 모두에서 가장 안정적인 상위권 성과를 보였다.",
            "특히 spread 기준으로는 6M, 12M, 6-1M, 12-1M에서 positivity가 Sharpe 1위였고, MDD도 다른 momentum 계열보다 작았다.",
            "12-1M spread에서 positivity는 CAGR 17.54%, MDD -20.36%, Sharpe 1.051로, return momentum과 high Sharpe 대비 수익의 질이 더 좋았다.",
            "이는 positivity가 급등 magnitude보다 반복적인 비하락 일수를 보면서, 이벤트성 급등과 이후 반락 위험을 덜 선택하기 때문으로 해석된다.",
        ]
    )

    doc.heading("2. 성과지표 요약", level=1)
    doc.paragraph("아래 표는 각 horizon에서 Sharpe 1위 팩터와 positivity의 성과를 나란히 비교한 것이다.")
    doc.table(_winner_table(summary), widths=[900, 1200, 1700, 1000, 1000, 1000, 900, 1000, 1000, 1000])
    doc.paragraph("성과지표의 전체 비교는 Q5 long-only와 Q5-Q1 spread를 나누어 정리했다.")
    doc.heading("Q5 Long-only 성과", level=2)
    doc.table(_metrics_table(summary, "q5"), widths=[900, 1350, 1100, 1000, 1000, 950, 950])
    doc.heading("Q5-Q1 Spread 성과", level=2)
    doc.table(_metrics_table(summary, "q5_minus_q1"), widths=[900, 1350, 1100, 1000, 1000, 950, 950])

    doc.heading("3. Spread Return", level=1)
    doc.paragraph(
        "Spread return은 상위 quintile(Q5)을 매수하고 하위 quintile(Q1)을 매도한 factor payoff다. "
        "시장 전체 상승장의 도움을 제거하고, 해당 신호가 실제로 종목 간 상대성과를 구분했는지를 보는 데 유용하다."
    )
    doc.image(INPUT_DIR / "spread_equity_subplots.png", "그림 1. Horizon별 Q5-Q1 누적 spread return")
    doc.image(INPUT_DIR / "spread_drawdown_subplots.png", "그림 2. Horizon별 Q5-Q1 spread drawdown")
    doc.image(INPUT_DIR / "spread_metric_heatmaps.png", "그림 3. Q5-Q1 spread 성과 heatmap")

    doc.heading("4. Q5 Long-only 및 Return Distribution", level=1)
    doc.paragraph(
        "Q5 long-only는 실제 롱 포트폴리오 관점의 성과를 보여준다. Spread가 신호의 순수 판별력을 본다면, Q5는 시장 노출을 포함한 실전형 성과에 가깝다."
    )
    doc.image(INPUT_DIR / "q5_equity_subplots.png", "그림 4. Horizon별 Q5 누적수익률")
    doc.image(INPUT_DIR / "q5_drawdown_subplots.png", "그림 5. Horizon별 Q5 drawdown")
    doc.image(INPUT_DIR / "q5_metric_heatmaps.png", "그림 6. Q5 성과지표 heatmap")
    doc.image(REPORT_DIR / "q5_return_histograms.png", "그림 7. Q5 daily return distribution")
    doc.image(REPORT_DIR / "spread_return_histograms.png", "그림 8. Q5-Q1 daily spread return distribution")

    doc.heading("5. 왜 Positivity가 더 좋은가", level=1)
    doc.paragraph(
        "Positivity는 특정 기간 중 일간 수익률이 0 이상인 날의 비율이다. 12M positivity가 60%라면 최근 252거래일 중 약 151일이 "
        "하락하지 않았다는 뜻이다. 따라서 이 지표는 누적 수익률의 크기보다 상승 경로의 빈도와 안정성을 우선한다."
    )
    doc.paragraph(
        "첫째, return momentum은 소수의 큰 상승일에 과민하다. 한두 번의 급등이 12개월 누적수익률을 대부분 설명할 수 있고, "
        "이런 종목은 이후 차익실현과 과열 해소에 노출된다. 반면 positivity는 같은 누적수익률이라도 더 많은 날에 걸쳐 완만하게 오른 종목을 선호한다."
    )
    doc.paragraph(
        "둘째, high Sharpe와도 다르다. High Sharpe는 평균수익률을 변동성으로 나눈 비율이라 변동성 추정에 민감하고, 극단 수익률의 크기를 여전히 반영한다. "
        "Positivity는 수익률의 부호에 초점을 맞추므로 tail event의 magnitude를 덜 반영한다. 이 때문에 winner를 고를 때 수익률 분포의 오른쪽 꼬리보다 "
        "상승의 반복성과 하락 회피 성향을 더 강하게 반영한다."
    )
    doc.paragraph(
        "셋째, 행동재무 관점에서 positivity는 점진적 정보 확산을 포착한다. Da, Gurun, Warachka의 frog-in-the-pan 연구는 같은 누적 가격효과라도 "
        "정보가 여러 번에 걸쳐 작게 들어올 때 투자자가 덜 주목하고, 그 결과 더 지속적인 momentum이 나타날 수 있음을 보인다. Positivity가 높은 종목은 "
        "큰 뉴스 한 방보다 여러 거래일에 걸친 완만한 재평가 가능성이 높다."
    )
    doc.paragraph(
        "넷째, Chen, Jiang, Liu, Zhu(2026)의 positivity 논문은 positivity가 과거수익률 및 return consistency보다 장기 horizon에서 더 오래 지속되는 "
        "예측력을 가진다고 보고한다. 또한 high positivity winner가 투기적 glamour growth가 아니라, 상대적으로 덜 주목받는 value 성격과 견조한 펀더멘털, "
        "높은 이익 성장의 조합으로 설명된다고 제시한다. 이는 한국 시장에서도 급등주보다 꾸준한 개선주가 spread 안정성에 기여한다는 해석과 맞닿아 있다."
    )
    doc.paragraph(
        "다섯째, 한국 시장의 미시구조와도 맞는다. 개인투자자 비중, 테마성 순환매, 공매도 제약은 급등 이후 반락 또는 가격발견 지연을 키울 수 있다. "
        "이 환경에서는 magnitude 중심 momentum보다 빈도 중심 momentum이 과열된 급등주를 덜 담고, 상대적으로 저평가된 점진적 개선 종목을 더 잘 고를 수 있다."
    )

    doc.heading("6. 최신 12M Positivity Q5 구성종목", level=1)
    latest_date = str(portfolio["date"].iloc[0])
    doc.paragraph(f"아래 구성은 원천 데이터의 최신 신호 snapshot 기준일 {latest_date}의 12M positivity Q5 포트폴리오다.")
    doc.heading("섹터 비중", level=2)
    doc.table(_sector_table(sector_summary), widths=[2600, 1200, 1400])
    doc.heading("구성종목", level=2)
    doc.table(_portfolio_table(portfolio), widths=[900, 1700, 2400, 900, 1200, 1100])

    doc.heading("7. 참고문헌 및 데이터 출처", level=1)
    doc.table(_sources_table(), widths=[900, 2500, 5200])


def _winner_table(summary: pd.DataFrame) -> list[list[object]]:
    rows: list[list[object]] = [[
        "Leg",
        "Horizon",
        "Sharpe 1위",
        "Best CAGR",
        "Best MDD",
        "Best Sharpe",
        "Pos Rank",
        "Pos CAGR",
        "Pos MDD",
        "Pos Sharpe",
    ]]
    for leg in ["q5", "q5_minus_q1"]:
        for horizon in HORIZON_ORDER:
            sub = summary.loc[(summary["leg"].eq(leg)) & (summary["horizon"].eq(horizon))].copy()
            sub = sub.sort_values(["sharpe", "cagr"], ascending=False).reset_index(drop=True)
            best = sub.iloc[0]
            pos_idx = int(sub.index[sub["factor"].eq("positivity")][0])
            pos = sub.loc[pos_idx]
            rows.append(
                [
                    LEG_LABELS[leg],
                    horizon,
                    FACTOR_LABELS[str(best["factor"])],
                    _pct(best["cagr"]),
                    _pct(best["mdd"]),
                    _num(best["sharpe"]),
                    pos_idx + 1,
                    _pct(pos["cagr"]),
                    _pct(pos["mdd"]),
                    _num(pos["sharpe"]),
                ]
            )
    return rows


def _metrics_table(summary: pd.DataFrame, leg: str) -> list[list[object]]:
    rows: list[list[object]] = [["Horizon", "Factor", "CAGR", "MDD", "Sharpe", "Win rate", "Total return"]]
    frame = summary.loc[summary["leg"].eq(leg)].copy()
    frame["horizon"] = pd.Categorical(frame["horizon"], HORIZON_ORDER, ordered=True)
    frame["factor"] = pd.Categorical(frame["factor"], FACTOR_ORDER, ordered=True)
    frame = frame.sort_values(["horizon", "factor"])
    for _, row in frame.iterrows():
        rows.append(
            [
                row["horizon"],
                FACTOR_LABELS[str(row["factor"])],
                _pct(row["cagr"]),
                _pct(row["mdd"]),
                _num(row["sharpe"]),
                _pct(row["daily_win_rate"]),
                _pct(row["total_return"]),
            ]
        )
    return rows


def _latest_2026_positivity_12m_portfolio() -> pd.DataFrame:
    store = ParquetStore(ROOT.parquet_path)
    close = store.read("qw_adj_c").astype(float)
    membership = store.read("qw_k200_yn").reindex(index=close.index, columns=close.columns).fillna(False).astype(bool)
    names = pd.read_parquet(ROOT.parquet_path / "map__ticker_name_gics_sector_map.parquet")
    meta = names.set_index("TICKER")[["NAME", "GICS_SECTOR_NAME"]]

    returns = close.pct_change(fill_method=None)
    score = positivity_score(returns, lookback=252, min_periods=252).where(membership)
    ranks = score.rank(axis=1, method="first", pct=True)
    q5_mask = ranks.gt(0.8) & score.notna() & membership
    counts = q5_mask.sum(axis=1)
    weights = q5_mask.div(counts.where(counts.gt(0)), axis=0).fillna(0.0)
    active_2026 = weights.loc["2026":].sum(axis=1).gt(0.0)
    latest_date = weights.loc["2026":].loc[active_2026].index.max()
    row = weights.loc[latest_date]
    tickers = row.loc[row.gt(0.0)].sort_values(ascending=False).index.tolist()
    frame = pd.DataFrame(
        {
            "date": latest_date.date().isoformat(),
            "ticker": tickers,
            "code": [ticker.replace("A", "", 1) for ticker in tickers],
            "name": [str(meta.loc[ticker, "NAME"]) if ticker in meta.index else ticker for ticker in tickers],
            "sector": [
                str(meta.loc[ticker, "GICS_SECTOR_NAME"]) if ticker in meta.index else "Unknown"
                for ticker in tickers
            ],
            "weight": [float(row.loc[ticker]) for ticker in tickers],
            "positivity_12m": [float(score.loc[latest_date, ticker]) for ticker in tickers],
            "rank_pct": [float(ranks.loc[latest_date, ticker]) for ticker in tickers],
        }
    )
    return frame.sort_values(["sector", "name"]).reset_index(drop=True)


def _sector_summary(portfolio: pd.DataFrame) -> pd.DataFrame:
    return (
        portfolio.groupby("sector", as_index=False)
        .agg(count=("ticker", "count"), weight=("weight", "sum"))
        .sort_values(["weight", "count"], ascending=False)
        .reset_index(drop=True)
    )


def _sector_table(sector_summary: pd.DataFrame) -> list[list[object]]:
    rows: list[list[object]] = [["Sector", "종목 수", "비중"]]
    for _, row in sector_summary.iterrows():
        rows.append([row["sector"], int(row["count"]), _pct(row["weight"])])
    return rows


def _portfolio_table(portfolio: pd.DataFrame) -> list[list[object]]:
    rows: list[list[object]] = [["Code", "Name", "Sector", "Weight", "Positivity", "Rank pct"]]
    for _, row in portfolio.iterrows():
        rows.append(
            [
                row["code"],
                row["name"],
                row["sector"],
                _pct(row["weight"]),
                _pct(row["positivity_12m"]),
                _pct(row["rank_pct"]),
            ]
        )
    return rows


def _sources_table() -> list[list[object]]:
    return [
        ["ID", "Source", "보고서 활용"],
        [
            "S1",
            "Chen, Jiang, Liu, Zhu (2026), Positivity and long-lasting momentum, Journal of Empirical Finance. https://ideas.repec.org/a/eee/empfin/v87y2026ics0927539826000095.html",
            "Positivity 정의, 장기 예측력, high positivity winner의 value/펀더멘털 해석.",
        ],
        [
            "S2",
            "Da, Gurun, Warachka (2014), Frog in the Pan: Continuous Information and Momentum, Review of Financial Studies. https://ideas.repec.org/a/oup/rfinst/v27y2014i7p2171-2218..html",
            "점진적 정보 유입과 투자자 부주의가 더 지속적인 momentum을 만든다는 행동재무 논거.",
        ],
        [
            "S3",
            "Papailias, Liu, Thomakos (2021), Return Signal Momentum, Journal of Banking & Finance. https://ideas.repec.org/a/eee/jbfina/v124y2021ics0378426621000212.html",
            "수익률 부호 기반 momentum이 기존 time-series momentum 대비 Sharpe와 drawdown에서 강할 수 있다는 배경.",
        ],
        [
            "S4",
            "Jegadeesh and Titman (1993), Returns to Buying Winners and Selling Losers. https://ideas.repec.org/a/bla/jfinan/v48y1993i1p65-91.html",
            "전통적 3-12개월 momentum의 기준선.",
        ],
        [
            "S5",
            "Grinblatt and Moskowitz (2004), Predicting stock price movements from past returns: the role of consistency and tax-loss selling. https://ideas.repec.org/a/eee/jfinec/v71y2004i3p541-579.html",
            "과거 수익률의 부호와 일관성이 기대수익률과 관련된다는 선행 연구.",
        ],
        [
            "S6",
            "Moskowitz and Grinblatt (1999), Do Industries Explain Momentum? https://ideas.repec.org/a/bla/jfinan/v54y1999i4p1249-1290.html",
            "momentum 성과에 산업/섹터 효과가 중요하다는 구성종목 섹터 점검의 배경.",
        ],
    ]


def _pct(value: object) -> str:
    numeric = float(value)
    return f"{numeric * 100:.2f}%"


def _num(value: object) -> str:
    return f"{float(value):.3f}"


def _format_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
