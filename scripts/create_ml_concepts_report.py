from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.shapes import Circle, Drawing, Line, Polygon, Rect, String
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "pdf"
OUT_FILE = OUT_DIR / "quant_ml_engineer_core_concepts_kr_bias_variance_expanded.pdf"

FONT_REGULAR = Path(r"C:\Windows\Fonts\malgun.ttf")
FONT_BOLD = Path(r"C:\Windows\Fonts\malgunbd.ttf")


def register_fonts() -> tuple[str, str]:
    if not FONT_REGULAR.exists() or not FONT_BOLD.exists():
        raise FileNotFoundError("Malgun Gothic font files were not found in C:\\Windows\\Fonts")
    pdfmetrics.registerFont(TTFont("Malgun", str(FONT_REGULAR)))
    pdfmetrics.registerFont(TTFont("Malgun-Bold", str(FONT_BOLD)))
    return "Malgun", "Malgun-Bold"


def make_styles(font: str, bold: str) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleKR",
            parent=base["Title"],
            fontName=bold,
            fontSize=22,
            leading=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#17324D"),
            spaceAfter=10,
        ),
        "subtitle": ParagraphStyle(
            "SubtitleKR",
            parent=base["BodyText"],
            fontName=font,
            fontSize=10.5,
            leading=16,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#52616F"),
            spaceAfter=16,
        ),
        "h1": ParagraphStyle(
            "Heading1KR",
            parent=base["Heading1"],
            fontName=bold,
            fontSize=15,
            leading=20,
            textColor=colors.HexColor("#17324D"),
            spaceBefore=12,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "Heading2KR",
            parent=base["Heading2"],
            fontName=bold,
            fontSize=11.5,
            leading=16,
            textColor=colors.HexColor("#244A6B"),
            spaceBefore=8,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "BodyKR",
            parent=base["BodyText"],
            fontName=font,
            fontSize=9.4,
            leading=15,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#1F2933"),
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "SmallKR",
            parent=base["BodyText"],
            fontName=font,
            fontSize=8.3,
            leading=12.5,
            textColor=colors.HexColor("#46515C"),
        ),
        "bullet": ParagraphStyle(
            "BulletKR",
            parent=base["BodyText"],
            fontName=font,
            fontSize=8.9,
            leading=13.6,
            leftIndent=8,
            firstLineIndent=0,
            textColor=colors.HexColor("#1F2933"),
        ),
        "table_header": ParagraphStyle(
            "TableHeaderKR",
            parent=base["BodyText"],
            fontName=bold,
            fontSize=8.2,
            leading=11,
            alignment=TA_CENTER,
            textColor=colors.white,
        ),
        "table_cell": ParagraphStyle(
            "TableCellKR",
            parent=base["BodyText"],
            fontName=font,
            fontSize=7.6,
            leading=10.8,
            textColor=colors.HexColor("#1F2933"),
        ),
        "callout": ParagraphStyle(
            "CalloutKR",
            parent=base["BodyText"],
            fontName=font,
            fontSize=8.7,
            leading=13,
            textColor=colors.HexColor("#1F2933"),
            backColor=colors.HexColor("#F6F8FA"),
            borderColor=colors.HexColor("#D8DEE6"),
            borderWidth=0.7,
            borderPadding=7,
            spaceBefore=4,
            spaceAfter=8,
        ),
        "formula": ParagraphStyle(
            "FormulaKR",
            parent=base["BodyText"],
            fontName=bold,
            fontSize=9,
            leading=14,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#17324D"),
            backColor=colors.HexColor("#EEF6FF"),
            borderColor=colors.HexColor("#B7CCE5"),
            borderWidth=0.7,
            borderPadding=7,
            spaceBefore=4,
            spaceAfter=8,
        ),
    }


def p(text: str, styles: dict[str, ParagraphStyle], style: str = "body") -> Paragraph:
    return Paragraph(text, styles[style])


def diagram_title(text: str, styles: dict[str, ParagraphStyle]) -> Paragraph:
    return p(text, styles, "h2")


def add_centered_text(
    drawing: Drawing,
    x: float,
    y: float,
    text: str,
    *,
    size: float = 8,
    bold: bool = False,
    color=colors.HexColor("#17324D"),
) -> None:
    drawing.add(
        String(
            x,
            y,
            text,
            fontName="Malgun-Bold" if bold else "Malgun",
            fontSize=size,
            fillColor=color,
            textAnchor="middle",
        )
    )


def add_left_text(
    drawing: Drawing,
    x: float,
    y: float,
    text: str,
    *,
    size: float = 7.3,
    bold: bool = False,
    color=colors.HexColor("#334155"),
) -> None:
    drawing.add(
        String(
            x,
            y,
            text,
            fontName="Malgun-Bold" if bold else "Malgun",
            fontSize=size,
            fillColor=color,
        )
    )


def add_arrow(drawing: Drawing, x1: float, y1: float, x2: float, y2: float, color=colors.HexColor("#64748B")) -> None:
    drawing.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=1.2))
    head = 4
    drawing.add(
        Polygon(
            points=[x2, y2, x2 - head, y2 + head / 2, x2 - head, y2 - head / 2],
            fillColor=color,
            strokeColor=color,
        )
    )


def pipeline_diagram() -> Drawing:
    width, height = 180 * mm, 42 * mm
    drawing = Drawing(width, height)
    box_w, box_h = 28 * mm, 14 * mm
    y = 16 * mm
    labels = [
        ("Data", "prices, factors"),
        ("Features X", "known at time t"),
        ("Model f", "learn pattern"),
        ("Prediction", "return/risk/class"),
        ("Evaluation", "future period"),
    ]
    x_positions = [4 * mm, 40 * mm, 76 * mm, 112 * mm, 148 * mm]
    for index, (title, subtitle) in enumerate(labels):
        x = x_positions[index]
        drawing.add(Rect(x, y, box_w, box_h, rx=2, ry=2, fillColor=colors.HexColor("#F8FAFC"), strokeColor=colors.HexColor("#94A3B8")))
        add_centered_text(drawing, x + box_w / 2, y + 8.5 * mm, title, size=8, bold=True)
        add_centered_text(drawing, x + box_w / 2, y + 4 * mm, subtitle, size=6.4, color=colors.HexColor("#475569"))
        if index < len(labels) - 1:
            add_arrow(drawing, x + box_w + 2 * mm, y + box_h / 2, x_positions[index + 1] - 2 * mm, y + box_h / 2)
    add_left_text(drawing, 4 * mm, 5 * mm, "Core loop: define a target, learn from past data, and test on unseen future data.", size=7)
    return drawing


def supervised_unsupervised_diagram() -> Drawing:
    width, height = 180 * mm, 58 * mm
    drawing = Drawing(width, height)
    panel_w = 84 * mm
    for x0, title in [(0, "Supervised: X -> y"), (96 * mm, "Unsupervised: structure only")]:
        drawing.add(Rect(x0 + 2 * mm, 7 * mm, panel_w, 42 * mm, fillColor=colors.white, strokeColor=colors.HexColor("#CBD5E1")))
        add_centered_text(drawing, x0 + 44 * mm, 52 * mm, title, size=9, bold=True)
        drawing.add(Line(x0 + 10 * mm, 14 * mm, x0 + 78 * mm, 14 * mm, strokeColor=colors.HexColor("#94A3B8")))
        drawing.add(Line(x0 + 10 * mm, 14 * mm, x0 + 10 * mm, 43 * mm, strokeColor=colors.HexColor("#94A3B8")))

    supervised_points = [
        (17, 18, "#2563EB"), (24, 22, "#2563EB"), (31, 21, "#2563EB"), (39, 27, "#2563EB"),
        (49, 32, "#DC2626"), (57, 35, "#DC2626"), (65, 37, "#DC2626"), (72, 40, "#DC2626"),
    ]
    for x, y, color in supervised_points:
        drawing.add(Circle(x * mm, y * mm, 2.2, fillColor=colors.HexColor(color), strokeColor=colors.white))
    drawing.add(Line(15 * mm, 16 * mm, 76 * mm, 42 * mm, strokeColor=colors.HexColor("#111827"), strokeWidth=1.2))
    add_left_text(drawing, 15 * mm, 9 * mm, "Labels define what the model must predict.", size=6.7)

    cluster_points = [
        (110, 20, "#2563EB"), (116, 23, "#2563EB"), (122, 19, "#2563EB"), (118, 28, "#2563EB"),
        (142, 34, "#DC2626"), (150, 38, "#DC2626"), (155, 33, "#DC2626"), (147, 29, "#DC2626"),
        (162, 19, "#059669"), (169, 22, "#059669"), (158, 15, "#059669"), (172, 16, "#059669"),
    ]
    for x, y, color in cluster_points:
        drawing.add(Circle(x * mm, y * mm, 2.2, fillColor=colors.HexColor(color), strokeColor=colors.white))
    add_left_text(drawing, 107 * mm, 9 * mm, "No target y: the algorithm searches for groups or axes.", size=6.7)
    return drawing


def regression_classification_diagram() -> Drawing:
    width, height = 180 * mm, 58 * mm
    drawing = Drawing(width, height)
    for x0, title in [(0, "Regression: predict a number"), (96 * mm, "Classification: predict a class/probability")]:
        drawing.add(Rect(x0 + 2 * mm, 7 * mm, 84 * mm, 42 * mm, fillColor=colors.white, strokeColor=colors.HexColor("#CBD5E1")))
        add_centered_text(drawing, x0 + 44 * mm, 52 * mm, title, size=8.5, bold=True)
        drawing.add(Line(x0 + 10 * mm, 14 * mm, x0 + 78 * mm, 14 * mm, strokeColor=colors.HexColor("#94A3B8")))
        drawing.add(Line(x0 + 10 * mm, 14 * mm, x0 + 10 * mm, 43 * mm, strokeColor=colors.HexColor("#94A3B8")))

    reg_points = [(17, 18), (24, 21), (31, 22), (38, 27), (45, 25), (53, 32), (61, 35), (70, 38)]
    for x, y in reg_points:
        drawing.add(Circle(x * mm, y * mm, 2.1, fillColor=colors.HexColor("#2563EB"), strokeColor=colors.white))
    drawing.add(Line(15 * mm, 17 * mm, 76 * mm, 40 * mm, strokeColor=colors.HexColor("#DC2626"), strokeWidth=1.5))
    add_left_text(drawing, 15 * mm, 9 * mm, "Example y: next-month return = 1.4%", size=6.7)

    class_points = [
        (112, 19, "#2563EB"), (118, 23, "#2563EB"), (126, 21, "#2563EB"), (132, 27, "#2563EB"),
        (146, 30, "#DC2626"), (152, 34, "#DC2626"), (159, 37, "#DC2626"), (168, 36, "#DC2626"),
    ]
    for x, y, color in class_points:
        drawing.add(Circle(x * mm, y * mm, 2.2, fillColor=colors.HexColor(color), strokeColor=colors.white))
    drawing.add(Line(139 * mm, 14 * mm, 139 * mm, 43 * mm, strokeColor=colors.HexColor("#111827"), strokeWidth=1.2))
    add_left_text(drawing, 107 * mm, 9 * mm, "Example y: top quintile = 1, otherwise = 0", size=6.7)
    return drawing


def loss_curve_diagram() -> Drawing:
    width, height = 180 * mm, 48 * mm
    drawing = Drawing(width, height)
    drawing.add(Rect(8 * mm, 7 * mm, 164 * mm, 34 * mm, fillColor=colors.white, strokeColor=colors.HexColor("#CBD5E1")))
    drawing.add(Line(20 * mm, 15 * mm, 160 * mm, 15 * mm, strokeColor=colors.HexColor("#94A3B8")))
    drawing.add(Line(20 * mm, 15 * mm, 20 * mm, 36 * mm, strokeColor=colors.HexColor("#94A3B8")))
    points = [(30, 34), (45, 27), (60, 22), (76, 18), (92, 16), (108, 18), (126, 23), (145, 32)]
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        drawing.add(Line(x1 * mm, y1 * mm, x2 * mm, y2 * mm, strokeColor=colors.HexColor("#2563EB"), strokeWidth=1.8))
    drawing.add(Circle(92 * mm, 16 * mm, 3.2, fillColor=colors.HexColor("#DC2626"), strokeColor=colors.white))
    add_left_text(drawing, 83 * mm, 9 * mm, "minimum loss", size=6.8, bold=True, color=colors.HexColor("#DC2626"))
    add_left_text(drawing, 22 * mm, 38 * mm, "Loss", size=7, bold=True)
    add_left_text(drawing, 147 * mm, 10 * mm, "model parameter", size=7)
    add_left_text(drawing, 32 * mm, 43 * mm, "Training changes parameters in the direction that lowers this curve.", size=7)
    return drawing


def train_validation_test_diagram() -> Drawing:
    width, height = 180 * mm, 43 * mm
    drawing = Drawing(width, height)
    y, h = 21 * mm, 10 * mm
    segments = [
        (12 * mm, 88 * mm, "#2563EB", "Train", "learn parameters"),
        (100 * mm, 38 * mm, "#F59E0B", "Validation", "select model"),
        (138 * mm, 30 * mm, "#059669", "Test", "final estimate"),
    ]
    drawing.add(Line(12 * mm, 17 * mm, 168 * mm, 17 * mm, strokeColor=colors.HexColor("#64748B"), strokeWidth=1.1))
    for x, w, color, title, subtitle in segments:
        drawing.add(Rect(x, y, w, h, fillColor=colors.HexColor(color), strokeColor=colors.white))
        add_centered_text(drawing, x + w / 2, y + 6.2 * mm, title, size=8, bold=True, color=colors.white)
        add_centered_text(drawing, x + w / 2, y - 4 * mm, subtitle, size=6.6, color=colors.HexColor("#334155"))
    add_left_text(drawing, 12 * mm, 36 * mm, "For finance, preserve time order: past -> present -> future.", size=7.5, bold=True)
    add_left_text(drawing, 12 * mm, 6 * mm, "The test set should stay unused until the very end.", size=7)
    return drawing


def overfitting_diagram() -> Drawing:
    width, height = 180 * mm, 50 * mm
    drawing = Drawing(width, height)
    drawing.add(Rect(8 * mm, 7 * mm, 164 * mm, 35 * mm, fillColor=colors.white, strokeColor=colors.HexColor("#CBD5E1")))
    drawing.add(Line(20 * mm, 15 * mm, 160 * mm, 15 * mm, strokeColor=colors.HexColor("#94A3B8")))
    drawing.add(Line(20 * mm, 15 * mm, 20 * mm, 37 * mm, strokeColor=colors.HexColor("#94A3B8")))
    train = [(28, 34), (45, 28), (62, 23), (80, 20), (100, 18), (124, 16), (150, 15)]
    val = [(28, 35), (45, 29), (62, 25), (80, 23), (100, 24), (124, 28), (150, 34)]
    for points, color in [(train, "#2563EB"), (val, "#DC2626")]:
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            drawing.add(Line(x1 * mm, y1 * mm, x2 * mm, y2 * mm, strokeColor=colors.HexColor(color), strokeWidth=1.8))
    add_left_text(drawing, 126 * mm, 17 * mm, "Train loss", size=7, bold=True, color=colors.HexColor("#2563EB"))
    add_left_text(drawing, 126 * mm, 35 * mm, "Validation loss", size=7, bold=True, color=colors.HexColor("#DC2626"))
    add_left_text(drawing, 22 * mm, 39 * mm, "Loss", size=7, bold=True)
    add_left_text(drawing, 132 * mm, 10 * mm, "model complexity", size=7)
    add_left_text(drawing, 30 * mm, 45 * mm, "Overfitting starts when validation loss rises while train loss keeps falling.", size=7)
    return drawing


def bias_variance_diagram() -> Drawing:
    width, height = 180 * mm, 56 * mm
    drawing = Drawing(width, height)
    panels = [(3 * mm, "High Bias"), (62 * mm, "Balanced"), (121 * mm, "High Variance")]
    for x0, title in panels:
        drawing.add(Rect(x0, 8 * mm, 52 * mm, 36 * mm, fillColor=colors.white, strokeColor=colors.HexColor("#CBD5E1")))
        add_centered_text(drawing, x0 + 26 * mm, 47 * mm, title, size=8.5, bold=True)
        drawing.add(Line(x0 + 6 * mm, 13 * mm, x0 + 47 * mm, 13 * mm, strokeColor=colors.HexColor("#94A3B8")))
        drawing.add(Line(x0 + 6 * mm, 13 * mm, x0 + 6 * mm, 38 * mm, strokeColor=colors.HexColor("#94A3B8")))
        for x, y in [(12, 16), (18, 21), (24, 20), (30, 28), (36, 31), (43, 35)]:
            drawing.add(Circle(x0 + x * mm, y * mm, 1.8, fillColor=colors.HexColor("#2563EB"), strokeColor=colors.white))

    drawing.add(Line(11 * mm, 22 * mm, 48 * mm, 26 * mm, strokeColor=colors.HexColor("#DC2626"), strokeWidth=1.4))
    balanced = [(70, 17), (78, 20), (86, 24), (95, 29), (104, 33)]
    for (x1, y1), (x2, y2) in zip(balanced, balanced[1:]):
        drawing.add(Line(x1 * mm, y1 * mm, x2 * mm, y2 * mm, strokeColor=colors.HexColor("#DC2626"), strokeWidth=1.4))
    wavy = [(129, 18), (134, 31), (141, 20), (148, 37), (157, 22), (165, 36)]
    for (x1, y1), (x2, y2) in zip(wavy, wavy[1:]):
        drawing.add(Line(x1 * mm, y1 * mm, x2 * mm, y2 * mm, strokeColor=colors.HexColor("#DC2626"), strokeWidth=1.4))
    add_left_text(drawing, 6 * mm, 3 * mm, "Too simple", size=6.7)
    add_left_text(drawing, 73 * mm, 3 * mm, "Captures signal", size=6.7)
    add_left_text(drawing, 130 * mm, 3 * mm, "Chases noise", size=6.7)
    return drawing


def bias_variance_target_diagram() -> Drawing:
    width, height = 180 * mm, 66 * mm
    drawing = Drawing(width, height)
    panels = [
        (4 * mm, "Low Bias\nLow Variance", [(24, 30), (25, 31), (23, 29), (26, 30), (24, 32)]),
        (48 * mm, "Low Bias\nHigh Variance", [(66, 37), (73, 25), (65, 21), (78, 33), (70, 42)]),
        (92 * mm, "High Bias\nLow Variance", [(119, 33), (121, 34), (120, 31), (118, 32), (122, 33)]),
        (136 * mm, "High Bias\nHigh Variance", [(151, 38), (164, 24), (172, 36), (158, 18), (169, 44)]),
    ]
    for x0, title, dots in panels:
        cx, cy = x0 + 22 * mm, 31 * mm
        drawing.add(Rect(x0, 10 * mm, 40 * mm, 45 * mm, fillColor=colors.white, strokeColor=colors.HexColor("#CBD5E1")))
        for radius, stroke in [(15 * mm, "#CBD5E1"), (10 * mm, "#94A3B8"), (5 * mm, "#64748B")]:
            drawing.add(Circle(cx, cy, radius, fillColor=None, strokeColor=colors.HexColor(stroke), strokeWidth=0.9))
        drawing.add(Circle(cx, cy, 2.2, fillColor=colors.HexColor("#DC2626"), strokeColor=colors.white))
        first, second = title.split("\n")
        add_centered_text(drawing, x0 + 20 * mm, 60 * mm, first, size=7.4, bold=True)
        add_centered_text(drawing, x0 + 20 * mm, 56 * mm, second, size=7.4, bold=True)
        for x, y in dots:
            drawing.add(Circle(x * mm, y * mm, 1.8, fillColor=colors.HexColor("#2563EB"), strokeColor=colors.white))
    add_left_text(drawing, 8 * mm, 4 * mm, "Red center = true target. Blue shots = model predictions from different training samples.", size=7)
    return drawing


def bias_variance_decomposition_diagram() -> Drawing:
    width, height = 180 * mm, 52 * mm
    drawing = Drawing(width, height)
    colors_map = [
        ("Bias^2", "#F59E0B", "wrong average prediction"),
        ("Variance", "#2563EB", "unstable across samples"),
        ("Noise", "#64748B", "irreducible randomness"),
    ]
    x0, y0 = 18 * mm, 15 * mm
    h = 16 * mm
    widths = [46 * mm, 57 * mm, 41 * mm]
    cursor = x0
    for (label, color, subtitle), segment_w in zip(colors_map, widths):
        drawing.add(Rect(cursor, y0, segment_w, h, fillColor=colors.HexColor(color), strokeColor=colors.white))
        add_centered_text(drawing, cursor + segment_w / 2, y0 + 9.5 * mm, label, size=9, bold=True, color=colors.white)
        add_centered_text(drawing, cursor + segment_w / 2, y0 + 4 * mm, subtitle, size=5.9, color=colors.white)
        cursor += segment_w
    add_centered_text(drawing, width / 2, 39 * mm, "Expected prediction error = Bias^2 + Variance + irreducible noise", size=9, bold=True)
    add_left_text(drawing, 18 * mm, 7 * mm, "Only Bias^2 and Variance are controlled by model design. Noise is part of the market/data process.", size=7)
    return drawing


def model_complexity_error_diagram() -> Drawing:
    width, height = 180 * mm, 58 * mm
    drawing = Drawing(width, height)
    drawing.add(Rect(8 * mm, 7 * mm, 164 * mm, 41 * mm, fillColor=colors.white, strokeColor=colors.HexColor("#CBD5E1")))
    drawing.add(Line(20 * mm, 15 * mm, 160 * mm, 15 * mm, strokeColor=colors.HexColor("#94A3B8")))
    drawing.add(Line(20 * mm, 15 * mm, 20 * mm, 42 * mm, strokeColor=colors.HexColor("#94A3B8")))
    bias = [(28, 38), (48, 33), (70, 27), (92, 22), (116, 18), (145, 16)]
    variance = [(28, 16), (48, 17), (70, 19), (92, 23), (116, 30), (145, 39)]
    total = [(28, 40), (48, 35), (70, 30), (92, 27), (116, 29), (145, 36)]
    for points, color, width_line in [(bias, "#F59E0B", 1.7), (variance, "#2563EB", 1.7), (total, "#DC2626", 2.1)]:
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            drawing.add(Line(x1 * mm, y1 * mm, x2 * mm, y2 * mm, strokeColor=colors.HexColor(color), strokeWidth=width_line))
    drawing.add(Line(94 * mm, 15 * mm, 94 * mm, 43 * mm, strokeColor=colors.HexColor("#111827"), strokeWidth=0.8, strokeDashArray=[3, 2]))
    add_left_text(drawing, 98 * mm, 43 * mm, "sweet spot", size=7, bold=True)
    add_left_text(drawing, 128 * mm, 39 * mm, "Variance", size=7, bold=True, color=colors.HexColor("#2563EB"))
    add_left_text(drawing, 35 * mm, 36 * mm, "Bias^2", size=7, bold=True, color=colors.HexColor("#F59E0B"))
    add_left_text(drawing, 126 * mm, 31 * mm, "Test error", size=7, bold=True, color=colors.HexColor("#DC2626"))
    add_left_text(drawing, 22 * mm, 44 * mm, "Error", size=7, bold=True)
    add_left_text(drawing, 124 * mm, 10 * mm, "model complexity", size=7)
    add_left_text(drawing, 30 * mm, 3 * mm, "As complexity rises, Bias tends to fall but Variance tends to rise.", size=7)
    return drawing


def sample_variance_diagram() -> Drawing:
    width, height = 180 * mm, 56 * mm
    drawing = Drawing(width, height)
    panels = [(4 * mm, "Dataset A"), (63 * mm, "Dataset B"), (122 * mm, "Dataset C")]
    line_sets = [
        [(13, 18), (24, 24), (35, 27), (47, 34)],
        [(72, 33), (84, 27), (96, 24), (108, 18)],
        [(131, 19), (143, 33), (155, 18), (167, 35)],
    ]
    for (x0, title), line_points in zip(panels, line_sets):
        drawing.add(Rect(x0, 9 * mm, 52 * mm, 35 * mm, fillColor=colors.white, strokeColor=colors.HexColor("#CBD5E1")))
        add_centered_text(drawing, x0 + 26 * mm, 47 * mm, title, size=8.2, bold=True)
        drawing.add(Line(x0 + 7 * mm, 14 * mm, x0 + 46 * mm, 14 * mm, strokeColor=colors.HexColor("#94A3B8")))
        drawing.add(Line(x0 + 7 * mm, 14 * mm, x0 + 7 * mm, 38 * mm, strokeColor=colors.HexColor("#94A3B8")))
        for x, y in [(12, 17), (18, 22), (24, 20), (31, 30), (39, 29), (45, 36)]:
            drawing.add(Circle(x0 + x * mm, y * mm, 1.7, fillColor=colors.HexColor("#2563EB"), strokeColor=colors.white))
        for (x1, y1), (x2, y2) in zip(line_points, line_points[1:]):
            drawing.add(Line(x1 * mm, y1 * mm, x2 * mm, y2 * mm, strokeColor=colors.HexColor("#DC2626"), strokeWidth=1.4))
    add_left_text(drawing, 10 * mm, 3 * mm, "High Variance: a small change in training sample creates a very different fitted model.", size=7)
    return drawing


def cross_validation_diagram() -> Drawing:
    width, height = 180 * mm, 62 * mm
    drawing = Drawing(width, height)
    x0, y0 = 31 * mm, 12 * mm
    cell_w, cell_h = 16 * mm, 8 * mm
    add_left_text(drawing, 6 * mm, 52 * mm, "Time Series Split example", size=8.5, bold=True)
    for i in range(8):
        add_centered_text(drawing, x0 + i * cell_w + cell_w / 2, 49 * mm, f"T{i+1}", size=6.5)
    for row in range(4):
        add_left_text(drawing, 6 * mm, y0 + (3 - row) * 10 * mm + 2 * mm, f"Fold {row+1}", size=7.2, bold=True)
        for col in range(8):
            fill = colors.HexColor("#E2E8F0")
            label = ""
            if col <= row + 2:
                fill = colors.HexColor("#2563EB")
                label = "train" if col == row + 1 else ""
            if col == row + 3:
                fill = colors.HexColor("#F59E0B")
                label = "valid"
            drawing.add(Rect(x0 + col * cell_w, y0 + (3 - row) * 10 * mm, cell_w - 1, cell_h, fillColor=fill, strokeColor=colors.white))
            if label:
                add_centered_text(drawing, x0 + col * cell_w + cell_w / 2, y0 + (3 - row) * 10 * mm + 2.5 * mm, label, size=5.5, color=colors.white)
    add_left_text(drawing, 31 * mm, 5 * mm, "Each fold trains on the past and validates on the next unseen period.", size=7)
    return drawing


def bullets(items: list[str], styles: dict[str, ParagraphStyle]) -> ListFlowable:
    return ListFlowable(
        [ListItem(p(item, styles, "bullet"), leftIndent=4) for item in items],
        bulletType="bullet",
        start="circle",
        leftIndent=14,
        bulletFontName="Malgun",
        bulletFontSize=5,
        spaceAfter=6,
    )


def make_table(
    rows: list[list[str]],
    widths: list[float],
    styles: dict[str, ParagraphStyle],
    header: bool = True,
) -> Table:
    data = []
    for row_index, row in enumerate(rows):
        style_name = "table_header" if header and row_index == 0 else "table_cell"
        data.append([p(cell, styles, style_name) for cell in row])
    table = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324D") if header else colors.white),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white if header else colors.HexColor("#1F2933")),
                ("FONTNAME", (0, 0), (-1, -1), "Malgun"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D8DEE6")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def add_page_number(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Malgun", 8)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(18 * mm, 12 * mm, "Quant ML Engineer Core Concepts")
    canvas.drawRightString(A4[0] - 18 * mm, 12 * mm, f"{doc.page}")
    canvas.restoreState()


def build_story(styles: dict[str, ParagraphStyle]) -> list:
    story = []

    story.append(p("퀀트 기반 ML Engineer 입문 보고서", styles, "title"))
    story.append(
        p(
            "머신러닝의 핵심 개념을 Supervised / Unsupervised, Regression / Classification, "
            "Loss Function, Overfitting, Bias-Variance Tradeoff, Train / Validation / Test, "
            "Cross Validation 중심으로 정리한 학습용 문서 - Bias-Variance 수식/그림 확장판",
            styles,
            "subtitle",
        )
    )
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1"), spaceAfter=10))
    story.append(
        p(
            "<b>목표</b>: 이 보고서는 수학과 코딩을 막 시작하는 단계에서도 읽을 수 있게 직관을 먼저 설명하고, "
            "퀀트 모델링에서 왜 중요한지 연결한다. 예시는 투자 조언이 아니라 ML 학습용 예시다.",
            styles,
            "callout",
        )
    )
    story.append(
        p(
            "<b>이번 확장판에서 크게 바뀐 부분</b>: 7번 Bias-Variance Tradeoff 섹션에 오차 분해 수식, "
            "Bias^2 / Variance / Noise 구성 그림, 모델 복잡도 곡선, 표본 변화에 따른 모델 흔들림 그림, "
            "퀀트 팩터 모델 진단 표를 추가했다.",
            styles,
            "callout",
        )
    )

    story.append(p("1. 머신러닝이란?", styles, "h1"))
    story.append(
        p(
            "머신러닝은 사람이 모든 규칙을 직접 코딩하는 대신, 데이터에서 패턴을 학습해 예측이나 분류, "
            "군집화 같은 결정을 수행하게 만드는 방법이다. 핵심은 <b>데이터 -> 학습 -> 일반화</b>다. "
            "학습 데이터에서는 잘 맞지만 새로운 데이터에서 무너지면 좋은 머신러닝이 아니다.",
            styles,
        )
    )
    story.append(diagram_title("그림 1. 머신러닝의 기본 흐름", styles))
    story.append(pipeline_diagram())
    story.append(
        p(
            "위 흐름에서 가장 중요한 경계는 <b>학습에 사용한 과거</b>와 <b>아직 보지 않은 미래</b>다. "
            "퀀트 모델은 과거 가격, 재무제표, 거래량, 거시 지표 같은 데이터를 특성 X로 만들고, "
            "다음 기간 수익률이나 위험 같은 목표 y를 예측한다. 이때 모델이 배운 것은 확정 규칙이 아니라 "
            "확률적인 관계다. 그래서 한 번의 예측이 맞았는가보다 여러 기간에서 안정적으로 일반화되는지가 더 중요하다.",
            styles,
        )
    )
    story.append(
        make_table(
            [
                ["구분", "전통적 프로그래밍", "머신러닝"],
                ["입력", "규칙과 데이터", "데이터와 정답 또는 데이터 구조"],
                ["출력", "규칙에 따른 결과", "학습된 모델 또는 예측 함수"],
                ["퀀트 예시", "PER < 10이면 매수 같은 고정 규칙", "과거 재무/가격/거래량으로 다음 기간 수익률 또는 위험을 예측"],
                ["평가 기준", "규칙이 의도대로 실행되는가", "보지 못한 기간과 종목에서도 성능이 유지되는가"],
            ],
            [30 * mm, 70 * mm, 80 * mm],
            styles,
        )
    )

    story.append(p("2. 지도학습과 비지도학습", styles, "h1"))
    story.append(
        p(
            "가장 먼저 볼 구분은 정답 라벨이 있는가다. 지도학습은 입력 X와 정답 y의 관계를 배우고, "
            "비지도학습은 정답 없이 데이터 내부 구조를 찾는다.",
            styles,
        )
    )
    story.append(diagram_title("그림 2. 정답 라벨이 있으면 지도학습, 없으면 비지도학습", styles))
    story.append(supervised_unsupervised_diagram())
    story.append(
        p(
            "왼쪽 그림의 지도학습에서는 각 점이 이미 정답 라벨을 갖고 있다. 예를 들어 파란 점은 다음 달 시장보다 "
            "못 오른 종목, 빨간 점은 시장보다 잘 오른 종목이라고 정의할 수 있다. 모델은 두 그룹을 가르는 경계나 "
            "확률 함수를 배운다. 오른쪽 그림의 비지도학습에서는 정답이 없으므로 알고리즘이 데이터의 거리, 상관, "
            "밀도 같은 구조를 이용해 그룹을 찾는다.",
            styles,
        )
    )
    story.append(
        make_table(
            [
                ["개념", "지도학습: Supervised Learning", "비지도학습: Unsupervised Learning"],
                ["정답 y", "있음. 예: 다음 달 수익률, 부도 여부, 상승/하락 라벨", "없음. 데이터의 패턴, 군집, 축, 이상치를 찾음"],
                ["대표 과제", "회귀, 분류", "군집화, 차원 축소, 이상치 탐지"],
                ["퀀트 예시", "팩터 데이터로 다음 달 초과수익률 예측", "종목을 유사한 리스크/스타일 그룹으로 묶기"],
                ["주의점", "라벨 정의와 데이터 누수에 민감", "결과 해석이 주관적일 수 있음"],
            ],
            [25 * mm, 77 * mm, 77 * mm],
            styles,
        )
    )
    story.append(
        p(
            "<b>퀀트 관점</b>: 지도학습을 할 때 y를 어떻게 정의하느냐가 모델의 목적을 결정한다. "
            "예를 들어 y가 다음 달 수익률이면 회귀 문제가 되고, y가 상위 20% 여부면 분류 문제가 된다.",
            styles,
            "callout",
        )
    )

    story.append(PageBreak())
    story.append(p("3. 회귀와 분류", styles, "h1"))
    story.append(
        p(
            "지도학습은 보통 회귀와 분류로 나뉜다. 회귀는 연속적인 숫자를 예측하고, 분류는 카테고리나 "
            "확률을 예측한다. 같은 금융 문제도 라벨을 어떻게 만들면 회귀가 될 수도, 분류가 될 수도 있다.",
            styles,
        )
    )
    story.append(diagram_title("그림 3. 회귀는 숫자, 분류는 경계나 확률", styles))
    story.append(regression_classification_diagram())
    story.append(
        p(
            "회귀 모델은 점들을 가장 잘 설명하는 연속적인 예측 함수를 만든다. 예측값은 다음 달 수익률 1.4%, "
            "예상 변동성 18.2%처럼 숫자로 나온다. 분류 모델은 특정 클래스에 속할 가능성을 계산한다. "
            "예측값이 상승 확률 63%라면, 그 숫자를 실제 매수/매도 결정으로 바꾸는 임계값은 별도로 정해야 한다.",
            styles,
        )
    )
    story.append(
        make_table(
            [
                ["구분", "Regression", "Classification"],
                ["예측값", "연속값. 예: 0.012, -0.034, 변동성 18%", "클래스 또는 확률. 예: 상승 확률 63%, 리스크 그룹 B"],
                ["대표 손실", "MSE, MAE, Huber Loss", "Log Loss, Cross Entropy, Hinge Loss"],
                ["대표 지표", "RMSE, MAE, R2, Rank IC", "Accuracy, Precision, Recall, AUC, F1"],
                ["퀀트 사용", "수익률, 변동성, 거래비용 추정", "상승/하락, 부도/정상, 리밸런싱 후보 분류"],
            ],
            [25 * mm, 77 * mm, 77 * mm],
            styles,
        )
    )
    story.append(
        bullets(
            [
                "수익률 예측은 노이즈가 매우 크므로 RMSE가 낮아도 실제 포트폴리오 성과와 바로 연결되지 않을 수 있다.",
                "분류 확률은 임계값 선택이 중요하다. 상승 확률 51%를 모두 매수하면 거래비용 때문에 손실이 날 수 있다.",
                "퀀트에서는 예측 정확도뿐 아니라 순위 품질, 턴오버, 비용 차감 후 성과를 함께 봐야 한다.",
            ],
            styles,
        )
    )
    story.append(p("라벨 설계 예시", styles, "h2"))
    story.append(
        make_table(
            [
                ["목적", "라벨 y 정의", "문제 유형", "해석"],
                ["수익률 크기 예측", "다음 20거래일 수익률", "회귀", "예측값이 높을수록 기대수익률이 높다고 해석"],
                ["상위 종목 선별", "다음 달 수익률이 전체 종목 중 상위 20%면 1", "분류", "상위 그룹에 들어갈 확률을 추정"],
                ["리스크 관리", "다음 20거래일 변동성", "회귀", "포지션 크기나 위험 한도에 활용"],
                ["이상 상황 탐지", "급락/거래정지/부도 이벤트 여부", "분류", "드문 이벤트라 클래스 불균형을 고려"],
            ],
            [36 * mm, 57 * mm, 28 * mm, 58 * mm],
            styles,
        )
    )
    story.append(
        p(
            "처음 공부할 때는 같은 원자료로 라벨만 바꾸어 회귀와 분류를 둘 다 만들어보면 차이가 빨리 잡힌다. "
            "단, 라벨 계산에는 반드시 예측 시점 이후의 값만 쓰고, 특성 X에는 예측 시점 이전 정보만 둔다.",
            styles,
            "callout",
        )
    )

    story.append(PageBreak())
    story.append(p("4. Loss Function: 모델이 무엇을 틀렸다고 보는가", styles, "h1"))
    story.append(
        p(
            "손실 함수는 모델의 예측이 정답과 얼마나 다른지를 숫자로 나타낸다. 학습 알고리즘은 보통 이 손실을 "
            "줄이는 방향으로 파라미터를 조정한다. 따라서 손실 함수는 모델에게 주는 업무 지시서에 가깝다.",
            styles,
        )
    )
    story.append(diagram_title("그림 4. 손실 함수는 모델이 내려가려는 지형", styles))
    story.append(loss_curve_diagram())
    story.append(
        p(
            "손실 곡선은 파라미터 선택에 따라 모델이 얼마나 틀리는지를 보여준다. 학습은 이 곡선에서 낮은 지점을 "
            "찾는 과정이다. 다만 금융 데이터에서는 학습 손실이 낮아졌다고 바로 좋은 전략이 되는 것은 아니다. "
            "모델이 줄인 손실이 실제 목적, 예를 들어 비용 차감 후 수익률이나 리스크 조정 성과와 연결되는지 확인해야 한다.",
            styles,
        )
    )
    story.append(
        make_table(
            [
                ["손실 함수", "직관", "사용 상황", "주의점"],
                ["MSE = mean((y - pred)^2)", "큰 오차를 강하게 벌함", "수익률/가격/변동성 회귀", "이상치에 민감"],
                ["MAE = mean(abs(y - pred))", "오차 크기를 선형으로 벌함", "노이즈와 이상치가 큰 회귀", "미분이 불안정한 구간이 있음"],
                ["Log Loss", "정답 클래스에 낮은 확률을 주면 크게 벌함", "상승/하락, 부도 여부 같은 분류", "확률 보정이 중요"],
                ["Ranking Loss", "정확한 값보다 순위를 중시", "종목 랭킹, 롱숏 후보 선정", "평가와 구현이 복잡할 수 있음"],
            ],
            [35 * mm, 48 * mm, 51 * mm, 45 * mm],
            styles,
        )
    )
    story.append(
        p(
            "<b>핵심 질문</b>: 내가 진짜 줄이고 싶은 손실은 무엇인가? 예측 오차인가, 잘못된 방향성인가, "
            "상위 종목을 놓치는 것인가, 거래비용을 감안한 포트폴리오 손실인가?",
            styles,
            "callout",
        )
    )

    story.append(p("5. Train / Validation / Test", styles, "h1"))
    story.append(
        p(
            "데이터를 한 번에 모두 학습에 쓰면 모델이 진짜로 일반화되는지 알 수 없다. 그래서 보통 학습용, "
            "검증용, 테스트용으로 나눈다.",
            styles,
        )
    )
    story.append(diagram_title("그림 5. Train / Validation / Test 시간 분할", styles))
    story.append(train_validation_test_diagram())
    story.append(
        p(
            "Train은 모델 파라미터를 학습하는 구간, Validation은 모델 종류와 하이퍼파라미터를 고르는 구간, "
            "Test는 마지막에 한 번만 보는 최종 평가 구간이다. 퀀트에서는 시간 순서를 지키는 것이 특히 중요하다. "
            "미래의 재무제표 수정치, 미래 구성 종목, 미래 생존 종목만 남긴 데이터가 학습에 섞이면 성능이 실제보다 좋아 보인다.",
            styles,
        )
    )
    story.append(
        make_table(
            [
                ["구간", "역할", "모델 개발 중 사용 여부", "퀀트에서의 주의점"],
                ["Train", "모델이 패턴을 학습하는 데이터", "매우 자주 사용", "미래 정보를 포함하면 안 됨"],
                ["Validation", "하이퍼파라미터와 모델 선택", "개발 중 반복 사용", "반복 사용이 많으면 검증 구간에도 과적합됨"],
                ["Test", "최종 성능 추정", "마지막에만 사용", "한 번 본 테스트는 더 이상 순수한 테스트가 아님"],
            ],
            [26 * mm, 56 * mm, 45 * mm, 52 * mm],
            styles,
        )
    )
    story.append(
        p(
            "시계열 금융 데이터에서는 무작위 셔플이 위험하다. 2025년 데이터를 학습에 넣고 2022년을 테스트하면 "
            "미래 정보를 과거로 흘려보내는 셈이 될 수 있다. 일반적으로 과거 -> 현재 -> 미래 순서로 나누고, "
            "특성 생성도 각 시점에서 알 수 있었던 정보만 사용한다.",
            styles,
        )
    )

    story.append(p("6. Overfitting: 학습 데이터 암기", styles, "h1"))
    story.append(
        p(
            "과적합은 모델이 학습 데이터의 진짜 패턴뿐 아니라 우연한 노이즈까지 외워서, 새로운 데이터에서는 "
            "성능이 떨어지는 현상이다. 퀀트에서는 백테스트가 좋아 보이는데 실거래나 이후 기간에서 망가지는 "
            "대표 원인이다.",
            styles,
        )
    )
    story.append(diagram_title("그림 6. Train 손실과 Validation 손실이 갈라지는 순간", styles))
    story.append(overfitting_diagram())
    story.append(
        p(
            "복잡한 모델은 학습 데이터에서는 계속 더 잘 맞을 수 있다. 하지만 어느 순간부터 검증 손실이 다시 올라가면 "
            "그 이후의 성능 개선은 일반화가 아니라 암기일 가능성이 크다. 퀀트에서는 변수가 많고 표본이 제한적이기 때문에 "
            "이 현상이 자주 발생한다. 특히 팩터를 수백 개 만들고 그중 성과가 좋은 조합만 고르면 데이터 스누핑에 빠지기 쉽다.",
            styles,
        )
    )
    story.append(
        make_table(
            [
                ["신호", "의미", "대응"],
                ["Train 성능은 매우 좋고 Validation 성능은 나쁨", "학습 데이터에 맞춰진 모델", "모델 단순화, 정규화, 특성 축소"],
                ["특정 기간/종목군에서만 성과가 좋음", "시장 국면 또는 표본에 의존", "기간별, 섹터별, 스타일별 성능 분해"],
                ["파라미터를 조금 바꾸면 성과가 급변", "전략이 불안정", "민감도 분석, 더 단순한 규칙 선호"],
                ["수많은 실험 중 최고 결과만 선택", "데이터 스누핑 위험", "실험 로그 관리, 별도 테스트 유지"],
            ],
            [43 * mm, 57 * mm, 79 * mm],
            styles,
        )
    )

    story.append(PageBreak())
    story.append(p("7. Bias-Variance Tradeoff", styles, "h1"))
    story.append(
        p(
            "Bias와 Variance는 모델의 예측 오차를 두 가지 원인으로 나누어 보는 관점이다. "
            "Bias는 모델의 평균적인 예측이 진짜 관계에서 얼마나 벗어나 있는지를 뜻하고, Variance는 "
            "학습 데이터 표본이 조금 바뀌었을 때 모델의 예측이 얼마나 흔들리는지를 뜻한다. "
            "퀀트 ML에서는 이 구분이 특히 중요하다. 금융 데이터는 신호보다 노이즈가 큰 경우가 많고, "
            "백테스트를 많이 반복하면 우연히 좋아 보이는 모델을 고르기 쉽기 때문이다.",
            styles,
        )
    )
    story.append(diagram_title("그림 7-0. 다트판 비유로 보는 Bias와 Variance", styles))
    story.append(bias_variance_target_diagram())
    story.append(
        p(
            "빨간 중심은 우리가 맞히고 싶은 진짜 관계이고, 파란 점들은 학습 표본이 달라질 때마다 모델이 내놓는 "
            "예측이라고 보면 된다. Low Bias는 예측들의 평균 위치가 중심에 가깝다는 뜻이고, Low Variance는 "
            "예측들이 서로 가깝게 모여 있다는 뜻이다. High Bias는 예측들이 한쪽으로 치우친 상태이고, "
            "High Variance는 예측들이 넓게 흩어진 상태다.",
            styles,
        )
    )
    story.append(p("오차 분해 수식", styles, "h2"))
    story.append(
        p(
            "Prediction error at x = E[(y - f_hat(x))^2]<br/>"
            "= (E[f_hat(x)] - f(x))^2 + E[(f_hat(x) - E[f_hat(x)])^2] + sigma^2<br/>"
            "= Bias^2 + Variance + irreducible noise",
            styles,
            "formula",
        )
    )
    story.append(
        p(
            "여기서 f(x)는 데이터 안에 존재한다고 가정하는 진짜 관계이고, f_hat(x)는 우리가 학습한 모델의 예측이다. "
            "E[f_hat(x)]는 같은 방식으로 데이터를 여러 번 뽑아 모델을 학습했을 때 평균적으로 나오는 예측이다. "
            "Bias^2는 이 평균 예측이 진짜 관계 f(x)에서 얼마나 멀리 떨어져 있는지를 나타낸다. Variance는 "
            "각 학습 표본에서 나온 f_hat(x)가 평균 예측 주변에서 얼마나 흩어지는지를 나타낸다. sigma^2는 "
            "시장 자체의 우연성, 측정 오차, 뉴스 충격처럼 모델이 아무리 좋아도 제거할 수 없는 노이즈다.",
            styles,
        )
    )
    story.append(diagram_title("그림 7-1. 예측 오차는 Bias^2, Variance, 제거 불가능한 노이즈로 나뉜다", styles))
    story.append(bias_variance_decomposition_diagram())
    story.append(
        p(
            "실무적으로 모델 개발자가 직접 줄일 수 있는 것은 Bias와 Variance다. 노이즈 자체를 없앨 수는 없으므로, "
            "노이즈가 큰 문제에서는 예측값 하나의 정확도보다 예측 순위의 안정성, 여러 기간에서의 일관성, 비용 차감 후 "
            "성과가 더 중요한 판단 기준이 된다.",
            styles,
        )
    )

    story.append(diagram_title("그림 7-2. 너무 단순한 모델, 적절한 모델, 너무 복잡한 모델", styles))
    story.append(bias_variance_diagram())
    story.append(
        p(
            "왼쪽의 High Bias 모델은 데이터의 큰 방향을 충분히 따라가지 못한다. 예를 들어 실제로는 모멘텀, 밸류, "
            "변동성, 섹터, 금리 국면이 함께 작동하는데 PER 하나만으로 수익률을 설명하려는 경우다. 오른쪽의 "
            "High Variance 모델은 각 점을 지나치게 따라가며 노이즈까지 설명하려 한다. 이런 모델은 학습 구간에서는 "
            "매우 좋아 보이지만, 기간이 바뀌거나 종목군이 바뀌면 예측이 크게 흔들린다.",
            styles,
        )
    )

    story.append(diagram_title("그림 7-3. 모델 복잡도가 올라갈 때 Bias와 Variance의 방향", styles))
    story.append(model_complexity_error_diagram())
    story.append(
        p(
            "모델 복잡도를 올리면 보통 Bias는 감소한다. 더 많은 패턴을 표현할 수 있기 때문이다. 하지만 일정 수준을 "
            "넘으면 Variance가 증가한다. 모델이 진짜 신호뿐 아니라 학습 표본의 우연한 잡음까지 따라가기 때문이다. "
            "따라서 좋은 모델 선택은 가장 복잡한 모델을 고르는 일이 아니라, 검증 오차가 가장 낮고 미래 데이터에서도 "
            "안정적인 지점을 찾는 일이다.",
            styles,
        )
    )

    story.append(diagram_title("그림 7-4. Variance는 표본이 바뀔 때 모델이 흔들리는 정도다", styles))
    story.append(sample_variance_diagram())
    story.append(
        p(
            "같은 종목 universe와 같은 특성을 사용해도 학습 기간을 2015-2019, 2016-2020, 2017-2021처럼 조금씩 "
            "바꾸면 모델이 매우 다른 규칙을 학습할 수 있다. 이것이 Variance다. 퀀트에서는 이 문제를 줄이기 위해 "
            "Walk-Forward 검증, 기간별 성능 분해, 파라미터 민감도 분석, 거래비용 포함 검증을 함께 수행한다.",
            styles,
        )
    )
    story.append(p("Train / Validation 성능으로 진단하기", styles, "h2"))
    story.append(
        p(
            "진단은 Train 성능과 Validation 성능을 함께 보면서 시작한다. Train과 Validation이 둘 다 나쁘면 "
            "모델이 너무 단순하거나 특성이 약한 High Bias 상태일 가능성이 크다. Train은 좋은데 Validation이 나쁘면 "
            "학습 데이터에 과적합된 High Variance 상태일 가능성이 크다. 둘 다 어느 정도 좋고 차이가 작다면, "
            "그 모델은 일반화 가능성이 상대적으로 높다.",
            styles,
        )
    )
    story.append(
        make_table(
            [
                ["상태", "특징", "예시", "개선 방향"],
                ["High Bias", "Train과 Validation 모두 성능이 낮음", "선형 모델 하나로 복잡한 비선형 패턴을 설명", "특성 개선, 더 표현력 있는 모델"],
                ["High Variance", "Train은 좋지만 Validation은 낮음", "깊은 트리로 작은 표본의 노이즈까지 분리", "정규화, 앙상블, 데이터 확대, 모델 단순화"],
                ["균형", "Train과 Validation 차이가 작고 둘 다 충분함", "간단한 모델이 여러 기간에서 안정적", "테스트와 운영 모니터링으로 확인"],
            ],
            [30 * mm, 55 * mm, 54 * mm, 40 * mm],
            styles,
        )
    )
    story.append(p("학습 곡선으로 빠르게 진단하기", styles, "h2"))
    story.append(
        make_table(
            [
                ["관찰", "가능성이 큰 원인", "다음 실험"],
                ["Train 손실 높음, Validation 손실도 높음", "모델이 너무 단순하거나 특성이 약함", "새 특성 추가, 비선형 모델 시도, 라벨 재검토"],
                ["Train 손실 낮음, Validation 손실 높음", "과적합 또는 데이터 누수 후 붕괴", "정규화, 모델 단순화, 시간 분할 재점검"],
                ["기간별 성능 편차가 큼", "시장 국면 의존 또는 표본 부족", "Walk-Forward 검증, 국면별 성능 분해"],
                ["작은 파라미터 변화에 성능 급변", "Variance가 큰 불안정 모델", "민감도 분석, 앙상블, 더 단순한 의사결정 규칙"],
            ],
            [54 * mm, 56 * mm, 69 * mm],
            styles,
        )
    )
    story.append(
        p(
            "퀀트 모델에서는 복잡한 모델이 항상 이기지 않는다. 작은 예측 개선이 거래비용, 슬리피지, 리밸런싱 제약을 "
            "넘지 못하면 운영 관점에서는 더 단순하고 안정적인 모델이 낫다. 특히 수익률 예측에서는 노이즈가 커서 "
            "Train 성능이 지나치게 좋은 모델을 오히려 의심해야 한다.",
            styles,
            "callout",
        )
    )
    story.append(PageBreak())
    story.append(p("퀀트 예시: 팩터 모델에서의 Bias와 Variance", styles, "h2"))
    story.append(
        make_table(
            [
                ["상황", "문제 해석", "개선 방향"],
                ["PER 하나만으로 다음 달 수익률을 예측", "시장 국면, 섹터, 모멘텀을 놓치는 High Bias 가능성", "핵심 팩터 추가, 국면 변수 추가"],
                ["팩터 300개와 깊은 트리로 학습 구간 Sharpe가 매우 높음", "우연한 조합을 외운 High Variance 가능성", "특성 축소, 트리 깊이 제한, 별도 테스트 유지"],
                ["Validation은 좋지만 특정 1년 성과에 대부분 의존", "기간 의존 Variance 가능성", "연도별/국면별 성능 분해"],
                ["파라미터를 조금 바꾸면 종목 선택이 크게 달라짐", "불안정한 의사결정 경계", "민감도 분석, 앙상블, 더 넓은 재학습 창"],
            ],
            [54 * mm, 70 * mm, 55 * mm],
            styles,
        )
    )
    story.append(
        p(
            "좋은 퀀트 ML 모델의 목표는 과거 데이터를 완벽하게 맞추는 것이 아니다. 목표는 미래 구간에서도 유지되는 "
            "작고 반복 가능한 신호를 찾는 것이다. 그래서 Bias를 줄이되 Variance가 폭발하지 않게 제어하는 것이 "
            "모델링의 핵심이다.",
            styles,
            "callout",
        )
    )

    story.append(p("8. Cross Validation", styles, "h1"))
    story.append(
        p(
            "교차검증은 데이터를 여러 조각으로 나누어, 일부는 학습하고 일부는 검증하는 과정을 반복하는 방법이다. "
            "목적은 특정 한 번의 분할에 운 좋게 맞은 모델을 고르는 일을 줄이는 것이다.",
            styles,
        )
    )
    story.append(diagram_title("그림 8. 시계열 교차검증은 항상 과거로 학습하고 다음 구간을 검증", styles))
    story.append(cross_validation_diagram())
    story.append(
        p(
            "일반 K-Fold는 데이터를 무작위로 섞는 경우가 많지만, 금융 시계열에서는 이 방식이 위험하다. "
            "시계열 교차검증은 각 Fold에서 과거 구간만 학습하고 바로 다음 미래 구간을 검증한다. "
            "이 구조가 실제 운영의 재학습 과정을 더 잘 흉내 낸다.",
            styles,
        )
    )
    story.append(
        make_table(
            [
                ["방식", "어떻게 나누는가", "적합한 상황", "퀀트 주의점"],
                ["K-Fold", "데이터를 K개로 나누고 하나씩 검증", "독립 표본에 가까운 일반 데이터", "시계열에는 그대로 쓰면 누수 위험"],
                ["Stratified K-Fold", "클래스 비율을 유지하며 분할", "불균형 분류", "시간 순서 보존이 더 중요할 수 있음"],
                ["Time Series Split", "과거로 학습하고 미래 구간으로 검증", "가격, 재무, 거시 지표", "운영 시점의 정보 집합을 엄격히 재현"],
                ["Walk-Forward", "학습 창을 앞으로 굴리며 반복 평가", "전략 백테스트와 모델 재학습", "거래비용, 리밸런싱 주기 포함 필요"],
            ],
            [32 * mm, 52 * mm, 42 * mm, 53 * mm],
            styles,
        )
    )
    story.append(
        p(
            "<b>실무 원칙</b>: 금융 시계열에서는 일반 K-Fold보다 Time Series Split 또는 Walk-Forward 검증을 먼저 고려한다. "
            "검증 방식은 모델 성능 숫자만이 아니라 실제 운영 프로세스를 흉내 내야 한다.",
            styles,
            "callout",
        )
    )

    story.append(p("9. 전체 파이프라인으로 연결하기", styles, "h1"))
    story.append(
        make_table(
            [
                ["단계", "질문", "산출물"],
                ["문제 정의", "예측하려는 y는 무엇인가? 회귀인가 분류인가?", "라벨 정의, 투자/리스크 목적"],
                ["데이터 준비", "각 시점에서 실제로 알 수 있던 X인가?", "특성 테이블, 누수 점검"],
                ["분할", "Train/Validation/Test가 시간 순서를 지키는가?", "분할 기준일, 검증 프로토콜"],
                ["모델 학습", "어떤 손실 함수를 최소화하는가?", "학습된 모델, 하이퍼파라미터"],
                ["검증", "과적합과 Bias-Variance 상태는 어떤가?", "성능표, 기간별 분석, 민감도 분석"],
                ["테스트", "최종 미사용 구간에서도 성과가 유지되는가?", "최종 테스트 리포트"],
                ["운영", "재학습 주기, 모니터링, 비용 반영이 있는가?", "운영 규칙, 알림, 성능 드리프트 점검"],
            ],
            [27 * mm, 82 * mm, 70 * mm],
            styles,
        )
    )

    roadmap = [
        ["주차", "핵심 목표", "공부 내용", "실습 산출물"],
        ["1주차", "ML 문제 구조 이해", "지도/비지도, 회귀/분류, 라벨과 특성", "작은 CSV로 회귀/분류 문제 각각 1개 정의"],
        ["2주차", "손실과 평가 지표 이해", "MSE, MAE, Log Loss, Accuracy, AUC, Rank IC", "같은 모델을 여러 지표로 비교한 노트"],
        ["3주차", "검증 체계 만들기", "Train/Validation/Test, Cross Validation, Time Series Split", "누수 없는 시계열 분할 코드"],
        ["4주차", "과적합 통제", "Overfitting, Bias-Variance, 정규화, 모델 복잡도", "간단한 모델과 복잡한 모델의 성능 비교 리포트"],
    ]
    story.append(
        KeepTogether(
            [
                p("10. 4주 학습 로드맵", styles, "h1"),
                make_table(roadmap, [20 * mm, 42 * mm, 58 * mm, 59 * mm], styles),
            ]
        )
    )

    story.append(p("11. 추천 미니 프로젝트", styles, "h1"))
    story.append(
        bullets(
            [
                "<b>프로젝트 A - 변동성 회귀</b>: 과거 20일 변동성, 거래량 변화, 수익률 모멘텀으로 다음 20일 변동성을 예측한다.",
                "<b>프로젝트 B - 상승 확률 분류</b>: 다음 1개월 수익률이 시장 수익률보다 높은지 여부를 분류한다.",
                "<b>프로젝트 C - 종목 군집화</b>: 수익률 상관, 변동성, 베타, 섹터 노출을 이용해 유사 종목군을 만든다.",
                "<b>프로젝트 D - Walk-Forward 검증</b>: 매월 모델을 재학습하고 다음 달만 예측하는 검증 루프를 만든다.",
            ],
            styles,
        )
    )
    story.append(p("미니 프로젝트 수행 템플릿", styles, "h2"))
    story.append(
        make_table(
            [
                ["항목", "작성 내용", "예시"],
                ["문제 정의", "회귀/분류/비지도 중 무엇인지 명시", "다음 20거래일 변동성을 예측하는 회귀"],
                ["라벨 y", "예측 시점 이후 값으로만 계산", "t+1일부터 t+20일까지의 실현 변동성"],
                ["특성 X", "예측 시점 이전에 알 수 있는 값만 사용", "과거 20일 변동성, 거래량 변화, 최근 수익률"],
                ["검증 방식", "시간 순서와 재학습 주기를 반영", "2018-2022 train, 2023 validation, 2024 test"],
                ["성공 기준", "모델 지표와 퀀트 관점 지표를 함께 기록", "MAE, Rank IC, 비용 차감 후 성과"],
            ],
            [27 * mm, 74 * mm, 78 * mm],
            styles,
        )
    )

    story.append(PageBreak())
    story.append(p("12. 자기 점검 질문", styles, "h1"))
    story.append(
        make_table(
            [
                ["질문", "스스로 답해야 할 기준"],
                ["지도학습과 비지도학습의 차이를 한 문장으로 말할 수 있는가?", "정답 y의 존재 여부와 목적 차이를 설명할 수 있다."],
                ["회귀 문제를 분류 문제로 바꾸면 무엇이 달라지는가?", "라벨, 손실 함수, 평가 지표, 의사결정 임계값이 바뀐다."],
                ["손실 함수와 평가 지표가 다르면 어떤 문제가 생기는가?", "학습 목표와 실제 목표가 어긋날 수 있다."],
                ["테스트 성능을 여러 번 보고 모델을 고르면 왜 위험한가?", "테스트 구간에 간접 과적합되기 때문이다."],
                ["금융 시계열에서 무작위 K-Fold가 위험한 이유는?", "미래 정보가 학습에 섞이는 누수 가능성 때문이다."],
                ["Bias와 Variance 중 어느 문제가 더 큰지 어떻게 판단하는가?", "Train과 Validation 성능의 절대 수준과 차이를 비교한다."],
            ],
            [77 * mm, 102 * mm],
            styles,
        )
    )

    story.append(p("13. 핵심 요약", styles, "h1"))
    story.append(
        bullets(
            [
                "머신러닝은 데이터에서 패턴을 학습해 보지 못한 데이터에 일반화하려는 방법이다.",
                "Supervised와 Unsupervised의 첫 차이는 정답 라벨 y의 존재다.",
                "Regression은 숫자 예측, Classification은 클래스나 확률 예측이다.",
                "Loss Function은 모델이 무엇을 틀렸다고 느끼는지 정의한다.",
                "Train/Validation/Test 분리는 일반화 성능을 추정하기 위한 최소 장치다.",
                "Overfitting은 퀀트 백테스트에서 특히 치명적이며, 데이터 누수와 함께 가장 먼저 의심해야 한다.",
                "Bias-Variance Tradeoff는 모델 복잡도를 조절하는 기준이다.",
                "Cross Validation은 한 번의 운 좋은 분할을 피하기 위한 도구이며, 시계열에서는 시간 순서가 우선이다.",
            ],
            styles,
        )
    )

    story.append(
        p(
            "<b>다음 학습 순서</b>: Python 기초 -> pandas/numpy -> scikit-learn -> 선형회귀/로지스틱회귀 -> 트리/랜덤포레스트/부스팅 -> "
            "시계열 검증 -> 백테스트와 거래비용 -> 포트폴리오 구성.",
            styles,
            "callout",
        )
    )

    return story


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    font, bold = register_fonts()
    styles = make_styles(font, bold)
    doc = SimpleDocTemplate(
        str(OUT_FILE),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=20 * mm,
        title="퀀트 기반 ML Engineer 입문 보고서",
        author="Codex",
    )
    story = build_story(styles)
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(OUT_FILE)


if __name__ == "__main__":
    main()
