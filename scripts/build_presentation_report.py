"""Render the canonical PPT generation report as a PDF."""
from __future__ import annotations

from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import LongTable, Paragraph, SimpleDocTemplate, Spacer, TableStyle


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports" / "PPT_Generation_Report.md"
OUTPUT = SOURCE.with_suffix(".pdf")


def build() -> None:
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        "ReportBody", parent=styles["BodyText"], fontSize=9.5, leading=14,
        spaceAfter=7, textColor=colors.HexColor("#243b53"),
    ))
    styles.add(ParagraphStyle(
        "ReportCell", parent=styles["ReportBody"], fontSize=8, leading=11, spaceAfter=0,
    ))
    styles.add(ParagraphStyle(
        "ReportHeaderCell", parent=styles["ReportCell"], fontName="Helvetica-Bold",
        textColor=colors.white,
    ))
    styles.add(ParagraphStyle(
        "ReportBullet", parent=styles["ReportBody"], leftIndent=12, firstLineIndent=-8,
        spaceAfter=4,
    ))
    for name in ("Title", "Heading1", "Heading2"):
        styles[name].textColor = colors.HexColor("#102a43")

    def paragraph(text: str, style: str = "ReportBody") -> Paragraph:
        return Paragraph(escape(text), styles[style])

    def footer(canvas, doc) -> None:
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#cbd5e1"))
        canvas.line(doc.leftMargin, 1.35 * cm, A4[0] - doc.rightMargin, 1.35 * cm)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#52606d"))
        canvas.drawString(doc.leftMargin, cm, "Multimodal Deepfake Detection Studio | PPT generation report")
        canvas.drawRightString(A4[0] - doc.rightMargin, cm, str(doc.page))
        canvas.restoreState()

    document = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4, leftMargin=1.65 * cm, rightMargin=1.65 * cm,
        topMargin=1.65 * cm, bottomMargin=1.8 * cm,
        title="Multimodal Deepfake Detection Studio - PPT Generation Report",
        author="Multimodal Deepfake Project", pageCompression=1,
    )
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    story = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                cells = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
                if not all(cell and set(cell) <= {"-", ":", " "} for cell in cells):
                    rows.append([paragraph(cell, "ReportHeaderCell" if not rows else "ReportCell") for cell in cells])
                index += 1
            columns = len(rows[0])
            widths = [document.width / columns] * columns
            if columns == 2:
                widths = [document.width * 0.38, document.width * 0.62]
            table = LongTable(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#102a43")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.extend([table, Spacer(1, 9)])
            continue
        if line.startswith("### "):
            story.append(paragraph(line[4:], "Heading2"))
        elif line.startswith("## "):
            story.append(paragraph(line[3:], "Heading1"))
        elif line.startswith("# "):
            story.append(paragraph(line[2:], "Title"))
        elif line.startswith("- "):
            story.append(paragraph("- " + line[2:], "ReportBullet"))
        else:
            parts = [line]
            while index + 1 < len(lines) and lines[index + 1].strip() and not lines[index + 1].lstrip().startswith(("#", "|", "- ")):
                index += 1
                parts.append(lines[index].strip())
            story.append(paragraph(" ".join(parts)))
        index += 1

    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Created {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    build()
