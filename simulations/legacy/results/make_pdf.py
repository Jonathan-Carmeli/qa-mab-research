#!/usr/bin/env python3
"""Generate a professional academic-style PDF from the QA-MAB narrative markdown."""

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


# ------- Config -------
SRC_MD = Path("/Users/jon_claw/qa-mab-research/simulations/results/qa-mab-narrative.md")
OUT_PDF = Path("/Users/jon_claw/qa-mab-research/simulations/results/qa-mab-narrative.pdf")

PAGE = A4
MARGIN = 2.5 * cm

NAVY = colors.HexColor("#1a1a2e")
ACCENT = colors.HexColor("#2d4059")
SUBTLE = colors.HexColor("#4a5568")
LINE = colors.HexColor("#cbd5e0")
CODE_BG = colors.HexColor("#f5f5f5")
ROW_ALT = colors.HexColor("#f0f4f8")
ROW_BASE = colors.white


# ------- Styles -------
def build_styles():
    base = getSampleStyleSheet()
    styles = {}

    styles["TitleMain"] = ParagraphStyle(
        "TitleMain",
        parent=base["Title"],
        fontName="Helvetica-Bold",
        fontSize=26,
        leading=32,
        alignment=TA_CENTER,
        textColor=NAVY,
        spaceAfter=18,
    )
    styles["Subtitle"] = ParagraphStyle(
        "Subtitle",
        fontName="Helvetica",
        fontSize=13,
        leading=18,
        alignment=TA_CENTER,
        textColor=ACCENT,
        spaceAfter=10,
    )
    styles["TitleMeta"] = ParagraphStyle(
        "TitleMeta",
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        alignment=TA_CENTER,
        textColor=SUBTLE,
        spaceAfter=4,
    )
    styles["H1"] = ParagraphStyle(
        "H1",
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=NAVY,
        spaceBefore=18,
        spaceAfter=10,
        keepWithNext=1,
    )
    styles["H2"] = ParagraphStyle(
        "H2",
        fontName="Helvetica-Bold",
        fontSize=13.5,
        leading=17,
        textColor=ACCENT,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=1,
    )
    styles["H3"] = ParagraphStyle(
        "H3",
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=14,
        textColor=ACCENT,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=1,
    )
    styles["Body"] = ParagraphStyle(
        "Body",
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.black,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
    )
    styles["Bullet"] = ParagraphStyle(
        "Bullet",
        parent=styles["Body"],
        leftIndent=18,
        bulletIndent=6,
        spaceAfter=3,
    )
    styles["Numbered"] = ParagraphStyle(
        "Numbered",
        parent=styles["Body"],
        leftIndent=22,
        bulletIndent=6,
        spaceAfter=3,
    )
    styles["Code"] = ParagraphStyle(
        "Code",
        fontName="Courier",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#222222"),
        leftIndent=8,
        rightIndent=8,
        spaceBefore=4,
        spaceAfter=4,
        backColor=CODE_BG,
        borderPadding=(6, 6, 6, 6),
    )
    styles["TOCEntry"] = ParagraphStyle(
        "TOCEntry",
        fontName="Helvetica",
        fontSize=11,
        leading=18,
        textColor=colors.black,
        leftIndent=0,
    )
    styles["TOCEntryH2"] = ParagraphStyle(
        "TOCEntryH2",
        fontName="Helvetica",
        fontSize=10,
        leading=15,
        textColor=SUBTLE,
        leftIndent=18,
    )
    styles["SectionLabel"] = ParagraphStyle(
        "SectionLabel",
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=NAVY,
        alignment=TA_CENTER,
        spaceAfter=20,
    )
    return styles


# ------- Inline formatting (bold, escaping) -------
def inline_format(text: str) -> str:
    # Escape XML special characters for reportlab Paragraph (which uses mini-HTML)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Convert **bold** -> <b>bold</b>
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    # Convert inline `code` -> <font name="Courier">code</font>
    text = re.sub(
        r"`([^`]+?)`",
        r'<font name="Courier" backColor="#f0f0f0">\1</font>',
        text,
    )
    # Remove leftover backslashes used to escape * in source
    text = text.replace("\\*", "*")
    return text


def code_format(text: str) -> str:
    """Format code-block lines, preserving leading whitespace as nbsp."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Preserve leading spaces with non-breaking spaces
    stripped = text.lstrip(" ")
    leading = len(text) - len(stripped)
    return ("&nbsp;" * leading) + stripped


# ------- Markdown parsing -------
def split_into_blocks(md_text: str):
    """Yield typed blocks: heading/para/list/code/hr/table."""
    lines = md_text.splitlines()
    i = 0
    blocks = []
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Blank line
        if not stripped:
            i += 1
            continue

        # Horizontal rule
        if re.match(r"^-{3,}$", stripped):
            blocks.append(("hr", None))
            i += 1
            continue

        # Heading
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            level = len(m.group(1))
            blocks.append(("h", (level, m.group(2).strip())))
            i += 1
            continue

        # Code fence
        if stripped.startswith("```"):
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            if i < len(lines):
                i += 1  # skip closing fence
            blocks.append(("code", code_lines))
            continue

        # Table (line starts with '|' and next line is a separator)
        if stripped.startswith("|") and i + 1 < len(lines) and re.match(
            r"^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?\s*$", lines[i + 1]
        ):
            table_lines = [line]
            i += 1
            # skip separator
            i += 1
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            blocks.append(("table", table_lines))
            continue

        # List item (bulleted)
        if re.match(r"^\s*[-*]\s+\S", line):
            list_items = []
            while i < len(lines) and re.match(r"^\s*[-*]\s+\S", lines[i]):
                item_text = re.sub(r"^\s*[-*]\s+", "", lines[i]).rstrip()
                # capture continuation lines
                i += 1
                while (
                    i < len(lines)
                    and lines[i].strip()
                    and not re.match(r"^\s*[-*]\s+\S", lines[i])
                    and not re.match(r"^\s*\d+\.\s+\S", lines[i])
                    and not lines[i].strip().startswith("#")
                    and not lines[i].strip().startswith("```")
                    and not lines[i].strip().startswith("|")
                    and not re.match(r"^-{3,}$", lines[i].strip())
                    and lines[i].startswith(" ")
                ):
                    item_text += " " + lines[i].strip()
                    i += 1
                list_items.append(item_text)
            blocks.append(("ulist", list_items))
            continue

        # Numbered list item
        if re.match(r"^\s*\d+\.\s+\S", line):
            list_items = []
            while i < len(lines) and re.match(r"^\s*\d+\.\s+\S", lines[i]):
                m2 = re.match(r"^\s*(\d+)\.\s+(.*)$", lines[i])
                item_text = m2.group(2).rstrip()
                num = m2.group(1)
                i += 1
                while (
                    i < len(lines)
                    and lines[i].strip()
                    and not re.match(r"^\s*[-*]\s+\S", lines[i])
                    and not re.match(r"^\s*\d+\.\s+\S", lines[i])
                    and not lines[i].strip().startswith("#")
                    and not lines[i].strip().startswith("```")
                    and not lines[i].strip().startswith("|")
                    and not re.match(r"^-{3,}$", lines[i].strip())
                    and lines[i].startswith(" ")
                ):
                    item_text += " " + lines[i].strip()
                    i += 1
                list_items.append((num, item_text))
            blocks.append(("olist", list_items))
            continue

        # Paragraph
        para_lines = [line]
        i += 1
        while (
            i < len(lines)
            and lines[i].strip()
            and not lines[i].strip().startswith("#")
            and not lines[i].strip().startswith("```")
            and not lines[i].strip().startswith("|")
            and not re.match(r"^-{3,}$", lines[i].strip())
            and not re.match(r"^\s*[-*]\s+\S", lines[i])
            and not re.match(r"^\s*\d+\.\s+\S", lines[i])
        ):
            para_lines.append(lines[i])
            i += 1
        blocks.append(("p", " ".join(p.strip() for p in para_lines)))

    return blocks


def parse_table(table_lines):
    rows = []
    for tl in table_lines:
        s = tl.strip()
        if s.startswith("|"):
            s = s[1:]
        if s.endswith("|"):
            s = s[:-1]
        cells = [c.strip() for c in s.split("|")]
        rows.append(cells)
    return rows


# ------- PDF building -------
def make_doc():
    styles = build_styles()
    # Use BaseDocTemplate with custom frame so we can apply different first-page templates
    doc = BaseDocTemplate(
        str(OUT_PDF),
        pagesize=PAGE,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
        title="QA-MAB Research Narrative",
        author="Yehonatan Carmeli",
        subject="QA-MAB UAV Routing Research",
    )

    frame = Frame(
        doc.leftMargin,
        doc.bottomMargin,
        doc.width,
        doc.height,
        id="normal",
        showBoundary=0,
    )

    def on_page(canvas, _doc):
        canvas.saveState()
        # Footer page number (skip title page = page 1)
        page_num = canvas.getPageNumber()
        if page_num > 1:
            canvas.setFont("Helvetica", 9)
            canvas.setFillColor(SUBTLE)
            canvas.drawCentredString(PAGE[0] / 2.0, 1.2 * cm, f"{page_num}")
            # subtle header rule
            canvas.setStrokeColor(LINE)
            canvas.setLineWidth(0.4)
            canvas.line(MARGIN, PAGE[1] - 1.5 * cm, PAGE[0] - MARGIN, PAGE[1] - 1.5 * cm)
            canvas.setFont("Helvetica-Oblique", 8)
            canvas.setFillColor(SUBTLE)
            canvas.drawString(MARGIN, PAGE[1] - 1.25 * cm, "QA-MAB Research Narrative")
            canvas.drawRightString(
                PAGE[0] - MARGIN, PAGE[1] - 1.25 * cm, "Yehonatan Carmeli  ·  v1.0"
            )
        canvas.restoreState()

    def on_title_page(canvas, _doc):
        canvas.saveState()
        # Decorative top bar
        canvas.setFillColor(NAVY)
        canvas.rect(0, PAGE[1] - 0.8 * cm, PAGE[0], 0.8 * cm, stroke=0, fill=1)
        canvas.setFillColor(ACCENT)
        canvas.rect(0, 0, PAGE[0], 0.8 * cm, stroke=0, fill=1)
        canvas.restoreState()

    doc.addPageTemplates([
        PageTemplate(id="Title", frames=[frame], onPage=on_title_page),
        PageTemplate(id="Body", frames=[frame], onPage=on_page),
    ])
    return doc, styles


def make_table_flowable(rows):
    if not rows:
        return Spacer(1, 0)
    body_style = ParagraphStyle(
        "TableBody",
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.black,
    )
    header_style = ParagraphStyle(
        "TableHeader",
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=12,
        textColor=colors.white,
    )

    formatted = []
    for r_idx, row in enumerate(rows):
        formatted_row = []
        for cell in row:
            style = header_style if r_idx == 0 else body_style
            formatted_row.append(Paragraph(inline_format(cell), style))
        formatted.append(formatted_row)

    # Compute approximate column widths from total available width
    n_cols = max(len(r) for r in rows)
    avail = PAGE[0] - 2 * MARGIN
    col_widths = [avail / n_cols] * n_cols

    tbl = Table(formatted, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, NAVY),
        ("BOX", (0, 0), (-1, -1), 0.4, LINE),
    ]
    for r_idx in range(1, len(rows)):
        bg = ROW_ALT if r_idx % 2 == 0 else ROW_BASE
        style_cmds.append(("BACKGROUND", (0, r_idx), (-1, r_idx), bg))
    tbl.setStyle(TableStyle(style_cmds))
    return tbl


def build_title_page(styles, title, subtitle):
    story = []
    story.append(Spacer(1, 5.5 * cm))
    story.append(Paragraph(title, styles["TitleMain"]))
    story.append(Spacer(1, 0.2 * cm))
    story.append(HRFlowable(width="55%", thickness=0.8, color=ACCENT, hAlign="CENTER"))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(subtitle, styles["Subtitle"]))
    story.append(Spacer(1, 3.5 * cm))
    story.append(Paragraph("<b>Author</b>", styles["TitleMeta"]))
    story.append(Paragraph("Yehonatan Carmeli", styles["TitleMeta"]))
    story.append(Spacer(1, 0.6 * cm))
    story.append(Paragraph("<b>Date</b>", styles["TitleMeta"]))
    story.append(Paragraph("May 2026", styles["TitleMeta"]))
    story.append(Spacer(1, 0.6 * cm))
    story.append(Paragraph("<b>Version</b>", styles["TitleMeta"]))
    story.append(Paragraph("1.0", styles["TitleMeta"]))
    return story


def build_toc(styles, headings):
    story = []
    story.append(Paragraph("Table of Contents", styles["SectionLabel"]))
    story.append(HRFlowable(width="100%", thickness=0.6, color=NAVY))
    story.append(Spacer(1, 0.6 * cm))
    for level, text in headings:
        if level == 1:
            story.append(Paragraph(f"<b>{inline_format(text)}</b>", styles["TOCEntry"]))
        elif level == 2:
            story.append(Paragraph(inline_format(text), styles["TOCEntryH2"]))
    return story


def main():
    md_text = SRC_MD.read_text(encoding="utf-8")
    blocks = split_into_blocks(md_text)

    doc, styles = make_doc()
    story = []

    # ----- Title page -----
    title_line = "QA-MAB Research Narrative"
    subtitle_line = "How We Built the Case for Quantum Annealing"
    # Skip the leading H1 and author/date metadata paragraph from md
    consumed = 0
    while consumed < len(blocks):
        b = blocks[consumed]
        if b[0] == "h" and b[1][0] == 1:
            # use the first H1 as title (but split on em-dash)
            full = b[1][1]
            if "—" in full:
                title_line, subtitle_line = [s.strip() for s in full.split("—", 1)]
            else:
                title_line = full
            consumed += 1
            continue
        if b[0] == "p" and ("Author" in b[1] or "Version" in b[1]):
            consumed += 1
            continue
        if b[0] == "hr":
            consumed += 1
            break
        break

    from reportlab.platypus import NextPageTemplate

    # Pre-collect H1/H2 from remaining blocks for the TOC
    headings = []
    for b in blocks[consumed:]:
        if b[0] == "h" and b[1][0] in (1, 2):
            headings.append((b[1][0], b[1][1]))

    story = []
    story.extend(build_title_page(styles, title_line, subtitle_line))
    story.append(NextPageTemplate("Body"))
    story.append(PageBreak())
    story.extend(build_toc(styles, headings))
    story.append(PageBreak())

    # ----- Body content -----
    in_first_section = True
    for b in blocks[consumed:]:
        kind = b[0]

        if kind == "hr":
            story.append(Spacer(1, 0.2 * cm))
            story.append(HRFlowable(width="100%", thickness=0.5, color=LINE))
            story.append(Spacer(1, 0.2 * cm))
            continue

        if kind == "h":
            level, text = b[1]
            if level == 1:
                if not in_first_section:
                    story.append(PageBreak())
                in_first_section = False
                story.append(Paragraph(inline_format(text), styles["H1"]))
                story.append(HRFlowable(width="100%", thickness=0.6, color=NAVY))
                story.append(Spacer(1, 0.15 * cm))
            elif level == 2:
                story.append(Paragraph(inline_format(text), styles["H2"]))
            else:
                story.append(Paragraph(inline_format(text), styles["H3"]))
            continue

        if kind == "p":
            story.append(Paragraph(inline_format(b[1]), styles["Body"]))
            continue

        if kind == "ulist":
            for item in b[1]:
                story.append(
                    Paragraph(inline_format(item), styles["Bullet"], bulletText="•")
                )
            story.append(Spacer(1, 0.15 * cm))
            continue

        if kind == "olist":
            for num, item in b[1]:
                story.append(
                    Paragraph(
                        inline_format(item),
                        styles["Numbered"],
                        bulletText=f"{num}.",
                    )
                )
            story.append(Spacer(1, 0.15 * cm))
            continue

        if kind == "code":
            code_lines = b[1]
            # Combine into single Paragraph with <br/> for nice background box
            formatted = "<br/>".join(code_format(cl) for cl in code_lines)
            if not formatted:
                formatted = "&nbsp;"
            story.append(KeepTogether(Paragraph(formatted, styles["Code"])))
            story.append(Spacer(1, 0.15 * cm))
            continue

        if kind == "table":
            rows = parse_table(b[1])
            story.append(Spacer(1, 0.1 * cm))
            story.append(make_table_flowable(rows))
            story.append(Spacer(1, 0.25 * cm))
            continue

    doc.build(story)
    print(f"Wrote {OUT_PDF} ({OUT_PDF.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
