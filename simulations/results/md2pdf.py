#!/usr/bin/env python3
"""Convert qa-mab-narrative.md to PDF using reportlab."""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
import re

MD_PATH = "/Users/jon_claw/qa-mab-research/simulations/results/qa-mab-narrative.md"
PDF_PATH = "/Users/jon_claw/qa-mab-research/simulations/results/qa-mab-narrative.pdf"

def md_to_text(md_lines):
    """Convert markdown to plain text with basic formatting preserved."""
    out = []
    in_table = False
    table_lines = []

    for raw in md_lines:
        line = raw.rstrip()

        # Tables
        if '|' in line and line.strip().startswith('|'):
            table_lines.append(line)
            continue
        elif table_lines:
            out.append(('TABLE', table_lines))
            table_lines = []

        # Headings
        m = re.match(r'^(#{1,6})\s+(.*)', line)
        if m:
            out.append(('H', int(len(m.group(1))), m.group(2).strip()))
        # Code blocks
        elif line.strip().startswith('```'):
            continue
        elif line.startswith('    ') or line.startswith('\t'):
            out.append(('CODE', line.strip()))
        elif line.startswith('|'):
            continue
        elif line == '---':
            out.append(('RULE',))
        elif not line:
            out.append(('BLANK',))
        else:
            # Inline formatting
            text = re.sub(r'\*\*(.+?)\*\*', r'\1', line)
            text = re.sub(r'__(.+?)__', r'\1', text)
            text = re.sub(r'\*(.+?)\*', r'\1', text)
            text = re.sub(r'`(.+?)`', r'\1', text)
            out.append(('P', text))

    if table_lines:
        out.append(('TABLE', table_lines))
    return out

def parse_table(lines):
    rows = []
    for l in lines:
        cells = [c.strip() for c in l.split('|')[1:-1]]
        rows.append(cells)
    # Remove separator rows
    clean = [r for r in rows if not all(re.match(r'^[-:]+$', c) for c in r)]
    return clean

def build_pdf(md_text, pdf_path):
    doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                           leftMargin=2.5*cm, rightMargin=2.5*cm,
                           topMargin=2.5*cm, bottomMargin=2.5*cm)
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle('Title', parent=styles['Normal'],
                                  fontSize=22, fontName='Helvetica-Bold',
                                  spaceAfter=8, alignment=TA_CENTER)
    subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'],
                                    fontSize=11, fontName='Helvetica',
                                    spaceAfter=20, alignment=TA_CENTER, textColor=colors.gray)
    h1_style = ParagraphStyle('H1', parent=styles['Normal'],
                              fontSize=16, fontName='Helvetica-Bold',
                              spaceBefore=18, spaceAfter=6, textColor=colors.HexColor('#1a1a2e'))
    h2_style = ParagraphStyle('H2', parent=styles['Normal'],
                              fontSize=13, fontName='Helvetica-Bold',
                              spaceBefore=12, spaceAfter=4)
    h3_style = ParagraphStyle('H3', parent=styles['Normal'],
                              fontSize=11, fontName='Helvetica-Bold',
                              spaceBefore=8, spaceAfter=3)
    body_style = ParagraphStyle('Body', parent=styles['Normal'],
                                 fontSize=10, fontName='Helvetica',
                                 spaceAfter=6, leading=14)
    code_style = ParagraphStyle('Code', parent=styles['Normal'],
                                 fontSize=8, fontName='Courier',
                                 backColor=colors.HexColor('#f5f5f5'),
                                 spaceBefore=4, spaceAfter=4, leading=12)
    bold_body = ParagraphStyle('BoldBody', parent=body_style,
                                fontName='Helvetica-Bold')

    story = []

    for item in md_text:
        typ = item[0]

        if typ == 'H':
            level, text = item[1], item[2]
            if level == 1:
                story.append(Paragraph(text, title_style))
            elif level == 2:
                story.append(Paragraph(text, h1_style))
            elif level == 3:
                story.append(Paragraph(text, h2_style))
            else:
                story.append(Paragraph(text, h3_style))
        elif typ == 'P':
            # Bold numbers/keywords
            parts = re.split(r'(\b[\w\s]+:\s)', item[1])
            # Simple: bold the leading term if it looks like a label
            story.append(Paragraph(item[1], body_style))
        elif typ == 'CODE':
            story.append(Paragraph(item[1], code_style))
        elif typ == 'BLANK':
            story.append(Spacer(1, 4))
        elif typ == 'RULE':
            from reportlab.platypus import HRFlowable
            story.append(HRFlowable(width='100%', thickness=0.5, color=colors.lightgrey, spaceAfter=6, spaceBefore=6))
        elif typ == 'TABLE':
            rows = parse_table(item[1])
            if not rows:
                continue
            t = Table(rows, colWidths=[4*cm, 5*cm, 5*cm])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a1a2e')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,-1), 9),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f9f9f9')]),
                ('TOPPADDING', (0,0), (-1,-1), 4),
                ('BOTTOMPADDING', (0,0), (-1,-1), 4),
                ('LEFTPADDING', (0,0), (-1,-1), 6),
            ]))
            story.append(t)
            story.append(Spacer(1, 10))

    doc.build(story)
    print(f"PDF written to {pdf_path}")

with open(MD_PATH) as f:
    lines = f.readlines()

md_text = md_to_text(lines)
build_pdf(md_text, PDF_PATH)
