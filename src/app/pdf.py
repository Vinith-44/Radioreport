"""Step 10 PDF export with ReportLab, reusing the layout of the existing template
(Image-Based-Radiology-Report-Generation-using-Medgemma/src/medgemma_report_generator.py, ClinicalPDFGenerator):
same A4 margins, Helvetica styles and colours, centred title block, key-value info table, centred figure with
caption, plain section paragraphs and an italic footer disclaimer. Additions: per-sentence verdict colours,
supporting cases and (optionally) the patient summary.

Indian scripts need a TTF font that covers them (e.g. Noto Sans Devanagari / Tamil / Telugu) and ReportLab's
text shaping (pip install uharfbuzz); without `indic_font_ttf` the translated summary is left out of the PDF
(and the app says so) rather than printed as empty boxes.
"""
from __future__ import annotations

import datetime as _dt
import io

from src.app.render import COLORS, DISCLAIMER


def _styles(indic_font: str | None):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.lib.styles import ParagraphStyle as PS

    gry, dgr, blu = colors.HexColor("#555555"), colors.HexColor("#222222"), colors.HexColor("#1A237E")
    s = {"title": PS("t", fontSize=15, fontName="Helvetica-Bold", textColor=blu, leading=20, spaceAfter=2,
                     alignment=TA_CENTER),
         "subtitle": PS("st", fontSize=8.5, fontName="Helvetica", textColor=gry, leading=12, alignment=TA_CENTER),
         "ik": PS("ik", fontSize=9, fontName="Helvetica-Bold", textColor=dgr, leading=13),
         "iv": PS("iv", fontSize=9, fontName="Helvetica", textColor=colors.black, leading=13),
         "shdr": PS("sh", fontSize=9.5, fontName="Helvetica-Bold", leading=13, spaceBefore=4, spaceAfter=1),
         "body": PS("bd", fontSize=9.5, fontName="Helvetica", leading=15, spaceAfter=2, alignment=TA_JUSTIFY),
         "figcap": PS("fc", fontSize=8, fontName="Helvetica-Oblique", textColor=gry, leading=11,
                      alignment=TA_CENTER, spaceAfter=3),
         "disc": PS("di", fontSize=7, fontName="Helvetica-Oblique", textColor=gry, leading=10, alignment=TA_CENTER)}
    if indic_font:
        s["indic"] = PS("in", fontSize=10, fontName=indic_font, leading=16)
    return s


def _esc(t: str) -> str:
    return (t or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")


def build_pdf(result: dict, image, out_path: str, title: str, subtitle: str, patient_id: str = "-",
              summary: dict | None = None, indic_font_ttf: str | None = None) -> str:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    from reportlab.platypus import Image as RLImage

    font = None
    if indic_font_ttf:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont

        pdfmetrics.registerFont(TTFont("IndicFont", indic_font_ttf))
        font = "IndicFont"
    S = _styles(font)
    W = A4[0] - 40 * mm
    HR = lambda t=0.6: HRFlowable(width="100%", thickness=t, color=colors.HexColor("#BBBBBB"),  # noqa: E731
                                  spaceBefore=3, spaceAfter=3)
    SP = lambda n: Spacer(1, n * mm)  # noqa: E731
    story = [SP(1), Paragraph(_esc(title), S["title"]), Paragraph(_esc(subtitle), S["subtitle"]), HR(1.0), SP(3)]
    rows = [[Paragraph("PATIENT NUMBER:", S["ik"]), Paragraph(_esc(patient_id), S["iv"]),
             Paragraph("STUDY DATE:", S["ik"]), Paragraph(_dt.datetime.now().strftime("%d %B %Y"), S["iv"])],
            [Paragraph("EXAM:", S["ik"]), Paragraph("CHEST X-RAY (frontal)", S["iv"]),
             Paragraph("SYSTEM:", S["ik"]), Paragraph(_esc(result.get("system", "-")), S["iv"])]]
    if result.get("indication"):
        rows.append([Paragraph("INDICATION:", S["ik"]), Paragraph(_esc(result["indication"]), S["iv"]),
                     Paragraph("", S["ik"]), Paragraph("", S["iv"])])
    t = Table(rows, colWidths=[W * 0.18, W * 0.32, W * 0.20, W * 0.30])
    t.setStyle(TableStyle([("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                           ("LEFTPADDING", (0, 0), (-1, -1), 0), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story += [t, SP(2), HR(), SP(3)]
    if image is not None:
        buf = io.BytesIO()
        th = image.convert("RGB").copy()
        th.thumbnail((560, 560))
        th.save(buf, format="JPEG", quality=92)
        buf.seek(0)
        iw = 90 * mm
        img = Table([[RLImage(buf, width=iw, height=iw * th.size[1] / th.size[0])]], colWidths=[W])
        img.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
        story += [img, Paragraph("Figure 1. Input chest X-ray with the boxes behind the sentence verdicts.", S["figcap"]),
                  HR(), SP(3)]
    cur = None
    for s in result.get("sentences", []):
        if s["section"] != cur:
            cur = s["section"]
            story.append(Paragraph(f"{cur.upper()}:", S["shdr"]))
        col = COLORS.get(s.get("verdict", "NOT_CHECKED"), "#555555")
        story.append(Paragraph(f"<font color='{col}'>[{s.get('verdict', 'NOT_CHECKED')}]</font> {_esc(s['text'])}",
                               S["body"]))
    if result.get("cases"):
        story += [SP(2), Paragraph("SUPPORTING CASES (other patients, style reference only):", S["shdr"])]
        for c in result["cases"]:
            story.append(Paragraph(_esc(f"{c['study_id']} (similarity {c['sim']:.2f}, label agreement "
                                        f"{c['agreement']:.2f}): {c['snippet']}"), S["body"]))
    if summary:
        story += [SP(2), Paragraph("PATIENT SUMMARY (plain English):", S["shdr"]), Paragraph(_esc(summary["summary_en"]), S["body"])]
        if summary.get("translation"):
            if font:
                story += [Paragraph(f"PATIENT SUMMARY ({summary['language'].upper()}):", S["shdr"]),
                          Paragraph(_esc(summary["translation"]), S["indic"])]
            else:
                story.append(Paragraph("(Translated summary omitted: no Indic font configured - app.pdf.indic_font_ttf.)",
                                       S["figcap"]))
            if not summary.get("faithful", True):
                story.append(Paragraph("<font color='#B71C1C'>Translation FLAGGED: its back-translation does not match "
                                       f"the report's findings ({_esc(', '.join(summary.get('lost', []) + summary.get('added', [])))}).</font>",
                                       S["body"]))
    story += [SP(4), HR(), Paragraph(_esc(DISCLAIMER), S["disc"])]
    SimpleDocTemplate(out_path, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=16 * mm,
                      bottomMargin=16 * mm).build(story)
    return out_path
