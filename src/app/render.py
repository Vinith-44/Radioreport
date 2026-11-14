"""Step 10 display helpers: colour-coded sentence verdicts (HTML) and finding boxes drawn on the X-ray."""
from __future__ import annotations

import html

from PIL import Image, ImageDraw

COLORS = {"VERIFIED": "#1B5E20", "UNCERTAIN": "#E65100", "FLAGGED": "#B71C1C", "NOT_CHECKED": "#555555"}
BG = {"VERIFIED": "#E8F5E9", "UNCERTAIN": "#FFF3E0", "FLAGGED": "#FFEBEE", "NOT_CHECKED": "#F5F5F5"}
DISCLAIMER = ("Research / radiologist-assist tool. NOT a diagnosis: every sentence must be checked by a "
              "qualified radiologist.")


def sentences_html(sentences: list[dict]) -> str:
    """sentences: [{section, text, verdict, claims:[{reason}]}] -> HTML with one coloured span per sentence."""
    parts, cur = [], None
    for s in sentences:
        if s["section"] != cur:
            cur = s["section"]
            parts.append(f"<div style='margin-top:8px;font-weight:bold'>{html.escape(cur.upper())}:</div>")
        v = s.get("verdict", "NOT_CHECKED")
        why = "; ".join(c.get("reason", "") for c in s.get("claims", []) if c.get("reason")) or v
        parts.append(f"<span title='{html.escape(why, quote=True)}' style='background:{BG[v]};color:{COLORS[v]};"
                     f"padding:1px 4px;margin:1px;border-radius:3px;display:inline-block'>{html.escape(s['text'])}"
                     f" <sup>{v[0]}</sup></span>")
    legend = " ".join(f"<span style='color:{COLORS[k]}'>&#9632; {k}</span>" for k in COLORS)
    return ("<div style='font-family:sans-serif;line-height:1.7'>" + "".join(parts) +
            f"<div style='margin-top:10px;font-size:12px'>{legend}</div>"
            f"<div style='margin-top:6px;font-size:11px;color:#555'>{html.escape(DISCLAIMER)}</div></div>")


def draw_boxes(image: Image.Image, boxes: list[dict]) -> Image.Image:
    """boxes: [{finding, box: [x0, y0, x1, y1] in 0-1 of the image, verdict}]"""
    im = image.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    w, h = im.size
    lw = max(2, w // 300)
    for b in boxes:
        x0, y0, x1, y1 = b["box"]
        col = COLORS.get(b.get("verdict", "NOT_CHECKED"), "#1A237E")
        d.rectangle([x0 * w, y0 * h, x1 * w, y1 * h], outline=col, width=lw)
        d.text((x0 * w + lw + 2, max(0, y0 * h - 14)), b["finding"], fill=col)
    return im
