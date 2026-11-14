"""Bounding boxes from MedGemma 1.5's chest X-ray localization.

Verified against Google's official notebook google-health/medgemma
notebooks/cxr_anatomy_localization_with_hugging_face.ipynb (repo commit a60a660, 2026-04-07):
  - the prompt below (verbatim, with the queried object substituted);
  - the image is zero-padded to a square first;
  - output: a ```json [...]``` list of {"box_2d": [y0, x0, y1, x1], "label": ...}, coordinates normalised to
    [0, 1000] of the PADDED square image; the model may "think" between <unused94> and <unused95> first.
CAVEAT: Google's example and the model card's 38.0 IoU (Chest ImaGenome) are for ANATOMY boxes; boxes for
FINDINGS are not documented, so step3_eval_boxes.py measures them on NIH ChestX-ray14 boxes before they are
trusted. In images, the patient's RIGHT is on the image's LEFT (frontal view, standard orientation).
"""
from __future__ import annotations

import json
import re

import numpy as np
from PIL import Image

LOCALIZE_PROMPT = """Instructions:
The following user query will require outputting bounding boxes. The format of bounding boxes coordinates is [y0, x0, y1, x1] where (y0, x0) must be top-left corner and (y1, x1) the bottom-right corner. This implies that x0 < x1 and y0 < y1. Always normalize the x and y coordinates the range [0, 1000], meaning that a bounding box starting at 15% of the image width would be associated with an x coordinate of 150. You MUST output a single parseable json list of objects enclosed into ```json...``` brackets, for instance ```json[{{"box_2d": [800, 3, 840, 471], "label": "car"}}, {{"box_2d": [400, 22, 600, 73], "label": "dog"}}]``` is a valid output. Now answer to the user query.

Remember "left" refers to the patient's left side where the heart is and sometimes underneath an L in the upper right corner of the image.

Query:
Where is the {object_name}? Don't give a final answer without reasoning. Output the final answer in the format "Final Answer: X" where X is a JSON list of objects. The object needs a "box_2d" and "label" key. Answer:"""


def localize_prompt(object_name: str) -> str:
    return LOCALIZE_PROMPT.format(object_name=object_name)


def pad_to_square(im: Image.Image) -> tuple[Image.Image, dict]:
    """Zero-pad (centred, like the official notebook) and return the padding needed to map boxes back."""
    a = np.asarray(im.convert("RGB"))
    h, w = a.shape[:2]
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    out = np.zeros((s, s, 3), dtype=a.dtype)
    out[top:top + h, left:left + w] = a
    return Image.fromarray(out), {"size": s, "top": top, "left": left, "h": h, "w": w}


_JSON_BLOCK = re.compile(r"```json\s*(.*?)```", re.S)


def _strip_thinking(text: str) -> str:
    return text.split("<unused95>", 1)[1] if "<unused95>" in text else text


def parse_boxes(response: str) -> tuple[list[dict] | None, str]:
    """-> (boxes, status). boxes = [{"box_2d": [y0,x0,y1,x1] (0-1000 padded), "label"}] or None if unparseable.
    status: ok | empty (model says absent) | unparseable. Swapped corners within an axis are re-ordered."""
    text = _strip_thinking(response or "")
    cands = _JSON_BLOCK.findall(text)
    if not cands and "Final Answer:" in text:
        tail = text.rsplit("Final Answer:", 1)[1]
        i, j = tail.find("["), tail.rfind("]")
        if i != -1 and j > i:
            cands = [tail[i:j + 1]]
    for raw in reversed(cands):   # the final answer comes last
        try:
            data = json.loads(raw.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            continue
        boxes = []
        for it in data:
            b = it.get("box_2d") if isinstance(it, dict) else None
            if not (isinstance(b, (list, tuple)) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b)):
                continue
            y0, x0, y1, x1 = (float(v) for v in b)
            y0, y1 = sorted((y0, y1))
            x0, x1 = sorted((x0, x1))
            if not (0 <= y0 < y1 <= 1000 and 0 <= x0 < x1 <= 1000):
                continue
            boxes.append({"box_2d": [y0, x0, y1, x1], "label": str(it.get("label", ""))})
        return boxes, ("ok" if boxes else "empty")
    return None, "unparseable"


def to_image_box(box_2d, pad: dict) -> list[float]:
    """[y0,x0,y1,x1] in 0-1000 of the padded square -> [x0,y0,x1,y1] in 0-1 of the ORIGINAL image (clipped)."""
    s, t, l, h, w = pad["size"], pad["top"], pad["left"], pad["h"], pad["w"]
    y0, x0, y1, x1 = (v / 1000.0 * s for v in box_2d)
    xs = [min(max((x - l) / w, 0.0), 1.0) for x in (x0, x1)]
    ys = [min(max((y - t) / h, 0.0), 1.0) for y in (y0, y1)]
    return [xs[0], ys[0], xs[1], ys[1]]


def patient_side(box_xyxy: list[float], band: float = 0.05) -> str:
    """Side of the PATIENT a box lies on: image-left half = patient's right."""
    x0, _, x1, _ = box_xyxy
    if x0 < 0.5 - band and x1 > 0.5 + band:
        return "both"
    cx = (x0 + x1) / 2
    if abs(cx - 0.5) <= band:
        return "center"
    return "right" if cx < 0.5 else "left"


def box_zone(box_xyxy: list[float]) -> str:
    cy = (box_xyxy[1] + box_xyxy[3]) / 2
    return "upper" if cy < 0.4 else "lower" if cy > 0.6 else "middle"


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0
