"""Report text cleaning and section extraction.

The MIMIC section parser follows the logic of the official MIT-LCP `section_parser.py`
(headings followed by a colon, synonym groups), simplified: a study with no explicit FINDINGS
heading gets empty findings and is dropped by `require_findings`; we never guess FINDINGS from
free text.
"""
from __future__ import annotations

import re

# De-identification placeholders: MIMIC uses "___", IU-Xray uses "XXXX".
_PHI_RE = re.compile(r"_{2,}|\bX{2,}\b")
_WS_RE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:])")
_REPEAT_PUNCT_RE = re.compile(r"([,.;:])(\s*[,.;:])+")

NON_INFORMATIVE = {
    "", "as above", "see above", "none", "n/a", "na", "nil", "---", "--", "-",
    "see impression", "see findings", "same", "unchanged", "no findings", "not applicable",
}

# Heading synonyms -> canonical section. Order matters for "FINDINGS AND IMPRESSION".
SECTION_SYNONYMS = {
    "indication": [
        "INDICATION", "INDICATIONS", "CLINICAL INDICATION", "HISTORY", "CLINICAL HISTORY",
        "REASON FOR EXAM", "REASON FOR EXAMINATION", "CLINICAL INFORMATION", "CLINICAL DETAILS",
    ],
    "findings": ["FINDINGS AND IMPRESSION", "FINDINGS", "FINDING"],
    "impression": ["IMPRESSION", "IMPRESSIONS", "CONCLUSION", "CONCLUSIONS", "SUMMARY", "OPINION"],
    "comparison": ["COMPARISON", "COMPARISONS", "PRIOR", "PRIORS"],
    "technique": ["TECHNIQUE", "EXAMINATION", "EXAM", "PROCEDURE", "STUDY"],
}
_HEAD_TO_SECTION = {h: s for s, hs in SECTION_SYNONYMS.items() for h in hs}
# A heading = start of line (or start of text), an ALL-CAPS phrase, then a colon.
_HEADING_RE = re.compile(r"(?:^|\n)\s*([A-Z][A-Z /&()-]{1,40}?)\s*:", re.MULTILINE)


def remove_phi(text: str | None) -> str:
    """Strip de-id placeholders and tidy the punctuation/spacing they leave behind."""
    if not text or not isinstance(text, str):
        return ""
    t = _PHI_RE.sub(" ", text)
    t = _WS_RE.sub(" ", t)
    t = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", t)
    t = _REPEAT_PUNCT_RE.sub(r"\1", t)
    return t.strip(" ,;:")


def normalize(text: str | None) -> str:
    return _WS_RE.sub(" ", remove_phi(text)).strip()


def is_non_informative(text: str | None, min_alpha_chars: int = 10) -> bool:
    t = normalize(text)
    key = re.sub(r"[^a-z/ -]", "", t.lower()).strip(" -")
    if key in NON_INFORMATIVE or t.strip(" .-") in {"", "---"}:
        return True
    return sum(c.isalpha() for c in t) < min_alpha_chars


def parse_sections(report: str) -> dict[str, str]:
    """Split a free-text report into canonical sections. Unknown headings are ignored.

    If a canonical section appears twice (e.g. 'FINDINGS:' then 'REPORT:'), the texts are joined.
    """
    out = {k: "" for k in SECTION_SYNONYMS}
    if not report:
        return out
    text = report.replace("\r\n", "\n").replace("\r", "\n")
    matches = list(_HEADING_RE.finditer(text))
    for i, m in enumerate(matches):
        head = _WS_RE.sub(" ", m.group(1)).strip()
        section = _HEAD_TO_SECTION.get(head)
        if section is None:
            continue
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = normalize(text[m.end() : end])
        if body:
            out[section] = (out[section] + " " + body).strip() if out[section] else body
    return out


def format_report(findings: str, impression: str) -> str:
    """Target format shared by both models (CLAUDE.md §5)."""
    return f"FINDINGS: {findings}\nIMPRESSION: {impression}"


_GEN_FIND_RE = re.compile(r"\*{0,2}\s*FINDINGS?\s*\*{0,2}\s*:\s*\*{0,2}", re.IGNORECASE)
_GEN_IMP_RE = re.compile(r"\*{0,2}\s*IMPRESSIONS?\s*\*{0,2}\s*:\s*\*{0,2}", re.IGNORECASE)


def parse_generated(text: str) -> dict:
    """Parse model output into findings/impression.

    Returns parse_ok=False when the FINDINGS header is missing; then the whole text is used as findings
    (scored as-is, never repaired) so format failures are penalised, not hidden.
    """
    t = (text or "").strip()
    f = _GEN_FIND_RE.search(t)
    i = _GEN_IMP_RE.search(t)
    if f and i and i.start() > f.start():
        findings, impression = t[f.end() : i.start()], t[i.end() :]
    elif f:
        findings, impression = t[f.end() :], ""
    elif i:
        findings, impression = t[: i.start()], t[i.end() :]
    else:
        findings, impression = t, ""
    clean = lambda s: _WS_RE.sub(" ", s.replace("*", " ")).strip()
    return {
        "findings": clean(findings),
        "impression": clean(impression),
        "parse_ok": bool(f),
    }
