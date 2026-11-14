"""Sentence verdicts (VERIFIED / UNCERTAIN / FLAGGED / NOT_CHECKED) and how they are evaluated.

Evidence for one claim (finding f, polarity, side, zone):
  probe : Step 2 probe probability p_f vs its val-tuned threshold t_f. "strong positive": p >= t + m(1-t);
          "strong negative": p <= t(1-m)  (m = verify.strong_margin).
  box   : MedGemma 1.5 localisation of f (positive claims only): absent (model returned []), present (sides /
          zones of its boxes), or unknown (unparseable / not queried).
Rules (fixed here, before any result is seen):
  positive claim, localisable finding:
    FLAGGED   if a box exists but ONLY on the opposite side of the stated side (laterality contradiction),
              or the probe is negative AND the box query says absent (no image support)
    VERIFIED  if probe positive AND a box exists whose side matches (or no side was stated) and zone does not
              contradict
    UNCERTAIN otherwise (one source agrees, zone mismatch, or box unknown)
  positive "No Finding" claim (normal study): VERIFIED if probe positive, FLAGGED if strong negative, else UNCERTAIN
  negative claim ("no effusion"): VERIFIED if probe negative, FLAGGED if strong positive, else UNCERTAIN
  uncertain claim ("possible pneumonia"): UNCERTAIN
  sentence with no CheXbert finding: NOT_CHECKED.  Sentence verdict = worst claim (FLAGGED > UNCERTAIN > VERIFIED).
"""
from __future__ import annotations

ORDER = {"VERIFIED": 0, "UNCERTAIN": 1, "FLAGGED": 2}
NO_BOX = {"No Finding"}
OPPOSITE = {"left": "right", "right": "left"}


def probe_state(p: float, t: float, m: float) -> dict:
    return {"p": round(float(p), 4), "threshold": round(float(t), 4), "positive": p >= t,
            "strong_positive": p >= t + m * (1 - t), "strong_negative": p <= t * (1 - m)}


def side_match(claim_side: str | None, box_sides: list[str]) -> str:
    """'match' | 'contradiction' | 'none' (no usable boxes)."""
    if not box_sides:
        return "none"
    if claim_side is None:
        return "match"
    if claim_side == "bilateral":
        s = set(box_sides)
        return "match" if ("both" in s or {"left", "right"} <= s or "center" in s) else "partial"
    if claim_side in box_sides or "both" in box_sides or "center" in box_sides:
        return "match"
    return "contradiction" if all(b == OPPOSITE[claim_side] for b in box_sides) else "match"


def claim_verdict(claim: dict, probe: dict, box: dict | None) -> tuple[str, str]:
    """-> (verdict, reason). box = {"status": ok|empty|unparseable, "sides": [...], "zones": [...]} or None."""
    f, pol = claim["finding"], claim["polarity"]
    if pol == "uncertain":
        return "UNCERTAIN", "the sentence itself is uncertain"
    if pol == "negative":
        if probe["strong_positive"]:
            return "FLAGGED", f"says no {f}, probe strongly positive (p={probe['p']})"
        if not probe["positive"]:
            return "VERIFIED", "probe agrees it is absent"
        return "UNCERTAIN", f"says no {f}, probe weakly positive (p={probe['p']})"
    if f in NO_BOX:
        if probe["positive"]:
            return "VERIFIED", "probe agrees the study is normal"
        if probe["strong_negative"]:
            return "FLAGGED", "says normal, probe strongly disagrees"
        return "UNCERTAIN", "says normal, probe weakly disagrees"
    status = (box or {}).get("status", "unknown")
    sides, zones = (box or {}).get("sides", []), (box or {}).get("zones", [])
    sm = side_match(claim.get("laterality"), sides) if status == "ok" else "none"
    if sm == "contradiction":
        return "FLAGGED", f"says {claim['laterality']}, box only on the {OPPOSITE[claim['laterality']]}"
    if not probe["positive"] and status == "empty":
        return "FLAGGED", "no image support (probe negative, no box found)"
    zone_bad = (status == "ok" and claim.get("zone") in ("upper", "lower") and zones
                and all(z == ("lower" if claim["zone"] == "upper" else "upper") for z in zones))
    if probe["positive"] and status == "ok" and sm in ("match",) and not zone_bad:
        return "VERIFIED", "probe positive and box on the stated side"
    parts = [f"probe {'positive' if probe['positive'] else 'negative'} (p={probe['p']})",
             {"ok": f"box side {sm}" + (", zone mismatch" if zone_bad else ""), "empty": "no box found"}
             .get(status, "box unavailable")]
    return "UNCERTAIN", "; ".join(parts)


def sentence_verdict(claim_verdicts: list[str]) -> str:
    if not claim_verdicts:
        return "NOT_CHECKED"
    return max(claim_verdicts, key=ORDER.__getitem__)


# ------------------------------------------------------------------ evaluation against the reference report
def claim_correct(claim: dict, ref_class: int, ref_sides: set | None) -> bool | None:
    """True / False / None (not judgeable). ref_class: raw CheXbert class of the finding in the REFERENCE report
    (0 blank, 1 pos, 2 neg, 3 unc). A positive claim is wrong if the reference does not report the finding
    (blank/negative) or reports it only on the other side; a negative claim is wrong if the reference reports it."""
    pol = claim["polarity"]
    if pol == "uncertain" or ref_class == 3:
        return None
    if claim["finding"] == "No Finding":
        if pol != "positive":
            return None
        return ref_class == 1
    if pol == "positive":
        if ref_class != 1:
            return False
        side, rs = claim.get("laterality"), ref_sides or set()
        if side in ("left", "right") and rs and side not in rs and "bilateral" not in rs:
            return False
        return True
    return ref_class != 1   # negative claim


def sentence_correct(judgements: list[bool | None]) -> bool | None:
    js = [j for j in judgements if j is not None]
    if not js:
        return None
    return all(js)


def drop_flagged(sentences: list[dict]) -> str:
    return " ".join(s["text"] for s in sentences if s["verdict"] != "FLAGGED")
