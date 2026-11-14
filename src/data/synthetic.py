"""Rule-based template reports from image-level labels (CheXpert, NIH ChestX-ray14) for Step 4 alignment.

No external API. Labels-only datasets are used here ONLY as training material for the projector, never as report
ground truth for evaluation (CLAUDE.md §3). Templates never state laterality (the labels do not have it) and
never contradict a label: boilerplate negatives are only added for findings that are not positive/uncertain.

Verified file layouts (Kaggle, 2026-11-14):
  CheXpert (ashery/chexpert, "small"): train.csv / valid.csv; Path "CheXpert-v1.0-small/train/patientX/studyY/
    viewZ_frontal.jpg" while the files live at <root>/train/...; labels 1 / 0 / -1 (uncertain) / blank.
  NIH (nih-chest-xrays/data): Data_Entry_2017.csv ("Finding Labels" pipe-separated, "View Position" PA/AP),
    official patient-disjoint train_val_list.txt / test_list.txt, images under images_XXX/images/.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import pandas as pd

from src.data.text_clean import format_report

# finding -> (positive variants, uncertain variants, negative variant)
PHRASES = {
    "Enlarged Cardiomediastinum": (["The cardiomediastinal silhouette is enlarged.", "The mediastinum is widened."],
                                   ["The cardiomediastinal silhouette may be enlarged."],
                                   "The cardiomediastinal silhouette is within normal limits."),
    "Cardiomegaly": (["The heart is enlarged.", "There is cardiomegaly.", "Moderate cardiomegaly is present."],
                     ["The heart size is borderline enlarged."], "The heart size is normal."),
    "Lung Opacity": (["There are patchy airspace opacities.", "There is an area of increased lung opacity."],
                     ["A subtle opacity may be present."], None),
    "Lung Lesion": (["There is a pulmonary nodule.", "A lung mass is seen."],
                    ["A possible nodular density is noted."], None),
    "Edema": (["There is pulmonary edema.", "Findings are consistent with pulmonary vascular congestion and edema."],
              ["Mild interstitial edema is possible."], "There is no pulmonary edema."),
    "Consolidation": (["There is focal consolidation.", "Airspace consolidation is present."],
                      ["Consolidation cannot be excluded."], "There is no focal consolidation."),
    "Pneumonia": (["Findings are concerning for pneumonia."], ["Pneumonia is a possibility."], None),
    "Atelectasis": (["There is atelectasis.", "Linear atelectasis is noted."],
                    ["Possible minimal atelectasis."], None),
    "Pneumothorax": (["There is a pneumothorax."], ["A small pneumothorax cannot be excluded."],
                     "There is no pneumothorax."),
    "Pleural Effusion": (["There is a pleural effusion.", "A small pleural effusion is present."],
                         ["A trace pleural effusion is possible."], "There is no pleural effusion."),
    "Pleural Other": (["There is pleural thickening."], ["Possible pleural thickening."], None),
    "Fracture": (["A rib fracture is seen."], ["A possible rib fracture is noted."], "No acute fracture is seen."),
    "Support Devices": (["Support devices are in place.", "Lines and tubes are present."],
                        ["A support device may be present."], None),
}
IMPRESSION_NAME = {"Enlarged Cardiomediastinum": "enlarged cardiomediastinum", "Cardiomegaly": "cardiomegaly",
                   "Lung Opacity": "lung opacity", "Lung Lesion": "pulmonary nodule or mass", "Edema": "pulmonary edema",
                   "Consolidation": "consolidation", "Pneumonia": "pneumonia", "Atelectasis": "atelectasis",
                   "Pneumothorax": "pneumothorax", "Pleural Effusion": "pleural effusion",
                   "Pleural Other": "pleural thickening", "Fracture": "fracture", "Support Devices": "support devices"}
NORMAL = ["The lungs are clear.", "The heart size is normal.", "There is no pleural effusion.",
          "There is no pneumothorax."]
BOILERPLATE_FOR = {"The lungs are clear.": {"Lung Opacity", "Lung Lesion", "Edema", "Consolidation", "Pneumonia",
                                            "Atelectasis"},
                   "The heart size is normal.": {"Cardiomegaly", "Enlarged Cardiomediastinum"},
                   "There is no pleural effusion.": {"Pleural Effusion"},
                   "There is no pneumothorax.": {"Pneumothorax"}}

# NIH ChestX-ray14 label -> template finding (or an extra phrase for labels CheXpert does not have)
NIH_MAP = {"Atelectasis": "Atelectasis", "Cardiomegaly": "Cardiomegaly", "Consolidation": "Consolidation",
           "Edema": "Edema", "Effusion": "Pleural Effusion", "Infiltration": "Lung Opacity", "Mass": "Lung Lesion",
           "Nodule": "Lung Lesion", "Pneumonia": "Pneumonia", "Pneumothorax": "Pneumothorax",
           "Pleural_Thickening": "Pleural Other"}
NIH_EXTRA = {"Emphysema": ("The lungs are hyperinflated, consistent with emphysema.", "emphysema"),
             "Fibrosis": ("There are fibrotic changes in the lungs.", "pulmonary fibrosis"),
             "Hernia": ("A hiatal hernia is present.", "hiatal hernia")}


def template_report(pos: list[str], unc: list[str], neg: list[str], rng: random.Random,
                    extra_pos: list[tuple[str, str]] = ()) -> str:
    """pos/unc/neg: template findings (PHRASES keys). extra_pos: (sentence, impression name) pairs."""
    pos, unc = [f for f in pos if f in PHRASES], [f for f in unc if f in PHRASES and f not in pos]
    sents = [rng.choice(PHRASES[f][0]) for f in pos] + [s for s, _ in extra_pos]
    rng.shuffle(sents)
    sents += [rng.choice(PHRASES[f][1]) for f in unc]
    claimed = set(pos) | set(unc)
    negs = [PHRASES[f][2] for f in neg if f in PHRASES and PHRASES[f][2] and f not in claimed]
    for s in NORMAL:   # boilerplate only where no label says otherwise
        if not (BOILERPLATE_FOR[s] & claimed) and s not in negs:
            if not pos or rng.random() < 0.5:
                negs.append(s)
    seen, findings = set(), []
    for s in sents + negs:
        if s not in seen:
            seen.add(s)
            findings.append(s)
    names = [IMPRESSION_NAME[f] for f in pos] + [n for _, n in extra_pos]
    if names:
        imp = (names[0].capitalize() if len(names) == 1 else
               ", ".join(names[:-1]).capitalize() + " and " + names[-1]) + "."
    elif unc:
        imp = f"Possible {IMPRESSION_NAME[unc[0]]}."
    else:
        imp = "No acute cardiopulmonary abnormality."
    return format_report(" ".join(findings), imp)


CHEXPERT_FINDINGS = list(PHRASES)


def load_chexpert(root: Path, split: str, n: int | None, seed: int, frontal_views=("AP", "PA")) -> pd.DataFrame:
    """split 'train' -> train.csv, 'val' -> valid.csv. Frontal AP/PA only. Seeded subset of n images."""
    root = Path(root)
    df = pd.read_csv(root / ("train.csv" if split == "train" else "valid.csv"))
    df = df[(df["Frontal/Lateral"] == "Frontal") & df["AP/PA"].isin(frontal_views)].copy()
    rel = df["Path"].str.replace(r"^CheXpert-v1\.0(-small)?/", "", regex=True)
    df["image_path"] = [str(root / r) for r in rel]
    df["patient_id"] = "chexpert:" + rel.str.extract(r"(patient\d+)")[0]
    df["study_id"] = "chexpert:" + rel.str.replace(r"\.jpg$", "", regex=True)
    if n and n < len(df):
        df = df.sample(n=n, random_state=seed)
    rng = random.Random(seed)
    rows = []
    for r in df.sort_values("study_id").to_dict("records"):
        pos = [f for f in CHEXPERT_FINDINGS if r.get(f) == 1]
        unc = [f for f in CHEXPERT_FINDINGS if r.get(f) == -1]
        neg = [f for f in CHEXPERT_FINDINGS if r.get(f) == 0]
        rows.append({"dataset": "chexpert_synth", "study_id": r["study_id"], "patient_id": r["patient_id"],
                     "split": split, "image_path": r["image_path"], "indication": "",
                     "target": template_report(pos, unc, neg, rng), "source": "synthetic"})
    return pd.DataFrame(rows)


def _nih_index(root: Path) -> dict[str, str]:
    out = {}
    for d in sorted(Path(root).glob("images_*/images")):
        with os.scandir(d) as it:
            for e in it:
                out[e.name] = str(d / e.name)
    return out


def load_nih(root: Path, n: int | None, seed: int) -> pd.DataFrame:
    """Official train_val list only (test_list never used); every NIH image is frontal (PA/AP)."""
    root = Path(root)
    df = pd.read_csv(root / "Data_Entry_2017.csv")
    tv = set((root / "train_val_list.txt").read_text().split())
    df = df[df["Image Index"].isin(tv) & df["View Position"].isin(["PA", "AP"])]
    if n and n < len(df):
        df = df.sample(n=n, random_state=seed)
    idx = _nih_index(root)
    rng = random.Random(seed)
    rows = []
    for r in df.sort_values("Image Index").to_dict("records"):
        if r["Image Index"] not in idx:
            continue
        labs = [x for x in str(r["Finding Labels"]).split("|") if x and x != "No Finding"]
        pos = sorted({NIH_MAP[x] for x in labs if x in NIH_MAP})
        extra = [NIH_EXTRA[x] for x in labs if x in NIH_EXTRA]
        rows.append({"dataset": "nih_synth", "study_id": "nih:" + r["Image Index"],
                     "patient_id": f"nih:{r['Patient ID']}", "split": "train", "image_path": idx[r["Image Index"]],
                     "indication": "", "target": template_report(pos, [], [], rng, extra), "source": "synthetic"})
    return pd.DataFrame(rows)
