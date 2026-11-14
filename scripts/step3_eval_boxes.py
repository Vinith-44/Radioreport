"""Step 3e: how good are MedGemma 1.5's FINDING boxes? Evaluated on NIH ChestX-ray14 expert boxes.

  python scripts/step3_eval_boxes.py --run-name v1 [--smoke]

Same model, prompt and query phrases as step3_localize.py (NIH label -> CheXbert finding -> box.phrases), so the
numbers describe exactly the box evidence the verifier uses. NIH is a labels-only dataset: used here ONLY to
evaluate localisation, never as report ground truth. MS-CXR (the spec's choice) needs MIMIC access.
Metrics per NIH label: answer parse rate, "absent" rate (model found nothing although an expert box exists),
best IoU, hit rate at IoU >= 0.3 / 0.5, pointing accuracy (predicted centre inside an expert box), side accuracy.
Writes outputs/runs/step3/<run>/nih_boxes.jsonl (resumable) and nih_box_eval.json / .md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.common import load_config, read_jsonl, run_dir, set_seed, write_run_info  # noqa: E402
from src.verify.boxes import iou, patient_side  # noqa: E402


def read_nih_boxes(csv_path: Path) -> pd.DataFrame:
    """BBox_List_2017.csv: 'Image Index','Finding Label','Bbox [x','y','w','h]' + empty trailing columns."""
    df = pd.read_csv(csv_path)
    df = df.iloc[:, :6]
    df.columns = ["image", "label", "x", "y", "w", "h"]
    return df


def find_images(root: Path, names: set[str]) -> dict[str, str]:
    out = {}
    for d in sorted(root.glob("images_*/images")):
        for n in names - set(out):
            p = d / n
            if p.exists():
                out[n] = str(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/step3.yaml")
    ap.add_argument("--run-name", default="v1")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg["seed"])
    set_seed(seed)
    nc, bc = cfg["nih_boxes"], cfg["box"]
    band = float(cfg["verify"]["side_center_band"])
    rdir = run_dir(cfg, "step3", args.run_name + ("_smoke" if args.smoke else ""))
    root = Path(nc["root"])
    boxes = read_nih_boxes(root / nc["csv"])
    per = 2 if args.smoke else int(nc["per_label"])
    # one query per (image, label); all expert boxes of that pair count as ground truth
    pairs = boxes.groupby(["image", "label"]).size().reset_index()[["image", "label"]]
    pairs = (pairs.groupby("label", group_keys=False).apply(lambda g: g.sample(min(per, len(g)), random_state=seed))
             .sort_values(["label", "image"]).reset_index(drop=True))
    paths = find_images(root, set(pairs["image"]))
    pairs = pairs[pairs["image"].isin(paths)]
    write_run_info(rdir, cfg, args, {"step": "step3_eval_boxes", "n_pairs": len(pairs)})
    print(f"NIH: {len(boxes)} boxes; evaluating {len(pairs)} (image, label) pairs: "
          f"{pairs['label'].value_counts().to_dict()}", flush=True)

    out_p = rdir / "nih_boxes.jsonl"
    done = {(r["image"], r["label"]) for r in read_jsonl(out_p)}
    todo = [{"study_id": r["image"], "dataset": "nih", "finding": nc["label_map"][r["label"]], "label": r["label"],
             "image_path": paths[r["image"]]} for r in pairs.to_dict("records") if (r["image"], r["label"]) not in done]
    if todo:
        from scripts.step3_localize import locate  # noqa: E402  (same batching/parsing as the verifier)
        from src.models.medgemma_zeroshot import load_model

        model, processor = load_model(cfg["models"][bc["model"]], bc["dtype"])
        tmp = rdir / "nih_boxes.tmp.jsonl"
        tmp.unlink(missing_ok=True)
        locate(model, processor, todo, bc["phrases"], bc, band, tmp, log=lambda m: print(m, flush=True))
        rows = []
        for r in read_jsonl(tmp):   # each row carries its own NIH label (Mass and Nodule share one finding)
            r["image"] = r.pop("study_id")
            rows.append(r)
        from src.common import append_jsonl

        append_jsonl(out_p, rows)
        tmp.unlink(missing_ok=True)

    # score
    from PIL import Image

    res, per_label = [], {}
    for r in read_jsonl(out_p):
        gt = boxes[(boxes["image"] == r["image"]) & (boxes["label"] == r["label"])]
        with Image.open(paths[r["image"]]) as im:   # NIH images are 1024x1024, but never assume
            W, H = im.size
        g = [[x / W, y / H, (x + w) / W, (y + h) / H] for x, y, w, h in gt[["x", "y", "w", "h"]].values]
        pred = r["boxes"]
        best = max((iou(p, q) for p in pred for q in g), default=0.0)
        point = any(q[0] <= (p[0] + p[2]) / 2 <= q[2] and q[1] <= (p[1] + p[3]) / 2 <= q[3] for p in pred for q in g)
        g_sides = {patient_side(q, band) for q in g} - {"center", "both"}
        p_sides = {patient_side(p, band) for p in pred} - {"center", "both"}
        side_ok = None if not g_sides or not p_sides else bool(p_sides & g_sides)
        res.append({"label": r["label"], "status": r["status"], "best_iou": best, "point": point, "side_ok": side_ok})
    df = pd.DataFrame(res)
    for lab_, g in df.groupby("label"):
        s = g["side_ok"].dropna()
        per_label[lab_] = {"n": len(g), "parsed": round(float((g["status"] != "unparseable").mean()), 3),
                           "absent (missed)": round(float((g["status"] == "empty").mean()), 3),
                           "mean_best_iou": round(float(g["best_iou"].mean()), 3),
                           "hit@0.3": round(float((g["best_iou"] >= 0.3).mean()), 3),
                           "hit@0.5": round(float((g["best_iou"] >= 0.5).mean()), 3),
                           "pointing": round(float(g["point"].mean()), 3),
                           "side_acc": None if s.empty else round(float(s.astype(float).mean()), 3), "side_n": len(s)}
    s = df["side_ok"].dropna()
    overall = {"n": len(df), "parsed": round(float((df["status"] != "unparseable").mean()), 3),
               "absent (missed)": round(float((df["status"] == "empty").mean()), 3),
               "mean_best_iou": round(float(df["best_iou"].mean()), 3), "hit@0.3": round(float((df["best_iou"] >= 0.3).mean()), 3),
               "pointing": round(float(df["point"].mean()), 3),
               "side_acc": None if s.empty else round(float(s.astype(float).mean()), 3), "side_n": len(s)}
    out = {"overall": overall, "per_label": per_label, "note": "NIH ChestX-ray14 expert boxes; MS-CXR not available"}
    (rdir / "nih_box_eval.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    head = list(overall)
    md = ["| NIH label | " + " | ".join(head) + " |", "|" + "---|" * (len(head) + 1)]
    for k, v in [*per_label.items(), ("ALL", overall)]:
        md.append(f"| {k} | " + " | ".join("-" if v[h] is None else str(v[h]) for h in head) + " |")
    (rdir / "nih_box_eval.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
