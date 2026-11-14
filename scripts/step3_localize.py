"""Step 3b: MedGemma 1.5 boxes for every POSITIVE claim (one query per study x finding). Resumable.

  python scripts/step3_localize.py --run-name v1 [--smoke]

Official localisation prompt + square padding (src/verify/boxes.py). Appends to outputs/runs/step3/<run>/
boxes.jsonl: status ok|empty|unparseable, boxes in original-image coordinates [x0,y0,x1,y1] (0-1), the patient
side and zone of each box, and the raw answer (for audit). Re-running skips finished (study, finding) pairs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.common import append_jsonl, data_dir, load_config, read_jsonl, run_dir, set_seed, write_run_info  # noqa: E402
from src.verify.boxes import box_zone, localize_prompt, pad_to_square, parse_boxes, patient_side, to_image_box  # noqa: E402
from src.verify.verdict import NO_BOX  # noqa: E402


def locate(model, processor, items: list[dict], phrases: dict, bc: dict, band: float, out_p: Path, log=print) -> None:
    from src.models.medgemma_zeroshot import generate_batch, load_image

    bs = int(bc["batch_size"])
    t0, n_bad = time.time(), 0
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        imgs, pads = [], []
        for it in chunk:
            im, pad = pad_to_square(load_image(it["image_path"]))
            imgs.append(im)
            pads.append(pad)
        outs = generate_batch(model, processor, imgs, [localize_prompt(phrases[it["finding"]]) for it in chunk],
                              int(bc["max_new_tokens"]))
        lines = []
        for it, o, pad in zip(chunk, outs, pads):
            boxes, status = parse_boxes(o["text"])
            img_boxes = [to_image_box(b["box_2d"], pad) for b in (boxes or [])]
            n_bad += status == "unparseable"
            lines.append({**{k: it[k] for k in ("study_id", "dataset", "finding", "label") if k in it},
                          "query": phrases[it["finding"]],
                          "status": status, "boxes": [[round(v, 4) for v in b] for b in img_boxes],
                          "sides": [patient_side(b, band) for b in img_boxes], "zones": [box_zone(b) for b in img_boxes],
                          "hit_max_tokens": o["hit_max_tokens"], "raw": o["text"][-1500:]})
        append_jsonl(out_p, lines)
        done = i + len(chunk)
        rate = (time.time() - t0) / done
        log(f"  {done}/{len(items)} queries  {rate:.1f}s/query  ETA {rate * (len(items) - done) / 60:.0f} min  "
            f"unparseable so far {n_bad}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/step3.yaml")
    ap.add_argument("--run-name", default="v1")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dtype", choices=["fp32", "fp16", "nf4"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    bc = dict(cfg["box"])
    if args.dtype:
        bc["dtype"] = args.dtype
    sfx = "_smoke" if args.smoke else ""
    rdir = run_dir(cfg, "step3", args.run_name + sfx)
    out_p = rdir / "boxes.jsonl"
    meta = {"model_id": cfg["models"][bc["model"]], "dtype": bc["dtype"], "max_new_tokens": int(bc["max_new_tokens"]),
            "phrases": bc["phrases"], "prompt": localize_prompt("{object}")}
    mp = rdir / "boxes_meta.json"
    if mp.exists() and json.loads(mp.read_text(encoding="utf-8")) != meta:
        raise SystemExit("boxes.jsonl was made with other settings; use a new --run-name")
    mp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_run_info(rdir, cfg, args, {"step": "step3_localize", **meta})

    paths = pd.read_parquet(data_dir(cfg) / f"test{sfx}.parquet", columns=["study_id", "image_path"])
    paths = dict(zip(paths["study_id"].astype(str), paths["image_path"]))
    want = {}
    for r in read_jsonl(rdir / "claims.jsonl"):
        for s in r["sentences"]:
            for c in s["claims"]:
                if c["polarity"] == "positive" and c["finding"] not in NO_BOX:
                    want[(r["study_id"], c["finding"])] = {"study_id": r["study_id"], "dataset": r["dataset"],
                                                           "finding": c["finding"], "image_path": paths[r["study_id"]]}
    done = {(r["study_id"], r["finding"]) for r in read_jsonl(out_p)}
    todo = [v for k, v in sorted(want.items()) if k not in done]
    print(f"{len(want)} (study, finding) queries, {len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        return
    from src.models.medgemma_zeroshot import load_model

    t0 = time.time()
    model, processor = load_model(meta["model_id"], bc["dtype"])
    print(f"loaded {meta['model_id']} ({bc['dtype']}) in {time.time() - t0:.0f}s", flush=True)
    locate(model, processor, todo, bc["phrases"], bc, float(cfg["verify"]["side_center_band"]), out_p,
           log=lambda m: print(m, flush=True))
    rs = read_jsonl(out_p)
    st = pd.Series([r["status"] for r in rs]).value_counts().to_dict()
    print(f"\nstatus counts: {st}; hit max tokens: {sum(r['hit_max_tokens'] for r in rs)}")
    for r in rs[:2]:
        print(f"--- {r['study_id']} / {r['finding']}: {r['status']} {r['boxes']} sides={r['sides']}\n{r['raw'][-400:]}")


if __name__ == "__main__":
    main()
