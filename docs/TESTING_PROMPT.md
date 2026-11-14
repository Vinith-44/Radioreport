# Prompt: test Radioreport Steps 0-10 in detail (paste everything below the line into Claude Code)

---

You are testing the research code in this repository (github.com/Vinith-44/Radioreport, local copy
`C:\Users\VANJANGI VINITH\Radioreport`). Read `CLAUDE.md` first: its rules override everything here. The code for
Steps 0-10 is written and unit-tested on CPU (134 tests), but **no step has run on real data yet**. Your job is to run
every step on Kaggle in order, check each output against the checks below, fix real bugs (small, tested fixes),
and report results honestly. A step only counts as done when its CLAUDE.md acceptance check passes on real data.

## Ground rules (never break these)
1. **No invented numbers.** Every metric must come from a `predictions.jsonl` produced in that run and a
   `metrics.json` written by the scoring scripts. If something did not run, say "not run" - never estimate.
2. **No leakage.** Train/val/test are patient-disjoint; retrieval index, probe, GRPO prompts and training use the
   TRAIN split only; test is used only for scoring. Every script asserts this - a failing assert is a real bug in
   the data, never something to silence.
3. **The fixed test subset never changes.** `data/step0_eval_ids.json` written by Step 0 must be identical in every
   later session (the notebooks check this). If it changes, all comparisons are void: stop and find out why.
4. **Smoke first** (`--smoke`, ~50 studies, < 10 min), show its output, then the full run.
5. **Privacy.** ReXGradient (licence: no redistribution) and MIMIC must stay in PRIVATE Kaggle notebooks/outputs.
   Never commit images, parquet files, tokens or patient text to git. The user pasted an HF token and a Kaggle
   token in an earlier chat: remind them to revoke both. A `gradio --share` link is public: never share it.
6. Before changing code against a library, check the INSTALLED version's API (source / `help()`); the pinned
   versions are in `requirements*.txt` and were chosen because their source was read.
7. Commit / push only when the user asks. Ask before anything outward-facing.

## Environment facts you can rely on (verified 2026-11-14/15)
- Kaggle, 2x Tesla T4 (no bf16; Gemma 3 overflows in fp16 -> the code uses fp32 / nf4-with-fp32-compute /
  Unsloth's forced-float32 mode; never "fix" an overflow by switching to fp16).
- Secret `HF_TOKEN` must be TICKED in the notebook (Add-ons -> Secrets). Accept the HF terms of: google/medgemma-4b-it,
  google/medgemma-1.5-4b-it, google/medsiglip-448, google/gemma-3-1b-it, google/medasr,
  ai4bharat/indictrans2-en-indic-1B, ai4bharat/indictrans2-indic-en-1B, rajpurkarlab/ReXGradient-160K.
- Datasets: IU-Xray `raddar/chest-xrays-indiana-university`; CheXpert-small `ashery/chexpert`; NIH
  `organizations/nih-chest-xrays/data` (BBox_List_2017.csv, official train_val/test lists); CheXphoto copy
  `gajahterbang111/chexphoto-split` (UNOFFICIAL re-upload - ask the user about the licence before using it).
  MIMIC-CXR is NOT available (the Kaggle copy is unofficial) - Step 0 runs without it; say so in results.
- ReXGradient test images: output of the user's notebook `vinithvanjangi/rexgradient-test-fetch` (17,029 PNGs,
  9.6 GB, missing_count 0). **Kaggle's "New Dataset from output" copies only ~500 files** - attach the NOTEBOOK OUTPUT
  directly (Add Input -> Notebook Output). If that is also capped, change the fetch to write a few large tar
  shards (no file-count limit) and extract them to /kaggle/tmp at the start of each session.
- ReXGradient train subset (Step 1+): run the job in `kaggle/fetch_rexgradient_train/run.py` (CPU, ~2-3 h), attach
  its output the same way. `step0_prepare_data.py` finds both folders under /kaggle/input automatically.
- Each step has a thin notebook `notebooks/stepN_kaggle.ipynb`; restore earlier steps' outputs via the `PREV` /
  `STEP0` cells (copy `<input>/project/outputs` and `data` into /kaggle/working/project).

## Step-by-step test plan

For every step: run the smoke cell, read the whole output, then the full run, then the checks. Record for the
user: what ran, wall time, the key numbers (from files), PASS / NOT YET with the reason, and anything you changed.

### Step 0 - data + zero-shot baseline (`notebooks/step0_kaggle.ipynb`)
Run: `step0_prepare_data.py --inspect`, pytest, smoke (prepare/generate x2/score), full.
Check:
- `--inspect`: ReXGradient `studies with >=1 image present in the image roots` ~ 10000/10000 for test; IU columns found.
- `data/manifest.json`: drop reasons are plausible; studies per split; `assert_no_leakage` passed (patients, study ids).
- Smoke generation: outputs are non-empty, `parse_ok` mostly true, no NaN/empty-output abort (fp16 guard).
  Print 3 REF/HYP pairs and read them: the hypothesis must describe a chest X-ray, not echo the prompt.
- `step0_eval_ids.json`: >= 1,000 studies per dataset (`eval.min_test_studies`). IU-Xray: the code uses a seeded
  patient split (R2Gen's test split is only ~590 studies) - confirm the user accepted this; report it.
- Full run: 2,000 studies per dataset (if available) for medgemma-1.5 and medgemma-1, fp32 sharded over 2 GPUs.
  `metrics.json` per (model, dataset), `significance.json`, `outputs/RESULTS.md` STEP0 block.
- Sanity of numbers: RadGraph F1 / CheXbert F1 for zero-shot MedGemma should be well below fine-tuned SOTA
  (roughly 10-30 range x100); values near 0 or near 100 mean a bug (empty hyps, refs == hyps, wrong target).
**Accept:** metrics on >= 1,000 real test studies per dataset; leakage assert passed.

### Step 1 - Model A QLoRA (`notebooks/step1_kaggle.ipynb`)
Precondition: train-subset images attached; `step0_prepare_data.py --config configs/step1.yaml` shows ReXGradient
train rows (~22k) and the eval ids are unchanged.
Check:
- The trainer's SANITY CHECK line: 1 `<bos>`, 256 image tokens, supervised text starts with `FINDINGS:` and ends
  with `<end_of_turn>`, prompt not in the loss. `UNSLOTH_FORCE_FLOAT32=1` printed (Gemma 3 on T4).
- Loss falls and is finite; s/step and ETA printed; checkpoints every 100 steps; resume works (stop and rerun).
- If training would exceed Kaggle's weekly GPU quota, reduce `train.datasets` (CLAUDE.md: start with 20-30k).
- `step1_generate.py` uses the SAME eval ids/prompt/decoding; `generation_meta.json` locks the adapter sha256.
- `step1_score.py`: `acceptance.json` per dataset with RadGraph zero-shot -> Model A, paired p, n.
**Accept:** RadGraph F1 beats zero-shot MedGemma 1.5 with a significant paired test. If RadGraph F1 < ~24, note the
CLAUDE.md advice (lateral view + indication + more data before RL).

### Step 2 - retrieval ablation (`notebooks/step2_kaggle.ipynb`)
Check:
- Embeddings: train/val/test counts match the parquet; no `FloatingPointError` (else `--dtype fp32`);
  norms == 1; `test_ids.parquet` holds NO report text.
- CheXbert label prevalence (% positive per label) looks clinically plausible (e.g. No Finding / Support Devices high,
  Fracture low).
- Probe: val macro AUROC (expect clearly > 0.5; ~0.7-0.85 is plausible); labels with few val positives say so.
- Retrieval: index = train only (assert); `retrieval_summary_rerank.json`: agree_chosen >= agree_top_sim;
  read 3 examples - references must come from OTHER patients and be distinct.
- Model A + retrieval generation, `step2_score.py`: KEEP/DROP per dataset by the pre-registered rule.
**Accept:** ablation table produced; retrieval kept only if it does not hurt clinical metrics. If KEPT, set
`verify.source` (Step 3) to the retrieval system.

### Step 3 - verify-then-report (`notebooks/step3_kaggle.ipynb`)
Check FIRST: `step3_eval_boxes.py` on NIH boxes (MedGemma's documented localisation is for ANATOMY; finding boxes
are unknown). Report parse rate, "absent" rate, hit@0.3, pointing and side accuracy per label. If side accuracy
is near 50% or most answers are unparseable, tell the user box evidence is unreliable before trusting verdicts.
Then: claims (sentence counts), localize (status counts, unparseable rate, hit_max_tokens), verify, score.
- Read 5 `verify.jsonl` reports: do VERIFIED/FLAGGED reasons make sense? Laterality: image-left = patient-right.
- `verify_eval.json`: share of WRONG sentences caught (FLAGGED or UNCERTAIN) AND the false-alarm rate on correct
  sentences - report both, never one without the other.
**Accept:** both numbers + metrics after dropping FLAGGED sentences (paired vs before).

### Step 4 - Model B projector alignment (`notebooks/step4_kaggle.ipynb`)
Check: sanity line (supervised text starts with FINDINGS, ends with `<end_of_turn>`, finite initial loss, 256 image
tokens of dim 1152); only the projector trains; `torchrun --nproc_per_node 2` works (else run on 1 GPU);
s/step + ETA; `acceptance.json` (convergence rule + coherence: format_ok >= 0.9, repeated <= 0.1); READ
`val_samples.jsonl` - sentences must be coherent radiology language.
**Accept:** loss converges and val reports are coherent.

### Step 5 - Model B joint tuning (`notebooks/step5_kaggle.ipynb`)
Check: projector initialised from Step 4; LoRA trainable params printed; two lr groups; priors appear in some
prompts (train has fewer priors than test - note it); `step5_generate.py` row format; `step5_score.py`
`winner.json` (pre-registered rule) and paired tests.
**Accept:** full RadEval comparison vs Model A; winner recorded; the other model reported as an ablation.

### Step 6 - robustness (`notebooks/step6_kaggle.ipynb`)
Check: robust twin trained with `augment_p 0.4` and otherwise the same recipe; `step6_make_photo_test.py` op
counts; look at 5 photo-test images (they must look like phone photos of an X-ray, still readable); CheXphoto
pairs exclude every training patient (count printed). `drop.json`: photo - clean per metric for baseline and
robust; CheXphoto label micro-F1 clean vs photo.
**Accept:** the clean -> photo drop table, with vs without robustness training. Say clearly that the synthetic
photo test shares the training corruption family and CheXphoto is the real-photo check.

### Step 7 - Indian-language summaries (`notebooks/step7_kaggle.ipynb`)
Check: IndicTrans2 loads with this transformers version (if not, `--backend llm` and report that); read 5
summaries per language with the user if possible; `faithfulness_<backend>.json` agreement per language; flagged
examples saved; note that the label check ignores laterality (left/right swaps are NOT detected).
**Accept:** per-language label-agreement rate; flagged examples saved for review.

### Step 8 - second reader (`notebooks/step8_kaggle.ipynb`)
Check: thresholds come from VAL (`thresholds.json`); recall and precision on the deletion simulation; flags per
UNMODIFIED report (false alarms or genuine misses - indistinguishable); key findings (nodule, pneumothorax,
consolidation) recall. If false alarms are high, lower `target_far` (it compounds over 13 conditions).
**Accept:** precision and recall of flagged misses; false alarms low.

### Step 9 - GRPO (`notebooks/step9_kaggle.ipynb`)
Check per step: reward and each component logged, KL small and finite, truncation rate, completion length;
the first sample completion is readable; reward rises over training (log first vs last 10%). Watch for reward
hacking: very short reports, repeated "no X" sentences to farm the verify reward, format-only outputs - read 10
test outputs of the GRPO model. Note the deviation: a custom GRPO loop (TRL's objective, num_iterations=1) instead of
TRL GRPOTrainer, because TRL cannot train custom Model B; the verify reward uses the probe only (no boxes).
**Accept:** significant paired gain on CheXbert-14 micro-F1 or GREEN vs the SFT model, RadGraph not lower.

### Step 10 - app + offline (`notebooks/step10_kaggle.ipynb`)
Check in the app: upload a test image (from the private data) -> report with coloured sentences and hover reasons,
boxes drawn on the image, supporting cases from other patients, patient summary with the faithfulness flag,
second-reader tab, voice dictation (English MedASR; Indic via the configured ASR), PDF export opens and matches
the old ReportLab layout. `step10_offline_benchmark.py`: load time, median/p90 latency, peak memory, on a laptop
with an NVIDIA GPU for the 4-bit claim (on CPU the script reports the unquantised fallback - say so).
**Accept:** app works end to end; latency and memory reported. Optional: IN-CXR normal-vs-abnormal test.

## When something fails
- Read the full traceback (Kaggle's Python 3.13 display can hide it; the scripts print "the real problem" first).
- `ConnectionError` from `kaggle_secrets` = HF_TOKEN not ticked for this notebook.
- `401/403` from Hugging Face = the HF terms of that model/dataset were not accepted with this token.
- OOM: reduce batch size first; never switch Gemma 3 to fp16 on T4.
- Empty / NaN outputs: precision problem (fp16) - the empty-output guard aborts generation.
- Resume: rerun the same command with the same `--run-name` after restoring outputs; changed settings raise
  "settings differ" by design - use a new run name instead of deleting files.
- After any code fix: add or update a unit test, run `python -m pytest tests -q` (all must pass), and re-run the
  affected smoke test before the full run.

## Final report to the user
A table per step: status (PASS / NOT YET / not run), key numbers with n, the file each number came from, wall time,
and deviations from CLAUDE.md (MIMIC missing, IU split, CheXphoto unofficial, custom GRPO, probe-only verify reward,
NIH instead of MS-CXR, translation laterality not checked). Update `outputs/RESULTS.md` only via the scripts.
