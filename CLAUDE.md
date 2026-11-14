# CLAUDE.md — Chest X-ray Report Generation (verifiable, robust, multilingual)

## 1. Project goal
Build a research system that generates chest X-ray radiology reports (FINDINGS + IMPRESSION)
from a frontal CXR + clinical indication (+ optional prior study), and:
- verifies each generated sentence against image evidence (bounding box + classifier),
- stays accurate on smartphone photos / WhatsApp-compressed X-rays,
- produces a patient-friendly summary in Indian languages with a faithfulness check,
- can act as a "second reader" that flags findings missing from a radiologist's report.
This is a research / radiologist-assist tool. Never describe it as diagnostic.

## 2. Hard constraints
- Hardware: Kaggle, 2x Tesla T4 (16 GB each). T4 has NO bf16 -> use fp16 compute, 4-bit QLoRA.
- Use Unsloth for fine-tuning where it supports the model; otherwise HF transformers + peft + trl.
- Python 3.10+. Paths: inputs under /kaggle/input, all outputs under /kaggle/working/project/.
- Every script must support `--smoke` (runs on ~50 samples in < 5 min) and be resumable from checkpoints.
- Before writing code against any library (transformers, trl, peft, unsloth, RadEval, faiss),
  check the installed version's API (help(), docs, or source). APIs change; do not guess signatures.
- Gated models (MedGemma, MedSigLIP, MAIRA-2) need an HF token from Kaggle Secrets ("HF_TOKEN").

## 3. Non-negotiable rules (research integrity)
- NEVER hard-code, invent, or "demo" metric values. Every reported number must come from
  running the model on the real test split and saving (study_id, reference, hypothesis) to JSONL.
- Split by PATIENT (or use official splits). No patient may appear in two splits. Assert this in code.
- Retrieval indexes and training use the TRAIN split only. Test split is only for final scoring.
- Label-only datasets (CheXpert, NIH-14) are never used as report ground truth for evaluation.
- Log config, seed, git hash/commit note, and library versions with every run.

## 4. Repository layout
project/
  configs/            YAML config per step (paths, hyperparams, model ids)
  data/               cleaned parquet splits + manifests
  src/
    data/             loaders, cleaning, splitting, augmentation
    models/           model A wrapper, model B (encoder + projector + LLM), probe classifier
    retrieval/        MedSigLIP embedding + FAISS index + CheXbert re-rank
    verify/           sentence-level verification (text vs box vs classifier)
    multilingual/     Indic summary generation + back-translation check
    eval/             RadEval wrappers, significance tests, result tables
    rl/               GRPO reward functions + trainer
    app/              Gradio UI, PDF export, voice input
  scripts/            one entry script per step (step0_*.py ... step10_*.py)
  outputs/            runs/<step>/<timestamp>/ (checkpoints, logs, predictions, metrics.json)
  notebooks/          thin Kaggle notebooks that only call scripts

## 5. Shared definitions
- Models:
  - Model A = google/medgemma-1.5-4b-it + QLoRA (baseline, safety net).
  - Model B = frozen medical encoder (microsoft/rad-dino or google/medsiglip-448)
              + custom projector (2x2 patch-merge -> 2-layer MLP with GELU)
              + Gemma 3 text LLM (start with 1B, try 4B) with LoRA.
  - Probe = small linear/MLP head on frozen MedSigLIP image embeddings predicting 14 CheXpert labels.
- Prompt format (both models):
  input  = image(s) + "Indication: {indication}" (+ "Prior report: {prior}" when available)
  output = "FINDINGS: ...\nIMPRESSION: ..."
- Metrics (via the RadEval package): RadGraph F1, CheXbert F1 (5 and 14 labels), RaTEScore,
  GREEN, RadCliQ; BLEU-4 / ROUGE-L only as secondary. Use RadEval's paired significance test
  when comparing systems.

## 6. Steps (build in this order; each step must pass its acceptance check before the next)

### Step 0 — Data cleaning + honest baseline
Inputs: MIMIC-CXR (official split), ReXGradient-160K (official split), IU-Xray (R2Gen split).
Do:
- Extract INDICATION, FINDINGS, IMPRESSION (official MIMIC section parser or equivalent).
- Remove MIMIC "___" PHI placeholders; drop empty/non-informative reports ("As above.", "None.", "---").
- Keep frontal views (AP/PA) first; record lateral + prior-study links for later.
- Write train/val/test parquet with columns: study_id, patient_id, image_path(s), view,
  indication, findings, impression, prior_study_id, dataset.
- Assert zero patient overlap across splits.
- Run zero-shot MedGemma 1 4B-it and MedGemma 1.5 4B-it on the test splits; score with RadEval.
Outputs: data/*.parquet, outputs/runs/step0/.../predictions.jsonl, metrics.json, a markdown results table.
Accept: metrics computed on >= 1,000 real test studies per dataset; leakage assert passes.

### Step 1 — Model A: QLoRA fine-tune of MedGemma 1.5 4B
Do: Unsloth 4-bit, LoRA r=16 (try 32), alpha=16, dropout 0.05, target all linear layers,
lr 2e-4, effective batch 16 (per-device 1-2 + grad accum), 1-3 epochs, gradient checkpointing,
vision encoder frozen, fp16. Mask prompt tokens in labels. Start with 20-30k studies.
Outputs: LoRA adapter, test predictions, metrics vs Step 0.
Accept: RadGraph F1 beats zero-shot MedGemma 1.5 with a significant paired test.
If RadGraph F1 < ~24: add lateral view + indication + more data before moving to RL.

### Step 2 — Fact-aware case retrieval
Do: embed all TRAIN images with google/medsiglip-448 (fp16), FAISS inner-product index.
At inference: top-20 candidates -> re-rank by CheXbert label agreement with the Probe's predictions
-> top-3 reports inserted into the prompt as "Reports from OTHER patients (style reference only)".
Outputs: index files, retrieval module, ablation (with vs without retrieval) on Model A.
Accept: ablation table produced; keep retrieval only if it does not hurt clinical metrics.

### Step 3 — Verify-then-report module (main novelty)
Do:
- Train the Probe (14 CheXpert labels) on frozen MedSigLIP embeddings (train split).
- Get per-finding bounding boxes (MedGemma 1.5 localization prompts; MS-CXR for evaluation).
- Split the generated report into sentences; extract the finding + laterality per sentence (CheXbert/RadGraph).
- Verdict per sentence: VERIFIED (text, box side/region, and probe all agree),
  UNCERTAIN (partial agreement), FLAGGED (contradiction or no image support).
- Option to drop FLAGGED sentences.
Outputs: verify/ module, JSON output per report with sentence verdicts + boxes + supporting case ids.
Accept: report the share of clinically wrong sentences (vs reference) that land in FLAGGED/UNCERTAIN,
and clinical metrics after dropping FLAGGED sentences.

### Step 4 — Model B, phase 1: projector alignment
Do: build Model B. Freeze encoder + LLM; train only the projector.
Data: synthetic template reports generated from CheXpert/NIH labels (rule-based templates, no external API),
plus MIMIC train reports.
Accept: training loss converges; the model produces coherent finding sentences on val images.

### Step 5 — Model B, phase 2: joint tuning
Do: freeze encoder; train projector (higher lr, e.g. 1e-4 to 2e-4) + LoRA on the LLM (lower lr, e.g. 2e-5 to 5e-5)
on real MIMIC + ReXGradient reports. Introduce indication and prior report inputs.
Accept: full RadEval comparison vs Model A. Whichever model is better carries the features
in Steps 6-10; the other is reported as an ablation.

### Step 6 — Robustness training (phone photos / WhatsApp)
Do: augmentation module: perspective/angle warp, screen glare, monitor moire, printed-film texture,
JPEG quality 10-40, low exposure/contrast, rotation, blur. Train with 60% clean / 40% augmented.
Evaluate on clean test + CheXphoto (photos of CXRs).
Accept: table of the clean -> photo score drop, with vs without robustness training.

### Step 7 — Indian-language patient summary
Do: generate a simple-language summary from the final English report in Hindi, Tamil, Telugu
(then Bengali, Marathi, Kannada, Malayalam, Gujarati). Use the LLM itself or ai4bharat/IndicTrans2.
Faithfulness check: back-translate to English -> CheXbert labels must match the English report's labels;
mismatched summaries are flagged.
Accept: per-language label-agreement rate; flagged examples saved for review.

### Step 8 — Second-reader mode
Do: input = image + a radiologist's report. Compare report findings (CheXbert/RadGraph) with Probe + box
evidence and flag possible misses (e.g. nodule, pneumothorax, consolidation). Simulate by deleting one
positive finding from real reports and measuring detection.
Accept: precision and recall of flagged misses on the simulated set; keep false alarms low.

### Step 9 — GRPO with multiple rewards (stretch)
Do: TRL GRPOTrainer on the best model's LoRA. Reward = weighted sum of
(a) clinical accuracy vs reference (RadGraph partial F1 + CheXbert F1),
(b) verification consistency from Step 3 (+ per verified sentence, - per flagged sentence),
(c) format reward (both FINDINGS and IMPRESSION present, no repeated sentences),
(d) small length penalty.
T4 settings: num_generations=4, max_completion_length ~256, a few thousand prompts.
Accept: significant gain on CheXbert F1 or GREEN vs the SFT model without lower RadGraph F1.

### Step 10 — App + deployment
Do: Gradio app: upload image (+ indication, optional prior) -> report with colour-coded sentence verdicts,
boxes drawn on the image, supporting cases, Indian-language summary, second-reader tab,
voice editing (google/medasr for English; pluggable Indic ASR), PDF export (reuse existing ReportLab template).
Offline build: 4-bit quantized model running on a laptop; report latency and memory.
Optional external tests: IN-CXR (normal vs abnormal from the generated report).

## 7. Final results table (must be produced from saved predictions)
Rows: zero-shot MedGemma 1, zero-shot MedGemma 1.5, MAIRA-2 (if it fits), Model A, Model B,
best model + retrieval, + verification, + robustness, + GRPO.
Columns: RadGraph F1, CheXbert F1 (5/14), RaTEScore, GREEN, RadCliQ, BLEU-4, ROUGE-L, n studies.

## 8. How to work
- Before coding a step: restate the step, list files you will create, and point out risks (memory, API changes).
- Write small, tested functions; add a pytest or assert-based sanity check for data, shapes and leakage.
- Run --smoke first, show the output, then the full run.
- After each step: update outputs/RESULTS.md with real numbers and a 3-line summary of what changed.
- If something does not fit on a T4, propose the smallest change (resolution, batch, sequence length,
  model size) before suggesting other hardware.
