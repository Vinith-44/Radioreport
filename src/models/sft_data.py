"""Step 1 training data: Step 0 parquet rows -> chat examples for supervised fine-tuning.

The user turn is EXACTLY the zero-shot prompt (same template, same image-then-text order as
medgemma_zeroshot.generate_batch), so the fine-tuned model is evaluated with the prompt it was trained on.
The assistant turn is the cleaned reference report in the shared output format ("FINDINGS: ...\nIMPRESSION: ...").
Loss masking (prompt tokens -> -100) is done by Unsloth's UnslothVisionDataCollator(train_on_responses_only=True).
"""
from __future__ import annotations

import pandas as pd

from src.data.text_clean import format_report
from src.models.medgemma_zeroshot import build_prompt, load_image


def build_messages(row: dict, template: str, empty_indication: str, image=None) -> list[dict]:
    user = [{"type": "image"} if image is None else {"type": "image", "image": image},
            {"type": "text", "text": build_prompt(template, row.get("indication"), empty_indication)}]
    target = format_report(row.get("findings") or "", row.get("impression") or "")
    return [{"role": "user", "content": user},
            {"role": "assistant", "content": [{"type": "text", "text": target}]}]


def select_train_rows(df: pd.DataFrame, per_dataset: dict, seed: int, split: str = "train") -> pd.DataFrame:
    """Seeded subset per dataset from ONE split. per_dataset = {dataset: max studies, or None/'all'}.
    Datasets that are not listed are not used. Raises if a non-`split` row could slip in."""
    parts = []
    for ds, n in per_dataset.items():
        sub = df[(df["dataset"] == ds) & (df["split"] == split)].sort_values("study_id")
        if n not in (None, "all") and int(n) < len(sub):
            sub = sub.sample(n=int(n), random_state=seed).sort_values("study_id")
        parts.append(sub)
    out = pd.concat(parts, ignore_index=True) if parts else df.iloc[:0]
    if len(out) and set(out["split"]) != {split}:
        raise AssertionError(f"select_train_rows: rows from {set(out['split'])} (expected only {split})")
    # Fixed, seeded order that interleaves datasets (the trainer shuffles again each epoch with its own seed).
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


class SFTImageDataset:
    """Lazy map-style dataset: images are read from disk in __getitem__ (25k PIL images would not fit in RAM).

    `column_names = None` makes TRL's dataset helpers fall back to inspecting the first example."""

    column_names = None

    def __init__(self, rows: list[dict], template: str, empty_indication: str, augment_p: float = 0.0, seed: int = 0):
        self.rows, self.template, self.empty = rows, template, empty_indication
        self.augment_p, self.seed, self.calls = augment_p, seed, 0

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        if i >= len(self.rows) or i < -len(self.rows):
            raise IndexError(i)
        r = self.rows[i]
        im = load_image(r["image_path"])
        if self.augment_p > 0:   # Step 6: phone-photo / WhatsApp corruptions on augment_p of the examples
            import random

            from src.data.augment import photo_augment

            self.calls += 1
            rng = random.Random(hash((self.seed, i, self.calls)))
            if rng.random() < self.augment_p:
                im = photo_augment(im, rng)
        return {"messages": build_messages(r, self.template, self.empty, image=im)}

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
