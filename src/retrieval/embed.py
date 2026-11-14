"""MedSigLIP image embeddings (google/medsiglip-448).

Verified against the model card and the transformers 5.17 source:
  - AutoModel / AutoProcessor; image resolution 448x448.
  - The card's reference preprocessing resizes with bilinear interpolation and NO antialiasing (Big Vision);
    PIL's resize antialiases, so the 448 resize is done here with torch (bilinear, align_corners=False,
    antialias=False == TF2 tf.image.resize bilinear). The processor then only rescales/normalises (448 -> 448).
  - transformers 5.x: SiglipModel.get_image_features returns the vision tower output object (not a tensor);
    the image embedding is its pooler_output (what SiglipModel.forward uses), L2-normalised here.

Every image (train-subset PNG already stored at 896x896, or a full-size test PNG) goes through the SAME path:
to_rgb8 -> grayscale -> 896x896 PIL bilinear (identity for the stored train images; the exact shrink_png step)
-> 448x448 torch bilinear, no antialias. Index images and query images are therefore processed identically.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from src.models.medgemma_zeroshot import to_rgb8

CANON = 896      # = shrink_png size used for the stored train-subset images
SIZE = 448       # MedSigLIP input resolution
PREPROC_VERSION = "rgb8-L-896bilinearPIL-448bilinearTorchNoAA-v1"


def canonical_image(path: str) -> Image.Image:
    with Image.open(path) as im:
        im.load()
        g = to_rgb8(im).convert("L")
    if g.size != (CANON, CANON):
        g = g.resize((CANON, CANON), Image.BILINEAR)
    return g


def resize_448(g: Image.Image) -> Image.Image:
    import torch
    import torch.nn.functional as F

    t = torch.from_numpy(np.asarray(g, dtype=np.float32))[None, None]
    r = F.interpolate(t, size=(SIZE, SIZE), mode="bilinear", align_corners=False, antialias=False)
    a = r[0, 0].clamp(0, 255).round().to(torch.uint8).numpy()
    return Image.fromarray(a).convert("RGB")


def load_for_siglip(path: str) -> Image.Image:
    return resize_448(canonical_image(path))


class SigLIPEmbedder:
    def __init__(self, model_id: str, dtype: str = "fp16", device: str = "cuda", token: str | None = None):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.dtype = {"fp16": torch.float16, "fp32": torch.float32}[dtype]
        self.device = device
        self.model = AutoModel.from_pretrained(model_id, token=token, dtype=self.dtype).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(model_id, token=token)

    def __call__(self, images: list[Image.Image]) -> np.ndarray:
        torch = self.torch
        px = self.processor(images=images, return_tensors="pt")["pixel_values"].to(self.device, self.dtype)
        with torch.inference_mode():
            out = self.model.get_image_features(pixel_values=px)
            emb = out if torch.is_tensor(out) else out.pooler_output
            emb = emb.float()
        if not torch.isfinite(emb).all():
            raise FloatingPointError("non-finite MedSigLIP embeddings (fp16 overflow?) - rerun with --dtype fp32")
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return emb.cpu().numpy().astype(np.float32)


class _ImageDS:
    def __init__(self, paths: list[str]):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return load_for_siglip(self.paths[i])


def _collate(b):
    return b


def embed_rows(paths: list[str], embed_fn, out_dir, split: str, chunk: int = 2048, batch_size: int = 32,
               num_workers: int = 0, log=print) -> np.ndarray:
    """Embed images in chunks saved as <split>_part<k>.npy (resumable: finished chunks are skipped), then
    return the concatenation in input order."""
    from pathlib import Path

    from torch.utils.data import DataLoader

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for k, s in enumerate(range(0, len(paths), chunk)):
        p = out_dir / f"{split}_part{k:04d}.npy"
        sub = paths[s:s + chunk]
        if p.exists() and np.load(p, mmap_mode="r").shape[0] == len(sub):
            parts.append(np.load(p))
            continue
        dl = DataLoader(_ImageDS(sub), batch_size=batch_size, num_workers=num_workers, collate_fn=_collate)
        embs = [embed_fn(imgs) for imgs in dl]
        e = np.concatenate(embs).astype(np.float32)
        np.save(p.with_suffix(".tmp.npy"), e)
        Path(p.with_suffix(".tmp.npy")).replace(p)
        parts.append(e)
        log(f"  [{split}] {min(s + chunk, len(paths))}/{len(paths)} images embedded")
    return np.concatenate(parts) if parts else np.zeros((0, 0), np.float32)
