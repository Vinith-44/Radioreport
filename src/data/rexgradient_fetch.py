"""Stream the ReXGradient-160K image archive from Hugging Face and keep only the studies we need.

Verified on the real repo (rajpurkarlab/ReXGradient-160K, 2026-11-14):
  - images = deid_png.part00..part09 (10 x 15.5 GB). Despite the README ("cat ... > deid_png.tar"), the
    concatenation is a ZSTD-compressed tar (magic 28b52ffd).
  - members are "deid_png/<PatientID>/<Accession>/studies/<StudyUID>/series/<SeriesUID>/instances/<UID>.png",
    i.e. metadata ImagePath without the leading "../"; ordered by PatientID, so any split is spread over all
    parts and the whole archive has to be streamed once (nothing is stored except the wanted PNGs).
License: non-commercial research, NO redistribution -> any copy you keep on Kaggle must stay PRIVATE.
"""
from __future__ import annotations

import io
import json
import os
import tarfile
import time
from pathlib import Path
from typing import Callable

PARTS = [f"deid_png.part{i:02d}" for i in range(10)]


class ConcatParts(io.RawIOBase):
    """Read the parts back to back as one stream; on a network error, reopen at the same offset and retry."""

    def __init__(self, open_part: Callable[[str], io.IOBase], parts: list[str], retries: int = 8,
                 byte_limit: int | None = None):
        self.open_part, self.parts, self.retries, self.limit = open_part, parts, retries, byte_limit
        self.i, self.offset, self.total, self.f = 0, 0, 0, None

    def readable(self) -> bool:
        return True

    def _open(self):
        self.f = self.open_part(self.parts[self.i])
        if self.offset:
            self.f.seek(self.offset)

    def readinto(self, b) -> int:
        if self.limit is not None and self.total >= self.limit:
            return 0
        want = len(b) if self.limit is None else min(len(b), self.limit - self.total)
        for attempt in range(self.retries + 1):
            try:
                while self.i < len(self.parts):
                    if self.f is None:
                        self._open()
                    data = self.f.read(want)
                    if data:
                        n = len(data)
                        b[:n] = data
                        self.offset += n
                        self.total += n
                        return n
                    self.f.close()
                    self.f, self.i, self.offset = None, self.i + 1, 0
                return 0
            except (OSError, IOError, ConnectionError, TimeoutError) as e:
                if attempt == self.retries:
                    raise
                wait = min(60, 2 ** attempt)
                print(f"  network error in {self.parts[self.i]} at {self.offset}: {e!r}; retry in {wait}s", flush=True)
                try:
                    if self.f is not None:
                        self.f.close()
                except Exception:
                    pass
                self.f = None
                time.sleep(wait)
        return 0


def zstd_reader(raw: io.RawIOBase):
    """Streaming zstd decoder that reads across frames (Python 3.14 stdlib, else the `zstandard` package)."""
    buffered = io.BufferedReader(raw, buffer_size=16 * 2**20)
    try:
        from compression import zstd  # Python >= 3.14

        return zstd.ZstdFile(buffered)
    except ImportError:
        import zstandard  # pip install zstandard

        return zstandard.ZstdDecompressor().stream_reader(buffered, read_across_frames=True)


def member_rel(name: str) -> str:
    """'deid_png/GRDN.../x.png' or './deid_png/...' -> 'GRDN.../x.png'."""
    n = name.lstrip("./")
    return n.split("deid_png/", 1)[1] if "deid_png/" in n else n


def wanted_from_metadata(meta: dict) -> set[str]:
    """Relative image paths (under deid_png/) for every image of the given studies."""
    out = set()
    for v in meta.values():
        for p in v.get("ImagePath") or []:
            out.add(member_rel(p.replace("\\", "/").removeprefix("../")))
    return out


def first_frontal(paths: list, views: list, frontal_views: set[str]) -> str | None:
    """The image the Step 0 loader uses as `image_path`: frontal views only, PA before AP, then by path."""
    from src.data.loaders import _norm_view

    cands = [(0 if _norm_view(v) == "PA" else 1, str(p)) for p, v in zip(paths or [], views or [])
             if _norm_view(v) in frontal_views]
    return min(cands)[1] if cands else None


def select_subset(meta: dict, n: int | None, seed: int, frontal_views: list[str], min_alpha_chars: int,
                  require_findings: bool = True) -> dict[str, str]:
    """Seeded random subset of studies that the Step 0 filters would KEEP (frontal view + informative findings),
    returned as {study_key: relative path of its first frontal image}. Used for the train split, where fetching
    every image (~140 GB) does not fit on Kaggle. n=None -> all eligible studies."""
    import random

    from src.data.text_clean import is_non_informative, normalize

    fv = {v.upper() for v in frontal_views}
    elig = {}
    for key, v in meta.items():
        if require_findings and is_non_informative(normalize(v.get("Findings")), min_alpha_chars):
            continue
        f = first_frontal(v.get("ImagePath") or [], v.get("ImageViewPosition") or [], fv)
        if f is not None:
            elig[key] = member_rel(f.replace("\\", "/").removeprefix("../"))
    keys = sorted(elig)
    if n is not None and n < len(keys):
        keys = sorted(random.Random(seed).sample(keys, n))
    return {k: elig[k] for k in keys}


def shrink_png(data: bytes, size: int) -> bytes:
    """Any X-ray PNG -> 8-bit grayscale PNG resized to size x size with bilinear filtering.

    This is what MedGemma's image processor does to every input anyway (Gemma3ImageProcessor: resize to
    896x896, BILINEAR, no pan-and-scan, transformers 5.23 source), after the same 16-bit -> 8-bit rescale used
    at inference (to_rgb8). Storing the result cuts a train image from ~1.5 MB to ~0.4 MB."""
    from PIL import Image

    from src.models.medgemma_zeroshot import to_rgb8

    with Image.open(io.BytesIO(data)) as im:
        im.load()
        g = to_rgb8(im).convert("L").resize((size, size), Image.BILINEAR)
    buf = io.BytesIO()
    g.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def extract_wanted(fileobj, wanted: set[str], out_dir: Path, progress_every: int = 20000,
                   report: Callable[[dict], None] | None = None,
                   transform: Callable[[bytes], bytes] | None = None) -> dict:
    """Stream a tar (already decompressed) and write only wanted members to out_dir/<rel>. Skips files that
    already exist with the right size (so a re-run after a crash only re-downloads, never re-writes).
    With `transform` (e.g. shrink_png) the stored bytes differ from the archive, so any existing file counts."""
    stats = {"members_seen": 0, "written": 0, "already_present": 0, "bytes_written": 0}
    found: set[str] = set()
    t0 = time.time()
    tf = tarfile.open(fileobj=fileobj, mode="r|")
    for m in tf:
        stats["members_seen"] += 1
        if m.isfile():
            rel = member_rel(m.name)
            if rel in wanted:
                found.add(rel)
                dst = out_dir / rel
                if dst.exists() and (transform is not None or dst.stat().st_size == m.size):
                    stats["already_present"] += 1
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dst.with_suffix(dst.suffix + ".part")
                    src = tf.extractfile(m)
                    n_out = 0
                    with open(tmp, "wb") as f:
                        if transform is not None:
                            n_out = f.write(transform(src.read()))
                        else:
                            while True:
                                chunk = src.read(4 * 2**20)
                                if not chunk:
                                    break
                                n_out += f.write(chunk)
                    os.replace(tmp, dst)
                    stats["written"] += 1
                    stats["bytes_written"] += n_out
        if report and stats["members_seen"] % progress_every == 0:
            report({**stats, "found": len(found), "wanted": len(wanted), "elapsed_s": round(time.time() - t0)})
        if len(found) == len(wanted):
            break  # everything we need is on disk; stop streaming early
    stats["found"] = len(found)
    stats["missing"] = sorted(wanted - found)
    return stats


def hf_open_part_factory(repo: str, token: str | None):
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem(token=token)
    base = f"datasets/{repo}/"
    return lambda part: fs.open(base + part, "rb", block_size=32 * 2**20)


def download_metadata(repo: str, token: str | None, out_dir: Path, files: list[str]) -> list[Path]:
    from huggingface_hub import hf_hub_download

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in files:
        p = Path(hf_hub_download(repo, f"metadata/{f}", repo_type="dataset", token=token,
                                 local_dir=str(out_dir.parent)))
        paths.append(p)
    return paths


def load_meta(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)  # files contain bare NaN tokens: fine for Python's json
