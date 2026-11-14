"""Shared helpers: config, paths, run logging (config/seed/git/versions), JSONL I/O."""
from __future__ import annotations

import datetime as _dt
import importlib.metadata as _md
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

CODE_ROOT = Path(__file__).resolve().parents[1]  # .../project (where src/, scripts/ live)


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | os.PathLike) -> dict:
    """YAML config. `inherit: <other.yaml>` (relative to this file) loads that config first and overrides it,
    so later steps share Step 0's data paths, prompt and generation settings instead of copying them."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("inherit", None)
    if base:
        bp = Path(base) if Path(base).is_absolute() else path.parent / base
        cfg = _deep_merge(load_config(bp), cfg)
    return cfg


def project_root(cfg: dict) -> Path:
    """Where outputs go. Env PROJECT_ROOT overrides the config (used for local tests)."""
    root = Path(os.environ.get("PROJECT_ROOT") or cfg["paths"]["project_root"])
    root.mkdir(parents=True, exist_ok=True)
    return root


def data_dir(cfg: dict) -> Path:
    d = project_root(cfg) / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_dir(cfg: dict, step: str, run_name: str | None) -> Path:
    """outputs/runs/<step>/<run_name>. Reusing a run_name resumes that run."""
    name = run_name or _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    d = project_root(cfg) / "outputs" / "runs" / step / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def git_info() -> dict:
    def _git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], cwd=CODE_ROOT, capture_output=True, text=True, timeout=10, check=True
            ).stdout.strip()
        except Exception:
            return None

    return {
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
        "note": os.environ.get("RUN_NOTE"),  # free-text commit note when git is unavailable (e.g. Kaggle upload)
    }


_TRACKED_LIBS = [
    "torch", "transformers", "accelerate", "bitsandbytes", "peft", "trl", "unsloth", "unsloth_zoo",
    "radeval", "pandas", "pyarrow", "numpy", "pillow", "faiss-cpu", "faiss-gpu",
]


def lib_versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for lib in _TRACKED_LIBS:
        try:
            out[lib] = _md.version(lib)
        except _md.PackageNotFoundError:
            pass
    try:
        import torch

        out["cuda"] = torch.version.cuda
        out["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    return out


def write_run_info(out_dir: Path, cfg: dict, args: Any, extra: dict | None = None) -> Path:
    """Log config, seed, git hash, library versions and CLI args (CLAUDE.md §3). One file per invocation."""
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    info = {
        "timestamp": stamp,
        "argv": sys.argv,
        "args": vars(args) if hasattr(args, "__dict__") else args,
        "seed": cfg.get("seed"),
        "config": cfg,
        "git": git_info(),
        "versions": lib_versions(),
        **(extra or {}),
    }
    path = out_dir / f"run_info_{stamp}.json"
    path.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    return path


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # A truncated last line from a killed session: ignore it; that study is regenerated.
                    continue
    return rows


def append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def iter_batches(items: list, n: int) -> Iterator[list]:
    for i in range(0, len(items), n):
        yield items[i : i + n]
