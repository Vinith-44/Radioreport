"""Kaggle job (CPU, internet ON): fetch the Step 1 ReXGradient-160K TRAIN/VAL subset into /kaggle/working.

Seeded subset (22,000 train + 1,000 val studies that pass the Step 0 filters), first frontal image only, stored as
8-bit 896x896 PNGs (what MedGemma's processor makes of every image anyway) -> ~10-13 GB.

Needs the Kaggle Secret HF_TOKEN attached to this notebook (Add-ons -> Secrets, checkbox ticked). The output
(/kaggle/working/rexgradient-160k-train-subset) becomes the private dataset rexgradient-160k-train-subset.
License: non-commercial, no redistribution -> keep this notebook and its output PRIVATE.

All child-process output is streamed into this cell: in the Kaggle editor a subprocess's own stdout does not
show up in the notebook, and Kaggle's traceback display (Python 3.13) can crash and hide the real error.
"""
import os
import subprocess
import sys
import traceback

REPO = "https://github.com/Vinith-44/Radioreport"
BRANCH = "main"
OUT = "/kaggle/working/rexgradient-160k-train-subset"


def step(title, cmd, cwd=None):
    print(f"\n===== {title}\n$ {' '.join(cmd)}", flush=True)
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="", flush=True)
    if p.wait() != 0:
        raise RuntimeError(f"step '{title}' failed with exit code {p.returncode} (see the lines above)")


def main():
    print("Python", sys.version.split()[0], flush=True)
    try:
        from kaggle_secrets import UserSecretsClient

        tok = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as e:
        raise RuntimeError(f"could not read the HF_TOKEN secret ({e!r}). In the editor: Add-ons -> Secrets -> "
                           f"add HF_TOKEN and TICK its checkbox for this notebook.") from None
    if not tok or not tok.startswith("hf_"):
        raise RuntimeError("HF_TOKEN secret is empty or does not look like a Hugging Face token (hf_...).")
    os.environ["HF_TOKEN"] = tok
    print("HF_TOKEN secret: found", flush=True)

    if os.path.isdir("/tmp/repo"):
        step("update code", ["git", "-C", "/tmp/repo", "pull", "-q", "--ff-only"])
    else:
        step("download code", ["git", "clone", "-q", "--depth", "1", "-b", BRANCH, REPO, "/tmp/repo"])
    step("install packages", [sys.executable, "-m", "pip", "install", "-q", "zstandard", "huggingface_hub>=0.30", "pyyaml", "pillow"])
    step("fetch ReXGradient train/val subset", [sys.executable, "-u", "scripts/step0_fetch_rexgradient.py",
                                                "--splits", "train", "val", "--train-studies", "22000",
                                                "--val-studies", "1000", "--resize", "896", "--out-dir", OUT],
         cwd="/tmp/repo")
    step("size on disk", ["du", "-sh", OUT])
    print("\nDONE. Output folder:", OUT, flush=True)


try:
    main()
except Exception:
    # Print the real error ourselves, before Kaggle's (sometimes broken) traceback display runs.
    print("\n" + "=" * 70 + "\nERROR - this is the real problem:\n", flush=True)
    traceback.print_exc(file=sys.stdout)
    sys.stdout.flush()
    raise SystemExit(1)
