"""Kaggle job (CPU, internet ON): fetch the official ReXGradient-160K TEST images into /kaggle/working.

Needs the Kaggle Secret HF_TOKEN attached to this notebook (Add-ons -> Secrets). The output
(/kaggle/working/rexgradient-160k-test-images, ~10 GB) is then attached to the Step 0 notebook as an input.
License: non-commercial, no redistribution -> keep this notebook and its output PRIVATE.
"""
import os
import subprocess
import sys

REPO = "https://github.com/Vinith-44/Radioreport"
BRANCH = "main"

try:
    from kaggle_secrets import UserSecretsClient

    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
except Exception as e:
    sys.exit(f"HF_TOKEN secret is not attached to this notebook ({e!r}). Add-ons -> Secrets -> HF_TOKEN, then rerun.")

subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", BRANCH, REPO, "/tmp/repo"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "zstandard", "huggingface_hub>=0.30"], check=True)
os.chdir("/tmp/repo")
subprocess.run([sys.executable, "scripts/step0_fetch_rexgradient.py", "--splits", "test",
                "--out-dir", "/kaggle/working/rexgradient-160k-test-images"], check=True)
subprocess.run(["du", "-sh", "/kaggle/working/rexgradient-160k-test-images"])
