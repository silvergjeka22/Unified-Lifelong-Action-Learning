"""
ULAL bootstrap — repo checkout, dataset download, path setup.

Replaces bash/fetch_src.sh and bash/setup_colab.sh.

What was wrong with the old scripts:

  fetch_src.sh
    - had to already exist on Drive before you could run it (chicken-and-egg)
    - downloaded every file individually through the GitHub API (hundreds of
      requests, slow, rate-limitable)
    - `BRANCH="GAN&ActiveLearning"` unquoted in shell -> the `&` backgrounds
      the command

  setup_colab.sh
    - rewrote src/config/config.py in place with sed, so the working tree
      differed from git and a re-clone silently reverted your paths
    - forced a kernel restart in the middle of a notebook run

Now: one `git clone --depth 1` (a single request), and paths travel through
environment variables so config.py is never modified.

Usage in a notebook — one self-contained cell, no Drive dependency:

    import os, sys, subprocess
    from getpass import getpass
    BRANCH, REPO, ROOT = "GAN&ActiveLearning", "silvergjeka22/Unified-Lifelong-Action-Learning", "/content/ulal"
    tok = os.environ.get("GITHUB_TOKEN") or getpass("GitHub token: ")
    if os.path.isdir(ROOT + "/.git"):
        subprocess.run(["git","-C",ROOT,"fetch","--depth","1","origin",BRANCH], check=True)
        subprocess.run(["git","-C",ROOT,"reset","--hard","FETCH_HEAD"], check=True)
    else:
        subprocess.run(["git","clone","--depth","1","--branch",BRANCH,
                        f"https://{tok}@github.com/{REPO}.git", ROOT], check=True)
    sys.path.insert(0, ROOT)

Then:

    from src.bootstrap import setup
    setup(drive=True, dataset=True)
    %run /content/ulal/src/imports.py
"""

import os
import subprocess
import sys

DEFAULT_ROOT = "/content/ulal"


# ── Repo checkout ─────────────────────────────────────────────────────────────
def clone_or_pull(repo: str, branch: str, root: str = DEFAULT_ROOT, token: str = None):
    """
    Shallow-clone the repo, or fast-forward it if already present.

    subprocess is called with a LIST, never a shell string — that is what makes
    a branch name containing '&' safe. The old bash script interpolated the
    branch into a shell command, where 'GAN&ActiveLearning' splits into
    'GAN' backgrounded plus a bogus 'ActiveLearning' command.
    """
    token = token or os.environ.get("GITHUB_TOKEN", "")
    if not token:
        raise RuntimeError(
            "No GitHub token. The repo is private.\n"
            "  os.environ['GITHUB_TOKEN'] = '...'   or use getpass in the bootstrap cell."
        )

    if os.path.isdir(os.path.join(root, ".git")):
        subprocess.run(["git", "-C", root, "fetch", "--depth", "1", "origin", branch], check=True)
        subprocess.run(["git", "-C", root, "reset", "--hard", "FETCH_HEAD"], check=True)
        action = "updated"
    else:
        url = f"https://{token}@github.com/{repo}.git"
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", branch, url, root], check=True
        )
        action = "cloned"

    if root not in sys.path:
        sys.path.insert(0, root)

    sha = subprocess.run(
        ["git", "-C", root, "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()
    print(f"Repo {action}: {repo}@{branch} ({sha}) -> {root}")
    return root


# ── Drive ─────────────────────────────────────────────────────────────────────
def mount_drive():
    try:
        from google.colab import drive
    except ImportError:
        print("Not on Colab — skipping Drive mount.")
        return False
    if not os.path.isdir("/content/drive/MyDrive"):
        drive.mount("/content/drive")
    print("Drive mounted.")
    return True


# ── Dataset ───────────────────────────────────────────────────────────────────
def setup_dataset(kaggle_version: str = "4"):
    """
    Download UCF101 via kagglehub and export the paths as env vars.

    Sets ULAL_DATASET_ROOT / ULAL_OUTPUT_ROOT, which config.py reads. No file is
    modified and no kernel restart is needed.
    """
    if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        raise RuntimeError("Set KAGGLE_USERNAME and KAGGLE_KEY before calling setup_dataset().")

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kagglehub"], check=True)
    import kagglehub

    path = kagglehub.dataset_download("matthewjansen/ucf101-action-recognition")

    cache = f"/root/.cache/kagglehub/datasets/matthewjansen/ucf101-action-recognition/versions/{kaggle_version}"
    if os.path.isdir(cache):
        dataset_root, output_root = cache.rstrip("/") + "/", "/content/ucf101-processed/"
    elif os.path.isdir("/kaggle/input/ucf101-action-recognition"):
        dataset_root, output_root = "/kaggle/input/ucf101-action-recognition/", "/kaggle/working/ucf101-processed/"
    else:
        dataset_root, output_root = path.rstrip("/") + "/", "/content/ucf101-processed/"

    os.environ["ULAL_DATASET_ROOT"] = dataset_root
    os.environ["ULAL_OUTPUT_ROOT"]  = output_root
    print(f"ULAL_DATASET_ROOT = {dataset_root}")
    print(f"ULAL_OUTPUT_ROOT  = {output_root}")
    return dataset_root, output_root


# ── One-call setup ────────────────────────────────────────────────────────────
def setup(drive: bool = True, dataset: bool = True, data_root: str = "/content/UCF101",
          drive_project: str = "/content/drive/MyDrive/apai", kaggle_version: str = "4"):
    """Mount Drive, download the dataset, export every path config.py reads."""
    os.environ.setdefault("ULAL_DATA_ROOT", data_root)
    os.environ.setdefault("ULAL_DRIVE_PROJECT", drive_project)

    if drive:
        mount_drive()
    if dataset:
        setup_dataset(kaggle_version)

    for d in ("checkpoints", "results", "cache"):
        os.makedirs(f"{drive_project}/{d}", exist_ok=True)
    os.makedirs(data_root, exist_ok=True)

    print(f"ULAL_DATA_ROOT     = {data_root}")
    print(f"ULAL_DRIVE_PROJECT = {drive_project}")
    print("Setup complete — no config.py edit, no kernel restart.")
