import os
import subprocess
import sys

DEFAULT_ROOT = "/content/ulal"


# Drive
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


# Dataset
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


# One-call setup
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
