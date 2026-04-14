#!/bin/bash
# ============================================================
# setup_colab.sh
#
# WHAT THIS SCRIPT DOES:
#   1. Install kagglehub
#   2. Download the UCF101 dataset
#   3. Detect and PRINT the real dataset path
#
# HOW TO USE IN COLAB:
#   Cell 1:
#     from google.colab import drive
#     drive.mount('/content/drive')
#     import os
#     os.environ["KAGGLE_USERNAME"] = "your_kaggle_username"
#     os.environ["KAGGLE_KEY"]      = "your_kaggle_key"
#
#   Cell 2:
#     !bash /content/drive/MyDrive/apai/setup_colab.sh
# ============================================================

set -e

KAGGLE_VERSION="4"
CACHE_PATH="/root/.cache/kagglehub/datasets/matthewjansen/ucf101-action-recognition/versions/${KAGGLE_VERSION}"
KAGGLE_INPUT_PATH="/kaggle/input/ucf101-action-recognition"

echo ""
echo "========================================"
echo " UCF101 — Dataset Download Script"
echo "========================================"

# ──────────────────────────────────────────
# STEP 0 — Check Kaggle credentials
# ──────────────────────────────────────────
echo ""
echo "[0/3] Checking Kaggle credentials..."

if [ -z "$KAGGLE_USERNAME" ] || [ -z "$KAGGLE_KEY" ]; then
    echo "  ✗ ERROR: KAGGLE_USERNAME or KAGGLE_KEY not set."
    echo ""
    echo "  Run this in Colab before the script:"
    echo "    import os"
    echo "    os.environ['KAGGLE_USERNAME'] = 'your_kaggle_username'"
    echo "    os.environ['KAGGLE_KEY']      = 'your_kaggle_key'"
    exit 1
fi

echo "  ✓ Kaggle credentials found."

# ──────────────────────────────────────────
# STEP 1 — Install kagglehub
# ──────────────────────────────────────────
echo ""
echo "[1/3] Installing kagglehub..."
pip install -q kagglehub
echo "  ✓ kagglehub ready."

# ──────────────────────────────────────────
# STEP 2 — Download dataset
# ──────────────────────────────────────────
echo ""
echo "[2/3] Downloading UCF101 dataset..."
echo "  (skipped automatically if already cached)"

python3 - "$KAGGLE_USERNAME" "$KAGGLE_KEY" <<'PYEOF'
import sys, os, kagglehub

os.environ["KAGGLE_USERNAME"] = sys.argv[1]
os.environ["KAGGLE_KEY"]      = sys.argv[2]

path = kagglehub.dataset_download("matthewjansen/ucf101-action-recognition")
print(f"  kagglehub returned path: {path}")
PYEOF

# ──────────────────────────────────────────
# STEP 3 — Detect and print dataset path
# ──────────────────────────────────────────
echo ""
echo "[3/3] Detecting dataset path..."

if [ -d "$CACHE_PATH" ]; then
    DETECTED_PATH="${CACHE_PATH%/}/"
    echo ""
    echo "  ✓ Dataset found in kagglehub cache:"
    echo "    $DETECTED_PATH"
    echo ""
    echo "  ➜ Use this in config.py:"
    echo "    DATASET_ROOT = \"$DETECTED_PATH\""
    echo "    OUTPUT_ROOT  = \"/content/ucf101-processed/\""

elif [ -d "$KAGGLE_INPUT_PATH" ]; then
    DETECTED_PATH="${KAGGLE_INPUT_PATH%/}/"
    echo ""
    echo "  ✓ Dataset found in Kaggle input:"
    echo "    $DETECTED_PATH"
    echo ""
    echo "  ➜ Use this in config.py:"
    echo "    DATASET_ROOT = \"$DETECTED_PATH\""
    echo "    OUTPUT_ROOT  = \"/kaggle/working/ucf101-processed/\""

else
    echo ""
    echo "  ✗ Dataset not found at expected paths:"
    echo "    $CACHE_PATH"
    echo "    $KAGGLE_INPUT_PATH"
    echo ""
    echo "  Full cache contents for debug:"
    find /root/.cache/kagglehub/datasets/matthewjansen/ -maxdepth 4 2>/dev/null || echo "    (cache empty or not found)"
    exit 1
fi

echo ""
echo "========================================"
echo " Done! Copy the DATASET_ROOT above"
echo " into your config.py and run the"
echo " full setup script next."
echo "========================================"
