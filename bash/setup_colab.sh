set -e

# CONFIG only edit this block
# full path
DRIVE_PROJECT_PATH="/content/drive/MyDrive/apai"

# Kaggle dataset version
KAGGLE_VERSION="4"

# DO NOT EDIT BELOW THIS LINE
CACHE_PATH="/root/.cache/kagglehub/datasets/matthewjansen/ucf101-action-recognition/versions/${KAGGLE_VERSION}"
KAGGLE_INPUT_PATH="/kaggle/input/ucf101-action-recognition"
CONFIG_FILE="${DRIVE_PROJECT_PATH}/src/config/config.py"

echo ""
echo "UCF101 Project — Colab Setup Script"

# Validate environment
echo ""
echo "Checking environment..."

ERRORS=0
[ -z "$KAGGLE_USERNAME" ] && echo "KAGGLE_USERNAME not set" && ERRORS=1
[ -z "$KAGGLE_KEY"      ] && echo "KAGGLE_KEY not set"      && ERRORS=1

if [ $ERRORS -ne 0 ]; then
    echo ""
    echo "Set these in Colab Cell 1 before running:"
    echo "    import os"
    echo "    os.environ['KAGGLE_USERNAME'] = 'your_username'"
    echo "    os.environ['KAGGLE_KEY']      = 'your_key'"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "config.py not found at: $CONFIG_FILE"
    echo ""
    echo "Run this in Colab to find it:"
    echo "    !find /content/drive/MyDrive -name 'config.py'"
    echo "Then update DRIVE_PROJECT_PATH in this script."
    exit 1
fi

echo "KAGGLE_USERNAME set"
echo "KAGGLE_KEY set"
echo "config.py found"

# Install kagglehub
echo ""
echo "Installing kagglehub..."
pip install -q kagglehub
echo "Done."

# Download dataset
echo ""
echo "Downloading UCF101 dataset (skipped if already cached)..."

python3 - "$KAGGLE_USERNAME" "$KAGGLE_KEY" <<'PYEOF'
import sys, os, kagglehub
os.environ["KAGGLE_USERNAME"] = sys.argv[1]
os.environ["KAGGLE_KEY"]      = sys.argv[2]
path = kagglehub.dataset_download("matthewjansen/ucf101-action-recognition")
print(f"Dataset path: {path}")
PYEOF

# Detect path + update config.py
echo ""
echo "Detecting path and updating config.py..."

if [ -d "$CACHE_PATH" ]; then
    DETECTED_PATH="${CACHE_PATH%/}/"
    OUTPUT_PATH="/content/ucf101-processed/"
    echo "Found in kagglehub cache: $DETECTED_PATH"
elif [ -d "$KAGGLE_INPUT_PATH" ]; then
    DETECTED_PATH="${KAGGLE_INPUT_PATH%/}/"
    OUTPUT_PATH="/kaggle/working/ucf101-processed/"
    echo "Found in Kaggle input: $DETECTED_PATH"
else
    echo "Dataset not found at:"
    echo "      $CACHE_PATH"
    echo "      $KAGGLE_INPUT_PATH"
    echo ""
    echo "Debug — cache contents:"
    find /root/.cache/kagglehub/datasets/matthewjansen/ -maxdepth 4 2>/dev/null \
        || echo "    (cache empty)"
    exit 1
fi

python3 - "$CONFIG_FILE" "$DETECTED_PATH" "$OUTPUT_PATH" <<'PYEOF'
import sys, re

config_path  = sys.argv[1]
dataset_root = sys.argv[2]
output_root  = sys.argv[3]

with open(config_path, "r") as f:
    content = f.read()

content = re.sub(
    r'DATASET_ROOT\s*=\s*["\'].*?["\']',
    f'DATASET_ROOT = "{dataset_root}"',
    content
)
content = re.sub(
    r'OUTPUT_ROOT\s*=\s*["\'].*?["\']',
    f'OUTPUT_ROOT  = "{output_root}"',
    content
)

with open(config_path, "w") as f:
    f.write(content)

print(f"DATASET_ROOT = '{dataset_root}'")
print(f"OUTPUT_ROOT  = '{output_root}'")
PYEOF

# Restart kernel after 5 seconds
echo ""
echo "Setup complete!"
echo "config.py updated on Drive."
echo "Restarting kernel in 5 seconds..."
echo "Dataset cache stays on disk."
echo "Run Cell 3 after restart to verify."

python3 - <<'PYEOF'
import time
from IPython import get_ipython

with open("/content/setup_done.flag", "w") as f:
    f.write("ok")

for i in range(5, 0, -1):
    print(f"Restarting in {i}...")
    time.sleep(1)

ip = get_ipython()
if ip is not None:
    ip.kernel.do_shutdown(True)
PYEOF
