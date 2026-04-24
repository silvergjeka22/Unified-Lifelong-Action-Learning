set -e

# CONFIG edit these lines only
GITHUB_USER="silvergjeka22"
GITHUB_REPO="Unified-Lifelong-Action-Learning"
BRANCH="naive"
FOLDER="src"           # folder inside the repo to download
DEST="/content"        # root destination files go to /content/src/


# stop edit
echo " fetch_src.sh — Download /${FOLDER} from GitHub"

# Check token
echo ""
echo "Checking environment..."

if [ -z "$GITHUB_TOKEN" ]; then
    echo " GITHUB_TOKEN is not set."
    echo " Set it in Colab Cell 1:"
    echo "    os.environ['GITHUB_TOKEN'] = 'ghp_yourtoken'"
    exit 1
fi
echo "GITHUB_TOKEN set"

# Use GitHub API to get all file URLs recursively
echo "Fetching file list from GitHub API..."

API_URL="https://api.github.com/repos/${GITHUB_USER}/${GITHUB_REPO}/git/trees/${BRANCH}?recursive=1"

# Get all file paths under src/ and extract their raw download URLs
python3 - "$API_URL" "$GITHUB_TOKEN" "$FOLDER" "$BRANCH" "$GITHUB_USER" "$GITHUB_REPO" "$DEST" <<'PYEOF'
import sys, os, json, urllib.request

api_url     = sys.argv[1]
token       = sys.argv[2]
folder      = sys.argv[3]
branch      = sys.argv[4]
user        = sys.argv[5]
repo        = sys.argv[6]
dest        = sys.argv[7]

# Call GitHub API to get the full repo tree
req = urllib.request.Request(api_url, headers={
    "Authorization": f"token {token}",
    "Accept": "application/vnd.github.v3+json"
})

with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read())

# Filter only files (blobs) inside the target folder
files = [
    item for item in data["tree"]
    if item["type"] == "blob" and item["path"].startswith(f"{folder}/")
]

if not files:
    print(f"No files found under '{folder}/' in the repo.")
    print(f"Check that the folder exists on branch '{branch}'.")
    sys.exit(1)

print(f"Found {len(files)} file(s) under /{folder}/")

# Download each file preserving directory structure
for item in files:
    path       = item["path"]                         # e.g. src/hello.py
    raw_url    = f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}"
    local_path = os.path.join(dest, path)             # e.g. /content/src/hello.py
    local_dir  = os.path.dirname(local_path)

    os.makedirs(local_dir, exist_ok=True)

    # wget-style download using urllib
    req_file = urllib.request.Request(raw_url, headers={
        "Authorization": f"token {token}"
    })

    with urllib.request.urlopen(req_file) as r, open(local_path, "wb") as f:
        f.write(r.read())

    print(f"{path} -> {local_path}")

print(f"\n All files saved under {dest}/{folder}/")
PYEOF

# Verify contents
echo "Contents of ${DEST}/${FOLDER}/:"
find "${DEST}/${FOLDER}" -type f | sort

echo ""
echo "Done! /${FOLDER} is ready at ${DEST}/${FOLDER}"
echo "Now Import in Colab"

