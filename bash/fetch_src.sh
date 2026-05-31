#!/usr/bin/env bash
# =============================================================================
# fetch_src.sh — Download /src from a specific GitHub branch to Colab + Drive
#
# HOW TO CHANGE BRANCH:
#   Option A — edit BRANCH below (one line, commit, re-run)
#   Option B — override from Colab before calling this script:
#               os.environ["BRANCH"] = "kd&mamal"
#               !bash /content/fetch_src.sh
# =============================================================================
set -e

# ─── EDIT ONLY THIS BLOCK ────────────────────────────────────────────────────
GITHUB_USER="silvergjeka22"
GITHUB_REPO="Unified-Lifelong-Action-Learning"
BRANCH="${BRANCH:-kd&mamal}"          # ← CHANGE BRANCH HERE (or set env var)
FOLDER="src"                          # folder inside repo to download
DEST="/content"                       # local destination  → /content/src/
DRIVE_DEST="/content/drive/MyDrive/apai/src"  # Drive backup (set "" to skip)
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "======================================================"
echo "  fetch_src.sh — GitHub Source Fetcher"
echo "======================================================"
echo "  Repo   : ${GITHUB_USER}/${GITHUB_REPO}"
echo "  Branch : ${BRANCH}"
echo "  Folder : /${FOLDER}"
echo "  Local  : ${DEST}/${FOLDER}/"
echo "  Drive  : ${DRIVE_DEST:-disabled}"
echo "======================================================"
echo ""

# ── Token check ──────────────────────────────────────────────────────────────
if [ -z "$GITHUB_TOKEN" ]; then
    echo "  ERROR: GITHUB_TOKEN is not set."
    echo ""
    echo "  Set it in Colab Cell 1:"
    echo "    import os"
    echo "    os.environ['GITHUB_TOKEN'] = 'ghp_yourtoken'"
    exit 1
fi
echo "  [OK] GITHUB_TOKEN set"

# ── Drive check (optional) ───────────────────────────────────────────────────
if [ -n "$DRIVE_DEST" ]; then
    if [ ! -d "/content/drive/MyDrive" ]; then
        echo "  [!] Drive not mounted — Drive backup will be skipped."
        echo "      Mount first: from google.colab import drive; drive.mount('/content/drive')"
        DRIVE_DEST=""
    else
        echo "  [OK] Drive mounted"
    fi
fi

# ── Fetch files via GitHub API ────────────────────────────────────────────────
echo ""
echo "  Fetching file list from GitHub API..."

API_URL="https://api.github.com/repos/${GITHUB_USER}/${GITHUB_REPO}/git/trees/${BRANCH}?recursive=1"

python3 - \
    "$API_URL" "$GITHUB_TOKEN" "$FOLDER" "$BRANCH" \
    "$GITHUB_USER" "$GITHUB_REPO" "$DEST" "$DRIVE_DEST" \
<<'PYEOF'
import sys, os, json, urllib.request, shutil

api_url    = sys.argv[1]
token      = sys.argv[2]
folder     = sys.argv[3]
branch     = sys.argv[4]
user       = sys.argv[5]
repo       = sys.argv[6]
dest       = sys.argv[7]
drive_dest = sys.argv[8]

req = urllib.request.Request(api_url, headers={
    "Authorization": f"token {token}",
    "Accept":        "application/vnd.github.v3+json"
})
try:
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
except urllib.error.HTTPError as e:
    print(f"\n  ERROR: GitHub API returned {e.code}")
    if e.code == 404:
        print(f"  Branch '{branch}' not found — check GITHUB_TOKEN and BRANCH.")
    sys.exit(1)

files = [
    item for item in data.get("tree", [])
    if item["type"] == "blob" and item["path"].startswith(f"{folder}/")
]

if not files:
    print(f"  ERROR: No files found under '{folder}/' on branch '{branch}'.")
    sys.exit(1)

print(f"  Found {len(files)} file(s) under /{folder}/\n")

for item in files:
    path       = item["path"]
    raw_url    = f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}"
    local_path = os.path.join(dest, path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    req_file = urllib.request.Request(raw_url, headers={"Authorization": f"token {token}"})
    with urllib.request.urlopen(req_file) as r, open(local_path, "wb") as f:
        f.write(r.read())
    print(f"  [OK] {path}")

print(f"\n  Downloaded {len(files)} file(s) -> {dest}/{folder}/")

if drive_dest:
    print(f"\n  Syncing to Drive: {drive_dest} ...")
    src_local = os.path.join(dest, folder)
    if os.path.exists(drive_dest):
        shutil.rmtree(drive_dest)
    shutil.copytree(src_local, drive_dest)
    print(f"  [OK] Synced to Drive.")
else:
    print("\n  Drive backup skipped.")
PYEOF

echo ""
echo "  Contents of ${DEST}/${FOLDER}/:"
find "${DEST}/${FOLDER}" -type f | sort | sed 's/^/    /'
echo ""
echo "  Done. Add to path in Colab with:"
echo "    import sys; sys.path.insert(0, '/content')"
echo ""
