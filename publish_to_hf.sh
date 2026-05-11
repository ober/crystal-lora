#!/usr/bin/env bash
# Publish a quantized GGUF + Modelfile + README to Hugging Face.
#
# This wraps the steps in REPRODUCE.md §9d that used to be raw `hf upload` commands.
# Uses hf-xet for chunked / resumable upload (handles 32 GB GGUFs over flaky links).
#
# Usage:
#   ./publish_to_hf.sh                                         # defaults: jaimef21/crystal-qwen-v3-30b-gguf
#   ./publish_to_hf.sh crystal-qwen-v3.Q8_0.gguf               # specific local GGUF
#   ./publish_to_hf.sh GGUF REPO                               # custom HF repo (user/name)
#   ./publish_to_hf.sh GGUF REPO REMOTE_GGUF_NAME              # rename in repo
#   ./publish_to_hf.sh GGUF REPO REMOTE_GGUF README_DIR        # source README + Modelfile from a different dir
#
# What it does:
#   1. Verifies `hf auth` is logged in.
#   2. Creates the repo if it doesn't exist (`hf repo create`).
#   3. Uploads README.md → README.md.
#   4. Uploads Modelfile → Modelfile.
#   5. Uploads the GGUF (renamed if requested).
#
# After publish, users pull via:
#   ollama pull hf.co/<repo>:Q8_0
# or:
#   hf download <repo> <gguf> --local-dir .
#   ollama create <name> -f Modelfile

set -euo pipefail

LOCAL_GGUF="${1:-crystal-qwen-v3.Q8_0.gguf}"
REPO="${2:-jaimef21/crystal-qwen-v3-30b-gguf}"
REMOTE_GGUF="${3:-crystal-qwen-v3-30b.gguf}"
DOC_DIR="${4:-hf-upload-v3}"

if [[ ! -s "$LOCAL_GGUF" ]]; then
  echo "ERROR: GGUF '$LOCAL_GGUF' not found. Run ./convert_to_gguf.sh first." >&2
  exit 1
fi

if [[ ! -f "$DOC_DIR/README.md" || ! -f "$DOC_DIR/Modelfile" ]]; then
  echo "ERROR: '$DOC_DIR/README.md' or '$DOC_DIR/Modelfile' missing." >&2
  echo "       Edit the v3 ones in hf-upload-v3/ as a template, or pass a different DOC_DIR." >&2
  exit 1
fi

require() {
  command -v "$1" >/dev/null 2>&1 || { echo "ERROR: \`$1\` not in PATH. Install: $2" >&2; exit 1; }
}
require hf "pip install -U huggingface_hub hf-xet"

if ! hf auth whoami >/dev/null 2>&1; then
  echo "ERROR: not logged in. Run \`hf auth login\` (needs a write-scope token)." >&2
  exit 1
fi

echo "=== 1/4 Ensure repo $REPO exists ==="
if hf repo create "$REPO" --repo-type model --exist-ok 2>&1 | grep -qi 'already exist'; then
  echo "  repo already exists — proceeding"
else
  echo "  repo created"
fi

echo "=== 2/4 Upload README.md ==="
hf upload "$REPO" "$DOC_DIR/README.md" README.md

echo "=== 3/4 Upload Modelfile ==="
hf upload "$REPO" "$DOC_DIR/Modelfile" Modelfile

echo "=== 4/4 Upload $LOCAL_GGUF as $REMOTE_GGUF (~$(du -h "$LOCAL_GGUF" | cut -f1); ~1-2 hours via hf-xet) ==="
hf upload "$REPO" "$LOCAL_GGUF" "$REMOTE_GGUF"

echo
echo "Done. Published: https://huggingface.co/$REPO"
echo
echo "Users can now pull via Ollama:"
echo "  ollama pull hf.co/$REPO"
echo "  ollama run  hf.co/$REPO 'Write a Crystal class Item.'"
