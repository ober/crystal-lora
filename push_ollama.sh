#!/bin/bash
# Push the Crystal Qwen model to the Ollama registry.
#
# Steps:
#   1. Verify the model exists locally
#   2. Tag as USERNAME/crystal-qwen if needed
#   3. Push to Ollama registry
#   4. Verify push succeeded
#
# Usage:
#   ./push_ollama.sh [USERNAME]
#
# Example:
#   ./push_ollama.sh jaimef

set -euo pipefail

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    echo "Usage: $0 [USERNAME]"
    echo ""
    echo "Push the local crystal-qwen model to the Ollama registry."
    echo ""
    echo "  USERNAME   Ollama registry username (default: jaimef)"
    echo ""
    echo "Example:"
    echo "  $0 jaimef"
    exit 0
fi

USERNAME="${1:-jaimef}"
LOCAL_MODEL="crystal-qwen"
REMOTE_MODEL="${USERNAME}/crystal-qwen"

echo "=== Push Crystal Qwen to Ollama Registry ==="
echo ""
echo "  Local model:  $LOCAL_MODEL"
echo "  Remote model: $REMOTE_MODEL"
echo ""

if ! command -v ollama &>/dev/null; then
    echo "ERROR: 'ollama' not found. Install from: https://ollama.com"
    exit 1
fi

# ── Step 1: Verify model exists locally ──────────────────────────────
echo "=== Step 1: Verify local model ==="

if ! ollama list 2>/dev/null | grep -q "$LOCAL_MODEL"; then
    echo "ERROR: Model '$LOCAL_MODEL' not found locally."
    echo ""
    echo "Build it first:"
    echo "  ./download_and_convert.sh"
    echo ""
    echo "Or pull from registry:"
    echo "  ollama pull $REMOTE_MODEL"
    exit 1
fi

echo "Found local model: $LOCAL_MODEL"
ollama list 2>/dev/null | grep "$LOCAL_MODEL" || true
echo ""

# ── Step 2: Tag for registry ─────────────────────────────────────────
echo "=== Step 2: Tag as $REMOTE_MODEL ==="

if ollama list 2>/dev/null | grep -q "$REMOTE_MODEL"; then
    echo "Already tagged as $REMOTE_MODEL"
else
    echo "Tagging $LOCAL_MODEL -> $REMOTE_MODEL ..."
    ollama cp "$LOCAL_MODEL" "$REMOTE_MODEL"
    echo "Tagged."
fi
echo ""

# ── Step 3: Push to registry ─────────────────────────────────────────
echo "=== Step 3: Push to Ollama registry ==="
echo "Pushing $REMOTE_MODEL ..."
echo "(This may take a few minutes for the first push)"
echo ""

ollama push "$REMOTE_MODEL"

echo ""

# ── Step 4: Verify ───────────────────────────────────────────────────
echo "=== Step 4: Verify ==="
echo ""
echo "Model pushed successfully!"
echo ""
echo "  Registry URL: https://ollama.com/$REMOTE_MODEL"
echo ""
echo "  Anyone can now pull it with:"
echo "    ollama pull $REMOTE_MODEL"
echo ""
echo "Done!"
