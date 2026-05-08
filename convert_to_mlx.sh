#!/bin/bash
# Convert a merged HuggingFace model dir into a 4-bit MLX bundle for local use.
#
# Usage:  ./convert_to_mlx.sh <hf_model_dir> <mlx_out_dir>
# Default: ./convert_to_mlx.sh runpod-pipeline-final crystal-mlx-4bit
set -euo pipefail

SRC="${1:-runpod-pipeline-final}"
DST="${2:-crystal-mlx-4bit}"

if [ ! -d "$SRC" ]; then
  echo "error: $SRC not found. Run 'python3 runpod_train.py pull' first." >&2
  exit 1
fi

if ! python3 -c "import mlx_lm" 2>/dev/null; then
  echo "Installing mlx-lm..."
  pip install --quiet mlx-lm
fi

echo "Converting $SRC -> $DST (4-bit MLX)..."
python3 -m mlx_lm.convert --hf-path "$SRC" --mlx-path "$DST" -q --q-bits 4

echo
echo "Done. Smoke test:"
echo "  python3 -m mlx_lm.generate --model $DST --prompt 'How do I parse JSON into a typed class in Crystal?' --max-tokens 256"
echo
echo "Verify against the test suite:"
echo "  python3 -m mlx_lm.server --model $DST --port 8080 &"
echo "  python3 verify_model.py --base-url http://localhost:8080/v1 --model $DST -v"
