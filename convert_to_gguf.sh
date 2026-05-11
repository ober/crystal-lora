#!/usr/bin/env bash
# Convert a merged HuggingFace checkpoint to GGUF, quantize, and register with Ollama.
#
# This wraps the steps in REPRODUCE.md §9a/9b that used to be raw llama.cpp commands.
#
# Usage:
#   ./convert_to_gguf.sh                                    # defaults: runpod-pipeline-merged-v3 → crystal-qwen-v3.Q8_0.gguf
#   ./convert_to_gguf.sh runpod-pipeline-merged-v4         # other source dir
#   ./convert_to_gguf.sh SRC OUTBASE                        # custom output basename (no extension)
#   ./convert_to_gguf.sh SRC OUTBASE Q4_K_M                 # custom quantization level
#   ./convert_to_gguf.sh SRC OUTBASE Q8_0 my-ollama-name    # custom Ollama tag (default: OUTBASE)
#
# What it does:
#   1. Verifies / clones llama.cpp into ./llama.cpp (gitignored).
#   2. python3 llama.cpp/convert_hf_to_gguf.py SRC --outfile OUTBASE.gguf
#   3. ./llama.cpp/llama-quantize OUTBASE.gguf OUTBASE.QUANT.gguf QUANT
#   4. Writes Modelfile.<outbase> (Qwen3 chatml TEMPLATE + Crystal SYSTEM block).
#   5. ollama create OLLAMA_TAG -f Modelfile.<outbase>
#
# Disk: needs ~95 GB scratch (BF16 GGUF ~60 GB + Q8_0 ~32 GB). Q4_K_M needs ~78 GB total.
# Time: ~10 min on a Mac Studio for the full convert+quantize pass on a 30B-A3B model.
#
# Why Q8_0 by default: see EVAL_VERDICT.md — Q4_K_M previously erased the LoRA delta.
# Republishing v3 at Q4_K_M for the quant-confound A/B (see v4.md) is a known v4 task.

set -euo pipefail

SRC="${1:-runpod-pipeline-merged-v3}"
OUTBASE="${2:-crystal-qwen-v3}"
QUANT="${3:-Q8_0}"
OLLAMA_TAG="${4:-$OUTBASE}"

if [[ ! -d "$SRC" ]]; then
  echo "ERROR: source dir '$SRC' not found. Run \`python3 runpod_train.py pull\` first." >&2
  exit 1
fi

require() {
  command -v "$1" >/dev/null 2>&1 || { echo "ERROR: \`$1\` not in PATH. Install: $2" >&2; exit 1; }
}
require python3 "system package manager"
require ollama  "https://ollama.com/download"

if [[ ! -d llama.cpp ]]; then
  echo "=== Cloning llama.cpp (~150 MB; gitignored) ==="
  git clone --depth=1 https://github.com/ggerganov/llama.cpp.git
fi

if [[ ! -x llama.cpp/build/bin/llama-quantize && ! -x llama.cpp/llama-quantize ]]; then
  echo "=== Building llama.cpp (cmake; ~3-5 min) ==="
  ( cd llama.cpp && cmake -B build -DGGML_METAL=ON >/dev/null && cmake --build build --config Release -j --target llama-quantize )
fi

# Locate llama-quantize (cmake default path vs. legacy Makefile path)
if   [[ -x llama.cpp/build/bin/llama-quantize ]]; then QUANTIZE=llama.cpp/build/bin/llama-quantize
elif [[ -x llama.cpp/llama-quantize ]];           then QUANTIZE=llama.cpp/llama-quantize
else echo "ERROR: built llama-quantize not found" >&2; exit 1
fi

BF16_GGUF="${OUTBASE}.gguf"
QUANT_GGUF="${OUTBASE}.${QUANT}.gguf"
MODELFILE="Modelfile.${OUTBASE}"

if [[ -s "$BF16_GGUF" ]]; then
  echo "=== Skip: BF16 GGUF $BF16_GGUF already exists ($(du -h "$BF16_GGUF" | cut -f1)) ==="
else
  echo "=== 1/3 BF16 conversion: $SRC → $BF16_GGUF (~5-10 min, ~60 GB output) ==="
  python3 llama.cpp/convert_hf_to_gguf.py "$SRC" --outfile "$BF16_GGUF"
fi

if [[ -s "$QUANT_GGUF" ]]; then
  echo "=== Skip: $QUANT_GGUF already exists ($(du -h "$QUANT_GGUF" | cut -f1)) ==="
else
  echo "=== 2/3 Quantize $QUANT: $BF16_GGUF → $QUANT_GGUF (~3-5 min) ==="
  "$QUANTIZE" "$BF16_GGUF" "$QUANT_GGUF" "$QUANT"
fi

if [[ ! -f "$MODELFILE" ]]; then
  echo "=== 3/3 Writing $MODELFILE ==="
  cat > "$MODELFILE" <<MODELFILE_EOF
FROM ./${QUANT_GGUF}

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
"""

SYSTEM """You are an expert in Crystal, a statically-typed, Ruby-syntax-inspired language that compiles to native code via LLVM. You provide accurate, idiomatic Crystal code with correct types, require statements, and method signatures. Crystal looks like Ruby but is type-checked at compile time: no method_missing, no eval, methods overload on argument types, unions are first-class (String | Nil, abbreviated String?), generics use parens (Array(Int32), Hash(String, Int32)), and macros run at compile time. Use getter/setter/property instead of Ruby's attr_*. Concurrency uses fibers and channels (spawn, Channel(T)). The standard test framework is the built-in Spec (require "spec"), not RSpec. When writing code, always include the needed require statements."""

PARAMETER temperature 0.2
PARAMETER num_ctx 32768
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
MODELFILE_EOF
fi

echo "=== Registering with Ollama as '$OLLAMA_TAG' ==="
ollama create "$OLLAMA_TAG" -f "$MODELFILE"

echo
echo "Done. Try: ollama run $OLLAMA_TAG 'Write a Crystal class Item with name and quantity that serializes to JSON.'"
echo
echo "Next: eval-gate before publishing (REPRODUCE.md §9c):"
echo "  python3 eval_holdout.py    --models $OLLAMA_TAG qwen3-coder:30b --out eval_holdout_v3.json"
echo "  python3 eval_similarity.py --models $OLLAMA_TAG qwen3-coder:30b --out eval_similarity_full.json"
echo
echo "Then: ./publish_to_hf.sh $QUANT_GGUF"
