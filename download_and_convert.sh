#!/bin/bash
# Download LoRA adapter from Together AI and convert to GGUF for Ollama.
#
# Prerequisites:
#   pip install together
#   pip install peft safetensors  # for the converter
#
# Usage:
#   ./download_and_convert.sh
#
# This script:
#   1. Downloads the LoRA adapter from Together AI (~50-200MB)
#   2. Clones llama.cpp (if needed) for the converter script
#   3. Converts the adapter to GGUF format
#   4. Creates the Ollama model with the adapter
#
# No GPU required. No 32GB RAM required. Works on any machine.
#
# JOB_ID is auto-detected from .together_state.json (set by train_together.py).
# Override with: JOB_ID=ft-xxx ./download_and_convert.sh

set -euo pipefail
cd "$(dirname "$0")"

if [ -z "${JOB_ID:-}" ]; then
    if [ -f ".together_state.json" ]; then
        JOB_ID=$(python3 -c "import json; print(json.load(open('.together_state.json')).get('job_id',''))")
    fi
fi

if [ -z "${JOB_ID:-}" ]; then
    echo "ERROR: No JOB_ID found."
    echo "Either set it manually:    JOB_ID=ft-xxx ./download_and_convert.sh"
    echo "Or train first:            python3 train_together.py train"
    exit 1
fi

ADAPTER_DIR="./together-adapter"
LLAMA_CPP_DIR="./llama.cpp"
ADAPTER_GGUF="./crystal-lora-adapter.gguf"

echo "=== Step 1: Download LoRA adapter from Together AI ==="
echo "Job: $JOB_ID"
if [ -d "$ADAPTER_DIR" ] && [ -f "$ADAPTER_DIR/adapter_model.safetensors" ]; then
    echo "Adapter already downloaded at $ADAPTER_DIR"
else
    echo "Downloading adapter for job $JOB_ID ..."
    mkdir -p "$ADAPTER_DIR"
    together fine-tuning download "$JOB_ID" --checkpoint-type adapter --output_dir "$ADAPTER_DIR"
    echo "Downloaded to $ADAPTER_DIR"
fi

# Together AI downloads a zstd-compressed tar — extract it
if [ ! -f "$ADAPTER_DIR/adapter_config.json" ]; then
    echo "Extracting compressed archive ..."
    cd "$ADAPTER_DIR"
    for f in *; do
        if file "$f" | grep -q "Zstandard"; then
            tar --zstd -xf "$f" && rm "$f"
            break
        fi
    done
    cd ..
fi

if [ ! -f "$ADAPTER_DIR/adapter_config.json" ]; then
    echo "ERROR: adapter_config.json not found in $ADAPTER_DIR"
    ls -la "$ADAPTER_DIR"/
    exit 1
fi

echo ""
echo "=== Step 2: Get llama.cpp converter ==="
if [ -f "$LLAMA_CPP_DIR/convert_lora_to_gguf.py" ]; then
    echo "llama.cpp already cloned"
else
    echo "Cloning llama.cpp (sparse, just the converter scripts) ..."
    git clone --depth 1 --filter=blob:none --sparse https://github.com/ggml-org/llama.cpp.git "$LLAMA_CPP_DIR"
    cd "$LLAMA_CPP_DIR"
    git sparse-checkout set --skip-checks convert_lora_to_gguf.py gguf-py scripts
    cd ..
    echo "Done"
fi

if ! python3 -c "import gguf" 2>/dev/null; then
    echo "Installing gguf Python package ..."
    python3 -m pip install --break-system-packages gguf
fi
if ! python3 -c "import transformers" 2>/dev/null; then
    echo "Installing transformers ..."
    python3 -m pip install --break-system-packages transformers safetensors
fi
if ! python3 -c "import torch" 2>/dev/null; then
    echo "Installing PyTorch (CPU-only) ..."
    python3 -m pip install --break-system-packages torch --index-url https://download.pytorch.org/whl/cpu
fi

echo ""
echo "=== Step 3: Convert adapter to GGUF ==="
if [ -f "$ADAPTER_GGUF" ]; then
    echo "GGUF adapter already exists: $ADAPTER_GGUF"
else
    echo "Converting adapter to GGUF ..."
    python3 "$LLAMA_CPP_DIR/convert_lora_to_gguf.py" \
        --outfile "$ADAPTER_GGUF" \
        "$ADAPTER_DIR"
    echo "Converted: $ADAPTER_GGUF"
fi

echo ""
echo "=== Step 4: Pull base model in Ollama ==="
echo "Pulling qwen2.5-coder:7b-instruct (one-time ~4.7GB download) ..."
ollama pull qwen2.5-coder:7b-instruct

echo ""
echo "=== Step 5: Create Ollama model with adapter ==="
echo "Creating crystal-qwen model ..."
ollama create crystal-qwen -f Modelfile

echo ""
echo "=== Done! ==="
echo ""
echo "Test it:"
echo "  ollama run crystal-qwen 'How do I declare a class with type-annotated properties in Crystal?'"
echo ""
echo "Full verification:"
echo "  python3 verify_model.py --base-url http://localhost:11434/v1 --model crystal-qwen -v"
echo ""
echo "Configure OpenCode:"
echo "  Base URL: http://localhost:11434/v1"
echo "  Model:    crystal-qwen"
