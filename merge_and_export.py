#!/usr/bin/env python3
"""
Merge LoRA adapter into base model and export to GGUF for Ollama.

Prerequisites:
  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

Usage:
  # From Unsloth training output:
  python3 merge_and_export.py

  # From Together AI download:
  together fine-tuning download <job-id> --output ./together-adapter
  python3 merge_and_export.py --adapter ./together-adapter --base Qwen/Qwen3-Coder-30B-A3B-Instruct

  # Custom quantization:
  python3 merge_and_export.py --quant q5_k_m

Quantization options (default: q4_k_m):
  q4_k_m   — 4-bit, good balance of quality/size
  q5_k_m   — 5-bit, better quality
  q8_0     — 8-bit, near-lossless
  f16      — full float16, no quantization

After export:
  ollama create crystal-qwen -f Modelfile
  ollama run crystal-qwen
"""

import argparse
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ADAPTER_DIR = os.path.join(SCRIPT_DIR, "crystal-lora-output")
DEFAULT_GGUF_DIR    = os.path.join(SCRIPT_DIR, "crystal-qwen-gguf")
DEFAULT_BASE_MODEL  = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
MAX_SEQ_LENGTH      = 4096


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter and export to GGUF")
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER_DIR,
                        help=f"Path to LoRA adapter directory (default: {DEFAULT_ADAPTER_DIR})")
    parser.add_argument("--base", default=None,
                        help=f"Base model name for Together AI adapters (default: auto-detect, "
                             f"fallback: {DEFAULT_BASE_MODEL})")
    parser.add_argument("--quant", default="q4_k_m",
                        choices=["q4_k_m", "q5_k_m", "q8_0", "f16"],
                        help="Quantization method (default: q4_k_m)")
    parser.add_argument("--output", default=DEFAULT_GGUF_DIR,
                        help=f"Output directory for GGUF (default: {DEFAULT_GGUF_DIR})")
    args = parser.parse_args()

    adapter_dir = os.path.abspath(args.adapter)
    if not os.path.isdir(adapter_dir):
        print(f"Adapter directory not found: {adapter_dir}")
        print()
        print("If you trained with Unsloth:")
        print("  python3 train_unsloth.py   # creates ./crystal-lora-output/")
        print()
        print("If you trained with Together AI:")
        print("  together fine-tuning download <job-id> --output ./together-adapter")
        print("  python3 merge_and_export.py --adapter ./together-adapter \\")
        print("       --base Qwen/Qwen3-Coder-30B-A3B-Instruct")
        sys.exit(1)

    adapter_config = os.path.join(adapter_dir, "adapter_config.json")
    needs_base = False
    if os.path.exists(adapter_config):
        import json
        with open(adapter_config) as f:
            config = json.load(f)
        base_from_config = config.get("base_model_name_or_path", "")
        if base_from_config and not args.base:
            print(f"Auto-detected base model: {base_from_config}")
            args.base = base_from_config
        elif not base_from_config and not args.base:
            needs_base = True

    if needs_base and not args.base:
        args.base = DEFAULT_BASE_MODEL
        print(f"No base model detected, using default: {args.base}")

    from unsloth import FastLanguageModel

    if args.base:
        print(f"Loading base model {args.base} ...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.base,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )
        print(f"Applying adapter from {adapter_dir} ...")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
    else:
        print(f"Loading adapter from {adapter_dir} ...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=adapter_dir,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )

    gguf_dir = os.path.abspath(args.output)
    print(f"Exporting to GGUF ({args.quant}) → {gguf_dir} ...")
    os.makedirs(gguf_dir, exist_ok=True)

    model.save_pretrained_gguf(
        gguf_dir,
        tokenizer,
        quantization_method=args.quant,
    )

    gguf_files = [f for f in os.listdir(gguf_dir) if f.endswith(".gguf")]
    if gguf_files:
        gguf_path = os.path.join(gguf_dir, gguf_files[0])
        size_mb = os.path.getsize(gguf_path) / 1024 / 1024
        print(f"\nGGUF saved: {gguf_path} ({size_mb:.0f} MB)")
    else:
        print(f"\nGGUF files saved to {gguf_dir}/")

    print(f"\nNext steps:")
    print(f"  ollama create crystal-qwen -f Modelfile")
    print(f"  ollama run crystal-qwen 'How do I declare a class with type-annotated properties in Crystal?'")


if __name__ == "__main__":
    main()
