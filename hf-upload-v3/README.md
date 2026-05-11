---
license: apache-2.0
language:
- en
base_model: Qwen/Qwen3-Coder-30B-A3B-Instruct
tags:
- crystal
- crystal-lang
- code
- gguf
- qwen3
- lora
- fine-tuned
pipeline_tag: text-generation
---

# crystal-qwen-v3-30b-gguf

LoRA-fine-tuned [Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct)
specialised for the [Crystal](https://crystal-lang.org/) programming language.
Quantized to **Q8_0** GGUF for local inference via
[Ollama](https://ollama.com/) or [llama.cpp](https://github.com/ggerganov/llama.cpp).

This is **v3** — a full-pipeline retrain on a substantially larger and cleaner
corpus than [v2 (`crystal-qwen3.6-30b-gguf`)](https://huggingface.co/jaimef21/crystal-qwen3.6-30b-gguf).

## Quick start (Ollama)

```bash
# Download the GGUF + Modelfile from this repo, then:
ollama create crystal-qwen-v3 -f Modelfile
ollama run crystal-qwen-v3 "Write a Crystal class Item with name and quantity that serializes to JSON."
```

## What v3 changes vs v2

| | v2 (`3.6`) | **v3** |
|---|---|---|
| CPT tokens | ~3 M | ~50 M (top-500 Crystal GitHub repos + stdlib + book + RFCs) |
| SFT pairs | ~600 | ~3 000 (mined + Claude-Haiku-4.5 augmented, compile-gated) |
| DPO pairs | 37 | 74 (chosen = idiomatic Crystal; rejected = Ruby-style or wrong) |
| LoRA rank / α | 32 / 64 | 64 / 128 |
| Quantization | Q4_K_M | **Q8_0** |
| Targets | attn only | attn + MLP + MoE `experts.gate_up_proj/down_proj` |

## Eval

Held-out eval (30 natural-language Crystal tasks; idiom score + `crystal build --no-codegen` compile gate). Higher is better.

| Model | Idiom | Compile pass | Total |
|---|---|---|---|
| **crystal-qwen-v3 (this model)** | **+76** | **26/28 (93 %)** | **+206** |
| jaimef21/crystal-qwen3.6-30b (v2) | +67 | 21/28 (75 %) | +172 |
| Qwen3-Coder-30B-A3B-Instruct (base) | +71 | 21/28 (75 %) | +176 |

v3 beats both prior baselines on every axis. v2 actually slightly *trailed* the
base model on idiom (+67 vs +71) — v3 is the first checkpoint of this lineage
that is unambiguously better than vanilla Qwen3-Coder at Crystal.

Caveat: v3 ships at Q8_0 while v2 and base measurements above are Q4_K_M, so
some compile-gate fidelity gap is attributable to less quantization noise.

## Files

- `crystal-qwen-v3-30b.gguf` — 32 GB, Q8_0 (8.51 BPW)
- `Modelfile` — Ollama Modelfile (chatml template + Crystal system prompt)
- `README.md` — this file

## Reproducing

Full reproduction pipeline (data scraping → training on RunPod H200 → GGUF
quantization → eval) is documented at:
[github.com/jaimef/crystal-lora](https://github.com/jaimef/crystal-lora) — see
`REPRODUCE.md`.

## License

Inherits Qwen's [Apache 2.0 license](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct/blob/main/LICENSE)
from the base model.
