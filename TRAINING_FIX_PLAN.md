# Why training failed and how to fix it

> **STATUS 2026-05-09 — Option B EXECUTED.** The corrected hyperparams in this plan (DPO lr 5e-6 × 3 epochs, LoRA r=64/α=128, expanded DPO pairs, Q8_0 quantization) were applied. Eval-gate result: trained beats base on both held-out coding (`36` vs `9`) and full similarity (`+0.357` vs `+0.124` tok-lean) — see [`EVAL_VERDICT.md`](EVAL_VERDICT.md) for the reversal note.
>
> The v3 round (now in flight) keeps the corrected hyperparams and expands data further: CPT 31.6M tokens (top 500 GitHub repos + book/RFCs/stdlib docs), DPO 374 pairs (235 compile-validated), SFT mined from stdlib doc-comments + READMEs + specs with compile-gate. See [`README.md`](README.md) and [`TODO.md`](TODO.md) for current state. **Publish target is hf.co, NOT ollama.com.**
>
> ---

## Root cause: the DPO stage was a no-op, and the rest barely moved the needle

I read the actual axolotl configs and dataset sizes. Here's what was actually trained:

| stage | examples | batch | epochs | steps | LR | LoRA r/α | issue |
|---|---|---|---|---|---|---|---|
| CPT | 1,756 | 8 | 1 | ~220 | **5.0e-6** | 16/32 | LR ~10× too low for CPT |
| SFT | 5,430 | 8 | 2 | ~1,357 | 1.0e-4 | 16/32 | reasonable |
| DPO | **74** | 4 | 1 | **~18** | **5.0e-7** | 16/32 | **catastrophic — LR 100× too low, dataset 10× too small, epochs 3× too few** |

### The math

For DPO, the cumulative weight update is roughly:
```
ΔW ≈ steps × lr × ||grad|| × scaling
   ≈ 18 × 5e-7 × O(1) × (32/16)
   ≈ 2e-5
```

That matches the LoRA delta we measured during the local merge (1e-7 to 1e-5 range). It's at or below the BF16 precision floor for non-zero weights, and **completely below Q4_K_M quantization noise** (~1-2% per weight). The DPO signal was erased before it ever ran in Ollama.

CPT at lr=5e-6 with 220 steps was also too conservative — typical CPT for code uses 2e-5 to 1e-4. SFT at lr=1e-4 with 1357 steps actually probably did move weights, but the regression from undertrained CPT + the verbosity drift may have washed it out.

### Why the eval showed regression, not just "no effect"

If training had been a true no-op, trained ≈ base. Instead trained is *worse* than base on 18/25 pairs. Two things contributed:

1. **CPT on 1,756 examples nudged the token distribution slightly** away from Qwen3-Coder's well-calibrated baseline, without adding enough new signal to compensate.
2. **Style drift**: the trained model writes more terse, less standard responses — minor catastrophic-forgetting-lite.

DPO had no role in either. With LR=5e-7, it was a $-spending no-op.

## What to fix — three options ordered by cost/value

### Option A — Cheapest: skip training, ship a system-prompted base ($0)

Most of the "Crystal-aware" behavior in `jaimef/crystal-qwen3.6-30b` actually comes from the SYSTEM block in the Modelfile (Crystal idioms, no method_missing, getter/setter, etc.). On `qwen3-coder:30b` with that same system prompt, you get the same behavior — measurably better, per our eval.

**Action:** publish the Modelfile (system prompt + template) as `crystal-system-prompted`, base on `qwen3-coder:30b`. No training, no $.

This is the honest answer. Try this first.

### Option B — One more cheap RunPod run with corrected hyperparams (~$10–20, 3–5h)

If you want to actually move the weights, the fixes are:

**DPO (the broken stage):**
- `learning_rate: 5.0e-6` (was 5e-7) — 10× higher
- `num_epochs: 3` (was 1) — more passes over the small set
- `lora_r: 64`, `lora_alpha: 128` (was 16/32) — bigger delta, survives Q4
- Expand `dpo_pairs.jsonl` to **300+ pairs** (was 74). Easy: programmatically generate Ruby→Crystal pairs from the Crystal stdlib docs (each `attr_*` method → 1 pair, each `Thread.new` example → 1 pair, etc.)

**CPT:**
- `learning_rate: 2.0e-5` (was 5e-6) — 4× higher
- `num_epochs: 2` (was 1)

**SFT:**
- Leave as is, settings were reasonable.

**Quantization:**
- Ship `Q8_0` (was Q4_K_M) — half the size savings, but LoRA contributions actually survive. ~32 GB instead of ~17 GB; still fits on a 64 GB Mac.

**Eval gate:**
- Run `eval_similarity.py` against base BEFORE pushing to ollama.com. If trained doesn't beat base on >50% of pairs, do not publish.

### Option C — Real Crystal model (~$100–500, 1–3 days, possibly weeks of data work)

The fundamental constraint is dataset scale. To push a 30B model meaningfully you want:

- **CPT corpus: 100M+ tokens of real Crystal code.** Scrape `crystal-lang/crystal`, `manastech`, `kemalcr`, `lucky-framework`, top-100 shards from shards.info. Currently we have 1,756 examples — likely <5M tokens.
- **SFT corpus: 20K+ Q/A pairs.** Generate from compiler error messages, stdlib docs, tutorial chapters. Currently 5,430.
- **DPO corpus: 1,000+ preference pairs.** Generate by taking working Crystal code and synthesizing the equivalent Ruby-ism for each (deterministic). Currently 74.
- **Multi-GPU (4× A100 or H100s)** for a full-rank or high-rank LoRA pass without 8-bit optimizer compromises.

This is the only path to a model that beats `qwen3-coder:30b` by a margin that matters. Anything less and you're paying RunPod to add noise.

## My recommendation

**Do Option A this week.** Ship the system-prompted base, prove to yourself the user-visible behavior is the same. If after a month of using it you find specific Crystal patterns where it still fails, *those* failures become your DPO dataset for Option B. Don't train blind.

Do not run another training pass with the current hyperparameters. Do not run another DPO at lr=5e-7. Do not push another model to ollama.com without an eval gate.

## Dataset expansion sketch (if you do Option B)

The single highest-leverage change: **300+ DPO pairs**, generated programmatically. Rough recipe:

```
For each Ruby idiom in [attr_accessor, attr_reader, attr_writer,
                        Thread.new, Thread.start, Mutex.new, Queue.new,
                        method_missing, define_method, instance_eval,
                        Array<T>, Hash<T>, Class<T>,
                        require "rspec", RSpec.describe, expect(...).to,
                        Proc.new, &block, block.call, block_given?, ...]:
  generate 15–20 short Q/A pairs
  - "Write a Crystal class with two String accessors" → chosen=property/rejected=attr_accessor
  - "Create a thread-safe counter in Crystal" → chosen=Channel+spawn / rejected=Thread.new+Mutex.new
  - etc.
```

That's ~25 idioms × ~15 variants = ~375 pairs. Half a day of careful generation. Combined with the LR fix, this alone might be enough to imprint.

## Eval gate (mandatory before any future push)

```bash
python3 eval_similarity.py \
  --models <new-trained> qwen3-coder:30b \
  --pairs-file dpo_pairs.jsonl \
  --out eval_pre_push.json
```

Required to publish: trained beats base on **≥60% of pairs** (currently: 7/25 = 28%, base wins).
