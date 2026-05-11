# Reproducing Crystal-LoRA v3 from Scratch

This guide walks a third party from "fresh checkout" to "merged Crystal-LoRA-v3 GGUF, eval-gated and uploaded to Hugging Face." It's the same recipe used to produce the v3 model now hosted on `hf.co/jaimef/crystal-qwen-gguf`.

It is *not* a from-scratch language-model training guide — you start from `Qwen/Qwen3-Coder-30B-A3B-Instruct` and add a Crystal-specific LoRA on top.

---

## 0. What you'll end up with

| Artifact | Size | Purpose |
|---|---|---|
| `cpt_corpus_v3_merged.jsonl` | ~120 MB / ~31.6 M tokens | Continued-pre-training corpus (real `.cr` source + Crystal book/RFCs/website/stdlib docs) |
| `sft_v3_mined.jsonl` | ~25 MB / ~9,958 pairs | Compile-gated chat-format Q/A pairs |
| `dpo_pairs_v3.jsonl` | ~150 KB / 374 pairs | Ruby→Crystal preference triples for DPO |
| `runpod-pipeline-final/` (or `runpod-pipeline-merged-v3/`) | ~57 GB | Merged BF16 HuggingFace checkpoint |
| `crystal-qwen.Q8_0.gguf` | ~30 GB | Quantized GGUF for Ollama / llama.cpp |

Final model behaviour (vs the base `qwen3-coder:30b`): wins both the held-out coding eval and the 74-pair similarity eval. See [`EVAL_VERDICT.md`](EVAL_VERDICT.md) for the v1 post-mortem and what changed.

---

## 1. Prerequisites

### Hardware (local)
- ~150 GB free disk for the merged checkpoint + intermediate shards.
- Apple Silicon Mac or Linux box for orchestration. No local GPU needed (training is on RunPod).
- A machine that can run Ollama for the eval/use phase: 24 GB+ unified memory or 24 GB+ GPU.

### Hardware (rented)
- 1× **NVIDIA H200 SXM 141 GB** on RunPod (community pricing ~$3.99/hr at time of writing).
- The pipeline takes ~14 hours of *active* GPU time (CPT 6 h + SFT 5 h + DPO 3 h). Expect ~$60 of pod cost if everything goes right; closer to ~$90 if you hit the kind of hiccups described in §10.

### Software (local)
| Tool | Why | Install |
|---|---|---|
| `git` | Cloning sources | OS package manager |
| `gh` (GitHub CLI) | `gh search repos` for top-N Crystal repos | https://cli.github.com — then `gh auth login` |
| `python3` 3.10+ | Build scripts | OS package manager |
| `crystal` compiler | Compile-gates SFT / DPO pairs | https://crystal-lang.org/install/ |
| `runpod` Python pkg | Pod lifecycle | `pip install runpod` |
| `rsync`, `ssh`, `scp` | Pulling the merged checkpoint | preinstalled on Mac/Linux |
| `llama.cpp` | GGUF conversion + quantization | clone & `make` from https://github.com/ggerganov/llama.cpp |
| `ollama` | Local inference / eval | https://ollama.com/download |
| `hf` (Hugging Face CLI) | Publishing | `pip install -U huggingface_hub` then `hf auth login` |

### Accounts / tokens
- **RunPod API key**: write to `~/.runpod.token`. Get one at https://www.runpod.io/console/user/settings.
- **OpenRouter API key**: only needed if you want to LLM-augment SFT (skip and you keep ~5 K compile-gated SFT pairs instead of ~9,958). Set `OPENROUTER_API_KEY` env var. Costs ~$15-20 for one full augmentation pass.
- **Hugging Face token** (write scope): `hf auth login` — needed only for the publish step.

---

## 2. Fetch the upstream sources

```bash
./fetch_training_sources.sh
```

This script clones (idempotent):
- `crystal-lang/crystal` → `~/mine/crystal` *(stdlib + compiler source — used by SFT mining and doc harvest)*
- `crystal-lang/crystal-book` → `/tmp/crystal-book`
- `crystal-lang/crystal-website` → `/tmp/crystal-lang_crystal-website`
- `crystal-lang/rfcs` → `/tmp/crystal-lang_rfcs`
- `veelenga/awesome-crystal` → `/tmp/veelenga_awesome-crystal`

It does **not** clone the 500 GitHub repos for the CPT corpus — those are pulled by `build_cpt_corpus_v2.py` because dedup happens during the same pass.

If you already have a Crystal stdlib checkout at a different path, edit `CRYSTAL_STDLIB` / `CRYSTAL_SRC` near the top of `build_cpt_docs.py` and `build_cpt_corpus_v2.py` (or symlink `~/mine/crystal` to your checkout).

---

## 3. Build the three v3 datasets

Run from the repo root in this order. **Order matters**: the LLM-augment step reads the merged CPT corpus.

```bash
# 3a. CPT — top 500 Crystal repos via gh CLI + shallow clone (~30-60 min, ~3-5 GB temp)
python3 build_cpt_corpus_v2.py --out cpt_corpus_v3.jsonl --limit 500 --keep-clones
# Output: cpt_corpus_v3.jsonl  (~29 K records)
# Side-effect: crystal_corpus_raw/  (kept; build_sft_v3.py reads it for spec/README mining)

# 3b. CPT — Crystal book + RFCs + website + awesome-crystal + stdlib doc-comments (~30 sec)
python3 build_cpt_docs.py
# Output: cpt_docs.jsonl  (~1,659 records)

# 3c. CPT — cross-file SHA dedup + merge
python3 merge_cpt_v3.py
# Output: cpt_corpus_v3_merged.jsonl  (~31,292 records, ~31.6 M tokens, ~120 MB)

# 3d. SFT — mine compile-gated Q/A from stdlib doc-comments + READMEs + spec files (~5 min on 14-core box)
python3 build_sft_v3.py --out sft_v3_mined.jsonl
# Output: sft_v3_mined.jsonl  (~1,800 pairs from miner alone; merge with v1 below)

# 3e. SFT — (optional) LLM-augment via Claude Haiku 4.5 → costs ~$15-20 of OpenRouter (~30-60 min)
export OPENROUTER_API_KEY=sk-or-v1-...
python3 build_sft_llm.py --files 3000 --out sft_v3_llm.jsonl
# Output: sft_v3_llm.jsonl  (~4,000 pairs after compile-gate)
# Skip this step and you'll train on ~5 K SFT pairs instead of ~10 K.

# 3f. SFT — merge mined + LLM-augmented + (optionally) any v1 hand-curated pairs
# The training pipeline reads sft_v3_mined.jsonl by default. If you want the
# LLM-augmented set in the run, concatenate (with dedup) into sft_v3_mined.jsonl,
# or point runpod_train.py at a different file (see §5).

# 3g. DPO — programmatic Ruby→Crystal pair generators, compile-validated chosen blocks (~10 sec)
python3 build_dpo_pairs_v3.py --out dpo_pairs_v3.jsonl
# Output: dpo_pairs_v3.jsonl  (~374 pairs, 235 of which compile-validate)
```

### Expected dataset sizes (sanity check)

If your numbers are **wildly** different — say <10 K records in `cpt_corpus_v3_merged.jsonl` — something failed silently. Common causes: `gh search repos` returned <100 results (re-auth), `crystal` compiler missing (compile-gate rejects everything in step 3d), or `~/mine/crystal/src` empty.

| File | v3 reference | What to check if very different |
|---|---|---|
| `cpt_corpus_v3.jsonl` | ~29,633 records | `gh search` results, network, repo clone failures |
| `cpt_docs.jsonl` | ~1,659 records | All five `/tmp/*` dirs cloned successfully |
| `cpt_corpus_v3_merged.jsonl` | ~31,292 records | (sum of above minus dedup) |
| `sft_v3_mined.jsonl` | ~1,800 pairs (mined-only) | `crystal` compiler in PATH; `~/mine/crystal/src` populated |
| `sft_v3_llm.jsonl` | ~4,000 pairs | OpenRouter key valid; compile-gate passing |
| `dpo_pairs_v3.jsonl` | 374 pairs | Should be deterministic — the rule generators are seedless |

---

## 4. Provision the RunPod box

Save your RunPod token:
```bash
echo "your-runpod-key" > ~/.runpod.token
```

The orchestrator reads it from there.

```bash
pip install runpod
python3 runpod_train.py up
# Provisions an H200 SXM 141 GB pod with the axolotl image,
# waits for SSH, and writes pod info to .runpod_state.json.
```

Cost meter starts when the pod is `RUNNING` (~30-60 s after `up`).

---

## 5. Push configs + datasets to the pod

```bash
python3 runpod_train.py push
```

This `scp`s:
- `axolotl_crystal_cpt.yaml`
- `axolotl_crystal_sft.yaml`
- `axolotl_crystal_dpo.yaml`
- `cpt_corpus_v3_merged.jsonl` → `/workspace/data/cpt.jsonl`
- `sft_v3_mined.jsonl` → `/workspace/data/sft.jsonl`
- `dpo_pairs_v3.jsonl` → `/workspace/data/dpo.jsonl`

To use a different SFT file (e.g. the merged mined+LLM file), edit `PIPELINE_STAGES[1]["data_local"]` near line 80 of `runpod_train.py`.

---

## 6. Train (CPT → merge → SFT → merge → DPO → merge)

```bash
python3 runpod_train.py train
```

This launches three back-to-back tmux sessions on the pod:
- `stage_cpt` — runs CPT, then `axolotl merge-lora` to fold the LoRA into the base, leaving the result at `/workspace/cpt_merged`.
- `stage_sft` — base = `cpt_merged`. Trains SFT, merges to `/workspace/sft_merged`.
- `stage_dpo` — base = `sft_merged`. Trains DPO, merges to `/workspace/dpo_merged`.

Each stage drops a `.done` marker file (`cpt.done`, `sft.done`, `dpo.done`) on success/failure. The pipeline is **idempotent**: re-running `runpod_train.py train` will skip stages whose `.done` says `OK`.

### Hyperparameters (LoRA r=64, α=128 on attn + MLP + MoE experts, all stages)

| Stage | LR | Epochs | Effective batch | Notes |
|---|---|---|---|---|
| CPT | 2e-5 | 2 | 8 (mb 1 × ga 8) | `merge_method: legacy` to avoid AdaLoRA false-positive on fused MoE experts |
| SFT | 1e-4 | 2 | 8 | chat-format completion loss only on assistant tokens |
| DPO | 5e-6 | 3 | 8 | `chatml.argilla` format, β=0.1 |

### Monitoring

- `python3 runpod_train.py status` — pod state + per-stage `.done` markers
- `python3 runpod_train.py tail` — follows the active stage's log
- `ssh -p $PORT root@$HOST "tmux ls"` — see live tmux sessions on the pod

### Expected wall time

| Stage | Time | Cost @ $3.99/hr |
|---|---|---|
| CPT (~31.6 M tok × 2 ep) | ~6 h | ~$24 |
| Merge after CPT | ~1 min | <$0.10 |
| SFT (~10 K pairs × 2 ep) | ~5 h | ~$20 |
| Merge after SFT | ~1 min | <$0.10 |
| DPO (~370 pairs × 3 ep) | ~3 h | ~$12 |
| Merge after DPO | ~5 min | <$0.50 |
| **Total active** | **~14 h** | **~$57** |

**Watch out for**: pod-side disk pressure during the DPO merge step. With CPT/SFT/DPO checkpoints + the SFT-merged base + the DPO-merged output all on disk, peak usage approaches the 200 GB container limit. The pipeline cleans intermediate adapters as it progresses, but if you hit ENOSPC during merge, see §10.

### Eval signals during training

You should see `eval_loss` drop:
- CPT: ~0.93 → ~0.55 (-41%)
- SFT: ~1.01 → ~0.43 (-58%)
- DPO: `rewards/accuracies` should saturate near 1.0; `rewards/margins` > 5

If CPT eval-loss doesn't move, learning rate is wrong or your corpus is empty. If SFT eval-loss climbs, you've imprinted bad data — check the compile-gate output of step 3d.

---

## 7. Pull the merged checkpoint

```bash
python3 runpod_train.py pull
```

This `rsync`s `/workspace/dpo_merged/` → `runpod-pipeline-final/` on your local box. ~61 GB; expect ~10-100 min depending on your link to the RunPod region. (~10 MB/s is typical; opening a parallel `scp` for shard 2 can roughly double throughput — see §10.)

---

## 8. Terminate the pod

```bash
python3 runpod_train.py down
```

**Critical**: the pod meter keeps running between `pull` finishing and `down` being invoked. Don't forget this step. The bottom of `.runpod_state.json` records `pod_id` if you need to terminate manually via the RunPod UI.

---

## 9. Convert merged checkpoint → GGUF, eval-gate, publish

### 9a. Quantize to Q8_0

```bash
# In a llama.cpp checkout
python3 llama.cpp/convert_hf_to_gguf.py runpod-pipeline-final/ \
    --outfile crystal-qwen.gguf
./llama.cpp/llama-quantize crystal-qwen.gguf crystal-qwen.Q8_0.gguf Q8_0
```

**Q8_0, not Q4_K_M.** v1 was killed by Q4_K_M precision loss erasing the LoRA delta — see [`EVAL_VERDICT.md`](EVAL_VERDICT.md). Q8_0 is the sweet spot of size (~30 GB) and faithfulness for a LoRA-tuned 30B-A3B MoE.

### 9b. Load into Ollama

```bash
ollama create crystal-qwen -f Modelfile
# Modelfile = `FROM ./crystal-qwen.Q8_0.gguf` + the Crystal SYSTEM block + Qwen3 TEMPLATE
```

### 9c. Eval gate (mandatory before publishing)

```bash
# Held-out coding eval — questions never seen during training
python3 eval_holdout.py --models crystal-qwen qwen3-coder:30b --out eval_holdout.json

# Full 74-pair similarity eval — token-lean toward chosen vs rejected
python3 eval_similarity.py --models crystal-qwen qwen3-coder:30b --out eval_similarity_full.json
```

**Required to publish**: trained must beat base on both metrics. Reference v3 numbers:

| Metric | base `qwen3-coder:30b` | trained v3 |
|---|---|---|
| Held-out total | 9 | 36 |
| Tok-lean toward chosen | +0.124 | +0.357 |

If the trained model loses, **don't publish**. Diagnose: most likely culprits are (a) Q4_K_M instead of Q8_0, (b) DPO LR set back to v1's 5e-7, (c) corpus merge step skipped so CPT trained on duplicates.

### 9d. Publish to Hugging Face (NOT ollama.com)

```bash
hf auth login
hf upload jaimef/crystal-qwen-gguf crystal-qwen.Q8_0.gguf

# Optionally upload the merged HF source for downstream re-quantization
hf upload jaimef/crystal-qwen-30b runpod-pipeline-final/
```

The `Modelfile` in this repo points at `./crystal-qwen.Q8_0.gguf`; users do `hf download jaimef/crystal-qwen-gguf crystal-qwen.Q8_0.gguf --local-dir .` then `ollama create crystal-qwen -f Modelfile`.

---

## 10. Common failure modes & fixes

### Pod sits idle while you sleep
Solo'd from the v3 run: the local orchestrator process can die between stages, leaving the pod billing at $3.99/hr while no training runs. v3 added a pod-side watchdog pattern (a tmux session that polls for `.done` markers and launches the next stage) that survives local crashes. If you're scripting your own re-run, deploy the watchdog before walking away. Always check pod GPU util (not just "tmux session alive") at ≤30 min intervals.

### `ENOSPC` during DPO merge
The DPO merge needs ~57 GB scratch alongside the existing SFT-merged base (~57 GB) and adapter (~15 GB) and intermediate checkpoint (~23 GB). Near 200 GB total. Recovery:
1. `ssh root@pod "rm -rf /workspace/output_dpo/checkpoint-* /workspace/output_cpt /workspace/output_sft /workspace/dpo_merged_parent /workspace/dpo.done"` — frees ~70 GB while preserving the DPO adapter at `/workspace/output_dpo/adapter_model.safetensors` and the SFT-merged base.
2. Re-launch just the merge:
   ```bash
   ssh root@pod "tmux new-session -d -s dpo_remerge \
     'export PATH=/workspace/axolotl-venv/bin:\$PATH; \
      axolotl merge-lora /workspace/axolotl_crystal_dpo.yaml \
        --lora-model-dir=/workspace/output_dpo \
        --output-dir=/workspace/dpo_merged_parent && \
      mv /workspace/dpo_merged_parent/merged /workspace/dpo_merged && \
      rm -rf /workspace/dpo_merged_parent && \
      echo OK > /workspace/dpo.done'"
   ```

### Slow rsync pull
~10 MB/s on a single rsync. Open a parallel `scp` for shard 2 to roughly double throughput:
```bash
scp -P $PORT root@$HOST:/workspace/dpo_merged/model-00002-of-00002.safetensors \
    runpod-pipeline-final/.shard2.partial &
# When it finishes, rename to drop the .partial suffix so rsync skips it.
```

### "AdaLoRA detected" / merge errors
If `axolotl merge-lora` fails complaining about AdaLoRA, the YAML is missing `merge_method: legacy`. The fused MoE expert targets (`experts.gate_up_proj`, `experts.down_proj`) write a non-empty `rank_pattern` to `adapter_config.json` that the memory-efficient merger misidentifies as AdaLoRA. The legacy merger uses PEFT's `merge_and_unload()` directly and works.

### Compile-gate rejects everything
`crystal` compiler not in PATH, or `crystal_corpus_raw/` empty. The gate is intentionally lenient on missing third-party shards (`can't find file '...'` is treated as PASS) but strict on `syntax error`, `undefined method`, `undefined constant`, `expected ... but`. If even hand-curated pairs are rejected, your `crystal` install is broken — try `crystal eval 'puts 1'`.

### gh search returns <100 repos
Re-run `gh auth login` (the search endpoint requires auth even for public results), or check `gh api rate_limit`.

---

## 11. What changed from v1 (and why)

v1 was a no-op in trained weights. The DPO LoRA delta sat below the BF16 precision floor and Q4_K_M quantization erased the rest. Three changes made v3 actually move the needle:

| | v1 | v3 | Why it matters |
|---|---|---|---|
| Data volume (CPT) | ~9 MB | ~120 MB | LR-corrected gradient updates need real signal; 9 MB underfits |
| LR scaling | CPT 5e-6, DPO 5e-7 | CPT 2e-5, DPO 5e-6 | v1 LRs were 4-10× too low for a 30B-A3B MoE adapter |
| LoRA rank | r=16, α=32 | r=64, α=128 | Larger delta survives Q8_0 quantization |
| Quantization | Q4_K_M | Q8_0 | Q4 erased the (already-tiny) v1 delta entirely |

The full post-mortem is in [`EVAL_VERDICT.md`](EVAL_VERDICT.md); the corrective plan rationale is in [`TRAINING_FIX_PLAN.md`](TRAINING_FIX_PLAN.md).

---

## 12. Iterating on your own LoRA

Once everything works end-to-end, the cycle for adding a new Ruby→Crystal idiom or extra training data is:

1. Edit `build_dpo_pairs_v3.py` to add a `rule_*` generator → `python3 build_dpo_pairs_v3.py`
2. (Optional) re-mine SFT: `python3 build_sft_v3.py`
3. `python3 runpod_train.py up && push && train && pull && down`
4. Eval-gate (§9c). Don't publish a model that loses to the base.
5. Re-quantize, re-publish.

The whole loop is ~15-18 hours of wall time and ~$60-90 of pod cost. Plan accordingly.
