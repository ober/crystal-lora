# Crystal LoRA — v3 Retrain Status

Current focus: re-run the full CPT → SFT → DPO pipeline on RunPod with the corrected hyperparams + v3 data, eval-gate, and publish to **Hugging Face (hf.co)**.

## Status (2026-05-09)

### Data (v3)
- [x] Scrape top 500 Crystal GitHub repos → `cpt_corpus_v3.jsonl` (29,633 `.cr` files)
- [x] Harvest Crystal book + RFCs + website + awesome-crystal + stdlib docs → `cpt_docs.jsonl` (1,659 entries)
- [x] Cross-file SHA-dedupe + merge → `cpt_corpus_v3_merged.jsonl` (31,292 records, ~31.6M tokens, 120 MB)
- [x] Expand DPO from 37 → 374 pairs (74 hand + 187 v2 rules + 113 new v3 rules); 235/374 compile-validated
- [x] Build SFT miner with parallel compile-gate (`build_sft_v3.py`, 14 workers)
- [x] Build LLM-augment backup (`build_sft_llm.py`, Claude Haiku 4.5 via OpenRouter, lenient missing-shard gate)
- [x] SFT mining run → `sft_v3_mined.jsonl` (1,774 stdlib + 9 readme + merged with v1)
- [x] LLM-augment SFT via Claude Haiku 4.5 → +3,813 pairs from 3,000 sampled corpus files
- [x] Merge mined + LLM-augmented + v1 with SHA-dedup → **9,958 final compile-gated SFT pairs** (~$8 spent)

### Training
- [x] Update axolotl YAMLs: r=64/α=128, CPT lr=2e-5/2 ep, SFT lr=1e-4/2 ep, DPO lr=5e-6/3 ep
- [x] Point `runpod_train.py` at v3 datasets (`cpt_corpus_v3_merged.jsonl`, `sft_v3_mined.jsonl`, `dpo_pairs_v3.jsonl`)
- [ ] Run full v3 retrain on RunPod (single A100 80GB)
- [ ] Pull merged checkpoint → `runpod-pipeline-final/`
- [ ] Terminate pod

### Eval gate (mandatory before publishing)
- [x] Held-out Crystal coding eval (`eval_holdout.py`)
- [x] Full 74-pair similarity eval (`eval_similarity.py`)
- [x] **Verdict reversed**: with corrected hyperparams + v2 data, trained beats base (`+0.357` vs `+0.124` tok-lean; `36` vs `9` held-out total)
- [ ] Re-run both evals on the v3-trained checkpoint; require trained > base on both

### Publish
- [ ] Convert merged checkpoint → GGUF (Q8_0) via `llama.cpp/convert_hf_to_gguf.py` + `llama-quantize`
- [ ] Upload to **Hugging Face** (`hf upload jaimef/crystal-qwen-gguf …`) — NOT ollama.com
- [ ] Optionally upload merged HF checkpoint at `runpod-pipeline-final/` as source
- [ ] Update Modelfile to point at the published GGUF; smoke-test via `ollama create … -f Modelfile`

## Why this round (vs the v1 run that flopped)

See [`EVAL_VERDICT.md`](EVAL_VERDICT.md) for the v1 post-mortem and [`TRAINING_FIX_PLAN.md`](TRAINING_FIX_PLAN.md) for the corrective hyperparams. Short version: v1 DPO LR was 100× too low — the LoRA delta sat below the BF16 precision floor and Q4_K_M quantization erased the rest. v3 fixes that and adds 10–100× more data across all three stages.
