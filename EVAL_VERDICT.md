# Verdict: training did not help. Stop spending.

> **UPDATE 2026-05-09 — VERDICT REVERSED.** The corrective retrain (see [`TRAINING_FIX_PLAN.md`](TRAINING_FIX_PLAN.md)) actually moved the weights. With DPO lr raised from 5e-7 → 5e-6 (3 epochs), LoRA r/α from 16/32 → 64/128, expanded data, and Q8_0 (not Q4_K_M) quantization, the trained model now beats `qwen3-coder:30b`:
>
> - **Full 74-pair similarity (`eval_similarity_full.json`)**: trained tok_lean = `+0.357` vs base `+0.124` → trained wins by `+0.233`. trained wins_chosen = 6 vs base 2 (out of pairs that scored). Char-lean is roughly tied (within noise).
> - **Held-out coding eval (`eval_holdout_baseline.json`)**: trained total = `36` vs base `9` (idiom + compile-pass scoring on never-seen prompts). Trained generated code in 7 of 7 tasks vs base in 2 of 7.
>
> The original verdict below was based on the small-sample 25-pair eval against the v1 (broken) checkpoint and is preserved as an honest record of what we believed at the time. **Do not act on it.** The active plan is the v3 retrain documented in [`README.md`](README.md) and [`TODO.md`](TODO.md).
>
> ---
>
> *Original verdict text follows, unchanged:*

## The numbers (data leakage by design — we tested ON the training set)

We ran 25 of the 74 DPO training pairs through both models and measured per-pair similarity (token Jaccard + character SequenceMatcher) of each model's response against the "chosen" Crystal-style answer vs the "rejected" Ruby-ism answer. `tok_lean = sim(response, chosen) - sim(response, rejected)`. Higher = leans toward chosen.

| metric | trained (`jaimef/crystal-qwen3.6-30b`) | base (`qwen3-coder:30b`) | winner |
|---|---|---|---|
| Sum token-similarity lean | **+1.36** | **+2.32** | base by 71% |
| Mean tok-lean / pair | +0.054 | +0.093 | base |
| Sum char-similarity lean | +0.74 | +1.50 | base by 2× |
| Pairs leaning toward chosen | 19/25 | 25/25 | base wins all 25 |
| Pairs leaning toward rejected | 6/25 | 0/25 | base never picks Ruby |
| Head-to-head per-pair wins | **7/25** | **18/25** | base 2.5× more often |
| Worst trained loss | −0.18 | — | bad |
| Best trained gain | +0.06 | — | tiny |

### Head-to-head per-pair (token-similarity to chosen training answer)
- Trained beats base: **7/25 pairs**
- Base beats trained: **18/25 pairs**
- Ties: 0/25

### Worst pairs for trained (where trained is much further from chosen than base)
- pair 24: trained-base delta = −0.182
- pair 21: trained-base delta = −0.154
- pair 12: trained-base delta = −0.151
- pair 13: trained-base delta = −0.131
- pair 15: trained-base delta = −0.073

### Best pairs for trained (training actually helped)
- pair 9:  +0.047
- pair 0:  +0.052
- pair 7:  +0.057
- pair 19: +0.058
- pair 4:  +0.062

## Why this is the strongest possible test

We tested the model against its own training data. If training had imprinted at all, the trained model should have memorized these "chosen" answers and matched them more closely than the base. Instead, the **base model wrote responses more similar to the chosen training data on 18/25 pairs**.

This isn't a noise issue. It isn't a verbosity issue (the metric is volume-invariant — Jaccard ratio and char-similarity both normalize for length). It's a real signal that the training run did not move the model toward its target.

## Why training failed

1. **Qwen3-Coder-30B already speaks Crystal.** Crystal is in its pretraining data. There was no headroom for our 74 DPO pairs / small SFT corpus to add anything new.

2. **DPO LoRA delta was ~1e-7** — below the BF16 precision floor. Q4_K_M quantization erased it entirely. Sanity-check on layer 0 during merge confirmed: q_proj delta max = 1.9e-6, gate_up_proj delta max = 3e-5, both vanishingly small relative to base weight magnitudes (~1e-2).

3. **CPT+SFT+DPO on a tiny corpus introduced minor regression** (catastrophic-forgetting-lite) without adding new capability. The model now writes slightly less standard Crystal than it did before training.

4. **Earlier eval (`eval_crystal.py`) looked like a tie (11 vs 11)** because that test rewards short specific outputs and the trained model is more terse — that masked the regression. The volume-invariant similarity test (`eval_similarity.py`) exposes it.

## Recommendation

**Stop training. Don't spend more on RunPod for this configuration.**

If you genuinely want a Crystal-specialized model, the path forward is:

- **10–100× more data**: real Crystal repos (shards), spec files, the standard library source. 74 DPO pairs is not enough to overcome a strong base.
- **Higher rank LoRA (r=64+) and higher learning rate** so the delta survives quantization.
- **Train at higher precision and ship the merged model unquantized or only Q8_0**, not Q4_K_M, if you want LoRA contributions to survive.
- **Or: skip training entirely.** A good system prompt on `qwen3-coder:30b` gets you the same Crystal output for $0. The Ollama Modelfile SYSTEM block we wrote already encodes the Crystal idioms — most of the visible "Crystal-aware" behavior in the deployed model is actually coming from the system prompt, not the weights.

For the current Ollama tag you already pushed: it is not measurably better than `qwen3-coder:30b` for Crystal. If you keep it, frame it as a "Crystal-system-prompted Qwen3-Coder variant," not a "Crystal-trained model."

## Files generated for this evaluation

- `eval_crystal.py` — 10 hand-crafted Crystal vs Ruby divergence prompts (regex pattern test). Result: 11 vs 11 tie, but verbosity-confounded.
- `eval_dpo_preference.py` — Counts Crystal idiom regex hits vs Ruby-ism regex hits in responses to DPO instructions. Smoke test on 3 pairs: trained −4 vs base +9. Verbosity-confounded — base writes more, hits more patterns.
- `eval_similarity.py` — Per-pair similarity to chosen vs rejected. Volume-invariant. **This is the eval to trust.** Result above.
- `eval_results.json`, `eval_similarity.json` — Raw outputs.
