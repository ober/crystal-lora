# Option C — Real Crystal Model: Concrete Breakdown

> **STATUS 2026-05-09 — IN EXECUTION (lighter variant).** Most of this plan is being implemented for the v3 round, on a single A100 80GB rather than the 4× A100 multi-GPU setup originally proposed. Publish target is **hf.co**, not the Ollama registry.
>
> | Step | Plan | What we're doing for v3 |
> |------|------|--------------------------|
> | CPT corpus | 100M+ tokens from top 500 shards | **31.6M tokens** from top 500 GitHub Crystal repos + book/RFCs/website/stdlib docs (`cpt_corpus_v3_merged.jsonl`) |
> | SFT corpus | 20K+ Q/A, compile-gated synthetic | Mined from stdlib doc-comments + READMEs + spec describe/it blocks, compile-gated; Claude Haiku 4.5 augmenter as backup |
> | DPO corpus | 1K+ programmatic pairs | **374 pairs** (235 compile-validated) across ~50 Ruby idioms |
> | Hyperparams | r=64/α=128, CPT 2e-5×2, SFT 5e-5×3, DPO 5e-6×3 | r=64/α=128, **CPT 2e-5×2**, **SFT 1e-4×2**, **DPO 5e-6×3** (matches plan) |
> | Hardware | 4× A100 / 2× H100 | **single A100 80GB** (cheaper, slower; LoRA fits) |
> | Eval gate | Held-out 100 prompts, ≥15% compile-pass beat | `eval_holdout.py` (already shows trained `36` vs base `9` total on v2 round) |
>
> The data scraping skeleton at the bottom of this doc is essentially what `build_cpt_corpus_v2.py` now does. See [`README.md`](README.md) for the live pipeline.
>
> ---

The "real" path: a Crystal-specialized model that genuinely beats `qwen3-coder:30b` by a margin worth paying for. ~1 week of focused work + ~$250 in compute.

## What you'd need to build

### 1. CPT corpus: 100M+ tokens of real Crystal code (1–3 days)

**Scrape sources:**
- `crystal-lang/crystal` (compiler + stdlib, ~500K LOC)
- Top 100–500 shards from `shards.info` (community libs)
- Notable projects: Lucky framework, Kemal, Amber, Marten, Crystal-DB, Crystal-Redis
- GitHub search `language:Crystal` sorted by stars (top 1000)

**Tools:** `gh repo list`, `git clone --depth=1` in a loop, then walk for `.cr` files

**Clean:**
- dedup near-duplicates (file SHA)
- strip generated/vendored code
- filter test fixtures

**Format:** one `.cr` file per JSONL row with `text` field; CPT is unsupervised next-token

**Realistic size:** 50–200M tokens depending on dedup aggressiveness

**Storage:** ~500 MB–2 GB of cleaned text

### 2. SFT corpus: 20K+ Q/A pairs (2–5 days)

**Sources:**
- Crystal API docs (`crystal-lang.org/api`) — every method becomes "How do I X?" → "Use `Foo#bar(...)`"
- Crystal compiler error messages → "I got error X, what's wrong?" → "You need to..."
- Crystal tutorial chapters (book.crystal-lang.org) → restructured as Q/A
- Community Q/A: stackoverflow, gitter logs, GitHub Discussions on crystal-lang/crystal
- Synthetic generation: feed Claude/GPT-4 each stdlib module → "generate 30 idiomatic Q/A pairs"

**Quality gate:** every "answer" code block must `crystal build` cleanly. Auto-filter via:
```bash
echo "$code" | crystal build --no-codegen -f main.cr  # syntax check
```

**Cost:** if synthetic, ~$50 in Claude API calls for 20K pairs

### 3. DPO corpus: 1K+ preference pairs (1–2 days)

**Programmatic generation** from Ruby↔Crystal divergence rules:
- For each Ruby idiom (~30 of them), generate 30+ surface variants
- Chosen = Crystal idiomatic; rejected = direct Ruby translation

**Examples:**
- chosen: `property name : String` / rejected: `attr_accessor :name`
- chosen: `spawn { ... }` + `Channel(Int32).new` / rejected: `Thread.new { ... }` + `Mutex.new`
- chosen: `Array(Int32).new` / rejected: `Array<Int32>.new`

**Validation:** every `chosen` block must `crystal build`; every `rejected` must NOT (Ruby-isms = compile errors)

### 4. Hyperparams (the easy part)

**Hardware:** 4× A100-80GB or 2× H100. RunPod community: ~$8–14/hr

| stage | LR | epochs | LoRA r/α | duration |
|---|---|---|---|---|
| CPT | 2e-5 | 2 | 64/128 | 6–10 hrs |
| SFT | 5e-5 | 3 | 64/128 | 4–8 hrs |
| DPO | 5e-6 | 3 | 64/128, β=0.1 | 1–2 hrs |

**Total compute:** ~15–25 hours × $10/hr = **$150–250 in GPU time**

### 5. Eval gate (1 day to build properly)

Held-out test set: 100 Crystal questions the model has *never* seen during training.

**Metrics:**
- Compile pass rate of generated code (`crystal build`)
- Spec pass rate (does the code's spec actually run?)
- Idiom adherence: % of responses using `property` vs `attr_*`, `spawn` vs `Thread.new`, etc.
- Head-to-head similarity vs base on the held-out set

**Gate:** trained must beat base by ≥15% on compile pass rate. Otherwise don't publish.

## Realistic total cost

| component | time | $ |
|---|---|---|
| Data scraping + cleaning | 2 days | $0 (your time) |
| Synthetic SFT generation | 1 day + Claude API | ~$50 |
| DPO pair generation | 1 day | $0 (your time) |
| Compile-validation harness | 1 day | $0 |
| Multi-GPU training | 25 hrs | ~$200 |
| Eval harness build + run | 1 day | ~$10 in eval inference |
| **Total** | **6–8 days work** | **~$250–300** |

## Where it can go wrong

1. **Crystal corpus is small.** Total Crystal code on GitHub is <1% of Ruby/Python. You may cap out at 50M tokens. That's fine for a 30B model with LoRA, but you're not getting Llama-scale data.
2. **Compile-gate filtering is slow.** Running `crystal build` on 20K SFT examples takes hours. Parallelize aggressively.
3. **MoE expert LoRA is still suboptimal.** The current setup broadcasts the same delta to all 128 experts. For really specialized capability you'd want per-expert LoRAs, which 5×'s the LoRA parameter count.
4. **Qwen3-Coder may already be at the ceiling.** If the base already gets 80% of Crystal idioms right, even a great training run only buys you the last 20%. Diminishing returns.
5. **Crystal evolves.** Every new compiler release deprecates idioms. Your model will rot in 6–12 months.

## Honest take

Option C is a real engineering project — 1 week of focused work + $250 in compute. Worth it only if:
- You actually need a Crystal-specialized model (you write Crystal full-time)
- You're willing to maintain it as Crystal evolves
- You want to learn the multi-stage training pipeline well enough to apply it to other niches

Otherwise: Option A (system-prompted `qwen3-coder:30b`) gets you 90% of the way for $0 and zero maintenance.

## Recommended order if you do this

1. **Build the eval harness FIRST** — held-out test set + compile-gate. You need this to know if any of it worked.
2. Run the eval against `qwen3-coder:30b` baseline. Whatever score it gets is the bar to clear.
3. Build dataset (CPT + SFT + DPO).
4. Train.
5. Re-run eval. If it doesn't beat baseline by ≥15%, do not publish — diagnose what went wrong.

## Quick-start: data scraping skeleton

```bash
# Top Crystal repos by stars
gh search repos --language=crystal --sort=stars --limit=500 \
  --json fullName,stargazerCount > crystal_repos.json

# Clone all of them shallow
mkdir -p crystal_corpus_raw
jq -r '.[].fullName' crystal_repos.json | while read repo; do
  git clone --depth=1 "https://github.com/$repo" "crystal_corpus_raw/${repo//\//_}" 2>/dev/null
done

# Walk for .cr files, dedup by SHA, write JSONL
python3 build_cpt_corpus.py crystal_corpus_raw cpt_corpus_v2.jsonl
```

Total expected output: ~50–100M tokens after dedup. Enough to actually move the needle on a 30B model.
