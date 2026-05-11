# Crystal LoRA

A fine-tuned **Qwen3-Coder-30B-A3B-Instruct** LoRA that knows the Crystal programming language. Trained as a staged **CPT → SFT → DPO** pipeline on real Crystal source from the top 500 GitHub repos, the official book + RFCs + website, the stdlib doc-comments, the spec suite, and hand-curated Ruby→Crystal preference pairs. Trained on RunPod (single A100 80GB), shipped as **GGUF on Hugging Face (hf.co)**, run locally via Ollama.

Crystal is statically-typed, Ruby-syntax-inspired, and compiles via LLVM. This LoRA teaches the base model Crystal's actual surface area: type unions (`String?`), generics with `(T)` (not `<T>`), `property`/`getter`/`setter` (not `attr_accessor`), fibers and `Channel(T)` (not threads), the `Spec` framework (not RSpec), `JSON::Serializable`, `HTTP::Server`, FFI via `lib`/`fun`, macros, and the dozens of small ways Crystal diverges from Ruby despite the syntactic resemblance.

## Status (v3 retrain in flight — 2026-05-09)

The original v1 training (37 DPO pairs, ~9 MB CPT, lr=5e-7 DPO) was **a no-op in the trained weights** — the DPO LoRA delta sat below the BF16 precision floor, and Q4_K_M quantization erased the rest. See [`EVAL_VERDICT.md`](EVAL_VERDICT.md) for the post-mortem and [`TRAINING_FIX_PLAN.md`](TRAINING_FIX_PLAN.md) for the corrective plan now being executed.

What changed for v3:

| Stage | v1 (failed) | v3 (now)                          | Why |
|-------|-------------|-----------------------------------|-----|
| CPT data    | 1,750 files / ~9 MB | **31,292 records / ~31.6M tok**   | Top 500 GitHub repos + book/RFCs/website/stdlib docs |
| SFT data    | 5,430 pairs | **9,958 pairs** (mined + Claude-augmented, all compile-gated) | +83% bigger, end-to-end compile-validated |
| DPO data    | 37 pairs    | **374 pairs** (235 compile-validated) | Ruby→Crystal rule generators × ~50 idioms |
| CPT lr      | 5e-6, 1 ep  | **2e-5, 2 ep**                    | v1 was 10× too low |
| DPO lr      | 5e-7, 1 ep  | **5e-6, 3 ep**                    | v1 delta was below BF16 floor |
| LoRA r/α    | 16 / 32     | **64 / 128**                      | Bigger delta → survives quantization |
| Quantization| Q4_K_M      | **Q8_0 (target)**                 | LoRA contributions actually survive |
| Publish     | ~~ollama.com~~ | **hf.co** | User correction: we publish to Hugging Face, not the Ollama registry |

After running the corrective hyperparams + the v2 (intermediate) data, the trained model **already beats base** on both the full 74-pair similarity eval (`+0.357` vs base `+0.124` token-lean to chosen) and the held-out coding eval (`36` vs `9` total score). The v3 expansion is intended to widen that margin and reduce regression on out-of-distribution prompts.

## Quick Start (once v3 is published)

```bash
# Pull GGUF from Hugging Face (NOT ollama.com)
hf download jaimef/crystal-qwen-gguf crystal-qwen.Q8_0.gguf --local-dir .
ollama create crystal-qwen -f Modelfile        # Modelfile: FROM ./crystal-qwen.Q8_0.gguf + Crystal SYSTEM block
ollama run crystal-qwen "How do I parse JSON into a typed class in Crystal?"
```

## Deployment Options

| Option | Cost | Speed | Setup |
|--------|------|-------|-------|
| **Local Ollama (merged GGUF, 24GB+ GPU)** | Free | 20–30 tok/s | Pull from hf.co, `ollama create … -f Modelfile` |
| **RunPod Serverless (48GB GPU)** | $0 idle, ~$0.69/hr active | 25–35 tok/s | `./deploy_runpod.sh` (uploads merged HF checkpoint, creates vLLM endpoint) |
| **Local Ollama (CPU)** | Free | 2–5 tok/s | Same; not recommended for 30B |

The 30B MoE has 3B active parameters at inference time, so latency is closer to a 3B dense model — but it still needs ~40 GB VRAM (Q8_0) to load weights, hence the 48 GB-GPU target for serverless.

## Use with OpenCode

### Local Ollama
```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ollama": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Ollama (local)",
      "options": { "baseURL": "http://localhost:11434/v1" },
      "models": { "crystal-qwen": { "name": "Crystal Qwen" } }
    }
  }
}
```
Or run `./configure_opencode.sh ollama`.

### RunPod Serverless
```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "runpod": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "RunPod (serverless)",
      "options": {
        "baseURL": "https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1",
        "apiKey": "<RUNPOD_API_KEY>"
      },
      "models": { "crystal-qwen": { "name": "Crystal Qwen 30B" } }
    }
  }
}
```

## Reproducing v3 from scratch

Want to rebuild the v3 model end-to-end (fetch every dataset, train, eval-gate, publish)? See **[REPRODUCE.md](REPRODUCE.md)** — full step-by-step with prerequisites, cost estimates (~$60-90 of pod time), expected metrics, and recovery procedures for the failure modes we actually hit.

The short version is one bootstrap command followed by the steps below:
```bash
./fetch_training_sources.sh    # clones Crystal stdlib + book + RFCs + website + awesome-crystal
```

## Build from Source — staged CPT → SFT → DPO

Training runs as a **staged CPT → SFT → DPO pipeline** on a single A100 80GB pod, orchestrated by `runpod_train.py`. LoRA adapters from each stage are merged into the base before the next stage trains, so DPO sees the SFT-absorbed weights, not stacked adapters. All three stages target attention + MLP + MoE experts (`experts.gate_up_proj`, `experts.down_proj`) at bf16.

| Stage | Data (v3) | LR | Epochs | LoRA r/α |
|-------|-----------|----|--------|----------|
| **1. CPT** — token-distribution shift | `cpt_corpus_v3_merged.jsonl` (31,292 docs / ~31.6M tok) | 2e-5 | 2 | 64 / 128 |
| **2. SFT** — chat-format Q/A | `sft_v3_mined.jsonl` (compile-gated, mined from stdlib doc-comments + READMEs + spec describe/it blocks) | 1e-4 | 2 | 64 / 128 |
| **3. DPO** — actively suppresses Ruby/RSpec/Java hallucinations | `dpo_pairs_v3.jsonl` (374 Ruby→Crystal preference triples, 235 compile-validated) | 5e-6 | 3 | 64 / 128 |

### 1. Generate the three v3 datasets

```bash
# CPT: scrape + dedupe + merge
python3 build_cpt_corpus_v2.py        # → cpt_corpus_v3.jsonl  (top 500 GitHub Crystal repos, SHA-deduped)
python3 build_cpt_docs.py             # → cpt_docs.jsonl       (book + RFCs + website + awesome-crystal + stdlib)
python3 merge_cpt_v3.py               # → cpt_corpus_v3_merged.jsonl  (cross-file SHA dedup)

# SFT: mine from real code (compile-gated). Optional LLM augmenter as a backup.
python3 build_sft_v3.py               # → sft_v3_mined.jsonl   (stdlib doc-comments + READMEs + specs, parallel compile-validated)
python3 build_sft_llm.py --files 2000 # → sft_v3_llm.jsonl     (Claude Haiku 4.5 via OpenRouter, ~$15–20)

# DPO: programmatic Ruby→Crystal pair generators (~50 idioms × multiple variants), compile-validated
python3 build_dpo_pairs_v3.py         # → dpo_pairs_v3.jsonl   (374 pairs, 235 validated)
```

### 2. Train on RunPod (~A100 80GB, single pod)

```bash
pip install runpod
echo "your-runpod-key" > ~/.runpod.token

python3 runpod_train.py up      # provision A100-SXM4 80GB, wait for SSH
python3 runpod_train.py push    # scp axolotl YAMLs + datasets
python3 runpod_train.py train   # CPT → merge → SFT → merge → DPO; idempotent (resumes via .done markers)
python3 runpod_train.py pull    # fetch /workspace/dpo_merged → ./runpod-pipeline-final/
python3 runpod_train.py down    # terminate the pod
```

`runpod_train.py status` shows pod state and per-stage `.done` markers. `runpod_train.py tail` follows the active log. Each stage runs inside its own tmux session on the pod, so SSH drops don't kill training.

### 3. Convert merged checkpoint → GGUF

```bash
# llama.cpp conversion (NOT convert_to_mlx.sh — that's for the sibling jerboa-lora project)
python3 llama.cpp/convert_hf_to_gguf.py runpod-pipeline-final/ --outfile crystal-qwen.gguf
./llama.cpp/llama-quantize crystal-qwen.gguf crystal-qwen.Q8_0.gguf Q8_0
```

### 4. Eval gate (mandatory before publishing)

```bash
# Held-out coding eval — questions the model never saw during training
python3 eval_holdout.py --models crystal-qwen qwen3-coder:30b --out eval_holdout.json

# Full 74-pair similarity eval — token-lean toward chosen vs rejected
python3 eval_similarity.py --models crystal-qwen qwen3-coder:30b --out eval_similarity_full.json
```

Required to publish: trained must beat base on both metrics. The v2 run already cleared this bar (`+0.357` vs `+0.124` tok-lean; `36` vs `9` held-out total) — v3 should widen the margin further.

### 5. Publish to Hugging Face (NOT ollama.com)

```bash
hf auth login
hf upload jaimef/crystal-qwen-gguf crystal-qwen.Q8_0.gguf
# Optionally upload the merged HF checkpoint as source:
hf upload jaimef/crystal-qwen-30b runpod-pipeline-final/
```

## Datasets — v3

### CPT corpus (`cpt_corpus_v3_merged.jsonl`) — 31,292 records, ~31.6M tokens

- **29,633 `.cr` source files** scraped from the top 500 Crystal GitHub repos (`gh search repos --language=crystal --sort=stars --limit=500` then shallow clone). `compiler/`, `llvm/`, vendored shards, and generated code are filtered out.
- **1,659 documentation entries** — Crystal book chapters, RFCs, website content, awesome-crystal entries, and stdlib `.md`/`.cr` doc files.
- One JSON record per file, `text` field with a `# FILE: <path>` header so the model learns that imports cluster with file structure.
- SHA-dedup both within and across the two corpora (the GitHub scrape pulls `crystal-lang/crystal` whose `src/` overlaps with the doc harvest).

### SFT dataset (`sft_v3_mined.jsonl`) — 9,958 pairs, all compile-gated

Four sources, all compile-validated in parallel via `crystal build --no-codegen`:

1. **Stdlib doc-comments** (`build_sft_v3.py`) — 1,774 pairs from `# ```` examples preceding `def`s in `~/mine/crystal/src/**/*.cr`. Question templated 5 ways for diversity, 2 emitted per def.
2. **README usage sections** (`build_sft_v3.py`) — 9 pairs from code blocks under "Usage"/"Example" headers in cloned repo READMEs (most fail compile-gate due to shard deps).
3. **Spec describe/it blocks** (`build_sft_v3.py`) — extracted via line-by-line do/end depth tracking, currently 0 kept (specs need project context for compile, gate over-rejects — known limitation).
4. **LLM-augmented from real source** (`build_sft_llm.py`) — 4,054 pairs generated by Claude Haiku 4.5 (via OpenRouter) from 3,000 sampled corpus files. Asked for 2 Q/A per file, ~64% pass compile-gate.
5. **Merged with v1's 5,430 hand/scripted pairs** (1,348 dedup'd).

Compile-gate is **lenient on missing third-party shards** (treated as a dependency issue, not a Crystal error) but strict on `syntax error`, `undefined method`, `undefined constant`, `expected … but`. This keeps real-world examples that import absent shards while still rejecting Claude/Ruby slop.

### DPO pairs (`dpo_pairs_v3.jsonl`) — 374 pairs, 235 compile-validated

74 hand-curated + 187 from v2 rule generators + 113 from new v3 rule generators covering ~50 Ruby idioms: `attr_accessor` → `property`, `Thread.new` → `spawn`, `Class<T>` → `Class(T)`, `require "rspec"` → `require "spec"`, `extern "C"` → `lib LibC ... fun`, untyped `[]` → `[] of T`, `JSON.parse` → `from_json`, `Mutex.new` → `Mutex.new` (Crystal-style), `Queue.new` → `Channel(T).new`, `Pathname` → `Path`, `URI.parse`, `Base64`, `Digest::MD5`, `YAML::Serializable`, `DB.query`, `HTTP::Client.get/post`, `Comparable`, `Enumerable#tally/group_by/each_with_object/partition`, etc. Each entry is `{system, instruction, chosen_response, rejected_response}` in axolotl `chatml.argilla` format. Compile validation runs the **chosen** code and rejects the pair if it fails (avoids imprinting Claude's mistakes).

## Scripts

| Script | Purpose |
|--------|---------|
| `build_cpt_corpus_v2.py` | Scrape top 500 Crystal GitHub repos → `cpt_corpus_v3.jsonl` |
| `build_cpt_docs.py` | Harvest Crystal book + RFCs + website + awesome-crystal + stdlib docs → `cpt_docs.jsonl` |
| `merge_cpt_v3.py` | SHA-dedupe and concat → `cpt_corpus_v3_merged.jsonl` |
| `build_sft_v3.py` | Mine SFT pairs from stdlib doc-comments + READMEs + spec blocks; parallel compile-gate |
| `build_sft_llm.py` | LLM-augment SFT via Claude Haiku 4.5 (OpenRouter); compile-gated |
| `build_dpo_pairs_v2.py` | First wave of programmatic Ruby→Crystal pair generators |
| `build_dpo_pairs_v3.py` | Imports v2 + adds 32 new rule generators; compile-validates chosen blocks |
| `fetch_training_sources.sh` | Clone every upstream source the build_*.py scripts read (stdlib + Crystal book + RFCs + website + awesome-crystal). Idempotent. |
| `runpod_train.py` | Provision A100 pod, run staged CPT → SFT → DPO, pull merged model |
| `axolotl_crystal_{cpt,sft,dpo}.yaml` | Per-stage axolotl configs (LoRA r=64/α=128, MoE-expert targets, bf16) |
| `eval_holdout.py` | Held-out Crystal coding eval (idiom regex hits + compile pass rate) |
| `eval_similarity.py` | Per-pair similarity to chosen vs rejected — volume-invariant, **trust this one** |
| `eval_crystal.py` | 10 hand-crafted Crystal-vs-Ruby divergence prompts (verbosity-confounded; deprecated) |
| `eval_dpo_preference.py` | Crystal/Ruby idiom-hit counter (verbosity-confounded; deprecated) |
| `merge_lora_local.py` | Local streaming LoRA merge of the pulled adapter |
| `merge_and_export.py` | Merge adapter + base to GGUF |
| `download_and_convert.sh` | Pull GGUF from hf.co, register with Ollama |
| `deploy_runpod.sh` | Upload merged checkpoint to hf.co, create RunPod vLLM endpoint |
| `manage_runpod.sh` | RunPod endpoint lifecycle (list, health, delete, purge, restore) |
| `configure_opencode.sh` | Generate OpenCode config for Ollama/RunPod |
| `verify_model.py` | Run 10 Crystal-specific test prompts |
| `convert_to_mlx.sh` | **Do not run** — leftover scaffolding from sibling `jerboa-lora` (MLX path), not used here |
| `Modelfile` | Ollama model definition (FROM ./crystal-qwen.Q8_0.gguf + SYSTEM block + TEMPLATE) |

## Iterating

1. Add a new Ruby→Crystal idiom: edit `build_dpo_pairs_v3.py` (add a `rule_*` generator) → `python3 build_dpo_pairs_v3.py`
2. Add more SFT data: extend the source list in `build_sft_v3.py` or up `--files` on `build_sft_llm.py`
3. Re-train: `runpod_train.py up && push && train && pull && down`
4. **Eval gate**: `eval_holdout.py` and `eval_similarity.py` against base; trained must beat base
5. Convert + publish: `llama.cpp/convert_hf_to_gguf.py` → `llama-quantize Q8_0` → `hf upload`

## Why a Crystal-specific model?

Crystal looks like Ruby but is not Ruby. Out-of-the-box LLMs constantly slip Ruby (and sometimes Java/C#) into "Crystal" answers:

- `attr_accessor :name` instead of `property name : String`
- `Thread.new { ... }` instead of `spawn { ... }`
- `class Box<T>` instead of `class Box(T)`
- `require "rspec"` instead of `require "spec"`
- `String | Nil` written as `Optional[String]` or just dropped
- `JSON.parse(s).as(MyClass)` instead of `MyClass.from_json(s)` (with `include JSON::Serializable`)
- `puts foo.length if foo` without realising `foo` is `String?` and needs `if foo = foo` or `try`
- `extern "C"` blocks instead of `lib LibC ... fun ... end`
- Method-name guessing (`each_with_object` exists; `inject_with_index` doesn't)

The v3 staged pipeline trains the model on what Crystal actually is — 31.6M tokens of real source, compile-gated SFT pairs, and 374 DPO pairs that actively *push the model away* from each Ruby-ism — so generated code parses, type-checks, and runs.

## Sister project

The sibling `jerboa-lora` project trains a similar LoRA but ships via **MLX** (Apple Silicon path). Crystal-LoRA ships via **GGUF/Ollama** and publishes to **hf.co** — do not confuse the runtimes. Files like `convert_to_mlx.sh` exist in this repo from earlier scaffolding; they are not part of the live pipeline.
