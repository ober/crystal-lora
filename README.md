# Crystal LoRA

A fine-tuned Qwen3-Coder-30B model that knows the Crystal programming language. Trained as a staged **CPT → SFT → DPO** pipeline on Crystal source, doc-comments, spec examples, and hand-curated Ruby→Crystal preference pairs. Run on RunPod, host with Ollama (local merged GGUF), or pull from the registry.

Crystal is a statically-typed, Ruby-syntax-inspired language compiled via LLVM. This LoRA teaches the base model Crystal's actual surface area: type unions (`String?`), generics with `(T)` (not `<T>`), `property`/`getter`/`setter` (not `attr_accessor`), fibers and `Channel(T)` (not threads), the `Spec` framework (not RSpec), `JSON::Serializable`, `HTTP::Server`, FFI via `lib`/`fun`, macros, and the dozens of small ways Crystal diverges from Ruby despite the syntactic resemblance.

## Quick Start (Local merged model)

```bash
ollama pull jaimef/crystal-qwen
ollama run jaimef/crystal-qwen "How do I parse JSON into a typed class in Crystal?"
```

## Deployment Options

| Option | Cost | Speed | Setup |
|--------|------|-------|-------|
| **RunPod Serverless (48GB GPU)** | $0 idle, ~$0.69/hr active | 25–35 tok/s | `./deploy_runpod.sh` |
| **Local Ollama (merged GGUF, 24GB+ GPU)** | Free | 20–30 tok/s | `./download_and_convert.sh` |
| **Local Ollama (CPU)** | Free | 2–5 tok/s | Same; not recommended for 30B |
| **Together AI Endpoint** | Always-on, ~$8/hr | Fast | Not recommended |

### Estimated RunPod monthly costs (48GB GPU, scale-to-zero)

| Usage | Hours/month | Cost/month |
|-------|-------------|------------|
| Idle (scale-to-zero) | 0 | **$0** |
| Light (1hr/day) | ~30 | **~$21** |
| Moderate (3hr/day) | ~90 | **~$62** |
| Heavy (8hr/day) | ~240 | **~$166** |

The 30B MoE has 3B active parameters at inference time, so latency is closer to a 3B dense model — but it still needs ~40GB of VRAM to load weights, which is why we target a 48GB GPU.

## Use with OpenCode

### Option A: Local Ollama

Add to `~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ollama": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Ollama (local)",
      "options": {
        "baseURL": "http://localhost:11434/v1"
      },
      "models": {
        "crystal-qwen": {
          "name": "Crystal Qwen"
        }
      }
    }
  }
}
```

Or run `./configure_opencode.sh ollama` to write that for you.

### Option B: RunPod Serverless (recommended — 30B is heavy for local)

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
      "models": {
        "crystal-qwen": {
          "name": "Crystal Qwen 30B"
        }
      }
    }
  }
}
```

Replace `<ENDPOINT_ID>` and `<RUNPOD_API_KEY>` with your values.

## Deploy to RunPod

One script handles everything: downloads merged model, uploads to HuggingFace, creates a 48GB-GPU RunPod endpoint via API.

```bash
# Prerequisites
pip install together huggingface_hub
export TOGETHER_API_KEY="your-key"
export RUNPOD_API_KEY="your-key"
hf auth login

# Deploy (after training completes)
./deploy_runpod.sh jaimef21/crystal-qwen-30b
```

The script reads the Together AI job ID from `.together_state.json` (set by `train_together.py train`), so you don't need to paste it in.

Endpoint URL: `https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1`

## Build from Source

Training runs as a **staged CPT → SFT → DPO pipeline** on a single A100 80GB pod, orchestrated by `runpod_train.py`. Each stage addresses a different gap that single-stage SFT alone leaves open:

| Stage | What it does | Data | LR | Epochs |
|-------|--------------|------|-----|--------|
| **1. CPT** (Continued Pre-Training) | Teaches the model that Crystal source code exists — the token distribution, which `require`s cluster with which idioms, what the stdlib actually looks like in the wild | `cpt_corpus.jsonl` (raw `.cr` / `.md` from `~/mine/crystal` and `~/mine/crystal-mcp`) | 5e-6 | 1 |
| **2. SFT** (Supervised Fine-Tuning) | Teaches the model to *answer* Crystal questions in chat format | `training_data_together.jsonl` (5,430 Q/A pairs from doc-comments, specs, conventions, divergence) | 1e-4 | 2 |
| **3. DPO** (Direct Preference Optimization) | Actively suppresses Ruby/RSpec/Java hallucinations — DPO pushes log P(chosen) up and log P(rejected) down, which plain SFT can't do | `dpo_pairs.jsonl` (37 hand-curated Ruby→Crystal preference triples) | 5e-7 | 1 |

LoRA adapters from each stage are merged into the base before the next stage trains, so DPO sees the SFT-absorbed weights, not stacked adapters. All three stages target attention + MLP + MoE experts (`experts.gate_up_proj`, `experts.down_proj`) at bf16.

### 1. Generate the three datasets

```bash
# These should already exist on your machine:
#   ~/mine/crystal       (Crystal compiler + stdlib source)
#   ~/mine/crystal-mcp   (Crystal MCP server)

python3 convert_training_data.py   # → training_data*.jsonl (5,430 SFT entries)
python3 build_cpt_corpus.py        # → cpt_corpus.jsonl     (~9 MB raw source)
python3 build_dpo_pairs.py         # → dpo_pairs.jsonl      (37 Ruby→Crystal pairs)
```

### 2. Train on RunPod (~$11, ~5–6 hours, A100 80GB)

```bash
pip install runpod
echo "your-runpod-key" > ~/.runpod.token

python3 runpod_train.py up      # provision A100-SXM4 80GB pod, wait for SSH
python3 runpod_train.py push    # scp all three configs + datasets
python3 runpod_train.py train   # run CPT → SFT → DPO; idempotent (resumes on rerun)
python3 runpod_train.py pull    # fetch /workspace/dpo_merged → ./runpod-pipeline-final/
python3 runpod_train.py down    # terminate the pod
```

`runpod_train.py status` shows pod state and per-stage `.done` markers. `runpod_train.py tail` follows the active log. Each stage runs inside its own tmux session on the pod, so SSH drops don't kill training.

#### Alternative: single-stage SFT on Together AI (~$5–8, ~10–15 minutes)

Faster and cheaper, but Together's hosted whitelist is attention-only — no MoE experts, no gate/up/down. The model still learns surface forms but the fix is shallower than CPT + SFT + DPO. Use this for quick iteration, the staged pipeline for the production model.

```bash
pip install together
export TOGETHER_API_KEY="your-key"   # or write it to ~/.together-ai.token

python3 train_together.py upload
python3 train_together.py train
python3 train_together.py status
```

### 3. Deploy

**Hosted (RunPod serverless, recommended):**
```bash
export RUNPOD_API_KEY="your-key"
hf auth login
./deploy_runpod.sh YOUR_USERNAME/crystal-qwen-30b
```

**Local (with 24GB+ GPU):**
```bash
./download_and_convert.sh
```

### 4. Verify

```bash
python3 verify_model.py \
  --base-url http://localhost:11434/v1 \
  --model crystal-qwen -v
```

### 5. Push to Ollama registry

```bash
./push_ollama.sh YOUR_USERNAME
```

## Training Data

Three datasets, one per pipeline stage.

### CPT corpus (`cpt_corpus.jsonl`)

Raw `.cr` / `.md` files from `~/mine/crystal` and `~/mine/crystal-mcp`, one JSON record per file with a `# FILE: <path>` header so the model learns that imports cluster with file structure. `compiler/`, `llvm/`, and `lib/` (vendored shards) are skipped — we want idiomatic user code, not LLVM bindings or AST internals. Roughly 1,750 files, ~9 MB.

### SFT dataset (`training_data_together.jsonl`)

**5,430 entries** generated from the Crystal source tree, the standard-library doc-comments, the spec suite, and hand-written convention/divergence examples.

| Source | Count | Description |
|--------|-------|-------------|
| stdlib | 4,804 | Doc-comments from `~/mine/crystal/src/**/*.cr` (everything except `compiler/` and `llvm/`) |
| spec | 270 | Verified-correct usage from `~/mine/crystal/spec/std/**/*_spec.cr` |
| divergence | 120 | Ruby → Crystal "wrong → right" pairs (40 entries × 3 phrasing variants) |
| sample | 66 | Example programs from `~/mine/crystal/samples/` |
| convention | 57 | 19 hand-written Crystal-idiom teaching examples (3× weighted) |
| doc | 36 | `~/mine/crystal/docs/**/*.md` chunked by section |
| mcp-doc | 25 | `~/mine/crystal-mcp/docs/**/*.md` |
| changelog | 21 | Recent entries from `CHANGELOG.md` |
| man | 13 | `~/mine/crystal/man/crystal.1` sections |
| mcp-spec | 10 | `~/mine/crystal-mcp/spec/**/*_spec.cr` examples |
| mcp-src | 8 | `~/mine/crystal-mcp/src/**/*.cr` doc-comments |

### DPO pairs (`dpo_pairs.jsonl`)

37 hand-curated Ruby→Crystal preference triples covering the highest-impact divergences: `attr_accessor` → `property`, `Thread.new` → `spawn`, `Class<T>` → `Class(T)`, RSpec → `Spec`, `extern "C"` → `lib LibC ... fun`, untyped `[]` → `[] of T`, `&.` → `if x = ...`, etc. Each entry is `{system, instruction, chosen_response, rejected_response}` in axolotl `chatml.argilla` format.

### Output formats

| File | Format | Use with |
|------|--------|----------|
| `training_data_together.jsonl` | `{messages: [...]}` | Axolotl chat_template, Together AI fine-tuning |
| `training_data.jsonl` | ChatML/ShareGPT | LLaMA-Factory, Unsloth |
| `training_data_alpaca.jsonl` | Alpaca JSONL | Unsloth, HuggingFace |
| `cpt_corpus.jsonl` | `{text: "..."}` | Axolotl `type: completion` |
| `dpo_pairs.jsonl` | `{system, instruction, chosen_response, rejected_response}` | Axolotl `type: chatml.default`, `rl: dpo` |

## Scripts

| Script | Purpose |
|--------|---------|
| `convert_training_data.py` | Generate SFT data from `~/mine/crystal` + `~/mine/crystal-mcp` |
| `build_cpt_corpus.py` | Walk Crystal source for `.cr`/`.md`, emit CPT corpus |
| `build_dpo_pairs.py` | Emit hand-curated Ruby→Crystal DPO preference pairs |
| `runpod_train.py` | Provision A100 pod, run staged CPT → SFT → DPO, pull merged model |
| `axolotl_crystal_{cpt,sft,dpo}.yaml` | Per-stage axolotl configs (LoRA, MoE-expert targets, bf16) |
| `train_together.py` | Single-stage SFT on Together AI (alternative to RunPod pipeline) |
| `download_and_convert.sh` | Download adapter, convert to GGUF, set up Ollama |
| `deploy_runpod.sh` | Upload merged model to HuggingFace, create RunPod endpoint |
| `manage_runpod.sh` | RunPod endpoint lifecycle (list, health, delete, purge, restore) |
| `push_ollama.sh` | Tag and push model to Ollama registry |
| `configure_opencode.sh` | Generate OpenCode config for Ollama/RunPod |
| `verify_model.py` | Run 10 Crystal-specific test prompts |
| `train_unsloth.py` | Local GPU training fallback (uses Qwen2.5-Coder-7B as base) |
| `merge_and_export.py` | Merge adapter + base to GGUF |
| `train_runpod.sh` | One-shot training on a rented GPU |
| `Modelfile` | Ollama model definition |

## Iterating

To improve the model with more training data:

1. Edit the source datasets:
   - **Convention/divergence Q/A** — `CONVENTION_EXAMPLES` and `DIVERGENCE_ENTRIES` in `convert_training_data.py`
   - **DPO Ruby→Crystal pairs** — `PAIRS` in `build_dpo_pairs.py` (add the wrong→right pair when you catch a new Ruby-ism)
   - **CPT corpus** — pull new code into `~/mine/crystal-mcp` (or upstream Crystal); the next `build_cpt_corpus.py` run picks it up
2. Regenerate:
   ```bash
   python3 convert_training_data.py
   python3 build_cpt_corpus.py
   python3 build_dpo_pairs.py
   ```
3. Retrain:
   - **Full pipeline:** `python3 runpod_train.py up && python3 runpod_train.py push && python3 runpod_train.py train && python3 runpod_train.py pull`
   - **Quick iteration:** `python3 train_together.py upload && python3 train_together.py train`
4. Deploy: `./deploy_runpod.sh` (hosted) or `./download_and_convert.sh` (local)

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

The staged pipeline trains the model on what Crystal actually is, so generated code parses, type-checks, and runs:

- **CPT** — ~1,750 raw Crystal source files (~9 MB) so the model has actually *seen* Crystal in the wild, not just answered questions about it
- **SFT** — 4,804 stdlib doc-comments + 270 verified spec examples + 120 divergence Q/A + 57 weighted convention examples (5,430 total)
- **DPO** — 37 Ruby→Crystal preference pairs that actively *push the model away* from `attr_accessor`, `Thread.new`, RSpec, `Class<T>`, etc., instead of just showing the right form once
