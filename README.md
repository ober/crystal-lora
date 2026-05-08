# Crystal LoRA

A fine-tuned Qwen3-Coder-30B model that knows the Crystal programming language. Run on RunPod, host with Ollama (local merged GGUF), or pull from the registry.

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

### 1. Generate training data

```bash
# These should already exist on your machine:
#   ~/mine/crystal       (Crystal compiler + stdlib source)
#   ~/mine/crystal-mcp   (Crystal MCP server)

python3 convert_training_data.py
# → training_data.jsonl (~9.6MB, 5,430 entries)
```

### 2. Train on Together AI (~$5–8, ~10–15 minutes)

```bash
pip install together
export TOGETHER_API_KEY="your-key"   # or write it to ~/.together-ai.token

python3 train_together.py upload
python3 train_together.py train
python3 train_together.py status
```

The 30B MoE costs more than a 7B dense model to train, but only one epoch is needed at this scale.

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

### Output formats

| File | Format | Use with |
|------|--------|----------|
| `training_data_together.jsonl` | Together AI messages | Together AI fine-tuning |
| `training_data.jsonl` | ChatML/ShareGPT | LLaMA-Factory, Axolotl, Unsloth |
| `training_data_alpaca.jsonl` | Alpaca JSONL | Unsloth, HuggingFace |

## Scripts

| Script | Purpose |
|--------|---------|
| `convert_training_data.py` | Generate training data from `~/mine/crystal` + `~/mine/crystal-mcp` |
| `train_together.py` | Upload, train, and monitor on Together AI |
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

1. Add idioms to the `CONVENTION_EXAMPLES` list in `convert_training_data.py`
2. Add Ruby→Crystal traps to the `DIVERGENCE_ENTRIES` list when you spot new hallucinations
3. `python3 convert_training_data.py`
4. `python3 train_together.py upload && python3 train_together.py train`
5. `./deploy_runpod.sh` (hosted) or `./download_and_convert.sh` (local)

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

The **120 divergence entries** + **4,804 stdlib doc-comments** + **270 verified spec examples** + **57 weighted convention examples** train the model on what Crystal actually is, so generated code parses, type-checks, and runs.
# crystal-lora
