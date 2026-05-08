# Crystal LoRA Training — Together AI

## Status

- [x] Scaffold scripts (forked from jerboa-lora)
- [x] Generate training data (5,430 entries from stdlib doc-comments, spec, samples, divergence, convention, docs)
- [ ] Upload training data to Together AI
- [ ] Start fine-tuning job (Qwen3-Coder-30B-A3B-Instruct, LoRA r=16, α=32, 1 epoch)
- [ ] Wait for training to complete (~10–15 min expected)
- [ ] Deploy to RunPod serverless (`./deploy_runpod.sh jaimef21/crystal-qwen-30b`)
- [ ] Verify model with `verify_model.py`
- [ ] Connect to OpenCode (`./configure_opencode.sh runpod <ENDPOINT_ID>`)
- [ ] (Optional) Download merged model and run via Ollama locally
- [ ] (Optional) Push to Ollama registry (`./push_ollama.sh jaimef`)

---

## Training Data

Generated **5,430 training entries** in `~/mine/crystal-lora/`:

| File                           | Format                 | Size   |
|--------------------------------|------------------------|--------|
| `training_data_together.jsonl` | Together AI (messages) | 9.4 MB |
| `training_data.jsonl`          | ChatML/ShareGPT        | 9.6 MB |
| `training_data_alpaca.jsonl`   | Alpaca JSONL           | 5.1 MB |

### Source breakdown

| Source | Count |
|--------|-------|
| stdlib | 4,804 |
| spec | 270 |
| divergence | 120 |
| sample | 66 |
| convention | 57 |
| doc | 36 |
| mcp-doc | 25 |
| changelog | 21 |
| man | 13 |
| mcp-spec | 10 |
| mcp-src | 8 |

Regenerate: `python3 convert_training_data.py`

---

## Step 1: Setup

```bash
pip install together
export TOGETHER_API_KEY="your-key-here"   # or save to ~/.together-ai.token
```

## Step 2: Upload

```bash
python3 train_together.py upload
```

Saves `file_id` to `.together_state.json`.

## Step 3: Train

```bash
python3 train_together.py train
```

Saves `job_id` to `.together_state.json`. Training settings:
- LoRA r=16, alpha=32
- 1 epoch, learning rate 1e-5, batch size 2
- Base model: `Qwen/Qwen3-Coder-30B-A3B-Instruct` (30B MoE, 3B active)

Single epoch is intentional — at 5,430 examples and a 30B base, more epochs risk overfitting on the synthetic divergence/convention triples.

## Step 4: Wait & Status

```bash
python3 train_together.py status
```

When done, the model name (e.g. `jaimef_xxxx/Qwen3-Coder-30B-A3B-Instruct-yyyyyyyy`) is saved to state.

---

## Step 5: Deploy — Choose Your Option

### Option A: RunPod Serverless (recommended — 30B is heavy for local, ~$0.69/hr active, $0 idle)

```bash
export RUNPOD_API_KEY="your-key"
hf auth login
./deploy_runpod.sh jaimef21/crystal-qwen-30b
```

The script auto-reads JOB_ID from `.together_state.json`, downloads + merges the adapter, uploads the merged model to HuggingFace, then provisions a 48GB-GPU vLLM endpoint.

Configure OpenCode:
```bash
./configure_opencode.sh runpod <ENDPOINT_ID>
```

### Option B: Local Ollama (free, needs 24GB+ GPU for usable speed)

```bash
./download_and_convert.sh
```

This downloads the LoRA adapter, merges it into the base, converts to GGUF (q4_k_m by default), and registers `crystal-qwen` with Ollama.

Or pull from the registry once published:
```bash
ollama pull jaimef/crystal-qwen
```

Configure OpenCode:
```bash
./configure_opencode.sh ollama
```

### Option C: Local Unsloth training (free, needs 24GB+ GPU, smaller base)

If you want to iterate quickly on a smaller model first:
```bash
python3 train_unsloth.py        # → ./crystal-lora-output/   (uses Qwen2.5-Coder-7B as base)
python3 merge_and_export.py     # → ./crystal-qwen-gguf/
ollama create crystal-qwen -f Modelfile
```

This is a 7B fallback — quality will be lower than the Together AI 30B run, but it's useful for catching data-formatting bugs before paying for the big training job.

---

## Verification

```bash
# Local
python3 verify_model.py --base-url http://localhost:11434/v1 --model crystal-qwen -v

# RunPod
python3 verify_model.py \
  --base-url https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1 \
  --model jaimef21/crystal-qwen-30b \
  --api-key $RUNPOD_API_KEY -v
```

10 test cases covering: type-annotated properties, `attr_accessor` → `property`, fibers + Channels, `JSON::Serializable`, `HTTP::Server`, the `Spec` framework, `(T)` generic syntax, `Tuple` vs `NamedTuple`, FFI via `lib`/`fun`, nilable types (`Nil`/`?`).

---

## Iteration

To improve quality:
1. Add Crystal idioms to the `CONVENTION_EXAMPLES` list in `convert_training_data.py`
2. Add Ruby → Crystal divergence entries for any new hallucinations you catch
3. `python3 convert_training_data.py`
4. `python3 train_together.py upload && python3 train_together.py train`
5. Redeploy
