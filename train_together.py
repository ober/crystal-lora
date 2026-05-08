#!/usr/bin/env python3
"""
Train a Crystal-language LoRA on Together AI.

Prerequisites:
  pip install together
  export TOGETHER_API_KEY="your-key-here"

Usage:
  python3 train_together.py upload     # Upload training data
  python3 train_together.py train      # Start fine-tuning (after upload)
  python3 train_together.py status     # Check training status
  python3 train_together.py test       # Test the fine-tuned model
  python3 train_together.py config     # Print OpenCode config
"""

import sys
import os
import json
from pathlib import Path

# Auto-load API key from ~/.together-ai.token if env not set
if not os.environ.get("TOGETHER_API_KEY"):
    tok = Path.home() / ".together-ai.token"
    if tok.exists():
        os.environ["TOGETHER_API_KEY"] = tok.read_text().strip()

try:
    from together import Together
except ImportError:
    print("Install the Together SDK: pip install together")
    sys.exit(1)

# ── Config ──────────────────────────────────────────────────────────
BASE_MODEL      = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
TRAINING_FILE   = os.path.join(os.path.dirname(__file__), "training_data_together.jsonl")
VALIDATION_FILE = os.path.join(os.path.dirname(__file__), "mlx_data", "valid.jsonl")
DPO_FILE        = os.path.join(os.path.dirname(__file__), "hard_data", "crystal_hard_dpo.jsonl")
HARD_TRAIN_FILE = os.path.join(os.path.dirname(__file__), "hard_data", "crystal_mixed_sft_train.jsonl")
HARD_VALID_FILE = os.path.join(os.path.dirname(__file__), "hard_data", "crystal_mixed_sft_valid.jsonl")
STATE_FILE      = os.path.join(os.path.dirname(__file__), ".together_state.json")

LORA_R         = 64
LORA_ALPHA     = 128
EPOCHS         = 2
LEARNING_RATE  = 2e-5
BATCH_SIZE     = 2            # Together max for Qwen3-Coder-30B-A3B-Instruct
N_EVALS        = 4            # Run validation 4 times during training


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def cmd_upload():
    """Upload training (and validation) data to Together AI."""
    client = Together()
    state = load_state()

    print(f"Uploading training file: {TRAINING_FILE}")
    response = client.files.upload(file=TRAINING_FILE, purpose="fine-tune")
    state["file_id"] = response.id
    print(f"  → file_id = {response.id}")

    if os.path.exists(VALIDATION_FILE):
        print(f"Uploading validation file: {VALIDATION_FILE}")
        vresp = client.files.upload(file=VALIDATION_FILE, purpose="fine-tune")
        state["validation_file_id"] = vresp.id
        print(f"  → validation_file_id = {vresp.id}")
    else:
        print(f"(no validation file at {VALIDATION_FILE} — skipping)")

    save_state(state)
    print(f"\nState written to {STATE_FILE}")
    print(f"\nNext: python3 {sys.argv[0]} train")


def cmd_upload_hard():
    """Upload compiler-checked hard SFT and DPO data."""
    client = Together()
    state = load_state()

    print(f"Uploading hard SFT: {HARD_TRAIN_FILE}")
    resp = client.files.upload(file=HARD_TRAIN_FILE, purpose="fine-tune")
    state["hard_file_id"] = resp.id
    print(f"  hard_file_id = {resp.id}")

    if os.path.exists(HARD_VALID_FILE):
        print(f"Uploading hard validation: {HARD_VALID_FILE}")
        vresp = client.files.upload(file=HARD_VALID_FILE, purpose="fine-tune")
        state["hard_validation_file_id"] = vresp.id
        print(f"  hard_validation_file_id = {vresp.id}")

    print(f"Uploading DPO: {DPO_FILE}")
    dresp = client.files.upload(file=DPO_FILE, purpose="fine-tune")
    state["dpo_file_id"] = dresp.id
    print(f"  dpo_file_id = {dresp.id}")

    save_state(state)
    print("Next: train-hard, then train-dpo after SFT completes.")


def cmd_train():
    """Start a fine-tuning job."""
    client = Together()
    state = load_state()

    file_id = state.get("file_id")
    if not file_id:
        print("No file_id found. Run 'upload' first.")
        sys.exit(1)
    validation_file_id = state.get("validation_file_id")

    print(f"Starting fine-tune on {BASE_MODEL}")
    print(f"  Training file: {file_id}")
    if validation_file_id:
        print(f"  Validation file: {validation_file_id} (n_evals={N_EVALS})")
    print(f"  Epochs: {EPOCHS}, LR: {LEARNING_RATE}, Batch: {BATCH_SIZE}")
    print(f"  LoRA r={LORA_R}, alpha={LORA_ALPHA}")

    kwargs = dict(
        model=BASE_MODEL,
        training_file=file_id,
        n_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        lora=True,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
    )
    if validation_file_id:
        kwargs["validation_file"] = validation_file_id
        kwargs["n_evals"] = N_EVALS

    response = client.fine_tuning.create(**kwargs)

    job_id = response.id
    state["job_id"] = job_id
    save_state(state)

    print(f"\nJob started! ID: {job_id}")
    print(f"Saved to {STATE_FILE}")
    print(f"\nMonitor: python3 {sys.argv[0]} status")


def cmd_train_hard():
    """Start hard-data SFT."""
    client = Together()
    state = load_state()
    file_id = state.get("hard_file_id")
    if not file_id:
        print("No hard_file_id found. Run upload-hard first.")
        sys.exit(1)

    kwargs = dict(
        model=BASE_MODEL,
        training_file=file_id,
        n_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        lora=True,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        train_on_inputs=False,
    )
    if state.get("hard_validation_file_id"):
        kwargs["validation_file"] = state["hard_validation_file_id"]
        kwargs["n_evals"] = N_EVALS

    print(f"Starting hard SFT: epochs={EPOCHS}, lr={LEARNING_RATE}, r={LORA_R}")
    resp = client.fine_tuning.create(**kwargs)
    state["hard_job_id"] = resp.id
    save_state(state)
    print(f"hard_job_id = {resp.id}")


def cmd_train_dpo():
    """Start DPO from the hard SFT model."""
    client = Together()
    state = load_state()
    model = state.get("hard_model_name") or state.get("model_name")
    file_id = state.get("dpo_file_id")
    if not model or not file_id:
        print("Need hard_model_name/model_name and dpo_file_id. Run status-hard and upload-hard first.")
        sys.exit(1)

    print(f"Starting DPO from {model}")
    resp = client.fine_tuning.create(
        model=model,
        training_file=file_id,
        n_epochs=1,
        learning_rate=1e-5,
        batch_size=BATCH_SIZE,
        lora=True,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        training_method="dpo",
        dpo_beta=0.2,
    )
    state["dpo_job_id"] = resp.id
    save_state(state)
    print(f"dpo_job_id = {resp.id}")


def cmd_status_hard():
    client = Together()
    state = load_state()
    job_id = state.get("hard_job_id")
    if not job_id:
        print("No hard_job_id found.")
        sys.exit(1)
    resp = client.fine_tuning.retrieve(id=job_id)
    print(f"hard_job_id={job_id} status={resp.status}")
    if getattr(resp, "output_name", None):
        state["hard_model_name"] = resp.output_name
        save_state(state)
        print(f"hard_model_name={resp.output_name}")


def cmd_status():
    """Check fine-tuning job status."""
    client = Together()
    state = load_state()

    job_id = state.get("job_id")
    if not job_id:
        print("No job_id found. Run 'train' first.")
        sys.exit(1)

    response = client.fine_tuning.retrieve(id=job_id)

    print(f"Job: {job_id}")
    print(f"Status: {response.status}")

    if hasattr(response, "output_name") and response.output_name:
        state["model_name"] = response.output_name
        save_state(state)
        print(f"Model: {response.output_name}")
        print(f"\nReady! Test: python3 {sys.argv[0]} test")

    if hasattr(response, "events") and response.events:
        print("\nRecent events:")
        for event in response.events[-5:]:
            print(f"  {event}")


def cmd_test():
    """Test the fine-tuned model."""
    client = Together()
    state = load_state()

    model_name = state.get("model_name")
    if not model_name:
        print("No model_name found. Check 'status' — training may still be running.")
        sys.exit(1)

    test_prompts = [
        "How do I declare a class with type-annotated properties in Crystal?",
        "What's the Crystal equivalent of Ruby's attr_accessor?",
        "Show me how to spawn a fiber and use a Channel in Crystal.",
        "How do I parse JSON into a typed class in Crystal?",
        "How do I write a basic HTTP server in Crystal?",
        "Show me how Crystal macros differ from Ruby's method_missing.",
    ]

    for prompt in test_prompts:
        print(f"\n{'='*60}")
        print(f"Q: {prompt}")
        print(f"{'='*60}")

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert in Crystal, a statically-typed, "
                        "Ruby-syntax-inspired language compiled via LLVM. "
                        "Provide accurate, idiomatic Crystal code with correct "
                        "types, requires, and method signatures."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
            temperature=0.2,
        )

        print(response.choices[0].message.content)


def cmd_opencode_config():
    """Print OpenCode configuration for the fine-tuned model."""
    state = load_state()
    model_name = state.get("model_name")
    if not model_name:
        print("No model_name found yet. Train first.")
        sys.exit(1)

    print("Add to your OpenCode config:\n")
    config = {
        "provider": "openai-compatible",
        "base_url": "https://api.together.xyz/v1",
        "api_key": "${TOGETHER_API_KEY}",
        "model": model_name,
    }
    print(json.dumps(config, indent=2))


COMMANDS = {
    "upload": cmd_upload,
    "upload-hard": cmd_upload_hard,
    "train": cmd_train,
    "train-hard": cmd_train_hard,
    "train-dpo": cmd_train_dpo,
    "status": cmd_status,
    "status-hard": cmd_status_hard,
    "test": cmd_test,
    "config": cmd_opencode_config,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(f"Usage: python3 {sys.argv[0]} <command>")
        print(f"Commands: {', '.join(COMMANDS.keys())}")
        print()
        print("Workflow:")
        print("  1. upload  — Upload training_data_together.jsonl to Together AI")
        print("  2. train   — Start LoRA fine-tuning job")
        print("  3. status  — Check if training is done")
        print("  4. test    — Run verification prompts")
        print("  5. config  — Print OpenCode configuration")
        sys.exit(1)

    COMMANDS[sys.argv[1]]()
