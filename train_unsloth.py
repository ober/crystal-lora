#!/usr/bin/env python3
"""
Train a Crystal-language LoRA locally using Unsloth.

Prerequisites:
  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
  pip install --no-deps xformers trl peft accelerate bitsandbytes triton

Requirements:
  - GPU with 24GB+ VRAM (RTX 4090, 3090, A100, etc.) — base model is large
  - ~40GB disk space

Usage:
  python3 train_unsloth.py

Output:
  ./crystal-lora-output/   — LoRA adapter weights
  Then run: python3 merge_and_export.py
"""

import os

# ── Config ──────────────────────────────────────────────────────────
# Use a smaller coder-tuned base for local training; the Together AI flow
# uses the bigger Qwen3-Coder-30B-A3B-Instruct.
MODEL_NAME            = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"
OUTPUT_DIR            = os.path.join(os.path.dirname(__file__), "crystal-lora-output")
TRAINING_FILE         = os.path.join(os.path.dirname(__file__), "training_data.jsonl")
MAX_SEQ_LENGTH        = 4096
EPOCHS                = 3
BATCH_SIZE            = 2
GRADIENT_ACCUMULATION = 4
LEARNING_RATE         = 2e-4
LORA_R                = 16
LORA_ALPHA            = 32


def main():
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import load_dataset

    print(f"Loading {MODEL_NAME} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    print(f"Applying LoRA (r={LORA_R}, alpha={LORA_ALPHA}) ...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    print(f"Loading {TRAINING_FILE} ...")
    dataset = load_dataset("json", data_files=TRAINING_FILE, split="train")
    print(f"Loaded {len(dataset)} examples")

    def format_chatml(example):
        messages = example["conversations"]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    dataset = dataset.map(format_chatml, num_proc=2)

    print("Starting training ...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=True,
        args=TrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION,
            warmup_steps=10,
            num_train_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            fp16=True,
            logging_steps=10,
            optim="adamw_8bit",
            save_strategy="epoch",
            save_total_limit=2,
            seed=42,
            report_to="none",
        ),
    )

    stats = trainer.train()
    print(f"\nTraining complete!")
    print(f"  Total steps: {stats.global_step}")
    print(f"  Training loss: {stats.training_loss:.4f}")
    print(f"  Runtime: {stats.metrics['train_runtime']:.0f}s")

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nLoRA adapter saved to {OUTPUT_DIR}/")
    print(f"\nNext step: python3 merge_and_export.py")


if __name__ == "__main__":
    main()
