#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aria_esc.utils.io import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA SFT for chat-formatted counseling dialogues.")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--validation-file")
    parser.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--dry-run", action="store_true", help="Validate input without loading a model.")
    return parser.parse_args()


def validate(rows: list[dict]) -> None:
    for index, row in enumerate(rows):
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError(f"row {index}: messages must contain at least two turns")
        if messages[-1].get("role") != "assistant":
            raise ValueError(f"row {index}: the final turn must be assistant")


def main() -> None:
    args = parse_args()
    train_rows = read_jsonl(args.train_file)
    validation_rows = read_jsonl(args.validation_file) if args.validation_file else []
    if not train_rows:
        raise ValueError(f"No training dialogues in {args.train_file}")
    validate(train_rows)
    validate(validation_rows)
    if args.dry_run:
        print(f"Validated {len(train_rows)} train and {len(validation_rows)} validation dialogues")
        return

    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))

    def tokenize(row: dict) -> dict:
        text = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
        encoded = tokenizer(text, truncation=True, max_length=args.max_length, add_special_tokens=False)
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded

    train_dataset = Dataset.from_list(train_rows).map(tokenize, remove_columns=list(train_rows[0]))
    eval_dataset = None
    if validation_rows:
        eval_dataset = Dataset.from_list(validation_rows).map(tokenize, remove_columns=list(validation_rows[0]))
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_dataset is not None else "no",
        bf16=torch.cuda.is_available(),
        report_to=args.report_to,
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100),
        processing_class=tokenizer,
    )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with (Path(args.output_dir) / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
