#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aria_esc.utils.io import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DPO from an SFT LoRA adapter.")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--validation-file")
    parser.add_argument("--sft-adapter", required=True)
    parser.add_argument("--base-model", help="Defaults to the base model recorded in the SFT adapter config")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate(rows: list[dict]) -> None:
    required = {"prompt", "chosen", "rejected"}
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"row {index}: missing {sorted(missing)}")
        if row["chosen"].strip() == row["rejected"].strip():
            raise ValueError(f"row {index}: chosen and rejected are identical")


def main() -> None:
    args = parse_args()
    train_rows = read_jsonl(args.train_file)
    validation_rows = read_jsonl(args.validation_file) if args.validation_file else []
    if not train_rows:
        raise ValueError(f"No training pairs in {args.train_file}")
    validate(train_rows)
    validate(validation_rows)
    if args.dry_run:
        print(f"Validated {len(train_rows)} train and {len(validation_rows)} validation pairs")
        return

    import torch
    from datasets import Dataset
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import DPOConfig, DPOTrainer

    set_seed(args.seed)
    base_model = args.base_model or PeftConfig.from_pretrained(args.sft_adapter).base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(args.sft_adapter)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    policy = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype),
        args.sft_adapter,
        is_trainable=True,
    )
    reference = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype),
        args.sft_adapter,
        is_trainable=False,
    )
    reference.eval()
    train_dataset = Dataset.from_list(train_rows)
    eval_dataset = Dataset.from_list(validation_rows) if validation_rows else None
    config = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        beta=args.beta,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=20,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_dataset is not None else "no",
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        bf16=torch.cuda.is_available(),
        report_to=args.report_to,
        seed=args.seed,
        remove_unused_columns=False,
    )
    trainer = DPOTrainer(
        model=policy,
        ref_model=reference,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
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
