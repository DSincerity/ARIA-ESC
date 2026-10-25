#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from aria_esc.utils.io import read_jsonl


def format_context(context: list[dict]) -> str:
    """Predictor input: the previous client utterance and the support-provider response."""
    return " [SEP] ".join(f"{turn.get('speaker', '?')}: {turn.get('utterance', '')}" for turn in context)


def make_predictor(encoder: str, *, pretrained: bool = True):
    """RoBERTa CLS representation followed by a small binary head."""
    from torch import nn
    from transformers import AutoConfig, AutoModel

    class Predictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = (AutoModel.from_pretrained(encoder) if pretrained
                            else AutoModel.from_config(AutoConfig.from_pretrained(encoder)))
            self.head = nn.Sequential(nn.Linear(self.encoder.config.hidden_size, 256), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(256, 2))

        def forward(self, input_ids, attention_mask):
            hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
            return self.head(hidden)

    return Predictor()


def predict_probabilities(model_dir: str | Path, contexts: list[list[dict]], *, device: str = "cuda",
                          batch_size: int = 32) -> list[float]:
    """Temperature-calibrated P(delayed valence improvement) from a trained predictor directory."""
    import torch
    from transformers import AutoTokenizer

    from aria_esc.train.calibration import apply_temperature

    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    calibration = json.loads((model_dir / "calibration.json").read_text(encoding="utf-8"))
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = make_predictor(config["encoder"], pretrained=False)
    model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=device, weights_only=True))
    model.to(device).eval()
    logits = []
    with torch.no_grad():
        for start in range(0, len(contexts), batch_size):
            encoded = tokenizer([format_context(c) for c in contexts[start:start + batch_size]], truncation=True,
                                padding=True, max_length=config["max_length"], return_tensors="pt").to(device)
            logits.append(model(encoded["input_ids"], encoded["attention_mask"]).cpu())
    if not logits:
        return []
    return apply_temperature(torch.cat(logits), calibration["calibration"]["temperature"]).tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a binary delayed-valence turn-reward predictor.")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--validation-file", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--encoder", default="roberta-base")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate(rows: list[dict]) -> None:
    for index, row in enumerate(rows):
        if not isinstance(row.get("context"), list) or row.get("delayed_valence_reward") is None:
            raise ValueError(f"row {index}: context and delayed_valence_reward are required")


def main() -> None:
    args = parse_args()
    splits = {name: read_jsonl(path) for name, path in (
        ("train", args.train_file), ("validation", args.validation_file), ("test", args.test_file)
    )}
    for name, rows in splits.items():
        if not rows:
            raise ValueError(f"The {name} split is empty")
        validate(rows)
    if len({float(row["delayed_valence_reward"]) > 0 for row in splits["validation"]}) < 2:
        raise ValueError("The validation split needs both classes to fit the temperature")
    if args.dry_run:
        print("Validated " + ", ".join(f"{name}={len(rows)}" for name, rows in splits.items()))
        return

    import numpy as np
    import torch
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)

    class TurnDataset(Dataset):
        def __init__(self, rows: list[dict]):
            self.rows = rows
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, index):
            row = self.rows[index]
            encoded = tokenizer(format_context(row["context"]), truncation=True, padding="max_length",
                                max_length=args.max_length, return_tensors="pt")
            return {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "label": torch.tensor(1 if float(row["delayed_valence_reward"]) > 0 else 0),
            }

    loaders = {
        name: DataLoader(TurnDataset(rows), batch_size=args.batch_size, shuffle=name == "train")
        for name, rows in splits.items()
    }
    model = make_predictor(args.encoder).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = max(1, len(loaders["train"]) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)
    criterion = nn.CrossEntropyLoss()

    def epoch(loader, train=False):
        model.train(train)
        losses, predictions, labels = [], [], []
        with torch.set_grad_enabled(train):
            for batch in loader:
                logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                target = batch["label"].to(device)
                loss = criterion(logits, target)
                if train:
                    optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()
                losses.append(loss.item())
                predictions.extend(logits.argmax(-1).detach().cpu().tolist())
                labels.extend(target.detach().cpu().tolist())
        return {
            "loss": sum(losses) / max(1, len(losses)),
            "accuracy": accuracy_score(labels, predictions),
            "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
            "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        }

    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    best, stale, history = -1.0, 0, []
    for number in range(1, args.epochs + 1):
        train_metrics = epoch(loaders["train"], train=True)
        validation_metrics = epoch(loaders["validation"])
        history.append({"epoch": number, "train": train_metrics, "validation": validation_metrics})
        if validation_metrics["macro_f1"] > best:
            best, stale = validation_metrics["macro_f1"], 0
            torch.save(model.state_dict(), output / "best_model.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(torch.load(output / "best_model.pt", map_location=device, weights_only=True))
    from aria_esc.train.calibration import calibration_metrics, fit_temperature

    def collect_logits(loader):
        model.eval()
        logits, labels = [], []
        with torch.no_grad():
            for batch in loader:
                logits.append(model(batch["input_ids"].to(device), batch["attention_mask"].to(device)).cpu())
                labels.append(batch["label"])
        return torch.cat(logits), torch.cat(labels)

    validation_logits, validation_labels = collect_logits(loaders["validation"])
    temperature = fit_temperature(validation_logits, validation_labels)
    test_logits, test_labels = collect_logits(loaders["test"])
    calibration = {"method": "temperature", "fit_split": "validation",
                   "calibration": {"temperature": temperature},
                   "validation_before": calibration_metrics(validation_logits, validation_labels, 1.0),
                   "validation_after": calibration_metrics(validation_logits, validation_labels, temperature),
                   "test_before": calibration_metrics(test_logits, test_labels, 1.0),
                   "test_after": calibration_metrics(test_logits, test_labels, temperature)}
    (output / "calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    results = {"best_validation_macro_f1": best, "test": epoch(loaders["test"]), "history": history, "config": vars(args)}
    (output / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    (output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    tokenizer.save_pretrained(output)


if __name__ == "__main__":
    main()
