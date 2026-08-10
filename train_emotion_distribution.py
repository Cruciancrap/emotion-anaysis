
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd


EMOTIONS = [
    "amusement",
    "anger",
    "awe",
    "contentment",
    "disgust",
    "excitement",
    "fear",
    "sadness",
    "something else",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("processed/artwork_emotion_distribution.csv"),
    )
    parser.add_argument("--model-name", default="microsoft/deberta-v3-base")
    parser.add_argument("--output-dir", type=Path, default=Path("models/emotion_distribution"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Training device. Use 'cuda' to require an NVIDIA GPU.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable CUDA mixed-precision training.",
    )
    return parser.parse_args()


def set_seed(seed: int, torch_module) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def probability_columns() -> list[str]:
    return [f"prob_{emotion.replace(' ', '_')}" for emotion in EMOTIONS]


def js_divergence(prediction, target, torch_module):
    eps = 1e-8
    prediction = prediction.clamp_min(eps)
    target = target.clamp_min(eps)
    middle = 0.5 * (prediction + target)
    left = (prediction * (prediction.log() - middle.log())).sum(dim=-1)
    right = (target * (target.log() - middle.log())).sum(dim=-1)
    return 0.5 * (left + right)


def main() -> None:
    args = parse_args()
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from tqdm.auto import tqdm
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            get_linear_schedule_with_warmup,
        )
    except ImportError as exc:
        raise SystemExit(
            "Missing training dependency. Install packages from requirements.txt first."
        ) from exc

    set_seed(args.seed, torch)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA was requested but is unavailable. Install a CUDA-enabled PyTorch build "
            "and verify it with: python -c \"import torch; print(torch.cuda.is_available())\""
        )
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        print(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        print("Device: CPU")
    print(f"Mixed precision (AMP): {use_amp}")

    frame = pd.read_csv(args.data)
    required = {"model_input", "split", *probability_columns()}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Prepared data is missing columns: {sorted(missing)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(EMOTIONS),
        id2label={index: label for index, label in enumerate(EMOTIONS)},
        label2id={label: index for index, label in enumerate(EMOTIONS)},
        dtype=torch.float32,
    )
    print(f"Master parameter dtype: {next(model.parameters()).dtype}")

    class DistributionDataset(Dataset):
        def __init__(self, data: pd.DataFrame):
            self.texts = data["model_input"].astype(str).tolist()
            self.labels = data[probability_columns()].to_numpy(dtype=np.float32)

        def __len__(self) -> int:
            return len(self.texts)

        def __getitem__(self, index: int) -> dict:
            encoded = tokenizer(
                self.texts[index],
                truncation=True,
                max_length=args.max_length,
            )
            encoded["labels"] = self.labels[index]
            return encoded

    def collate(batch: list[dict]) -> dict:
        labels = torch.tensor(np.stack([item.pop("labels") for item in batch]))
        encoded = tokenizer.pad(batch, padding=True, return_tensors="pt")
        encoded["labels"] = labels
        return encoded

    datasets = {
        split: DistributionDataset(frame[frame["split"].eq(split)].reset_index(drop=True))
        for split in ["train", "validation", "test"]
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=args.num_workers,
            collate_fn=collate,
            pin_memory=device.type == "cuda",
        )
        for split, dataset in datasets.items()
    }

    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    total_steps = max(1, args.epochs * len(loaders["train"]))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=math.ceil(total_steps * 0.1),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def evaluate(split: str) -> dict[str, float]:
        model.eval()
        js_values: list[np.ndarray] = []
        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        with torch.no_grad():
            for batch in tqdm(loaders[split], desc=f"Evaluating {split}", leave=False):
                labels = batch.pop("labels").to(device)
                inputs = {key: value.to(device) for key, value in batch.items()}
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=use_amp
                ):
                    logits = model(**inputs).logits
                probabilities = torch.softmax(logits.float(), dim=-1)
                js_values.append(js_divergence(probabilities, labels, torch).cpu().numpy())
                predictions.append(probabilities.cpu().numpy())
                targets.append(labels.cpu().numpy())

        prediction = np.concatenate(predictions)
        target = np.concatenate(targets)
        target_top = target.argmax(axis=1)
        predicted_top = prediction.argmax(axis=1)
        predicted_top3 = np.argpartition(prediction, -3, axis=1)[:, -3:]
        return {
            "jsd": float(np.concatenate(js_values).mean()),
            "top1_accuracy": float((predicted_top == target_top).mean()),
            "recall_at_3": float(
                np.mean([truth in guesses for truth, guesses in zip(target_top, predicted_top3)])
            ),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_jsd = float("inf")
    history: list[dict] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(
            loaders["train"],
            desc=f"Epoch {epoch}/{args.epochs}",
            unit="batch",
        )
        for step, batch in enumerate(progress, start=1):
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                logits = model(**inputs).logits
                probabilities = torch.softmax(logits.float(), dim=-1)
                loss = js_divergence(probabilities, labels, torch).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item()
            progress.set_postfix(loss=f"{running_loss / step:.4f}")

        validation_metrics = evaluate("validation")
        epoch_result = {
            "epoch": epoch,
            "train_jsd": running_loss / max(1, len(loaders["train"])),
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result, ensure_ascii=False))

        if validation_metrics["jsd"] < best_jsd:
            best_jsd = validation_metrics["jsd"]
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

    best_model = AutoModelForSequenceClassification.from_pretrained(
        args.output_dir, dtype=torch.float32
    ).to(device)
    model = best_model
    test_metrics = evaluate("test")
    results = {
        "model_name": args.model_name,
        "device": str(device),
        "emotions": EMOTIONS,
        "history": history,
        "test": test_metrics,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"test": test_metrics}, ensure_ascii=False, indent=2))
    print(f"Best model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
