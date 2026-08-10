"""Train T5/FLAN-T5 and optionally run a local Top-3 emotion narrative demo."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
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

UNSUPPORTED_PATTERNS = [
    r"\bthe artist intended\b",
    r"\bthe painter intended\b",
    r"\bduring the war\b",
    r"\bafter (?:his|her|the artist's) (?:wife|husband|child|lover) died\b",
    r"\bpainted (?:this|it) after\b",
    r"\bin the year \d{3,4}\b",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["train", "demo"], default="train")
    parser.add_argument(
        "--data", type=Path, default=Path("processed/narrative_pairs.csv")
    )
    parser.add_argument("--model-name", default="google/flan-t5-base")
    parser.add_argument("--output-dir", type=Path, default=Path("models/narrative_generator"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-input-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=96)
    parser.add_argument(
        "--eval-max-samples",
        type=int,
        default=0,
        help=(
            "Maximum number of test rows to evaluate after training. "
            "Use 0 (the default) to evaluate the complete test set."
        ),
    )
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
        help="Compatibility flag: force full FP32 training.",
    )
    parser.add_argument(
        "--precision",
        choices=["auto", "bf16", "fp16", "fp32"],
        default="auto",
        help=(
            "Training precision. 'auto' uses BF16 on supported CUDA GPUs, "
            "otherwise FP16 on CUDA or FP32 on CPU."
        ),
    )
    parser.add_argument(
        "--distribution-model",
        type=Path,
        default=Path("models/emotion_distribution"),
        help="Local distribution model used by demo mode.",
    )
    parser.add_argument("--style", default="Realism")
    parser.add_argument("--painting", default="unknown-painting")
    parser.add_argument(
        "--description", default="A woman sits alone beside a table."
    )
    return parser.parse_args()


def set_seed(seed: int, torch_module) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def narrative_input(style: str, painting: str, description: str, emotion: str) -> str:
    return (
        f"generate an art explanation | emotion: {emotion} | "
        f"style: {style.replace('_', ' ')} | "
        f"painting: {painting.replace('-', ' ')} | description: {description}"
    )


def distribution_input(style: str, painting: str, description: str) -> str:
    return (
        f"style: {style.replace('_', ' ')} | "
        f"painting: {painting.replace('-', ' ')} | description: {description}"
    )


def unsupported_claim_count(text: str) -> int:
    lowered = text.lower()
    return sum(bool(re.search(pattern, lowered)) for pattern in UNSUPPORTED_PATTERNS)


def content_tokens(text: str) -> set[str]:
    stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "have", "in", "is", "it", "of", "on", "or", "that", "the",
        "this", "to", "was", "with", "i", "me", "my", "makes", "feel",
    }
    return {
        token
        for token in re.findall(r"[a-z]+", text.lower())
        if len(token) > 2 and token not in stopwords
    }


def lightweight_grounding_score(description: str, candidate: str) -> float:
    """Reward visible-word support and strongly penalize invented history/intent."""
    source = content_tokens(description)
    output = content_tokens(candidate)
    overlap = len(source.intersection(output)) / max(1, len(source))
    return overlap - 2.0 * unsupported_claim_count(candidate)


def lcs_length(first: list[str], second: list[str]) -> int:
    previous = [0] * (len(second) + 1)
    for token_a in first:
        current = [0]
        for index, token_b in enumerate(second, start=1):
            if token_a == token_b:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(reference: str, hypothesis: str) -> float:
    reference_tokens = reference.lower().split()
    hypothesis_tokens = hypothesis.lower().split()
    if not reference_tokens or not hypothesis_tokens:
        return 0.0
    common = lcs_length(reference_tokens, hypothesis_tokens)
    precision = common / len(hypothesis_tokens)
    recall = common / len(reference_tokens)
    return 2 * precision * recall / max(1e-12, precision + recall)


def corpus_bleu_4(references: list[str], hypotheses: list[str]) -> float:
    """Small dependency-free BLEU-4 implementation with add-one smoothing."""
    clipped = Counter()
    totals = Counter()
    reference_length = 0
    hypothesis_length = 0
    for reference, hypothesis in zip(references, hypotheses):
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        reference_length += len(ref_tokens)
        hypothesis_length += len(hyp_tokens)
        for n in range(1, 5):
            ref_ngrams = Counter(tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1))
            hyp_ngrams = Counter(tuple(hyp_tokens[i : i + n]) for i in range(len(hyp_tokens) - n + 1))
            clipped[n] += sum(min(count, ref_ngrams[gram]) for gram, count in hyp_ngrams.items())
            totals[n] += sum(hyp_ngrams.values())
    precisions = [(clipped[n] + 1) / (totals[n] + 1) for n in range(1, 5)]
    brevity = 1.0 if hypothesis_length > reference_length else math.exp(
        1 - reference_length / max(1, hypothesis_length)
    )
    return float(brevity * math.exp(sum(math.log(value) for value in precisions) / 4))


def select_device(args, torch):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA was requested but is unavailable. Install a CUDA-enabled PyTorch build "
            "and verify it with: python -c \"import torch; print(torch.cuda.is_available())\""
        )
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    return torch.device(device_name)


def select_precision(args, device, torch) -> str:
    """Choose a numerically stable precision for sequence-to-sequence training."""
    if args.no_amp:
        return "fp32"
    if device.type != "cuda":
        if args.precision not in {"auto", "fp32"}:
            raise SystemExit(f"{args.precision} requires a CUDA device.")
        return "fp32"
    if args.precision == "auto":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise SystemExit(
            "BF16 was requested but this GPU does not support it. "
            "Use --precision fp16 or --precision fp32."
        )
    return args.precision


def run_demo(args, torch, model_classes) -> None:
    AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer = model_classes
    device = select_device(args, torch)
    print(
        f"Device: CUDA ({torch.cuda.get_device_name(0)})"
        if device.type == "cuda"
        else "Device: CPU"
    )

    distribution_tokenizer = AutoTokenizer.from_pretrained(args.distribution_model)
    distribution_model = AutoModelForSequenceClassification.from_pretrained(
        args.distribution_model, dtype=torch.float32
    ).to(device)
    generator_tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
    generator = AutoModelForSeq2SeqLM.from_pretrained(
        args.output_dir, dtype=torch.float32
    ).to(device)
    distribution_model.eval()
    generator.eval()

    encoded = distribution_tokenizer(
        distribution_input(args.style, args.painting, args.description),
        return_tensors="pt",
        truncation=True,
        max_length=args.max_input_length,
    ).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(distribution_model(**encoded).logits, dim=-1)[0]
    top_indices = probabilities.topk(3).indices.tolist()

    outputs = []
    for emotion_index in top_indices:
        emotion = distribution_model.config.id2label.get(emotion_index, EMOTIONS[emotion_index])
        prompt = narrative_input(args.style, args.painting, args.description, emotion)
        generator_inputs = generator_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_input_length,
        ).to(device)
        with torch.no_grad():
            sequences = generator.generate(
                **generator_inputs,
                max_new_tokens=args.max_target_length,
                num_beams=4,
                num_return_sequences=3,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
        candidates = generator_tokenizer.batch_decode(sequences, skip_special_tokens=True)
        selected = max(
            candidates,
            key=lambda text: lightweight_grounding_score(args.description, text),
        )
        outputs.append(
            {
                "emotion": emotion,
                "probability": float(probabilities[emotion_index].cpu()),
                "narrative": selected,
                "grounding_score": lightweight_grounding_score(args.description, selected),
                "blocked_claims": unsupported_claim_count(selected),
            }
        )
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from tqdm.auto import tqdm
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoModelForSequenceClassification,
            AutoTokenizer,
            get_linear_schedule_with_warmup,
        )
    except ImportError as exc:
        raise SystemExit(
            "Missing training dependency. Install packages from requirements.txt first."
        ) from exc

    model_classes = (
        AutoModelForSeq2SeqLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )
    set_seed(args.seed, torch)
    if args.mode == "demo":
        run_demo(args, torch, model_classes)
        return

    device = select_device(args, torch)
    precision = select_precision(args, device, torch)
    use_amp = precision in {"bf16", "fp16"}
    use_scaler = precision == "fp16"
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        print(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        print("Device: CPU")
    print(f"Training precision: {precision}")
    print(f"Gradient scaler: {use_scaler}")

    frame = pd.read_csv(args.data)
    required = {"model_input", "target_utterance", "target_emotion", "split"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Prepared data is missing columns: {sorted(missing)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name, dtype=torch.float32
    )
    print(f"Master parameter dtype: {next(model.parameters()).dtype}")

    class NarrativeDataset(Dataset):
        def __init__(self, data: pd.DataFrame):
            self.inputs = data["model_input"].astype(str).tolist()
            self.targets = data["target_utterance"].astype(str).tolist()

        def __len__(self) -> int:
            return len(self.inputs)

        def __getitem__(self, index: int) -> dict[str, str]:
            return {"input": self.inputs[index], "target": self.targets[index]}

    def collate(batch: list[dict[str, str]]) -> dict:
        encoded = tokenizer(
            [item["input"] for item in batch],
            padding=True,
            truncation=True,
            max_length=args.max_input_length,
            return_tensors="pt",
        )
        labels = tokenizer(
            text_target=[item["target"] for item in batch],
            padding=True,
            truncation=True,
            max_length=args.max_target_length,
            return_tensors="pt",
        )["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        encoded["labels"] = labels
        return encoded

    split_frames = {
        split: frame[frame["split"].eq(split)].reset_index(drop=True)
        for split in ["train", "validation", "test"]
    }
    datasets = {split: NarrativeDataset(data) for split, data in split_frames.items()}
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = max(1, args.epochs * len(loaders["train"]))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=math.ceil(total_steps * 0.1),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    def validation_loss() -> float:
        model.eval()
        losses = []
        with torch.no_grad():
            for batch in tqdm(
                loaders["validation"], desc="Evaluating validation", leave=False
            ):
                batch = {key: value.to(device) for key, value in batch.items()}
                with torch.autocast(
                    device_type=device.type, dtype=amp_dtype, enabled=use_amp
                ):
                    loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "Non-finite validation loss detected. Try --precision fp32 "
                        "and a lower --learning-rate."
                    )
                losses.append(loss.item())
        return float(np.mean(losses))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(
            loaders["train"],
            desc=f"Epoch {epoch}/{args.epochs}",
            unit="batch",
        )
        for step, batch in enumerate(progress, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=use_amp
            ):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss at epoch {epoch}, step {step}. "
                    "Stop training and retry with --precision bf16 or fp32."
                )
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            progress.set_postfix(loss=f"{running_loss / step:.4f}")
        val_loss = validation_loss()
        record = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, len(loaders["train"])),
            "validation_loss": val_loss,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        if val_loss < best_loss:
            best_loss = val_loss
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.output_dir, dtype=torch.float32
    ).to(device)
    model.eval()
    test = split_frames["test"].copy()
    if args.eval_max_samples > 0:
        test = test.head(args.eval_max_samples).copy()
    hypotheses = []
    for start in range(0, len(test), args.batch_size):
        texts = test["model_input"].iloc[start : start + args.batch_size].tolist()
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=args.max_input_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            sequences = model.generate(
                **encoded,
                max_new_tokens=args.max_target_length,
                num_beams=4,
                no_repeat_ngram_size=3,
            )
        hypotheses.extend(tokenizer.batch_decode(sequences, skip_special_tokens=True))

    references = test["target_utterance"].astype(str).tolist()
    metrics = {
        "bleu_4": corpus_bleu_4(references, hypotheses),
        "rouge_l_f1": float(
            np.mean([rouge_l_f1(ref, hyp) for ref, hyp in zip(references, hypotheses)])
        ),
        "unsupported_claim_rate": float(
            np.mean([unsupported_claim_count(text) > 0 for text in hypotheses])
        ),
        "evaluated_samples": len(test),
    }
    test["generated_utterance"] = hypotheses
    test["grounding_score"] = [
        lightweight_grounding_score(description, output)
        for description, output in zip(test["objective_description"], hypotheses)
    ]
    test.to_csv(args.output_dir / "test_generations.csv", index=False, encoding="utf-8-sig")
    results = {
        "model_name": args.model_name,
        "device": str(device),
        "history": history,
        "test": metrics,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"test": metrics}, ensure_ascii=False, indent=2))
    print(f"Best model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
