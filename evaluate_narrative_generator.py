"""Evaluate the saved narrative generator on the complete test set.

The evaluator does not train or update model weights. It groups repeated test
rows by unique model input, treats the corresponding ArtEmis utterances as
multiple valid references, generates several candidates, and compares the
original best beam with a lightweight grounding-aware reranked candidate.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from train_narrative_generator import (
    lightweight_grounding_score,
    rouge_l_f1,
    unsupported_claim_count,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("processed/narrative_pairs.csv")
    )
    parser.add_argument(
        "--model-dir", type=Path, default=Path("models/narrative_generator")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/narrative_generator/full_test_evaluation"),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-candidates", type=int, default=3)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-input-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--precision", choices=["auto", "bf16", "fp32"], default="auto"
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--grounding-weight",
        type=float,
        default=1.0,
        help="Weight of lexical grounding and unsupported-claim rules.",
    )
    parser.add_argument(
        "--model-score-weight",
        type=float,
        default=0.15,
        help="Weight of the length-normalized beam score during reranking.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=0,
        help="Evaluate only the first N unique prompts; 0 evaluates all prompts.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def choose_device(args, torch):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")
    if args.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(args.device)


def choose_precision(args, device, torch) -> str:
    if device.type != "cuda":
        if args.precision == "bf16":
            raise SystemExit("BF16 evaluation requires a CUDA device.")
        return "fp32"
    if args.precision == "auto":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp32"
    if args.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise SystemExit("This CUDA device does not support BF16.")
    return args.precision


def ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1))


def multi_reference_bleu(
    references: list[list[str]], hypotheses: list[str], max_order: int = 4
) -> float:
    """Corpus BLEU with multiple references and add-one smoothing."""
    clipped = Counter()
    totals = Counter()
    total_reference_length = 0
    total_hypothesis_length = 0

    for reference_set, hypothesis in zip(references, hypotheses):
        reference_tokens = [reference.lower().split() for reference in reference_set]
        hypothesis_tokens = hypothesis.lower().split()
        total_hypothesis_length += len(hypothesis_tokens)

        lengths = [len(tokens) for tokens in reference_tokens]
        closest_length = min(lengths, key=lambda length: (abs(length - len(hypothesis_tokens)), length))
        total_reference_length += closest_length

        for order in range(1, max_order + 1):
            hypothesis_ngrams = ngrams(hypothesis_tokens, order)
            maximum_reference_counts: Counter = Counter()
            for tokens in reference_tokens:
                reference_ngrams = ngrams(tokens, order)
                for gram, count in reference_ngrams.items():
                    maximum_reference_counts[gram] = max(
                        maximum_reference_counts[gram], count
                    )
            clipped[order] += sum(
                min(count, maximum_reference_counts[gram])
                for gram, count in hypothesis_ngrams.items()
            )
            totals[order] += sum(hypothesis_ngrams.values())

    precisions = [
        (clipped[order] + 1) / (totals[order] + 1)
        for order in range(1, max_order + 1)
    ]
    if total_hypothesis_length == 0:
        return 0.0
    brevity_penalty = (
        1.0
        if total_hypothesis_length > total_reference_length
        else math.exp(1 - total_reference_length / total_hypothesis_length)
    )
    return float(
        brevity_penalty
        * math.exp(sum(math.log(precision) for precision in precisions) / max_order)
    )


def group_test_prompts(frame: pd.DataFrame) -> list[dict]:
    test = frame[frame["split"].eq("test")].copy()
    groups: list[dict] = []
    for model_input, group in test.groupby("model_input", sort=False):
        first = group.iloc[0]
        groups.append(
            {
                "art_style": first["art_style"],
                "painting": first["painting"],
                "objective_description": first["objective_description"],
                "target_emotion": first["target_emotion"],
                "model_input": model_input,
                "references": list(dict.fromkeys(group["target_utterance"].astype(str))),
            }
        )
    return groups


def score_candidate(
    description: str,
    candidate: str,
    model_score: float,
    grounding_weight: float,
    model_score_weight: float,
) -> float:
    return (
        grounding_weight * lightweight_grounding_score(description, candidate)
        + model_score_weight * model_score
    )


def summarize_predictions(rows: list[dict], field: str) -> dict[str, float]:
    hypotheses = [row[field] for row in rows]
    references = [row["references"] for row in rows]
    best_rouge = [
        max(rouge_l_f1(reference, hypothesis) for reference in reference_set)
        for reference_set, hypothesis in zip(references, hypotheses)
    ]
    average_rouge = [
        float(np.mean([rouge_l_f1(reference, hypothesis) for reference in reference_set]))
        for reference_set, hypothesis in zip(references, hypotheses)
    ]
    grounding = [
        lightweight_grounding_score(row["objective_description"], row[field])
        for row in rows
    ]
    unsupported = [unsupported_claim_count(text) for text in hypotheses]
    return {
        "multi_reference_bleu_4": multi_reference_bleu(references, hypotheses),
        "multi_reference_max_rouge_l_f1": float(np.mean(best_rouge)),
        "multi_reference_mean_rouge_l_f1": float(np.mean(average_rouge)),
        "mean_grounding_score": float(np.mean(grounding)),
        "unsupported_claim_rate": float(np.mean(np.asarray(unsupported) > 0)),
        "mean_generated_words": float(
            np.mean([len(hypothesis.split()) for hypothesis in hypotheses])
        ),
        "unique_generation_rate": len(set(hypotheses)) / max(1, len(hypotheses)),
    }


def main() -> None:
    args = parse_args()
    if args.num_candidates < 2:
        raise ValueError("--num-candidates must be at least 2 for reranking.")
    if args.num_beams < args.num_candidates:
        raise ValueError("--num-beams must be greater than or equal to --num-candidates.")

    prediction_path = args.output_dir / "unique_prompt_predictions.csv"
    metrics_path = args.output_dir / "full_test_metrics.json"
    if not args.overwrite and (prediction_path.exists() or metrics_path.exists()):
        raise FileExistsError(
            f"Evaluation output already exists in {args.output_dir}. Use --overwrite."
        )
    if not args.model_dir.exists():
        raise FileNotFoundError(f"Saved model not found: {args.model_dir}")

    try:
        import torch
        from tqdm.auto import tqdm
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install the packages in requirements.txt first.") from exc

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = choose_device(args, torch)
    precision = choose_precision(args, device, torch)
    model_dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    print(
        f"Device: CUDA ({torch.cuda.get_device_name(0)})"
        if device.type == "cuda"
        else "Device: CPU"
    )
    print(f"Evaluation precision: {precision}")

    frame = pd.read_csv(args.data)
    required = {
        "art_style",
        "painting",
        "objective_description",
        "target_emotion",
        "target_utterance",
        "model_input",
        "split",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Evaluation data is missing columns: {sorted(missing)}")

    prompts = group_test_prompts(frame)
    total_unique_prompts = len(prompts)
    if args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]
    print(
        f"Test rows: {int(frame['split'].eq('test').sum())}; "
        f"unique prompts: {total_unique_prompts}; evaluating: {len(prompts)}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_dir, dtype=model_dtype
    ).to(device)
    model.eval()

    evaluated: list[dict] = []
    progress = tqdm(
        range(0, len(prompts), args.batch_size),
        desc="Generating candidates",
        unit="batch",
    )
    for start in progress:
        batch_prompts = prompts[start : start + args.batch_size]
        encoded = tokenizer(
            [row["model_input"] for row in batch_prompts],
            padding=True,
            truncation=True,
            max_length=args.max_input_length,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                num_return_sequences=args.num_candidates,
                no_repeat_ngram_size=3,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True,
            )
        decoded = tokenizer.batch_decode(generated.sequences, skip_special_tokens=True)
        if generated.sequences_scores is None:
            sequence_scores = [0.0] * len(decoded)
        else:
            sequence_scores = generated.sequences_scores.float().cpu().tolist()

        for batch_index, row in enumerate(batch_prompts):
            begin = batch_index * args.num_candidates
            end = begin + args.num_candidates
            candidates = decoded[begin:end]
            candidate_model_scores = sequence_scores[begin:end]
            combined_scores = [
                score_candidate(
                    row["objective_description"],
                    candidate,
                    model_score,
                    args.grounding_weight,
                    args.model_score_weight,
                )
                for candidate, model_score in zip(candidates, candidate_model_scores)
            ]
            selected_index = int(np.argmax(combined_scores))
            baseline = candidates[0]
            reranked = candidates[selected_index]
            evaluated.append(
                {
                    **row,
                    "baseline_generation": baseline,
                    "reranked_generation": reranked,
                    "reranked_candidate_index": selected_index,
                    "baseline_model_score": candidate_model_scores[0],
                    "reranked_model_score": candidate_model_scores[selected_index],
                    "baseline_grounding_score": lightweight_grounding_score(
                        row["objective_description"], baseline
                    ),
                    "reranked_grounding_score": lightweight_grounding_score(
                        row["objective_description"], reranked
                    ),
                    "baseline_unsupported_claims": unsupported_claim_count(baseline),
                    "reranked_unsupported_claims": unsupported_claim_count(reranked),
                    "candidate_texts": candidates,
                    "candidate_model_scores": candidate_model_scores,
                    "candidate_combined_scores": combined_scores,
                }
            )

    baseline_metrics = summarize_predictions(evaluated, "baseline_generation")
    reranked_metrics = summarize_predictions(evaluated, "reranked_generation")
    changed = [
        row["baseline_generation"] != row["reranked_generation"] for row in evaluated
    ]
    metrics = {
        "model_dir": str(args.model_dir),
        "data": str(args.data),
        "device": str(device),
        "precision": precision,
        "test_rows": int(frame["split"].eq("test").sum()),
        "total_unique_test_prompts": total_unique_prompts,
        "evaluated_unique_prompts": len(evaluated),
        "total_references": int(sum(len(row["references"]) for row in evaluated)),
        "num_candidates": args.num_candidates,
        "num_beams": args.num_beams,
        "reranking": {
            "grounding_weight": args.grounding_weight,
            "model_score_weight": args.model_score_weight,
            "changed_generation_rate": float(np.mean(changed)),
        },
        "baseline_best_beam": baseline_metrics,
        "grounding_reranked": reranked_metrics,
    }

    export_rows = []
    for row in evaluated:
        export = dict(row)
        for field in [
            "references",
            "candidate_texts",
            "candidate_model_scores",
            "candidate_combined_scores",
        ]:
            export[field] = json.dumps(export[field], ensure_ascii=False)
        export["reference_count"] = len(row["references"])
        export_rows.append(export)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(export_rows).to_csv(
        prediction_path, index=False, encoding="utf-8-sig"
    )
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved predictions: {prediction_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
