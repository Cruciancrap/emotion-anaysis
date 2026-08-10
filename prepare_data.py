"""Prepare ArtEmis + OLA data for the simplified EI conference pipeline.

This script never modifies the two raw CSV files. It creates:
1. one artwork-level emotion-distribution table;
2. one emotion-conditioned narrative-pair table;
3. one metadata JSON file containing data audit statistics.
"""

from __future__ import annotations

import argparse
import json
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
KEY_COLUMNS = ["art_style", "painting"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match OLA with ArtEmis and build leakage-free train/val/test data."
    )
    parser.add_argument(
        "--artemis",
        type=Path,
        default=Path("artemis_dataset_release_v0.csv"),
        help="Path to the raw ArtEmis CSV.",
    )
    parser.add_argument(
        "--ola",
        type=Path,
        default=Path("ola_dataset_release_v0.csv"),
        help="Path to the raw OLA CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("processed"),
        help="Directory for derived data.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing derived output files.",
    )
    return parser.parse_args()


def validate_columns(frame: pd.DataFrame, expected: set[str], name: str) -> None:
    missing = expected.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def emotion_slug(emotion: str) -> str:
    return emotion.replace(" ", "_")


def stratified_artwork_split(
    artworks: pd.DataFrame,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> dict[tuple[str, str], str]:
    """Split by artwork while approximately preserving dominant-emotion ratios."""
    if not (0 < train_ratio < 1 and 0 <= val_ratio < 1):
        raise ValueError("train_ratio and val_ratio must be within [0, 1].")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be smaller than 1.")

    rng = np.random.default_rng(seed)
    assignments: dict[tuple[str, str], str] = {}
    for _, group in artworks.groupby("dominant_emotion", sort=True):
        order = rng.permutation(len(group))
        shuffled = group.iloc[order]
        n_train = int(len(shuffled) * train_ratio)
        n_val = int(len(shuffled) * val_ratio)

        for position, row in enumerate(shuffled.itertuples(index=False)):
            if position < n_train:
                split = "train"
            elif position < n_train + n_val:
                split = "validation"
            else:
                split = "test"
            assignments[(row.art_style, row.painting)] = split
    return assignments


def main() -> None:
    args = parse_args()
    output_paths = {
        "distribution": args.output_dir / "artwork_emotion_distribution.csv",
        "narratives": args.output_dir / "narrative_pairs.csv",
        "metadata": args.output_dir / "metadata.json",
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not args.overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output already exists: {joined}. Use --overwrite to replace it.")

    artemis = pd.read_csv(args.artemis)
    ola = pd.read_csv(args.ola)
    validate_columns(
        artemis,
        {"art_style", "painting", "emotion", "utterance", "repetition"},
        "ArtEmis",
    )
    validate_columns(ola, {"art_style", "painting", "utterance"}, "OLA")

    unknown_emotions = sorted(set(artemis["emotion"].unique()).difference(EMOTIONS))
    if unknown_emotions:
        raise ValueError(f"Unexpected emotion labels: {unknown_emotions}")

    raw_artemis_rows = len(artemis)
    duplicate_artemis_rows = int(artemis.duplicated().sum())
    artemis = artemis.drop_duplicates().copy()
    ola = ola.drop_duplicates(KEY_COLUMNS).copy()
    ola = ola.rename(columns={"utterance": "objective_description"})

    overlap = ola.merge(artemis, on=KEY_COLUMNS, how="inner", validate="one_to_many")
    overlap_keys = overlap[KEY_COLUMNS].drop_duplicates()
    unmatched_ola = ola.merge(overlap_keys, on=KEY_COLUMNS, how="left", indicator=True)
    unmatched_ola = unmatched_ola[unmatched_ola["_merge"].eq("left_only")]

    counts = (
        overlap.groupby(KEY_COLUMNS + ["emotion"], sort=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=EMOTIONS, fill_value=0)
    )
    probabilities = counts.div(counts.sum(axis=1), axis=0)

    distribution = overlap[
        KEY_COLUMNS + ["objective_description"]
    ].drop_duplicates(KEY_COLUMNS)
    distribution = distribution.set_index(KEY_COLUMNS)
    for emotion in EMOTIONS:
        slug = emotion_slug(emotion)
        distribution[f"count_{slug}"] = counts[emotion]
        distribution[f"prob_{slug}"] = probabilities[emotion]
    distribution["annotation_count"] = counts.sum(axis=1)
    distribution["dominant_emotion"] = counts.idxmax(axis=1)
    distribution = distribution.reset_index()

    split_map = stratified_artwork_split(
        distribution,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    distribution["split"] = [
        split_map[(style, painting)]
        for style, painting in distribution[KEY_COLUMNS].itertuples(index=False, name=None)
    ]
    distribution["model_input"] = (
        "style: "
        + distribution["art_style"].str.replace("_", " ", regex=False)
        + " | painting: "
        + distribution["painting"].str.replace("-", " ", regex=False)
        + " | description: "
        + distribution["objective_description"]
    )

    narratives = overlap.rename(
        columns={"emotion": "target_emotion", "utterance": "target_utterance"}
    )[
        KEY_COLUMNS
        + ["objective_description", "target_emotion", "target_utterance"]
    ].copy()
    narratives["split"] = [
        split_map[(style, painting)]
        for style, painting in narratives[KEY_COLUMNS].itertuples(index=False, name=None)
    ]
    narratives["model_input"] = (
        "generate an art explanation | emotion: "
        + narratives["target_emotion"]
        + " | style: "
        + narratives["art_style"].str.replace("_", " ", regex=False)
        + " | painting: "
        + narratives["painting"].str.replace("-", " ", regex=False)
        + " | description: "
        + narratives["objective_description"]
    )

    split_artworks = distribution["split"].value_counts().to_dict()
    split_narratives = narratives["split"].value_counts().to_dict()
    metadata = {
        "seed": args.seed,
        "raw_artemis_rows": raw_artemis_rows,
        "removed_exact_artemis_duplicates": duplicate_artemis_rows,
        "raw_ola_rows": int(len(ola)),
        "matched_artworks": int(len(distribution)),
        "unmatched_ola_artworks": int(len(unmatched_ola)),
        "narrative_pairs": int(len(narratives)),
        "emotions": EMOTIONS,
        "artworks_by_split": {key: int(value) for key, value in split_artworks.items()},
        "narratives_by_split": {
            key: int(value) for key, value in split_narratives.items()
        },
        "emotion_counts_in_pairs": {
            key: int(value)
            for key, value in narratives["target_emotion"].value_counts().items()
        },
        "note": "All rows for the same artwork are assigned to the same split.",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    distribution.to_csv(output_paths["distribution"], index=False, encoding="utf-8-sig")
    narratives.to_csv(output_paths["narratives"], index=False, encoding="utf-8-sig")
    output_paths["metadata"].write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Saved: {output_paths['distribution']}")
    print(f"Saved: {output_paths['narratives']}")
    print(f"Saved: {output_paths['metadata']}")


if __name__ == "__main__":
    main()
