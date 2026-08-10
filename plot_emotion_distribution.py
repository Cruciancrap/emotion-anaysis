import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    metadata_path = Path("processed") / "metadata.json"
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Cannot find {metadata_path}. Please run prepare_data.py first."
        )

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    emotion_counts = metadata["emotion_counts_in_pairs"]
    sorted_items = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)

    emotions = [item[0] for item in sorted_items]
    counts = [item[1] for item in sorted_items]

    plt.figure(figsize=(9, 4.8))
    bars = plt.bar(emotions, counts, color="#4C78A8")

    plt.xlabel("Emotion Category")
    plt.ylabel("Number of Narrative Pairs")
    plt.title("Distribution of Emotion Categories")
    plt.xticks(rotation=35, ha="right")

    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count:,}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()

    png_path = output_dir / "emotion_category_distribution.png"
    pdf_path = output_dir / "emotion_category_distribution.pdf"
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()

    print(f"Saved PNG figure to: {png_path}")
    print(f"Saved PDF figure to: {pdf_path}")


if __name__ == "__main__":
    main()
