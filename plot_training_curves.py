"""Create a paper-ready SVG training-curve figure without extra packages."""

from __future__ import annotations

import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "figures"
WIDTH, HEIGHT = 1440, 630

TRAIN_COLOR = "#2F6B9A"
VALIDATION_COLOR = "#D97732"
BEST_COLOR = "#B42318"
GRID_COLOR = "#D5D9DE"
TEXT_COLOR = "#202124"


def read_history(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["history"]


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def path(points: list[tuple[float, float]]) -> str:
    return " ".join(
        ("M" if index == 0 else "L") + f" {x:.2f} {y:.2f}"
        for index, (x, y) in enumerate(points)
    )


def panel(
    *,
    x0: float,
    y0: float,
    width: float,
    height: float,
    title: str,
    ylabel: str,
    epochs: list[int],
    train: list[float],
    validation: list[float],
    ymin: float,
    ymax: float,
    yticks: list[float],
    train_label: str,
    validation_label: str,
    best_label: str,
    y_format: str,
) -> str:
    def sx(epoch: int) -> float:
        return x0 + (epoch - min(epochs)) / (max(epochs) - min(epochs)) * width

    def sy(value: float) -> float:
        return y0 + height - (value - ymin) / (ymax - ymin) * height

    train_points = [(sx(epoch), sy(value)) for epoch, value in zip(epochs, train)]
    val_points = [(sx(epoch), sy(value)) for epoch, value in zip(epochs, validation)]
    best_index = min(range(len(validation)), key=validation.__getitem__)
    best_x, best_y = val_points[best_index]

    out = [f'<g aria-label="{esc(title)}">']
    out.append(
        f'<text x="{x0 + width / 2:.1f}" y="35" class="panel-title" '
        f'text-anchor="middle">{esc(title)}</text>'
    )

    for value in yticks:
        y = sy(value)
        out.append(
            f'<line x1="{x0}" y1="{y:.2f}" x2="{x0 + width}" y2="{y:.2f}" '
            f'class="grid"/>'
        )
        out.append(
            f'<text x="{x0 - 14}" y="{y + 5:.2f}" class="tick" '
            f'text-anchor="end">{format(value, y_format)}</text>'
        )

    out.append(
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + height}" class="axis"/>'
    )
    out.append(
        f'<line x1="{x0}" y1="{y0 + height}" x2="{x0 + width}" '
        f'y2="{y0 + height}" class="axis"/>'
    )

    for epoch in epochs:
        x = sx(epoch)
        out.append(
            f'<line x1="{x:.2f}" y1="{y0 + height}" x2="{x:.2f}" '
            f'y2="{y0 + height + 6}" class="axis"/>'
        )
        out.append(
            f'<text x="{x:.2f}" y="{y0 + height + 26}" class="tick" '
            f'text-anchor="middle">{epoch}</text>'
        )

    out.append(
        f'<path d="{path(train_points)}" fill="none" stroke="{TRAIN_COLOR}" '
        f'class="series-line"/>'
    )
    out.append(
        f'<path d="{path(val_points)}" fill="none" stroke="{VALIDATION_COLOR}" '
        f'class="series-line"/>'
    )
    for x, y in train_points:
        out.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.2" fill="{TRAIN_COLOR}"/>')
    for x, y in val_points:
        out.append(
            f'<rect x="{x - 4.2:.2f}" y="{y - 4.2:.2f}" width="8.4" height="8.4" '
            f'fill="{VALIDATION_COLOR}"/>'
        )

    out.append(
        f'<circle cx="{best_x:.2f}" cy="{best_y:.2f}" r="10" fill="none" '
        f'stroke="{BEST_COLOR}" stroke-width="3"/>'
    )
    label_x = min(x0 + width - 185, best_x + 65)
    label_y = max(y0 + 80, best_y - 55)
    out.append(
        f'<path d="M {label_x - 8:.2f} {label_y + 8:.2f} L {best_x + 8:.2f} '
        f'{best_y - 6:.2f}" fill="none" stroke="{BEST_COLOR}" stroke-width="1.8" '
        f'marker-end="url(#arrow)"/>'
    )
    first, second = best_label.split("\n", 1)
    out.append(
        f'<text x="{label_x:.2f}" y="{label_y:.2f}" class="annotation">'
        f'<tspan x="{label_x:.2f}" dy="0">{esc(first)}</tspan>'
        f'<tspan x="{label_x:.2f}" dy="20">{esc(second)}</tspan></text>'
    )

    legend_x, legend_y = x0 + width - 185, y0 + 28
    out.append(
        f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 34}" y2="{legend_y}" '
        f'stroke="{TRAIN_COLOR}" class="series-line"/>'
    )
    out.append(f'<circle cx="{legend_x + 17}" cy="{legend_y}" r="4.2" fill="{TRAIN_COLOR}"/>')
    out.append(
        f'<text x="{legend_x + 44}" y="{legend_y + 5}" class="legend">{esc(train_label)}</text>'
    )
    out.append(
        f'<line x1="{legend_x}" y1="{legend_y + 30}" x2="{legend_x + 34}" '
        f'y2="{legend_y + 30}" stroke="{VALIDATION_COLOR}" class="series-line"/>'
    )
    out.append(
        f'<rect x="{legend_x + 12.8}" y="{legend_y + 25.8}" width="8.4" height="8.4" '
        f'fill="{VALIDATION_COLOR}"/>'
    )
    out.append(
        f'<text x="{legend_x + 44}" y="{legend_y + 35}" class="legend">'
        f'{esc(validation_label)}</text>'
    )

    out.append(
        f'<text x="{x0 + width / 2:.2f}" y="{y0 + height + 62}" class="axis-label" '
        f'text-anchor="middle">Epoch</text>'
    )
    center_y = y0 + height / 2
    out.append(
        f'<text x="{x0 - 70}" y="{center_y:.2f}" class="axis-label" '
        f'text-anchor="middle" transform="rotate(-90 {x0 - 70} {center_y:.2f})">'
        f'{esc(ylabel)}</text>'
    )
    out.append("</g>")
    return "\n".join(out)


def main() -> None:
    emotion = read_history(ROOT / "models" / "emotion_distribution" / "metrics.json")
    generator = read_history(ROOT / "models" / "narrative_generator" / "metrics.json")

    emotion_panel = panel(
        x0=105,
        y0=80,
        width=545,
        height=420,
        title="(a) Emotion distribution predictor",
        ylabel="Jensen–Shannon divergence",
        epochs=[row["epoch"] for row in emotion],
        train=[row["train_jsd"] for row in emotion],
        validation=[row["validation_jsd"] for row in emotion],
        ymin=0.12,
        ymax=0.28,
        yticks=[0.12, 0.16, 0.20, 0.24, 0.28],
        train_label="Training JSD",
        validation_label="Validation JSD",
        best_label="Best: epoch 5\nVal. JSD = 0.2152",
        y_format=".2f",
    )
    generator_panel = panel(
        x0=820,
        y0=80,
        width=545,
        height=420,
        title="(b) Emotion-conditioned narrative generator",
        ylabel="Cross-entropy loss",
        epochs=[row["epoch"] for row in generator],
        train=[row["train_loss"] for row in generator],
        validation=[row["validation_loss"] for row in generator],
        ymin=2.70,
        ymax=3.30,
        yticks=[2.70, 2.80, 2.90, 3.00, 3.10, 3.20, 3.30],
        train_label="Training loss",
        validation_label="Validation loss",
        best_label="Best: epoch 7\nVal. loss = 2.8380",
        y_format=".2f",
    )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}"
viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">
<title id="title">Training and validation curves</title>
<desc id="desc">Two panels show training and validation curves across ten epochs.
The emotion distribution predictor has its best validation JSD at epoch five,
and the narrative generator has its best validation loss at epoch seven.</desc>
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6"
  markerHeight="6" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="{BEST_COLOR}"/>
  </marker>
</defs>
<style>
  text {{ font-family: "Times New Roman", "Liberation Serif", serif; fill: {TEXT_COLOR}; }}
  .panel-title {{ font-size: 22px; font-weight: 600; }}
  .axis-label {{ font-size: 19px; }}
  .tick {{ font-size: 16px; }}
  .legend {{ font-size: 16px; }}
  .annotation {{ font-size: 16px; font-weight: 600; fill: {BEST_COLOR}; }}
  .grid {{ stroke: {GRID_COLOR}; stroke-width: 1; }}
  .axis {{ stroke: #4B4F55; stroke-width: 1.5; }}
  .series-line {{ stroke-width: 3; stroke-linejoin: round; stroke-linecap: round; }}
</style>
<rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" fill="white"/>
{emotion_panel}
{generator_panel}
</svg>'''

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / "training_curves.svg"
    output.write_text(svg, encoding="utf-8")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
