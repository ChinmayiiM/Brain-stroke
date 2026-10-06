"""Development/debugging figure: the input MRI with its predicted class and probabilities.

Not a user interface — writes a PNG (and optionally opens a window) for quick checks.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.data.visualization import GRID, INK_PRIMARY, INK_SECONDARY, SERIES_COLOURS, SURFACE, _style_axes
from src.inference.predictor import PredictionResult
from src.utils.config import PROJECT_ROOT


def save_prediction_figure(result: PredictionResult, out_path: Path, show: bool = False) -> Path:
    """Original image (left) + class probabilities (right), titled with the prediction."""
    image_file = Path(result.image_path)
    if not image_file.is_absolute():
        image_file = PROJECT_ROOT / image_file
    with Image.open(image_file) as img:
        pixels = np.asarray(img.convert("L"))

    names = list(result.probabilities)
    values = [result.probabilities[n] for n in names]
    colours = [SERIES_COLOURS[i % len(SERIES_COLOURS)] for i in range(len(names))]

    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(9, 4.2), facecolor=SURFACE,
                                         gridspec_kw={"width_ratios": [1, 1.1]})
    ax_img.imshow(pixels, cmap="gray", vmin=0, vmax=255)
    ax_img.axis("off")
    ax_img.set_title(Path(result.image_path).name, fontsize=8, color=INK_SECONDARY)

    _style_axes(ax_bar)
    bars = ax_bar.barh(names[::-1], values[::-1], color=colours[::-1], height=0.55)
    for bar, value in zip(bars, values[::-1]):
        ax_bar.annotate(f"{value:.1%}", (value, bar.get_y() + bar.get_height() / 2), va="center",
                        fontsize=9, color=INK_PRIMARY, xytext=(4, 0), textcoords="offset points")
    ax_bar.set_xlim(0, 1.15)
    ax_bar.set_xlabel("Model probability")
    ax_bar.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax_bar.set_axisbelow(True)

    fig.suptitle(f"Predicted: {result.predicted_class} ({result.confidence:.1%})", x=0.02, ha="left",
                 fontsize=12, color=INK_PRIMARY)
    fig.text(0.02, 0.01, result.disclaimer, fontsize=8, color=INK_SECONDARY)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    if show:
        plt.show()
    plt.close(fig)
    return out_path
