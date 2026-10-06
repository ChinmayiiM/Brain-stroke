"""Dataset visualisations (Matplotlib). All figures are drawn from real records."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — figures are written to disk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

# Categorical slots (fixed order, validated for <= 3 series all-pairs);
# text stays in neutral ink, never in the series colour.
SERIES_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.yaxis.label.set_color(INK_SECONDARY)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.title.set_color(INK_PRIMARY)


def _new_figure(width: float, height: float) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(width, height), facecolor=SURFACE)
    _style_axes(ax)
    return fig, ax


def _colour_map(class_names: list[str]) -> dict[str, str]:
    return {name: SERIES_COLOURS[i % len(SERIES_COLOURS)] for i, name in enumerate(class_names)}


def plot_class_distribution(df: pd.DataFrame, class_names: list[str], out_path: Path, title: str) -> None:
    """Bar chart of images per class with count + percentage labels."""
    counts = df["folder_label"].value_counts().reindex(class_names, fill_value=0)
    colours = _colour_map(class_names)
    fig, ax = _new_figure(6.5, 4)
    bars = ax.bar(counts.index, counts.values, width=0.55, color=[colours[c] for c in counts.index])
    total = counts.sum()
    for bar, value in zip(bars, counts.values):
        ax.annotate(
            f"{value}  ({value / total:.0%})",
            (bar.get_x() + bar.get_width() / 2, value),
            ha="center", va="bottom", fontsize=9, color=INK_PRIMARY, xytext=(0, 3), textcoords="offset points",
        )
    ax.set_ylabel("Images")
    ax.set_title(title, fontsize=11, loc="left")
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_images_per_patient(df: pd.DataFrame, class_names: list[str], out_path: Path) -> None:
    """Horizontal bars: images contributed by each (pseudonymised) patient, coloured by class."""
    counts = df.groupby(["folder_label", "patient_id"]).size().reset_index(name="images")
    counts["order"] = counts["folder_label"].map({c: i for i, c in enumerate(class_names)})
    counts = counts.sort_values(["order", "patient_id"], ascending=[False, False])
    colours = _colour_map(class_names)
    fig, ax = _new_figure(6.5, 0.32 * len(counts) + 1.2)
    ax.barh(counts["patient_id"], counts["images"], height=0.65, color=counts["folder_label"].map(colours))
    for y, value in enumerate(counts["images"]):
        ax.annotate(str(value), (value, y), va="center", fontsize=8, color=INK_SECONDARY, xytext=(3, 0),
                    textcoords="offset points")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colours[c]) for c in class_names]
    ax.legend(handles, class_names, frameon=False, fontsize=9, loc="lower right", labelcolor=INK_PRIMARY)
    ax.set_xlabel("Images")
    ax.set_title("Images per patient (pseudonymised IDs)", fontsize=11, loc="left")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sequence_by_class(df: pd.DataFrame, class_names: list[str], out_path: Path) -> None:
    """Annotated count matrix of MRI sequence vs class — exposes acquisition confounding."""
    table = pd.crosstab(df["sequence"], df["folder_label"]).reindex(columns=class_names, fill_value=0)
    fig, ax = _new_figure(5.5, 0.55 * len(table) + 1.5)
    ax.imshow(table.values, cmap="Blues", aspect="auto", vmin=0)
    ax.set_xticks(range(len(table.columns)), table.columns)
    ax.set_yticks(range(len(table.index)), table.index)
    vmax = table.values.max() or 1
    for (i, j), value in np.ndenumerate(table.values):
        ax.text(j, i, str(value), ha="center", va="center", fontsize=10,
                color="#ffffff" if value > 0.55 * vmax else INK_PRIMARY)
    ax.set_title("Images per MRI sequence and class", fontsize=11, loc="left")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_dimension_distribution(df: pd.DataFrame, out_path: Path) -> None:
    """Bar chart of distinct (width x height) image sizes."""
    readable = df[df["readable"]]
    sizes = (readable["image_width"].astype(str) + "×" + readable["image_height"].astype(str)).value_counts()
    fig, ax = _new_figure(5.5, 3.5)
    bars = ax.bar(sizes.index, sizes.values, width=0.5, color=SERIES_COLOURS[0])
    for bar, value in zip(bars, sizes.values):
        ax.annotate(str(value), (bar.get_x() + bar.get_width() / 2, value), ha="center", va="bottom",
                    fontsize=9, color=INK_PRIMARY, xytext=(0, 3), textcoords="offset points")
    ax.set_xlabel("Width × height (px)")
    ax.set_ylabel("Images")
    ax.set_title("Image dimensions", fontsize=11, loc="left")
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_split_distribution(df: pd.DataFrame, class_names: list[str], splits: list[str], out_path: Path) -> None:
    """Grouped bars: images per class within each split."""
    table = pd.crosstab(df["split"], df["folder_label"]).reindex(index=splits, columns=class_names, fill_value=0)
    colours = _colour_map(class_names)
    fig, ax = _new_figure(7, 4)
    width = 0.8 / len(class_names)
    x = np.arange(len(splits))
    for i, label in enumerate(class_names):
        positions = x + (i - (len(class_names) - 1) / 2) * width
        bars = ax.bar(positions, table[label].values, width=width * 0.92, color=colours[label], label=label)
        for bar, value in zip(bars, table[label].values):
            ax.annotate(str(value), (bar.get_x() + bar.get_width() / 2, value), ha="center", va="bottom",
                        fontsize=8, color=INK_SECONDARY, xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(x, splits)
    ax.set_ylabel("Images")
    ax.set_title("Images per class in each split (patient-level)", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_PRIMARY)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sample_grid(
    df: pd.DataFrame,
    class_names: list[str],
    out_path: Path,
    per_class: int,
    seed: int,
    path_column: str = "source_path",
) -> None:
    """Grid of real images: one row per class, samples drawn reproducibly across patients."""
    fig, axes = plt.subplots(len(class_names), per_class, figsize=(2.1 * per_class, 2.4 * len(class_names)),
                             facecolor=SURFACE)
    axes = np.atleast_2d(axes)
    for row, label in enumerate(class_names):
        subset = df[df["folder_label"] == label]
        # Round-robin across patients so one patient does not dominate the row.
        shuffled = subset.sample(frac=1.0, random_state=seed)
        shuffled = shuffled.assign(_rank=shuffled.groupby("patient_id").cumcount()).sort_values("_rank", kind="stable")
        picks = shuffled.head(per_class)
        for col in range(per_class):
            ax = axes[row, col]
            ax.axis("off")
            if col >= len(picks):
                continue
            item = picks.iloc[col]
            with Image.open(item[path_column]) as img:
                ax.imshow(np.asarray(img.convert("L")), cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"{item['patient_id']} · {item['sequence']}", fontsize=8, color=INK_SECONDARY)
        axes[row, 0].text(-0.08, 0.5, label, transform=axes[row, 0].transAxes, rotation=90, ha="right",
                          va="center", fontsize=11, color=INK_PRIMARY)
    fig.suptitle("Sample MRI images per class (real data)", fontsize=12, color=INK_PRIMARY, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
