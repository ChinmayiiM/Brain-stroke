"""Command-line inference.

Usage (project root, venv active):
    python -m src.inference.cli --image dataset/test/Ischemic/ISC-P05_DWI_s020.png
    python -m src.inference.cli --image <path> --json
    python -m src.inference.cli --image <path> --save-figure results/inference/debug.png
    python -m src.inference.cli --dir dataset/test/Normal --output results/inference/normal_batch.csv
    python -m src.inference.cli --images a.png b.jpg --model models/best/efficientnetb0_stroke_mri.keras

Exit codes: 0 success · 2 invalid input image(s) · 3 model/configuration problem.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from src.inference.errors import ImageValidationError, InferenceError  # noqa: E402
from src.inference.predictor import PredictionResult, Predictor, batch_to_frame  # noqa: E402
from src.utils.config import resolve_path  # noqa: E402

EXIT_OK, EXIT_INPUT, EXIT_MODEL = 0, 2, 3


def format_result(result: PredictionResult) -> str:
    width = max(len(name) for name in result.probabilities)
    lines = [
        "-" * 48,
        "MRI PREDICTION",
        "-" * 48,
        f"Image:           {result.image_path}",
        f"Predicted class: {result.predicted_class}",
        f"Confidence:      {result.confidence:.2%}",
        "",
        "Probabilities:",
        *[f"  {name:<{width}}  {p:7.2%}" for name, p in result.probabilities.items()],
        "",
        f"Model: {result.model_path}",
        "-" * 48,
        result.disclaimer,
    ]
    return "\n".join(lines)


def _collect_paths(args: argparse.Namespace, allowed: tuple[str, ...]) -> list[Path]:
    if args.dir:
        directory = Path(args.dir)
        if not directory.is_dir():
            raise ImageValidationError(f"Directory not found: {directory}")
        return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in allowed)
    return [Path(p) for p in args.images]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="path to one MRI image")
    source.add_argument("--images", nargs="+", help="several image paths (batch)")
    source.add_argument("--dir", help="folder of images (batch, non-recursive)")
    parser.add_argument("--model", help="override config inference.model_path")
    parser.add_argument("--json", action="store_true", help="print the structured result as JSON")
    parser.add_argument("--save-figure", help="single image: save a debug figure to this PNG path")
    parser.add_argument("--output", help="batch: CSV path (default: <inference.batch_output_dir>/batch_predictions.csv)")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # Windows cp1252 pipes cannot encode e.g. the em dash
        if (getattr(stream, "encoding", "") or "").lower() not in ("utf-8", "utf8") and hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    try:
        predictor = Predictor(model_path=args.model)
    except InferenceError as exc:
        print(f"ERROR (model): {exc}", file=sys.stderr)
        return EXIT_MODEL

    if args.image:
        try:
            result = predictor.predict(args.image)
        except ImageValidationError as exc:
            print(f"ERROR (image): {exc}", file=sys.stderr)
            return EXIT_INPUT
        except InferenceError as exc:
            print(f"ERROR (prediction): {exc}", file=sys.stderr)
            return EXIT_MODEL
        print(json.dumps(result.to_dict(), indent=2) if args.json else format_result(result))
        if args.save_figure:
            from src.inference.visualize import save_prediction_figure

            out = save_prediction_figure(result, resolve_path(args.save_figure))
            print(f"Debug figure: {out}")
        return EXIT_OK

    try:
        paths = _collect_paths(args, predictor.settings.allowed_extensions)
    except ImageValidationError as exc:
        print(f"ERROR (input): {exc}", file=sys.stderr)
        return EXIT_INPUT
    if not paths:
        print("ERROR (input): no images with a supported extension were found", file=sys.stderr)
        return EXIT_INPUT
    try:
        items = predictor.predict_batch(paths)
    except InferenceError as exc:
        print(f"ERROR (prediction): {exc}", file=sys.stderr)
        return EXIT_MODEL

    frame = batch_to_frame(items, predictor.loaded.class_names)
    output = resolve_path(args.output) if args.output else predictor.settings.batch_output_dir / "batch_predictions.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    n_ok = int((frame["status"] == "ok").sum())
    if args.json:
        print(frame.to_json(orient="records", indent=2))
    else:
        print(frame.drop(columns=["error"]).to_string(index=False))
    for item in items:
        if not item.ok:
            print(f"ERROR (image): {item.error}", file=sys.stderr)
    print(f"\n{n_ok}/{len(items)} images predicted. CSV: {output}")
    print(predictor.settings.disclaimer)
    return EXIT_OK if n_ok == len(items) else EXIT_INPUT


if __name__ == "__main__":
    sys.exit(main())
