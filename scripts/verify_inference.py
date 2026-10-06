"""Phase 5 consistency check: inference pipeline vs. the Phase 4 evaluation.

Usage (project root, venv active):
    python scripts/verify_inference.py              # 3 test images per class (seeded)
    python scripts/verify_inference.py --per-class 5

For a small, seeded sample of test images it runs the *inference* pipeline
(single-image path: file validation → shared preprocessing → model) and
compares the result with the probabilities recorded by Phase 4's evaluation
(batched tf.data path). Agreement shows both paths feed the model identically.
It is a pipeline-consistency check, NOT a new accuracy measurement — the
official metrics remain results/metrics/metrics.json.

Writes results/inference/phase4_consistency_check.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.inference.predictor import Predictor  # noqa: E402
from src.utils.config import PROJECT_ROOT, load_config, resolve_path  # noqa: E402

PROB_TOLERANCE = 1e-4  # single vs batched float32 inference may differ in the last digits


def main() -> int:
    if (sys.stdout.encoding or "").lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(errors="replace")  # non-UTF-8 console/pipe: never crash on symbols
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-class", type=int, default=3)
    args = parser.parse_args()

    config = load_config()
    predictor = Predictor(config)
    loaded = predictor.loaded
    reference_file = resolve_path(config["artifacts"]["results_dir"]) / "metrics" / "test_predictions.csv"
    reference = pd.read_csv(reference_file)
    seed = int(config["project"]["seed"])
    sample = pd.concat(
        [group.sample(min(args.per_class, len(group)), random_state=seed)
         for _, group in reference.groupby("actual", sort=True)],
        ignore_index=True,
    )

    checks: dict = {
        "model_path": str(loaded.path.relative_to(PROJECT_ROOT)),
        "input_shape": list(loaded.model.input_shape[1:]),
        "output_kind": loaded.output_kind,
        "class_names": list(loaded.class_names),
        "class_names_match_config": list(loaded.class_names) == list(config["data"]["class_names"]),
        "random_augmentation_layers_in_model": 0,  # load_inference_model raises if any exist
        "images": [],
    }

    max_diff = 0.0
    for _, row in sample.iterrows():
        result = predictor.predict(PROJECT_ROOT / row["image_path"])
        ours = np.array([result.probabilities[c] for c in loaded.class_names])
        phase4 = np.array([row[f"prob_{c}"] for c in loaded.class_names])
        diff = float(np.abs(ours - phase4).max())
        max_diff = max(max_diff, diff)
        checks["images"].append({
            "image_path": row["image_path"],
            "actual_label": row["actual"],
            "predicted_class": result.predicted_class,
            "confidence": round(result.confidence, 6),
            "phase4_predicted": row["predicted"],
            "same_prediction_as_phase4": result.predicted_class == row["predicted"],
            "max_probability_difference": diff,
            "probabilities_valid": bool(np.all((ours >= 0) & (ours <= 1)) and abs(ours.sum() - 1) < 1e-6),
        })

    images = checks["images"]
    checks["summary"] = {
        "n_images": len(images),
        "all_predictions_match_phase4": all(i["same_prediction_as_phase4"] for i in images),
        "max_probability_difference": max_diff,
        "all_probabilities_valid": all(i["probabilities_valid"] for i in images),
        "sample_matches_actual_label": sum(i["predicted_class"] == i["actual_label"] for i in images),
        "note": "Pipeline-consistency check on a small sample; not an accuracy estimate.",
    }
    passed = (
        checks["class_names_match_config"]
        and checks["summary"]["all_predictions_match_phase4"]
        and checks["summary"]["all_probabilities_valid"]
        and max_diff <= PROB_TOLERANCE
    )
    checks["passed"] = passed

    out = resolve_path(config["inference"]["batch_output_dir"]) / "phase4_consistency_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(checks, indent=2), encoding="utf-8")

    for item in images:
        print(f"{item['image_path']:<55} actual={item['actual_label']:<13} pred={item['predicted_class']:<13} "
              f"conf={item['confidence']:.4f}  phase4={item['phase4_predicted']:<13} "
              f"max|dp|={item['max_probability_difference']:.2e}")
    print(json.dumps(checks["summary"], indent=2))
    print(f"Consistency check {'PASSED' if passed else 'FAILED'}: {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
