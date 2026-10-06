"""Phase 7 end-to-end integration checks.

Usage (project root, venv active):
    python scripts/run_integration_checks.py            # all checks incl. Streamlit server + app tests
    python scripts/run_integration_checks.py --quick    # skip the Streamlit server start and AppTest suite

Every check below is executed for real; the report records the observed values:
    results/integration_test_report.md   (human-readable)
    results/integration_test_report.json (machine-readable)

Nothing here trains, modifies or overwrites the model, dataset or Phase 4 artifacts.
Edge-case files are created in a temporary directory that is deleted afterwards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from src.data.preparation import OutputDirs, verify_prepared_dataset  # noqa: E402
from src.evaluation.leakage import check_leakage  # noqa: E402
from src.inference.errors import ImageValidationError, ModelNotFoundError, ModelOutputError  # noqa: E402
from src.inference.model_loader import clear_model_cache  # noqa: E402
from src.inference.predictor import Predictor  # noqa: E402
from src.training.data_loader import build_dataset, get_split, load_metadata  # noqa: E402
from src.training.settings import TrainingSettings  # noqa: E402
from src.utils.config import PROJECT_ROOT, load_config, resolve_path  # noqa: E402

PYTHON = sys.executable


@dataclass
class Check:
    area: str
    name: str
    expected: str
    actual: str
    status: str  # PASS / FAIL
    notes: str = ""


CHECKS: list[Check] = []


def _portable(text: str) -> str:
    """Mask machine-specific locations so the committed report contains no local paths/user names."""
    for path, label in ((tempfile.gettempdir(), "<TMP>"), (str(PROJECT_ROOT), "<PROJECT>"), (str(Path.home()), "<HOME>")):
        text = text.replace(path, label).replace(path.replace("\\", "\\\\"), label)
    return text


def record(area: str, name: str, expected: str, actual: str, passed: bool, notes: str = "") -> None:
    actual, notes = _portable(actual), _portable(notes)
    CHECKS.append(Check(area, name, expected, actual, "PASS" if passed else "FAIL", notes))
    print(f"[{'PASS' if passed else 'FAIL'}] {area} · {name}: {actual}")


def guarded(area: str, name: str, expected: str):
    """Decorator: run a check function; any exception becomes a FAIL with the error text."""
    def wrap(fn):
        def inner(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                record(area, name, expected, f"{type(exc).__name__}: {exc}", False)
        return inner
    return wrap


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------------------------

@guarded("Dataset", "Prepared dataset integrity", "0 problems")
def check_dataset(config: dict, settings: TrainingSettings) -> pd.DataFrame:
    metadata = load_metadata(settings)
    problems = verify_prepared_dataset(metadata, OutputDirs.from_config(config, resolve_path),
                                       list(settings.class_names))
    counts = pd.crosstab(metadata["split"], metadata["class_name"]).to_dict(orient="index")
    record("Dataset", "Prepared dataset integrity", "files exist, open, match hashes/labels/splits; 0 problems",
           f"{len(problems)} problems; {len(metadata)} images; per split/class {counts}", not problems,
           "verify_prepared_dataset opens every image and checks hashes, label folders, split folders")
    patients = metadata.groupby("patient_id")["split"].nunique()
    hashes = metadata.groupby("file_sha256")["split"].nunique()
    record("Dataset", "Split separation", "no patient or identical image in >1 split",
           f"patients in >1 split: {int((patients > 1).sum())}; identical images in >1 split: {int((hashes > 1).sum())}",
           (patients > 1).sum() == 0 and (hashes > 1).sum() == 0)
    shares = (metadata["class_name"].value_counts(normalize=True) * 100).round(1).to_dict()
    record("Dataset", "Class balance (informational)", "documented imbalance", f"class share % {shares}", True,
           "Imbalanced by design of the source dataset; handled with class weights (Phase 3/4)")
    return metadata


# ---------------------------------------------------------------------------------------------
# 2. Preprocessing consistency: training pipeline vs CLI path vs Streamlit (bytes) path
# ---------------------------------------------------------------------------------------------

@guarded("Preprocessing", "Training = inference = Streamlit inputs", "bitwise identical arrays")
def check_preprocessing(settings: TrainingSettings, predictor: Predictor, metadata: pd.DataFrame) -> None:
    spec = get_split(metadata, "validation", settings)
    take = min(settings.batch_size, spec.size)
    train_path_batch = np.asarray(next(iter(build_dataset(spec, settings, training=False)))[0])[:take]
    cli_batch = np.stack([predictor.preprocess(p) for p in spec.paths[:take]])
    app_batch = np.stack([predictor.preprocess_bytes(Path(p).read_bytes(), Path(p).name) for p in spec.paths[:take]])
    same = np.array_equal(train_path_batch, cli_batch) and np.array_equal(cli_batch, app_batch)
    record("Preprocessing", "Training = inference = Streamlit inputs", "bitwise identical arrays",
           f"{take} validation images; identical={same}; shape={cli_batch.shape[1:]}, dtype={cli_batch.dtype}, "
           f"range=[{cli_batch.min():.0f}, {cli_batch.max():.0f}]",
           same and cli_batch.dtype == np.float32 and cli_batch.shape[1:] == settings.input_shape,
           "All three paths call src.preprocessing.image_ops.preprocess_image")


# ---------------------------------------------------------------------------------------------
# 3. Model integrity and class mapping
# ---------------------------------------------------------------------------------------------

@guarded("Model", "Model integrity", "loads; shapes match config; is the selected checkpoint")
def check_model(config: dict, predictor: Predictor) -> None:
    loaded = predictor.loaded
    model = loaded.model
    summary = json.loads((resolve_path(config["artifacts"]["results_dir"]) / "metrics" / "training_summary.json")
                         .read_text(encoding="utf-8"))
    selection = json.loads((resolve_path(config["artifacts"]["results_dir"]) / "metrics" / "model_selection.json")
                           .read_text(encoding="utf-8"))
    checkpoint = resolve_path(config["artifacts"]["checkpoints_dir"]) / selection["selected_checkpoint"]
    same_file = sha256(loaded.path) == sha256(checkpoint)
    expected_params = summary["stage1_freeze_check"]["total"]
    record("Model", "Model integrity", "loads; input (None,224,224,3); output (None,3) softmax; "
           "file identical to validation-selected checkpoint",
           f"input {model.input_shape}, output {model.output_shape} {loaded.output_kind}, "
           f"params {model.count_params():,} (training record {expected_params:,}); "
           f"identical to {checkpoint.name}: {same_file}",
           model.input_shape[1:] == tuple(settings_input(config)) and model.output_shape[-1] == len(loaded.class_names)
           and model.count_params() == expected_params and same_file,
           f"model file SHA-256 {sha256(loaded.path)[:16]}…")


def settings_input(config: dict) -> list[int]:
    return [*config["preprocessing"]["image_size"], config["preprocessing"]["channels"]]


@guarded("Class mapping", "Index → class name is identical everywhere", "same order in all components")
def check_class_mapping(config: dict, predictor: Predictor, metadata: pd.DataFrame) -> None:
    results_dir = resolve_path(config["artifacts"]["results_dir"])
    config_names = list(config["data"]["class_names"])
    model_meta = list(predictor.loaded.metadata["class_names"])
    dataset_map = metadata.drop_duplicates("label").sort_values("label")["class_name"].tolist()
    test_pred = pd.read_csv(results_dir / "metrics" / "test_predictions.csv")
    eval_cols = [c.removeprefix("prob_") for c in test_pred.columns if c.startswith("prob_")]
    eval_index_ok = all(config_names[i] == n for i, n in zip(test_pred["predicted_index"], test_pred["predicted"]))
    cm_labels = json.loads((results_dir / "metrics" / "metrics.json").read_text(encoding="utf-8"))[
        "confusion_matrix"]["labels"]
    sources = {"config": config_names, "dataset metadata (label idx)": dataset_map, "model_metadata.json": model_meta,
               "inference Predictor": list(predictor.loaded.class_names), "evaluation prob columns": eval_cols,
               "confusion matrix labels": cm_labels}
    consistent = all(v == config_names for v in sources.values()) and eval_index_ok
    record("Class mapping", "Index → class name is identical everywhere", "same order in all components",
           f"{' | '.join(f'{k}: {v}' for k, v in sources.items())}; evaluation index→name consistent: {eval_index_ok}",
           consistent, "Streamlit renders PredictionResult.probabilities, whose keys come from the Predictor")


# ---------------------------------------------------------------------------------------------
# 4. Phase 4 artifacts and leakage
# ---------------------------------------------------------------------------------------------

@guarded("Evaluation", "Phase 4 artifacts present", "all files exist")
def check_artifacts(config: dict) -> None:
    results = resolve_path(config["artifacts"]["results_dir"])
    required = ["metrics/metrics.json", "metrics/classification_report.txt", "metrics/test_predictions.csv",
                "metrics/training_history.csv", "metrics/model_selection.json", "metrics/stage_comparison.csv",
                "plots/accuracy.png", "plots/loss.png", "plots/confusion_matrix.png", "plots/roc_curves.png",
                "analysis/misclassified_images.csv", "analysis/leakage_checks.json"]
    missing = [r for r in required if not (results / r).is_file()]
    m = json.loads((results / "metrics/metrics.json").read_text(encoding="utf-8"))
    record("Evaluation", "Phase 4 artifacts present", f"{len(required)} files exist",
           f"missing: {missing or 'none'}; recorded test accuracy {m['accuracy']:.4f}, macro F1 {m['macro_f1']:.4f}, "
           f"macro ROC-AUC {m['roc_auc']['macro_ovr']:.4f} (n={m['n_images']})", not missing,
           "Values read from the Phase 4 file — not recomputed and not derived from app uploads")


@guarded("Leakage", "No test data in training/selection", "leakage checks pass")
def check_no_leakage(config: dict, metadata: pd.DataFrame) -> None:
    results = resolve_path(config["artifacts"]["results_dir"])
    report = check_leakage(metadata, results / "metrics" / "model_selection.json",
                           resolve_path(config["paths"]["reports"]) / "data_analysis" / "near_duplicates.csv")
    aug_in_model = any(type(l).__name__.startswith("Random") for l in _layers(Predictor(config).loaded.model))
    record("Leakage", "No test data in training/selection", "all blocking checks pass; augmentation train-only",
           f"passed={report['passed']}; selection recorded before test={report['model_selection_recorded_before_test']}; "
           f"train.py requests test split={report['training_code_reads_test_split']}; "
           f"test↔fit near-duplicates={len(report['near_duplicate_pairs_test_vs_fit'] or [])}; "
           f"augmentation layers inside saved model={aug_in_model}",
           report["passed"] and not aug_in_model,
           "Augmentation is applied only in the training tf.data pipeline (data_loader.build_dataset training=True)")


def _layers(model):
    out = []
    for layer in model.layers:
        out.append(layer)
        if hasattr(layer, "layers"):
            out.extend(_layers(layer))
    return out


# ---------------------------------------------------------------------------------------------
# 5. CLI inference on real images
# ---------------------------------------------------------------------------------------------

@guarded("CLI", "Real images via CLI", "valid JSON, probs sum to 1, match Phase 4")
def check_cli(config: dict, metadata: pd.DataFrame) -> None:
    results = resolve_path(config["artifacts"]["results_dir"])
    reference = pd.read_csv(results / "metrics" / "test_predictions.csv").set_index("image_path")
    picks = metadata[metadata["split"] == "test"].groupby("class_name").head(1)["image_path"].tolist()
    outputs = []
    for rel in picks:
        start = time.perf_counter()
        proc = subprocess.run([PYTHON, "-m", "src.inference.cli", "--image", rel, "--json"], cwd=PROJECT_ROOT,
                              capture_output=True, text=True, encoding="utf-8", errors="replace")
        elapsed = time.perf_counter() - start
        result = json.loads(proc.stdout)
        probs = result["probabilities"]
        ref = reference.loc[rel]
        diff = max(abs(probs[c] - ref[f"prob_{c}"]) for c in probs)
        outputs.append((rel, proc.returncode, result["predicted_class"], result["confidence"], sum(probs.values()),
                        diff, elapsed))
    ok = all(rc == 0 and abs(total - 1) < 1e-6 and diff < 1e-4 for _, rc, _, _, total, diff, _ in outputs)
    distinct = len({round(c, 6) for *_, c, _, _, _ in [(o[0], o[1], o[2], o[3], o[4], o[5], o[6]) for o in outputs]})
    detail = "; ".join(f"{Path(r).name} → {cls} {conf:.2%} (Σp={tot:.6f}, |Δp| vs Phase 4={d:.1e}, {t:.1f}s)"
                       for r, _, cls, conf, tot, d, t in outputs)
    record("CLI", "Real images via CLI", "exit 0, valid JSON, Σp=1, same as Phase 4 evaluation", detail,
           ok and distinct > 1, f"{distinct} distinct confidences across {len(outputs)} images (not a fixed output)")


# ---------------------------------------------------------------------------------------------
# 6. Edge cases (temporary files, deleted afterwards)
# ---------------------------------------------------------------------------------------------

def check_edge_cases(config: dict, predictor: Predictor, metadata: pd.DataFrame) -> None:
    source = PROJECT_ROOT / metadata[metadata["split"] == "test"].iloc[0]["image_path"]
    base = Image.open(source)
    with tempfile.TemporaryDirectory(prefix="stroke_edge_") as tmp:
        tmp = Path(tmp)

        def expect_ok(name: str, path: Path, note: str = "") -> None:
            try:
                r = predictor.predict(path)
                valid = abs(sum(r.probabilities.values()) - 1) < 1e-6 and r.predicted_class in predictor.loaded.class_names
                record("Edge cases", name, "valid prediction", f"{r.predicted_class} {r.confidence:.2%}", valid, note)
            except Exception as exc:  # noqa: BLE001
                record("Edge cases", name, "valid prediction", f"{type(exc).__name__}: {exc}", False, note)

        def expect_error(name: str, fn, error_type, note: str = "") -> None:
            try:
                fn()
                record("Edge cases", name, f"{error_type.__name__} with clear message", "no error raised", False, note)
            except error_type as exc:
                record("Edge cases", name, f"{error_type.__name__} with clear message", str(exc)[:150], True, note)
            except Exception as exc:  # noqa: BLE001
                record("Edge cases", name, f"{error_type.__name__}", f"{type(exc).__name__}: {exc}", False, note)

        base.convert("RGB").save(tmp / "scan.jpg", quality=95)
        base.convert("RGB").save(tmp / "scan.jpeg", quality=95)
        base.convert("L").save(tmp / "gray.png")
        base.convert("RGB").save(tmp / "rgb.png")
        base.convert("L").resize((32, 32)).save(tmp / "tiny.png")
        base.convert("L").resize((2048, 2048)).save(tmp / "large.png")
        (tmp / "corrupted.png").write_bytes(b"\x89PNG\r\n\x1a\n" + os.urandom(64))
        base.convert("RGB").save(tmp / "scan.bmp")
        (tmp / "notes.txt").write_text("not an image", encoding="utf-8")

        expect_ok("Valid PNG (original)", source)
        expect_ok("Valid JPG", tmp / "scan.jpg", "JPEG re-encoding is lossy; prediction may differ slightly")
        expect_ok("Valid JPEG", tmp / "scan.jpeg")
        expect_ok("Grayscale (mode L)", tmp / "gray.png")
        expect_ok("RGB (mode RGB)", tmp / "rgb.png")
        expect_ok("Large image (2048×2048)", tmp / "large.png", "downscaled to 224×224 by the shared preprocessing")
        expect_error("Very small image (32×32)", lambda: predictor.predict(tmp / "tiny.png"), ImageValidationError,
                     "below config cleaning.min_dimension")
        expect_error("Corrupted image", lambda: predictor.predict(tmp / "corrupted.png"), ImageValidationError)
        expect_error("Unsupported format (.bmp)", lambda: predictor.predict(tmp / "scan.bmp"), ImageValidationError)
        expect_error("Non-image file (.txt)", lambda: predictor.predict(tmp / "notes.txt"), ImageValidationError)
        expect_error("Missing file", lambda: predictor.predict(tmp / "missing.png"), ImageValidationError)
        expect_error("Empty upload (bytes)", lambda: predictor.predict_bytes(b"", "scan.png"), ImageValidationError)
        expect_error("Missing model", lambda: Predictor(config, model_path=tmp / "absent.keras"), ModelNotFoundError,
                     "real model untouched; a non-existent path is passed instead")
        expect_error("Unexpected model output (wrong shape)",
                     lambda: predictor._probabilities(np.ones((1, 5))), ModelOutputError,
                     "output-validation branch exercised directly")
        expect_error("Unexpected model output (not a distribution)",
                     lambda: predictor._probabilities(np.full((1, 3), 0.9)), ModelOutputError)
    record("Edge cases", "Temporary files cleaned up", "temp dir removed", f"exists after run: {tmp.exists()}",
           not tmp.exists())


# ---------------------------------------------------------------------------------------------
# 7. Performance
# ---------------------------------------------------------------------------------------------

@guarded("Performance", "Model load and prediction latency", "measured")
def check_performance(config: dict, metadata: pd.DataFrame) -> None:
    clear_model_cache()
    start = time.perf_counter()
    predictor = Predictor(config)
    load_s = time.perf_counter() - start
    start = time.perf_counter()
    Predictor(config)
    cached_s = time.perf_counter() - start
    image = PROJECT_ROOT / metadata[metadata["split"] == "test"].iloc[1]["image_path"]
    predictor.predict(image)  # warm-up (graph tracing)
    times = []
    for _ in range(5):
        start = time.perf_counter()
        predictor.predict(image)
        times.append(time.perf_counter() - start)
    record("Performance", "Model load and prediction latency", "model cached after first load (latency recorded, CPU-dependent)",
           f"first load {load_s:.2f}s; cached load {cached_s:.3f}s; prediction mean {np.mean(times) * 1000:.0f} ms "
           f"(min {min(times) * 1000:.0f}, max {max(times) * 1000:.0f}) on {platform.processor() or 'CPU'}",
           cached_s < load_s, "Streamlit wraps the Predictor in st.cache_resource → one load per server process")


# ---------------------------------------------------------------------------------------------
# 8. Streamlit
# ---------------------------------------------------------------------------------------------

@guarded("Streamlit", "Server starts", "health endpoint ok, page served")
def check_streamlit_server() -> None:
    port = 8599
    proc = subprocess.Popen(
        [PYTHON, "-m", "streamlit", "run", "app/streamlit_app.py", "--server.headless", "true",
         "--server.port", str(port)], cwd=PROJECT_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        start, health, page = time.perf_counter(), None, None
        while time.perf_counter() - start < 120 and health != "ok":
            try:
                health = urllib.request.urlopen(f"http://localhost:{port}/_stcore/health", timeout=5).read().decode()
            except OSError:
                time.sleep(1)
        if health == "ok":
            page = urllib.request.urlopen(f"http://localhost:{port}/", timeout=10).status
        record("Streamlit", "Server starts", "`streamlit run app/streamlit_app.py` → health 'ok', HTTP 200",
               f"health={health!r}, page status={page}, startup {time.perf_counter() - start:.1f}s",
               health == "ok" and page == 200, "local only (localhost); started headless on a test port")
    finally:
        proc.terminate()
        proc.wait(timeout=30)


@guarded("Streamlit", "App behaviour (AppTest suite)", "all app tests pass")
def check_app_tests() -> None:
    proc = subprocess.run([PYTHON, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-W", "ignore::DeprecationWarning",
                           "tests/test_app.py", "-rA"], cwd=PROJECT_ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    lines = [l for l in proc.stdout.splitlines()
             if re.match(r"^(PASSED|FAILED|SKIPPED|ERROR) tests/", l)]  # result lines only, not log lines
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr[-300:]
    for line in lines:
        status, _, name = line.partition(" ")
        record("Streamlit", name.split("::")[-1].split(" ")[0], "see tests/test_app.py", status, status == "PASSED",
               "AppTest drives the real script with the real model")
    record("Streamlit", "App behaviour (AppTest suite)", "all app tests pass", tail, proc.returncode == 0)


# ---------------------------------------------------------------------------------------------
# 9. Privacy & wording (static)
# ---------------------------------------------------------------------------------------------

@guarded("Privacy", "No persistence / no external calls", "uploads handled in memory only")
def check_privacy() -> None:
    sources = list((PROJECT_ROOT / "app").glob("*.py")) + list((PROJECT_ROOT / "src" / "inference").glob("*.py"))
    # builtin open( (not Image.open / _open), file writes, temp files, network clients
    pattern = re.compile(r"(?<![\w.])open\(|\.write_bytes|\.write_text|tempfile|requests\.|urllib|socket|https?://")
    unexpected = []
    for path in sources:
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#")[0]
            if pattern.search(code) and "dataset_url" not in code and "source_url" not in code:
                unexpected.append(f"{path.relative_to(PROJECT_ROOT)}:{i}: {code.strip()[:80]}")
    stats_off = "gatherUsageStats = false" in (PROJECT_ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    record("Privacy", "No persistence / no external calls", "no file writes, temp files or network calls in app/inference; "
           "telemetry off", f"unexpected hits: {unexpected or 'none'}; Streamlit usage stats disabled: {stats_off}",
           not unexpected and stats_off,
           "Uploads stay in memory (UploadedFile.getvalue()); Streamlit keeps them in its in-memory file manager "
           "for the session only")


@guarded("Safety wording", "No overclaiming language in UI", "no forbidden phrases; disclaimer present")
def check_wording() -> None:
    forbidden = ["100% accurate", "guaranteed", "definitively", "doctor replacement", "diagnosis certainty",
                 "detects stroke with certainty", "clinically validated"]
    text = "\n".join(p.read_text(encoding="utf-8").lower() for p in (PROJECT_ROOT / "app").glob("*.py"))
    found = [f for f in forbidden if f in text]
    confidence_label = 'st.metric("confidence"' in text  # labelled "Confidence", not "certainty"
    record("Safety wording", "No overclaiming language in UI", "no forbidden phrases; 'Confidence' label",
           f"forbidden found: {found or 'none'}; 'Confidence' label: {confidence_label}",
           not found and confidence_label,
           "The on-screen disclaimer was removed at the owner's request; it remains in README.md")


# ---------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------

def write_report(config: dict, started: datetime) -> Path:
    out_dir = resolve_path(config["artifacts"]["results_dir"])
    passed = sum(c.status == "PASS" for c in CHECKS)
    lines = [
        "# Integration Test Report (Phase 7)",
        "",
        f"Generated by `python scripts/run_integration_checks.py` on {started:%Y-%m-%d %H:%M} "
        f"({platform.system()} {platform.release()}, Python {platform.python_version()}).",
        "Every row is the observed outcome of a check that was actually executed.",
        "",
        f"**Result: {passed}/{len(CHECKS)} checks passed.**",
        "",
        "| Area | Test | Expected | Actual | Status | Notes |",
        "|---|---|---|---|---|---|",
    ]
    for c in CHECKS:
        cells = [c.area, c.name, c.expected, c.actual, f"**{c.status}**", c.notes]
        lines.append("| " + " | ".join(str(x).replace("|", "\\|").replace("\n", " ") for x in cells) + " |")
    lines += ["", "Model metrics quoted here are read from the Phase 4 evaluation; app/CLI sample predictions are "
              "pipeline checks, not accuracy measurements. AI-assisted research prototype — not a medical diagnosis."]
    (out_dir / "integration_test_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "integration_test_report.json").write_text(
        json.dumps({"generated": started.isoformat(), "passed": passed, "total": len(CHECKS),
                    "checks": [asdict(c) for c in CHECKS]}, indent=2), encoding="utf-8")
    return out_dir / "integration_test_report.md"


def main() -> int:
    if (sys.stdout.encoding or "").lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="skip Streamlit server start and AppTest suite")
    args = parser.parse_args()
    started = datetime.now()

    config = load_config()
    settings = TrainingSettings.from_config(config)
    predictor = Predictor(config)

    metadata = check_dataset(config, settings)
    if metadata is None:
        metadata = load_metadata(settings)
    check_preprocessing(settings, predictor, metadata)
    check_model(config, predictor)
    check_class_mapping(config, predictor, metadata)
    check_artifacts(config)
    check_no_leakage(config, metadata)
    check_cli(config, metadata)
    check_edge_cases(config, predictor, metadata)
    check_performance(config, metadata)
    if not args.quick:
        check_streamlit_server()
        check_app_tests()
    check_privacy()
    check_wording()

    report = write_report(config, started)
    failed = [c for c in CHECKS if c.status != "PASS"]
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed. Report: {report}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
