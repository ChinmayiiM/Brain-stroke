"""Verify the Python environment is ready for the project.

Usage (from the project root, with the venv activated):
    python scripts/verify_environment.py

Checks Python version, imports every required package, reports versions,
and confirms TensorFlow can construct EfficientNetB0 (architecture only —
no pretrained weights are downloaded and nothing is trained).
"""

from __future__ import annotations

import importlib
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# (import name, display name)
REQUIRED = [
    ("tensorflow", "tensorflow"),
    ("keras", "keras"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("cv2", "opencv-python-headless"),
    ("PIL", "pillow"),
    ("sklearn", "scikit-learn"),
    ("matplotlib", "matplotlib"),
    ("streamlit", "streamlit"),
    ("yaml", "pyyaml"),
]


def main() -> int:
    ok = True
    py = sys.version_info
    print(f"Python      : {sys.version.split()[0]} ({sys.executable})")
    if not ((3, 9) <= (py.major, py.minor) <= (3, 12)):
        print("  WARNING: TensorFlow officially supports Python 3.9-3.12; 3.11 is recommended.")

    print("\nPackages:")
    for module_name, display in REQUIRED:
        try:
            module = importlib.import_module(module_name)
            print(f"  [OK]   {display:<24} {getattr(module, '__version__', '?')}")
        except Exception as exc:  # noqa: BLE001 — report any import failure
            ok = False
            print(f"  [FAIL] {display:<24} {type(exc).__name__}: {exc}")

    if ok:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        print(f"\nGPU devices : {len(gpus)} {'(training will run on CPU)' if not gpus else ''}")

        model = tf.keras.applications.EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
        print(f"EfficientNetB0 backbone builds: {model.count_params():,} parameters (no weights loaded)")

    print("\nEnvironment OK" if ok else "\nEnvironment has problems — see [FAIL] lines above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
