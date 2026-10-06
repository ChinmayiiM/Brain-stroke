"""Measure real Streamlit app timings: start-up, first and second Analyze (fresh process).

Usage (project root, venv active):
    python scripts/measure_app_performance.py

Drives app/streamlit_app.py with streamlit.testing.AppTest (same script, same cached
model, no browser), uploads two real test images, clicks "Analyze MRI" for each and
reports wall-clock times plus the timing lines the app writes to logs/streamlit_app.log.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

LOG = ROOT / "logs" / "streamlit_app.log"
IMAGES = [ROOT / "dataset" / "test" / "Normal" / "NOR-P06_DWI_s004.png",
          ROOT / "dataset" / "test" / "Haemorrhagic" / "HAE-P01_SWI_s013.png",
          ROOT / "dataset" / "test" / "Ischemic" / "ISC-P05_DWI_s020.png"]


def main() -> int:
    if (sys.stdout.encoding or "").lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(errors="replace")
    log_offset = LOG.stat().st_size if LOG.exists() else 0

    t = time.perf_counter()
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=600)
    at.run()
    print(f"App start (first run: TensorFlow + model load + warm-up): {time.perf_counter() - t:.2f} s")

    for n, image in enumerate(IMAGES, 1):
        t = time.perf_counter()
        at.file_uploader(key="mri_upload").set_value((image.name, image.read_bytes(), "image/png")).run()
        upload = time.perf_counter() - t
        t = time.perf_counter()
        at.button(key="analyze").click().run()
        analyze = time.perf_counter() - t
        result = next((m.value for m in at.markdown if m.value.startswith('<p class="result-class">')), "")
        label = result.split('result-class">')[1].split("<")[0] if result else "?"
        conf = next((m.value for m in at.metric if m.label == "Confidence"), "?")
        print(f"Image {n} ({image.name}): upload rerun {upload * 1000:.0f} ms | Analyze click rerun "
              f"{analyze * 1000:.0f} ms → {label} {conf}")

    new_log = LOG.read_text(encoding="utf-8", errors="replace")[log_offset:] if LOG.exists() else ""
    lines = [l for l in new_log.splitlines() if any(k in l for k in ("Loaded model", "Warm-up", "Model ready",
                                                                        "Timing", "Analyze click"))]
    print("\nlogs/streamlit_app.log (this run):")
    for line in lines:
        print("  " + line.split(" | ", 3)[-1])
    loads = sum("Loaded model" in l for l in lines)
    print(f"\n.keras model loads during this run: {loads} (expected 1)")
    return 0 if loads == 1 else 1


if __name__ == "__main__":
    sys.exit(main())
