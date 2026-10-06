"""Streamlit front end for the Brain Stroke Detection prototype.

Run from the project root:
    streamlit run app/streamlit_app.py

All validation, preprocessing and prediction is delegated to the Phase 5 pipeline
(src.inference.Predictor). The 3-class result is only *presented* as
STROKE / NO STROKE via src.inference.binary_view. For a STROKE result, the same cached
model is re-run on occluded copies of the image (src.inference.region_view) to outline the
area the stroke score depends on — model interpretability, not a lesion mask.

Start-up: the page is rendered first using only lightweight modules (no TensorFlow).
The model (TensorFlow import → .keras load → warm-up) is then loaded ONCE per server
process in a background thread; "Analyze MRI" waits for it only if it is not ready yet.
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from concurrent.futures import Future
from pathlib import Path

SCRIPT_STARTED = time.perf_counter()
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quieter TensorFlow start-up logs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # `streamlit run` puts app/ (not the root) on sys.path
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

from app import ui  # noqa: E402
from app.services import (  # noqa: E402
    build_model_card,
    model_files_problem,
    model_path_override,
    start_model_loading,
)
from src.inference.binary_view import BinaryViewConfig, to_binary  # noqa: E402
from src.inference.errors import ImageValidationError, InferenceError  # noqa: E402
from src.inference.validation import validate_image_bytes  # noqa: E402  (no TensorFlow)
from src.utils.config import load_config  # noqa: E402
from src.utils.logger import add_file_handler, get_logger  # noqa: E402

logger = get_logger("streamlit_app")
# Development timing (UI render, TF import, model load, warm-up, prediction) → logs/streamlit_app.log
add_file_handler(["streamlit_app", "src.inference.model_loader", "src.inference.predictor",
                  "src.inference.region_view"],
                 PROJECT_ROOT / "logs" / "streamlit_app.log")

st.set_page_config(page_title="Brain Stroke Detection", page_icon="🧠", layout="wide",
                   initial_sidebar_state="expanded")


@st.cache_resource(show_spinner=False)
def get_model_future(model_path: str | None) -> Future:
    """Start loading the model ONCE per server process (background thread); reruns reuse this Future."""
    return start_model_loading(model_path)


@st.cache_data
def get_config() -> dict:
    return load_config()


@st.cache_data
def get_binary_view() -> BinaryViewConfig:
    return BinaryViewConfig.from_config(load_config())


def _wait_for_predictor(model_path: str | None):
    """Return the cached Predictor, waiting for the background load if it is still running."""
    future = get_model_future(model_path)
    if not future.done():
        with st.spinner("Preparing the model (first use only)…"):
            started = time.perf_counter()
            future.result()
            logger.info("Analyze waited %.2f s for the background model load", time.perf_counter() - started)
    try:
        return future.result()
    except Exception:
        get_model_future.clear()  # don't keep a failed load cached — retry on the next run
        raise


def main() -> None:
    ui.inject_css()
    config = get_config()
    override = model_path_override()
    load_error = model_files_problem(config, override)  # cheap file check, no TensorFlow
    card = None
    view = get_binary_view()
    if load_error is None:
        try:
            card = build_model_card(config, override)  # from model_metadata.json, no model needed
            view.validate(card.class_names)
        except (InferenceError, OSError, ValueError, KeyError) as exc:
            load_error = str(exc)
    page = ui.current_page()
    ui.render_sidebar(page)
    ui.render_header()

    # Exactly one page per run; the sidebar buttons only change st.session_state["page"].
    if page in ("home", "upload"):
        if load_error is not None:
            logger.error("Model unavailable: %s", load_error)
            st.error(f"The trained model is not available, so analysis is disabled.\n\n**Details:** {load_error}",
                     icon=":material/error:")
        else:
            _render_analysis(config, view, override)
    elif page == "model":
        ui.render_model_page(card)
    elif page == "about":
        ui.render_about_page()
    elif page == "how":
        ui.render_how_page()

    if load_error is None:
        # The whole page is on screen now; start (or reuse) the one-time background model load.
        future = get_model_future(override)
        if not future.done():
            logger.info("UI rendered in %.2f s; model still loading in the background",
                        time.perf_counter() - SCRIPT_STARTED)


def _analyze(data: bytes, filename: str, view: BinaryViewConfig, override: str | None, config: dict) -> None:
    """Run the existing prediction (and, for STROKE, the region view) and keep the results in the session."""
    try:
        predictor = _wait_for_predictor(override)
        view.validate(predictor.loaded.class_names)
        with st.spinner("Analysing image…"):
            started = time.perf_counter()
            prediction = predictor.predict_bytes(data, filename)  # real 3-class output, cached model
            st.session_state["result"] = to_binary(prediction, view)
            logger.info("Analyze click: %.1f ms end-to-end, result %s (probabilities %s)",
                        (time.perf_counter() - started) * 1000, st.session_state["result"].label,
                        {k: round(v, 4) for k, v in prediction.probabilities.items()})
    except ImageValidationError as exc:
        st.error(f"The image could not be processed. {exc}", icon=":material/broken_image:")
        return
    except InferenceError as exc:
        logger.error("Inference failed: %s", exc)
        st.error(f"The model could not produce a valid prediction. {exc}", icon=":material/error:")
        return
    except Exception:  # noqa: BLE001 — never show a raw traceback to end users
        logger.exception("Unexpected error during prediction")
        st.error("An unexpected error occurred during analysis. Details were written to the "
                 "server log.", icon=":material/error:")
        return

    if st.session_state["result"].is_positive:
        from src.inference.region_view import RegionViewConfig, stroke_region  # OpenCV/SciPy, STROKE only

        with st.spinner("Measuring which image area the stroke score depends on…"):
            try:
                st.session_state["region"] = stroke_region(predictor, data, filename, view,
                                                           RegionViewConfig.from_config(config))
            except Exception:  # noqa: BLE001 — the prediction stands even if the region view fails
                logger.exception("Region view failed")


def _store_upload() -> None:
    """Keep the upload in the session: the uploader widget forgets its file while another page is shown."""
    uploaded = st.session_state.get("mri_upload")
    if uploaded is None:  # the user removed the file
        st.session_state.pop("upload", None)
    else:
        st.session_state["upload"] = (uploaded.getvalue(), uploaded.name)


def _render_analysis(config: dict, view: BinaryViewConfig, override: str | None) -> None:
    allowed = tuple(e.lower() for e in config["data"]["allowed_extensions"])
    data = filename = info = None
    upload_col, result_col = st.columns(2, gap="large")

    with upload_col:
        with st.container(border=True, height="stretch", key="mri_panel"):
            ui.render_upload_heading()
            uploaded = st.file_uploader(
                "Upload Brain MRI Image",
                type=[ext.lstrip(".") for ext in allowed],
                accept_multiple_files=False,
                help="One 2D MRI slice, 8-bit grayscale or RGB.",
                key="mri_upload",
                label_visibility="collapsed",
                on_change=_store_upload,
            )
            stored = st.session_state.get("upload")
            if stored is not None:
                data, filename = stored
                if uploaded is None:  # back from another page: same image, widget is empty again
                    st.caption("Showing your last uploaded image. Upload a new file to replace it.")
                image_key = hashlib.sha256(data).hexdigest()
                if st.session_state.get("result_key") != image_key:  # new image → discard previous results
                    for key in ("result", "region"):
                        st.session_state.pop(key, None)
                    st.session_state["result_key"] = image_key
                try:
                    info = validate_image_bytes(data, filename, allowed)  # Phase 5 validation, before preview
                    ui.render_upload_details(data, filename, info)
                except ImageValidationError as exc:
                    st.error(f"This file cannot be analysed. {exc}", icon=":material/broken_image:")

    with result_col:
        with st.container(border=True, height="stretch", key="result_panel"):
            ui.render_result_heading()
            if info is None:
                st.caption("Upload a brain MRI image to enable analysis.")
            else:
                if st.button("Analyze MRI", type="primary", width="stretch", icon=":material/neurology:",
                             key="analyze"):
                    _analyze(data, filename, view, override, config)
                if st.session_state.get("result") is not None:
                    ui.render_result(st.session_state["result"])
                else:
                    st.caption("Click **Analyze MRI** to run the trained model on this image.")

    result = st.session_state.get("result") if info is not None else None
    if result is not None and result.is_positive:
        min_drop = float(config["inference"]["region_view"]["min_drop"])
        ui.render_region_section(data, st.session_state.get("region"), min_drop)
    if result is not None:
        ui.render_summary(result)


main()
