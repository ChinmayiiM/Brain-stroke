"""Application-level tests for the Streamlit app (streamlit.testing AppTest).

They drive the real app script with the real trained model and real test-split
images, and compare what is displayed with a direct Phase 5 Predictor call
followed by the binary STROKE / NO STROKE view. Skipped if the model or dataset is missing.
"""

import re

import pytest
import streamlit as st

import app.services as services
from src.inference.binary_view import BinaryViewConfig, to_binary
from src.inference.predictor import Predictor
from src.utils.config import PROJECT_ROOT, load_config, resolve_path

APP_FILE = str(PROJECT_ROOT / "app" / "streamlit_app.py")
CONFIG = load_config()
MODEL_FILE = resolve_path(CONFIG["inference"]["model_path"])
TEST_DIR = PROJECT_ROOT / "dataset" / "test"
IMAGE_A = TEST_DIR / "Normal" / "NOR-P06_DWI_s004.png"
IMAGE_B = TEST_DIR / "Ischemic" / "ISC-P05_DWI_s020.png"
CARD_RE = re.compile(r'<div class="result-card result-card--(?:stroke|no-stroke)">.*?'
                     r'<p class="result-class">([^<]+)</p><p class="result-confidence">Confidence: ([^<]+)</p>')
REGION_TITLE = "Suspected Stroke Region"

pytestmark = pytest.mark.skipif(
    not (MODEL_FILE.is_file() and IMAGE_A.is_file() and IMAGE_B.is_file()),
    reason="trained model or test split not available",
)


def _app():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_FILE, default_timeout=240)
    at.run()
    return at


def _upload(at, path):
    at.file_uploader(key="mri_upload").set_value((path.name, path.read_bytes(), "image/png")).run()


def _displayed_result(at):
    match = next((CARD_RE.search(m.value) for m in at.main.markdown if CARD_RE.search(m.value)), None)
    return match.groups() if match else None


def _expected(path):
    prediction = Predictor(CONFIG).predict(path)
    return prediction, to_binary(prediction, BinaryViewConfig.from_config(CONFIG))


def test_original_layout_is_preserved():
    at = _app()
    assert not at.exception and not at.error
    html = "\n".join(m.value for m in at.markdown)
    # header
    assert "Brain Stroke Detection" in html and "Brain MRI Image Classification" in html
    assert "AI-Assisted Brain MRI Analysis" not in html
    # sidebar branding + one navigation button per page
    sidebar = "\n".join(m.value for m in at.sidebar.markdown)
    assert "Brain Stroke Detection" in sidebar and "Brain MRI Image Classification" in sidebar
    assert [b.label for b in at.sidebar.button] == ["Home", "Upload MRI", "Model & Method", "About", "How It Works"]
    # Home = upload + result cards only
    headings = [s.value for s in at.main.subheader]
    assert len(headings) == 2 and headings[0].endswith("Upload MRI") and headings[1].endswith("Result")
    assert at.file_uploader(key="mri_upload").label == "Upload Brain MRI Image"


PAGE_HEADINGS = {
    "home": ["Upload MRI", "Result"],
    "upload": ["Upload MRI", "Result"],
    "model": ["Model", "Purpose", "Details", "Measured on the held-out test split", "Pipeline"],
    "about": ["About this project"],
    "how": ["From upload to result"],
}


@pytest.mark.parametrize("page", list(PAGE_HEADINGS))
def test_each_sidebar_item_renders_only_its_own_page(page):
    at = _app()
    at.sidebar.button(key=f"nav_{page}").click().run()
    assert not at.exception and not at.error
    headings = [s.value.split(": ")[-1] if s.value.startswith(":material/") else s.value for s in at.main.subheader]
    assert headings == PAGE_HEADINGS[page]
    assert len(at.main.get("file_uploader")) == (1 if page in ("home", "upload") else 0)


def test_model_page_shows_real_test_metrics():
    at = _app()
    at.sidebar.button(key="nav_model").click().run()
    meta = Predictor(CONFIG).loaded.metadata["test_evaluation"]
    text = "\n".join(m.value for m in at.main.markdown)
    assert f"{meta['accuracy']:.1%}" in text and f"{meta['balanced_accuracy']:.1%}" in text
    assert f"{meta['macro_f1']:.3f}" in text
    assert any("68 test images from 3 patients" in c.value for c in at.main.caption)


def test_upload_and_result_survive_switching_pages():
    at = _app()
    _upload(at, IMAGE_A)
    at.button(key="analyze").click().run()
    before = _displayed_result(at)
    at.sidebar.button(key="nav_model").click().run()
    assert _displayed_result(at) is None                              # analysis not rendered on Model & Method
    at.sidebar.button(key="nav_home").click().run()
    assert _displayed_result(at) == before and len(at.get("image")) == 1


def test_requested_elements_are_removed():
    at = _app()
    assert len(at.radio) == 0                     # "Sample from the dataset's test split" option removed
    assert len(at.info) == 0                      # blue upload box and large blue disclaimer box removed
    main_text = "\n".join(m.value for m in at.main.markdown)
    assert "Upload a brain MRI image to begin" not in main_text
    assert "must not be used as a substitute" not in main_text  # long disclaimer not in the main area
    assert "Research prototype" not in main_text                     # top label removed
    sidebar_captions = [c.value for c in at.sidebar.caption]
    assert "AI-assisted MRI image classification prototype" not in sidebar_captions  # sidebar subtitle removed
    # short Important note at the bottom of the sidebar (separator + 3 lines), not the long paragraph
    assert len(at.sidebar.divider) == 1
    assert sidebar_captions[-1] == "**Important:**  \nAI-assisted research prototype.  \nNot a medical diagnosis."
    assert not any("must not be used as a substitute" in c for c in sidebar_captions)


def test_upload_preview_and_binary_result():
    at = _app()
    _upload(at, IMAGE_A)
    assert not at.exception and not at.error
    assert len(at.get("image")) == 1              # uploaded MRI preview
    assert _displayed_result(at) is None          # nothing before clicking

    at.button(key="analyze").click().run()
    assert not at.exception and not at.error
    prediction, expected = _expected(IMAGE_A)
    label, confidence = _displayed_result(at)
    assert label == expected.label and label in ("STROKE", "NO STROKE")
    assert confidence == f"{expected.confidence:.1%}"     # binary confidence, not the 3-class max
    assert not any(REGION_TITLE in h.value for h in at.main.subheader)  # NO STROKE: no region claimed
    assert len(at.main.dataframe) == 0 and len(at.main.table) == 0     # class-probability table removed
    analyze_text = "\n".join(m.value for m in at.main.markdown)
    assert "Class probabilities" not in analyze_text
    assert not any(f"{p:.2%}" in analyze_text for p in prediction.probabilities.values())
    assert f"● {label} DETECTED" in analyze_text                       # centred summary under the cards


def test_second_image_updates_result_and_reuses_cached_model(monkeypatch):
    st.cache_resource.clear()
    calls = []
    real_create = services.create_predictor

    def counting_create(model_path=None):
        calls.append(model_path)
        return real_create(model_path)

    monkeypatch.setattr(services, "create_predictor", counting_create)
    at = _app()
    _upload(at, IMAGE_A)
    at.button(key="analyze").click().run()
    first = _displayed_result(at)
    _upload(at, IMAGE_B)
    assert _displayed_result(at) is None          # stale result cleared for the new image
    at.button(key="analyze").click().run()
    second = _displayed_result(at)
    expected_b = _expected(IMAGE_B)[1]
    assert second == (expected_b.label, f"{expected_b.confidence:.1%}")
    assert first != second
    assert len(calls) == 1                        # model loaded once for both images and all reruns


def test_stroke_result_shows_original_and_region_view():
    at = _app()
    _upload(at, IMAGE_B)
    at.button(key="analyze").click().run()
    assert not at.exception and not at.error
    expected = _expected(IMAGE_B)[1]
    assert expected.is_positive and _displayed_result(at) == (expected.label, f"{expected.confidence:.1%}")
    assert any(REGION_TITLE in h.value for h in at.main.subheader)
    assert len(at.get("image")) == 3                                  # upload preview, original, highlighted
    captions = "\n".join(c.value for c in at.main.caption)
    assert "does not represent a confirmed medical lesion" in captions
    assert "occlusion sensitivity" in captions


def test_corrupted_upload_is_reported_not_crashing():
    at = _app()
    at.file_uploader(key="mri_upload").set_value(("broken.png", b"\x89PNG\r\n\x1a\nnot-an-image", "image/png")).run()
    assert not at.exception
    assert any("cannot be analysed" in e.value for e in at.error)
    assert not any(b.key == "analyze" for b in at.button)


def test_non_image_with_allowed_extension_is_reported():
    at = _app()
    at.file_uploader(key="mri_upload").set_value(("notes.png", b"just some text", "image/png")).run()
    assert not at.exception
    assert any("cannot be analysed" in e.value for e in at.error)


def test_uploader_only_accepts_configured_formats():
    at = _app()
    configured = {e.lower().lstrip(".") for e in CONFIG["data"]["allowed_extensions"]}
    widget = {t.lower().lstrip(".") for t in at.file_uploader(key="mri_upload").allowed_type}
    assert widget == configured


def test_missing_model_is_reported(monkeypatch, tmp_path):
    monkeypatch.setenv(services.MODEL_PATH_ENV, str(tmp_path / "missing.keras"))
    at = _app()
    assert not at.exception
    assert any("trained model is not available" in e.value for e in at.error)
    assert not any(b.key == "analyze" for b in at.button)


def test_model_card_uses_real_metadata():
    predictor = Predictor(CONFIG)
    card = services.build_model_card(CONFIG)  # built from model_metadata.json, without loading the model
    meta = predictor.loaded.metadata
    assert card.class_names == tuple(meta["class_names"])
    assert card.test_accuracy == meta["test_evaluation"]["accuracy"]


def test_app_imports_do_not_load_tensorflow():
    # The page must render before TensorFlow initialises: importing the UI modules must not import it.
    import subprocess
    import sys

    code = ("import sys; sys.path.insert(0, r'%s'); import app.ui, app.services, src.inference.validation, "
            "src.inference.binary_view; print('tensorflow' in sys.modules)" % PROJECT_ROOT)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=PROJECT_ROOT)
    assert out.stdout.strip() == "False", out.stderr[-500:]


def test_app_code_has_no_deploy_ui():
    source = "\n".join(p.read_text(encoding="utf-8").lower() for p in (PROJECT_ROOT / "app").glob("*.py"))
    assert "deploy" not in source


def test_streamlit_toolbar_hides_developer_controls():
    # Streamlit's own "Deploy" control belongs to the developer toolbar; viewer mode hides it.
    import tomllib

    config = tomllib.loads((PROJECT_ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert config["client"]["toolbarMode"] == "viewer"
