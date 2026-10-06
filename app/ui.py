"""Presentation helpers for the Streamlit app. All values rendered come from real objects."""

from __future__ import annotations

from html import escape
from typing import TYPE_CHECKING

import streamlit as st

from app.services import ModelCard
from src.inference.binary_view import BinaryResult
from src.inference.validation import ImageInfo

if TYPE_CHECKING:  # type hint only — region_view pulls in OpenCV/SciPy, loaded on first STROKE result
    from src.inference.region_view import RegionResult

# Neutral colours come from the Streamlit theme (.streamlit/config.toml, incl. the navy sidebar); only
# the STROKE / NO STROKE status uses its own subtle red / green tones. Text rules are scoped to the
# markdown container so they outrank Streamlit's own `p` styles.
CSS = """
<style>
.block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem; }
html { scroll-behavior: smooth; }
.app-title { font-size: 2.1rem; font-weight: 700; line-height: 1.15; margin: 0; }
[data-testid="stMarkdownContainer"] .app-subtitle { font-size: 1.15rem; font-weight: 550; opacity: 0.7;
    margin: 0.3rem 0 0.6rem 0; }

/* sidebar: brand, navigation, important note */
[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
[data-testid="stMarkdownContainer"] .brand-title { font-size: 1.2rem; font-weight: 700; margin: 0; }
[data-testid="stMarkdownContainer"] .brand-subtitle { font-size: 0.85rem; opacity: 0.8; margin: 0.15rem 0 0 0; }
.st-key-nav button { justify-content: flex-start; border: none; border-radius: 8px; padding: 0.5rem 0.8rem;
    min-height: 2.6rem; }
.st-key-nav button > div { justify-content: flex-start; width: 100%; }
.st-key-nav button p { font-weight: 500; }
.st-key-nav button[kind="primary"] { background: #1f5fae; color: #ffffff; }
.st-key-nav button[kind="tertiary"]:hover { background: rgba(255, 255, 255, 0.08); color: #ffffff; }
.st-key-nav [data-testid="stElementContainer"] { margin-bottom: -0.55rem; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color: #c3d2e6; opacity: 1; }

/* cards */
.st-key-mri_panel, .st-key-result_panel, .st-key-region_card, .st-key-mm_model, .st-key-mm_details,
.st-key-mm_metrics, .st-key-mm_pipeline, .st-key-about_card, .st-key-how_card {
    background: #ffffff; border-radius: 14px; padding: 1.1rem 1.25rem; }

/* information pages: blue headings, metric tiles, pipeline steps */
[data-testid="stMarkdownContainer"] .page-title { font-size: 1.05rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: #1f5fae; margin: 0.4rem 0 0.2rem 0; }
[class*="st-key-mm_"] h3, .st-key-about_card h3, .st-key-how_card h3 { color: #1f5fae; }
[data-testid="stMarkdownContainer"] .metric-tile { background: #f4f7fb; border: 1px solid #dfe7f1; border-radius: 12px;
    padding: 1rem 1.1rem; margin: 0; }
.metric-tile .metric-label { display: block; font-size: 0.85rem; opacity: 0.75; }
.metric-tile .metric-value { display: block; font-size: 1.8rem; font-weight: 700; color: #1f5fae; margin-top: 0.2rem; }
[data-testid="stMarkdownContainer"] .pipeline { display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem;
    margin: 0.3rem 0 0 0; }
.pipeline .step { background: #eef3fa; border: 1px solid #d3deec; border-radius: 999px; padding: 0.35rem 0.85rem;
    font-size: 0.92rem; font-weight: 550; }
.pipeline .step--model { background: #1f5fae; border-color: #1f5fae; color: #ffffff; }
.pipeline .arrow { opacity: 0.55; }
.st-key-region_original, .st-key-region_highlight { border-radius: 12px; padding: 0.8rem 0.9rem; }
.st-key-mri_panel img, .st-key-region_card img { max-height: 420px; object-fit: contain; border-radius: 8px;
    background: #000000; }
[data-testid="stMarkdownContainer"] .panel-label { font-size: 0.95rem; font-weight: 650; margin: 0; }
[data-testid="stMarkdownContainer"] .region-empty { min-height: 220px; display: flex; align-items: center;
    justify-content: center; text-align: center; padding: 1rem; border-radius: 8px; background: #f4f7fb;
    font-size: 0.92rem; opacity: 0.85; margin: 0; }

/* result */
.result-card { display: flex; align-items: center; gap: 1.1rem; border: 1px solid; border-radius: 12px;
    padding: 1.4rem 1.5rem; margin: 0.4rem 0 0.9rem 0; }
.result-card--no-stroke { background: #eef7f1; border-color: #bfdcc9; }
.result-card--stroke { background: #fbf0ed; border-color: #ecc5bc; }
.result-icon { flex: 0 0 auto; width: 2.6rem; height: 2.6rem; border-radius: 50%; color: #ffffff;
    display: flex; align-items: center; justify-content: center; font-size: 1.45rem; font-weight: 700; }
.result-card--no-stroke .result-icon { background: #2e8b57; }
.result-card--stroke .result-icon { background: #c0392b; }
[data-testid="stMarkdownContainer"] .result-class { font-size: 2.1rem; font-weight: 700; letter-spacing: 0.02em;
    line-height: 1.1; margin: 0; }
.result-card--no-stroke .result-class, .result-card--no-stroke .result-confidence { color: #1f6b42; }
.result-card--stroke .result-class, .result-card--stroke .result-confidence { color: #a83a2c; }
[data-testid="stMarkdownContainer"] .result-confidence { font-size: 1.15rem; font-weight: 600; margin: 0.5rem 0 0 0; }
[data-testid="stMarkdownContainer"] .result-explanation { font-size: 0.95rem; opacity: 0.8; margin: 0; }
[data-testid="stMarkdownContainer"] .result-summary { text-align: center; font-size: 1.1rem; font-weight: 700;
    letter-spacing: 0.05em; margin: 1.2rem 0 0.4rem 0; }
.result-summary--no-stroke { color: #1f6b42; }
.result-summary--stroke { color: #a83a2c; }
[data-testid="stMetricValue"] { font-size: 1.6rem; }
</style>
"""

# (page key, icon, label) — one sidebar button per page; only the selected page is rendered
PAGES = (
    ("home", ":material/home:", "Home"),
    ("upload", ":material/upload:", "Upload MRI"),
    ("model", ":material/settings:", "Model & Method"),
    ("about", ":material/info:", "About"),
    ("how", ":material/help:", "How It Works"),
)
PAGE_KEYS = tuple(key for key, _, _ in PAGES)

REGION_NOTE = ("The highlighted region shows the area that contributed most to the model's prediction. "
               "It is a model-interpretability visualization and does not represent a confirmed medical lesion.")


def _weights_name(card: ModelCard) -> str:
    name = card.pretrained_weights or ""
    return "ImageNet" if name.lower() == "imagenet" else name


def _transfer_text(card: ModelCard) -> str:
    return f"yes, {_weights_name(card)} pre-trained weights" if card.pretrained_weights else "no"


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def render_header() -> None:
    st.markdown(
        '<h1 class="app-title">Brain Stroke Detection</h1>'
        '<p class="app-subtitle">Brain MRI Image Classification</p>',
        unsafe_allow_html=True,
    )


def current_page() -> str:
    """Selected page: session state, else the ?page= URL parameter (bookmarks/reloads), else Home."""
    if "page" not in st.session_state:
        requested = st.query_params.get("page")
        st.session_state["page"] = requested if requested in PAGE_KEYS else "home"
    return st.session_state["page"]


def _go_to(page: str) -> None:
    st.session_state["page"] = page
    st.query_params["page"] = page


def render_sidebar(page: str) -> None:
    with st.sidebar:
        st.markdown('<p class="brand-title">Brain Stroke Detection</p>'
                    '<p class="brand-subtitle">Brain MRI Image Classification</p>', unsafe_allow_html=True)
        st.write("")
        with st.container(key="nav"):
            for key, icon, label in PAGES:
                st.button(label, icon=icon, key=f"nav_{key}", width="stretch",
                          type="primary" if key == page else "tertiary", on_click=_go_to, args=(key,))
        st.divider()
        st.caption("**Important:**  \nAI-assisted research prototype.  \nNot a medical diagnosis.")


# ---- analysis: upload card -------------------------------------------------------------------
def render_upload_heading() -> None:
    st.subheader(":material/upload: Upload MRI", anchor=False)


def render_upload_details(data: bytes, name: str, info: ImageInfo) -> None:
    """The real uploaded MRI, full image, original aspect ratio."""
    st.caption(f"{name}  ·  {info.width} × {info.height} px  ·  {info.format or 'unknown format'}")
    st.image(data, width="stretch")


# ---- analysis: result card -------------------------------------------------------------------
def render_result_heading() -> None:
    st.subheader(":material/bar_chart: Result", anchor=False)


def _tone(result: BinaryResult) -> str:
    return "stroke" if result.is_positive else "no-stroke"


def render_result(result: BinaryResult) -> None:
    """Binary label + its confidence only (no per-class probabilities)."""
    explanation = ("The model classified this MRI as Stroke based on the combined stroke-class probability."
                   if result.is_positive else
                   "The model classified this MRI as No Stroke based on the predicted class probabilities.")
    st.markdown(
        f'<div class="result-card result-card--{_tone(result)}">'
        f'<span class="result-icon">{"!" if result.is_positive else "✓"}</span><div>'
        f'<p class="result-class">{escape(result.label)}</p>'
        f'<p class="result-confidence">Confidence: {result.confidence:.1%}</p>'
        f'</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<p class="result-explanation">{explanation}</p>', unsafe_allow_html=True)


def render_summary(result: BinaryResult) -> None:
    st.markdown(f'<p class="result-summary result-summary--{_tone(result)}">● {escape(result.label)} DETECTED</p>',
                unsafe_allow_html=True)


# ---- analysis: suspected region (STROKE only) ------------------------------------------------
def render_region_section(data: bytes, region: RegionResult | None, min_drop: float) -> None:
    """Original MRI next to the occlusion-sensitivity area (model interpretability, not a lesion mask)."""
    with st.container(border=True, key="region_card"):
        st.subheader(":material/image: Suspected Stroke Region", anchor=False)
        left, right = st.columns(2, gap="medium")
        with left:
            with st.container(border=True, height="stretch", key="region_original"):
                st.markdown('<p class="panel-label">Original MRI</p>', unsafe_allow_html=True)
                st.image(data, width="stretch")
        with right:
            with st.container(border=True, height="stretch", key="region_highlight"):
                st.markdown('<p class="panel-label">Highlighted Region</p>', unsafe_allow_html=True)
                if region is None:
                    message = "The region view could not be generated for this image."
                elif region.overlay_png is None:
                    message = (f"No single area lowered the model's stroke probability by {min_drop:.0%} or more "
                               "when hidden, so no region is highlighted.")
                else:
                    message = None
                    st.image(region.overlay_png, width="stretch")
                if message:
                    st.markdown(f'<p class="region-empty">{message}</p>', unsafe_allow_html=True)
        st.caption(f":material/info: {REGION_NOTE}")
        if region is not None and region.overlay_png is not None:
            st.caption(f"Method: occlusion sensitivity. Hiding the most influential patch lowers the model's "
                       f"stroke probability from {region.stroke_probability:.1%} to "
                       f"{region.occluded_probability:.1%}. The dataset has no lesion masks, so this area is "
                       f"not checked against a real lesion location.")


# ---- information pages (each rendered on its own, never under the analysis page) -------------
def _page_title(text: str) -> None:
    st.markdown(f'<p class="page-title">{escape(text)}</p>', unsafe_allow_html=True)


def render_model_page(card: ModelCard | None) -> None:
    _page_title("Model & Method")
    if card is None:
        st.warning("The trained model could not be loaded, so model details are unavailable.")
        return
    stroke_classes = [c for c in card.class_names if c != "Normal"]

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        with st.container(border=True, height="stretch", key="mm_model"):
            st.subheader("Model", anchor=False)
            st.write(f"**{card.backbone}** with transfer learning ({_transfer_text(card)}), "
                     f"fine-tuned in two stages: first only a new classification head was trained on a frozen "
                     f"backbone, then the top backbone layers were fine-tuned with a small learning rate.")
            st.subheader("Purpose", anchor=False)
            st.write("The model learns visual patterns from labelled brain MRI slices and predicts which class "
                     "of the training dataset an image most resembles:")
            st.markdown("\n".join(f"- {name}" for name in card.class_names))
            st.write("The application combines:")
            st.markdown(f"- {' + '.join(stroke_classes)} → **STROKE**\n- Normal → **NO STROKE**")
    with right:
        with st.container(border=True, height="stretch", key="mm_details"):
            st.subheader("Details", anchor=False)
            st.markdown(
                f"- **Output:** {card.output}\n"
                f"- **Input size:** {card.input_size}\n"
                f"- **Classes ({len(card.class_names)}):** {', '.join(card.class_names)}\n"
                f"- **Dataset:** [{card.dataset_name}]({card.dataset_url})\n"
                f"- **Model file:** `{card.model_file}`"
                + (f"\n- **Selected training stage:** {card.selected_stage} (fine-tuned)" if card.selected_stage else "")
            )

    if card.test_accuracy is not None:
        with st.container(border=True, key="mm_metrics"):
            st.subheader("Measured on the held-out test split", anchor=False)
            tiles = (("Accuracy", f"{card.test_accuracy:.1%}"),
                     ("Balanced accuracy", f"{card.test_balanced_accuracy:.1%}"),
                     ("Macro F1", f"{card.test_macro_f1:.3f}"))
            for column, (label, value) in zip(st.columns(3), tiles):
                column.markdown(f'<p class="metric-tile"><span class="metric-label">{label}</span>'
                                f'<span class="metric-value">{value}</span></p>', unsafe_allow_html=True)
            patients = f" from {card.test_patients} patients" if card.test_patients else ""
            st.caption(f"3-class metrics computed once on {card.test_images} test images{patients} of this "
                       "research dataset. Part of this score reflects MRI-sequence differences between classes "
                       "rather than pathology (see README). These figures do not describe clinical performance.")

    with st.container(border=True, key="mm_pipeline"):
        st.subheader("Pipeline", anchor=False)
        height, width = card.input_size.split(" × ")[:2]
        steps = ["MRI image", "validation", f"preprocessing (RGB, {height}×{width})", card.backbone,
                 "feature extraction", "classification", "STROKE / NO STROKE + confidence"]
        chips = '<span class="arrow">→</span>'.join(
            f'<span class="step{" step--model" if step == card.backbone else ""}">{escape(step)}</span>'
            for step in steps)
        st.markdown(f'<div class="pipeline">{chips}</div>', unsafe_allow_html=True)


def render_about_page() -> None:
    _page_title("About")
    with st.container(border=True, key="about_card"):
        st.subheader("About this project", anchor=False)
        st.write("A deep-learning prototype that classifies a single 2D brain MRI slice into the "
                 "classes of a public research dataset. Built for academic study of transfer "
                 "learning on medical images.")


def render_how_page() -> None:
    _page_title("How It Works")
    with st.container(border=True, key="how_card"):
        st.subheader("From upload to result", anchor=False)
        st.markdown(
            "1. **Upload** an MRI slice (PNG/JPG).\n"
            "2. **Preprocessing** — the same function used during training: RGB conversion, "
            "resize to the model input size.\n"
            "3. **EfficientNetB0** extracts image features.\n"
            "4. **Classification** — a softmax layer gives one probability per class "
            "(Haemorrhagic, Ischemic, Normal).\n"
            "5. **Result** — Haemorrhagic + Ischemic are combined as **STROKE**, Normal is "
            "**NO STROKE**; the more likely result is shown with its confidence.\n"
            "6. **Suspected region (STROKE only)** — square patches of the image are hidden one at a time and "
            "the same model is run again; the area whose hiding lowers the stroke probability most is outlined. "
            "This shows what the model relied on. It is not lesion detection: the dataset has no lesion masks."
        )
