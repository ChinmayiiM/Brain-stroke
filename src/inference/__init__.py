"""Inference: load the trained model and predict MRI images.

Public API::

    from src.inference import Predictor
    result = Predictor().predict("dataset/test/Ischemic/ISC-P05_DWI_s020.png")

Depends on src.preprocessing (shared with training), never on src.training at runtime.
Exports are resolved lazily so that light submodules (validation, binary_view, errors)
can be imported without pulling in TensorFlow.
"""

__all__ = ["Predictor", "PredictionResult", "BatchItem"]


def __getattr__(name: str):
    if name in __all__:
        from src.inference import predictor

        return getattr(predictor, name)
    raise AttributeError(f"module 'src.inference' has no attribute {name!r}")
