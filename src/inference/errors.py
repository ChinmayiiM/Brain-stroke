"""Inference error types — each carries a user-readable message.

The CLI catches :class:`InferenceError` and prints its message; anything else
is a genuine bug and is allowed to surface with a traceback.
"""


class InferenceError(Exception):
    """Base class for expected, explainable inference failures."""


class ModelNotFoundError(InferenceError):
    """The trained model file (or its metadata) does not exist."""


class ModelLoadError(InferenceError):
    """The model file exists but could not be deserialised."""


class ClassMappingError(InferenceError):
    """Model metadata, model output size and configured class names disagree."""


class PreprocessingMismatchError(InferenceError):
    """The configured preprocessing differs from what the model was trained with."""


class ImageValidationError(InferenceError):
    """The input image is missing, unsupported, unreadable or unusable."""


class ModelOutputError(InferenceError):
    """The model returned an output with an unexpected shape or invalid probabilities."""
