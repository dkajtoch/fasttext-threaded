"""fastText-compatible parallel inference."""

from fasttext_threaded.api import FastTextThreaded, load_model
from fasttext_threaded.exceptions import (
    FastTextThreadedError,
    InvalidInputError,
    ModelLoadError,
    PredictionError,
)

__all__ = [
    "FastTextThreaded",
    "FastTextThreadedError",
    "InvalidInputError",
    "ModelLoadError",
    "PredictionError",
    "load_model",
]
