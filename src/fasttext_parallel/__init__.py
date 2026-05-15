"""fastText-compatible parallel inference."""

from fasttext_parallel.api import FastTextParallel, load_model
from fasttext_parallel.exceptions import (
    FastTextParallelError,
    InvalidInputError,
    ModelLoadError,
    PredictionError,
)

__all__ = [
    "FastTextParallel",
    "FastTextParallelError",
    "InvalidInputError",
    "ModelLoadError",
    "PredictionError",
    "load_model",
]
