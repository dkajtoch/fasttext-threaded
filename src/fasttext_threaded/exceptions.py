"""Exception classes exposed by fasttext-threaded."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:

    class FastTextThreadedError(RuntimeError):
        """Base error for fasttext-threaded."""

    class ModelLoadError(FastTextThreadedError):
        """Raised when a model cannot be loaded."""

    class PredictionError(FastTextThreadedError):
        """Raised when prediction fails inside the native extension."""

    class InvalidInputError(FastTextThreadedError):
        """Raised when native input validation fails."""

else:
    try:
        from fasttext_threaded._fasttext_threaded import (  # noqa: F401
            FastTextThreadedError,
            InvalidInputError,
            ModelLoadError,
            PredictionError,
        )
    except ImportError:

        class FastTextThreadedError(RuntimeError):
            """Base error for fasttext-threaded."""

        class ModelLoadError(FastTextThreadedError):
            """Raised when a model cannot be loaded."""

        class PredictionError(FastTextThreadedError):
            """Raised when prediction fails inside the native extension."""

        class InvalidInputError(FastTextThreadedError):
            """Raised when native input validation fails."""


__all__ = [
    "FastTextThreadedError",
    "InvalidInputError",
    "ModelLoadError",
    "PredictionError",
]
