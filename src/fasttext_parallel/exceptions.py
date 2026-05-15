"""Exception classes exposed by fasttext-parallel."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:

    class FastTextParallelError(RuntimeError):
        """Base error for fasttext-parallel."""

    class ModelLoadError(FastTextParallelError):
        """Raised when a model cannot be loaded."""

    class PredictionError(FastTextParallelError):
        """Raised when prediction fails inside the native extension."""

    class InvalidInputError(FastTextParallelError):
        """Raised when native input validation fails."""

else:
    try:
        from fasttext_parallel._fasttext_parallel import (  # noqa: F401
            FastTextParallelError,
            InvalidInputError,
            ModelLoadError,
            PredictionError,
        )
    except ImportError:

        class FastTextParallelError(RuntimeError):
            """Base error for fasttext-parallel."""

        class ModelLoadError(FastTextParallelError):
            """Raised when a model cannot be loaded."""

        class PredictionError(FastTextParallelError):
            """Raised when prediction fails inside the native extension."""

        class InvalidInputError(FastTextParallelError):
            """Raised when native input validation fails."""


__all__ = [
    "FastTextParallelError",
    "InvalidInputError",
    "ModelLoadError",
    "PredictionError",
]
