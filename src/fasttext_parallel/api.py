"""Public Python API for fasttext-parallel."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final, TypeAlias, cast, overload

import numpy as np

from fasttext_parallel import _fasttext_parallel as _native
from fasttext_parallel._typing import ProbabilityArray
from fasttext_parallel.exceptions import InvalidInputError

LabelBatch: TypeAlias = list[list[str]]
ProbabilityBatch: TypeAlias = list[ProbabilityArray]

_NEWLINE_ERROR: Final = "predict processes one line at a time (remove '\\n')"


def _check_line(entry: str) -> str:
    if entry.find("\n") != -1:
        raise ValueError(_NEWLINE_ERROR)
    return f"{entry}\n"


class FastTextParallel:
    """fastText-compatible model wrapper with native parallel batch inference."""

    __slots__ = ("f",)

    def __init__(self, model_path: str | os.PathLike[str], threads: int | None = None):
        if threads is None:
            threads = os.cpu_count() or 1
        if threads <= 0:
            raise InvalidInputError("thread_count must be greater than zero")
        self.f = _native.FastTextParallelModel(str(Path(model_path)), threads)

    @overload
    def predict(
        self,
        text: str,
        k: int = 1,
        threshold: float = 0.0,
        on_unicode_error: str = "strict",
    ) -> tuple[tuple[str, ...], ProbabilityArray]: ...

    @overload
    def predict(
        self,
        text: list[str],
        k: int = 1,
        threshold: float = 0.0,
        on_unicode_error: str = "strict",
    ) -> tuple[LabelBatch, ProbabilityBatch]: ...

    def predict(
        self,
        text: str | list[str],
        k: int = 1,
        threshold: float = 0.0,
        on_unicode_error: str = "strict",
    ) -> tuple[tuple[str, ...], ProbabilityArray] | tuple[LabelBatch, ProbabilityBatch]:
        """Predict labels using the official fastText Python signature."""
        if type(text) is list:
            lines = [_check_line(entry) for entry in text]
            labels, probs = self.f.predict_many(lines, k, threshold, on_unicode_error)
            return labels, probs

        line = _check_line(cast(str, text))
        labels_batch, probs_batch = self.f.predict_many(
            [line], k, threshold, on_unicode_error
        )
        single_labels = tuple(labels_batch[0]) if labels_batch else ()
        single_probs = (
            np.array(probs_batch[0], dtype=np.float32, copy=False)
            if probs_batch
            else np.array([], dtype=np.float32)
        )
        return single_labels, single_probs

    def predict_batch(
        self,
        texts: list[str],
        k: int = 1,
        threshold: float = 0.0,
        on_unicode_error: str = "strict",
    ) -> tuple[LabelBatch, ProbabilityBatch]:
        """Alias for ``predict(list[str], ...)`` with identical semantics."""
        return self.predict(texts, k, threshold, on_unicode_error)

    @property
    def threads(self) -> int:
        """Number of native worker threads owned by this model."""
        return int(self.f.threads)


def load_model(
    path: str | os.PathLike[str], threads: int | None = None
) -> FastTextParallel:
    """Load a fastText model for parallel inference."""
    return FastTextParallel(path, threads=threads)
