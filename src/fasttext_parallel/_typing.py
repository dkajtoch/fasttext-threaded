"""Internal type protocols for untyped optional dependencies."""

from __future__ import annotations

from os import PathLike
from typing import Protocol, TypeAlias, overload

import numpy as np
import numpy.typing as npt

ProbabilityArray: TypeAlias = npt.NDArray[np.float32]
FastTextSinglePrediction: TypeAlias = tuple[tuple[str, ...], ProbabilityArray]
FastTextBatchPrediction: TypeAlias = tuple[list[list[str]], list[ProbabilityArray]]


class OfficialFastTextModel(Protocol):
    @overload
    def predict(
        self,
        text: str,
        k: int = 1,
        threshold: float = 0.0,
        on_unicode_error: str = "strict",
    ) -> FastTextSinglePrediction: ...

    @overload
    def predict(
        self,
        text: list[str],
        k: int = 1,
        threshold: float = 0.0,
        on_unicode_error: str = "strict",
    ) -> FastTextBatchPrediction: ...

    def save_model(self, path: str) -> None: ...


class OfficialFastTextModule(Protocol):
    def load_model(self, path: str | PathLike[str]) -> OfficialFastTextModel: ...

    def train_supervised(
        self,
        *,
        input: str,
        epoch: int = 5,
        lr: float = 0.1,
        dim: int = 100,
        wordNgrams: int = 1,
        minCount: int = 1,
        verbose: int = 2,
    ) -> OfficialFastTextModel: ...
