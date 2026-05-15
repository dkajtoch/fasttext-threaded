from __future__ import annotations

from pathlib import Path

import pytest

import fasttext_threaded
from fasttext_threaded import InvalidInputError, ModelLoadError


def test_missing_model_raises_model_load_error() -> None:
    with pytest.raises(ModelLoadError):
        fasttext_threaded.load_model("/definitely/missing/model.bin", threads=2)


def test_invalid_k_raises_invalid_input(tiny_model_path: Path) -> None:
    model = fasttext_threaded.load_model(tiny_model_path, threads=2)

    with pytest.raises(InvalidInputError):
        model.predict(["alpha beta"], k=0)


def test_invalid_threads_raises_invalid_input(tiny_model_path: Path) -> None:
    with pytest.raises(InvalidInputError):
        fasttext_threaded.load_model(tiny_model_path, threads=0)
