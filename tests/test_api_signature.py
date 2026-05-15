from __future__ import annotations

import inspect
from pathlib import Path

import fasttext_threaded


def test_predict_signature_matches_fasttext_shape(tiny_model_path: Path) -> None:
    model = fasttext_threaded.load_model(tiny_model_path, threads=2)

    signature = inspect.signature(model.predict)
    assert list(signature.parameters) == [
        "text",
        "k",
        "threshold",
        "on_unicode_error",
    ]
    assert signature.parameters["k"].default == 1
    assert signature.parameters["threshold"].default == 0.0
    assert signature.parameters["on_unicode_error"].default == "strict"


def test_predict_batch_alias_uses_same_parameters(tiny_model_path: Path) -> None:
    model = fasttext_threaded.load_model(tiny_model_path, threads=2)
    signature = inspect.signature(model.predict_batch)
    assert list(signature.parameters) == [
        "texts",
        "k",
        "threshold",
        "on_unicode_error",
    ]
