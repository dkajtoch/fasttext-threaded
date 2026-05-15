from __future__ import annotations

from pathlib import Path

import numpy as np

import fasttext_threaded
from fasttext_threaded._typing import OfficialFastTextModule


def test_single_prediction_matches_official_fasttext(
    tiny_model_path: Path,
    official_fasttext: OfficialFastTextModule,
) -> None:
    baseline = official_fasttext.load_model(str(tiny_model_path))
    parallel = fasttext_threaded.load_model(tiny_model_path, threads=2)

    expected_labels, expected_probs = baseline.predict("alpha beta", k=2)
    actual_labels, actual_probs = parallel.predict("alpha beta", k=2)

    assert actual_labels == expected_labels
    np.testing.assert_allclose(actual_probs, expected_probs, rtol=1e-6, atol=1e-6)


def test_batch_prediction_matches_official_fasttext(
    tiny_model_path: Path,
    official_fasttext: OfficialFastTextModule,
) -> None:
    texts = ["alpha beta", "gamma delta", "one two"]
    baseline = official_fasttext.load_model(str(tiny_model_path))
    parallel = fasttext_threaded.load_model(tiny_model_path, threads=2)

    expected_labels, expected_probs = baseline.predict(texts, k=2)
    actual_labels, actual_probs = parallel.predict(texts, k=2)

    assert actual_labels == expected_labels
    assert len(actual_probs) == len(expected_probs)
    for actual, expected in zip(actual_probs, expected_probs, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_newline_validation_matches_official_message(tiny_model_path: Path) -> None:
    parallel = fasttext_threaded.load_model(tiny_model_path, threads=2)

    try:
        parallel.predict("bad\ninput")
    except ValueError as exc:
        assert "predict processes one line at a time" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_predict_batch_alias_matches_predict(tiny_model_path: Path) -> None:
    parallel = fasttext_threaded.load_model(tiny_model_path, threads=2)
    texts = ["alpha beta", "gamma delta"]

    predict_labels, predict_probs = parallel.predict(texts, k=1)
    alias_labels, alias_probs = parallel.predict_batch(texts, k=1)

    assert alias_labels == predict_labels
    for alias, direct in zip(alias_probs, predict_probs, strict=True):
        np.testing.assert_array_equal(alias, direct)


def test_threshold_outside_probability_range_matches_official_fasttext(
    tiny_model_path: Path,
    official_fasttext: OfficialFastTextModule,
) -> None:
    texts = ["alpha beta", "gamma delta"]
    baseline = official_fasttext.load_model(str(tiny_model_path))
    parallel = fasttext_threaded.load_model(tiny_model_path, threads=2)

    for threshold in (-0.5, 1.5):
        expected_labels, expected_probs = baseline.predict(
            texts, k=2, threshold=threshold
        )
        actual_labels, actual_probs = parallel.predict(texts, k=2, threshold=threshold)

        assert actual_labels == expected_labels
        assert len(actual_probs) == len(expected_probs)
        for actual, expected in zip(actual_probs, expected_probs, strict=True):
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
