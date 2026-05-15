from __future__ import annotations


def test_import_smoke() -> None:
    import fasttext_parallel

    assert "load_model" in fasttext_parallel.__all__
