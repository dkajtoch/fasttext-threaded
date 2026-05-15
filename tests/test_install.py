from __future__ import annotations


def test_import_smoke() -> None:
    import fasttext_threaded

    assert "load_model" in fasttext_threaded.__all__
