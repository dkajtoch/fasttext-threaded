from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from fasttext_threaded._typing import OfficialFastTextModule


@pytest.fixture(scope="session")
def official_fasttext() -> OfficialFastTextModule:
    return cast(OfficialFastTextModule, pytest.importorskip("fasttext"))


@pytest.fixture(scope="session")
def tiny_model_path() -> Path:
    return Path(__file__).parent / "data" / "tiny_supervised.bin"
