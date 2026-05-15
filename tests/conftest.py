from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from fasttext_parallel._typing import OfficialFastTextModule


@pytest.fixture(scope="session")
def official_fasttext() -> OfficialFastTextModule:
    return cast(OfficialFastTextModule, pytest.importorskip("fasttext"))


@pytest.fixture(scope="session")
def tiny_model_path(
    tmp_path_factory: pytest.TempPathFactory,
    official_fasttext: OfficialFastTextModule,
) -> Path:
    tmp_path = tmp_path_factory.mktemp("fasttext_model")
    train_path = tmp_path / "train.txt"
    model_path = tmp_path / "tiny.bin"
    train_path.write_text(
        "\n".join(
            [
                "__label__alpha alpha beta alpha",
                "__label__alpha beta alpha beta",
                "__label__beta gamma delta gamma",
                "__label__beta delta gamma delta",
                "__label__gamma one two three",
                "__label__gamma two three one",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    model = official_fasttext.train_supervised(
        input=str(train_path),
        epoch=20,
        lr=0.5,
        dim=16,
        wordNgrams=1,
        minCount=1,
        verbose=0,
    )
    model.save_model(str(model_path))
    return model_path
