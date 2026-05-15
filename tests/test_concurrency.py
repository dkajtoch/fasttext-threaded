from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fasttext_parallel


def test_concurrent_python_callers_share_one_model(tiny_model_path: Path) -> None:
    model = fasttext_parallel.load_model(tiny_model_path, threads=4)
    texts = ["alpha beta", "gamma delta", "one two"] * 32

    def run_once() -> int:
        labels, probs = model.predict(texts, k=1)
        assert len(labels) == len(texts)
        assert len(probs) == len(texts)
        return len(labels)

    with ThreadPoolExecutor(max_workers=8) as executor:
        counts = list(executor.map(lambda _: run_once(), range(16)))

    assert counts == [len(texts)] * 16
