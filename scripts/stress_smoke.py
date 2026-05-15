from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import cast

from fasttext_parallel._typing import OfficialFastTextModule


def _write_training_data(path: Path) -> None:
    rows = [
        "__label__alpha alpha beta alpha",
        "__label__alpha beta alpha beta",
        "__label__beta gamma delta gamma",
        "__label__beta delta gamma delta",
        "__label__gamma one two three",
        "__label__gamma two three one",
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_input_data(path: Path, repeat: int) -> None:
    rows = [
        "alpha beta alpha",
        "gamma delta gamma",
        "one two three",
        "beta alpha beta",
        "delta gamma delta",
        "two three one",
    ]
    with path.open("w", encoding="utf-8") as output:
        for index in range(repeat):
            for row in rows:
                output.write(f"{row} {index}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--callers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--repeat", type=int, default=512)
    args = parser.parse_args()

    import fasttext

    fasttext_module = cast(OfficialFastTextModule, fasttext)
    with tempfile.TemporaryDirectory(prefix="fasttext-parallel-stress-") as temp:
        temp_dir = Path(temp)
        train_path = temp_dir / "train.txt"
        input_path = temp_dir / "input.txt"
        model_path = temp_dir / "tiny.bin"

        _write_training_data(train_path)
        _write_input_data(input_path, args.repeat)
        model = fasttext_module.train_supervised(
            input=str(train_path),
            epoch=20,
            lr=0.5,
            dim=16,
            wordNgrams=1,
            minCount=1,
            verbose=0,
        )
        model.save_model(str(model_path))

        subprocess.run(
            [
                sys.executable,
                "stress/stress_inference.py",
                "--model",
                str(model_path),
                "--input",
                str(input_path),
                "--threads",
                str(args.threads),
                "--callers",
                str(args.callers),
                "--batch-size",
                str(args.batch_size),
                "--duration",
                str(args.duration),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
