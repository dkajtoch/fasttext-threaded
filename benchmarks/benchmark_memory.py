from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import psutil

import fasttext_parallel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()

    process = psutil.Process(os.getpid())
    before = process.memory_info().rss / (1024 * 1024)
    start = time.perf_counter()
    model = fasttext_parallel.load_model(args.model, threads=args.threads)
    after = process.memory_info().rss / (1024 * 1024)
    print(f"load_seconds={time.perf_counter() - start:.4f}")
    print(f"threads={model.threads}")
    print(f"rss_before_mb={before:.2f}")
    print(f"rss_after_mb={after:.2f}")
    print(f"rss_delta_mb={after - before:.2f}")


if __name__ == "__main__":
    main()
