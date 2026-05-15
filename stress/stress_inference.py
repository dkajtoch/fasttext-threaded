from __future__ import annotations

import argparse
import os
import threading
import time
from pathlib import Path

import psutil

import fasttext_threaded


def _read_texts(path: Path, batch_size: int) -> list[str]:
    texts: list[str] = []
    with path.open(encoding="utf-8", errors="replace") as file:
        for line in file:
            text = line.rstrip("\n")
            if text:
                texts.append(text)
            if len(texts) >= batch_size:
                break
    if not texts:
        raise ValueError("input file did not contain any non-empty lines")
    return texts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--callers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--duration", type=float, default=60.0)
    args = parser.parse_args()

    texts = _read_texts(args.input, args.batch_size)
    model = fasttext_threaded.load_model(args.model, threads=args.threads)
    stop_at = time.monotonic() + args.duration
    failures: list[BaseException] = []
    lock = threading.Lock()
    calls = 0

    def worker() -> None:
        nonlocal calls
        while time.monotonic() < stop_at:
            try:
                labels, probs = model.predict(texts)
                if len(labels) != len(texts) or len(probs) != len(texts):
                    raise AssertionError("result length mismatch")
                with lock:
                    calls += 1
            except BaseException as exc:  # noqa: BLE001
                with lock:
                    failures.append(exc)
                return

    threads = [threading.Thread(target=worker) for _ in range(args.callers)]
    rss_start = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    rss_end = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

    print(f"calls={calls}")
    print(f"failures={len(failures)}")
    print(f"rss_start_mb={rss_start:.2f}")
    print(f"rss_end_mb={rss_end:.2f}")
    print(f"rss_delta_mb={rss_end - rss_start:.2f}")

    if failures:
        raise failures[0]


if __name__ == "__main__":
    main()
