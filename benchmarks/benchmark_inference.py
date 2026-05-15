from __future__ import annotations

import argparse
import os
import statistics
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import cast

import psutil

import fasttext_parallel
from fasttext_parallel._typing import OfficialFastTextModel, OfficialFastTextModule

_PROCESS_MODEL: OfficialFastTextModel | None = None


def _read_texts(path: Path, limit: int | None) -> list[str]:
    texts: list[str] = []
    with path.open(encoding="utf-8", errors="replace") as file:
        for line in file:
            text = line.rstrip("\n")
            if text:
                texts.append(text)
            if limit is not None and len(texts) >= limit:
                break
    return texts


def _batches(texts: list[str], batch_size: int) -> list[list[str]]:
    return [
        texts[index : index + batch_size] for index in range(0, len(texts), batch_size)
    ]


def _rss_mb() -> float:
    process = psutil.Process(os.getpid())
    rss = process.memory_info().rss
    for child in process.children(recursive=True):
        try:
            rss += child.memory_info().rss
        except psutil.Error:
            continue
    return float(rss / (1024 * 1024))


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return ordered[index]


def _measure(
    name: str, runner: Callable[[], int], repeats: int, model_copies: int
) -> None:
    latencies: list[float] = []
    total_docs = 0
    rss_before = _rss_mb()
    for _ in range(repeats):
        start = time.perf_counter()
        total_docs += runner()
        latencies.append(time.perf_counter() - start)
    rss_after = _rss_mb()
    total_time = sum(latencies)
    docs_per_second = total_docs / total_time if total_time else 0.0
    p50 = statistics.median(latencies)
    p95 = _percentile(latencies, 0.95)
    p99 = _percentile(latencies, 0.99)
    print(
        f"{name:36s} docs/s={docs_per_second:10.2f} "
        f"p50={p50:.4f}s p95={p95:.4f}s p99={p99:.4f}s "
        f"model_copies={model_copies:3d} "
        f"rss_delta={rss_after - rss_before:.2f}MB rss={rss_after:.2f}MB"
    )


def _init_process_model(model_path: str) -> None:
    global _PROCESS_MODEL
    import fasttext

    fasttext_module = cast(OfficialFastTextModule, fasttext)
    _PROCESS_MODEL = fasttext_module.load_model(model_path)


def _process_predict_batch(texts: list[str]) -> int:
    if _PROCESS_MODEL is None:
        raise RuntimeError("process model was not initialized")
    _PROCESS_MODEL.predict(texts)
    return len(texts)


def _thread_pool_runner(
    model: OfficialFastTextModel, batches: list[list[str]], max_workers: int
) -> Callable[[], int]:
    def run() -> int:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return sum(
                executor.map(lambda batch: len(model.predict(batch)[0]), batches)
            )

    return run


def _process_pool_runner(
    executor: ProcessPoolExecutor, batches: list[list[str]]
) -> Callable[[], int]:
    def run() -> int:
        return sum(executor.map(_process_predict_batch, batches))

    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--processes", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    import fasttext

    fasttext_module = cast(OfficialFastTextModule, fasttext)

    texts = _read_texts(args.input, args.limit)
    batches = _batches(texts, args.batch_size)
    print(
        f"docs={len(texts)} batches={len(batches)} batch_size={args.batch_size} "
        f"threads={args.threads} processes={args.processes}"
    )

    start = time.perf_counter()
    baseline = fasttext_module.load_model(str(args.model))
    print(f"official_fasttext_load={time.perf_counter() - start:.4f}s")

    start = time.perf_counter()
    parallel = fasttext_parallel.load_model(args.model, threads=args.threads)
    print(f"fasttext_parallel_load={time.perf_counter() - start:.4f}s")

    expected_labels, expected_probs = baseline.predict(texts[: min(8, len(texts))])
    actual_labels, actual_probs = parallel.predict(texts[: min(8, len(texts))])
    parity_ok = expected_labels == actual_labels and len(expected_probs) == len(
        actual_probs
    )
    print(f"parity_sample={parity_ok}")

    _measure(
        "official loop predict(str)",
        lambda: sum(len(baseline.predict(text)[0]) * 0 + 1 for text in texts),
        args.repeat,
        model_copies=1,
    )
    _measure(
        "official predict(list[str])",
        lambda: sum(len(baseline.predict(batch)[0]) for batch in batches),
        args.repeat,
        model_copies=1,
    )
    _measure(
        "official ThreadPoolExecutor",
        _thread_pool_runner(baseline, batches, args.threads),
        args.repeat,
        model_copies=1,
    )
    _measure(
        "fasttext_parallel predict(list[str])",
        lambda: sum(len(parallel.predict(batch)[0]) for batch in batches),
        args.repeat,
        model_copies=1,
    )

    with ProcessPoolExecutor(
        max_workers=args.processes,
        initializer=_init_process_model,
        initargs=(str(args.model),),
    ) as executor:
        _measure(
            "official ProcessPoolExecutor",
            _process_pool_runner(executor, batches),
            args.repeat,
            model_copies=args.processes,
        )


if __name__ == "__main__":
    main()
