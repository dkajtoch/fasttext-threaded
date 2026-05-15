from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import psutil

import fasttext_parallel
from fasttext_parallel._typing import OfficialFastTextModel, OfficialFastTextModule

_PROCESS_MODEL: OfficialFastTextModel | None = None


@dataclass(frozen=True)
class ScalingResult:
    method: str
    workers: int
    model_copies: int
    docs: int
    repeats: int
    batch_size: int
    load_wall_seconds: float
    wall_seconds: float
    cpu_seconds: float
    docs_per_second: float
    cpu_seconds_per_doc: float
    p50_seconds: float
    p95_seconds: float
    p99_seconds: float
    rss_before_mb: float
    rss_after_mb: float
    rss_delta_mb: float


@dataclass(frozen=True)
class Scenario:
    method: str
    workers: int
    model_copies: int


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


def _process_tree() -> list[psutil.Process]:
    process = psutil.Process(os.getpid())
    children: list[psutil.Process] = []
    for child in process.children(recursive=True):
        try:
            if child.is_running():
                children.append(child)
        except psutil.Error:
            continue
    return [process, *children]


def _process_tree_for(root: psutil.Process) -> list[psutil.Process]:
    processes = [root]
    try:
        processes.extend(root.children(recursive=True))
    except psutil.Error:
        return []
    return processes


def _rss_mb() -> float:
    rss = 0
    for process in _process_tree():
        try:
            rss += process.memory_info().rss
        except psutil.Error:
            continue
    return float(rss / (1024 * 1024))


def _rss_mb_for(processes: list[psutil.Process]) -> float:
    rss = 0
    for process in processes:
        try:
            rss += process.memory_info().rss
        except psutil.Error:
            continue
    return float(rss / (1024 * 1024))


def _cpu_seconds() -> float:
    total = 0.0
    for process in _process_tree():
        try:
            times = process.cpu_times()
            total += float(times.user + times.system)
        except psutil.Error:
            continue
    return total


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return ordered[index]


def _measure(
    *,
    method: str,
    workers: int,
    model_copies: int,
    docs_per_run: int,
    repeats: int,
    batch_size: int,
    load_wall_seconds: float,
    runner: Callable[[], int],
) -> ScalingResult:
    latencies: list[float] = []
    total_docs = 0
    rss_before = _rss_mb()
    cpu_before = _cpu_seconds()

    for _ in range(repeats):
        start = time.perf_counter()
        total_docs += runner()
        latencies.append(time.perf_counter() - start)

    cpu_after = _cpu_seconds()
    rss_after = _rss_mb()
    wall_seconds = sum(latencies)
    cpu_seconds = cpu_after - cpu_before
    docs_per_second = total_docs / wall_seconds if wall_seconds else 0.0
    cpu_seconds_per_doc = cpu_seconds / total_docs if total_docs else 0.0

    return ScalingResult(
        method=method,
        workers=workers,
        model_copies=model_copies,
        docs=docs_per_run,
        repeats=repeats,
        batch_size=batch_size,
        load_wall_seconds=load_wall_seconds,
        wall_seconds=wall_seconds,
        cpu_seconds=cpu_seconds,
        docs_per_second=docs_per_second,
        cpu_seconds_per_doc=cpu_seconds_per_doc,
        p50_seconds=statistics.median(latencies),
        p95_seconds=_percentile(latencies, 0.95),
        p99_seconds=_percentile(latencies, 0.99),
        rss_before_mb=rss_before,
        rss_after_mb=rss_after,
        rss_delta_mb=rss_after - rss_before,
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


def _official_thread_runner(
    model: OfficialFastTextModel, batches: list[list[str]], workers: int
) -> Callable[[], int]:
    def run() -> int:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return sum(
                executor.map(lambda batch: len(model.predict(batch)[0]), batches)
            )

    return run


def _parallel_runner(
    model: fasttext_parallel.FastTextParallel, batches: list[list[str]]
) -> Callable[[], int]:
    def run() -> int:
        return sum(len(model.predict(batch)[0]) for batch in batches)

    return run


def _official_batch_runner(
    model: OfficialFastTextModel, batches: list[list[str]]
) -> Callable[[], int]:
    def run() -> int:
        return sum(len(model.predict(batch)[0]) for batch in batches)

    return run


def _process_runner(
    executor: ProcessPoolExecutor, batches: list[list[str]]
) -> Callable[[], int]:
    def run() -> int:
        return sum(executor.map(_process_predict_batch, batches))

    return run


def _write_csv(path: Path, results: list[ScalingResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(results[0])))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def _write_metadata(path: Path, args: argparse.Namespace) -> None:
    metadata = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "physical_cpus": psutil.cpu_count(logical=False),
        "logical_cpus": psutil.cpu_count(logical=True),
        "total_memory_mb": psutil.virtual_memory().total / (1024 * 1024),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def _result_from_json(path: Path) -> ScalingResult:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return ScalingResult(**raw)


def _write_result_json(path: Path, result: ScalingResult) -> None:
    path.write_text(json.dumps(asdict(result), sort_keys=True), encoding="utf-8")


def _plot_results(path: Path, results: list[ScalingResult]) -> None:
    import matplotlib.pyplot as plt

    methods = sorted({result.method for result in results})
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    for method in methods:
        rows = sorted(
            [result for result in results if result.method == method],
            key=lambda result: result.workers,
        )
        workers = [row.workers for row in rows]
        axes[0][0].plot(
            workers, [row.docs_per_second for row in rows], marker="o", label=method
        )
        axes[0][1].plot(
            workers, [row.rss_after_mb for row in rows], marker="o", label=method
        )
        axes[1][0].plot(
            workers,
            [row.cpu_seconds_per_doc * 1_000_000 for row in rows],
            marker="o",
            label=method,
        )
        axes[1][1].plot(
            workers, [row.p95_seconds for row in rows], marker="o", label=method
        )

    axes[0][0].set_title("Throughput")
    axes[0][0].set_ylabel("docs/sec")
    axes[0][1].set_title("Resident Memory")
    axes[0][1].set_ylabel("RSS MB, parent + children")
    axes[1][0].set_title("CPU Time")
    axes[1][0].set_ylabel("microseconds/doc")
    axes[1][1].set_title("p95 Run Latency")
    axes[1][1].set_ylabel("seconds")

    for axis in axes.ravel():
        axis.set_xlabel("workers")
        axis.grid(True, alpha=0.3)
        axis.legend()

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _parse_workers(value: str) -> list[int]:
    workers = sorted({int(part) for part in value.split(",") if part.strip()})
    if not workers or any(worker <= 0 for worker in workers):
        raise argparse.ArgumentTypeError("workers must be positive integers")
    return workers


def _run_scenario(args: argparse.Namespace) -> ScalingResult:
    texts = _read_texts(args.input, args.limit)
    batches = _batches(texts, args.batch_size)
    if not batches:
        raise ValueError("input corpus is empty")

    method = args.scenario_method
    workers = args.scenario_workers

    if method == "official_batch":
        import fasttext

        fasttext_module = cast(OfficialFastTextModule, fasttext)

        start = time.perf_counter()
        model = fasttext_module.load_model(str(args.model))
        load_wall = time.perf_counter() - start
        return _measure(
            method=method,
            workers=workers,
            model_copies=1,
            docs_per_run=len(texts),
            repeats=args.repeat,
            batch_size=args.batch_size,
            load_wall_seconds=load_wall,
            runner=_official_batch_runner(model, batches),
        )

    if method == "official_python_threads":
        import fasttext

        fasttext_module = cast(OfficialFastTextModule, fasttext)

        start = time.perf_counter()
        model = fasttext_module.load_model(str(args.model))
        load_wall = time.perf_counter() - start
        return _measure(
            method=method,
            workers=workers,
            model_copies=1,
            docs_per_run=len(texts),
            repeats=args.repeat,
            batch_size=args.batch_size,
            load_wall_seconds=load_wall,
            runner=_official_thread_runner(model, batches, workers),
        )

    if method == "fasttext_parallel_threads":
        start = time.perf_counter()
        parallel_model = fasttext_parallel.load_model(args.model, threads=workers)
        load_wall = time.perf_counter() - start
        return _measure(
            method=method,
            workers=workers,
            model_copies=1,
            docs_per_run=len(texts),
            repeats=args.repeat,
            batch_size=args.batch_size,
            load_wall_seconds=load_wall,
            runner=_parallel_runner(parallel_model, batches),
        )

    if method == "official_python_processes":
        start = time.perf_counter()
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_process_model,
            initargs=(str(args.model),),
        ) as executor:
            warmup_batches = batches[: min(workers, len(batches))]
            list(executor.map(_process_predict_batch, warmup_batches))
            load_wall = time.perf_counter() - start
            return _measure(
                method=method,
                workers=workers,
                model_copies=workers,
                docs_per_run=len(texts),
                repeats=args.repeat,
                batch_size=args.batch_size,
                load_wall_seconds=load_wall,
                runner=_process_runner(executor, batches),
            )

    raise ValueError(f"unknown scenario method: {method}")


def _monitor_process(command: list[str], output_path: Path) -> ScalingResult:
    process = subprocess.Popen(command)
    root = psutil.Process(process.pid)
    peak_rss_mb = 0.0

    while process.poll() is None:
        processes = _process_tree_for(root)
        peak_rss_mb = max(peak_rss_mb, _rss_mb_for(processes))
        time.sleep(0.02)

    processes = _process_tree_for(root)
    peak_rss_mb = max(peak_rss_mb, _rss_mb_for(processes))

    if process.returncode != 0:
        raise RuntimeError(
            f"scenario process failed with exit code {process.returncode}"
        )

    result = _result_from_json(output_path)
    return ScalingResult(
        **{
            **asdict(result),
            "rss_before_mb": 0.0,
            "rss_after_mb": peak_rss_mb,
            "rss_delta_mb": peak_rss_mb,
        }
    )


def _run_isolated_scenario(
    args: argparse.Namespace, scenario: Scenario, temp_dir: Path
) -> ScalingResult:
    output_path = temp_dir / f"{scenario.method}-{scenario.workers}.json"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model",
        str(args.model),
        "--input",
        str(args.input),
        "--batch-size",
        str(args.batch_size),
        "--repeat",
        str(args.repeat),
        "--scenario-method",
        scenario.method,
        "--scenario-workers",
        str(scenario.workers),
        "--scenario-output",
        str(output_path),
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    return _monitor_process(command, output_path)


def _build_scenarios(
    worker_counts: list[int], process_worker_counts: list[int]
) -> list[Scenario]:
    scenarios = [Scenario("official_batch", 1, 1)]
    for workers in worker_counts:
        scenarios.append(Scenario("official_python_threads", workers, 1))
        scenarios.append(Scenario("fasttext_parallel_threads", workers, 1))
    for workers in process_worker_counts:
        scenarios.append(Scenario("official_python_processes", workers, workers))
    return scenarios


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--workers", type=_parse_workers, default=[1, 2, 4, 8])
    parser.add_argument("--process-workers", type=_parse_workers, default=None)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".cache") / "fasttext-parallel" / "scaling",
    )
    parser.add_argument("--scenario-method", default=None)
    parser.add_argument("--scenario-workers", type=int, default=None)
    parser.add_argument("--scenario-output", type=Path, default=None)
    args = parser.parse_args()

    if args.repeat <= 0:
        raise ValueError("--repeat must be greater than zero")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero")

    if args.scenario_method is not None:
        if args.scenario_workers is None or args.scenario_output is None:
            raise ValueError("scenario mode requires workers and output path")
        result = _run_scenario(args)
        _write_result_json(args.scenario_output, result)
        return

    texts = _read_texts(args.input, args.limit)
    batches = _batches(texts, args.batch_size)
    if not batches:
        raise ValueError("input corpus is empty")

    print(
        f"docs={len(texts)} batches={len(batches)} batch_size={args.batch_size} "
        f"repeats={args.repeat}"
    )

    process_workers = args.process_workers or args.workers
    scenarios = _build_scenarios(args.workers, process_workers)
    results: list[ScalingResult] = []

    with tempfile.TemporaryDirectory(prefix="fasttext-parallel-scaling-") as temp:
        temp_dir = Path(temp)
        for scenario in scenarios:
            print(f"running {scenario.method} workers={scenario.workers}")
            results.append(_run_isolated_scenario(args, scenario, temp_dir))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "scaling_results.csv"
    metadata_path = args.output_dir / "scaling_metadata.json"
    plot_path = args.output_dir / "scaling_plot.png"

    _write_csv(csv_path, results)
    _write_metadata(metadata_path, args)
    _plot_results(plot_path, results)

    for result in results:
        print(
            f"{result.method:28s} workers={result.workers:2d} "
            f"copies={result.model_copies:2d} docs/s={result.docs_per_second:10.2f} "
            f"cpu_us/doc={result.cpu_seconds_per_doc * 1_000_000:8.2f} "
            f"rss={result.rss_after_mb:8.2f}MB p95={result.p95_seconds:.4f}s"
        )

    print(f"csv={csv_path}")
    print(f"metadata={metadata_path}")
    print(f"plot={plot_path}")


if __name__ == "__main__":
    main()
