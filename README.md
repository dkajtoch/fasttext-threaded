# fasttext-parallel

`fasttext-parallel` is a Python package for running fastText supervised model
inference with one model loaded in memory and a native C++ thread pool doing
parallel batch prediction.

The package is designed to preserve the official Python fastText `predict`
interface while fixing the high-throughput inference case where Python threads
do not provide useful model-level parallelism and Python processes multiply
model memory.

## Install

Source builds are the supported v1 install path on Linux and macOS:

```bash
pip install fasttext-parallel
```

For local development:

```bash
uv sync --all-extras
uv run pre-commit install
```

You need a C++17 compiler and CMake.

## Usage

```python
from fasttext_parallel import load_model

model = load_model("model.bin", threads=16)

labels, probs = model.predict("some text", k=1, threshold=0.0)
batch_labels, batch_probs = model.predict(
    ["first document", "second document"],
    k=1,
    threshold=0.0,
)
```

The public prediction signature intentionally matches official fastText:

```python
predict(text, k=1, threshold=0.0, on_unicode_error="strict")
```

`predict(list[str], ...)` is the high-throughput path. It sends one batch into
C++, releases the GIL, and partitions rows across native worker threads while
using one loaded model.

`predict_batch(texts, k=1, threshold=0.0, on_unicode_error="strict")` is also
available as a convenience alias for `predict(texts, ...)`. It does not have
different semantics.

## Compatibility

The goal is API and behavior parity with official Python fastText for
prediction:

- `predict(str)` returns `(tuple[str, ...], numpy.ndarray)`.
- `predict(list[str])` returns `(list[list[str]], list[numpy.ndarray])`.
- newline-containing inputs are rejected with the same message shape as
  official fastText.
- `k`, `threshold`, and labels/probabilities are tested against official
  fastText.

## Why This Exists

Official fastText supports `predict(list[str])`, but the binding processes that
batch with a serial C++ loop. Python-side `ThreadPoolExecutor` is not a reliable
solution for throughput, and Python-side `ProcessPoolExecutor` loads one model
copy per process.

This package targets the common production inference shape:

- one model object in one Python process
- one native C++ thread pool
- large batches passed through `predict(list[str], ...)`
- no TCP, pickle, JSON, or multiprocessing serialization on the hot path

## Error Handling

The package exposes custom exceptions:

```python
from fasttext_parallel import (
    FastTextParallelError,
    InvalidInputError,
    ModelLoadError,
    PredictionError,
)
```

Input validation errors happen before worker threads are launched where
possible. If a C++ worker fails during batch prediction, the first exception is
captured, already-running workers finish, and the entire batch raises a Python
exception. Partial predictions are not returned silently.

## Benchmarks

Prepare a realistic language-identification benchmark using the official
fastText `lid.176.bin` supervised model:

```bash
uv run python benchmarks/prepare_lid_benchmark.py --model-kind bin --repeat 5000
```

This downloads the 126 MB `lid.176.bin` model into `.cache/` and writes a
multilingual input corpus. The compressed `lid.176.ftz` model is also supported
for quick smoke checks, but `lid.176.bin` is the meaningful default for
performance measurements.

Run:

```bash
uv run python benchmarks/benchmark_inference.py \
  --model .cache/fasttext-parallel/lid/lid.176.bin \
  --input .cache/fasttext-parallel/lid/lid_input.txt \
  --threads 16 \
  --batch-size 4096
```

The benchmark compares:

- official fastText single-thread `predict(str)` loop
- official fastText `predict(list[str])`
- official fastText with `ThreadPoolExecutor`
- official fastText with `ProcessPoolExecutor`
- `fasttext_parallel.predict(list[str])`

Reported metrics include throughput, latency percentiles, model load time,
RSS memory, model-copy count, worker count, and a small parity sample.

For scaling plots across native threads, Python threads, and Python processes:

```bash
uv run python benchmarks/benchmark_scaling.py \
  --model .cache/fasttext-parallel/lid/lid.176.bin \
  --input .cache/fasttext-parallel/lid/lid_input.txt \
  --workers 1,2,4,8,16 \
  --process-workers 1,2,4 \
  --batch-size 4096 \
  --repeat 3
```

This writes:

- `.cache/fasttext-parallel/scaling/scaling_results.csv`
- `.cache/fasttext-parallel/scaling/scaling_metadata.json`
- `.cache/fasttext-parallel/scaling/scaling_plot.png`

The scaling benchmark records wall time, parent-plus-child CPU time, RSS,
model-copy count, machine metadata, and a four-panel plot. Each measured
scenario runs in a fresh subprocess, and RSS is captured as peak parent-plus-child
memory for that scenario.

## Stress Test

Run:

```bash
uv run python stress/stress_inference.py \
  --model /path/to/model.bin \
  --input /path/to/input.txt \
  --threads 16 \
  --callers 32 \
  --duration 60
```

The stress test repeatedly calls one shared model from many Python threads and
checks for crashes, exceptions, output count mismatches, order corruption, and
RSS growth.

## Development

```bash
uv sync --all-extras
uv run pre-commit run --all-files
uv run pytest
```

Native C++ tests can be run with:

```bash
uv run cmake -S . -B build -DFASTTEXT_PARALLEL_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
uv run cmake --build build
uv run ctest --test-dir build --output-on-failure
```

Runtime-safety checks are available through sanitizer builds:

```bash
uv run cmake -S . -B build-asan \
  -DFASTTEXT_PARALLEL_BUILD_TESTS=ON \
  -DFASTTEXT_PARALLEL_ENABLE_ASAN=ON \
  -DCMAKE_BUILD_TYPE=Debug
uv run cmake --build build-asan
uv run ctest --test-dir build-asan --output-on-failure
```

Use `FASTTEXT_PARALLEL_ENABLE_TSAN=ON` instead of ASAN for ThreadSanitizer.
CI also runs a short shared-model stress smoke test.

Ruff and mypy are mandatory quality gates.

## Release

Releases are published by GitHub Actions using PyPI Trusted Publishing. No
long-lived PyPI token is required.

1. Configure the PyPI project trusted publisher for this repository,
   `.github/workflows/release.yml`, and the `pypi` GitHub environment.
2. Require manual approval on the `pypi` environment.
3. Push a SemVer tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```

The release workflow builds the source distribution plus Linux and macOS wheels
with `cibuildwheel`, validates all artifacts, and publishes them with PyPI
Trusted Publishing.

## Limitations

- v1 targets Linux/macOS source builds.
- Prebuilt wheels are not part of the initial release.
- The primary speedup comes from batching. Per-row calls still pay Python call
  overhead and should not be used for throughput measurements.
