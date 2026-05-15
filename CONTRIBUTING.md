# Contributing

Thanks for working on `fasttext-threaded`. This project has a small public API and a native C++ core, so changes should preserve fastText compatibility and be backed by tests.

## Development Setup

Use `uv` for local development:

```bash
uv sync --all-extras
uv run pre-commit install
```

You also need CMake and a C++17 compiler.

## Quality Gates

Run the default checks before opening a pull request:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

Run C++ tests with:

```bash
uv run cmake -S . -B build -DFASTTEXT_THREADED_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
uv run cmake --build build --parallel
uv run ctest --test-dir build --output-on-failure
```

Or run the combined local script:

```bash
scripts/run_all_checks.sh
```

## Compatibility Rules

The Python-facing prediction API must stay compatible with official fastText:

```python
predict(text, k=1, threshold=0.0, on_unicode_error="strict")
```

Important expectations:

- `predict(str)` returns labels and probabilities with the official fastText shape.
- `predict(list[str])` is the high-throughput parallel path.
- `predict_batch(...)` is only an alias for `predict(list[str], ...)`.
- Input ordering must be preserved.
- Do not reject inputs that official fastText accepts unless there is a documented safety reason.
- Worker failures should fail the whole batch; do not silently return partial predictions.

## Typing

Avoid `Any`. If an untyped dependency boundary is needed, add a narrow `Protocol` in `src/fasttext_threaded/_typing.py` and use an explicit `cast` at the boundary.

The repository runs mypy in strict mode. Keep public wrappers typed and include stubs for native extension symbols when needed.

## C++ Guidelines

- Keep native code C++17-compatible.
- Do not construct Python objects while the GIL is released or inside worker threads.
- Preserve result order by writing to preallocated per-input slots.
- Capture worker exceptions and rethrow after all submitted work finishes.
- Keep fastText model state read-only after loading.
- Add C++ tests for thread-pool behavior, model error handling, and concurrency-sensitive changes.

## Benchmarks

Use the official fastText language identification model for meaningful benchmark runs:

```bash
uv run python benchmarks/prepare_lid_benchmark.py --model-kind bin --repeat 5000
```

Run the scaling benchmark with:

```bash
uv run python benchmarks/benchmark_scaling.py \
  --model .cache/fasttext-threaded/lid/lid.176.bin \
  --input .cache/fasttext-threaded/lid/lid_input.txt \
  --workers 1,2,4,8,16 \
  --process-workers 1,2,4,8,16 \
  --batch-size 2048 \
  --repeat 5
```

Benchmark outputs under `.cache/` are local artifacts and should not be committed.

## Runtime Safety

Run sanitizer builds for native changes when practical:

```bash
uv run cmake -S . -B build-asan \
  -DFASTTEXT_THREADED_BUILD_TESTS=ON \
  -DFASTTEXT_THREADED_ENABLE_ASAN=ON \
  -DCMAKE_BUILD_TYPE=Debug
uv run cmake --build build-asan --parallel
uv run ctest --test-dir build-asan --output-on-failure
```

Use `FASTTEXT_THREADED_ENABLE_TSAN=ON` for ThreadSanitizer.

Run the full stress tool for shared-model concurrency checks:

```bash
uv run python stress/stress_inference.py \
  --model .cache/fasttext-threaded/lid/lid.176.bin \
  --input .cache/fasttext-threaded/lid/lid_input.txt \
  --threads 16 \
  --callers 32 \
  --batch-size 8192 \
  --duration 60
```

## Packaging And Release

Do not add build-time GitHub fetches. The PyPI sdist must be buildable from packaged sources and PyPI build dependencies.

Releases are tag-driven:

```bash
git tag v0.1.0
git push origin v0.1.0
```

GitHub Actions builds the sdist and prebuilt Linux/macOS wheels with `cibuildwheel`, validates artifacts, and publishes with PyPI Trusted Publishing.

## Pull Request Expectations

A good pull request includes:

- a short explanation of the behavioral change
- tests for user-visible behavior or native concurrency changes
- benchmark output for performance-sensitive changes
- no generated caches, wheels, benchmark outputs, or local IDE files
