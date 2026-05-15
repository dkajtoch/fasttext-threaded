#!/usr/bin/env bash
set -euo pipefail

uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
uv run cmake -S . -B build -DFASTTEXT_THREADED_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
uv run cmake --build build
uv run ctest --test-dir build --output-on-failure
