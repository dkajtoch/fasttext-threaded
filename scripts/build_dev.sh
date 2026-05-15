#!/usr/bin/env bash
set -euo pipefail

uv sync --all-extras
uv run cmake -S . -B build -DFASTTEXT_PARALLEL_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
uv run cmake --build build
