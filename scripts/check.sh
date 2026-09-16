#!/usr/bin/env bash
# Everything CI runs, in the order CI runs it. Run before every commit.
set -euo pipefail

PY="${PY:-.venv/bin/python}"

echo "── ruff ──────────────────────────────────────────"
"$PY" -m ruff check .

echo "── mypy ──────────────────────────────────────────"
"$PY" -m mypy

echo "── pytest ────────────────────────────────────────"
"$PY" -m pytest -q

echo
echo "All checks passed."
