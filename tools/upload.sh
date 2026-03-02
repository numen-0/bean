#!/bin/sh
set -eu

[ -d ".venv" ] && . ./.venv/bin/activate

./run_tests.sh

uv pip install --upgrade build
uv pip install --upgrade twine

for dir in bean-core; do (
    cd "$dir"
    [ -d "./dist" ] && rm -rf "./dist"
    python -m build
    python -m twine upload ./dist/*
) done
