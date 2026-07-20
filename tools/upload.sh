#!/bin/sh
set -eu

MODS="
bean-core
bean-config
"

[ -d ".venv" ] && . ./.venv/bin/activate

./run_tests.sh

uv pip install --upgrade build
uv pip install --upgrade twine

for dir in $MODS; do (
    cd "$dir"
    [ -d "./dist" ] && rm -rf "./dist"
    python -m build
    python -m twine upload --verbose ./dist/*
) done
