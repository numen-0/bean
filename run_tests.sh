#!/bin/sh
set -eu

[ -d ".venv" ] && . ./.venv/bin/activate

python -m unittest discover -s tests
