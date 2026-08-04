#!/bin/sh
set -eu

#* env

[ -d ".venv" ] && . ./.venv/bin/activate

#* loop, tag & push

for dir in bean-*; do
    module="${dir#bean-}"

    version="$(
        cd "$dir"
        python - <<PY
from bean.${module} import __version__
print(__version__)
PY
    )"

    tag="${dir}-${version}"

    if git rev-parse "$tag" >/dev/null 2>&1; then
        echo "'$tag' already exists."
        continue
    fi

    echo "Pushing tag: $tag"
    git tag "$tag"
    git push origin "$tag"
done
