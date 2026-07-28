#!/bin/sh
set -eu

MODS="
bean-core
bean-config
"

#* env

[ -d ".venv" ] && . ./.venv/bin/activate

./run_tests.sh

#* get packages

PUBLISH="$(
for dir in $MODS; do
    (
        cd "$dir"

        version="$(python - <<PY
from bean.${dir#bean-} import __version__
print(__version__)
PY
)"

        if ! curl -fsSL "https://pypi.org/pypi/$dir/json" \
            | jq -e --arg v "$version" '.releases | has($v)' >/dev/null
        then
            echo "$dir|$version"
        fi
    )
done
)"

[ -n "$PUBLISH" ] || {
    echo "Everything is already published."
    exit 0
}

#* ok?

echo "The following package(s) will be published:"
echo "$PUBLISH" | while IFS='|' read -r dir version; do
    printf "  %-16s %s\n" "$dir" "$version"
done

printf "\nContinue? [y/N] "
read -r ans

case "$ans" in
    y|Y|yes|YES) ;;
    *) echo "Aborted."; exit 0 ;;
esac

#* build & publish

python -m ensurepip --upgrade
python -m pip install -U pip build twine

echo "$PUBLISH" | while IFS='|' read -r dir version; do
    (
        cd "$dir"

        [ -d "./dist" ] && rm -rf "./dist"
        python -m build
        python -m twine check dist/*
        python -m twine upload --skip-existing dist/*
    )
done

