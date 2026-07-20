#!/bin/sh
set -eu

[ -d ".venv" ] && . ./.venv/bin/activate

SCRIPT="./tools/api_guard.py"
SNAPSHOT_DIR="./out/api_snapshots"
mkdir -p "$SNAPSHOT_DIR"

code=0
for mod_dir in bean-*; do
    mod="${mod_dir#bean-}"
    snapshot="$SNAPSHOT_DIR/${mod}.json"
    source="./${mod_dir}/src/bean/${mod}.py"

    echo "Checking module: $mod"

    [ ! -f "$snapshot" ] && {
        echo "    no snapshot for '$mod_dir', skipping..."
        echo "    run: python "$SCRIPT" snapshot $source $snapshot"
        continue
    }

    python "$SCRIPT" check "$source" "$snapshot" || code=1
done

exit "$code"
