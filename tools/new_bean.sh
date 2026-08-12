#!/bin/sh
set -eu

BEAN="$(
    printf "%s" "${1:-}"             \
        | tr '[:upper:]' '[:lower:]' \
        | sed -e 's/[^a-z0-9]/-/g'   \
              -e 's/-\+/-/g'         \
              -e 's/\(^-\|-$\)//g'
)"
MOD="bean.$BEAN"
DIR="bean-$BEAN"
PACKAGE="$DIR/src/bean/$BEAN.py"

[ "$#" -eq 1 ] && [ -n "$BEAN" ] || {
    cat <<EOF 1>&2
Usage: $0 <bean>
Description: Bootstrap new bean
EOF
    exit 1
}

################################################################################

CINF="\e[1;7;92m"
CCMD="\e[1;7;95m"
CDIE="\e[1;7;31m"
R="\e[0m"

info()    { printf "${CINF} info ${R} %s\n" "${*:-???}"; }
die()     { printf "${CDIE} error ${R} %s\n" "${*:-???}" >&2; exit 1; }
cmd()     { printf "${CCMD} cmd ${R} %s\n" "${*:-???}"; "$@"; }

################################################################################

dump_py() {
    cat <<EOF
# ============================================================================ #
#                                                                              #
#                               ,---.      ,---.                               #
#                              /     \`-<>-'  :D \                              #
#                              |                |                              #
#                               . .            .                               #
#                               .\`-~~~~~~~~~~-'                                #
#                                                                              #
#                            Bean there, done that.                            #
#                                                                              #
# ============================================================================ #

__version__ = "0.1.0"
__doc__     = "---description---" # TODO
__author__  = "numen-0"
__license__ = "MIT"

# ------------------------------------------------------------------------------
# api
# ------------------------------------------------------------------------------

__all__ = [ # TODO

]

# Hack: Hide imported stuff for \`dir(module)\`
__dir__ = lambda: __all__

# ------------------------------------------------------------------------------
# imports
# ------------------------------------------------------------------------------

import typing       as _

# ------------------------------------------------------------------------------
# ---A---
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# ---B---
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# ---C---
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
EOF
}

dump_md() {
    cat <<EOF
# $MOD # TODO

\`$MOD\` ---desc---.

> Just enough to ---joke--- beans.

---

## Overview # TODO

With \`$MOD\` you get:

- ---A---
- ---B---
- ---C---

\`\`\`py
from bean import ${BEAN}
...
\`\`\`

## Installation

Using \`pip\`:

\`\`\`sh
pip install --upgrade $DIR
\`\`\`

Using \`curl\` (direct download):

\`\`\`sh
curl -Ls \
    https://raw.githubusercontent.com/numen-0/bean/main/$PACKAGE
\`\`\`

## API # TODO

This is a quick reference for the main \`API\`.

For full details, see the [source code](/$PACKAGE).

### ---A---

\`\`\`py

\`\`\`

### ---B---

\`\`\`py

\`\`\`

### ---C---

\`\`\`py

\`\`\`

## Notes # TODO

- ---a---
- ---b---
- ---c---

## License

All the repo falls under the [MIT License](/LICENSE).

EOF
}

dump_pp() {
    cat <<EOF
[project]
name = "$DIR"
description = "---description---" # TODO
dynamic = ["version"]
dependencies = []
requires-python = ">=3.14"

authors = [
  { name = "numen-0", email = "numen.0x1dea@gmail.com" },
]
readme = "README.md"
license = { file = "LICENSE" }

keywords = [
    "bean",
    "$MOD",
]
classifiers = [
    "Development Status :: 2 - Pre-Alpha",
    "Environment :: Console",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: 3.14",
    "Programming Language :: Python :: 3.15",
    "Topic :: Scientific/Engineering",
    "Topic :: Software Development",
    "Topic :: Software Development :: Libraries",
    "Topic :: Software Development :: Libraries :: Application Frameworks",
    "Typing :: Typed",
]

[project.urls]
homepage = "https://github.com/numen-0/bean"
repository = "https://github.com/numen-0/bean"
issues = "https://github.com/numen-0/bean/issues"
documentation = "https://github.com/numen-0/bean/$PACKAGE"

[build-system]
requires = ["setuptools>=64", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.dynamic]
version = { attr = "${MOD}.__version__" }

[tool.setuptools.packages.find]
where = ["src"]
include = ["bean*"]
namespaces = true
EOF
}


################################################################################

[ ! -e "$DIR" ] || die "'$DIR' already exists"

info "The following will be created:"
cat <<EOF
$DIR/
 |- src/
 |   '- bean/
 |       '- ${BEAN}.py
 |- LICENSE
 |- README.md
 '- pyproject.toml
EOF

printf "Continue? [y/N] "
read -r answer

case "$answer" in
    y|Y|yes|YES) ;;
    *) die "Aborted." ;;
esac

info "Bootstraping bean..."

info "Creating dirs..."
cmd mkdir -p "$DIR/src/bean"

info "Creating files..."

info "Hard linking LICENSE..."
cmd ln ./LICENSE "$DIR/LICENSE"

info "Generating README.md..."
dump_md > "$DIR/README.md"

info "Generating pyproject.toml..."
dump_pp > "$DIR/pyproject.toml"

info "Generating $PACKAGE..."
dump_py > "$PACKAGE"

info "Bean '$BEAN', done..."
info "Has bean a pleasure working with you."

