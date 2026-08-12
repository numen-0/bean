# bean.py

```

                                ,---.      ,---.
                               /     `-<>-'  :D \
                               |                |
                                . .            .
                                .`-~~~~~~~~~~-'

                             Bean there, done that.

```

> Small, fun, dependency-free beans.

---

## What is `bean`?

`bean` is a tiny collection of reusable packages for small Python applications.

Designed to be:

- Small and self-contained (one file, one package)
- Dependency-free
- Easy to:
  - read & reason about
  - reuse
  - extend
  - distribute

Built for small services, humble tools, side quests, and mildly ambitious
revolutions.

> Be an *unbeanlievable* bean with bean.

## Project Structure

`bean` is a **monorepo** containing multiple packages.

Each package:

- Is self-contained.
- Can be installed independently.
- Shares the same namespace: `bean.*`.

> Meaning:
> ```py
> import bean.core
> import bean.test
> ```

## Packages

`bean` currently provides:

| Package                         | Description                                |
|:-------------------------------:|:-------------------------------------------|
| [`bean.core`](/bean-core)       | core runtime & application primitives      |
| [`bean.config`](/bean-config)   | minimal configuration framework            |
| [`bean.test`](/bean-test)       | lightweight testing utilities (WIP)        |

> Beware: More may sprout.

## Installation

Requirements:

- Python `3.14`

Using `pip`:

```sh
MOD="core"
pip install --upgrade "bean-$MOD"
```

Or grab a package directly with `curl`:

```sh
MOD="core"
curl -Ls \
    "https://raw.githubusercontent.com/numen-0/bean/main/bean-$MOD/src/bean/$MOD.py"
```

## Local Development

### Setup

- Using python:

  1. Setup environment:

  ```sh
  python3.14 -m venv .venv
  . ./.venv/bin/activate
  ```

  2. Install packages:

  ```sh
  for bean in ./bean-*; do python -m pip install -e ./bean-*; done
  ```

- Using `uv`:

```sh
uv python install 3.14
uv sync --all-packages
```

> Quick `bean` check (for `bean.core`):
>
> ```sh
> python -c "
>     import bean.core as bean
>     print('[bean.core]')
>     print(f'version: {bean.__version__}')
>     print(f'doc: {bean.__doc__}')
>     print(f'by: {bean.__author__}')
> "
> ```

### Running Tests

From repository root:

```sh
./run_tests.sh
```

## License

All the repo falls under the [MIT License](/LICENSE).

