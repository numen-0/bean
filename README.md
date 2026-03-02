# bean.py

```


                               ,---.      ,---.
                              /     `-<>-'  :D \
                              |                |
                               . .            .
                                `-~~~~~~~~~~-'

                             Bean there, done that
                           Bean there, debugged that
                               May contain nuts


```

> Small, fun, dependency-free beans.

---

## What is `bean`?

`bean` is a tiny library of reusable packages for small Python applications.

Designed to be:

- Single-file oriented
- Dependency-free
- Easy to:
    - distribute
    - read & reason about
    - reuse
    - extend

Built for small services, humble tools, side quests, and mildly ambitious
revolutions.

> Be an unbeanlievable bean with bean.

## Project Structure

This repository is a **monorepo** containing multiple packages.

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

- [`bean.core`](/bean-core): core runtime & application primitives
- [`bean.test`](/bean-test): lightweight testing utilities (WIP)

> Beware: More may sprout.

## Local Development

### Setup

1. setup environment:

```sh
uv python install 3.14
uv venv --python 3.14
. .venv/bin/activate
```

2. install packages:

```sh
uv pip install -e ./bean-*
```

3. quick `bean` check (for `bean.core`):

```sh
python -c "
    import bean.core as bean
    print('[bean.core]')
    print(f'version: {bean.__version__}')
    print(f'doc: {bean.__doc__}')
    print(f'by: {bean.__author__}')
"
```

### Running Tests

From repository root:

```sh
./run_tests.sh
```

## License

All the repo falls under the [MIT License](/LICENSE).

