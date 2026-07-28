# bean.config

`bean.config` is a minimal configuration framework for Python.

Define your configuration as a class and populate it from command-line
arguments, environment variables, defaults, overrides, or custom sources.

> Just enough to arrange some beans.

---

## Overview

With `bean.config` you get:

- Type-safe configuration classes.
- Automatic loading from multiple *built-in* or *user-defined* sources.
- Configurable source priority.
- Custom validators and normalizers.

```py
from enum import Enum
from bean import config

class Mode(Enum):
    DEV = "dev"
    PROD = "prod"

@config
class AppConfig:
    DEBUG: bool = False
    MODE: Mode = Mode.DEV
    HOST: str
    PORT: int = 8080

# `AppConfig` is now the loaded configuration instance.
print(config.dump_str(AppConfig))
```

> **Note**:
>
> To quickly inspect the public API:
>
> ```sh
> python -c "
>     import bean.config as bean
>     for v in sorted(dir(bean)):
>         print(f'- {v}')
> "
> ```

## Installation

Requirements:

- Python `3.14+`

Using `pip`:

```sh
pip install --upgrade bean-config
```

Using `curl` (direct download):

```sh
curl -Ls \
    https://raw.githubusercontent.com/numen-0/bean/refs/heads/main/bean-config/src/bean/config.py
```

## API

This is a quick reference for the main `API`.

For full details, see the [source code](/bean-config/src/bean/config.py).

### Loading

```py
cfg = config.load(
    Config,
    argv=["--host", "localhost"],
)

config.load(
    Config,
    env_prefix="APP",
    overrides={
        "HOST": "localhost",
    },
)

config.load(
    Config,
    priorities=("extra", "defaults"), # only load from extra and then defaults
    extra_sources={
        "extra": foo,                 # foo(field) -> value
    },
)
```

Built-in sources:

| source            | description                                              |
|:-----------------:|:---------------------------------------------------------|
| `args`            | Command-line arguments (`argparse`).                     |
| `envs`            | Environment variables.                                   |
| `defaults`        | Default class attributes.                                |
| `overrides`       | Explicit values passed via the `overrides` parameter.    |

### Validators

Validators can `raise` exceptions or return a boolean signaling success:

```
class Config:
    NAME: str
    PORT: int

    @config.validator("PORT")
    def port_is_valid(self, port: int) -> bool:
        return 0 < port <= 65535

    @config.validator("NAME")
    def name_is_valid(self, name: str):
      if name == "":
          raise ValueError("Empty NAME")
```

> **Note**: Validators may return `True`, `False` or `None`.

> **Note**: A validator fails only if it returns `False` or raises an exception.

### Normalizers

Normalizers can be used to finalize the configuration load.

```
class Config:
    NAME: str
    HOST: str

    @config.normalizer("NAME", "HOST")
    def lowercase(self, name: str, host: str) -> tuple[str, str]:
        return name.lower(), host.lower()

    @config.normalizer("NAME", "HOST", series=True)
    def strip(self, value: str) -> str:
        return value.strip()

    @config.normalizer()
    def normalize(self) -> None:
      ...
```

> **Note**:
>
> The return value must match the declared fields
>
> - No fields -> return `None`
> - One field -> return the normalized value
> - Multiple fields -> return a `tuple` with one value per field
> - `series=True` -> the function receives and returns one field at a time

## Notes

Validators and normalizers follow the same definition rules:

- The number of declared fields must match the function parameters.
- They work with instance methods, `@staticmethod`, and `@classmethod`.
- Execution order is determined by `priority`, then by function name.

## License

All the repo falls under the [MIT License](/LICENSE).

