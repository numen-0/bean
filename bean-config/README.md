# bean.config

`bean.config` is a minimal configuration framework for Python.

Define a configuration schema as a class, then populate it from one or more
config sources with type validation and customizable source priorities.

> Just enough to arrange some beans.

---

## Overview

With `bean.config` you get:

- **Typed configuration** from a simple class definition.
- **Multiple sources** (`args`, `envs`, `overrides`, custom sources).
- **Customizable priorities** between configuration sources.
- **Debugging helpers** to inspect loaded configurations.

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

### Config Example

```py
from enum import Enum
from bean import config

class Color(Enum):
    RED   = "#ff0000"
    GREEN = "#00ff00"
    BLUE  = "#0000ff"

@config
class MySimpleConfig:
    NAME: str
    EMAIL: str
    DEBUG: bool = False
    PORT: int = 8080
    HOST: str = "localhost"
    COLORS: list[Color] = [Color.RED]

print(config.dump_str(MySimpleConfig))
```

#### Sources

Built-in sources

| source            | description                                              |
|:-----------------:|:---------------------------------------------------------|
| `args`            | command-line arguments parsed with `argparse`.           |
| `envs`            | environment variables.                                   |
| `defaults`        | default values defined on the configuration class.       |
| `overrides`       | values supplied via the `overrides` parameter.           |

## License

All the repo falls under the [MIT License](/LICENSE).

