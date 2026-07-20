# ============================================================================ #
#                                                                              #
#                               ,---.      ,---.                               #
#                              /     `-<>-'  :D \                              #
#                              |                |                              #
#                               . .            .                               #
#                               .`-~~~~~~~~~~-'                                #
#                                                                              #
#                             Bean there, done that                            #
#                                                                              #
# ============================================================================ #

__version__ = "0.1.0"
__doc__     = "Minimal config framework"
__author__  = "numen-0"
__license__ = "MIT"

"""
Load a configuration from a class annotations

example:

```py
class Mode(Enum):
    DEV = "dev"
    PROD = "prod"

@config()
class Config:
    HOST    : str     # if no default value, it's assumed as required
    PORT    : int     = 8000
    DEBUG   : bool    = False
    MODE    : Mode    = Mode.DEV

    _internal:  int     # members started with "_" will be ignored
```
"""

# ------------------------------------------------------------------------------
# api
# ------------------------------------------------------------------------------

__all__ = [
    "load", "dump", "dump_str", "dump_meta",
    "Primitive", "FieldValue",
]

# Hack: Hide imported stuff for `dir(module)`
__dir__ = lambda: __all__

# ------------------------------------------------------------------------------
# imports
# ------------------------------------------------------------------------------

import argparse     as _argparse
import enum         as _enum
import inspect      as _inspect
import os           as _os
import sys          as _sys
import types        as _types
import typing       as _

# ------------------------------------------------------------------------------
# config
# ------------------------------------------------------------------------------

type Primitive  = bool|int|float|str|_enum.Enum
type FieldValue = Primitive|list[Primitive]|tuple[Primitive,...]|set[Primitive]

def load[T](
    cls: type[T],
    *,
    argv: _.Sequence[str]|None = None,
    env_prefix: str = "",
    overrides: dict[str, FieldValue]|None = None,
    priorities: _.Sequence[str] = ("overrides", "args", "envs", "defaults"),
    extra_sources: dict[str, _.Callable[[str], FieldValue|None]] = {},
) -> T:
    """ Build and populate a configuration instance from the given schema. """
    overrides = overrides or {}
    annotations: dict[str, type[FieldValue]] = {
        key: value
        for key, value in cls.__annotations__.items()
        if not key.startswith("_") # skip "private"
    }
    defaults: dict[str, FieldValue] = {
        k: getattr(cls, k)
        for k in annotations.keys()
        if hasattr(cls, k)
    }
    sources: dict[str, str] = {}

    def is_enum_t(t: type[FieldValue]) -> bool:
        return _inspect.isclass(t) and issubclass(t, _enum.Enum)

    def get_choices(e: type[_enum.Enum]) -> list[str]:
        return [m.name for m in e]

    def to_env(name: str) -> str:
        if env_prefix: return f"{env_prefix}_{name}".upper()
        return name.upper()
    def to_kebab(name: str) -> str: return name.lower().replace("_", "-")
    def to_cli(name: str) -> str:   return "--" + to_kebab(name)

    enum_map: dict[type[_enum.Enum], dict[str, _enum.Enum]] = {}
    def cast(
        value: str,
        to_type: type[FieldValue]
    ) -> tuple[FieldValue, Exception|None]:
        if to_type is bool:
            v = value.lower()
            if v in ("1", "true", "yes", "on"): return True, None
            if v in ("0", "false", "no", "off"): return False, None
            return value, ValueError(f"Invalid boolean: {value}")

        if to_type in (list, set, tuple):
            return to_type(v.strip()
                           for v in value.split(",")
                           if v.strip()), None

        if is_enum_t(to_type):
            mapping = enum_map.get(to_type) # type:ignore
            if not mapping:
                mapping = {m.name.lower(): m for m in to_type} # type:ignore
                enum_map[to_type] = mapping # type:ignore

            v = value.lower()
            if v in mapping: return mapping[v], None

            return value, ValueError(
                f"Invalid value '{value}' for {to_type.__name__}. "
                f"Expected one of: {', '.join(mapping.keys())}"
            )

        try:                   return to_type(value), None
        except Exception as e: return value, e

    def normalize(value: FieldValue) -> FieldValue:
        if isinstance(value, list):  return list(value)
        if isinstance(value, tuple): return tuple(value)
        if isinstance(value, set):   return set(value)

        return value

    def parse_cli() -> dict[str, FieldValue]:
        parser = _argparse.ArgumentParser()

        # build cli args
        for field, f_type in annotations.items():
            opts = {
                "default": None, # Note: if set will mask values
                "dest": field,
            }

            # extract `list` from `list[T]`
            origin = _.get_origin(f_type)

            if f_type is bool:
                # opts["type"] = bool # Note: `add_argument` doesn't allow it
                kebab_name = to_kebab(field)
                if field not in defaults:
                    parser.add_argument(f"--{kebab_name}",
                                        action="store_true", **opts)
                    parser.add_argument(f"--no-{kebab_name}",
                                        action="store_false", **opts)
                    continue

                if defaults.get(field): opts["action"] = "store_false"
                else:                   opts["action"] = "store_true"

            elif is_enum_t(f_type):
                opts["type"] = str
                opts["choices"] = get_choices(f_type) # type:ignore

            elif f_type in (list, tuple, set) or origin in (list, tuple, set):
                elem_type = str
                # extract `T` from `x[T]`
                args_t = _.get_args(f_type)

                if args_t:
                    elem_type = args_t[0]

                    if is_enum_t(elem_type):
                        opts["choices"] = get_choices(elem_type)
                        elem_type = str

                opts["type"] = elem_type
                opts["nargs"] = "*"

            else:
                opts["type"] = f_type

            parser.add_argument(to_cli(field), **opts)

        return vars(parser.parse_args(argv))

    def build() -> T:
        values: dict[str, FieldValue] = {}
        errors: list[Exception] = []
        args: dict[str, FieldValue] = parse_cli()

        source_fn_map: dict[str, _.Callable[[str], FieldValue|None]] = {
            "overrides":    lambda field: overrides.get(field),
            "args":         lambda field: args.get(field),
            "envs":         lambda field: _os.getenv(to_env(field)),
            "defaults":     lambda field: defaults.get(field),
        }
        source_fn_map.update(extra_sources)

        unknown = set(priorities) - source_fn_map.keys()
        if unknown:
            raise ValueError(f"Unknown config sources: {sorted(unknown)}")

        for field, f_type in annotations.items():
            value: FieldValue|None = None
            origin = _.get_origin(f_type)

            for source in priorities:
                getter = source_fn_map[source]
                if (value := getter(field)) is not None:
                    sources[field] = source
                    break

            if value is None:
                err = ValueError(
                    f"Missing required config '{field}'. "
                    f"Provide via cli ({to_cli(field)}), "
                    f"ENV ({to_env(field)}) "
                    f"or supply them directly in 'overrides' dict"
                )
                err.args = (f"{field}: {value}",)
                errors.append(err)
                continue

            if isinstance(value, str):
                v, ex = cast(value, f_type)

                if ex is not None:
                    ex.args = (f"{field}: {value}",)
                    errors.append(ex)
                    continue

                assert(v is not None)
                value = v

            elif ((to_type := f_type) in (list, tuple, set)
                  or (to_type := origin) in (list, tuple, set)):
                assert(isinstance(value, list))
                value = to_type(value)

            values[field] = value

        if errors: raise ExceptionGroup("Configuration errors", errors)

        instance = cls.__new__(cls)
        for k, v in values.items():
            setattr(instance, k, normalize(v))
        return instance

    instance = build()
    setattr(instance, "__config_schema__", {
        "cls": cls,
        "argv": argv,
        "env_prefix": env_prefix,
        "overrides": overrides,
        "annotations": annotations,
        "defaults": defaults,
        "sources": sources,
        "priorities": priorities,
    })

    return instance

def dump_meta(
    cfg: object,
) -> dict[str, tuple[FieldValue, type, str]]:
    """ Return configuration values with their type annotations and source. """
    schema: dict[str, _.Any]|None = getattr(cfg, "__config_schema__", None)
    if not isinstance(schema, dict):
        raise TypeError("Object was not created by config()")

    annotations: dict[str, type[FieldValue]] = schema["annotations"]
    sources: dict[str, str] = schema["sources"]

    return {
        field: (
            getattr(cfg, field),
            annotations[field],
            sources.get(field, "???")
        )
        for field in annotations.keys()
    }

def dump_str(
    cfg: object,
) -> str:
    """ Return a human-readable representation of the configuration. """
    schema: dict[str, _.Any]|None = getattr(cfg, "__config_schema__", None)
    if not isinstance(schema, dict):
        raise TypeError("Object was not created by config()")

    cls: type = schema["cls"]
    data: dict[str, tuple[FieldValue, type, str]] = dump_meta(cfg)

    def align(items: _.Iterable[str], max_align: int = 24) -> int:
        n = max((len(item) for item in items), default=0)
        n = min(n, max_align)
        return (n + 3) & ~3 # round up to multiple of 4

    def dump_type(tp: object) -> str:
        origin = _.get_origin(tp)

        if origin is None:
            if isinstance(tp, type): return tp.__name__
            return str(tp)

        args = _.get_args(tp)
        if not args: return origin.__name__

        return f"{origin.__name__}[{', '.join(dump_type(a) for a in args)}]"
    
    fields = sorted(data.keys())
    values = [str(data[f][0]) for f in fields]
    types = [dump_type(data[f][1]) for f in fields]
    srcs = [data[f][2] for f in fields]

    a_field = align(fields)
    a_tp = align(types)
    a_values = align(values)

    rep: str = f"[{cls.__name__}]"
    for f, v, t, s in zip(fields, values, types, srcs):
        rep += f"\n{f:{a_field}} : {t:{a_tp}} = {v:{a_values}} # {s}"

    return rep

def dump(
    cfg: object,
) -> dict[str, FieldValue]:
    """ Return the configuration as a flat dictionary. """
    return {
        field: value
        for field, (value, *_) in dump_meta(cfg).items()
    }

# ------------------------------------------------------------------------------
# callable module
# ------------------------------------------------------------------------------

# Hack: Makes the module callable, just for show

class _CallableModule(_types.ModuleType):
    def __call__[T](
        self,
        cls: type[T],
        *,
        argv: _.Sequence[str]|None = None,
        env_prefix: str = "",
        overrides: dict[str, FieldValue]|None = None,
        priorities: _.Sequence[str] = ("overrides", "args", "envs", "defaults"),
        extra_sources: dict[str, _.Callable[[str], FieldValue|None]] = {},
    ):
        return load(
            cls=cls,
            argv=argv,
            env_prefix=env_prefix,
            overrides=overrides,
            priorities=priorities,
            extra_sources=extra_sources,
        )

_sys.modules[__name__].__class__ = _CallableModule

del _CallableModule

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
