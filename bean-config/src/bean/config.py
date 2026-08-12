# ============================================================================ #
#                                                                              #
#                               ,---.      ,---.                               #
#                              /     `-<>-'  :D \                              #
#                              |                |                              #
#                               . .            .                               #
#                               .`-~~~~~~~~~~-'                                #
#                                                                              #
#                            Bean there, done that.                            #
#                                                                              #
# ============================================================================ #

__version__ = "0.2.1"
__doc__     = "Minimal config framework"
__author__  = "numen-0"
__license__ = "MIT"

# ------------------------------------------------------------------------------
# api
# ------------------------------------------------------------------------------

__all__ = [
    "Primitive", "FieldValue",

    "build",

    "validate", "validator", "is_valid",

    "normalize", "normalizer",

    "dump", "dump_str",

    "load",
]

# Hack: Hide imported stuff for `dir(module)`
__dir__ = lambda: __all__

# ------------------------------------------------------------------------------
# imports
# ------------------------------------------------------------------------------

import argparse     as _argparse
import dataclasses  as _dataclasses
import enum         as _enum
import inspect      as _inspect
import os           as _os
import sys          as _sys
import types        as _types
import typing       as _

# ------------------------------------------------------------------------------
# build
# ------------------------------------------------------------------------------

type Primitive  = bool|int|float|str|_enum.Enum
type FieldValue = Primitive|list[Primitive]|tuple[Primitive,...]|set[Primitive]

def build[T](
    cls: type[T],
    *,
    argv: _.Sequence[str]|None = None,
    env_prefix: str = "",
    overrides: dict[str, FieldValue]|None = None,
    priorities: _.Sequence[str] = ("overrides", "args", "envs", "defaults"),
    extra_sources: dict[str, _.Callable[[str], FieldValue|None]] = {},
) -> T:
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

        origin = _.get_origin(to_type)
        collection_type = origin or to_type

        if collection_type in (list, set, tuple):
            args = _.get_args(to_type)
            element_type = args[0] if args else str

            values: list[tuple[Primitive, Exception|None]] = [ # type:ignore
                cast(v.strip(), element_type)
                for v in value.split(",")
                if v.strip()
            ]

            errors = [v[1] for v in values if v[1] is not None]

            if errors:
                return value, ExceptionGroup(
                    f"Invalid values on container", errors
                )

            return collection_type(v[0] for v in values), None

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

    def norm(value: FieldValue) -> FieldValue:
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
            assert isinstance(value, (list, tuple, set))
            value = to_type(value)

        values[field] = value

    if errors: raise ExceptionGroup("Configuration errors", errors)

    instance = cls.__new__(cls)
    for k, v in values.items():
        setattr(instance, k, norm(v))
    # setattr(instance, "__bean_config__", True) # mark

    return instance

# ------------------------------------------------------------------------------
# validators
# ------------------------------------------------------------------------------

type ValidatorFn = _.Callable[..., bool|None]

@_dataclasses.dataclass(frozen=True, slots=True, eq=False)
class Validator:
    name: str
    fn: ValidatorFn
    fields: _.Sequence[str]
    priority: float = 100.0
    series: bool = False

    # = Note ================================================================= #
    #                                                                          #
    #  Exceptions raised while executing a validator are ambiguous:            #
    #                                                                          #
    #  - They may indicate a intentionally raised validation failure.          #
    #  - They may also be caused by an invalid validator definition (e.g. an   #
    #    incorrect function signature).                                        #
    #                                                                          #
    #  Since both cases raise from the validator call itself, they cannot be   #
    #  distinguished reliably at runtime. Therefore `test()` reports any       #
    #  raised exception back to the caller.                                    #
    #                                                                          #
    #  ```                                                                     #
    #  class Config:                                                           #
    #      DEBUG: bool = True                                                  #
    #                                                                          #
    #      # Definition error: missing required parameter                      #
    #      @config.validator("DEBUG")                                          #
    #      def foo(self):                                                      #
    #        return True                                                       #
    #                                                                          #
    #      # User-defined validation failure                                   #
    #      @config.validator()                                                 #
    #      def var(self):                                                      #
    #        raise Exception()                                                 #
    #  ```                                                                     #
    #                                                                          #
    #  Both cases raise during validator execution and are therefore treated   #
    #  identically.                                                            #
    #                                                                          #
    # ======================================================================== #
    def test(self, cfg: object) -> tuple[bool, Exception|None]:
        fn: ValidatorFn = getattr(cfg, self.name, self.fn)

        try:
            if self.series:
                if not self.fields:
                    return False, ValueError(
                        f"Validator '{self.name}' in series requires fields"
                    )

                ok = all( # Note: all(...) short circuits
                    fn(getattr(cfg, field)) is not False
                    for field in self.fields
                )

            elif self.fields:
                ok = fn(*(getattr(cfg, f) for f in self.fields))

            else:
                ok = fn()

        except Exception as ex:
            return False, ex

        return ok is not False, None

def validator[T: ValidatorFn](
    *fields: str,
    priority: float = 100.0,
    series: bool = False,
) -> _.Callable[[T], T]:
    """ Decorator to mark validation functions on config """

    def decorator(obj: T) -> T:
        # unwrap staticmethod/classmethod/instancemethod
        fn = getattr(obj, "__func__", obj)

        setattr(fn, "__bean_validator__", Validator(
            name=fn.__name__,
            fn=fn,
            fields=fields,
            priority=priority,
            series=series,
        ))

        return obj

    return decorator

def is_valid(cfg: object) -> bool:
    try:
        validate(cfg, short_circuit=True)
        return True
    except ExceptionGroup:
        return False

def validate[T: object](
    cfg: T,
    *,
    short_circuit: bool = False,
) -> T:
    """ Collect defined validation functions and validate the config """
    cls: type[T] = cfg.__class__
    members = vars(cls)

    validators: list[Validator] = sorted(
        [
            v
            for obj in members.values()
            if (v := getattr(
                    getattr(obj, "__func__", obj),
                    "__bean_validator__",
                    None
            )) is not None
        ],
        key=lambda v: (v.priority, v.name)
    )

    fields: set[str] = set().union(*(n.fields for n in validators))
    missing = sorted(
        field
        for field in fields
        if not hasattr(cfg, field)
    )
    if missing:
        raise KeyError(f"Validators reference unknown fields: {missing}")

    errors: list[Exception] = []

    for v in validators:
        ok, err = v.test(cfg)

        if not ok:
            errors.append(err or ValueError(
                f"Validator '{v.name}' failed for fields: {v.fields}"
            ))

            if short_circuit:
                break

    if errors:
        raise ExceptionGroup(
            f"Configuration validation failed for {cls.__name__}",
            errors
        )

    return cfg

# ------------------------------------------------------------------------------
# normalize
# ------------------------------------------------------------------------------

type NormalizerFn = _.Union[
    _.Callable[..., None],                  # 0 fields
    _.Callable[..., FieldValue],            # 1 fields or series=True
    _.Callable[..., tuple[FieldValue,...]], # N fields
]

@_dataclasses.dataclass(frozen=True, slots=True, eq=False)
class Normalizer:
    name: str
    fn: NormalizerFn
    fields: _.Sequence[str]
    priority: float = 100.0
    series: bool = False

    def apply[T: object](self, cfg: T) -> T:
        fn: NormalizerFn = getattr(cfg, self.name, self.fn)

        #* series (1 field)

        if self.series:
            if not self.fields:
                raise ValueError(
                    f"Normalizer '{self.name}' in series requires fields"
                )

            for field in self.fields:
                value = fn(getattr(cfg, field))

                if value is None:
                    raise TypeError(
                        f"Normalizer '{self.name}' returned None while "
                        f"on series for field '{field}'"
                    )

                setattr(cfg, field, value)
            return cfg

        #* 0 fields

        if not self.fields:
            if fn() is not None:
                raise TypeError(
                    f"Normalizer '{self.name}' must return None "
                    "when no fields are declared"
                )
            return cfg

        values = fn(*(getattr(cfg, f) for f in self.fields))
        if values is None: return cfg # SHOULD I DELETE THIS???

        #* 1 or N fields

        if len(self.fields) == 1:
            values = (values,)

        elif not isinstance(values, tuple):
            raise TypeError(
                f"Normalizer '{self.name}' must return a tuple "
                f"for {len(self.fields)} fields"
            )

        elif len(self.fields) != len(values):
            raise ValueError(
                f"Normalizer '{self.name}' returned "
                f"{len(values)} values for {len(self.fields)} fields"
            )

        for field, value in zip(self.fields, values):
            setattr(cfg, field, value)

        return cfg

def normalizer[T: NormalizerFn](
    *fields: str,
    priority: float = 100.0,
    series: bool = False,
) -> _.Callable[[T], T]:
    """ Decorator to mark normalization functions on config """

    def decorator(obj: T) -> T:
        # unwrap staticmethod/classmethod/instancemethod
        fn = getattr(obj, "__func__", obj)

        setattr(fn, "__bean_normalizer__", Normalizer(
            name=fn.__name__,
            fn=fn,
            fields=fields,
            priority=priority,
            series=series,
        ))

        return obj

    return decorator

def normalize[T](
    cfg: T,
    *,
    allow_overlap: bool = True,
) -> T:
    """ Collect defined normalization functions and normalize the config """
    cls: type[T] = cfg.__class__
    members = vars(cls)

    normalizers: list[Normalizer] = sorted(
        [
            n
            for obj in members.values()
            if (n := getattr(
                    getattr(obj, "__func__", obj),
                    "__bean_normalizer__",
                    None
            )) is not None
        ],
        key=lambda n: (n.priority, n.name)
    )

    fields: set[str] = set().union(*(n.fields for n in normalizers))
    missing = sorted(
        field
        for field in fields
        if not hasattr(cfg, field)
    )
    if missing:
        raise KeyError(f"Normalizers reference unknown fields: {missing}")

    if not allow_overlap:
        seen: set[str] = set()
        overlap: set[str] = set()

        for n in normalizers:
            fs = set(n.fields)
            overlap |= seen & fs
            seen |= fs

        if overlap:
            raise KeyError(
                f"Overlap in normalizers target fields: {sorted(overlap)}"
            )

    for n in normalizers:
        n.apply(cfg)

    return cfg

# ------------------------------------------------------------------------------
# dump
# ------------------------------------------------------------------------------

def dump(
    cfg: object,
) -> dict[str, FieldValue]:
    """ Return the configuration as a flat dictionary. """
    cls: type = type(cfg)

    return {
        field: getattr(cfg, field)
        for field in cls.__annotations__.keys()
        if not field.startswith("_") # skip "private"
    }

def dump_str(
    cfg: object,
) -> str:
    """ Return a human-readable representation of the configuration. """
    cls: type = type(cfg)
    data: dict[str, FieldValue] = dump(cfg)
    annotations: dict[str, type[FieldValue]] = {
        key: value
        for key, value in cls.__annotations__.items()
        if not key.startswith("_") # skip "private"
    }

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
    types = [dump_type(annotations[f]) for f in fields]

    a_field = align(fields)
    a_tp = align(types)

    rep: str = f"[{cls.__name__}]"
    for f, t in zip(fields, types):
        rep += f"\n{f:{a_field}} : {t:{a_tp}} = {data[f]}"

    return rep

# ------------------------------------------------------------------------------
# load
# ------------------------------------------------------------------------------

@_.overload
def load[T](
    cls: type[T],
    *,
    argv: _.Sequence[str]|None = None,
    env_prefix: str = "",
    overrides: dict[str, FieldValue]|None = None,
    priorities: _.Sequence[str] = ("overrides", "args", "envs", "defaults"),
    extra_sources: dict[str, _.Callable[[str], FieldValue|None]] = {},
    allow_overlap: bool = True,
    short_circuit: bool = False,
) -> T: ...

@_.overload
def load[T](
    *,
    argv: _.Sequence[str]|None = None,
    env_prefix: str = "",
    overrides: dict[str, FieldValue]|None = None,
    priorities: _.Sequence[str] = ("overrides", "args", "envs", "defaults"),
    extra_sources: dict[str, _.Callable[[str], FieldValue|None]] = {},
    allow_overlap: bool = True,
    short_circuit: bool = False,
) -> _.Callable[[type[T]], T]: ...

def load[T](
    cls: type[T] | None = None,
    *,
    argv: _.Sequence[str]|None = None,
    env_prefix: str = "",
    overrides: dict[str, FieldValue]|None = None,
    priorities: _.Sequence[str] = ("overrides", "args", "envs", "defaults"),
    extra_sources: dict[str, _.Callable[[str], FieldValue|None]] = {},
    allow_overlap: bool = True,
    short_circuit: bool = False,
) -> T|_.Callable[[type[T]], T]:
    """ Build and populate a configuration instance from the given schema. """
    if cls is None:
        def decorator(cls):
            return load(
                cls,
                argv=argv,
                env_prefix=env_prefix,
                overrides=overrides,
                priorities=priorities,
                extra_sources=extra_sources,
                allow_overlap=allow_overlap,
                short_circuit=short_circuit,
            )
        return decorator

    cfg = build(
        cls,
        argv=argv,
        env_prefix=env_prefix,
        overrides=overrides,
        priorities=priorities,
        extra_sources=extra_sources,
    )
    cfg = normalize(
        cfg,
        allow_overlap=allow_overlap,
    )
    cfg = validate(
        cfg,
        short_circuit=short_circuit,
    )

    return cfg

# Hack: Makes the module callable, just for show
class _CallableModule(_types.ModuleType):
    def __call__(self, *args, **kwargs):
        return load(*args, **kwargs)

_sys.modules[__name__].__class__ = _CallableModule
del _CallableModule

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
