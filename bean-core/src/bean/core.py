# =========================================================================== #
#                                                                             #
#                              ,---.      ,---.                               #
#                             /     `-<>-'  :D \                              #
#                             |                |                              #
#                              . .            .                               #
#                              .`-~~~~~~~~~~-'                                #
#                                                                             #
#                            Bean there, done that                            #
#                          Bean there, debugged that                          #
#                              May contain nuts                               #
#                                                                             #
# =========================================================================== #

__version__ = "0.1.0"
__doc__     = "tiny framework for bootstrapping apps"
__author__  = "numen-0"
__license__ = "MIT"

# -----------------------------------------------------------------------------
# api
# -----------------------------------------------------------------------------

__all__ = [
    "BeanApp",
    "Log", "Logger",
    "Result", "Success", "Predicate", "predicate",
    "Pipe",
    "Cmd", "sh", "cat", "tee", "stdout", "stderr",
    "install_signal_handlers", "shutdown_requested",
    "Scheduler",
    "BeanConfig", "ConfigField",
    "dirExists", "fileExists", "isDate", "isEmail", "isHost", "isIPv4",
    "isIPv6", "isNegative", "isPort", "isPositive", "isUrl", "nonEmpty",
    "pathExists",
    "main",
]

# -----------------------------------------------------------------------------
# imports
# -----------------------------------------------------------------------------

import re, signal, subprocess, sys, atexit
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from threading import Event, Thread
from time import sleep
from typing import (
    Callable, Literal, NoReturn, ClassVar, Protocol, Self,
    List, Dict, TextIO, Tuple, Iterable, Set,
    Any, TypeVar, Generic, Optional, override
)

# -----------------------------------------------------------------------------
# app
# -----------------------------------------------------------------------------

class _BeanMeta(ABCMeta):
    _initialized = False
    _guarded = { "NAME", "DEBUG" }

    def __getattribute__(cls, name):
        if (name in type.__getattribute__(cls, "_guarded")
            and not type.__getattribute__(cls, "_initialized")):
            raise RuntimeError(
                f"{cls.__name__} must be initialized before accessing '{name}'")
        return super().__getattribute__(name)

class BeanApp(ABC, metaclass=_BeanMeta):
    """ Bean module global config """

    NAME : str
    DEBUG : bool

    def __init__(
        self,
        name: str = "bean-app",
        debug: bool = False,
    ):
        if BeanApp._initialized:
            raise RuntimeError("BeanApp already initialized")

        BeanApp._initialized = True
        BeanApp.NAME = name
        BeanApp.DEBUG = debug

    def __getattribute__(self, name: str) -> Any:
        cls = type(self)
        if name in cls._guarded and not cls._initialized:
            raise RuntimeError(
                f"{cls.__name__} must be initialized before accessing '{name}'"
            )
        return super().__getattribute__(name)

    def startup(self) -> Optional[bool]: ...

    def shutdown(self) -> Optional[bool]: ...

    @abstractmethod
    def run(self) -> int: ...

del _BeanMeta

# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main(app: BeanApp) -> NoReturn:
    install_signal_handlers()

    try:
        if app.startup() is False:
            sys.exit(1)

        code = app.run()

    except Exception:
        raise

    finally:
        if app.shutdown() is False:
            code = 1

    sys.exit(code)

# -----------------------------------------------------------------------------
# logging
# -----------------------------------------------------------------------------

class Logger:
    """ Note:
    This began as a simple question:
    > "Why is the built-in logging module so complex?"

    After a brief episode of *NIH* syndrome and bit of shoveling, I ended up
    rebuilding a similar architecture.

    Turns out writing non-opinionated code is hard. ( >_<)
    """

    # logger classes & helpers

    class Level(Enum):
        FATAL = (4, "\033[1;7;91m")
        ERROR = (3, "\033[91m")
        WARN  = (2, "\033[93m")
        INFO  = (1, "\033[92m")
        DEBUG = (0, "\033[94m")

        def __init__(self, value: int, color: str):
            # https://docs.python.org/3/library/enum.html#enum.Enum._value_
            self._value_ = value # set internal Enum value
            self._color = color

        @property
        def color(self) -> str:
            return self._color

        # sugar

        @staticmethod
        def from_debug(debug: bool) -> Logger.Level:
            return Logger.Level.DEBUG if debug else Logger.Level.INFO

    @dataclass
    class Record:
        timestamp: datetime
        level: Logger.Level
        logger: str
        message: str

    class Handler(Protocol):
        @abstractmethod
        def log(self, record: Logger.Record): ...
        def flush(self): pass
        def close(self): pass

    @staticmethod
    def VoidHandler() -> Logger.Handler:

        class VoidHandler(Logger.Handler):
            @override
            def log(self, record: Logger.Record):
                 _ = record  # shut the warn
            def flush(self): pass
            def close(self): pass

        return VoidHandler()

    @staticmethod
    def CustomHandler(
        log: Callable[[Logger.Record], Any],
        flush: Callable[[], Any] = lambda: ...,
        close: Callable[[], Any] = lambda: ...,
    ) -> Logger.Handler:

        class CustomHandler(Logger.Handler):
            def __init__(self):
                from threading import RLock
                self._lock = RLock()
                self._closed = False

            @override
            def log(self, record: Logger.Record):
                with self._lock:
                    if self._closed: return
                    log(record)

            @override
            def flush(self):
                with self._lock:
                    if self._closed: return
                    flush()

            @override
            def close(self):
                with self._lock:
                    if self._closed: return
                    close()
                    self._closed = True

        return CustomHandler()

    @staticmethod
    def FileHandler(
        path: str,
        fmt_fn: Optional[Callable[[Logger.Record], str]] = None,
    ) -> Logger.Handler:
        fn = Logger.fmt(color=False) if fmt_fn is None else fmt_fn

        class FileHandler(Logger.Handler):
            def __init__(self):
                from threading import RLock
                self._lock = RLock()
                self._file = open(path, "a", buffering=1)  # line buffered
                self._closed = False

            @override
            def log(self, record: Logger.Record):
                with self._lock:
                    if self._closed: return
                    self._file.write(fn(record) + "\n")

            @override
            def flush(self):
                with self._lock:
                    if self._closed: return
                    self._file.flush()

            @override
            def close(self):
                with self._lock:
                    if self._closed: return
                    self._file.flush()
                    self._file.close()
                    self._closed = True

        return FileHandler()

    @staticmethod
    def TermHandler(
        fmt_fn: Optional[Callable[[Logger.Record], str]] = None,
        stream: Optional[TextIO] = None,
    ) -> Logger.Handler:
        fn = Logger.fmt() if fmt_fn is None else fmt_fn

        class TermHandler(Logger.Handler):
            def __init__(self):
                from threading import RLock
                self._lock = RLock()
                self._closed = False

            @override
            def log(self, record: Logger.Record) -> None:
                with self._lock:
                    if self._closed: return

                    if stream is not None:
                        out = stream
                    elif record.level in (Logger.Level.DEBUG,
                                          Logger.Level.INFO):
                        out = sys.stdout
                    else:
                        out = sys.stderr

                    print(fn(record), file=out, flush=True)

            @override
            def flush(self):
                with self._lock:
                    if self._closed: return
                    sys.stdout.flush()
                    sys.stderr.flush()

            @override
            def close(self):
                with self._lock:
                    self._closed = True

        return TermHandler()

    @staticmethod
    def fmt(
        *,
        color: bool = False,
        timestamp: bool = True,
        logger_name: bool = True,
    ) -> Callable[[Logger.Record], str]:

        def fmt(record: Logger.Record) -> str:
            parts = []

            if timestamp:
                parts.append(f"{record.timestamp:%Y-%m-%d %H:%M:%S}")

            parts.append(f"{record.level.name:<5}")

            if logger_name:
                parts.append(f"{record.logger:<24}")

            parts.append(record.message)

            line = " | ".join(parts)

            if color:
                return f"{record.level.color}{line}\033[0m"

            return line

        return fmt

    # Logger

    _active_loggers: Set[Logger] = set()
    def __init__(
        self,
        name: str,
        *,
        level: Level = Level.INFO,
        handlers: Optional[Iterable[Logger.Handler]] = None,
    ):
        self.name: str = name
        self.level: Logger.Level = level

        self._parent: Optional[Logger] = None
        self._children: List[Logger] = []

        self.handlers: List[Logger.Handler] = list(handlers) if handlers else []

        Logger._active_loggers.add(self)

    # context manager support

    def __enter__(self):
        return self
    def __exit__(self, *_):
        self.close()

        for child in self._children:
            del child

        if self._parent: self._parent._children.remove(self)
        Logger._active_loggers.discard(self)

        return False  # don't suppress exceptions

    def __del__(self):
        try: # attempt to flush before object is destroyed
            self.close()
            if self._parent: self._parent._children.remove(self)
            Logger._active_loggers.discard(self)
        except Exception: pass

    # logger gen/set

    @property
    def root(self) -> Logger:
        log = self
        while log._parent is not None:
            log = log._parent
        return log

    def child(
        self,
        name: str,
        *,
        level: Optional[Level] = None,
        handlers: Optional[List[Logger.Handler]] = None,
    ) -> Logger:
        handlers = list(handlers) if handlers else []

        log = Logger(
            name=name,
            level=level if level is not None else self.level,
            handlers=handlers + self.handlers,
        )

        log._parent = self
        self._children.append(log)

        return log

    def update(
        self,
        name: Optional[str] = None,
        *,
        level: Optional[Level] = None,
        handlers: Optional[Logger.Handler|Iterable[Logger.Handler]] = None,
        cascade: bool = True
    ) -> Self:
        if name is not None:  self.name = name
        if level is not None: self.set_level(level, cascade)
        if handlers is not None:  self.add_handlers(handlers, cascade=cascade)
        return self

    def set_level(self, level: Level, cascade: bool = True) -> Self:
        self.level = level
        if cascade:
            for child in self._children:
                child.set_level(level, cascade)
        return self

    def set_handlers(
        self,
        handlers: Optional[Logger.Handler|Iterable[Logger.Handler]] = None,
        cascade: bool = True
    ) -> Self:
        if handlers is None:
            self.handlers = []
        elif isinstance(handlers, Iterable):
            self.handlers = list(handlers)
        else:
            self.handlers = [handlers]

        if cascade:
            for child in self._children:
                child.set_handlers(self.handlers, cascade)

        return self

    def add_handlers(
        self,
        handlers: Optional[Logger.Handler|Iterable[Logger.Handler]] = None,
        cascade: bool = True
    ) -> Self:
        if handlers is None:
            return self
        elif isinstance(handlers, Iterable):
            handlers = list(handlers)
        else:
            handlers = [handlers]

        self.handlers.extend(handlers)
        if cascade:
            for child in self._children:
                child.add_handlers(handlers, cascade=cascade)

        return self

    # log

    def _log(self, level: Level, msg: str) -> Self:
        if level.value < self.level.value: return self

        def logger_path(logger: Logger) -> str:
            parts = []
            log = logger
            while log:
                parts.append(log.name)
                log = log._parent
            return ".".join(reversed(parts))

        record = Logger.Record(
            timestamp=datetime.now(),
            level=level,
            logger=logger_path(self),
            message=msg,
        )
        for handler in self.handlers: handler.log(record)

        return self

    # Standard log methods
    def debug(self, msg: str) -> Self:
        return self._log(Logger.Level.DEBUG, msg)
    def info(self, msg: str) -> Self:
        return self._log(Logger.Level.INFO, msg)
    def warning(self, msg: str) -> Self:
        return self._log(Logger.Level.WARN, msg)
    def error(self, msg: str) -> Self:
        return self._log(Logger.Level.ERROR, msg)
    def fatal(self, msg: str) -> Self:
        return self._log(Logger.Level.FATAL, msg)

    # mimic logging.exception
    def exception(self, msg: str) -> Self:
        """Log an error message with the current exception traceback."""
        import traceback

        if sys.exc_info()[0] is not None: # check if is there is any exception
            msg = f"{msg}\n{traceback.format_exc()}"

        return self._log(Logger.Level.ERROR, msg)

    def flush(self, cascade: bool = False) -> Self:
        for handler in self.handlers:
            handler.flush()

        if cascade:
            for child in self._children:
                child.flush(cascade=cascade)

        return self

    def close(self, cascade: bool = False) -> Self:
        for handler in self.handlers:
            handler.flush()
            handler.close()

        if cascade:
            for child in self._children:
                child.close()

        return self

    @atexit.register
    @staticmethod
    def close_all_loggers():
        for log in list(Logger._active_loggers):
            try:
                log.flush()
                log.close()
            except Exception: pass

    # sugar

    @staticmethod
    def init(
        name: str = "bean",
        level: Logger.Level = Level.INFO,
        fmt: Optional[Callable[[Logger.Record], str]] = None,
        handlers: Optional[List[Logger.Handler]] = None,
    ) -> Logger:
        """ Initialize root logger. """

        if fmt is None:
            fmt = Logger.fmt()
        if handlers is None:
            handlers = [Logger.TermHandler(fmt)]

        global Log

        Log.update(name, level=level, handlers=handlers)
        return Log

Log = Logger("bean")

# -----------------------------------------------------------------------------
# type classes
# -----------------------------------------------------------------------------

_S = TypeVar("_S")  # Success type
_E = TypeVar("_E")  # Error type
_R = TypeVar("_R")  # Remap type for maps

@dataclass(frozen=True)
class Result(Generic[_S, _E]):
    ok: bool = False
    value: Optional[_S] = None
    error: Optional[_E] = None

    # constructors

    @staticmethod
    def Ok(value: _S) -> Result[_S, _E]:
        return Result(value=value, ok=True)

    @staticmethod
    def Error(error: _E) -> Result[_S, _E]:
        return Result(error=error, ok=False)

    @staticmethod
    def from_tuple(t: Tuple[_S, bool]) -> Result[_S, _S]:
        """
        Convert a (value, ok) tuple into a Result.
        error is used if ok == False
        """
        val, ok = t
        if ok:
            return Result.Ok(val)
        return Result.Error(val)

    # methods

    def __bool__(self) -> bool:
        """ Allow `if result:` syntax to check success. """
        return self.ok

    def __eq__(self, other) -> bool:
        if isinstance(other, tuple) and len(other) == 2:
            return self.to_tuple() == other
        if isinstance(other, Result):
            return self.value == other.value and self.ok == other.ok
        return NotImplemented

    def to_success(self) -> Success[_S|_E]:
        return Success.from_tuple(self.to_tuple())
    def to_tuple(self) -> Tuple[_S, Literal[True]]|Tuple[_E, Literal[False]]:
        if self.ok: return (self.value, True) # type: ignore
        return (self.error, False) # type: ignore

    def unwrap(self) -> _S:
        """ Return the value. Raise exception if has no `.value` """
        if not self.ok:
            raise RuntimeError(f"Unwrapped error Result: {self.error}")
        return self.value # type: ignore

    def unwrap_or(self, default: _S) -> _S:
        """ Return the value if ok, else return default. """
        if not self.ok: return default
        return self.value # type: ignore

    def unwrap_err(self) -> _E:
        """ Return the error. Raise exception if has no `.value` """
        if self.ok: raise RuntimeError(f"Result has no error")
        return self.error # type: ignore

_T = TypeVar("_T")  # Generic type

@dataclass(frozen=True)
class Success(Generic[_T]):
    value: _T
    ok: bool

    # constructors

    @staticmethod
    def Ok(value: _T) -> Success[_T]:
        if value is None:
            raise TypeError("Success.Ok cannot hold None")
        return Success(value=value, ok=True)

    @staticmethod
    def Fail() -> Success[None]:
        return Success(value=None, ok=False)

    @staticmethod
    def from_tuple(t: Tuple[_T, bool]) -> Success[_T]:
        """
        Convert a (value, ok) tuple into a Success.
        Ignores the value if ok==False
        """
        return Success(*t)

    @staticmethod
    def from_obj(obj: Tuple[_T, bool]|Success[_T]) -> Success[_T]:
        if not isinstance(obj, Success):
            return Success.from_tuple(obj)
        return Success(obj.value, obj.ok)

    # methods

    def __bool__(self) -> bool:
        """ Allow `if result:` syntax to check success. """
        return self.ok

    def __eq__(self, other) -> bool:
        if isinstance(other, tuple) and len(other) == 2:
            return self.to_tuple() == other
        if isinstance(other, Success):
            return self.value == other.value and self.ok == other.ok
        return NotImplemented

    def to_result(self) -> Result[_T, _T]:
        return Result.from_tuple(self.to_tuple())
    def to_tuple(self) -> Tuple[_T, bool]:
        return (self.value, self.ok)

class Predicate(Generic[_T]):
    def __init__(self, fn: Callable[[_T], bool], name: str | None = None):
        self.fn = fn
        self.name = name or fn.__name__

    def __call__(self, value: _T) -> bool:
        return bool(self.fn(value))

    def __and__(self, other: Predicate[_T]) -> Predicate[_T]:
        return Predicate(
            lambda v: self(v) and other(v),
            name=f"({self.name} & {other.name})"
        )

    def __or__(self, other: Predicate[_T]) -> Predicate[_T]:
        return Predicate(
            lambda v: self(v) or other(v),
            name=f"({self.name} | {other.name})"
        )

    def __invert__(self) -> Predicate[_T]:
        return Predicate(
            lambda v: not self(v),
            name=f"(!{self.name})"
        )

    def __repr__(self):
        return f"<Predicate {self.name}>"

    def trace(self, value: _T, log=None) -> bool:
        res = self(value)
        if log: log.debug(f"check {self.name} -> {res}")
        return res

# Suggar
def predicate(fn: Callable[[_T], bool]) -> Predicate[_T]:
    """ Wrap a predicate into a Obj, enabling logical composition. """
    return Predicate(fn)

# -----------------------------------------------------------------------------
# pipes
# -----------------------------------------------------------------------------

_I = TypeVar("_I") # Input type
_O = TypeVar("_O") # Output type
_R = TypeVar("_R") # Result type
_E = TypeVar("_E") # Error type
_P = TypeVar("_P") # Pipe result

PipeResult = Success[_P] | tuple[_P, bool]

class Pipe(Generic[_I, _O]):
    """ A composable transformation pipeline.

    - Each pipe stage receives a value and returns a `Success[O]` or a
      equivalent `Tuple[O, bool]`.
    - If `ok == False`, the pipeline short-circuits and no further stages are
      executed.

    Typing note:
        Some `# type: ignore` annotations are used internally to preserve type
        flow across pipe composition. This is needed because Python's type
        system cannot fully express the short-circuit semantics of the pipeline
        without breaking the generic type chain.

        At runtime, only values from successful stages (`ok == True`) are passed
        to the next pipe, so the type assumptions remain safe in practice.
    """

    def __init__(
        self,
        fn: Callable[[_I], PipeResult[_O]] = lambda v: Success.Ok(v)
    ):
        self._fn = fn

    def __call__(self, value: _I) -> Success[_O]:
        return Success.from_obj(self._fn(value))

    def __or__(
        self,
        other: Callable[[_O], PipeResult[_R]]
    ) -> Pipe[_I, _R]:
        if not isinstance(other, Pipe):
            other = Pipe(other)

        def join(value: _I) -> Success[_R]:
            res = self(value)
            if res.ok: res = other(res.value)
            return res # type: ignore

        return Pipe(join)

    def retry(
        self,
        fn: Callable[[_O], PipeResult[_R]],
        attempts: int = 3,
        delay: float = 0.5,
    ) -> Pipe[_I, _R]:
        """ Wrap a pipe-compatible function with retry logic.

        The returned function will call `fn` up to `attempts` times, sleeping
        `delay` seconds between failures. Succeeds as soon as any attempt
        returns `ok == True`.

        Returns:
          - `(value,      True)`  if any try succeed
          - `(last_value, False)` if all functions fail
        """
        assert 0 < attempts

        def foo(value: _O) -> Success[_R]:
            res = None
            for _ in range(attempts):
                res = Success.from_obj(fn(value))
                if res.ok: return res
                if delay > 0: sleep(delay)

            assert res is not None
            return res

        return self | Pipe(foo)

    def map(
        self,
        fn: Callable[[_O], _R],
    ) -> Pipe[_I, _R]:
        """ Lift a pure function into a pipe-compatible function.

        Returns: `(fn(value), True)`
        """
        def foo(value: _O): return Success.Ok(fn(value))
        return self | Pipe(foo)

    def guard(
        self,
        fn: Callable[[_O], bool],
    ) -> Pipe[_I, _O]:
        """ Validate a value inside a pipe.

        Returns:
          - `(value, True)`  if guard passes (`fn(value) == True`)
          - `(value, False)` if guard fails
        """
        def foo(value: _O) -> Success[_O]:
            if fn(value): return Success.Ok(value)
            return Success(value, False)

        return self | Pipe(foo)

    def peek(
        self,
        fn: Callable[[_O], Any],
    ) -> Pipe[_I, _O]:
        """ Execute a side-effect without modifying the value.

        Commonly used for logging, metrics, or debugging.

        Returns: `(value, True)`
        """
        def foo(value: _O):
            fn(value)
            return Success.Ok(value)

        return self | Pipe(foo)

    def fallback(
        self,
        fn: Callable[[_O], PipeResult[_R]],
        fb: _E
    ) -> Pipe[_I, _R|_E]:
        """ Convert a failing function into a successful one with a fallback.

        Useful for optional steps that should not stop the pipe.

        Returns:
          - `(value,    True)` if fn passes
          - `(fb_value, True)` if fn fails
        """
        def foo(value: _O):
            res = Success.from_obj(fn(value))
            return Success.Ok(res.value if res.ok else fb)

        return self | Pipe(foo)

    def branch(
        self,
        cond_fn: Callable[[_O], bool],
        success_fn: Optional[Callable[[_O], PipeResult[_R]]] = None,
        fail_fn: Optional[Callable[[_O], PipeResult[_E]]] = None,
    ) -> Pipe[_I, _O|_R|_E]:
        """ Conditional branching inside a pipe.

        If the selected branch function is None, the value is passed through
        and the condition result is used as the ok flag.

        Returns:
          - `(value,   cond)` if no matching fn is set
          - `(s_value, bool)` if `cond_fn` passes and `success_fn` is provided
          - `(f_value, bool)` if `cond_fn` fails and `fail_fn` is provided
        """
        def foo(value: _O) -> Success[_O|_R|_E]:
            cond = cond_fn(value)
            if cond and success_fn is not None:
                return Success.from_obj(success_fn(value))
            elif not cond and fail_fn is not None:
                return Success.from_obj(fail_fn(value))
            return Success(value, cond)

        return self | Pipe(foo)

    def trigger(
        self,
        fn: Callable[[_O], bool],
        ex: type[Exception] = Exception,
        msg: Optional[str] = None
    ) -> Pipe[_I, _O]:
        """ Validate a value inside a pipe.

        Returns:
          - raise `ex(msg)` if trigger passes  (`fn(value) == True`)
          - `(value, True)` if trigger fails
        """
        def foo(value: _O) -> Success[_O]:
            if fn(value):
                raise ex(msg or f"Pipe trigger activated on value: {value}")
            return Success.Ok(value)

        return self | Pipe(foo)

# -----------------------------------------------------------------------------
# shell
# -----------------------------------------------------------------------------

class Cmd:
    @dataclass
    class Result:
        code: int
        out: str
        err: str

        def __bool__(self) -> bool:
            return self.code == 0

        def __str__(self) -> str:
            return self.out

        @property
        def ok(self) -> bool:
            return bool(self)

        @staticmethod
        def Fail(code: int = -1, err: str = "") -> Cmd.Result:
            return Cmd.Result(code, "", err)

        @staticmethod
        def from_tuple(t: Tuple[str, bool]) -> Cmd.Result:
            txt, ok = t
            if ok: return Cmd.Result(ok, txt, "")
            return Cmd.Result(ok, "", txt)

        def to_tuple(self) -> Tuple[str, bool]:
            return (self.out if self.ok else self.err, self.ok)
        def to_success(self) -> Success[str]:
            return Success.from_tuple(self.to_tuple())
        def to_result(self) -> Result[str, str]:
            return Result.from_tuple(self.to_tuple())

    def __init__(
        self,
        cmd: str | list[str],
        shell: Optional[bool] = None,
    ):
        self.cmd: str | list[str] = cmd
        if shell is None:
            self.shell = isinstance(cmd, str)
        else:
            self.shell: bool = shell

    def __call__(
        self,
        cwd: Optional[str] = None,
        timeout: Optional[float] = None,
        input: Optional[str] = None,
    ) -> Cmd.Result:
        return self.run(cwd, timeout, input)

    def run(
        self,
        cwd: Optional[str] = None,
        timeout: Optional[float] = None,
        input: Optional[str] = None,
    ) -> Cmd.Result:
        try:
            proc = subprocess.run(
                self.cmd,
                shell=self.shell,
                cwd=cwd,
                timeout=timeout,
                input=input,
                capture_output=True,
                text=True,
            )
            return Cmd.Result(proc.returncode, proc.stdout, proc.stderr)

        except Exception as e:
            return Cmd.Result.Fail(err=str(e))

def sh(
    cmd: str | list[str],
    shell: Optional[bool] = None,
    cwd: Optional[str] = None,
    timeout: Optional[float] = None,
) -> Pipe[Optional[str|Cmd.Result], Cmd.Result]:
    if shell is None: shell = isinstance(cmd, str)

    def foo(input: Optional[str|Cmd.Result]):
        if isinstance(input, Cmd.Result):
            input = input.out
        res = Cmd(cmd, shell).run(cwd, timeout, input)
        return Success(res, res.ok)

    return Pipe(foo)

def stdout() -> Pipe[Cmd.Result, str]:
    def foo(res: Cmd.Result):
        return Success(res.out, res.ok)
    return Pipe(foo)
def stderr() -> Pipe[Cmd.Result, str]:
    def foo(res: Cmd.Result):
        return Success(res.err, res.ok)
    return Pipe(foo)

def tee(
    *paths: str,
    append: bool = False,
) -> Pipe[Optional[str|Cmd.Result], Cmd.Result]:

    def foo(input):
        output = "" if input is None else str(input)

        mode = "a" if append else "w"
        ok = True
        for path in paths:
            try:
                with open(path, mode) as f:
                    f.write(output)
            except Exception:
                ok = False

        return Success(Cmd.Result(int(not ok), output, ""), ok)

    return Pipe(foo)

def cat(
    *paths: str
) -> Pipe[Optional[str|Cmd.Result], Cmd.Result]:

    def foo(input):
        output = "" if input is None else str(input)

        ok = True
        for path in paths:
            try:
                with open(path, "r") as f:
                    output += f.read()
            except Exception:
                ok = False

        return Success(Cmd.Result(int(not ok), output, ""), ok)

    return Pipe(foo)

# -----------------------------------------------------------------------------
# signals
# -----------------------------------------------------------------------------

_shutdown_event = Event()
_installed = False

def install_signal_handlers(
    cb: Optional[Callable[[int, Any], Any]] = None
):
    """ Install SIGINT / SIGTERM handlers.

    Signals set a shutdown flag that can be checked by the app.
    """
    global _installed
    if _installed: return
    _installed = True

    force = False

    def handler(signum, frame):
        nonlocal force

        if _shutdown_event.is_set():
            if force: sys.exit(1)

            force = True
            Log.warning(
                "shutdown already in progress, press Ctrl+C again to force exit")
            return

        Log.info(f"Gracefully shutting down. Please wait...")
        _shutdown_event.set()
        if cb is not None:
            cb(signum, frame)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, handler)

def shutdown_requested() -> bool:
    """ Return True if a shutdown signal was received. """
    return _shutdown_event.is_set()

# -----------------------------------------------------------------------------
# scheduler
# -----------------------------------------------------------------------------

class Scheduler:
    class Task:
        """ Wrap a callable with execution metadata """
        def __init__(
            self,
            fn: Callable,
            delay: float = 0,
            interval: Optional[float] = None,
            runs: Optional[int] = None,
        ):
            self.fn = fn
            self.delay = delay
            self.interval = interval
            self.runs = runs
            self._thread: Optional[Thread] = None
            self._stop_event = Event()

        def start(self):
            if not self._thread or not self._thread.is_alive():
                self._thread = Thread(target=self._run, daemon=True)
                self._thread.start()

        def _run(self):
            if self.delay > 0: sleep(self.delay)
            if self.runs is not None and self.runs < 1: return

            count = 0
            def active():
                if _shutdown_event.is_set() or self._stop_event.is_set():
                    return False

                if self.runs is None:
                    return True

                nonlocal count
                count += 1
                return count < self.runs

            while True:
                try:
                    self.fn()
                except Exception:
                    Log.exception(f"task {self.fn} failed")

                if not active(): break

                if self.interval is not None:
                    sleep(self.interval)

        def stop(self):
            self._stop_event.set()

        def join(self, timeout: Optional[float] = None):
            if (self._thread and self._thread.is_alive()
                    and self.runs is not None): # dont block with inf tasks
                self._thread.join(timeout)

    def __init__(self):
        self.tasks: List[Scheduler.Task] = []

    # context manager support

    def __enter__(self) -> Scheduler:
        return self.start()

    def __exit__(self, *_):
        self.stop().join()
        return False

    def task(
        self,
        fn: Callable,
        runs: Optional[int] = 1,
        delay: float = 0,
    ) -> Self:
        """ Add a N-run task (delay optional). """
        self.tasks.append(Scheduler.Task(fn, delay=delay, runs=runs))
        return self

    def job(
        self,
        fn: Callable,
        interval: float,
        runs: Optional[int] = None
    ) -> Self:
        """ Add a periodic job with the given interval. """
        self.tasks.append(Scheduler.Task(fn, interval=interval, runs=runs))
        return self

    def start(self) -> Self:
        """ Start all scheduled tasks/jobs. """
        for t in self.tasks: t.start()
        return self

    def join(self, timeout: Optional[float] = None) -> Self:
        """ Join all scheduled tasks/jobs and wait for threads to finish. """
        if timeout is None:
            for t in self.tasks: t.join(None)
            return self

        import time
        end = time.time() + timeout

        for t in self.tasks:
            remaining = end - time.time()
            if remaining <= 0: break
            t.join(remaining)

        return self

    def stop(self) -> Self:
        """ Stop all scheduled tasks/jobs. """
        for t in self.tasks: t.stop()
        return self

    def clear(self) -> Self:
        """ Remove all scheduled tasks/jobs from the scheduler. """
        self.tasks.clear()
        return self

# -----------------------------------------------------------------------------
# config
# -----------------------------------------------------------------------------

FieldValue = bool|int|float|str|List[str]

_C = TypeVar("_C", bound="BeanConfig")

class BeanConfig(ABC):
    """ Abstract base class for application configuration. """

    class _ConfigField(Generic[_T]):
        def __init__(
            self,
            type: type[_T],
            description: Optional[str] = None,
            default: Optional[_T] = None,
            validator: Optional[Predicate[_T]|Callable[[_T], bool]] = None,
            required: bool = False,
        ):
            self.type = type
            self.description = description
            self.default = default
            self.validator = validator
            self.required = required

        def __set_name__(self, _, name): self._name = name

        def __get__(self, instance, owner) -> _T:
            """ HACK!!!
            Descriptor access logic.

            This descriptor serves two roles depending on how/when it is accessed:

            1) Class access: `MyConfig.PORT`
                - Before configuration is built:
                    returns the _ConfigField descriptor itself.
                    This allows `BeanConfig.spec()` to discover the schema.
                - After configuration is built:
                    returns the resolved runtime value from the singleton instance.

            2) Instance access: `cfg.PORT`
                - Always returns the stored runtime value from the instance.

            This allows the same attribute to behave as:
                - a schema definition at class definition time
                - a typed runtime value after configuration is built

            Static type checkers interpret this descriptor as returning `T`, which
            provides correct autocomplete and type checking for config values when
            accessed from either the instance or the class.
            """

            # instance access
            if instance is not None:
                return instance.__dict__.get(self._name, self.default)

            # class access
            inst = getattr(owner, "_instance", None)

            if inst is None:
                return self     # type: ignore  # before build -> schema
            return inst.__dict__[self._name]    # after build  -> value

    _instance: ClassVar["BeanConfig | None"] = None
    _spec: ClassVar[Dict[str, _ConfigField]] = {}

    def __init_subclass__(cls, **kwargs):
        """ Build the immutable schema `_spec` at class definition time.

        - Scans the entire MRO to include inherited ConfigFields
        - Stores them on the class to avoid repeated introspection
        - Makes config schema "static"
        """
        super().__init_subclass__(**kwargs)

        fields: Dict[str, BeanConfig._ConfigField] = {}
        for base in reversed(cls.__mro__):      # python magic so we also set
            for k, v in base.__dict__.items():  # inherited fields...
                if isinstance(v, BeanConfig._ConfigField):
                    fields[k] = v

        cls._spec = fields

    @staticmethod
    def validate(
        field_name: str
    ) -> Callable[[Callable[[_T], bool]], Callable[[_T], bool]]:
        """ decorator to make custom validators on function definition """
        def wrapper(fn: Callable[[_T], bool]) -> Callable[[_T], bool]:
            setattr(fn, "_validate_field", field_name)
            return fn
        return wrapper

    @classmethod
    def _set_instance(cls, obj: BeanConfig):
        cls._instance = obj

    @classmethod
    def spec(cls) -> Dict[str, _ConfigField]:
        return cls._spec

    @classmethod
    def load(cls: type[_C]) -> BeanConfig._ConfigLoader[_C]:
        return BeanConfig._ConfigLoader(cls)

    @classmethod
    def cli_help(cls):
        import argparse

        def to_cli(key: str) -> str:
            return f"--{key.replace('_', '-').lower()}"

        parser = argparse.ArgumentParser()
        for k, f in cls.spec().items():
            arg_key = to_cli(k)
            t = f.type if f.type != list else str   # lists parsed from str
            parser.add_argument(
                arg_key,
                dest=k,
                type=t,
                default=f.default,
                help=f.description,
                required=f.required,
            )
        parser.print_help()

    @classmethod
    def print_config(
        cls,
        show_types: bool = False,
        show_defaults: bool = False,
        show_required: bool = False,
    ) -> None:
        """Pretty-print the current configuration."""

        if cls._instance is None:
            raise RuntimeError("Config not loaded")

        inst = cls._instance
        spec = cls.spec()

        # determine column width
        width = max(len(k) for k in spec.keys()) if spec else 0

        print(f"{cls.__name__}:")
        print("-" * min(width + 30, 80))

        for key, field in spec.items():
            value = getattr(inst, key, None)
            line = f"{key:<{width}} : {value}"

            extras = []
            if show_types: extras.append(f"type={field.type.__name__}")
            if show_defaults: extras.append(f"def={field.default}")
            if show_required: extras.append(f"req={field.required}")
            if extras: line += f" ({", ".join(extras)})"

            print(line)

    class _ConfigLoader(Generic[_C]):
        """ App config loader/builder

        Receives a schema, and loads sources to fill the config, validating it
        and generating a final config as a product
        """

        def __init__(self, config_cls: type[_C]):
            self.config_cls = config_cls
            self._values: Dict[str, Any] = {}
            self._locked = False

        # post config steps

        def validate(self) -> BeanConfig._ConfigLoader[_C]:
            """ validate current config """
            import inspect

            spec = self.config_cls.spec()
            # default and normal validators
            for k, f in spec.items():
                val = self._values.get(k, None)

                if val is None:
                    if f.default is not None:
                        val = f.default
                    elif f.required:
                        raise ValueError(f"Missing required config: {k}")

                elif not isinstance(val, f.type):
                    raise TypeError(f"Invalid type '{type(val).__name__}' for value {k} ({val})")

                elif f.validator and not f.validator(val):
                    raise ValueError(f"Invalid value for {k}: {val}")

                self._values[k] = val

            # class-level per-field validators
            for name, method in inspect.getmembers(self.config_cls,
                                                   predicate=callable):
                field: Optional[str] = getattr(method, "_validate_field", None)
                if field is None: continue

                if field not in spec:
                    raise KeyError(f"Class-level validator '{name}' key '{field}' is not in spec")

                val = self._values.get(field, None)
                if val is not None and not method(val):
                    raise ValueError(f"Class-level validation failed for {field}: {name}({val})")

            return self

        def build(self) -> _C:
            """validate and build the config"""
            obj = self.validate().config_cls()

            # Note: we override the instance attr not the class attributes
            for k, v in self._values.items(): setattr(obj, k, v)

            self.config_cls._set_instance(obj)
            return obj

        # access

        def as_dict(self):
            return dict(self._values)

        def __getitem__(self, key):
            return self._values[key]

        def __getattr__(self, name):
            if name in self._values:
                return self._values[name]
            raise AttributeError(name)

        # helpers

        @staticmethod
        def _cast_type(val: Any, field: BeanConfig._ConfigField):
            """ Casts value to match the expected field type"""
            if val is None: return None

            t = field.type
            if isinstance(val, t): return val

            try:
                if t is bool:
                    if not isinstance(val, str): return bool(val)
                    v = val.lower()
                    if v in ("1", "true", "yes", "on"): return True
                    if v in ("0", "false", "no", "off"): return False
                    return None

                if t is int:    return int(val)
                if t is float:  return float(val)
                if t is str:    return str(val)

                if t is list:
                    if not isinstance(val, str): return [val] # Type error?
                    return [x.strip() for x in val.split(",")]

            except Exception: pass
            return val

        @staticmethod
        def _file_exists(path: str) -> bool:
            conf_path = Path(path)
            return conf_path.exists() and conf_path.is_file()

        # source loaders

        def _from_source(
            self,
            source: str,
            data: Dict[str, Any],
            key_mapper: Callable[[str], str] = lambda k: k
        ) -> Self:
            """
            Generic loader: cast and store values from any source.

            - data: dict of raw key -> value
            - source: string for logging purposes
            - key_mapper: function to normalize source keys to config keys
            """

            spec = self.config_cls.spec()
            for raw_key, raw_val in data.items():
                key = key_mapper(raw_key)
                if key not in spec:
                    Log.debug(f"unknown key '{raw_key}' in config from '{source}'")
                    continue

                val = BeanConfig._ConfigLoader._cast_type(raw_val, spec[key])
                if val is None:
                    Log.warning(f"Invalid value for {raw_key} ({raw_val}) from '{source}', skipping...")
                    continue
                self._values[key] = val

            return self

        def from_dict(self, d: Dict) -> Self:
            return self._from_source("dict", d)

        def from_json(
            self,
            path: str,
            force: bool = False
        ) -> Self:
            if not BeanConfig._ConfigLoader._file_exists(path):
                if force: raise FileNotFoundError(f"File '{path}' not found.")
                return self

            import json
            with open(path) as f:
                data = json.load(f)

            return self._from_source(path, data,
                                    lambda k: k.upper().replace("-", "_"))

        def from_args(self, args: Optional[list[str]] = None) -> Self:
            import argparse

            def to_cli(key: str) -> str:
                return f"--{key.replace('_', '-').lower()}"

            parser = argparse.ArgumentParser()
            spec = self.config_cls.spec()
            for k, f in spec.items():
                arg_key = to_cli(k)
                t = f.type if f.type != list else str   # lists parsed from str
                parser.add_argument(
                    arg_key,
                    dest=k,             # store as spec key
                    type=t,
                    default=None,       # if set it will mask values
                    help=f.description
                )

            parsed = parser.parse_args(args)
            return self._from_source("args", vars(parsed))

        def from_env(self, prefix: str = "") -> Self:
            import os

            def normalize_keys(items):
                return { k.removeprefix(prefix).upper(): v for k, v in items }

            spec = self.config_cls.spec()
            return self._from_source("env", {
                    k: v
                    for k, v in normalize_keys(os.environ.items()).items()
                    if k in spec
                })

        def from_py(
            self,
            path: str,
            symbol: str = "Config",
            force: bool = False
        ) -> Self:
            if not BeanConfig._ConfigLoader._file_exists(path):
                if force: raise FileNotFoundError(f"File '{path}' not found.")
                return self

            import importlib.util as IU
            spec = IU.spec_from_file_location("user_config", path)
            if spec is None or spec.loader is None:
                raise FileNotFoundError(f"Cannot load config file: {path}")

            mod = IU.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if not hasattr(mod, symbol):
                raise ValueError(f"Python config must expose `{symbol}` object")

            obj = getattr(mod, symbol)
            if isinstance(obj, dict):
                return self._from_source(path, obj)

            data = {
                k: getattr(obj, k)
                for k, _ in self.config_cls.spec().items()
                if not k.startswith("_")
            }

            return self._from_source(path, data)

        def from_toml(
            self,
            path: str,
            root: str = "app",
            force: bool = False
        ) -> Self:
            if not BeanConfig._ConfigLoader._file_exists(path):
                if force: raise FileNotFoundError(f"File '{path}' not found.")
                return self

            import tomllib

            def flatten_toml(obj: dict, prefix: str = "") -> Dict[str, Any]:
                out: Dict[str, Any] = {}
                for k, v in obj.items():
                    full_key = f"{prefix}_{k}" if prefix else k
                    full_key = full_key.replace(".", "_").upper()
                    if isinstance(v, dict):
                        out.update(flatten_toml(v, full_key))
                    else:
                        out[full_key] = v
                return out

            with open(path, "rb") as f:
                data = tomllib.load(f).get(root, {})

            return self._from_source(path, flatten_toml(data))

        def from_ini(
            self,
            path: str,
            section: str = "app",
            force: bool = False
        ) -> Self:
            if not BeanConfig._ConfigLoader._file_exists(path):
                if force: raise FileNotFoundError(f"File '{path}' not found.")
                return self

            import configparser
            cfg = configparser.ConfigParser()
            cfg.read(path)

            def flatten_ini() -> Dict[str, Any]:
                out: Dict[str, Any] = {}

                for sec in cfg.sections():
                    if not sec.startswith(section):
                        continue
                    for key, value in cfg.items(sec):
                        full_key = f"{sec}.{key}"
                        norm_key = full_key.replace(".", "_").upper()
                        out[norm_key] = value

                prefix = f"{section}_".upper()
                return { k.removeprefix(prefix): v for k, v in out.items() }

            return self._from_source(path, flatten_ini())

# suggar
def ConfigField(
        type: type[_T],
        description: Optional[str] = None,
        default: Optional[_T] = None,
        validator: Optional[Predicate[_T]|Callable[[_T], bool]] = None,
        required: bool = False,
    ) -> _T:
    return BeanConfig._ConfigField(type, description, default,
                                   validator, required) # type: ignore

# cleanup
del _S, _E, _R, _T, _I, _O, _C, _P

# -----------------------------------------------------------------------------
# config validators
# -----------------------------------------------------------------------------

@predicate
def isPositive(n: int | float) -> bool:
    """Check that a number is negative `n > 0`."""
    return n > 0

@predicate
def isNegative(n: int | float) -> bool:
    """Check that a number is negative `n < 0`."""
    return n < 0

@predicate
def isPort(n: int) -> bool:
    """Check that a number is a valid TCP/UDP port."""
    return isinstance(n, int) and 0 < n <= 65535

@predicate
def nonEmpty(s: str) -> bool:
    """Check that a string is not empty / blank."""
    return bool(s.strip())

rEmail = re.compile(r'([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+')
@predicate
def isEmail(email: str) -> bool:
    """Check that a string is a simple valid email."""
    return bool(re.fullmatch(rEmail, email))

def isDate(fmt: str = "%Y-%m-%d") -> Predicate[str]:
    from datetime import datetime

    def check(date: str) -> bool:
        """Check that a string is a valid date in the given format."""
        try:
            datetime.strptime(date, fmt)
            return True
        except ValueError:
            return False

    return predicate(check)

@predicate
def isUrl(url: str) -> bool:
    """Check that a string is a valid URL."""
    from urllib.parse import urlparse
    result = urlparse(url)
    return all([result.scheme, result.netloc])

_rHostname = re.compile(
    r"^(?=.{1,253}$)(?!-)([A-Za-z0-9-]{1,63}\.)*[A-Za-z0-9-]{1,63}$"
)
@predicate
def isHost(hostname: str) -> bool:
    """Check that a string is a syntactically valid hostname."""
    if not hostname:
        return False
    if hostname.endswith("."):
        hostname = hostname[:-1]  # strip trailing dot
    return bool(_rHostname.fullmatch(hostname))

@predicate
def isIPv4(ip: str) -> bool:
    """Check that a string is a valid IPv4."""
    import ipaddress
    try:
        ipaddress.IPv4Address(ip)
        return True
    except Exception:
        return False

@predicate
def isIPv6(ip: str) -> bool:
    """Check that a string is a valid IPv6."""
    import ipaddress
    try:
        ipaddress.IPv6Address(ip)
        return True
    except Exception:
        return False

@predicate
def pathExists(path: str) -> bool:
    """Check that a string is a valid path to something."""
    return Path(path).exists()

@predicate
def fileExists(path: str) -> bool:
    """Check that a string is a valid path to a file."""
    p = Path(path)
    return p.exists() and p.is_file()

@predicate
def dirExists(path: str) -> bool:
    """Check that a string is a valid path to a dir."""
    p = Path(path)
    return p.exists() and p.is_dir()

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
