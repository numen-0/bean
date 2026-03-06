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

__version__ = "0.3.0"
__doc__     = "Tiny framework for bootstrapping apps"
__author__  = "numen-0"
__license__ = "MIT"

# -----------------------------------------------------------------------------
# api
# -----------------------------------------------------------------------------

__all__ = [
    "BeanApp",
    "Log", "Logger",
    "Result", "Success", "Predicate",
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
from abc import ABC, abstractmethod
from dataclasses import MISSING, dataclass, field, fields
from datetime import datetime
from enum import Enum
from functools import wraps
from pathlib import Path
from threading import Event, Thread
from time import sleep
from typing import (
    Any, Literal, Optional, Type,
    Callable, NoReturn,
    ClassVar, Protocol, Self,
    Dict, Iterable, List, Set, Tuple,
    TextIO,
    cast, dataclass_transform, get_args, get_origin, override
)

# -----------------------------------------------------------------------------
# app
# -----------------------------------------------------------------------------

class BeanApp(ABC):
    """ Abstract base class for a Bean application. """

    def __init__(self, name: str = "bean-app"):
        self.name: str = name

    def startup(self) -> Optional[bool]: ...

    def shutdown(self) -> Optional[bool]: ...

    @abstractmethod
    def run(self) -> int: ...

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
        """ Standard log levels with numeric severity and ANSI color codes. """

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
        level: bool = True,
        timestamp: bool = True,
        logger_name: bool = True,
    ) -> Callable[[Logger.Record], str]:

        def fmt(record: Logger.Record) -> str:
            parts = []

            if timestamp:
                parts.append(f"{record.timestamp:%Y-%m-%d %H:%M:%S}")

            if level:
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
        parent: Optional[Logger] = None,
    ):
        self._name: str = name
        self._full_name: str
        self._level: Logger.Level = level

        self._parent: Optional[Logger] = parent
        self._children: List[Logger] = []

        self.handlers: List[Logger.Handler] = list(handlers) if handlers else []

        Logger._active_loggers.add(self)

        if self._parent is None:
            self._full_name = self._name
        else:
            self._full_name = f"{self._parent._full_name}.{self._name}"

    # context manager support

    def __enter__(self):
        return self
    def __exit__(self, *_):
        self.close()

        for child in self._children:
            del child

        if self._parent and self in self._parent._children:
            self._parent._children.remove(self)
        Logger._active_loggers.discard(self)

        return False  # don't suppress exceptions

    def __del__(self):
        try: # attempt to flush before object is destroyed
            self.close()
            if self._parent and self in self._parent._children:
                self._parent._children.remove(self)
            Logger._active_loggers.discard(self)
        except Exception: pass

    # logger gen/set

    @property
    def name(self) -> str: return self._full_name

    @name.setter
    def name(self, name: str) -> None: self.set_name(name)

    @property
    def level(self) -> Level: return self._level

    @level.setter
    def level(self, value: Level): self.set_level(value)

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
            level=level if level is not None else self._level,
            handlers=handlers + self.handlers,
            parent=self,
        )

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
        if name is not None:     self.set_name(name)
        if level is not None:    self.set_level(level, cascade)
        if handlers is not None: self.add_handlers(handlers, cascade=cascade)
        return self

    def set_name(self, name: str) -> Self:
        self._name = name
        if self._parent is None:
            self._full_name = self._name
        else:
            self._full_name = f"{self._parent._full_name}.{self._name}"

        for child in self._children:
            child.set_name(child._name)
        return self

    def set_level(self, level: Level, cascade: bool = True) -> Self:
        self._level = level
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
        if level.value < self._level.value: return self

        record = Logger.Record(
            timestamp=datetime.now(),
            level=level,
            logger=self._full_name,
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
        """ Log an error message with the current exception traceback. """
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

        self.handlers.clear()  # detach handlers, don't close

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

@dataclass(frozen=True, slots=True)
class Result[S, E]:
    ok: bool = False
    value: Optional[S] = None
    error: Optional[E] = None

    # constructors

    @staticmethod
    def Ok(value: S) -> Result[S, E]:
        return Result(value=value, ok=True)

    @staticmethod
    def Error(error: E) -> Result[S, E]:
        return Result(error=error, ok=False)

    @staticmethod
    def from_tuple(t: Tuple[S, bool]) -> Result[S, S]:
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

    def to_success(self) -> Success[S|E]:
        return Success.from_tuple(self.to_tuple())
    def to_tuple(self) -> Tuple[S, Literal[True]]|Tuple[E, Literal[False]]:
        if self.ok: return (self.value, True) # type: ignore
        return (self.error, False) # type: ignore

    def unwrap(self) -> S:
        """ Return the value. Raise exception if has no `.value` """
        if not self.ok:
            raise RuntimeError(f"Unwrapped error Result: {self.error}")
        return self.value # type: ignore

    def unwrap_or(self, default: S) -> S:
        """ Return the value if ok, else return default. """
        if not self.ok: return default
        return self.value # type: ignore

    def unwrap_err(self) -> E:
        """ Return the error. Raise exception if has no `.value` """
        if self.ok: raise RuntimeError(f"Result has no error")
        return self.error # type: ignore

@dataclass(frozen=True, slots=True)
class Success[T]:
    value: T
    ok: bool

    # constructors

    @staticmethod
    def Ok(value: T) -> Success[T]:
        return Success(value=value, ok=True)

    @staticmethod
    def Fail() -> Success[None]:
        return Success(value=None, ok=False)

    @staticmethod
    def from_tuple(t: Tuple[T, bool]) -> Success[T]:
        """
        Convert a (value, ok) tuple into a Success.
        Ignores the value if ok==False
        """
        return Success(*t)

    @staticmethod
    def from_obj(obj: Tuple[T, bool]|Success[T]) -> Success[T]:
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

    def to_result(self) -> Result[T, T]:
        return Result.from_tuple(self.to_tuple())
    def to_tuple(self) -> Tuple[T, bool]:
        return (self.value, self.ok)

@dataclass(slots=True, frozen=True)
class Predicate[T]:
    fn: Callable[[T], bool]
    name: Optional[str] = None

    def __post_init__(self):
        if self.name is None:
            object.__setattr__(self, "name", self.fn.__name__)

    def __call__(self, value: T) -> bool:
        return bool(self.fn(value))

    def __and__(self, other: Predicate[T]) -> Predicate[T]:
        return Predicate(
            lambda v: self(v) and other(v),
            name=f"({self.name} & {other.name})"
        )

    def __or__(self, other: Predicate[T]) -> Predicate[T]:
        return Predicate(
            lambda v: self(v) or other(v),
            name=f"({self.name} | {other.name})"
        )

    def __invert__(self) -> Predicate[T]:
        return Predicate(
            lambda v: not self(v),
            name=f"!{self.name}"
        )

    def __repr__(self):
        return f"<Predicate {self.name}>"

    def trace(self, hook: Callable[[T, bool], Any]) -> Predicate[T]:
        return Predicate(lambda value: (
            res := self(value),
            hook(value, res),
        )[0], name=f"@{self.name}")

# -----------------------------------------------------------------------------
# pipes
# -----------------------------------------------------------------------------

type PipeResult[T] = Success[T] | Tuple[T, bool]

class Pipe[I, O]:
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
        fn: Optional[Callable[[I], PipeResult[O]]] = None
    ):
        if fn is None:
            def no_op(v): return Success.Ok(v)
            fn = no_op
        self._fn: Callable[[I], PipeResult[O]] = fn

    def __call__(self, value: I) -> Success[O]:
        return Success.from_obj(self._fn(value))

    def __or__[R](
        self,
        other: Callable[[O], PipeResult[R]]
    ) -> Pipe[I, R]:

        other = other if isinstance(other, Pipe) else Pipe(other)

        @wraps(other._fn)
        def join(value: I) -> Success[R]:
            res = self(value)
            return other(res.value) if res.ok else res  # type: ignore

        return Pipe(join)

    def retry[R](
        self,
        fn: Callable[[O], PipeResult[R]],
        attempts: int = 3,
        delay: float = 0.5,
    ) -> Pipe[I, R]:
        """ Wrap a pipe-compatible function with retry logic.

        The returned function will call `fn` up to `attempts` times, sleeping
        `delay` seconds between failures. Succeeds as soon as any attempt
        returns `ok == True`.

        Returns:
          - `(value,      True)`  if any try succeed
          - `(last_value, False)` if all functions fail
        """
        assert 0 < attempts

        def foo(value: O) -> Success[R]:
            res = None
            for _ in range(attempts):
                res = Success.from_obj(fn(value))
                if res.ok: return res
                if delay > 0: sleep(delay)

            assert res is not None
            return res

        return self | Pipe(foo)

    def map[R](
        self,
        fn: Callable[[O], R],
    ) -> Pipe[I, R]:
        """ Lift a pure function into a pipe-compatible function.

        Returns: `(fn(value), True)`
        """
        def foo(value: O): return Success.Ok(fn(value))
        return self | Pipe(foo)

    def guard[R](
        self,
        fn: Callable[[O], bool],
        err: Optional[R|Callable[[O], R]] = None
    ) -> Pipe[I, O]:
        """ Validate a value inside a pipe.

        Returns:
          - `(value, True)`         if   guard passes (`fn(value) == True`)
          - `(value, False)`        elif guard fails and err == None
          - `(err(value), False)`   elif guard fails and err is Callable
          - `(err, False)`          else
        """
        def foo(value: O) -> Success[O]:
            if fn(value): return Success.Ok(value)
            if err is not None:
                value = err(value) if callable(err) else err # type: ignore
            return Success(value, False)

        return self | Pipe(foo)

    def peek(
        self,
        fn: Callable[[O], Any],
    ) -> Pipe[I, O]:
        """ Execute a side-effect without modifying the value.

        Commonly used for logging, metrics, or debugging.

        Returns: `(value, True)`
        """
        def foo(value: O):
            fn(value)
            return Success.Ok(value)

        return self | Pipe(foo)

    def fallback[R, E](
        self,
        fn: Callable[[O], PipeResult[R]],
        fb: E
    ) -> Pipe[I, R|E]:
        """ Convert a failing function into a successful one with a fallback.

        Useful for optional steps that should not stop the pipe.

        Returns:
          - `(value,    True)` if fn passes
          - `(fb_value, True)` if fn fails
        """
        def foo(value: O):
            res = Success.from_obj(fn(value))
            return Success.Ok(res.value if res.ok else fb)

        return self | Pipe(foo)

    def branch[R, E](
        self,
        cond_fn: Callable[[O], bool],
        success_fn: Optional[Callable[[O], PipeResult[R]]] = None,
        fail_fn: Optional[Callable[[O], PipeResult[E]]] = None,
    ) -> Pipe[I, O|R|E]:
        """ Conditional branching inside a pipe.

        If the selected branch function is None, the value is passed through
        and the condition result is used as the ok flag.

        Returns:
          - `(value,   cond)` if no matching fn is set
          - `(s_value, bool)` if `cond_fn` passes and `success_fn` is provided
          - `(f_value, bool)` if `cond_fn` fails and `fail_fn` is provided
        """
        def foo(value: O) -> Success[O|R|E]:
            cond = cond_fn(value)
            if cond and success_fn is not None:
                return Success.from_obj(success_fn(value))
            elif not cond and fail_fn is not None:
                return Success.from_obj(fail_fn(value))
            return Success(value, cond)

        return self | Pipe(foo)

    def trigger(
        self,
        fn: Callable[[O], bool],
        ex: type[Exception] = Exception,
        msg: Optional[str] = None
    ) -> Pipe[I, O]:
        """ Validate a value inside a pipe.

        Returns:
          - raise `ex(msg)` if trigger passes  (`fn(value) == True`)
          - `(value, True)` if trigger fails
        """
        def foo(value: O) -> Success[O]:
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

def install_signal_handlers(
    cb: Optional[Callable[[int, Any], Any]] = None
):
    """ Install SIGINT / SIGTERM handlers.

    Signals set a shutdown flag that can be checked by the app.
    """
    if getattr(install_signal_handlers, "_installed", False): return
    setattr(install_signal_handlers, "_installed", True)

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

FieldValue = bool|int|float|str|Enum|list
type ConfigSource = Literal[
    "args", "dict", "env", "ini", "json", "py", "toml",
] | str

class BeanConfig(ABC):
    """ Abstract base class for declarative application configuration.

    `BeanConfig` provides a structured, type-safe configuration system based
      on declarative field specifications.

    Subclasses define configuration schema via `_Field` declarations. The
    framework handles:

        - Loading values from one or multiple sources (`json`, `env`, ...)
        - Type enforcement
        - Default resolution
        - Field-level normalization
        - Field-level validation
        - Class-level validation (with inheritance support)
        - CLI flag integration (if enabled)
    """

    @dataclass
    class _Field[F: FieldValue]:
        """ Declarative specification of a single configuration field.

        It is purely declarative and contains no runtime state.
        Runtime values are stored separately by the config loader.

        Lifecycle of a field value:

            1. Raw value is provided (cli, env, ...).
            2. Value is casted (if necessary).
            3. Default is applied (if necessary).
            4. Type is validated.
            5. `normalizer` is applied (if defined).
            6. `validator` is executed (if defined).
            7. Class-level validators run afterwards (if defined).

        @type:          Values must fall under type after config build
        @required:      If True, the field must be explicitly provided (cannot
                        rely on default)
        @default:       Optional default value applied when no value is provided
        @description:   Optional human-readable description
        @normalizer:    Optional transformation function applied before
                        validation. Intended for canonicalization (e.g. strip
                        strings, expand paths, ...)
        @validator:     Optional field-level validation function
        @long_flag:     Optional CLI long flag (e.g. "--port")
        @short_flag:    Optional CLI short flag (e.g. "-p")
        @shadow:        Optional set of sources the field can not be set
        """

        type: type[F]
        description: Optional[str] = None
        required: bool = False
        default: Optional[F] = None
        validator: Optional[Callable[[F], bool]] = None
        normalizer: Optional[Callable[[F], F]] = None
        long_flag: Optional[str] = None
        short_flag: Optional[str] = None
        shadow: set[str] = field(default_factory=set)

        def __post_init__(self):
            if isinstance(self.shadow, Iterable):
                self.shadow = set(self.shadow)

        def __set_name__(self, owner, name):
            self._name = name
            if self.default is None and not self.required:
                raise TypeError(
                    f"Unbound ConfigField, '{owner.__name__}.{self._name}' has no default and is not required"
                )

        def __get__(self, instance, owner) -> F:
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

        def cast_val(self, val: FieldValue) -> Optional[FieldValue]:
            exp_t = self.type
            origin = get_origin(exp_t)
            check_t = origin or exp_t

            if isinstance(val, check_t): return val

            enum_map: Dict[str, Enum] = {}
            def _cast_scalar(t: type, v):
                if isinstance(v, t): return v
                if t is str:         return str(v)
                if t is int:         return int(v)
                if t is float:       return float(v)

                if t is bool:
                    if not isinstance(v, str): return bool(v)

                    v = v.lower()
                    if v in ("1", "true", "yes", "on"):  return True
                    if v in ("0", "false", "no", "off"): return False
                    return None

                # Note: If an Enum name doesn't match, instead of raising a
                #       clean `ValueError`, it will cause a `TypeError` because
                #       on failure the raw value is returned. After all sources
                #       are loaded, it is then validated against the field type.
                if issubclass(t, Enum) and isinstance(v, str):
                    nonlocal enum_map # simple cache
                    if not enum_map: enum_map = {m.name.lower(): m for m in t}
                    return enum_map.get(v.lower(), None)

                return None

            try:
                if check_t is not list:
                    return _cast_scalar(exp_t,  val)

                args_t = get_args(exp_t)
                elem_t = args_t[0] if args_t else str

                if isinstance(val, list):
                    items = val
                elif isinstance(val, str):
                    items = [x.strip() for x in val.split(",")]
                else:
                    items = [val] # type error?

                out = []
                for x in items:
                    v = _cast_scalar(elem_t, x)
                    if v is None:
                        return None
                    out.append(v)
                return out

            except Exception: pass
            return None

    _instance: ClassVar[Optional[BeanConfig]] = None
    _spec: ClassVar[Dict[str, _Field]] = {}
    # allow override `{ FIELD_NAME: { FN_NAME: fn, ... }, ... }`
    _global_validators: Dict[str, Dict[str, Callable[[FieldValue], bool]]] = {}

    @staticmethod
    @dataclass_transform()
    def dataclass(bcls: Type[Any]) -> Type[Any]:
        bcls = dataclass(bcls)
        dc_cls = cast(type, bcls)

        allowed = get_args(FieldValue)
        def is_valid_field_type(t: type|type[Any]|Any) -> bool:
            check_t = get_origin(t) or t

            if check_t is list:
                if not (sub := get_args(t)): return True
                t = sub[0]

            return t in allowed or (isinstance(t, type) and issubclass(t, Enum))

        config_fields = {}
        for f in fields(dc_cls):
            t = f.type

            if not is_valid_field_type(t):
                raise TypeError(f"Field '{f.name}' has unsupported type: {t}")

            if f.default is not MISSING:
                required = False
                default  = f.default
            elif f.default_factory is not MISSING:
                required = False
                default  = f.default_factory()
            else:
                required = True
                default = None

            config_fields[f.name] = BeanConfig._Field(
                t, # type: ignore
                required=required,
                default=default,
            )

        bcls._spec.update(config_fields)
        return bcls

    def __init_subclass__(cls, **kwargs):
        """ Build the immutable schema `_spec` at class definition time.

        - Scans the entire MRO to include inherited ConfigFields
        - Stores them on the class to avoid repeated introspection
        - Makes config schema "static"
        """
        super().__init_subclass__(**kwargs)

        fields: Dict[str, BeanConfig._Field] = {}

        # normal ConfigField discovery
        for base in reversed(cls.__mro__):      # python magic so we also set
            for k, v in base.__dict__.items():  # inherited fields...
                if isinstance(v, BeanConfig._Field):
                    fields[k] = v

        cls._spec = fields

    @classmethod
    def validate[F: FieldValue](
        cls,
        *keys: str
    ) -> Callable[[Callable[[F], bool]], Callable[[F], bool]]:
        """ Decorator to make custom validators on function definition """

        def decorator(fn: Callable[[F], bool]) -> Callable[[F], bool]:
            orig_fn = getattr(fn, "__func__", fn) # unwrap staticmethod
            name = str(orig_fn.__name__)

            # allow override `cls.validators[FIELD_NAME][FN]`
            for key in keys:
                cls._global_validators.setdefault(
                    key,
                    {}
                )[name] = orig_fn # type: ignore
            return fn

        return decorator

    @classmethod
    def _set_instance(cls, obj: BeanConfig):
        cls._instance = obj

    @classmethod
    def spec(cls) -> Dict[str, _Field]:
        return cls._spec

    @classmethod
    def spec_for(cls, source: ConfigSource) -> Dict[str, _Field]:
        return {
            k: f
            for k, f in cls._spec.items()
            if source not in f.shadow
        }

    @classmethod
    def load[C: BeanConfig](
        cls: type[C],
        strict: bool = False,
        logger: Optional[Logger] = None,
    ) -> BeanConfig._ConfigLoader[C]:
        return BeanConfig._ConfigLoader(cls, strict, logger)

    @classmethod
    def print_config(
        cls,
        show_types: bool = False,
        show_defaults: bool = False,
        show_required: bool = False,
    ) -> None:
        """ Pretty-print the current configuration. """

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

    class _ConfigLoader[C: BeanConfig]:
        """ App config loader/builder

        Receives a schema, and loads sources to fill the config, validating it
        and generating a final config as a product
        """

        def __init__(
            self,
            config_cls: type[C],
            strict: bool = False,
            log: Optional[Logger] = None,
        ):
            self.config_cls: type[C] = config_cls
            self._values: Dict[str, Any] = {}
            self._locked = False
            self.strict = strict

            if log is not None: self.log = log
            else: self.log = Logger("dummy", handlers=[Logger.VoidHandler()])

        # post config steps

        def validate(self) -> BeanConfig._ConfigLoader[C]:
            """ Validate current config """

            errors = []
            spec = self.config_cls.spec()

            # field-level

            for k, f in spec.items():
                val = self._values.get(k, None)

                # default / required
                if val is None:
                    if f.required:
                        errors.append(ValueError(
                            f"Missing required config: {k}"))
                        continue

                    assert f.default is not None # already asserted at init
                    val = f.default

                # type check
                if not isinstance(val, get_origin(f.type) or f.type):
                    errors.append(TypeError(
                        f"Invalid type '{type(val).__name__}' for value {k} ({val})"))
                    continue

                # normalize
                if f.normalizer is not None: val = f.normalizer(val)


                # field-level validator
                if f.validator and not f.validator(val):
                    errors.append(ValueError(f"Invalid value for {k}: {val}"))
                    continue

                self._values[k] = val

            if errors: raise ExceptionGroup("Config validation errors", errors)

            # class-level per-field validators

            def collect_validators(
            ) -> Dict[str, Dict[str, Callable[[FieldValue], bool]]]:
                """ Dynamically collect validators via MRO. """
                validators = {}

                for base in reversed(self.config_cls.__mro__):
                    base_v = getattr(base, "_global_validators", None)
                    if not base_v: continue

                    for key, d in base_v.items():
                        validators.setdefault(key, {}).update(d)

                return validators

            for field, d in collect_validators().items():
                if field not in spec:
                    fmt = f"Class-level validator '%s' key '{field}' is not in spec"
                    errors.extend(
                        KeyError(fmt % name)
                        for name in d.keys()
                    )
                    continue

                val = self._values.get(field, None)
                assert val is not None

                fmt = f"Class-level validation failed for {field}: %s({val})"

                errors.extend(
                    ValueError(fmt % name)
                    for name, method in d.items()
                    if not method(val)
                )

            if errors: raise ExceptionGroup("Config validation errors", errors)

            return self

        def build(self) -> C:
            """ Validate and build the config """
            obj = object.__new__(self.validate().config_cls)

            # Note: we override the instance attr not the class attributes
            for k, v in self._values.items():
                setattr(obj, k, v)
                setattr(self.config_cls, k, v)

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

        # source loaders

        def _from_source(
            self,
            source: ConfigSource,
            data: Dict[str, Any],
            key_mapper: Callable[[str], str] = lambda k: k
        ) -> Self:
            """ Generic loader: cast and store values from any source.

            - data: dict of raw key -> value
            - source: string for logging purposes
            - key_mapper: function to normalize source keys to config keys
            """

            errors = []

            spec = self.config_cls.spec_for(source)
            for raw_key, raw_val in data.items():
                key = key_mapper(raw_key)

                if key not in spec:
                    msg = f"unknown key '{raw_key}' in config from '{source}'"
                    if self.strict: errors.append(KeyError(msg))
                    else: self.log.warning(msg)
                    continue

                # Note: auto-skip unset values (`None`). Useful for `from_args`
                #       so missing flags don't raise errors.
                #
                #       For other sources, this might silently mask config
                #       entries. `( '-')`
                if raw_val is None:
                    continue

                val = spec[key].cast_val(raw_val)
                if val is None:
                    errors.append(ValueError(
                        f"Invalid value for {raw_key} ({raw_val}) from '{source}'"))
                    continue

                self._values[key] = val

            if errors:
                raise ExceptionGroup("Config load errors", errors)

            return self

        def from_dict(self, d: Dict) -> Self:
            return self._from_source("dict", d)

        def from_json(
            self,
            path: str,
            force: bool = False
        ) -> Self:

            if not fileExists(path):
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
                return f"{key.replace('_', '-').lower()}"

            parser = argparse.ArgumentParser()
            spec = self.config_cls.spec_for("args")
            for k, f in spec.items():
                flags = []

                # Note: we don't mangle user flags
                if f.long_flag is not None:  flags.append(f.long_flag)
                else:                        flags.append(f"--{to_cli(k)}")
                if f.short_flag is not None: flags.append(f.short_flag)

                kwargs: Dict[str, Any] = dict(
                    dest=k,             # store as spec key
                    default=None,       # if set it will mask values
                    help=f.description,
                )

                origin = get_origin(f.type) # extract `list` from `list[T]`

                if f.type is bool:
                    if f.default is not None:
                        if f.default:   kwargs["action"] = "store_false"
                        else:           kwargs["action"] = "store_true"

                elif f.type is list or origin is list:
                    elem_type = str
                    args_t = get_args(f.type) # extract `T` from `list[T]`

                    if args_t:
                        elem_type = args_t[0]

                        if issubclass(elem_type, Enum):
                            kwargs["choices"] = [m.name for m in elem_type]
                            elem_type = str

                    kwargs["type"] = elem_type
                    kwargs["nargs"] = "*"

                elif issubclass(f.type, Enum):
                    kwargs["type"] = str
                    kwargs["choices"] = [m.name for m in f.type]

                else:
                    kwargs["type"] = f.type

                parser.add_argument(*flags, **kwargs)

            parsed = parser.parse_args(args)
            return self._from_source("args", vars(parsed))

        def from_env(self, prefix: str = "") -> Self:
            import os

            def normalize_keys(items):
                return { k.removeprefix(prefix).upper(): v for k, v in items }

            spec = self.config_cls.spec_for("env")
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
            if not fileExists(path):
                if force: raise FileNotFoundError(f"File '{path}' not found.")
                return self

            import importlib.util as IU
            spec = IU.spec_from_file_location("user_config", path)
            if spec is None or spec.loader is None:
                raise FileNotFoundError(f"Cannot load config file: {path}")

            mod = IU.module_from_spec(spec)
            spec.loader.exec_module(mod)
            obj = getattr(mod, symbol, None)

            if obj is None:
                raise ValueError(f"Python config must expose `{symbol}` object")

            if isinstance(obj, dict):
                return self._from_source(path, obj)

            data = {
                k: getattr(obj, k)
                for k, _ in self.config_cls.spec_for("py").items()
                if not k.startswith("_") and hasattr(obj, k)
            }

            return self._from_source(path, data)

        def from_toml(
            self,
            path: str,
            root: str = "app",
            force: bool = False
        ) -> Self:
            if not fileExists(path):
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
            if not fileExists(path):
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

# sugar

def ConfigField[F: FieldValue](
        type: type[F],
        *,
        description: Optional[str] = None,
        required: Optional[bool] = None,
        default: Optional[F] = None,
        normalizer: Optional[Callable[[F], F]] = None,
        validator: Optional[Predicate[F]|Callable[[F], bool]] = None,
        long_flag: Optional[str] = None,
        short_flag: Optional[str] = None,
        shadow: Optional[Iterable[str]] = None,
    ) -> F:
    """ Sugar to declare a BeanConfig fields.

    This wraps `BeanConfig._ConfigField` and auto-infers `required` if not
    provided:
        - If `default` is None and `required` is not explicitly set, the field
          becomes required.
        - Otherwise, `required` is set to `False`.
    """
    if required is None: required = default is None
    return BeanConfig._Field(
        type,
        description=description,
        required=required,
        default=default,
        normalizer=normalizer,
        validator=validator,
        long_flag=long_flag,
        short_flag=short_flag,
        shadow=set(shadow) if shadow is not None else set(),
    ) # type: ignore

# -----------------------------------------------------------------------------
# config validators
# -----------------------------------------------------------------------------

@Predicate
def isPositive(n: int | float) -> bool:
    """ Check that a number is negative `n > 0`. """
    return n > 0

@Predicate
def isNegative(n: int | float) -> bool:
    """ Check that a number is negative `n < 0`. """
    return n < 0

@Predicate
def isPort(n: int) -> bool:
    """ Check that a number is a valid TCP/UDP port. """
    return isinstance(n, int) and 0 < n <= 65535

@Predicate
def nonEmpty(s: str) -> bool:
    """ Check that a string is not empty / blank. """
    return bool(s.strip())

rEmail = re.compile(r'([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+')
@Predicate
def isEmail(email: str) -> bool:
    """ Check that a string is a simple valid email. """
    return bool(re.fullmatch(rEmail, email))

def isDate(fmt: str = "%Y-%m-%d") -> Predicate[str]:
    """ Check that a string is a valid date in the given format. """

    def check(date: str) -> bool:
        try:
            datetime.strptime(date, fmt)
            return True
        except ValueError:
            return False

    return Predicate(check, name=f"isDate[{fmt}]")

@Predicate
def isUrl(url: str) -> bool:
    """ Check that a string is a valid URL. """
    from urllib.parse import urlparse
    result = urlparse(url)
    return all([result.scheme, result.netloc])

_rHostname = re.compile(
    r"^(?=.{1,253}$)(?!-)([A-Za-z0-9-]{1,63}\.)*[A-Za-z0-9-]{1,63}$"
)
@Predicate
def isHost(hostname: str) -> bool:
    """ Check that a string is a syntactically valid hostname. """
    if not hostname:
        return False
    if hostname.endswith("."):
        hostname = hostname[:-1]  # strip trailing dot
    return bool(_rHostname.fullmatch(hostname))

@Predicate
def isIPv4(ip: str) -> bool:
    """ Check that a string is a valid IPv4. """
    import ipaddress
    try:
        ipaddress.IPv4Address(ip)
        return True
    except Exception:
        return False

@Predicate
def isIPv6(ip: str) -> bool:
    """ Check that a string is a valid IPv6. """
    import ipaddress
    try:
        ipaddress.IPv6Address(ip)
        return True
    except Exception:
        return False

@Predicate
def pathExists(path: str|Path) -> bool:
    """ Check that a string is a valid path to something. """
    return Path(path).exists()

@Predicate
def fileExists(path: str|Path) -> bool:
    """ Check that a string is a valid path to a file. """
    p = Path(path)
    return p.exists() and p.is_file()

@Predicate
def dirExists(path: str|Path) -> bool:
    """ Check that a string is a valid path to a dir. """
    p = Path(path)
    return p.exists() and p.is_dir()

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
