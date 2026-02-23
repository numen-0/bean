# bean.core

`bean.core` is a tiny Python toolkit to bootstrap small apps.

It gives you **ready-to-use building blocks** so you can focus on the
**business logic**, instead of boilerplate.

---

## Overview

With `bean.core` you get:

- **App lifecycle**: start, run, shutdown
- **Config**: load from multiple sources (`env`, `.py`, ...) with validation.
- **Logging**.
- **Pipes**: composable data flows.
- **Shell commands**: run commands directly in your flows.
- **Scheduler**: jobs & tasks.

> Just enough to cook some beans.

## Quick Examples

This is a quick reference for the main `API`.

For full details, peek at the [source code](/bean-core/src/bean/core.py) `:)`.

### Simple Loop App

In this example:

- define a custom app by overriding the `run` method and optionally `startup`
  and `shutdown`.
- set up logging.
- script a infinite loop on run.
- run `bean.main` with our `App()`

So, the app will loop until a `SIGINT` or `SIGTERM` signal is received. On
shutdown, the `shutdown` method is called, but you can force termination by
sending another signal.

```py
from time import sleep
from bean.core import BeanApp, Log, Logger, main, shutdown_requested

class App(BeanApp):
    def startup(self):
        Log.init(
            level=Logger.Level.from_debug(self.DEBUG),
            handlers=[Logger.TermHandler(Logger.fmt(color=True))]
        )
        Log.debug("starting...")

    def shutdown(self):
        Log.debug("ending...")

    def run(self):
        while not shutdown_requested():
            Log.info(":D doing nothing")
            sleep(2)

        Log.warning("xD shutdown requested, exiting run loop")
        return 0

if __name__ == "__main__":
    main(App("bean-app", debug=True))
```

### Config

BeanConfig can be populated from multiple sources, with a defined priority.
Available sources:

- `.from_args()`: parse command-line arguments with `argparse`
- `.from_dict()`: load configuration from a Python dictionary
- `.from_env()`: load environment variables matching `prefix + filedname`
- `.from_ini()`: Load from a `INI` file
- `.from_json()`: load from a `JSON` file
- `.from_py()`: load from a Python file exposing a `Config` object by default
- `.from_toml()`: Load from a `TOML` file
> Note: File loaders will skip missing files unless `force=True` is passed

```py
from bean.core import BeanConfig, ConfigField, isHost, isPort, isEmail

class MyConfig(BeanConfig):
    NAME: str = ConfigField(str)
    PORT: int = ConfigField(int, default=8080, validator=isPort)
    HOST: str = ConfigField(str, default="localhost", validator=isHost)
    EMAIL: str = ConfigField(str, validator=isEmail)

    @BeanConfig.validate("NAME")
    def check_empty_name(name: str):
        return len(name) > 0


( MyConfig.load()            # load priority:
    .from_env("APP_")        # 1. environment variables
    .from_py("./config.py")  # 2. Python file (ignored if not found)
    .from_args()             # 3. command-line arguments (auto --help)
    .build() )

MyConfig.print_config()
```
> Note: If your type checker complains about check_empty_name, add
>       `@staticmethod` or `# type: ignore`.


### Pipes

`Pipe` is a small, composable transformation pipeline.

Each stage:
- Receives a value
- Returns either:
    - `Success(value)` or `tuple(value, True)`
    - `tuple(value, ok)`
- If `ok == False`, the pipeline short-circuits

This makes it easy to build safe, expressive and composable data flows.

```py
from bean.core import Pipe

result = (
    Pipe()
        .guard(lambda x: x != 0)
        .map(lambda x: 10 / x)
)(5)

print(result)       # Success(2.0, ok=True)
```
> Note: pipes can be typed, e.g: `Pipe[float, float]()`

If the guard fails:

```py
result = (
    Pipe()
        .guard(lambda x: x != 0)
        .map(lambda x: 10 / x)
)(0) 

print(result)       # Success(value=0, ok=False)
```

The pipe short-circuits and the division step is never executed.

Available `Pipe` helpers:

- `.map(fn)`: transform value
- `.guard(fn)`: validate value (may short-circuit)
- `.peek(fn)`: side-effect without modifying value
- `.retry(fn, attempts, delay)`: retry a stage
- `.fallback(fn, fallback_value)`: recover from failure
- `.branch(cond, success_fn, fail_fn)`: conditional logic
- `.trigger(fn, ex, msg)`: raise exception if condition matches

> Pipes are fully composable using the `|` operator.

#### Shell Commands

Shell commands integrate directly into pipes.

`sh(cmd)` returns a `Pipe` that executes a shell command.

```py
from bean.core import sh, cat, tee, stdout

res = ()
print(
    (sh("echo 'hello bean'") 
        | sh("grep -F 'hello'")
        | tee("copy.txt")
        | stdout()
     )(None)
)
```

Available shell helpers:

- `sh(cmd)`: run command
- `cat(*paths)`: read files
- `tee(*paths)`: write to files
- `stdout()`: extract stdout
- `stderr()`: extract stderr

### Scheduler

Scheduler provides a minimal threaded task runner for delayed and periodic
execution.

It supports:
- One-shot tasks
- Delayed tasks
- Periodic jobs
- Limited or infinite runs
- Graceful shutdown

Basic example:

```py
from bean.core import Scheduler

( Scheduler()
    .task(lambda: print("Bean task once"))
).start() 
```

Periodic Job:

```py
from bean.core import Scheduler

schr = (
    Scheduler()
        .job(
            fn=lambda: print("Bean job running!"),
            interval=2.0,
            runs=5
        )
).start()

# ...

schr.join()  # wait until finite jobs complete
```


Mixed example:

```py
import time
from bean.core import Scheduler

with Scheduler()
    .task(lambda: (print("Init task"), time.sleep(5), print("Done")))
    .job(lambda: print("Heartbeat"), interval=1.0)
) as schr:
    ...
```
> Note: When exiting the `with` block, the scheduler automatically calls
> `schr.stop().join()` to gracefully stop infinite jobs.

API Overview:

- `.task(fn, runs=1, delay=0)` -> Schedule a task that runs:
    - Runs `runs` times (default: once) or indefinitely if `runs=None`
    - Starts after an optional `delay`

- `.job(fn, interval, runs=None)` -> Schedule a periodic job:
    - Runs every `interval` seconds
    - Runs indefinitely by default (`runs=None`)

Lifecycle Methods:

- `.start()`: start all tasks
- `.join(timeout=None)`: wait for completion (non-infinite only)
- `.stop()`: signal stop
- `.clear()`: remove all tasks

> All methods return `Self` for chaining.

Behavior Notes:

- Each task runs in its own **daemon thread**
- Finite tasks (`runs != None`) blocks `.join()` until completion or timeout
- Infinite tasks (`runs=None`) are ignored by `.join()`
- Tasks must exit voluntarily, blocking calls may delay shutdown
- Because threads are daemon, all tasks terminate when the main program exits

## License

All the repo falls under the [MIT License](/LICENSE).

