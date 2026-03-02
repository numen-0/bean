"""
# TIWI - The Intended Way Index

`TIWI` is a small, opinionated code quality signal generator.

- It does not enforce correctness. It's mostly nonsense.
- It does not judge. It spits numbers.
- It does not say anything meaningful. It just gives you a score to defend your
  already made-up ideas.

## Usage

```sh
python tiwi.py file1.py file2.py ...
```

## Notes

- This is a joke project.
- Built to test and experiment with
  [`bean.core`](https://github.com/numen-0/bean).
"""

import sys
from pathlib import Path

from bean.core import (
    BeanApp, Log, Logger, Pipe, fileExists, main, ConfigField, BeanConfig, 
)

class Config(BeanConfig):
    DEBUG = ConfigField(bool, default=False)

class App(BeanApp):
    def startup(self):
        Config.load().build()

        Log.init(
            self.name,
            level=Logger.Level.from_debug(Config.DEBUG),
            handlers=[ Logger.TermHandler(Logger.fmt(
                color=True, level=False, timestamp=False, logger_name=False,
            )) ],
        )
        Log.debug("starting...")

    def shutdown(self):
        Log.debug("ending...")

    def run(self):
        if len(sys.argv) < 2:
            Log.error("No path(s) provided.")
            return 1

        pipe = ( Pipe[str, str]()
            .guard(fileExists)
            .map(Path)
                .peek(lambda path: Log.info(f"[{path}]"))
            .map(analyze)
            .peek(lambda report: [
                Log.info(f"{k:<22}: {v}")
                for k, v in report.items()
            ])
            .map(lambda report: report["score"])
        )

        total_score = 0
        count = 0

        Log.info("========================================")
        for path in sys.argv[1:]:
            score, ok = pipe(path).to_tuple()

            if ok:
                total_score += score
                count += 1
            else:
                Log.error(f"Path not found: {path}")

        if count > 0:
            Log.info("----------------------------------------")
            Log.info(f"Total Score           : {total_score / count:.2f}")
        Log.info("========================================")

        return 0

# -------------------------

def analyze(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    total = len(lines)
    blank = sum(1 for l in lines if not l.strip())
    comments = sum(1 for l in lines if l.strip().startswith("#"))
    max_len = max((len(l) for l in lines), default=0)

    score = 100

    # file too big penalty
    if total > 500:
        score -= 20
    elif total > 300:
        score -= 10

    # long line penalty
    if max_len > 120:
        score -= 15
    elif max_len > 88:
        score -= 5

    # no comments penalty
    if comments == 0:
        score -= 10

    # too many blank lines penalty
    if total > 0 and blank / total > 0.4:
        score -= 5


    return {
        "score": max(0, score),
        "lines": total,
        "blank_lines": blank,
        "comments": comments,
        "max_line_length": max_len,
    }


if __name__ == "__main__":
    main(App("tiwi"))
