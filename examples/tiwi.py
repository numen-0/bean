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

- This is a joke script.
- Built to test and experiment with
  [`bean.core`](https://github.com/numen-0/bean).
- TIWI exists to let you defend your code with numbers nobody asked for.
"""

import keyword, math, re, sys
from collections import Counter
from pathlib import Path

from bean.core import (
    BeanApp, Log, Logger, Pipe, fileExists, main, ConfigField, BeanConfig,
)

class Config(BeanConfig):
    DEBUG = ConfigField(bool, default=False)
    IDENT = ConfigField(int, default=4)

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
            Log.info("----------------------------------------")

        if count > 0:
            Log.info(f"Total Score           : {total_score / count:.2f}")
        Log.info("========================================")

        return 0

# -------------------------

def indent_depth(line: str) -> int:
    spaces = len(line) - len(line.lstrip(" "))
    return spaces // Config.IDENT

def symbol_name_entropy(names: list[str]) -> float:
    symbols = [n for n in names if n not in keyword.kwlist]

    if not symbols: return 0.0

    freq = Counter(symbols)
    total = sum(freq.values())

    entropy = 0.0
    for count in freq.values():
        p = count / total
        entropy -= p * math.log2(p)

    return entropy

def analyze(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    lcode = [l for line in lines if (l := line.split("#")[0]).strip()]
    lcomments = [l := line[:line.index("#")] for line in lines if "#" in line]
    tcode = "\n".join(lcode)
    tccomments = "\n".join(lcomments)

    total = len(lines)
    blank = sum(1 for l in lines if not l.strip())
    comments = len(lcomments)
    max_len = max((len(l) for l in lines), default=0)

    # metrics

    #* Colon Density (CD)
    colon_count = tcode.count(":")
    colon_density = colon_count / total if total else 0

    #* Parenthesis Aggression Ratio (PAR)
    paren_count = sum(tcode.count(c) for c in "()<>{}[]")
    par_ratio = paren_count / total if total else 0

    #* Import Flexing Score (IFS)
    imports = sum(
        # Note: not exact, but try to count "import x, y, c, ..."
        len(l.split(",")) if l.strip().startswith("import ") else 1
        for l in lcode if l.strip().startswith(("import ", "from "))
    )
    functions = sum(1 for l in lcode if l.strip().startswith("def "))
    classes = sum(1 for l in lcode if l.strip().startswith("class "))
    import_flex = imports / ((functions + classes) or 1)

    #* Indentation Depth Peak (IDP)
    indent_depth_peak = max((
        indent_depth(l)
        for l in lcode if l.strip()),
        default=0
    )

    #* Symbol Name Entropy (SNE)
    symbols = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", tcode)
    sne = symbol_name_entropy(symbols)

    #* Magic Number Index (MNI)
    numeric_literals = len(re.findall(r"\b\d+\b", tcode))
    magic_number_index = numeric_literals / ((functions + classes) or 1)

    #* Despair Metric (DM)
    despair = sum(
        tccomments.count(tag) for tag in ("TODO", "FIXME", "HACK")
    )

    # score

    score = 100.0

    #* file too big penalty
    if total > 500:             score *= 0.93
    elif total > 300:           score *= 0.90

    #* long line penalty
    if max_len > 120:           score *= 0.92
    elif max_len > 100:         score *= 0.95
    elif max_len > 80:          score *= 0.97

    #* comments penalty
    if comments:                score *= 0.90

    #* blank lines penalty
    blank_ratio = blank / total if total else 0
    if blank_ratio > 0.4:       score *= 0.85
    elif blank_ratio > 0.2:     score *= 0.90
    elif blank_ratio > 0.1:     score *= 0.95

    #* Colon Density (CD)
    if colon_density > 0.5:     score *= 0.90
    elif colon_density < 0.1:   score *= 0.95

    #* Parenthesis Aggression Ratio (PAR)
    if par_ratio > 1.5:         score *= 0.93

    #* Import Flexing Score (IFS)
    if import_flex < 0.2:       score *= 0.95
    elif import_flex > 3:       score *= 0.90

    #* Indentation Depth Peak (IDP)
    if indent_depth_peak > 6:   score *= 0.93
    elif indent_depth_peak < 2: score *= 0.98

    #* Symbol Name Entropy (SNE)
    if sne > 8:                 score *= 0.97
    elif sne < 2:               score *= 0.98

    #* Magic Number Index (MNI)
    if magic_number_index > 10: score *= 0.95

    #* Despair Metric (DM)
    if despair > 5:             score *= 0.9

    score = max(0, min(score, 100))

    return {
        "score": round(score, 3),

        "lines": total,
        "code": len(lcode),
        "comments": comments,
        "blank-lines": blank,
        "max-line-length": max_len,

        "colon-density": round(colon_density, 3),
        "par-ratio": round(par_ratio, 3),
        "import-flex-score": round(import_flex, 3),
        "indent-depth-peak": indent_depth_peak,
        "variable-name-entropy": round(sne, 3),
        "magic-number-index": round(magic_number_index, 3),
        "despair-metric": despair,
    }


if __name__ == "__main__":
    main(App("tiwi"))
