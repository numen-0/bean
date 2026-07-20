import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Tuple, Literal
from textwrap import dedent


ChangeType = Literal["major", "minor", "patch", "none"]


# version handling

def extract_version(path: Path) -> str:
    tree = ast.parse(path.read_text())

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue

        for target in node.targets:
            if (isinstance(target, ast.Name)
                and target.id == "__version__"
                and isinstance(node.value, ast.Constant)
            ):
                return node.value.value # type: ignore

    raise RuntimeError("No __version__ found")

def parse_version(version: str) -> Tuple[int, int, int]:
    return tuple(map(int, version.split("."))) # type: ignore

def version_bump_type(old_v: str, new_v: str) -> ChangeType:
    old = parse_version(old_v)
    new = parse_version(new_v)

    if new[0] > old[0]:
        return "major"
    if new[1] > old[1]:
        return "minor"
    if new[2] > old[2]:
        return "patch"
    return "none"

def bump_version(version, change):
    if change == "none":
        return version

    major, minor, patch = map(int, version.split("."))
    if change == "patch":
        major, minor, patch = major, minor, patch + 1
    else:
        if change == "minor":
            major, minor, patch = major, minor + 1, 0
        else:
            major, minor, patch = major + 1, 0, 0

    return f"{major}.{minor}.{patch}"

def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

# api extraction

def extract_api(path: Path) -> Dict[str, Dict]:
    tree = ast.parse(path.read_text())
    api: Dict[str, Dict] = {}

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            methods = {}
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and not item.name.startswith("_"):
                    methods[item.name] = len(item.args.args)

            api[node.name] = {
                "type": "class",
                "methods": methods,
            }

        elif isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            api[node.name] = {
                "type": "function",
                "args": ast.dump(node.args),
            }

    return api

def classify_change(old: Dict, new: Dict) -> ChangeType:
    if old == new:
        return "none" # or "patch"

    # breacking change
    for name, entry in old.items():
        if name not in new:
            return "major"

        if entry != new[name]:
            return "major"

    # extension
    return "minor"

# snapshot handling

def create_snapshot(py_path: Path, out_path: Path) -> None:
    snapshot = {
        "__version__": extract_version(py_path),
        "__hash__": file_hash(py_path),
        "api": extract_api(py_path),
    }

    out_path.write_text(json.dumps(snapshot, indent=2))
    print(f"Snapshot written to {out_path}")


def check_snapshot(py_path: Path, snapshot_path: Path) -> int:
    snapshot = json.loads(snapshot_path.read_text())

    old_api = snapshot["api"]
    old_version = snapshot["__version__"]
    old_hash = snapshot["__hash__"]

    new_api = extract_api(py_path)
    new_version = extract_version(py_path)
    new_hash = file_hash(py_path)

    change = classify_change(old_api, new_api)

    
    if change == "none":
        if old_hash != new_hash:
            v = bump_version(new_version, change)
            print(f"File changed but API unchanged -> patch suggested ({v}).")
        else:
            print("No changes detected.")
        return 0

    bump = version_bump_type(old_version, new_version)

    if bump != change:
        print(dedent(f"""
            Version mismatch detected:
                API change detected: {change}
                Version bump detected: {bump}
                Old version: {old_version}
                New version: {new_version}
        """).strip())
        return 1

    print("API and version are consistent.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage:")
        print("  python api_guard.py snapshot <input.py> <snapshot.json>")
        print("  python api_guard.py check <input.py> <snapshot.json>")
        sys.exit(1)

    command = sys.argv[1]
    py_file = Path(sys.argv[2])
    snapshot_file = Path(sys.argv[3])

    if command == "snapshot":
        create_snapshot(py_file, snapshot_file)
        sys.exit(0)

    elif command == "check":
        exit_code = check_snapshot(py_file, snapshot_file)
        sys.exit(exit_code)

    else:
        print("Unknown command:", command)
        sys.exit(1)

