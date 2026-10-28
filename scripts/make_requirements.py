"""Day 23a: writes requirements.txt for the free host. Run from the project root:   python scripts/make_requirements.py
It reads which outside packages the project's own code really imports (not tests, not your whole virtual
environment) and pins the versions that are installed right now, so the host installs the same ones you tested with."""
import ast
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FOLDERS = ("app", "agents", "tools", "ui", "evals")
EXTRA = ("pyarrow",)                       # used by pandas for some file types; harmless to pin when installed
OWN = {"app", "agents", "tools", "ui", "evals", "scripts", "tests"}


def imported_names(folders=FOLDERS, root=ROOT) -> set:
    names = set()
    for folder in folders:
        for path in Path(root, folder).rglob("*.py"):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names |= {a.name.split(".")[0] for a in node.names}
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names.add(node.module.split(".")[0])
    return {n for n in names if n not in sys.stdlib_module_names and n not in OWN}


def pinned(names, extra=EXTRA) -> tuple:
    """(lines 'name==version', names that are not installed)."""
    owners = metadata.packages_distributions()
    dists, missing = {}, []
    for n in sorted(set(names) | set(extra)):
        found = owners.get(n)
        if not found:
            if n not in extra:
                missing.append(n)
            continue
        for d in found:
            dists[d] = metadata.version(d)
    return [f"{d}=={v}" for d, v in sorted(dists.items(), key=lambda x: x[0].lower())], missing


def main():
    lines, missing = pinned(imported_names())
    if missing:
        sys.exit("These imports are not installed in this environment, so they cannot be pinned: "
                 + ", ".join(missing) + ". Activate the project's virtual environment and try again.")
    (ROOT / "requirements.txt").write_text("\n".join(lines) + "\n")
    print(f"Wrote requirements.txt with {len(lines)} packages:")
    print("\n".join("  " + x for x in lines))


if __name__ == "__main__":
    main()
