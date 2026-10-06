"""Run dc-rl's whole test suite with bare `python3` (no pytest needed).

    python3 tests/run_all.py            # run everything
    python3 tests/run_all.py price      # only modules matching "price"

Discovers every `tests/test_*.py`, imports it, runs its `test_*` functions
via the shared harness, and reports a per-module and overall summary.
Exit code is the number of failing modules (0 = all green), so it drops
straight into CI or a pre-commit hook later.
"""
import importlib
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tests._harness import run_module_tests  # noqa: E402


def discover(pattern=None):
    """Every tests/test_*.py module name, optionally filtered by substring."""
    tests_dir = Path(__file__).resolve().parent
    names = sorted(p.stem for p in tests_dir.glob("test_*.py"))
    if pattern:
        names = [n for n in names if pattern in n]
    return names


def main(argv):
    pattern = argv[1] if len(argv) > 1 else None
    module_names = discover(pattern)
    if not module_names:
        print(f"No test modules found{f' matching {pattern!r}' if pattern else ''}.")
        return 1

    results = {}
    for name in module_names:
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
        module = importlib.import_module(f"tests.{name}")
        results[name] = run_module_tests(vars(module))

    print(f"\n{'=' * 70}\nSUITE SUMMARY\n{'=' * 70}")
    for name, failures in results.items():
        print(f"  {'FAIL' if failures else 'ok  '}  {name}"
              + (f"  ({failures} failing)" if failures else ""))
    failing_modules = sum(1 for f in results.values() if f)
    total_failures = sum(results.values())
    print(f"\n{len(results) - failing_modules}/{len(results)} modules passed"
          + (f", {total_failures} failing test(s)" if total_failures else ""))
    return failing_modules


if __name__ == "__main__":
    sys.exit(main(sys.argv))
