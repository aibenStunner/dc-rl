"""Shared test-runner harness for dc-rl's test suite.

This repo has no pytest dependency (it is not in requirements.txt and is not
installed in the `thesis-dcrl` env), so tests are written to be runnable with
bare `python3` -- the same convention this thesis's v1/v2a/v2b/v3 models use
in their `test_invariance.py` files: plain module-level `test_*` functions,
collected and run by a small runner.

Each test module ends with:

    if __name__ == "__main__":
        sys.exit(run_module_tests(globals()))

...so it stays individually runnable, while `tests/run_all.py` runs the
whole suite at once. Tests also work under pytest if it is ever installed
(see conftest.py), since they are ordinary `test_*` functions.
"""
import traceback


def run_module_tests(module_globals, extra_tests=None):
    """Collect and run every `test_*` callable in `module_globals`.

    Args:
        module_globals: the calling module's `globals()`.
        extra_tests: optional list of (name, fn) run after the collected
            ones -- for slow/opt-in checks not picked up automatically.

    Returns:
        int: number of failures (0 = all passed), suitable for sys.exit().
    """
    tests = [(n, f) for n, f in sorted(module_globals.items())
             if n.startswith("test_") and callable(f)]
    if extra_tests:
        tests.extend(extra_tests)

    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:                # noqa: BLE001 - report and continue
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
            traceback.print_exc()

    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed
