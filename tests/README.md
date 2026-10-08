# dc-rl tests

Upstream SustainDC shipped with no tests. This suite is being built up
alongside the thesis's adaptation of it (Hydro-Québec tariffs, PV, demand
charges, LP cross-checks).

## Running

```bash
PY=~/.pyenv/versions/3.10.14/envs/thesis-dcrl/bin/python

$PY tests/run_all.py                  # whole suite
$PY tests/run_all.py price            # only modules matching "price"
$PY tests/test_hydro_quebec_pricing.py  # one module directly
$PY tests/test_pricing_integration.py  # SustainDC pricing path
```

**Use the `thesis-dcrl` interpreter, not bare `python3`** — bare `python3` is
pyenv 3.11 and has no `gymnasium`. `test_flat_pricing.py` uses only the pricing package, but
`test_observation_layout.py` and `test_pricing_integration.py` build a real
`SustainDC` and need the full dependency stack.

No pytest required — it is not a dependency of this repo and is not
installed in the `thesis-dcrl` env. Tests are plain `test_*` functions
collected by `tests/_harness.py`. `pytest tests/` also works if you ever
install it (`tests/conftest.py` handles the import path).

## Adding a test module

Create `tests/test_<thing>.py`:

```python
import sys
from pathlib import Path

# repo root on sys.path -- running this file directly puts tests/ there, not the root
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tests._harness import run_module_tests
from utils.thing_under_test import Thing


def test_something_specific():
    assert Thing().value == 42


if __name__ == "__main__":
    sys.exit(1 if run_module_tests(globals()) else 0)
```

`run_all.py` picks it up automatically — no registration needed.

Conventions worth keeping:
- **Pin source figures in their own test.** `test_hydro_quebec_pricing.py`
  locks the cited tariff figures from `data/Pricing/hydro_quebec_2026.yaml`.
- **Hand-compute expected values** where arithmetic is the behavior under test,
  rather than asserting against current implementation output.
- **Use manager-level tests for billing state.** `PriceManager` owns shared
  billing-period energy and validation/rollback; individual models should be
  tested for their own tariff or schedule rules.

## Current modules

| Module | Covers |
|---|---|
| `test_pricing_contracts.py` | Public pricing dataclasses, immutable forecasts, and plugin protocol signatures |
| `test_pricing_loader.py` | Strict nested config, YAML validation, built-in registry, and import-path plugins |
| `test_flat_pricing.py` | Constant energy pricing |
| `test_time_of_use_pricing.py` | 24-hour schedules and midnight forecast wrapping |
| `test_time_series_pricing.py` | CSV prices, zero-order hold, source intervals, and series wrapping |
| `test_hydro_quebec_pricing.py` | Hydro-Québec Rate M, Rate L, pending data-centre scenario, and tariff validation |
| `test_pricing_manager.py` | Manager lifecycle, state/carry transactions, and standardized outputs |
| `test_pricing_integration.py` | Nested pricing config through a real `SustainDC` reset/step |
| `test_observation_layout.py` | Observation vector and shared-critic layout stability |
| `test_battery_efficiency.py` | Efficiency-aware bus/cell flows, round-trip energy, grid meter identity, throughput degradation reporting, and battery config propagation |
| `test_pv_manager.py` | pvlib Montréal EPW parsing, fixed-tilt AC production, capacity scaling, hourly-to-quarter-hour energy conservation, and PV/grid meter balance |
