# dc-rl tests

Upstream SustainDC shipped with no tests. This suite is being built up
alongside the thesis's adaptation of it (Hydro-Québec tariffs, PV, demand
charges, LP cross-checks).

## Running

```bash
PY=~/.pyenv/versions/3.10.14/envs/thesis-dcrl/bin/python

$PY tests/run_all.py                  # whole suite
$PY tests/run_all.py price            # only modules matching "price"
$PY tests/test_price_manager.py       # one module directly
```

**Use the `thesis-dcrl` interpreter, not bare `python3`** — bare `python3` is
pyenv 3.11 and has no `gymnasium`. `test_price_manager.py` happens to work
either way (numpy only), but `test_observation_layout.py` builds a real
`SustainDC` and needs the full dependency stack.

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
- **Pin real-world constants in their own test.** `test_price_manager.py`'s
  `test_rate_numbers_match_the_official_tariff_pdf` exists so a later edit
  can't silently drift a tariff figure away from the source document.
- **Hand-compute the expected value** where the arithmetic is the thing
  being tested (see the Rate M tier-boundary test), rather than asserting
  against whatever the code currently returns.

## Current modules

| Module | Covers |
|---|---|
| `test_price_manager.py` | `utils/price_manager.py` — Rate M two-tier energy + demand charge, Rate L flat energy + winter optimization charge, capacity-based rate auto-selection, running-peak demand ratchet, Rate M's 65% winter ratchet, HQ winter calendar |
| `test_observation_layout.py` | The observation contract — declared Box widths vs what the builders emit, the shared time/CI prefix, season features, and `SustainDC.DC_IDX`/`BAT_IDX`, which `harlsustaindc_env.py` consumes **positionally** to build the critic's shared observation (a stale index there fails silently, not loudly) |
