# Pluggable Pricing Models Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn SustainDC pricing into a first-class plug-in subsystem with built-in flat, time-of-use, time-series, and Hydro-Québec models while preserving current Rate M/Rate L billing behavior and the agents' three-value pricing observation block.

**Architecture:** `PriceManager` becomes a model-neutral orchestrator. A strict nested `pricing:` configuration selects a built-in registry entry or explicit `module:attribute` plug-in and loads a cited YAML parameter file; each pricing model implements one protocol and returns standardized observations and charge categories. `SustainDC` and reward functions consume only those standardized outputs.

**Tech Stack:** Python 3.10, dataclasses, `typing.Protocol`, `importlib`, NumPy, PyYAML, pandas/CSV for time-series prices, Gymnasium, standalone `test_*` harness.

**Spec:** `docs/superpowers/specs/2026-10-07-pluggable-pricing-design.md`

## Global Constraints

- Configuration is a clean break: exactly one nested `pricing:` mapping; removed flat pricing keys must be rejected.
- Built-in model names: `hydro_quebec`, `flat`, `time_of_use`, `time_series`.
- External models use an explicit `module:attribute` import path; there is no directory scanning.
- Tariff figures and source metadata live in YAML, not Python constants.
- The pricing observation block remains exactly three values in the current positions: normalized price, billing progress, normalized peak.
- Reward logic consumes standardized charge outputs and contains no tariff-specific billing rules.
- Existing Rate M, Rate L, and proposed data-centre scenario arithmetic must remain behaviorally identical.
- Parameter files use `schema_version: 1`; loaders use `yaml.safe_load`.
- Prices, meter readings, capacities, thresholds, normalized outputs, and forecast arrays must be finite; values constrained to `[0,1]` stay in that range.
- Implementation docstrings are self-contained and do not reference historical simulator versions.
- Run commands with `~/.pyenv/versions/3.10.14/envs/thesis-dcrl/bin/python`, not bare `python3`.

## Review Focus

- Missing, empty, non-mapping, or unknown-key `pricing` configuration must fail before environment construction with a field-specific error.
- Malformed import paths, import failures, and protocol-incompatible plug-ins must name the offending path and expected interface.
- Time-series and TOU forecasts crossing midnight/year end must wrap correctly and always return exactly `future_steps` finite normalized values.
- Negative, NaN, or infinite energy/demand readings must be rejected before pricing state mutates.
- A model returning the wrong forecast length, non-finite charges, or normalized values outside `[0,1]` must be rejected at the manager boundary.

---

## File Structure

**Create**

- `utils/pricing/__init__.py` — public pricing API and compatibility-free imports.
- `utils/pricing/contracts.py` — protocol and standardized dataclasses.
- `utils/pricing/loader.py` — strict config/YAML validation and built-in/import-path model resolution.
- `utils/pricing/manager.py` — model-neutral `PriceManager` lifecycle and getters.
- `utils/pricing/models/__init__.py` — built-in registry only.
- `utils/pricing/models/flat.py` — energy-only flat model.
- `utils/pricing/models/time_of_use.py` — cyclic 24-hour schedule model.
- `utils/pricing/models/time_series.py` — CSV-backed exogenous series model.
- `utils/pricing/models/hydro_quebec.py` — Rate M, Rate L, proposed data-centre scenario.
- `data/Pricing/hydro_quebec_2026.yaml` — official/cited rate parameters and pending-scenario metadata.
- `data/Pricing/flat_example.yaml` — runnable flat example.
- `data/Pricing/time_of_use_example.yaml` — runnable 24-hour schedule example.
- `data/Pricing/time_series_example.yaml` — runnable CSV model config.
- `data/Pricing/example_prices.csv` — one-year hourly example series.
- `tests/test_pricing_contracts.py` — dataclass and protocol invariants.
- `tests/test_pricing_loader.py` — nested config, YAML validation, registry, import-path loading.
- `tests/fixtures/custom_pricing_plugin.py` — valid/invalid test plug-ins.
- `tests/test_flat_pricing.py` — flat model behavior.
- `tests/test_time_of_use_pricing.py` — schedule and forecast wrap behavior.
- `tests/test_time_series_pricing.py` — CSV loading/resampling/wrap behavior.
- `tests/test_hydro_quebec_pricing.py` — current tariff regression suite, migrated.
- `sphinx/usage/custompricing.rst` — user-facing built-in and custom plug-in guide.

**Modify**

- `sustaindc_env.py` — consume nested pricing config and model-neutral manager API.
- `utils/reward_creator.py` — consume `total_price_cost_this_step_c` only.
- `harl/configs/envs_cfgs/sustaindc.yaml` — replace flat keys with `pricing:` block.
- `tests/test_observation_layout.py` — use nested config and pin three-value block stability.
- `tests/README.md` — list the pricing modules and focused commands.
- `sphinx/usage/index.rst` — add pricing guide to navigation.
- `sphinx/usage/mainconf.rst` — document strict nested config.
- `CLAUDE.md` in the parent repository — update pricing architecture and commands after integration succeeds.

**Delete after all imports/tests migrate**

- `utils/price_manager.py` — replaced by `utils/pricing/` package.
- `tests/test_price_manager.py` — split into focused model/loader tests.

---

### Task 1: Define the Model-Neutral Pricing Contracts

**Files:**
- Create: `utils/pricing/__init__.py`
- Create: `utils/pricing/contracts.py`
- Create: `tests/test_pricing_contracts.py`

**Interfaces:**
- Consumes: NumPy and standard-library dataclasses/typing.
- Produces: `PricingContext`, `PricingClock`, `MeterReading`, `PricingState`, `PricingObservation`, `PricingCharges`, `PricingModel`.

- [ ] **Step 1: Write failing contract tests**

Create `tests/test_pricing_contracts.py` with tests that pin constructor fields, `PricingCharges.total_cost_c`, runtime protocol conformance, and immutable output arrays:

```python
import sys
from pathlib import Path
from dataclasses import FrozenInstanceError

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tests._harness import run_module_tests
from utils.pricing.contracts import (
    MeterReading, PricingCharges, PricingClock, PricingContext,
    PricingModel, PricingObservation, PricingState,
)


def test_charges_total_is_sum_of_standard_categories():
    charges = PricingCharges(energy_cost_c=10.0,
                             demand_cost_increment_c=3.0,
                             additional_cost_increment_c=2.0)
    assert charges.total_cost_c == 15.0


def test_observation_requires_exact_forecast_shape_at_manager_boundary():
    obs = PricingObservation(current_price_c_per_kwh=5.0,
                             normalized_price=0.5,
                             forecast_normalized=np.array([0.5, 0.5]),
                             billing_progress=0.25,
                             normalized_peak=0.4)
    assert obs.forecast_normalized.shape == (2,)


def test_protocol_accepts_structural_implementation():
    class MinimalModel:
        name = "minimal"
        def reset(self, **kwargs): pass
        def step(self, **kwargs): return PricingCharges(0.0, 0.0, 0.0)
        def observe(self, **kwargs): return PricingObservation(
            0.0, 0.0, np.zeros(1, dtype=np.float32), 0.0, 0.0)
        def export_carry_state(self, state): return {}
    assert isinstance(MinimalModel(), PricingModel)


if __name__ == "__main__":
    sys.exit(1 if run_module_tests(globals()) else 0)
```

- [ ] **Step 2: Run the test and verify import failure**

Run:

```bash
PY=~/.pyenv/versions/3.10.14/envs/thesis-dcrl/bin/python
$PY tests/test_pricing_contracts.py
```

Expected: FAIL because `utils.pricing.contracts` does not exist.

- [ ] **Step 3: Implement the contracts**

Create `utils/pricing/contracts.py` with these exact public signatures:

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Mapping, Protocol, runtime_checkable
import numpy as np

@dataclass(frozen=True)
class PricingContext:
    datacenter_capacity_mw: float
    timestep_minutes: int
    future_steps: int

@dataclass(frozen=True)
class PricingClock:
    day_of_year: int
    hour: float

@dataclass(frozen=True)
class MeterReading:
    energy_kwh: float
    demand_kw: float

@dataclass
class PricingState:
    billing_period_energy_kwh: float = 0.0
    running_peak_kw: float = 0.0
    model_state: dict[str, object] = field(default_factory=dict)

@dataclass(frozen=True)
class PricingObservation:
    current_price_c_per_kwh: float
    normalized_price: float
    forecast_normalized: np.ndarray
    billing_progress: float
    normalized_peak: float

@dataclass(frozen=True)
class PricingCharges:
    energy_cost_c: float
    demand_cost_increment_c: float
    additional_cost_increment_c: float
    @property
    def total_cost_c(self) -> float:
        return (self.energy_cost_c + self.demand_cost_increment_c
                + self.additional_cost_increment_c)

@runtime_checkable
class PricingModel(Protocol):
    name: str
    def reset(self, *, context: PricingContext, state: PricingState,
              clock: PricingClock, carry_state: Mapping[str, object] | None,
              options: Mapping[str, object]) -> None: ...
    def step(self, *, context: PricingContext, state: PricingState,
             clock: PricingClock, reading: MeterReading) -> PricingCharges: ...
    def observe(self, *, context: PricingContext, state: PricingState,
                clock: PricingClock) -> PricingObservation: ...
    def export_carry_state(self, state: PricingState) -> Mapping[str, object]: ...
```

Export these names from `utils/pricing/__init__.py`.

- [ ] **Step 4: Run contract tests**

Run: `$PY tests/test_pricing_contracts.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add utils/pricing/__init__.py utils/pricing/contracts.py tests/test_pricing_contracts.py
git commit -m "feat: define pricing model contracts"
```

---

### Task 2: Implement Strict Config Loading and Plug-In Resolution

**Files:**
- Create: `utils/pricing/loader.py`
- Create: `utils/pricing/models/__init__.py`
- Create: `tests/test_pricing_loader.py`
- Create: `tests/fixtures/custom_pricing_plugin.py`

**Interfaces:**
- Consumes: `PricingModel` from Task 1.
- Produces: `PricingConfig`, `load_pricing_config(raw, repo_root)`, `load_parameter_file(path, expected_model)`, `resolve_pricing_model(selector)`, `create_pricing_model(pricing_config)`.

- [ ] **Step 1: Write failing loader tests for the five review-focus failures**

Create fixtures:

```python
# tests/fixtures/custom_pricing_plugin.py
from utils.pricing.contracts import PricingCharges, PricingObservation
import numpy as np

class ValidPlugin:
    name = "fixture_plugin"
    def __init__(self, parameters): self.parameters = parameters
    def reset(self, **kwargs): pass
    def step(self, **kwargs): return PricingCharges(0.0, 0.0, 0.0)
    def observe(self, context, **kwargs): return PricingObservation(
        1.0, 1.0, np.ones(context.future_steps, dtype=np.float32), 0.0, 0.0)
    def export_carry_state(self, state): return {}

class InvalidPlugin:
    pass
```

Create `tests/test_pricing_loader.py` with concrete tests:

```python
def test_missing_pricing_mapping_is_rejected():
    _assert_error({}, "pricing")

def test_old_flat_keys_are_rejected_with_migration_message():
    _assert_error({"pricing": {...}, "tariff_rate_override": "rate_l"},
                  "pricing.options.tariff")

def test_unknown_pricing_keys_are_rejected():
    _assert_error({"pricing": {"model": "flat", "config_file": "x.yaml",
                               "surprise": 1}}, "surprise")

def test_empty_yaml_is_rejected(tmp_path): ...
def test_unsupported_schema_version_is_rejected(tmp_path): ...
def test_unknown_registry_name_lists_builtins(): ...
def test_malformed_import_path_names_the_path(): ...
def test_import_failure_names_the_path(): ...
def test_incompatible_plugin_is_rejected(): ...
def test_valid_import_path_plugin_loads(): ...
def test_relative_config_path_resolves_from_repo_root(tmp_path): ...
```

The helper must assert both exception type `PricingConfigError` and a field/source substring.

- [ ] **Step 2: Run loader tests and verify failure**

Run: `$PY tests/test_pricing_loader.py`

Expected: FAIL because loader API does not exist.

- [ ] **Step 3: Implement strict nested config parsing**

In `loader.py` define:

```python
@dataclass(frozen=True)
class PricingConfig:
    model: str
    config_file: Path
    options: Mapping[str, object]

class PricingConfigError(ValueError):
    pass
```

`load_pricing_config` must:

- require `raw["pricing"]` to be a mapping;
- reject removed top-level `tariff_rate_override` and `demand_floor_kw` with exact replacement paths;
- require exactly `model`, `config_file`, optional `options`;
- reject unknown nested keys;
- resolve relative files from repository root;
- load YAML through `yaml.safe_load`;
- require a mapping with `schema_version == 1` and matching `model` for built-ins.

- [ ] **Step 4: Implement built-in and import-path resolution**

Create `utils/pricing/models/flat.py` now with the minimal real built-in required to prove registry resolution end-to-end:

```python
class FlatPricingModel:
    name = "flat"
    def __init__(self, parameters):
        self.price_c_per_kwh = float(parameters["energy_price_c_per_kwh"])
    def reset(self, **kwargs): pass
    def step(self, reading, **kwargs):
        return PricingCharges(reading.energy_kwh * self.price_c_per_kwh, 0.0, 0.0)
    def observe(self, context, **kwargs):
        normalized = 0.0 if self.price_c_per_kwh == 0 else 1.0
        return PricingObservation(self.price_c_per_kwh, normalized,
            np.full(context.future_steps, normalized, dtype=np.float32), 0.0, 0.0)
    def export_carry_state(self, state): return {}
```

In `models/__init__.py`, register it immediately:

```python
BUILTIN_PRICING_MODELS: Mapping[str, type[PricingModel]] = {
    "flat": FlatPricingModel,
}
```

In `loader.py`, treat selectors containing `:` as import paths. Split once; reject empty module/attribute; call `importlib.import_module`; retrieve the attribute; instantiate with the parsed parameter mapping; validate `isinstance(instance, PricingModel)`; wrap all failures in `PricingConfigError` naming the selector.

For built-ins, resolve `flat` successfully and fail unknown names with all current registry keys. Task 3 adds the remaining reusable built-ins; Task 4 adds `hydro_quebec`.

- [ ] **Step 5: Run loader tests**

Run: `$PY tests/test_pricing_loader.py`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add utils/pricing/loader.py utils/pricing/models/__init__.py \
  tests/test_pricing_loader.py tests/fixtures/custom_pricing_plugin.py
git commit -m "feat: load pricing models from config or import path"
```

---

### Task 3: Add Flat, TOU, and Time-Series Built-ins

**Files:**
- Modify: `utils/pricing/models/flat.py`
- Create: `utils/pricing/models/time_of_use.py`
- Create: `utils/pricing/models/time_series.py`
- Modify: `utils/pricing/models/__init__.py`
- Create: `data/Pricing/flat_example.yaml`
- Create: `data/Pricing/time_of_use_example.yaml`
- Create: `data/Pricing/time_series_example.yaml`
- Create: `data/Pricing/example_prices.csv`
- Create: `tests/test_flat_pricing.py`
- Create: `tests/test_time_of_use_pricing.py`
- Create: `tests/test_time_series_pricing.py`

**Interfaces:**
- Consumes: contracts from Task 1 and loader conventions from Task 2.
- Produces: `FlatPricingModel`, `TimeOfUsePricingModel`, `TimeSeriesPricingModel`; registry keys `flat`, `time_of_use`, `time_series`.

- [ ] **Step 1: Write failing flat-model tests**

Tests must assert:

```python
charges = model.step(... reading=MeterReading(energy_kwh=25.0, demand_kw=100.0))
assert charges == PricingCharges(energy_cost_c=125.0,
                                 demand_cost_increment_c=0.0,
                                 additional_cost_increment_c=0.0)
assert model.observe(...).forecast_normalized.shape == (8,)
assert model.observe(...).billing_progress == 0.0
```

Also reject negative/NaN `energy_price_c_per_kwh` through the loader.

- [ ] **Step 2: Complete `FlatPricingModel` validation and add example YAML**

Use exact parameter schema:

```yaml
schema_version: 1
model: flat
metadata:
  name: Example flat rate
  currency: CAD
energy_price_c_per_kwh: 5.0
```

Forecast is constant ones after normalization; price denominator is the configured price, with the zero-price case normalized to `0.0`.

- [ ] **Step 3: Run flat tests**

Run: `$PY tests/test_flat_pricing.py`

Expected: PASS.

- [ ] **Step 4: Write failing TOU tests**

Tests must cover:

- exactly 24 finite non-negative hourly prices;
- fractional hour selects `floor(hour) % 24`;
- forecast across 23:45 wraps to hour 00;
- output length equals `future_steps`;
- normalized values stay `[0,1]`;
- billing progress is `(hour % 24) / 24`.

- [ ] **Step 5: Implement `TimeOfUsePricingModel` and example YAML**

Parameter schema:

```yaml
schema_version: 1
model: time_of_use
metadata:
  name: Example daily TOU
  currency: CAD
hourly_prices_c_per_kwh: [5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 8, 8,
                          6, 6, 6, 6, 9, 9, 9, 9, 6, 6, 5, 5]
```

Forecast slot `k` uses `(clock.hour + (k+1)*context.timestep_minutes/60) % 24`.

- [ ] **Step 6: Run TOU tests**

Run: `$PY tests/test_time_of_use_pricing.py`

Expected: PASS.

- [ ] **Step 7: Write failing time-series tests**

Create a temporary CSV fixture and test:

- named price column;
- missing column error names the column;
- NaN/negative values rejected;
- hourly input becomes 15-minute values deterministically;
- forecast wraps at the end when `wrap: true`;
- `wrap: false` raises before reading beyond data;
- output length equals `future_steps`.

- [ ] **Step 8: Implement `TimeSeriesPricingModel`, example CSV, and YAML**

Parameter schema:

```yaml
schema_version: 1
model: time_series
metadata:
  name: Example hourly market price
  currency: CAD
csv_file: data/Pricing/example_prices.csv
price_column: price_c_per_kwh
source_interval_minutes: 60
wrap: true
```

Resolve CSV relative to repository root. Use zero-order hold when source interval is an integer multiple of simulation interval; reject unsupported non-integer ratios instead of silently interpolating billing prices.

- [ ] **Step 9: Populate registry and run all three model tests**

`models/__init__.py`:

```python
BUILTIN_PRICING_MODELS = {
    "flat": FlatPricingModel,
    "time_of_use": TimeOfUsePricingModel,
    "time_series": TimeSeriesPricingModel,
}
```

Run:

```bash
$PY tests/test_flat_pricing.py
$PY tests/test_time_of_use_pricing.py
$PY tests/test_time_series_pricing.py
```

Expected: all PASS.

- [ ] **Step 10: Commit**

```bash
git add utils/pricing/models data/Pricing tests/test_flat_pricing.py \
  tests/test_time_of_use_pricing.py tests/test_time_series_pricing.py
git commit -m "feat: add reusable pricing model builtins"
```

---

### Task 4: Extract Hydro-Québec Logic and Parameters

**Files:**
- Create: `utils/pricing/models/hydro_quebec.py`
- Create: `data/Pricing/hydro_quebec_2026.yaml`
- Modify: `utils/pricing/models/__init__.py`
- Create: `tests/test_hydro_quebec_pricing.py`
- Modify: `tests/test_pricing_loader.py`

**Interfaces:**
- Consumes: standardized contracts and loader.
- Produces: `HydroQuebecPricingModel`, registry key `hydro_quebec`; behavior matching current Rate M/Rate L/proposed-scenario tests.

- [ ] **Step 1: Move current regression assertions into failing model tests**

Port every behavior from `tests/test_price_manager.py`, but instantiate through:

```python
config = load_pricing_config({"pricing": {
    "model": "hydro_quebec",
    "config_file": "data/Pricing/hydro_quebec_2026.yaml",
    "options": {"tariff": "rate_m", "demand_floor_kw": 0.0},
}}, REPO_ROOT)
manager = PriceManager.from_config(config, datacenter_capacity_mw=2.0,
                                   future_steps=8, timestep_minutes=15)
```

Pin all official numbers from YAML, tier-boundary arithmetic, capacity auto-selection, pending scenario never auto-selection, running peak increments, winter ratchet carry, Rate L optimization charge, and winter calendar boundaries.

Add a parity test that drives old and new managers over one deterministic trace before old code is deleted:

```python
for energy_kwh, is_winter in trace:
    old.step(energy_kwh, is_winter)
    new.step(energy_kwh, day_of_year=day, hour=hour)
    assert new.get_total_cost_this_step_c() == old_energy + old_demand + old_optimization
```

- [ ] **Step 2: Run Hydro-Québec tests and verify failure**

Run: `$PY tests/test_hydro_quebec_pricing.py`

Expected: FAIL because model/YAML are absent.

- [ ] **Step 3: Write the cited parameter file**

`hydro_quebec_2026.yaml` must include:

```yaml
schema_version: 1
model: hydro_quebec
metadata:
  name: Hydro-Québec 2026 electricity rates
  currency: CAD
  effective_date: 2026-04-01
  source: hq-electricity-rates.pdf
  winter:
    start_day_of_year: 334
    end_day_of_year: 89
selection:
  rate_l_threshold_kw: 5000.0
rates:
  rate_m:
    approved: true
    citation: Chapter 4, Section 1, articles 4.1-4.8
    energy_tier1_c_per_kwh: 6.292
    energy_tier2_c_per_kwh: 4.666
    tier2_threshold_kwh: 210000.0
    demand_charge_c_per_kw_month: 1824.2
    minimum_billing_demand_kw: 0.0
    winter_ratchet_fraction: 0.65
    minimum_bill_c_1phase: 1542.6
    minimum_bill_c_3phase: 4627.8
  rate_l:
    approved: true
    citation: Chapter 5, Section 1, articles 5.1-5.9
    energy_price_c_per_kwh: 3.821
    demand_charge_c_per_kw_month: 1502.7
    minimum_billing_demand_kw: 5000.0
    optimization_charge_c_per_kw_day: 880.8
    optimization_charge_cap_c_per_kw_month: 2642.0
    optimization_overrun_fraction: 1.10
  new_dc_rate:
    approved: false
    approximate: true
    citation: Hydro-Québec press release, pending Régie approval
    energy_price_c_per_kwh: 13.0
    demand_charge_c_per_kw_month: 0.0
    minimum_billing_demand_kw: 5000.0
```

- [ ] **Step 4: Implement Hydro-Québec model from current behavior**

Move tariff selection, winter detection, tier arithmetic, demand increments, ratchet state, and optimization-charge state into `hydro_quebec.py`. Do not leave official numbers in Python. Store model-specific carry state under keys:

```python
{
    "winter_peak_kw": float,
    "ending_peak_kw": float,
}
```

Map current `optimization_charge_increment_c` to standardized `additional_cost_increment_c`.

- [ ] **Step 5: Register and validate YAML/model agreement**

Add `"hydro_quebec": HydroQuebecPricingModel` to the registry. Loader validation must reject missing `rates`, unsupported tariff options, YAML `model` mismatch, and non-finite/negative figures.

- [ ] **Step 6: Run Hydro-Québec and loader tests**

Run:

```bash
$PY tests/test_hydro_quebec_pricing.py
$PY tests/test_pricing_loader.py
```

Expected: PASS, including old/new parity.

- [ ] **Step 7: Commit**

```bash
git add utils/pricing/models/hydro_quebec.py \
  utils/pricing/models/__init__.py data/Pricing/hydro_quebec_2026.yaml \
  tests/test_hydro_quebec_pricing.py tests/test_pricing_loader.py
git commit -m "feat: add Hydro-Quebec pricing plugin"
```

---

### Task 5: Implement the Model-Neutral PriceManager

**Files:**
- Create: `utils/pricing/manager.py`
- Modify: `utils/pricing/__init__.py`
- Create: `tests/test_pricing_manager.py`

**Interfaces:**
- Consumes: `PricingConfig`, model loader, contracts.
- Produces: `PriceManager.from_config(...)`, reset/step/getters exactly listed in the spec.

- [ ] **Step 1: Write failing manager-boundary tests**

Cover all review-focus boundary failures:

```python
def test_negative_meter_energy_is_rejected_before_state_mutates(): ...
def test_nan_meter_energy_is_rejected_before_state_mutates(): ...
def test_plugin_wrong_forecast_length_is_rejected(): ...
def test_plugin_nonfinite_charge_is_rejected(): ...
def test_invalid_observation_restores_state_and_manager_outputs_in_place(): ...
def test_plugin_normalized_values_outside_unit_interval_are_rejected(): ...
def test_total_cost_equals_standard_categories(): ...
def test_exported_carry_state_round_trips(): ...
```

Use fixture plug-ins that intentionally violate one output invariant each.

- [ ] **Step 2: Run manager tests and verify failure**

Run: `$PY tests/test_pricing_manager.py`

Expected: FAIL because manager does not exist.

- [ ] **Step 3: Implement construction and lifecycle**

Required constructor:

```python
@classmethod
def from_config(cls, pricing_config: PricingConfig, *,
                datacenter_capacity_mw: float,
                future_steps: int,
                timestep_minutes: int = 15) -> "PriceManager": ...
```

Required lifecycle:

```python
def reset(self, *, init_day: int, init_hour: float,
          carry_state: Mapping[str, object] | None = None): ...
def step(self, *, metered_energy_kwh: float,
         day_of_year: int, hour: float): ...
```

On step, compute `demand_kw = energy_kwh / (timestep_minutes / 60)`. Deep-snapshot shared state before model interaction; keep model step, charge validation, manager billing-energy update, observation creation, and observation validation in one transaction. On any exception, restore shared state in place and preserve the prior manager charges, observation, and clock; only commit those outputs after every validation succeeds.

- [ ] **Step 4: Implement standardized getters**

Implement every getter listed in spec section 7. Return copies of NumPy forecasts to prevent callers mutating manager state.

- [ ] **Step 5: Run manager and model tests**

Run:

```bash
$PY tests/test_pricing_manager.py
$PY tests/run_all.py pricing
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add utils/pricing/manager.py utils/pricing/__init__.py tests/test_pricing_manager.py
git commit -m "feat: orchestrate pluggable pricing models"
```

---

### Task 6: Migrate SustainDC, Reward, and Configuration

**Files:**
- Modify: `sustaindc_env.py`
- Modify: `utils/reward_creator.py`
- Modify: `harl/configs/envs_cfgs/sustaindc.yaml`
- Modify: `tests/test_observation_layout.py`
- Create: `tests/test_pricing_integration.py`

**Interfaces:**
- Consumes: `load_pricing_config`, new `PriceManager`.
- Produces: unchanged three-value observation block and model-neutral reward/info keys.

- [ ] **Step 1: Write failing clean-break configuration tests**

In `tests/test_pricing_integration.py`, assert:

```python
def test_default_yaml_contains_nested_pricing_only():
    env_args = yaml.safe_load(Path(...).read_text())
    assert set(env_args["pricing"]) == {"model", "config_file", "options"}
    assert "tariff_rate_override" not in env_args
    assert "demand_floor_kw" not in env_args


def test_env_rejects_removed_flat_pricing_keys(): ...
def test_env_constructs_resets_and_steps_with_nested_pricing(): ...
def test_info_uses_model_neutral_charge_keys_only(): ...
def test_price_feature_block_remains_three_values(): ...
```

- [ ] **Step 2: Run integration tests and verify failure**

Run: `$PY tests/test_pricing_integration.py`

Expected: FAIL because environment still imports `utils.price_manager` and uses old keys.

- [ ] **Step 3: Replace environment defaults and construction**

In `EnvConfig.DEFAULT_CONFIG`, replace both flat keys with:

```python
"pricing": {
    "model": "hydro_quebec",
    "config_file": "data/Pricing/hydro_quebec_2026.yaml",
    "options": {"tariff": "auto", "demand_floor_kw": 0.0},
},
```

Deep-copy nested defaults in `EnvConfig.__init__` before applying overrides; a shallow `.copy()` would share `options` across environments.

Construct pricing through `load_pricing_config` and `PriceManager.from_config`. Remove `is_hq_winter_day` from `SustainDC`; pass `day_of_year`/`hour` into manager step.

- [ ] **Step 4: Preserve observation width with the neutral getter**

Change `_price_feature_block()` only:

```python
return np.array([
    self.price_m.get_current_price(),
    self.price_m.get_billing_progress_fraction(),
    self.price_m.get_normalized_peak(),
], dtype=np.float32)
```

Run `test_observation_layout.py` immediately. Expected widths remain LS 31, DC 19, battery 18; shared width remains 34.

- [ ] **Step 5: Migrate reward/info keys**

`_calculate_reward_params()` must expose:

```python
"energy_cost_this_step_c"
"demand_charge_increment_c"
"additional_charge_increment_c"
"total_price_cost_this_step_c"
```

Remove `optimization_charge_increment_c`. Change `default_price_reward` to:

```python
return -0.01 * params["total_price_cost_this_step_c"]
```

No tariff-specific names remain in reward code.

- [ ] **Step 6: Update default YAML**

Use:

```yaml
pricing:
  model: hydro_quebec
  config_file: data/Pricing/hydro_quebec_2026.yaml
  options:
    tariff: auto
    demand_floor_kw: 0.0
```

Remove old keys entirely.

- [ ] **Step 7: Run integration, observation, and reward tests**

Run:

```bash
$PY tests/test_pricing_integration.py
$PY tests/test_observation_layout.py
$PY tests/run_all.py
```

Expected: PASS.

- [ ] **Step 8: Run a short training smoke test**

Run:

```bash
timeout 220 $PY train_sustaindc.py --algo happo --exp_name pricing_plugin_smoke \
  --n_rollout_threads 1 --n_eval_rollout_threads 1 \
  --episode_length 25 --num_env_steps 50
```

Expected: exit 0; shared observation `(34,)`; agent observation `(31,)`; `Start Training`; no traceback.

- [ ] **Step 9: Commit**

```bash
git add sustaindc_env.py utils/reward_creator.py \
  harl/configs/envs_cfgs/sustaindc.yaml tests/test_observation_layout.py \
  tests/test_pricing_integration.py
git commit -m "refactor: integrate pluggable pricing into SustainDC"
```

---

### Task 7: Remove the Legacy Module and Finish Migration Coverage

**Files:**
- Delete: `utils/price_manager.py`
- Delete: `tests/test_price_manager.py`
- Modify: any remaining imports found by search.

**Interfaces:**
- Consumes: all new pricing package APIs.
- Produces: repository with no active import of deleted module and no compatibility fallback.

- [ ] **Step 1: Search for old API usage**

Run:

```bash
grep -RIn "utils.price_manager\|tariff_rate_override\|demand_floor_kw\|optimization_charge_increment_c\|get_tier2_progress_fraction" \
  --include='*.py' --include='*.yaml' --include='*.rst' --include='*.md' .
```

Expected: only migration/error-message tests and approved `pricing.options.demand_floor_kw`; no production old API calls.

- [ ] **Step 2: Delete old module and old monolithic tests**

```bash
rm utils/price_manager.py tests/test_price_manager.py
```

- [ ] **Step 3: Run import and suite checks**

Run:

```bash
$PY -c "from utils.pricing import PriceManager; import sustaindc_env"
$PY tests/run_all.py
```

Expected: PASS, no old-module import errors.

- [ ] **Step 4: Commit**

```bash
git add -A utils/price_manager.py tests/test_price_manager.py utils/pricing tests
git commit -m "refactor: remove monolithic pricing module"
```

---

### Task 8: Document the Pricing Plug-In System

**Files:**
- Create: `sphinx/usage/custompricing.rst`
- Modify: `sphinx/usage/index.rst`
- Modify: `sphinx/usage/mainconf.rst`
- Modify: `tests/README.md`
- Modify: parent repository `CLAUDE.md`

**Interfaces:**
- Consumes: final configuration and API from Tasks 1-7.
- Produces: user instructions that match executable behavior.

- [ ] **Step 1: Write the pricing guide**

Document:

- architecture diagram in text (`SustainDC -> PriceManager -> PricingModel`);
- built-in model table;
- canonical nested YAML;
- each parameter-file schema;
- import-path plug-in example implementing the exact protocol;
- units (`cents/kWh`, `cents/kW-month`, `kWh`, `kW`);
- carry-state semantics;
- observation fields and reward/info keys;
- explicit limitations and pending-rate status;
- errors users will see for malformed config.

- [ ] **Step 2: Add navigation and main-config reference**

Add `custompricing` to `sphinx/usage/index.rst`. Replace old inline pricing examples in `mainconf.rst` with the canonical `pricing:` block.

- [ ] **Step 3: Update test documentation**

List the new focused modules and commands in `tests/README.md`:

```bash
$PY tests/run_all.py pricing
$PY tests/test_hydro_quebec_pricing.py
$PY tests/test_pricing_integration.py
```

- [ ] **Step 4: Update parent CLAUDE.md**

Describe the nested `pricing:` config, built-in names, import-path plug-ins, parameter file location, and model-neutral observation/reward contract. Do not mention removed flat keys as supported behavior.

- [ ] **Step 5: Verify docs against live configuration**

Run a script that parses every YAML snippet/file mentioned by docs and calls `load_pricing_config` for each runnable example. Run:

```bash
$PY -m compileall utils/pricing tests
$PY tests/run_all.py
```

Expected: all examples load; suite passes.

- [ ] **Step 6: Commit docs**

In the simulator repo:

```bash
git add sphinx/usage/custompricing.rst sphinx/usage/index.rst \
  sphinx/usage/mainconf.rst tests/README.md
git commit -m "docs: explain pluggable pricing models"
```

In the parent repo, commit `CLAUDE.md` separately after the submodule pointer is advanced.

---

### Task 9: Final Verification and Submodule Update

**Files:**
- Modify: parent repo submodule pointer `dc-rl`.

**Interfaces:**
- Consumes: completed simulator commits.
- Produces: verified simulator branch and parent repository pinned to it.

- [ ] **Step 1: Run full static and unit verification**

```bash
PY=~/.pyenv/versions/3.10.14/envs/thesis-dcrl/bin/python
$PY -m compileall utils/pricing sustaindc_env.py utils/reward_creator.py tests
$PY tests/run_all.py
```

Expected: compile succeeds; all modules pass.

- [ ] **Step 2: Run all built-in configuration smoke tests**

For each built-in (`hydro_quebec`, `flat`, `time_of_use`, `time_series`), construct `SustainDC`, reset, step three times, and assert:

- finite observations/rewards;
- observation widths 31/19/18;
- pricing forecast length 8;
- `total_price_cost_this_step_c` equals component sum.

- [ ] **Step 3: Run custom import-path smoke test**

Construct `PriceManager` using `tests.fixtures.custom_pricing_plugin:ValidPlugin`; reset/step/observe and assert finite standardized outputs.

- [ ] **Step 4: Run HARL training smoke test**

```bash
timeout 220 $PY train_sustaindc.py --algo happo --exp_name pricing_plugin_final \
  --n_rollout_threads 1 --n_eval_rollout_threads 1 \
  --episode_length 25 --num_env_steps 50
```

Expected: exit 0, no traceback.

- [ ] **Step 5: Review diffs and historical-reference policy**

```bash
git diff --check
grep -RInE '\bv1\b|\bv2a?\b|\bv2b\b|\bv3\b' \
  --include='*.py' --include='*.md' --include='*.rst' utils/pricing tests sphinx/usage
```

Expected: no implementation-docstring references to earlier simulator versions; unrelated external version strings are reviewed manually.

- [ ] **Step 6: Update parent submodule pointer and verify clean state**

From parent repository:

```bash
git add dc-rl CLAUDE.md
git status --short
git commit -m "feat: adopt pluggable SustainDC pricing models"
```

Confirm both simulator and parent working trees are clean, or list unrelated pre-existing changes explicitly.
