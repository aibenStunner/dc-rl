# Pluggable Pricing Models Design

**Date:** 2026-10-07  
**Status:** Approved design, pending implementation  
**Scope:** Replace the Hydro-Québec-specific pricing module with a general pricing-model subsystem while preserving current billing behavior and the agents' observation width.

## 1. Purpose

SustainDC already treats workload, weather, and carbon intensity as configurable inputs. Pricing should have the same status: a user selects a built-in pricing model or supplies a Python plug-in, while `SustainDC` consumes one stable observation and billing interface.

The refactor must separate four responsibilities:

1. **PriceManager orchestration** — lifecycle, meter input, time context, carry-over state, standardized outputs.
2. **Pricing-model logic** — how a tariff calculates energy, demand, and additional charges.
3. **Parameter data** — tariff numbers, schedules, source metadata, and effective dates in YAML.
4. **Reward logic** — consumes standardized billing outputs but does not implement tariff rules.

## 2. Goals

- Select a built-in model with a short registry name.
- Load an external model with a Python import path such as `my_package.pricing:CampusTariff`.
- Move official tariff figures out of Python and into inspectable, cited YAML files.
- Preserve Rate M, Rate L, and the proposed large-data-centre scenario behavior.
- Support flat, time-of-use, and time-series pricing as reusable built-ins.
- Keep the agents' three pricing observation values and their positions unchanged.
- Expose model-neutral cost fields to reward functions and `info` dictionaries.
- Fail early and clearly for malformed configuration, unknown model names, invalid plug-ins, or missing files.

## 3. Non-goals

- Reward-weight tuning or redesigning the three agents' reward alignment.
- Changing observation-vector width or HARL model architecture.
- Implementing high-voltage supply credits, transformation losses, taxes, or account fees.
- Replacing the existing weather, workload, or carbon managers.
- Maintaining the old flat pricing keys. This is an explicit clean break.

## 4. Configuration contract

The environment has exactly one top-level `pricing` mapping:

```yaml
pricing:
  model: hydro_quebec
  config_file: data/Pricing/hydro_quebec_2026.yaml
  options:
    tariff: auto
    demand_floor_kw: 0.0
```

### 4.1 Allowed keys

`pricing` accepts exactly:

- `model` — required string. Either a built-in registry name or an import path in `module:attribute` form.
- `config_file` — required string. Absolute path or path relative to the `dc-rl` repository root.
- `options` — optional mapping, default `{}`. Runtime selections that are not tariff-source data.

Unknown keys are rejected. The removed top-level keys `tariff_rate_override` and `demand_floor_kw` are rejected with an error pointing to the new nested schema.

### 4.2 Built-in names

- `hydro_quebec`
- `flat`
- `time_of_use`
- `time_series`

Any string containing `:` is interpreted as an import path. Other unknown strings fail with the built-in names listed in the error.

### 4.3 Import-path plug-ins

Example:

```yaml
pricing:
  model: my_package.pricing:CampusTariff
  config_file: data/Pricing/campus_tariff.yaml
  options:
    account: research_site
```

The loader imports `my_package.pricing`, retrieves `CampusTariff`, instantiates it with the parsed parameter mapping, and validates the instance against the `PricingModel` protocol before the environment starts.

No plug-in directory scanning or dynamic registration is used. Import paths are explicit, reproducible, and visible in saved experiment configuration.

## 5. Package layout

```text
utils/pricing/
├── __init__.py              public API only
├── contracts.py             dataclasses and PricingModel protocol
├── loader.py                config-file loading, validation, registry/import-path resolution
├── manager.py               PriceManager orchestration and standard getters
└── models/
    ├── __init__.py          built-in registry
    ├── flat.py              constant energy price
    ├── time_of_use.py       recurring schedule-based energy prices
    ├── time_series.py       CSV-backed exogenous price series
    └── hydro_quebec.py      Rate M, Rate L, pending data-centre scenario

data/Pricing/
├── hydro_quebec_2026.yaml
├── flat_example.yaml
├── time_of_use_example.yaml
└── time_series_example.yaml
```

The existing `utils/price_manager.py` is removed after imports are migrated. There is one implementation of `PriceManager`, in `utils/pricing/manager.py`.

## 6. Core contracts

### 6.1 Context and readings

```python
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
```

`PriceManager` derives `demand_kw` from the 15-minute energy reading only once, then passes both values to the model.

### 6.2 Manager-owned state

```python
@dataclass
class PricingState:
    billing_period_energy_kwh: float = 0.0
    running_peak_kw: float = 0.0
    model_state: dict[str, object] = field(default_factory=dict)
```

`PriceManager` owns the `PricingState` instance. Models may update it, but callers never manipulate model internals directly.

### 6.3 Standardized outputs

```python
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
    def total_cost_c(self) -> float: ...
```

`billing_progress` is model-defined but always finite and in `[0,1]`:

- Rate M: progress toward the monthly tier threshold.
- Flat and Rate L: `0.0`.
- Time-of-use: progress through the current schedule period.
- Time-series: progress through the loaded series/billing period.

`additional_cost_increment_c` carries model-specific charges such as Rate L's winter optimization charge without leaking that tariff name into the environment or reward contract.

### 6.4 PricingModel protocol

```python
@runtime_checkable
class PricingModel(Protocol):
    name: str

    def reset(
        self,
        *,
        context: PricingContext,
        state: PricingState,
        clock: PricingClock,
        carry_state: Mapping[str, object] | None,
        options: Mapping[str, object],
    ) -> None: ...

    def step(
        self,
        *,
        context: PricingContext,
        state: PricingState,
        clock: PricingClock,
        reading: MeterReading,
    ) -> PricingCharges: ...

    def observe(
        self,
        *,
        context: PricingContext,
        state: PricingState,
        clock: PricingClock,
    ) -> PricingObservation: ...

    def export_carry_state(self, state: PricingState) -> Mapping[str, object]: ...
```

Built-ins and external plug-ins use this exact protocol.

## 7. PriceManager behavior

`PriceManager`:

1. Validates the nested `pricing` configuration.
2. Resolves the model through the built-in registry or import path.
3. Loads the YAML with `yaml.safe_load`.
4. Creates `PricingContext`, `PricingState`, and the model instance.
5. On `reset`, passes the new clock and previous carry state to the model.
6. On `step`, validates non-negative finite meter values, creates `MeterReading`, calls the model, stores `PricingCharges`, then refreshes `PricingObservation`.
7. Exposes model-neutral getters used by `SustainDC` and rewards.

Public getters:

```python
get_current_price() -> float                 # normalized
get_forecast_price() -> np.ndarray           # normalized
get_current_price_denorm() -> float          # cents/kWh
get_billing_progress_fraction() -> float
get_normalized_peak() -> float
get_energy_cost_this_step_c() -> float
get_demand_charge_increment_c() -> float
get_additional_charge_increment_c() -> float
get_total_cost_this_step_c() -> float
export_carry_state() -> Mapping[str, object]
```


## 8. Built-in models

### 8.1 HydroQuébec

The YAML contains source metadata, effective date, selection threshold, and parameter mappings for:

- Rate M
- Rate L
- Proposed large-data-centre scenario

Runtime option:

```yaml
options:
  tariff: auto       # auto | rate_m | rate_l | new_dc_rate
  demand_floor_kw: 0.0
```

`auto` selects Rate M below 5 MW and Rate L at or above 5 MW. The proposed scenario is never automatically selected.

The model preserves:

- Rate M two-tier energy pricing and exact threshold-straddling arithmetic.
- Rate M demand charge and observed winter-peak ratchet.
- Rate L flat energy price and minimum contract power.
- Rate L winter daily optimization charge and monthly cap.
- Existing cross-episode carry semantics.

The pending scenario remains explicitly marked unapproved and approximate in YAML metadata.

### 8.2 Flat

Parameter file:

```yaml
schema_version: 1
model: flat
metadata:
  name: Example flat rate
  currency: CAD
energy_price_c_per_kwh: 5.0
```

Energy-only model. Demand and additional charges are zero.

### 8.3 Time of use

Parameter file contains a 24-entry hourly schedule in cents/kWh. Every value must be finite and non-negative. The model maps fractional hours to the containing hour and forecasts the next `future_steps` 15-minute slots with cyclic day wrap.

### 8.4 Time series

Parameter file identifies a CSV, price column, source interval, and wrap behavior. The model loads and validates the series once, resamples/interpolates to the simulation interval, and advances from the episode's day/hour. This model is exogenous: demand and additional charges are zero.

## 9. SustainDC integration

`SustainDC` reads only `env_config["pricing"]` and constructs `PriceManager` with the facility capacity and standard horizon.

On reset:

```python
carry_state = self.price_m.export_carry_state()
self.price_m.reset(
    init_day=random_init_day,
    init_hour=random_init_hour,
    carry_state=carry_state,
)
```

On step, after battery operation determines meter energy:

```python
self.price_m.step(
    metered_energy_kwh=self.bat_info["bat_total_energy_with_battery_KWh"],
    day_of_year=day,
    hour=hour,
)
```

`SustainDC` does not import or call Hydro-Québec calendar helpers. Seasonal logic belongs to the selected model.

## 10. Stable observation contract

The three pricing observation values remain in the same positions and the vector widths do not change:

1. normalized current energy price
2. model-neutral billing progress
3. normalized running peak

`_price_feature_block()` changes only from `get_tier2_progress_fraction()` to `get_billing_progress_fraction()`.

Checkpoint input dimensions remain compatible with the current price-feature version of the environment.

## 11. Stable reward and info contract

Reward code receives:

- `energy_cost_this_step_c`
- `demand_charge_increment_c`
- `additional_charge_increment_c`
- `total_price_cost_this_step_c`
- `norm_price`
- `price_denorm_c_per_kwh`

`default_price_reward` uses `total_price_cost_this_step_c` and does not know which model produced it.

The old tariff-specific key `optimization_charge_increment_c` is removed in this clean break.

## 12. Validation and errors

Construction fails before training starts when:

- `pricing` is absent or not a mapping.
- Required keys are absent.
- Unknown top-level pricing keys are present.
- Old flat pricing keys are present.
- A config file is missing, unreadable, empty, or not a mapping.
- `schema_version` is unsupported.
- The YAML `model` does not match the selected built-in model.
- A registry name is unknown.
- An import path is malformed, cannot be imported, or resolves to an object that does not satisfy `PricingModel`.
- Prices, thresholds, capacities, or meter readings are negative or non-finite.
- Forecast output has the wrong length or any standardized observation leaves `[0,1]`.

Errors name the failing field and source file/import path.

## 13. Tests

Tests remain standalone-runnable with the `thesis-dcrl` interpreter and pytest-compatible.

Coverage includes:

- Parameter file parsing and strict schema validation.
- Built-in registry and import-path loading.
- Malformed plug-in rejection.
- Rate M/Rate L/proposed-scenario regression behavior.
- Flat, TOU, and time-series behavior including forecast wrap.
- Meter-input validation.
- Carry-state export/reset.
- Observation width and field-position stability.
- Reward parity: total cost equals the sum of the three standardized charge categories.
- A short `SustainDC` construction/reset/step smoke test with the new nested config.

## 14. Documentation

- Add a pricing guide showing built-in selection, custom import-path use, parameter-file format, units, carry state, and limitations.
- Update the environment config documentation and tests README.
- Keep implementation docstrings self-contained; do not compare against earlier simulator versions.
