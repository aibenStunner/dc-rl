"""Offline continuous-power battery peak-minimization feasibility bound.

This module is deliberately non-deployable: it uses an entire known facility/PV
trajectory to solve a continuous dispatch bound. It does not select SustainDC's
discrete battery action and must not be used for online control or training.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from utils.managers.pricing.loader import PricingConfig, load_pricing_config
from utils.managers.pricing.manager import PriceManager


@dataclass(frozen=True)
class OfflineBatteryOracleResult:
    """Continuous dispatch and cell-energy trajectory for one fixed horizon."""

    charge_bus_kwh: np.ndarray
    discharge_bus_kwh: np.ndarray
    grid_import_kwh: np.ndarray
    pv_curtailed_kwh: np.ndarray
    cell_energy_mwh: np.ndarray
    grid_peak_kw: float


@dataclass(frozen=True)
class TariffReplayResult:
    """Reconciled charges from replaying a fixed grid-meter trajectory."""

    energy_cost_c: float
    demand_cost_c: float
    additional_cost_c: float
    total_cost_c: float
    grid_peak_kw: float


@dataclass(frozen=True)
class TraceOracleResult:
    """Continuous bound and tariff replay for one complete matched trace."""

    oracle: OfflineBatteryOracleResult
    tariff: TariffReplayResult | None


def solve_peak_shaving_oracle(
    *,
    base_load_kwh: np.ndarray,
    pv_kwh: np.ndarray,
    capacity_mwh: float,
    initial_soc: float,
    round_trip_efficiency: float,
    rated_power_mw: float,
    step_hours: float,
) -> OfflineBatteryOracleResult:
    """Minimize grid-import peak with equal terminal cell energy.

    Inputs are fixed exogenous interval energies. Charge/discharge are bus-side
    energies; state is cell-side MWh. The linear program permits continuous
    power and is therefore an offline feasibility upper bound, not an action
    policy. It forbids export by constraining grid import to be non-negative.
    """
    load = _energy_series(base_load_kwh, "base_load_kwh")
    pv = _energy_series(pv_kwh, "pv_kwh")
    if load.shape != pv.shape:
        raise ValueError("base_load_kwh and pv_kwh must have the same nonzero length")
    if not math.isfinite(capacity_mwh) or capacity_mwh <= 0:
        raise ValueError("capacity_mwh must be finite and positive")
    if not math.isfinite(initial_soc) or not 0 <= initial_soc <= 1:
        raise ValueError("initial_soc must be finite in [0, 1]")
    if not math.isfinite(round_trip_efficiency) or not 0 < round_trip_efficiency <= 1:
        raise ValueError("round_trip_efficiency must be finite in (0, 1]")
    if not math.isfinite(rated_power_mw) or rated_power_mw <= 0:
        raise ValueError("rated_power_mw must be finite and positive")
    if not math.isfinite(step_hours) or step_hours <= 0:
        raise ValueError("step_hours must be finite and positive")

    steps = len(load)
    eta = math.sqrt(round_trip_efficiency)
    initial_cell_mwh = capacity_mwh * initial_soc
    max_bus_kwh = rated_power_mw * step_hours * 1000.0
    # Variable blocks: charge/discharge bus kWh, cell energy after each
    # interval (MWh), grid import kWh, PV curtailment kWh, and peak kWh.
    charge = np.arange(steps)
    discharge = np.arange(steps, 2 * steps)
    cell = np.arange(2 * steps, 3 * steps)
    grid = np.arange(3 * steps, 4 * steps)
    curtailed_pv = np.arange(4 * steps, 5 * steps)
    peak = 5 * steps
    variable_count = peak + 1

    objective = np.zeros(variable_count)
    objective[peak] = 1.0
    lower = np.zeros(variable_count)
    upper = np.full(variable_count, np.inf)
    upper[charge] = max_bus_kwh
    upper[discharge] = max_bus_kwh
    upper[curtailed_pv] = pv
    lower[cell], upper[cell] = 0.0, capacity_mwh

    # Each interval creates three sparse constraints, plus terminal cell
    # energy. Dense matrices would occupy gigabytes for a complete month.
    matrix = lil_matrix((3 * steps + 1, variable_count), dtype=float)
    lows = np.empty(3 * steps + 1)
    highs = np.empty(3 * steps + 1)
    for t in range(steps):
        grid_row = 3 * t
        # PV serves load first; surplus may be curtailed under the configured
        # no-export policy: grid = load - PV + curtail + charge - discharge.
        matrix[grid_row, grid[t]] = 1.0
        matrix[grid_row, charge[t]] = -1.0
        matrix[grid_row, discharge[t]] = 1.0
        matrix[grid_row, curtailed_pv[t]] = -1.0
        lows[grid_row] = highs[grid_row] = load[t] - pv[t]

        cell_row = grid_row + 1
        # cell[t] = cell[t-1] + eta * charge - discharge / eta
        matrix[cell_row, cell[t]] = 1.0
        matrix[cell_row, charge[t]] = -eta / 1000.0
        matrix[cell_row, discharge[t]] = 1.0 / (eta * 1000.0)
        if t == 0:
            lows[cell_row] = highs[cell_row] = initial_cell_mwh
        else:
            matrix[cell_row, cell[t - 1]] = -1.0
            lows[cell_row] = highs[cell_row] = 0.0

        peak_row_index = grid_row + 2
        # Peak is at least every grid interval.
        matrix[peak_row_index, peak] = 1.0
        matrix[peak_row_index, grid[t]] = -1.0
        lows[peak_row_index], highs[peak_row_index] = 0.0, np.inf

    # Equal end/start cell energy, not an unmetered post-hoc settlement.
    matrix[-1, cell[-1]] = 1.0
    lows[-1] = highs[-1] = initial_cell_mwh

    constraints = LinearConstraint(matrix.tocsr(), lows, highs)
    solved = milp(
        c=objective,
        integrality=np.zeros(variable_count, dtype=int),
        bounds=Bounds(lower, upper),
        constraints=constraints,
        options={"disp": False},
    )
    if not solved.success or solved.x is None:
        raise RuntimeError(f"offline battery oracle failed: {solved.message}")

    # Peak is the primary objective. With its optimum fixed, minimize imported
    # energy to avoid arbitrary cycling in the continuous feasibility bound.
    peak_limit = float(solved.x[peak]) + 1e-7
    peak_row = lil_matrix((1, variable_count), dtype=float)
    peak_row[0, peak] = 1.0
    energy_objective = np.zeros(variable_count)
    energy_objective[grid] = 1.0
    energy_solved = milp(
        c=energy_objective,
        integrality=np.zeros(variable_count, dtype=int),
        bounds=Bounds(lower, upper),
        constraints=(constraints, LinearConstraint(peak_row, [-np.inf], [peak_limit])),
        options={"disp": False},
    )
    if not energy_solved.success or energy_solved.x is None:
        raise RuntimeError(f"offline battery oracle energy refinement failed: {energy_solved.message}")
    values = energy_solved.x
    charge_kwh = np.maximum(0.0, values[charge])
    discharge_kwh = np.maximum(0.0, values[discharge])
    grid_kwh = np.maximum(0.0, values[grid])
    curtailed_pv_kwh = np.maximum(0.0, values[curtailed_pv])
    cell_energy = np.concatenate(([initial_cell_mwh], values[cell]))
    return OfflineBatteryOracleResult(
        charge_bus_kwh=charge_kwh,
        discharge_bus_kwh=discharge_kwh,
        grid_import_kwh=grid_kwh,
        pv_curtailed_kwh=curtailed_pv_kwh,
        cell_energy_mwh=cell_energy,
        grid_peak_kw=float(values[peak] / step_hours),
    )


def solve_trace_file_oracle(
    *,
    trace_path: Path,
    config_path: Path,
    repo_root: Path,
) -> dict[str, object]:
    """Solve a saved matched trace and return serializable provenance metrics."""
    with Path(trace_path).open(newline="", encoding="utf-8") as trace_file:
        rows = list(csv.DictReader(trace_file))
    try:
        saved_config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        env_config = saved_config["env_args"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("config_path must contain a saved configuration with env_args") from exc
    if not isinstance(env_config, dict):
        raise ValueError("saved env_args must be a mapping")
    pricing_config = load_pricing_config({"pricing": env_config["pricing"]}, Path(repo_root))
    rated_power_mw, capacity_mwh = resolve_saved_battery_sizing(env_config)
    battery = env_config.get("battery")
    if not isinstance(battery, dict):
        raise ValueError("saved battery configuration must be a mapping")
    datacenter_capacity_mw = float(env_config["datacenter_capacity_mw"])
    result = solve_trace_oracle(
        rows=rows,
        pricing_config=pricing_config,
        datacenter_capacity_mw=datacenter_capacity_mw,
        capacity_mwh=capacity_mwh,
        initial_soc=float(battery["initial_soc"]),
        round_trip_efficiency=float(battery["round_trip_efficiency"]),
        rated_power_mw=rated_power_mw,
    )
    baseline_grid = _trace_numeric_column(rows, "battery/bat_grid_import_kwh")
    baseline_days, baseline_hours = _trace_clock_columns(rows)
    baseline_tariff = replay_tariff(
        grid_import_kwh=baseline_grid,
        day_of_year=baseline_days,
        hour=baseline_hours,
        pricing_config=pricing_config,
        datacenter_capacity_mw=datacenter_capacity_mw,
        timestep_minutes=15,
    )
    summary: dict[str, object] = {
        "source_trace": str(trace_path),
        "saved_config": str(config_path),
        "steps": len(rows),
        "baseline_grid_peak_kw": baseline_tariff.grid_peak_kw,
        "baseline_tariff_energy_cost_c": baseline_tariff.energy_cost_c,
        "baseline_tariff_demand_cost_c": baseline_tariff.demand_cost_c,
        "baseline_tariff_additional_cost_c": baseline_tariff.additional_cost_c,
        "baseline_tariff_total_cost_c": baseline_tariff.total_cost_c,
        "oracle_grid_peak_kw": result.oracle.grid_peak_kw,
        "oracle_initial_cell_energy_mwh": float(result.oracle.cell_energy_mwh[0]),
        "oracle_final_cell_energy_mwh": float(result.oracle.cell_energy_mwh[-1]),
        "oracle_charge_bus_kwh": float(np.sum(result.oracle.charge_bus_kwh)),
        "oracle_discharge_bus_kwh": float(np.sum(result.oracle.discharge_bus_kwh)),
        "oracle_pv_curtailed_kwh": float(np.sum(result.oracle.pv_curtailed_kwh)),
        "tariff_replayed": result.tariff is not None,
    }
    if result.tariff is not None:
        summary.update({
            "oracle_tariff_grid_peak_kw": result.tariff.grid_peak_kw,
            "oracle_tariff_energy_cost_c": result.tariff.energy_cost_c,
            "oracle_tariff_demand_cost_c": result.tariff.demand_cost_c,
            "oracle_tariff_additional_cost_c": result.tariff.additional_cost_c,
            "oracle_tariff_total_cost_c": result.tariff.total_cost_c,
            "oracle_minus_baseline_tariff_total_cost_c": (
                result.tariff.total_cost_c - baseline_tariff.total_cost_c
            ),
        })
    return summary


def _trace_clock_columns(rows: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    try:
        days = np.asarray([int(row["billed_day"]) for row in rows], dtype=int)
        hours = np.asarray([float(row["billed_hour"]) for row in rows])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("trace must contain valid billed_day and billed_hour values") from exc
    _validate_quarter_hour_calendar(days, hours)
    return days, hours


def _trace_numeric_column(rows: list[dict[str, object]], key: str) -> np.ndarray:
    try:
        values = np.asarray([float(row[key]) for row in rows])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"trace must contain finite {key} values") from exc
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"trace must contain finite non-negative {key} values")
    return values


def resolve_saved_battery_sizing(env_config: dict[str, object]) -> tuple[float, float]:
    """Resolve a saved scaling BESS, accepting absent historical null fields."""
    facility = env_config.get("facility_sizing")
    battery = env_config.get("battery")
    if not isinstance(facility, dict) or facility.get("mode") != "scaling":
        raise ValueError("only scaling facility sizing is supported")
    if not isinstance(battery, dict):
        raise ValueError("battery configuration must be a mapping")
    try:
        datacenter_capacity_mw = float(env_config["datacenter_capacity_mw"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("datacenter_capacity_mw must be available") from exc
    rated_power = battery.get("rated_power_mw")
    if rated_power is None:
        fraction = battery.get("power_fraction")
        rated_power = datacenter_capacity_mw * (1.0 if fraction is None else float(fraction))
    capacity = battery.get("capacity_mwh")
    if capacity is None:
        duration = battery.get("duration_hours")
        capacity = float(rated_power) * (2.0 if duration is None else float(duration))
    if not math.isfinite(float(rated_power)) or float(rated_power) <= 0:
        raise ValueError("resolved rated battery power must be finite and positive")
    if not math.isfinite(float(capacity)) or float(capacity) <= 0:
        raise ValueError("resolved battery capacity must be finite and positive")
    return float(rated_power), float(capacity)


def solve_trace_oracle(
    *,
    rows: list[dict[str, object]],
    pricing_config: PricingConfig | None,
    datacenter_capacity_mw: float,
    capacity_mwh: float,
    initial_soc: float,
    round_trip_efficiency: float,
    rated_power_mw: float,
) -> TraceOracleResult:
    """Solve a complete matched trace and optionally replay its tariff meter.

    This consumes only fixed pre-battery load, PV, and billing clocks from the
    trace. Every day must contain exactly the 96 sequential 15-minute interval
    starts used by SustainDC, preventing partial diagnostics from appearing as
    complete monthly bills.
    """
    if not rows:
        raise ValueError("rows must be nonempty")
    required = ("battery/bat_dc_load_kwh", "pv/pv_ac_kwh", "billed_day", "billed_hour")
    try:
        load = np.asarray([float(row[key]) for row in rows for key in [required[0]]])
        pv = np.asarray([float(row[key]) for row in rows for key in [required[1]]])
        days = np.asarray([int(row[key]) for row in rows for key in [required[2]]], dtype=int)
        hours = np.asarray([float(row[key]) for row in rows for key in [required[3]]])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("trace must contain finite load, PV, day, and hour fields") from exc
    _validate_quarter_hour_calendar(days, hours)
    oracle = solve_peak_shaving_oracle(
        base_load_kwh=load,
        pv_kwh=pv,
        capacity_mwh=capacity_mwh,
        initial_soc=initial_soc,
        round_trip_efficiency=round_trip_efficiency,
        rated_power_mw=rated_power_mw,
        step_hours=0.25,
    )
    tariff = None
    if pricing_config is not None:
        tariff = replay_tariff(
            grid_import_kwh=oracle.grid_import_kwh,
            day_of_year=days,
            hour=hours,
            pricing_config=pricing_config,
            datacenter_capacity_mw=datacenter_capacity_mw,
            timestep_minutes=15,
        )
        if not math.isclose(tariff.grid_peak_kw, oracle.grid_peak_kw, rel_tol=1e-7, abs_tol=1e-5):
            raise RuntimeError("tariff replay grid peak does not match oracle meter peak")
    return TraceOracleResult(oracle=oracle, tariff=tariff)


def _validate_quarter_hour_calendar(days: np.ndarray, hours: np.ndarray) -> None:
    if days.ndim != 1 or hours.ndim != 1 or len(days) != len(hours):
        raise ValueError("day_of_year and hour must be same-length one-dimensional arrays")
    if len(days) % 96:
        raise ValueError("trace must contain complete days of 96 quarter-hour intervals")
    expected_hours = np.arange(96, dtype=float) / 4.0
    for start in range(0, len(days), 96):
        day_hours = hours[start:start + 96]
        if not np.allclose(day_hours, expected_hours, rtol=0.0, atol=1e-9):
            raise ValueError("each day must contain 96 quarter-hour intervals in order")
        if not np.all(days[start:start + 96] == days[start]):
            raise ValueError("each complete day must have one billed_day")
        if start and days[start] != days[start - 1] + 1:
            raise ValueError("billed_day must increase consecutively across complete days")


def replay_tariff(
    *,
    grid_import_kwh: np.ndarray,
    day_of_year: np.ndarray,
    hour: np.ndarray,
    pricing_config: PricingConfig,
    datacenter_capacity_mw: float,
    timestep_minutes: int,
) -> TariffReplayResult:
    """Replay a nonnegative interval meter series through the configured tariff.

    The replay is accounting only: it does not turn continuous oracle flows into
    SustainDC discrete actions. Clocks are the interval-start clocks recorded by
    a matched trace; a one-shot settlement includes deferred model charges.
    """
    energy = _energy_series(grid_import_kwh, "grid_import_kwh")
    days = _integer_series(day_of_year, "day_of_year")
    hours = np.asarray(hour, dtype=float)
    if days.shape != energy.shape or hours.ndim != 1 or hours.shape != energy.shape:
        raise ValueError("grid_import_kwh, day_of_year, and hour must have the same nonzero length")
    if not np.all(np.isfinite(hours)) or np.any((hours < 0.0) | (hours >= 24.0)):
        raise ValueError("hour must contain finite values in [0, 24)")
    manager = PriceManager.from_config(
        pricing_config,
        datacenter_capacity_mw=datacenter_capacity_mw,
        future_steps=0,
        timestep_minutes=timestep_minutes,
    )
    manager.reset(init_day=int(days[0]), init_hour=float(hours[0]))
    energy_cost_c = demand_cost_c = additional_cost_c = 0.0
    for energy_kwh, day, interval_hour in zip(energy, days, hours):
        charges = manager.step(
            metered_energy_kwh=float(energy_kwh),
            day_of_year=int(day),
            hour=float(interval_hour),
        )
        energy_cost_c += charges.energy_cost_c
        demand_cost_c += charges.demand_cost_increment_c
        additional_cost_c += charges.additional_cost_increment_c
    settlement = manager.settle_pending_period()
    energy_cost_c += settlement.energy_cost_c
    demand_cost_c += settlement.demand_cost_increment_c
    additional_cost_c += settlement.additional_cost_increment_c
    total_cost_c = energy_cost_c + demand_cost_c + additional_cost_c
    step_hours = timestep_minutes / 60.0
    return TariffReplayResult(
        energy_cost_c=energy_cost_c,
        demand_cost_c=demand_cost_c,
        additional_cost_c=additional_cost_c,
        total_cost_c=total_cost_c,
        grid_peak_kw=float(np.max(energy) / step_hours),
    )


def _integer_series(values: np.ndarray, name: str) -> np.ndarray:
    series = np.asarray(values)
    if series.ndim != 1 or not len(series) or not np.issubdtype(series.dtype, np.integer):
        raise ValueError(f"{name} must be a one-dimensional nonempty integer array")
    return series


def _energy_series(values: np.ndarray, name: str) -> np.ndarray:
    series = np.asarray(values, dtype=float)
    if series.ndim != 1 or not len(series):
        raise ValueError(f"{name} must be a one-dimensional nonempty array")
    if not np.all(np.isfinite(series)) or np.any(series < 0):
        raise ValueError(f"{name} must contain finite non-negative values")
    return series
