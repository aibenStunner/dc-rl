"""Tests for the offline continuous-power battery feasibility bound."""
import csv
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests._harness import run_module_tests
from utils.offline_battery_oracle import replay_tariff, solve_peak_shaving_oracle


def test_oracle_shaves_peak_and_restores_cell_energy():
    result = solve_peak_shaving_oracle(
        base_load_kwh=np.array([5.0, 15.0, 5.0, 5.0]),
        pv_kwh=np.zeros(4),
        capacity_mwh=0.010,
        initial_soc=0.5,
        round_trip_efficiency=1.0,
        rated_power_mw=0.020,
        step_hours=0.25,
    )
    assert math.isclose(result.grid_peak_kw, 40.0, abs_tol=1e-6)
    assert math.isclose(result.cell_energy_mwh[0], 0.005, abs_tol=1e-9)
    assert math.isclose(result.cell_energy_mwh[-1], 0.005, abs_tol=1e-9)
    assert np.all(result.charge_bus_kwh <= 5.0 + 1e-8)
    assert np.all(result.discharge_bus_kwh <= 5.0 + 1e-8)
    assert np.all(result.grid_import_kwh >= -1e-8)
    assert np.all(result.charge_bus_kwh * result.discharge_bus_kwh <= 1e-6)
    assert math.isclose(max(result.grid_import_kwh), 10.0, abs_tol=1e-6)


def test_oracle_allows_pv_surplus_curtailment_without_forced_battery_cycling():
    result = solve_peak_shaving_oracle(
        base_load_kwh=np.array([1.0, 1.0, 1.0, 1.0]),
        pv_kwh=np.array([2.0, 0.0, 0.0, 0.0]),
        capacity_mwh=0.010,
        initial_soc=1.0,
        round_trip_efficiency=1.0,
        rated_power_mw=0.020,
        step_hours=0.25,
    )
    assert math.isclose(result.pv_curtailed_kwh[0], 1.0, abs_tol=1e-6)
    assert math.isclose(sum(result.charge_bus_kwh), 0.0, abs_tol=1e-6)
    assert math.isclose(sum(result.discharge_bus_kwh), 0.0, abs_tol=1e-6)
    assert np.all(result.grid_import_kwh >= -1e-8)


def test_oracle_rejects_inconsistent_series_lengths():
    try:
        solve_peak_shaving_oracle(
            base_load_kwh=np.array([1.0]),
            pv_kwh=np.array([0.0, 1.0]),
            capacity_mwh=1.0,
            initial_soc=0.5,
            round_trip_efficiency=0.9,
            rated_power_mw=1.0,
            step_hours=0.25,
        )
    except ValueError as error:
        assert "same nonzero length" in str(error)
    else:
        raise AssertionError("expected inconsistent series lengths to fail")


def test_saved_battery_sizing_supplies_omitted_optional_fields():
    from utils.offline_battery_oracle import resolve_saved_battery_sizing

    rated_power_mw, capacity_mwh = resolve_saved_battery_sizing(
        {
            "datacenter_capacity_mw": 10,
            "facility_sizing": {"mode": "scaling"},
            "battery": {"power_fraction": 1.0, "duration_hours": 2.0},
        }
    )
    assert rated_power_mw == 10.0
    assert capacity_mwh == 20.0


def test_trace_file_oracle_records_source_and_baseline_metrics():
    from utils.offline_battery_oracle import solve_trace_file_oracle

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        trace_path = root / "trace.csv"
        config_path = root / "config.json"
        fields = ["battery/bat_dc_load_kwh", "pv/pv_ac_kwh", "billed_day", "billed_hour", "battery/bat_grid_import_kwh"]
        with trace_path.open("w", newline="") as trace_file:
            writer = csv.DictWriter(trace_file, fieldnames=fields)
            writer.writeheader()
            for step in range(96):
                writer.writerow({
                    "battery/bat_dc_load_kwh": "1.0",
                    "pv/pv_ac_kwh": "0.0",
                    "billed_day": "0",
                    "billed_hour": str(step / 4),
                    "battery/bat_grid_import_kwh": "1.0",
                })
        config_path.write_text(json.dumps({"env_args": {
            "datacenter_capacity_mw": 1,
            "facility_sizing": {"mode": "scaling"},
            "battery": {"power_fraction": 1.0, "duration_hours": 2.0, "initial_soc": 0.5, "round_trip_efficiency": 0.9},
            "pricing": {"model": "hydro_quebec", "config_file": "data/Pricing/hydro_quebec_2026.yaml", "options": {"tariff": "auto", "demand_floor_kw": 0.0}},
        }}))
        summary = solve_trace_file_oracle(trace_path=trace_path, config_path=config_path, repo_root=_REPO_ROOT)
    assert summary["source_trace"] == str(trace_path)
    assert summary["baseline_grid_peak_kw"] == 4.0
    assert summary["baseline_tariff_total_cost_c"] > 0.0
    assert summary["oracle_minus_baseline_tariff_total_cost_c"] == summary["oracle_tariff_total_cost_c"] - summary["baseline_tariff_total_cost_c"]
    assert summary["oracle_final_cell_energy_mwh"] == summary["oracle_initial_cell_energy_mwh"]
    assert summary["oracle_pv_curtailed_kwh"] >= 0.0
    assert summary["tariff_replayed"] is True


def test_trace_oracle_requires_complete_quarter_hour_day():
    from utils.offline_battery_oracle import solve_trace_oracle

    rows = [
        {
            "battery/bat_dc_load_kwh": "1.0",
            "pv/pv_ac_kwh": "0.0",
            "billed_day": "0",
            "billed_hour": str(step / 4),
        }
        for step in range(95)
    ]
    try:
        solve_trace_oracle(
            rows=rows,
            pricing_config=None,
            datacenter_capacity_mw=1.0,
            capacity_mwh=1.0,
            initial_soc=0.5,
            round_trip_efficiency=0.9,
            rated_power_mw=1.0,
        )
    except ValueError as error:
        assert "96 quarter-hour intervals" in str(error)
    else:
        raise AssertionError("expected incomplete day to fail")


def test_tariff_replay_reconciles_components_and_meter_peak():
    from utils.managers.pricing.loader import load_pricing_config

    pricing_config = load_pricing_config(
        {
            "pricing": {
                "model": "hydro_quebec",
                "config_file": "data/Pricing/hydro_quebec_2026.yaml",
                "options": {"tariff": "auto", "demand_floor_kw": 0.0},
            }
        },
        _REPO_ROOT,
    )
    replay = replay_tariff(
        grid_import_kwh=np.array([10.0, 20.0, 15.0]),
        day_of_year=np.array([0, 0, 0]),
        hour=np.array([0.0, 0.25, 0.5]),
        pricing_config=pricing_config,
        datacenter_capacity_mw=1.0,
        timestep_minutes=15,
    )
    assert math.isclose(replay.grid_peak_kw, 80.0, abs_tol=1e-9)
    assert math.isclose(replay.total_cost_c, replay.energy_cost_c + replay.demand_cost_c + replay.additional_cost_c, abs_tol=1e-9)
    assert replay.total_cost_c > 0.0


if __name__ == "__main__":
    sys.exit(1 if run_module_tests(globals()) else 0)
