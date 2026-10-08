"""Physical battery-flow and report-only degradation tests."""
import math
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests._harness import run_module_tests
from envs.battery_model import Battery2
from envs.bat_env_fwd_view import BatteryEnvFwd
from utils import reward_creator
from sustaindc_env import EnvConfig, SustainDC


def _env(**overrides):
    config = {
        "n_fwd_steps": 8,
        "max_dc_pw_MW": 2.0,
        "max_bat_cap": 1.0,
        "charging_rate": 0.5,
        "start_point": 0,
        "dcload_max": 2.0,
        "dcload_min": 0.1,
        "round_trip_efficiency": 0.90,
        "degradation_cost_c_per_kwh": 5.78,
    }
    config.update(overrides)
    return BatteryEnvFwd(config)


def test_env_config_deep_merges_partial_battery_options():
    config = EnvConfig({"battery": {"round_trip_efficiency": 1.0}})
    assert config["battery"]["round_trip_efficiency"] == 1.0
    assert config["battery"]["degradation_cost_c_per_kwh"] == 5.78


def test_real_sustaindc_exposes_battery_fidelity_metrics_without_changing_observation_width():
    env = SustainDC({
        "agents": ["agent_ls", "agent_dc", "agent_bat"],
        "location": "montreal",
        "cintensity_file": "QC_NG_&_avgCI.csv",
        "workload_file": "Alibaba_CPU_Data_Hourly_1.csv",
        "datacenter_capacity_mw": 1.0,
        "month": 6,
        "days_per_episode": 1,
        "sample_whole_year": False,
        "battery": {
            "round_trip_efficiency": 0.90,
            "degradation_cost_c_per_kwh": 5.78,
        },
        "pricing": {
            "model": "hydro_quebec",
            "config_file": "data/Pricing/hydro_quebec_2026.yaml",
            "options": {"tariff": "auto", "demand_floor_kw": 0.0},
        },
    })
    env.reset()
    actions = {agent: env.action_space[index].sample()
               for index, agent in enumerate(env.agents)}
    observations, _, _, _, info = env.step(actions)
    battery_info = info["agent_bat"]
    required = {
        "bat_charge_bus_KWh", "bat_discharge_bus_KWh",
        "bat_charge_cell_KWh", "bat_discharge_cell_KWh",
        "bat_cell_throughput_KWh", "bat_cumulative_cell_throughput_KWh",
        "bat_degradation_cost_c", "bat_cumulative_degradation_cost_c",
        "bat_round_trip_efficiency", "bat_charge_efficiency",
        "bat_discharge_efficiency",
    }
    assert required <= set(battery_info)
    assert all(np.isfinite(battery_info[key]) for key in required)
    assert [observations[agent].shape[0] for agent in env.agents] == [40, 28, 27]


def test_rte_derives_symmetric_one_way_efficiency():
    battery = Battery2(capacity=1.0, round_trip_efficiency=0.90)
    assert math.isclose(battery.eff_c, math.sqrt(0.90))
    assert math.isclose(battery.eff_d, math.sqrt(0.90))
    assert math.isclose(battery.eff_c * battery.eff_d, 0.90)


def test_complete_cycle_returns_configured_round_trip_energy():
    battery = Battery2(capacity=10.0, round_trip_efficiency=0.90,
                       c_lim=100.0, d_lim=100.0)
    charge = battery.charge(requested_bus_power_mw=1.0, duration_hours=1.0)
    discharge = battery.discharge(requested_bus_power_mw=10.0,
                                  duration_hours=1.0, dc_bus_limit_mw=10.0)
    assert math.isclose(charge.charge_bus_mwh, 1.0)
    assert math.isclose(discharge.discharge_bus_mwh / charge.charge_bus_mwh, 0.90, abs_tol=1e-8)
    assert math.isclose(battery.current_load, 0.0, abs_tol=1e-8)


def test_charge_and_discharge_flows_keep_bus_and_cell_boundaries_distinct():
    battery = Battery2(capacity=2.0, round_trip_efficiency=0.90,
                       c_lim=100.0, d_lim=100.0)
    charge = battery.charge(requested_bus_power_mw=1.0, duration_hours=0.25)
    assert charge.charge_bus_mwh == 0.25
    assert math.isclose(charge.charge_cell_mwh, 0.25 * math.sqrt(0.90))
    assert math.isclose(battery.current_load, charge.charge_cell_mwh, abs_tol=1e-8)

    discharge = battery.discharge(requested_bus_power_mw=0.1, duration_hours=0.25,
                                  dc_bus_limit_mw=10.0)
    assert math.isclose(discharge.discharge_bus_mwh, 0.025)
    assert math.isclose(discharge.discharge_cell_mwh, 0.025 / math.sqrt(0.90))


def test_efficiency_aware_capacity_bounds_and_ideal_efficiency():
    battery = Battery2(capacity=1.0, current_load=0.99, round_trip_efficiency=0.90,
                       c_lim=100.0, d_lim=100.0)
    charge = battery.charge(requested_bus_power_mw=10.0, duration_hours=1.0)
    assert battery.current_load <= 1.0
    assert math.isclose(charge.charge_bus_mwh, (1.0 - 0.99) / math.sqrt(0.90))

    discharge = battery.discharge(requested_bus_power_mw=10.0, duration_hours=1.0,
                                  dc_bus_limit_mw=10.0)
    assert battery.current_load >= 0.0
    assert discharge.discharge_bus_mwh <= math.sqrt(0.90)

    ideal = Battery2(capacity=1.0, round_trip_efficiency=1.0,
                     c_lim=100.0, d_lim=100.0)
    ideal_charge = ideal.charge(requested_bus_power_mw=0.4, duration_hours=1.0)
    ideal_discharge = ideal.discharge(requested_bus_power_mw=0.4, duration_hours=1.0,
                                      dc_bus_limit_mw=1.0)
    assert ideal_charge.charge_bus_mwh == ideal_charge.charge_cell_mwh
    assert ideal_discharge.discharge_bus_mwh == ideal_discharge.discharge_cell_mwh


def test_invalid_efficiency_is_rejected():
    for value in (0.0, -0.1, 1.01, math.nan, math.inf):
        try:
            Battery2(capacity=1.0, round_trip_efficiency=value)
        except ValueError as exc:
            assert "round_trip_efficiency" in str(exc)
        else:
            raise AssertionError(f"expected {value!r} to fail")


def test_invalid_degradation_cost_is_rejected():
    for value in (-0.1, math.nan, math.inf):
        try:
            _env(degradation_cost_c_per_kwh=value)
        except ValueError as exc:
            assert "degradation_cost_c_per_kwh" in str(exc)
        else:
            raise AssertionError(f"expected {value!r} to fail")


def test_grid_meter_uses_bus_energy_and_reported_degradation_uses_cell_throughput():
    env = _env()
    env.reset()
    env.set_dcload(1.0)
    _, _, _, _, info = env.step(0)  # charge

    base_kwh = 1.0 * 1000.0 * 0.25
    assert math.isclose(info["bat_total_energy_without_battery_KWh"], base_kwh)
    assert math.isclose(info["bat_total_energy_with_battery_KWh"],
                        info["bat_grid_import_kwh"])
    assert math.isclose(info["bat_grid_import_kwh"],
                        base_kwh + info["bat_charge_bus_KWh"])
    assert info["bat_charge_bus_KWh"] > info["bat_charge_cell_KWh"]
    assert math.isclose(info["bat_cell_throughput_KWh"],
                        info["bat_charge_cell_KWh"] + info["bat_discharge_cell_KWh"])
    assert math.isclose(info["bat_degradation_cost_c"],
                        info["bat_cell_throughput_KWh"] * 5.78)

    _, _, _, _, idle_info = env.step(2)
    assert idle_info["bat_charge_bus_KWh"] == 0.0
    assert idle_info["bat_discharge_bus_KWh"] == 0.0
    assert idle_info["bat_cell_throughput_KWh"] == 0.0
    assert idle_info["bat_cumulative_cell_throughput_KWh"] == info["bat_cell_throughput_KWh"]


def test_report_only_degradation_does_not_change_grid_meter_or_reward_input():
    costly = _env(degradation_cost_c_per_kwh=5.78)
    free = _env(degradation_cost_c_per_kwh=0.0)
    for env in (costly, free):
        env.reset()
        env.set_dcload(1.0)
    _, _, _, _, costly_info = costly.step(0)
    _, _, _, _, free_info = free.step(0)
    assert math.isclose(costly_info["bat_grid_import_kwh"], free_info["bat_grid_import_kwh"])
    assert costly_info["bat_degradation_cost_c"] > 0.0
    assert free_info["bat_degradation_cost_c"] == 0.0


def test_degradation_metrics_reset_per_episode_and_do_not_change_price_reward():
    env = _env()
    env.reset()
    env.set_dcload(1.0)
    _, _, _, _, info = env.step(0)
    assert info["bat_cumulative_degradation_cost_c"] > 0.0

    reward = reward_creator.default_price_reward({"total_price_cost_this_step_c": 123.0})
    assert reward == -1.23
    assert info["bat_degradation_cost_c"] not in {123.0}

    _, reset_info = env.reset()
    assert reset_info["bat_cumulative_cell_throughput_KWh"] == 0.0
    assert reset_info["bat_cumulative_degradation_cost_c"] == 0.0


if __name__ == "__main__":
    sys.exit(1 if run_module_tests(globals()) else 0)
