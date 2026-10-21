"""Workload and queue behaviour under different facility capacities.

The workload trace is a normalized utilization fraction, so identical seeds
and actions must produce identical utilization and queue trajectories at any
capacity. Only the physical compute behind the same fraction changes.
"""
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests._harness import run_module_tests
from sustaindc_env import SustainDC


def _env(capacity_mw):
    return SustainDC({
        "agents": ["agent_ls", "agent_dc", "agent_bat"],
        "location": "montreal",
        "cintensity_file": "QC_NG_&_avgCI.csv",
        "workload_file": "Alibaba_CPU_Data_Hourly_1.csv",
        "datacenter_capacity_mw": capacity_mw,
        "facility_sizing": {"mode": "scaling"},
        "battery": {"power_fraction": 1.0, "duration_hours": 2.0,
                    "round_trip_efficiency": 0.90,
                    "degradation_cost_c_per_kwh": 5.78, "initial_soc": 0.5},
        "month": 6, "days_per_episode": 1, "sample_whole_year": False,
        "pricing": {"model": "hydro_quebec",
                    "config_file": "data/Pricing/hydro_quebec_2026.yaml",
                    "options": {"tariff": "auto", "demand_floor_kw": 0.0}},
    })


def _trace(capacity_mw, steps=48):
    """Collect the load-shifting trajectory under a fixed action sequence."""
    random.seed(7)
    np.random.seed(7)
    env = _env(capacity_mw)
    env.reset()
    rows = []
    for step in range(steps):
        action = 0 if step % 3 == 0 else 1     # defer on some steps, run otherwise
        _, _, _, _, info = env.step({"agent_ls": action, "agent_dc": 1, "agent_bat": 2})
        ls = info["agent_ls"]
        rows.append((
            ls["ls_original_workload"],
            ls["ls_shifted_workload"],
            ls["ls_tasks_in_queue"],
            ls["ls_tasks_dropped"],
        ))
    return env, rows


def test_normalized_workload_and_queue_are_capacity_independent():
    small_env, small = _trace(1.0)
    large_env, large = _trace(10.0)
    assert len(small) == len(large)
    for index, (a, b) in enumerate(zip(small, large)):
        assert a[0] == b[0], f"original workload differs at step {index}"
        assert a[1] == b[1], f"shifted workload differs at step {index}"
        assert a[2] == b[2], f"queue depth differs at step {index}"
        assert a[3] == b[3], f"dropped tasks differ at step {index}"


def test_physical_scale_actually_differs_between_capacities():
    small_env, _ = _trace(1.0)
    large_env, _ = _trace(10.0)
    small_meta = small_env.dc_env.sizing_metadata
    large_meta = large_env.dc_env.sizing_metadata
    assert small_meta['server_count'] < large_meta['server_count']
    assert abs(small_meta['achieved_mw'] - 1.0) / 1.0 <= 0.02
    assert abs(large_meta['achieved_mw'] - 10.0) / 10.0 <= 0.02
    # A queue task is one percentage point of installed compute capacity, so
    # the same queue length means different absolute work at each capacity.
    assert large_meta['server_count'] > 5 * small_meta['server_count']


def test_battery_scales_with_the_facility():
    small_env, _ = _trace(1.0)
    large_env, _ = _trace(10.0)
    assert small_env.bat_env.battery.capacity == 2.0
    assert large_env.bat_env.battery.capacity == 20.0
    assert small_env.bat_env.battery.rated_power_mw == 1.0
    assert large_env.bat_env.battery.rated_power_mw == 10.0


if __name__ == '__main__':
    sys.exit(1 if run_module_tests(globals()) else 0)
