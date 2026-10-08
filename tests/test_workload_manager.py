"""Behavioral tests for the workload trace manager."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests._harness import run_module_tests
from utils.managers.workload_manager import Workload_Manager


_DEFAULT = _REPO_ROOT / "data" / "Workload" / "Alibaba_CPU_Data_Hourly_1.csv"
_SECOND = _REPO_ROOT / "data" / "Workload" / "Alibaba_CPU_Data_Hourly_2.csv"


def test_default_trace_expands_hourly_source_to_quarter_hourly():
    manager = Workload_Manager()
    source = pd.read_csv(_DEFAULT)["cpu_load"].to_numpy()[:8760]
    assert len(manager.original_data) == 365 * 24 * 4
    assert manager.original_data[0] == source[0]
    assert manager.original_data[-1] == source[-1]
    expected_first_interior = source[0] + (source[1] - source[0]) * 8760 / 35039
    assert np.isclose(manager.original_data[1], expected_first_interior)


def test_named_source_and_timezone_shift_select_and_roll_series():
    default = Workload_Manager()
    named = Workload_Manager(workload_filename=_SECOND.name)
    shifted = Workload_Manager(timezone_shift=1)
    assert not np.array_equal(default.original_data, named.original_data)
    assert np.array_equal(shifted.original_data, np.roll(default.original_data, -4))


def test_reset_caches_current_next_and_cyclic_lookahead():
    manager = Workload_Manager()
    state = np.random.get_state()
    try:
        np.random.seed(7)
        returned = manager.reset(init_day=100, init_hour=12)
    finally:
        np.random.set_state(state)
    assert returned == manager.get_current_workload()
    assert manager.get_next_workload() == manager.cpu_smooth[manager.time_step + 1]
    assert np.all((manager.cpu_smooth >= 0.0) & (manager.cpu_smooth <= 1.0))
    assert np.array_equal(manager.get_n_next_workloads(4), manager.cpu_smooth[manager.time_step + 1:manager.time_step + 5])


def test_lookahead_and_step_wrap_at_annual_boundary():
    manager = Workload_Manager()
    state = np.random.get_state()
    try:
        np.random.seed(9)
        manager.reset(init_day=20, init_hour=3)
    finally:
        np.random.set_state(state)
    manager.time_step = len(manager.cpu_smooth) - 1
    assert manager.get_next_workload() == manager.cpu_smooth[manager.init_time_step + 1] or manager.get_next_workload() == manager.cpu_smooth[0]
    wrapped = manager.get_n_next_workloads(4)
    assert len(wrapped) == 4
    assert wrapped[0] == manager.cpu_smooth[0]
    manager.step()
    assert manager.time_step == manager.init_time_step


def test_set_current_workload_mutates_current_sample():
    manager = Workload_Manager()
    state = np.random.get_state()
    try:
        np.random.seed(3)
        manager.reset(init_day=50, init_hour=1)
    finally:
        np.random.set_state(state)
    manager.set_current_workload(0.42)
    assert manager.get_current_workload() != 0.42  # cached value intentionally remains stale
    assert manager.cpu_smooth[manager.time_step] == 0.42


if __name__ == "__main__":
    sys.exit(1 if run_module_tests(globals()) else 0)
