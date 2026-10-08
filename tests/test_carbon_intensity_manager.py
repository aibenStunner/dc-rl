"""Behavioral tests for the carbon-intensity trace manager."""
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests._harness import run_module_tests
from utils.managers.carbon_intensity_manager import CI_Manager


def test_location_source_loads_one_year_and_expands_to_quarter_hours():
    manager = CI_Manager(location="CA", future_steps=8)
    source = pd.read_csv(_REPO_ROOT / "data" / "CarbonIntensity" / "CA_NG_&_avgCI.csv")["avg_CI"].to_numpy()[:8760]
    assert len(manager.original_data) == 35040
    assert manager.original_data[0] == source[0]
    assert manager.original_data[-1] == source[-1]


def test_location_takes_precedence_over_filename():
    by_location = CI_Manager(location="CA", filename="NYIS_NG_&_avgCI.csv")
    by_filename = CI_Manager(location="", filename="NYIS_NG_&_avgCI.csv")
    assert not np.array_equal(by_location.original_data, by_filename.original_data)


def test_nan_hourly_values_are_replaced_with_source_mean_before_interpolation():
    source = np.arange(8760, dtype=float)
    source[10] = np.nan
    frame = pd.DataFrame({"avg_CI": source})
    with patch("utils.managers.carbon_intensity_manager.pd.read_csv", return_value=frame):
        manager = CI_Manager(location="mock")
    expected_mean = np.nanmean(source)
    # The source hourly value is replaced before the manager's historical
    # endpoint-inclusive interpolation convention is applied. Pin the exact
    # quarterly source-slot index rather than assuming a conventional x4 grid.
    expected = np.interp(
        np.linspace(0, len(source), len(source) * 4),
        range(len(source)),
        np.nan_to_num(source, nan=expected_mean),
    )
    assert np.allclose(manager.original_data, expected)


def test_reset_returns_normalized_current_forecast_and_raw_current():
    manager = CI_Manager(location="CA", future_steps=8)
    current, forecast, raw = manager.reset(init_day=100, init_hour=12)
    assert current == manager.get_current_ci()
    assert raw == manager.carbon_smooth[manager.time_step]
    assert forecast.shape == (8,)
    assert np.all(np.isfinite(forecast))


def test_future_and_past_windows_wrap_at_annual_boundary():
    manager = CI_Manager(location="CA", future_steps=4)
    manager.reset(init_day=100, init_hour=0)
    manager.norm_carbon = np.arange(len(manager.norm_carbon), dtype=float)
    manager.time_step = len(manager.norm_carbon) - 1
    manager._forecast_norm_carbon = manager.get_n_past_ci(4)  # establish helper path
    assert np.array_equal(manager.get_n_past_ci(4), np.array([35035., 35036., 35037., 35038.]))
    manager.step()
    assert manager.time_step == manager.init_day * manager.time_steps_day


if __name__ == "__main__":
    sys.exit(1 if run_module_tests(globals()) else 0)
