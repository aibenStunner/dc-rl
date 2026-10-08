"""Behavioral tests for the weather and wet-bulb trace manager."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psychrolib as psy

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests._harness import run_module_tests
from utils.managers.weather_manager import Weather_Manager


_WEATHER = "USA_CA_San.Jose-Mineta.epw"


def test_epw_loads_interpolates_and_uses_si_wet_bulb():
    manager = Weather_Manager(location=_WEATHER)
    source = pd.read_csv(_REPO_ROOT / "data" / "Weather" / _WEATHER, skiprows=8, header=None).values
    temp, rh, pressure = source[0, 6], source[0, 8], source[0, 9]
    psy.SetUnitSystem(psy.SI)
    expected_wet_bulb = psy.GetTWetBulbFromRelHum(float(temp), float(rh) / 100, float(pressure))
    assert len(manager.temperature_data) == 35040
    assert manager.temperature_data[0] == float(temp)
    assert np.isclose(manager.wet_bulb_data[0], expected_wet_bulb)


def test_timezone_shift_rolls_raw_temperature_and_wet_bulb_series():
    plain = Weather_Manager(location=_WEATHER)
    shifted = Weather_Manager(location=_WEATHER, timezone_shift=1)
    assert np.array_equal(shifted.original_temp_data, np.roll(plain.original_temp_data, -4))
    assert np.array_equal(shifted.original_wb_data, np.roll(plain.original_wb_data, -4))


def test_seeded_reset_is_reproducible_and_getters_match_outputs():
    state = np.random.get_state()
    try:
        np.random.seed(11)
        first = Weather_Manager(location=_WEATHER)
        first_out = first.reset(init_day=100, init_hour=12)
        np.random.seed(11)
        second = Weather_Manager(location=_WEATHER)
        second_out = second.reset(init_day=100, init_hour=12)
    finally:
        np.random.set_state(state)
    assert np.allclose(first_out, second_out)
    assert np.array_equal(first.temperature_data, second.temperature_data)
    assert first.get_current_temperature() == first_out[1]
    assert first.get_current_wet_bulb() == first_out[2]
    assert np.all((first.temperature_data >= 0.0) & (first.temperature_data <= 45.0))


def test_future_temperature_window_wraps_at_annual_boundary():
    manager = Weather_Manager(location=_WEATHER, debug=True)
    manager.reset(init_day=100, init_hour=0)
    manager.time_step = len(manager.norm_temp_data) - 1
    assert len(manager.get_n_next_temperature(4)) == 4
    manager.step()
    assert manager.time_step == manager.init_day * manager.time_steps_day


if __name__ == "__main__":
    sys.exit(1 if run_module_tests(globals()) else 0)
