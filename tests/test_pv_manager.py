"""PV manager physics and EPW integration tests."""
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests._harness import run_module_tests
from utils.managers.pv_manager import PV_Manager
from envs.bat_env_fwd_view import BatteryEnvFwd


_EPW = "CAN_QC_Montreal-Trudeau.Intl.AP.716270_TMYx.2011-2025.epw"


def _manager(**overrides):
    config = dict(
        epw_filename=_EPW,
        datacenter_capacity_mw=1.0,
        enabled=True,
        capacity_fraction_of_datacenter=0.05,
        surface_tilt_deg=35.0,
        surface_azimuth_deg=180.0,
        dc_ac_ratio=1.20,
        gamma_pdc_per_deg_c=-0.0037,
        system_losses_fraction=0.1408,
        inverter_efficiency=0.96,
        temperature_model="close_mount_glass_glass",
        timezone_shift=0,
        init_day=0,
    )
    config.update(overrides)
    return PV_Manager(**config)


def test_disabled_pv_returns_zero_current_and_forecast():
    manager = _manager(enabled=False)
    current, forecast = manager.reset(init_day=180, init_hour=12, future_steps=8)
    assert current == 0.0
    assert np.array_equal(forecast, np.zeros(8, dtype=np.float32))
    assert manager.get_current_ac_norm() == 0.0


def test_montreal_epw_metadata_and_inputs_are_finite():
    manager = _manager()
    assert manager.metadata["latitude"] == 45.4667
    assert manager.metadata["longitude"] == -73.75
    assert manager.metadata["altitude"] == 36.0
    assert len(manager.hourly_weather) == 8760
    assert np.isfinite(manager.hourly_weather[["ghi", "dni", "dhi", "temp_air", "wind_speed"]].to_numpy()).all()


def test_pv_is_zero_at_night_and_positive_on_sunny_daytime():
    manager = _manager()
    night, _ = manager.reset(init_day=0, init_hour=0, future_steps=8)
    day, _ = manager.reset(init_day=171, init_hour=12, future_steps=8)
    assert night == 0.0
    assert day > 0.0


def test_ac_output_is_finite_nonnegative_and_inverter_limited():
    manager = _manager()
    assert np.isfinite(manager.hourly_ac_kw).all()
    assert (manager.hourly_ac_kw >= 0.0).all()
    assert manager.hourly_ac_kw.max() <= manager.capacity_ac_kw + 1e-9


def test_capacity_fraction_scales_dc_nameplate_exactly():
    small = _manager(capacity_fraction_of_datacenter=0.05)
    large = _manager(capacity_fraction_of_datacenter=0.10)
    assert small.capacity_dc_kw == 50.0
    assert large.capacity_dc_kw == 100.0
    assert np.isclose(large.hourly_ac_kw.sum(), 2 * small.hourly_ac_kw.sum())


def test_hourly_ac_energy_is_conserved_across_four_quarter_hour_steps():
    manager = _manager()
    manager.reset(init_day=171, init_hour=12, future_steps=8)
    hourly_index = manager.time_step // 4
    hourly_energy = manager.hourly_ac_kw[hourly_index]
    energies = [manager.get_current_ac_kwh()]
    for _ in range(3):
        manager.step(future_steps=8)
        energies.append(manager.get_current_ac_kwh())
    assert np.isclose(sum(energies), hourly_energy)


def test_annual_wrap_and_forecast_are_fixed_length():
    manager = _manager()
    current, forecast = manager.reset(init_day=364, init_hour=23, future_steps=8)
    assert np.isfinite(current)
    assert forecast.shape == (8,)
    for _ in range(4):
        manager.step(future_steps=8)
    assert manager.time_step == manager.init_day * manager.time_steps_day


def test_battery_meter_applies_pv_before_grid_import_and_reports_curtailment():
    env = BatteryEnvFwd({
        "n_fwd_steps": 8, "max_dc_pw_MW": 2.0, "max_bat_cap": 1.0,
        "charging_rate": 0.5, "start_point": 0, "dcload_max": 2.0,
        "dcload_min": 0.1, "round_trip_efficiency": 1.0,
        "degradation_cost_c_per_kwh": 0.0,
    })
    env.reset()
    env.set_dcload(1.0)
    env.set_pv_ac_energy_kwh(100.0)
    _, _, _, _, info = env.step(2)
    assert np.isclose(info["bat_dc_load_kwh"], 250.0)
    assert np.isclose(info["bat_grid_import_kwh"], 150.0)
    assert np.isclose(info["pv_self_consumed_kwh"], 100.0)
    assert info["pv_curtailed_kwh"] == 0.0

    env.set_pv_ac_energy_kwh(400.0)
    _, _, _, _, surplus = env.step(2)
    assert surplus["bat_grid_import_kwh"] == 0.0
    assert np.isclose(surplus["pv_self_consumed_kwh"], 250.0)
    assert np.isclose(surplus["pv_curtailed_kwh"], 150.0)


def test_pv_combines_with_bus_side_battery_charge_at_single_meter():
    env = BatteryEnvFwd({
        "n_fwd_steps": 8, "max_dc_pw_MW": 2.0, "max_bat_cap": 1.0,
        "charging_rate": 0.5, "start_point": 0, "dcload_max": 2.0,
        "dcload_min": 0.1, "round_trip_efficiency": 1.0,
        "degradation_cost_c_per_kwh": 0.0,
    })
    env.reset()
    env.set_dcload(1.0)
    env.set_pv_ac_energy_kwh(100.0)
    _, _, _, _, info = env.step(0)
    assert np.isclose(info["bat_grid_import_kwh"],
                      info["bat_dc_load_kwh"] + info["bat_charge_bus_KWh"] - info["pv_ac_kwh"])
    assert info["bat_grid_import_kwh"] >= 0.0


def test_invalid_physical_configuration_is_rejected():
    for kwargs, field in [
        ({"capacity_fraction_of_datacenter": -0.1}, "capacity_fraction"),
        ({"surface_tilt_deg": 100.0}, "surface_tilt"),
        ({"dc_ac_ratio": 0.0}, "dc_ac_ratio"),
        ({"system_losses_fraction": 1.1}, "system_losses"),
        ({"inverter_efficiency": 0.0}, "inverter_efficiency"),
        ({"temperature_model": "not_real"}, "temperature_model"),
    ]:
        try:
            _manager(**kwargs)
        except ValueError as exc:
            assert field in str(exc)
        else:
            raise AssertionError(f"expected {kwargs} to fail")


if __name__ == "__main__":
    sys.exit(1 if run_module_tests(globals()) else 0)
