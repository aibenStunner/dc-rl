"""Bus-rated BESS scaling and boundary constraints."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from envs.battery_model import Battery2
from envs.bat_env_fwd_view import BatteryEnvFwd
from tests._harness import run_module_tests


def _env(power, energy, **options):
    config = dict(n_fwd_steps=8, max_dc_pw_MW=20.0, max_bat_cap=energy,
                  charging_rate=0.5, dcload_max=20.0, dcload_min=0.0,
                  round_trip_efficiency=0.90, degradation_cost_c_per_kwh=5.78,
                  rated_power_mw=power)
    config.update(options)
    return BatteryEnvFwd(config)


def test_scaled_ratings_reach_bus_power_and_preserve_energy_balance():
    for power in (1.0, 10.0):
        env = _env(power, 2 * power, initial_soc=0.25)
        env.reset()
        env.set_dcload(power * 2)
        _, _, _, _, info = env.step(0)
        assert math.isclose(info['bat_charge_bus_KWh'], power * 250)
        assert math.isclose(info['bat_charge_cell_KWh'], power * 250 * math.sqrt(0.9))
        assert math.isclose(info['bat_grid_import_kwh'], power * 750)
        assert info['bat_rated_power_mw'] == power
        assert info['bat_capacity_mwh'] == 2 * power
        env.battery.reset(0.75)
        _, _, _, _, info = env.step(1)
        assert math.isclose(info['bat_discharge_bus_KWh'], power * 250)
        assert math.isclose(info['bat_discharge_cell_KWh'], power * 250 / math.sqrt(0.9))


def test_explicit_power_and_cell_c_rates_limit_both_directions():
    b = Battery2(capacity=2, current_load=1, round_trip_efficiency=0.81,
                 rated_power_mw=1, c_lim=0.1, d_lim=0.2)
    assert math.isclose(b.charge(10, 0.25).charge_bus_mwh, (0.2 / 0.9) * 0.25)
    assert math.isclose(b.discharge(10, 0.25, 10).discharge_bus_mwh, 0.4 * 0.9 * 0.25)
    b = Battery2(capacity=2, current_load=1, rated_power_mw=1, c_lim=10, d_lim=10)
    assert b.charge(10, 0.25).charge_bus_mwh == 0.25
    assert b.discharge(10, 0.25, 0.2).discharge_bus_mwh == 0.05


def test_soc_and_pv_limits_are_respected_at_explicit_rating():
    env = _env(1, 2, initial_soc=0.99)
    env.reset()
    env.set_dcload(0.1)
    env.set_pv_ac_energy_kwh(100)
    _, _, _, _, info = env.step(0)
    assert info['bat_SOC'] <= 1
    assert 0 < info['bat_charge_cell_KWh'] <= 20
    assert math.isclose(info['bat_grid_import_kwh'], 0)
    assert info['pv_curtailed_kwh'] > 0
    env.battery.reset(0.001)
    _, _, _, _, info = env.step(1)
    assert info['bat_SOC'] >= 0
    assert info['bat_discharge_bus_KWh'] <= 2 * math.sqrt(0.9)


def test_invalid_explicit_ratings_rejected():
    for value in (0, -1, math.nan, math.inf, True):
        try:
            _env(value, 2)
        except ValueError:
            pass
        else:
            raise AssertionError(f'invalid rating accepted: {value!r}')


if __name__ == '__main__':
    sys.exit(1 if run_module_tests(globals()) else 0)
