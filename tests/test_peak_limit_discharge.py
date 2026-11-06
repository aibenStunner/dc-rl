"""Opt-in peak-limit battery discharge: deliver only the excess over a target."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests._harness import run_module_tests
from envs.bat_env_fwd_view import BatteryEnvFwd
from sustaindc_env import EnvConfig, SustainDC
from utils.make_envs_pyenv import make_bat_fwd_env


def _battery(initial_soc=0.5, **extra):
    config = {
        'n_fwd_steps': 8, 'max_dc_pw_MW': 2.0, 'max_bat_cap': 2.0,
        'charging_rate': 0.5, 'dcload_max': 2.0, 'dcload_min': 0.0,
        'round_trip_efficiency': 0.90, 'degradation_cost_c_per_kwh': 5.78,
        'rated_power_mw': 1.0, 'initial_soc': initial_soc,
    }
    config.update(extra)
    battery = BatteryEnvFwd(config)
    battery.reset()
    return battery


def test_default_discharge_mode_is_fixed_rate():
    assert _battery().discharge_mode == 'fixed_rate'
    assert EnvConfig({})['battery']['discharge_mode'] == 'fixed_rate'


def test_peak_limit_discharges_only_the_excess_over_target():
    battery = _battery(discharge_mode='peak_limit')
    battery.set_dcload(0.8)
    battery.set_discharge_peak_target_kw(600.0)
    _, _, _, _, info = battery.step(1)
    assert math.isclose(info['bat_requested_power_mw'], 0.2)
    assert math.isclose(info['bat_delivered_bus_power_mw'], -0.2)
    assert math.isclose(info['bat_discharge_bus_KWh'], 50.0)
    assert math.isclose(info['bat_grid_import_kwh'], 150.0)
    assert math.isclose(info['bat_grid_import_kwh'] * 4, 600.0)


def test_peak_limit_subtracts_pv_from_the_excess():
    battery = _battery(discharge_mode='peak_limit')
    battery.set_dcload(0.8)
    battery.set_pv_ac_energy_kwh(25.0)
    battery.set_discharge_peak_target_kw(600.0)
    _, _, _, _, info = battery.step(1)
    assert math.isclose(info['bat_requested_power_mw'], 0.1)
    assert math.isclose(info['bat_discharge_bus_KWh'], 25.0)
    assert math.isclose(info['bat_grid_import_kwh'] * 4, 600.0)


def test_peak_limit_below_target_leaves_soc_unchanged():
    battery = _battery(discharge_mode='peak_limit')
    soc = battery.get_battery_soc()
    battery.set_dcload(0.5)
    battery.set_discharge_peak_target_kw(600.0)
    _, _, _, _, info = battery.step(1)
    assert info['bat_requested_power_mw'] == 0.0
    assert info['bat_discharge_bus_KWh'] == 0.0
    assert info['bat_cell_throughput_KWh'] == 0.0
    assert battery.get_battery_soc() == soc
    assert math.isclose(info['bat_grid_import_kwh'], 125.0)


def test_peak_limit_is_still_capped_by_the_soc_rate_curve():
    battery = _battery(initial_soc=0.1, discharge_mode='peak_limit')
    rate_cap = battery.discharging_rate_modifier(battery.battery)
    assert rate_cap < 0.8
    battery.set_dcload(0.8)
    battery.set_discharge_peak_target_kw(0.0)
    _, _, _, _, info = battery.step(1)
    assert math.isclose(info['bat_requested_power_mw'], rate_cap)
    assert math.isclose(info['bat_delivered_bus_power_mw'], -rate_cap)


def test_peak_limit_requires_a_target_and_reset_clears_it():
    battery = _battery(discharge_mode='peak_limit')
    battery.set_dcload(0.8)
    for _ in range(2):
        try:
            battery.step(1)
        except ValueError:
            pass
        else:
            raise AssertionError('peak_limit discharged without a target')
        battery.set_discharge_peak_target_kw(600.0)
        battery.step(1)
        battery.reset()
        battery.set_dcload(0.8)


def test_peak_limit_does_not_change_charge_or_idle():
    for action in (0, 2):
        limited = _battery(discharge_mode='peak_limit')
        fixed = _battery()
        for battery in (limited, fixed):
            battery.set_dcload(0.5)
        limited.set_discharge_peak_target_kw(0.0)
        info_limited = limited.step(action)[4]
        info_fixed = fixed.step(action)[4]
        for key in ('bat_requested_power_mw', 'bat_charge_bus_KWh',
                    'bat_discharge_bus_KWh', 'bat_grid_import_kwh', 'bat_SOC'):
            assert info_limited[key] == info_fixed[key], key


def test_fixed_rate_ignores_a_discharge_target():
    battery = _battery()
    battery.set_dcload(0.8)
    battery.set_discharge_peak_target_kw(600.0)
    _, _, _, _, info = battery.step(1)
    assert math.isclose(info['bat_delivered_bus_power_mw'], -0.8)


def test_invalid_discharge_mode_and_target_are_rejected():
    for mode in ('peak', None, 1, ''):
        try:
            _battery(discharge_mode=mode)
        except ValueError:
            pass
        else:
            raise AssertionError(f'accepted discharge_mode {mode!r}')
    battery = _battery(discharge_mode='peak_limit')
    for target in (-1.0, float('nan'), float('inf'), True):
        try:
            battery.set_discharge_peak_target_kw(target)
        except ValueError:
            pass
        else:
            raise AssertionError(f'accepted target {target!r}')
    for mode in ('peak', None, 1):
        try:
            EnvConfig({'battery': {'discharge_mode': mode}})
        except ValueError:
            pass
        else:
            raise AssertionError(f'EnvConfig accepted discharge_mode {mode!r}')


def test_factory_forwards_discharge_mode():
    battery = make_bat_fwd_env(month=0, discharge_mode='peak_limit')
    assert battery.discharge_mode == 'peak_limit'
    assert make_bat_fwd_env(month=0).discharge_mode == 'fixed_rate'


def _run_discharge(mode, steps=96):
    env = SustainDC(EnvConfig({
        'month': 6, 'episode_start_policy': 'valid_midnight',
        'battery': {'initial_soc': 0.5, 'discharge_mode': mode},
    }))
    env.reset(seed=102, options={'init_day': 14, 'init_hour': 0})
    rows = []
    for _ in range(steps):
        prior_peak = env.price_m.state.running_peak_kw
        rate_cap = env.bat_env.discharging_rate_modifier(env.bat_env.battery)
        _, _, _, _, info = env.step({'agent_ls': 1, 'agent_dc': 1, 'agent_bat': 1})
        rows.append((prior_peak, rate_cap, info['__common__']))
    return env, rows


def test_joint_peak_limit_holds_import_at_the_pre_step_billing_peak():
    env, rows = _run_discharge('peak_limit')
    assert env.bat_env.discharge_mode == 'peak_limit'
    held = 0
    for prior_peak, rate_cap, common in rows:
        net_kw = common['dc_total_power_kW'] - 4 * common['pv_ac_kwh']
        excess_kw = max(0.0, net_kw - prior_peak)
        assert math.isclose(common['bat_requested_power_mw'],
                            min(rate_cap, excess_kw / 1000.0), abs_tol=1e-9)
        if 0.0 < excess_kw / 1000.0 <= rate_cap and common['bat_SOC'] > 0.0:
            assert 4 * common['bat_grid_import_kwh'] <= prior_peak + 1e-6
            held += 1
    assert held > 0


def _run_armed_late(mode, arm_step=None, steps=192):
    env = SustainDC(EnvConfig({
        'month': 6, 'episode_start_policy': 'valid_midnight',
        'battery': {'initial_soc': 0.5, 'discharge_mode': mode},
    }))
    env.reset(seed=102, options={'init_day': 14, 'init_hour': 0})
    imports_kw = []
    for step in range(steps):
        action = 1 if arm_step is not None and step >= arm_step else 2
        _, _, _, _, info = env.step({'agent_ls': 1, 'agent_dc': 1, 'agent_bat': action})
        imports_kw.append(4 * info['__common__']['bat_grid_import_kwh'])
    return env.price_m.state.running_peak_kw, env.bat_env.get_battery_soc(), imports_kw


def test_joint_peak_limit_armed_late_shaves_where_fixed_rate_cannot():
    # Month-start caveat: the target is the running peak, so arming while it
    # is still low drains the battery exactly like fixed_rate. Arming once the
    # running peak is near the uncontrolled peak leaves enough energy to hold it.
    noop_peak, _, noop_imports = _run_armed_late('fixed_rate')
    arm_step = next(step for step, kw in enumerate(noop_imports) if kw >= 0.9 * noop_peak)
    limited_peak, limited_soc, _ = _run_armed_late('peak_limit', arm_step)
    fixed_peak, fixed_soc, _ = _run_armed_late('fixed_rate', arm_step)
    assert limited_peak < 0.95 * noop_peak
    assert limited_peak < fixed_peak
    assert limited_soc > fixed_soc


if __name__ == '__main__':
    run_module_tests(globals())
