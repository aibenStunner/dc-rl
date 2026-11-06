"""Accounting checks at the workload, cooling, and battery boundaries."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests._harness import run_module_tests
from envs.bat_env_fwd_view import BatteryEnvFwd
from harl.envs.sustaindc.metrics import FacilityMetricAccumulator
from sustaindc_env import EnvConfig, SustainDC
from utils.make_envs_pyenv import make_dc_pyeplus_env
from utils import reward_creator


def test_cooling_meter_includes_all_computed_electrical_equipment():
    env, _ = make_dc_pyeplus_env(
        dc_config_file='dc_config.json', location='QC',
        datacenter_capacity_mw=1.0, facility_sizing_mode='scaling',
    )
    env.reset()
    env.set_shifted_wklds(0.5)
    env.set_ambient_temp(20, 20)
    _, _, _, _, info = env.step(1)
    expected_hvac = (
        env.CRAC_Fan_load + env.CT_Cooling_load + env.Compressor_load
        + env.CW_pump_load + env.CT_pump_load
    ) / 1000
    assert math.isclose(info['dc_HVAC_total_power_kW'], expected_hvac)
    assert math.isclose(info['dc_total_power_kW'], info['dc_ITE_total_power_kW'] + expected_hvac)
    assert math.isclose(info['dc_CW_pump_power_kW'], env.CW_pump_load / 1000)
    assert math.isclose(info['dc_CT_pump_power_kW'], env.CT_pump_load / 1000)


def test_terminal_queue_liability_is_ls_only_and_battery_settlement_remains_separate():
    config = EnvConfig({
        'month': 6,
        'days_per_episode': 1,
        'reward': {'mode': 'team_tariff_targeted_safeguards',
                   'ls_terminal_backlog_penalty_weight': 0.5},
        'episode_start_policy': 'valid_midnight',
    })
    env = SustainDC(config)
    env.reset(seed=102, options={'init_day': 14, 'init_hour': 0})
    for _ in range(96):
        _, rewards, _, truncated, info = env.step(
            {'agent_ls': 0, 'agent_dc': 1, 'agent_bat': 2}
        )
    common = info['__common__']
    assert truncated['__all__']
    assert common['ls_tasks_in_queue'] > 0
    assert common['ls_terminal_backlog_tasks'] == common['ls_tasks_in_queue']
    assert common['ls_terminal_backlog_penalty_component'] == 0.5 * common['ls_tasks_in_queue']
    accumulator = FacilityMetricAccumulator('episode')
    accumulator.add(common)
    assert (accumulator.snapshot()['episode/reward/penalty/ls_terminal_backlog/sum']
            == common['ls_terminal_backlog_penalty_component'])
    assert math.isclose(rewards['agent_ls'], common['shared_tariff_cost_reward']
                        - common['ls_overdue_penalty_component']
                        - common['ls_dropped_penalty_component']
                        - common['ls_backlog_penalty_component']
                        - common['ls_terminal_backlog_penalty_component'])
    assert math.isclose(rewards['agent_bat'], common['shared_tariff_cost_reward']
                        - common['bat_terminal_soc_penalty_component']
                        - common['bat_degradation_penalty_component'])
    assert math.isclose(common['bat_grid_import_kwh'],
                        common['bat_dc_load_kwh'] - common['pv_self_consumed_kwh']
                        + common['bat_charge_bus_KWh'] - common['bat_discharge_bus_KWh'])


def test_rack_inlet_limit_is_distinct_from_crac_setpoint_bound():
    env, _ = make_dc_pyeplus_env(
        dc_config_file='dc_config.json', location='QC',
        datacenter_capacity_mw=1.0, facility_sizing_mode='scaling',
    )
    env.reset()
    env.set_shifted_wklds(0.5)
    env.set_ambient_temp(20, 20)
    env.raw_curr_stpt = 21.6
    _, _, _, _, info = env.step(1)
    assert info['dc_max_rack_inlet_c'] == max(env.dc.rackwise_inlet_temp)
    assert info['dc_max_rack_inlet_c'] > info['dc_thermal_limit_c']
    assert info['dc_max_rack_inlet_c'] < 28.0
    assert info['dc_constraint_violation'] == 0
    accumulator = FacilityMetricAccumulator('episode')
    accumulator.add(info)
    assert accumulator.snapshot()['episode/thermal/rack_inlet_c/max'] == info['dc_max_rack_inlet_c']


def test_cooling_meter_reaches_battery_when_battery_charges():
    env = SustainDC(EnvConfig({
        'month': 6,
        'reward': {'mode': 'team_tariff_targeted_safeguards'},
        'episode_start_policy': 'valid_midnight',
    }))
    env.reset(seed=102, options={'init_day': 14, 'init_hour': 0})
    _, _, _, _, info = env.step({'agent_ls': 1, 'agent_dc': 1, 'agent_bat': 0})
    common = info['__common__']
    dc = env.dc_env
    battery = env.bat_env
    assert math.isclose(common['dc_total_power_kW'],
                        common['dc_ITE_total_power_kW'] + common['dc_HVAC_total_power_kW'])
    assert math.isclose(common['dc_HVAC_total_power_kW'],
                        dc.HVAC_load / 1000)
    assert math.isclose(common['bat_dc_load_kwh'],
                        common['dc_total_power_kW'] / 4)
    assert math.isclose(common['bat_grid_import_kwh'],
                        common['bat_dc_load_kwh'] - common['pv_self_consumed_kwh']
                        + common['bat_charge_bus_KWh'] - common['bat_discharge_bus_KWh'])
    assert 0 <= battery.get_battery_soc() <= 1
    assert common['bat_charge_bus_KWh'] >= common['bat_charge_cell_KWh'] >= 0


def test_battery_info_distinguishes_requested_direction_from_delivered_flows():
    battery = BatteryEnvFwd({
        'n_fwd_steps': 8, 'max_dc_pw_MW': 2.0, 'max_bat_cap': 2.0,
        'charging_rate': 0.5, 'dcload_max': 2.0, 'dcload_min': 0.0,
        'round_trip_efficiency': 0.90, 'degradation_cost_c_per_kwh': 5.78,
        'rated_power_mw': 1.0, 'initial_soc': 0.0,
    })
    battery.reset()
    battery.set_dcload(0.0)
    _, _, _, _, info = battery.step(1)
    assert info['bat_requested_action'] == 'discharge'
    assert info['bat_requested_power_mw'] > 0.0
    assert info['bat_delivered_bus_power_mw'] == 0.0
    assert info['bat_discharge_bus_KWh'] == 0.0


def test_charge_stays_under_running_peak_without_wasting_available_pv():
    battery = BatteryEnvFwd({
        'n_fwd_steps': 8, 'max_dc_pw_MW': 2.0, 'max_bat_cap': 2.0,
        'charging_rate': 0.5, 'dcload_max': 2.0, 'dcload_min': 0.0,
        'round_trip_efficiency': 0.90, 'degradation_cost_c_per_kwh': 5.78,
        'rated_power_mw': 1.0, 'initial_soc': 0.5,
    })
    battery.reset()
    battery.set_dcload(0.7)
    battery.set_pv_ac_energy_kwh(25.0)
    battery.set_grid_import_peak_kw(750.0)
    _, _, _, _, info = battery.step(0)
    assert math.isclose(info['bat_charge_bus_KWh'], 37.5)
    assert math.isclose(info['bat_grid_import_kwh'], 187.5)
    assert math.isclose(info['bat_grid_import_kwh'] * 4, 750.0)


def test_charge_at_peak_uses_only_pv_surplus():
    battery = BatteryEnvFwd({
        'n_fwd_steps': 8, 'max_dc_pw_MW': 2.0, 'max_bat_cap': 2.0,
        'charging_rate': 0.5, 'dcload_max': 2.0, 'dcload_min': 0.0,
        'round_trip_efficiency': 0.90, 'degradation_cost_c_per_kwh': 5.78,
        'rated_power_mw': 1.0, 'initial_soc': 0.5,
    })
    battery.reset()
    battery.set_dcload(0.8)
    battery.set_grid_import_peak_kw(0.0)
    battery.set_pv_ac_energy_kwh(210.0)
    _, _, _, _, info = battery.step(0)
    assert math.isclose(info['bat_charge_bus_KWh'], 10.0)
    assert math.isclose(info['bat_grid_import_kwh'], 0.0)
    battery.set_pv_ac_energy_kwh(0.0)
    _, _, _, _, info = battery.step(0)
    assert info['bat_charge_bus_KWh'] == 0.0
    assert math.isclose(info['bat_grid_import_kwh'], 200.0)


def test_joint_battery_charge_respects_pre_step_billing_peak():
    env = SustainDC(EnvConfig({
        'month': 6, 'episode_start_policy': 'valid_midnight',
        'battery': {'peak_charge_headroom': True},
    }))
    env.reset(seed=102, options={'init_day': 14, 'init_hour': 0})
    prior_peak = env.price_m.state.running_peak_kw
    _, _, _, _, info = env.step({'agent_ls': 1, 'agent_dc': 1, 'agent_bat': 0})
    common = info['__common__']
    assert common['bat_charge_bus_KWh'] <= max(
        0.0, (prior_peak - common['dc_total_power_kW']) / 4 + common['pv_ac_kwh']
    ) + 1e-6


def test_peak_charge_limit_resets_between_episodes():
    battery = BatteryEnvFwd({
        'n_fwd_steps': 8, 'max_dc_pw_MW': 2.0, 'max_bat_cap': 2.0,
        'charging_rate': 0.5, 'dcload_max': 2.0, 'dcload_min': 0.0,
        'round_trip_efficiency': 0.90, 'degradation_cost_c_per_kwh': 5.78,
        'rated_power_mw': 1.0, 'initial_soc': 0.5,
    })
    battery.reset()
    battery.set_grid_import_peak_kw(0.0)
    battery.reset()
    battery.set_dcload(0.5)
    _, _, _, _, info = battery.step(0)
    assert info['bat_charge_bus_KWh'] > 0.0


def test_training_scenario_charging_cannot_create_new_demand_peak():
    import yaml

    config_path = Path(__file__).resolve().parent.parent / 'harl/configs/envs_cfgs/sustaindc.yaml'
    config = yaml.safe_load(config_path.read_text())
    env = SustainDC(config)
    env.reset(seed=102, options={'init_day': 0, 'init_hour': 0})
    for _ in range(8):
        previous_peak = env.price_m.state.running_peak_kw
        if env.battery_config['observation_timing'] == 'current':
            env.begin_step({'agent_ls': 1, 'agent_dc': 1})
        _, _, _, _, info = env.step({'agent_ls': 1, 'agent_dc': 1, 'agent_bat': 0})
        common = info['__common__']
        assert common['billing_running_peak_kw'] <= max(
            previous_peak, common['dc_total_power_kW']
            - 4 * common['pv_ac_kwh']) + 1e-6


def test_peak_charge_headroom_is_opt_in_and_validated():
    assert EnvConfig({})['battery']['peak_charge_headroom'] is False
    for invalid in (None, 1, 'yes'):
        try:
            EnvConfig({'battery': {'peak_charge_headroom': invalid}})
        except ValueError:
            pass
        else:
            raise AssertionError(f'accepted invalid peak_charge_headroom: {invalid!r}')


def test_nonterminal_backlog_has_no_terminal_charge():
    params = {
        'total_price_cost_this_step_c': 0, 'ls_overdue_penalty': 0,
        'ls_tasks_dropped': 0, 'ls_norm_tasks_in_queue': 0.5,
        'ls_tasks_in_queue': 500, 'isterminal': False,
        'dc_int_temperature': 20, 'dc_thermal_limit_c': 21.6,
        'dc_constraint_violation': 0, 'bat_terminal_soc_penalty_component': 0,
        'bat_degradation_cost_c': 0,
    }
    config = {
        'scale': .01, 'degradation_cost_weight': 0,
        'ls_overdue_penalty_weight': .5, 'ls_dropped_penalty_weight': .25,
        'ls_backlog_penalty_weight': .2, 'ls_terminal_backlog_penalty_weight': .5,
        'dc_thermal_penalty_weight': .75, 'dc_constraint_penalty_weight': .4,
    }
    _, components = reward_creator.team_tariff_targeted_safeguards(params, config)
    assert components['ls_terminal_backlog_penalty_component'] == 0
    params['isterminal'] = True
    _, components = reward_creator.team_tariff_targeted_safeguards(params, config)
    assert components['ls_terminal_backlog_penalty_component'] == 250


if __name__ == '__main__':
    sys.exit(1 if run_module_tests(globals()) else 0)
