import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests._harness import run_module_tests
from envs.datacenter import (DataCenter_ITModel, calculate_avg_CRAC_return_temp,
                             calculate_HVAC_power, chiller_available_cooling_capacity,
                             calculate_chiller_power)
from harl.envs.sustaindc.metrics import build_trace_row
from utils.dc_config_reader import DC_Config
from utils.make_envs_pyenv import make_dc_pyeplus_env


def test_cooling_reset_restores_initial_setpoint_after_control_actions():
    env, _ = make_dc_pyeplus_env(
        dc_config_file='dc_config.json', location='QC',
        datacenter_capacity_mw=1.0, facility_sizing_mode='scaling',
    )
    env.reset()
    initial = env.raw_curr_stpt
    env.set_shifted_wklds(0.5)
    env.set_ambient_temp(20.0, 20.0)
    env.step(0)
    assert env.raw_curr_stpt < initial
    env.reset()
    assert env.raw_curr_stpt == initial


def test_air_cooled_chiller_capacity_curve_uses_coherent_reference_and_rated_temperature():
    # EnergyPlus AirCooledElectricChiller.idf: capacity coefficients
    # 0.9949, -0.045954, -0.0013543; condenser design 35 C,
    # chilled-water design 6.67 C and temperature-rise coefficient 2.778.
    nominal_w = 100_000.0
    assert math.isclose(chiller_available_cooling_capacity(nominal_w, 35.0),
                        99_490.0, rel_tol=1e-10)
    assert math.isclose(chiller_available_cooling_capacity(nominal_w, 5.0),
                        nominal_w * (0.9949 - 0.045954 * (-30.0 / 2.778)
                                     - 0.0013543 * (-30.0 / 2.778)**2),
                        rel_tol=1e-10)
    assert chiller_available_cooling_capacity(nominal_w, 5.0) > 99_490.0
    assert calculate_chiller_power(nominal_w, nominal_w / 2, 5.0) < calculate_chiller_power(
        nominal_w, nominal_w / 2, 35.0)


def test_air_cooled_chiller_power_uses_example_nominal_cop():
    # AirCooledElectricChiller.idf: nominal COP 2.58 and the power curves.
    nominal_w = 100_000.0
    available_w = nominal_w * 0.9949
    capacity_ratio = 0.9949
    power_ratio = 2.333 - 1.975 * capacity_ratio + 0.6121 * capacity_ratio**2
    full_load_ratio = 0.03303 + 0.6852 + 0.2818
    expected_w = full_load_ratio * power_ratio * available_w / 2.58
    assert math.isclose(calculate_chiller_power(nominal_w, available_w, 35.0),
                        expected_w, rel_tol=1e-10)


def test_server_power_curve_respects_configured_full_load_endpoints():
    config = DC_Config('dc_config.json')
    dc = DataCenter_ITModel(config.NUM_RACKS, config.RACK_SUPPLY_APPROACH_TEMP_LIST,
                            config.RACK_CPU_CONFIG, config.MAX_W_PER_RACK, config)
    rack = dc.racks_list[0]
    for inlet, ratio in ((16.0, 1.0), (28.0, 1.02)):
        cpu_power, _ = rack.compute_instantaneous_pwr_vecd(inlet, 100.0)
        expected = sum(max(cpu.idle_pwr, cpu.full_load_pwr * ratio)
                       for cpu in rack.CPU_list)
        assert math.isclose(cpu_power, expected, rel_tol=1e-10)


def test_fan_flow_matches_configured_temperature_and_utilization_endpoints():
    config = DC_Config('dc_config.json')
    dc = DataCenter_ITModel(config.NUM_RACKS, config.RACK_SUPPLY_APPROACH_TEMP_LIST,
                            config.RACK_CPU_CONFIG, config.MAX_W_PER_RACK, config)
    rack = dc.racks_list[0]
    for inlet, utilization, expected_ratio in (
            (16.0, 25.0, 0.01), (16.0, 100.0, 0.225),
            (28.0, 25.0, 0.225), (28.0, 100.0, 1.0)):
        rack.compute_instantaneous_pwr_vecd(inlet, utilization)
        expected_flow = len(rack.CPU_list) * config.IT_FAN_FULL_LOAD_V * expected_ratio
        assert math.isclose(rack.get_total_rack_fan_v(), expected_flow, rel_tol=1e-10)


def test_rack_outlet_is_energy_balanced_and_zero_utilization_is_finite():
    config = DC_Config('dc_config.json')
    dc = DataCenter_ITModel(config.NUM_RACKS, config.RACK_SUPPLY_APPROACH_TEMP_LIST,
                            config.RACK_CPU_CONFIG, config.MAX_W_PER_RACK, config)
    for utilization in (0.0, 20.0, 50.0, 100.0):
        powers, fans, outlets = dc.compute_datacenter_IT_load_outlet_temp(
            [utilization] * config.NUM_RACKS, 18.0)
        for rack, inlet, power, fan, outlet in zip(
                dc.racks_list, dc.rackwise_inlet_temp, powers, fans, outlets):
            flow = rack.get_total_rack_fan_v()
            assert flow > 0
            assert math.isfinite(outlet) and outlet > inlet
            assert math.isclose(
                config.RHO_AIR * config.C_AIR * flow * (outlet - inlet),
                power + fan, rel_tol=1e-9)
        assert math.isfinite(calculate_avg_CRAC_return_temp(
            config.RACK_RETURN_APPROACH_TEMP_LIST, outlets))


def test_cooling_load_tracks_rack_heat_at_design_and_partial_utilization():
    env, _ = make_dc_pyeplus_env(dc_config_file='dc_config.json', location='montreal',
                                 datacenter_capacity_mw=1.0, facility_sizing_mode='scaling')
    config = env.DC_Config
    for utilization in (0.0, 50.0, 100.0):
        powers, fans, outlets = env.dc.compute_datacenter_IT_load_outlet_temp(
            [utilization] * config.NUM_RACKS, 21.6)
        it_heat_w = sum(powers) + sum(fans)
        return_temp = calculate_avg_CRAC_return_temp(
            config.RACK_RETURN_APPROACH_TEMP_LIST, outlets)
        _, _, cooling_w, _, _, _ = calculate_HVAC_power(
            21.6, return_temp, 33.0, it_heat_w, config)
        assert math.isclose(cooling_w, it_heat_w, rel_tol=1e-9)


def test_capacity_limited_chiller_reports_unmet_heat_without_claiming_thermal_safety():
    env, _ = make_dc_pyeplus_env(dc_config_file='dc_config.json', location='montreal',
                                 datacenter_capacity_mw=1.0, facility_sizing_mode='scaling')
    env.DC_Config.CHILLER_NOMINAL_COOLING_CAPACITY_W *= 0.5
    env.reset()
    env.set_shifted_wklds(1.0)
    env.set_ambient_temp(33.0, 25.0)
    env.raw_curr_stpt = 21.6
    _, _, _, _, info = env.step(1)
    assert info['dc_chiller_available_cooling_kW'] > 0
    assert info['dc_unmet_cooling_kW'] > 100
    assert math.isclose(info['dc_unmet_cooling_kW'],
                        info['dc_ITE_total_power_kW'] - info['dc_chiller_available_cooling_kW'],
                        rel_tol=1e-10)
    assert info['dc_cooling_capacity_exceeded'] == 1.0
    assert info['dc_steady_state_thermal_valid'] == 0.0
    trace = build_trace_row(info=info, actions=[[1], [1], [2]],
                            run_metadata={'step': 0}, terminal=False, truncated=False)
    assert math.isclose(trace['thermal/dc_unmet_cooling_kW'],
                        info['dc_unmet_cooling_kW'])
    assert trace['thermal/dc_steady_state_thermal_valid'] == 0.0


def test_sized_plant_can_remove_full_load_heat_at_design_ambient():
    for capacity_mw in (1.0, 10.0):
        env, _ = make_dc_pyeplus_env(dc_config_file='dc_config.json', location='montreal',
                                     datacenter_capacity_mw=capacity_mw,
                                     facility_sizing_mode='scaling')
        env.reset()
        env.set_shifted_wklds(1.0)
        env.set_ambient_temp(33.0, 25.0)
        env.raw_curr_stpt = 21.6
        _, _, _, _, info = env.step(1)
        assert info['dc_unmet_cooling_kW'] <= 1e-6
        assert info['dc_steady_state_thermal_valid'] == 1.0


def test_subcapacity_load_has_no_unmet_heat_and_valid_steady_state():
    env, _ = make_dc_pyeplus_env(dc_config_file='dc_config.json', location='montreal',
                                 datacenter_capacity_mw=1.0, facility_sizing_mode='scaling')
    env.reset()
    env.set_shifted_wklds(0.5)
    env.set_ambient_temp(33.0, 25.0)
    env.raw_curr_stpt = 21.6
    _, _, _, _, info = env.step(1)
    assert info['dc_chiller_available_cooling_kW'] > info['dc_ITE_total_power_kW']
    assert info['dc_unmet_cooling_kW'] == 0.0
    assert info['dc_cooling_capacity_exceeded'] == 0.0
    assert info['dc_steady_state_thermal_valid'] == 1.0


def test_air_cooled_facility_has_no_water_side_electricity_or_water_use():
    env, _ = make_dc_pyeplus_env(dc_config_file='dc_config.json', location='montreal',
                                 datacenter_capacity_mw=1.0, facility_sizing_mode='scaling')
    for ambient in (0.0, 33.0):
        env.reset()
        env.set_shifted_wklds(0.7)
        env.set_ambient_temp(ambient, ambient - 2.0)
        env.raw_curr_stpt = 18.0
        _, _, _, _, info = env.step(1)
        assert info['dc_CT_total_power_kW'] == 0.0
        chilled_water_pump_w = (env.DC_Config.CW_PRESSURE_DROP
                                * env.DC_Config.CW_WATER_FLOW_RATE
                                / env.DC_Config.CW_PUMP_EFFICIENCY)
        assert math.isclose(info['dc_CW_pump_power_kW'], chilled_water_pump_w / 1000)
        assert info['dc_CT_pump_power_kW'] == 0.0
        assert info['dc_water_usage'] == 0.0
        assert math.isclose(info['dc_HVAC_total_power_kW'],
                            (env.CRAC_Fan_load + env.Compressor_load + chilled_water_pump_w) / 1000,
                            rel_tol=1e-10)
        assert math.isclose(info['dc_total_power_kW'],
                            info['dc_ITE_total_power_kW'] + info['dc_HVAC_total_power_kW'],
                            rel_tol=1e-10)


def test_air_cooled_sizing_and_power_bounds_follow_the_electrical_meter():
    for capacity_mw in (1.0, 10.0):
        env, _ = make_dc_pyeplus_env(dc_config_file='dc_config.json', location='montreal',
                                     datacenter_capacity_mw=capacity_mw,
                                     facility_sizing_mode='scaling')
        meta = env.sizing_metadata
        assert meta['tower_fan_mw'] == meta['ct_pump_mw'] == 0.0
        assert meta['cw_pump_mw'] > 0.0
        env.reset()
        env.set_shifted_wklds(1.0)
        env.set_ambient_temp(meta['ambient_temperature_c'], 25.0)
        env.raw_curr_stpt = meta['crac_setpoint_c']
        _, _, _, _, info = env.step(1)
        assert math.isclose(meta['achieved_mw'] * 1000,
                            info['dc_total_power_kW'], rel_tol=1e-10)
        assert math.isclose(meta['hvac_mw'] * 1000,
                            info['dc_HVAC_total_power_kW'], rel_tol=1e-10)
        it_bound_w = env.ranges[
            'Facility Total Building Electricity Demand Rate(Whole Building)'][1] / 1.1
        crac_fan_w = env.DC_Config.CRAC_FAN_REF_P * (
            env.DC_Config.CRAC_SUPPLY_AIR_FLOW_RATE_pu
            / env.DC_Config.CRAC_REFRENCE_AIR_FLOW_RATE_pu) ** 3
        compressor_w = calculate_chiller_power(
            env.DC_Config.CHILLER_NOMINAL_COOLING_CAPACITY_W, it_bound_w,
            meta['ambient_temperature_c'])
        chilled_water_pump_w = (env.DC_Config.CW_PRESSURE_DROP
                                * env.DC_Config.CW_WATER_FLOW_RATE
                                / env.DC_Config.CW_PUMP_EFFICIENCY)
        hvac_bound_w = 1.1 * (crac_fan_w + compressor_w + chilled_water_pump_w)
        assert math.isclose(env.ranges[
            'Facility Total HVAC Electricity Demand Rate(Whole Building)'][1],
            hvac_bound_w, rel_tol=1e-10)
        assert math.isclose(env.ranges[
            'Facility Total Electricity Demand Rate(Whole Building)'][1],
            1.1 * it_bound_w + hvac_bound_w, rel_tol=1e-10)


def test_cooling_tower_fan_rating_is_distinct_from_chiller_cooling_capacity():
    env, _ = make_dc_pyeplus_env(dc_config_file='dc_config.json', location='montreal',
                                 datacenter_capacity_mw=1.0, facility_sizing_mode='scaling')
    config = env.DC_Config
    assert config.CHILLER_NOMINAL_COOLING_CAPACITY_W > 0
    assert math.isclose(config.CT_FAN_REF_P, 1000.0)
    assert config.CHILLER_NOMINAL_COOLING_CAPACITY_W > config.CT_FAN_REF_P
    env.reset()
    env.set_shifted_wklds(1.0)
    env.set_ambient_temp(33.0, 25.0)
    env.raw_curr_stpt = 21.6
    _, _, _, _, info = env.step(1)
    assert info['dc_CT_total_power_kW'] <= config.CT_FAN_REF_P / 1000
    assert info['dc_Compressor_total_power_kW'] > 0


if __name__ == '__main__':
    sys.exit(1 if run_module_tests(globals()) else 0)
