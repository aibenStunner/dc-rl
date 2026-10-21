import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests._harness import run_module_tests
from utils.make_envs_pyenv import make_dc_pyeplus_env


def test_design_envelope():
    for target in (1.0, 10.0):
        env, _ = make_dc_pyeplus_env(dc_config_file='dc_config.json', location='NYIS',
                                     datacenter_capacity_mw=target, facility_sizing_mode='scaling')
        meta = env.sizing_metadata
        assert abs(meta['relative_error']) <= 0.02
        assert meta['server_count'] == sum(len(r.CPU_list) for r in env.dc.racks_list)
        env.reset()
        env.set_shifted_wklds(1.0)
        env.set_ambient_temp(meta['ambient_temperature_c'], meta['ambient_temperature_c'])
        env.raw_curr_stpt = meta['crac_setpoint_c']
        _, _, _, _, info = env.step(1)
        assert math.isclose(info['dc_total_power_kW'] / 1000, meta['achieved_mw'])
        assert math.isclose(info['dc_ITE_total_power_kW'] / 1000, meta['it_mw'])
        assert math.isclose(info['dc_HVAC_total_power_kW'] / 1000, meta['hvac_mw'])


def test_montreal_design_ambient_matches_its_weather_file():
    env, _ = make_dc_pyeplus_env(dc_config_file='dc_config.json', location='QC',
                                 datacenter_capacity_mw=1.0, facility_sizing_mode='scaling')
    meta = env.sizing_metadata
    # Montreal-Trudeau TMYx maximum dry bulb is 33.0 C.
    assert meta['ambient_temperature_c'] == 33.0


def test_validation_and_determinism():
    for value in (0, -1, math.nan, math.inf):
        try:
            make_dc_pyeplus_env(dc_config_file='dc_config.json', datacenter_capacity_mw=value,
                                 facility_sizing_mode='scaling')
        except ValueError:
            pass
        else:
            raise AssertionError(f'accepted invalid capacity {value}')
    try:
        make_dc_pyeplus_env(dc_config_file='dc_config.json', facility_sizing_mode='unknown')
    except ValueError:
        pass
    else:
        raise AssertionError('accepted unknown mode')
    a, _ = make_dc_pyeplus_env(dc_config_file='dc_config.json', datacenter_capacity_mw=10, facility_sizing_mode='scaling')
    b, _ = make_dc_pyeplus_env(dc_config_file='dc_config.json', datacenter_capacity_mw=10, facility_sizing_mode='scaling')
    assert a.sizing_metadata == b.sizing_metadata
    assert a.DC_Config.RACK_CPU_CONFIG == b.DC_Config.RACK_CPU_CONFIG


def test_legacy_mode_is_unchanged():
    legacy, _ = make_dc_pyeplus_env(dc_config_file='dc_config.json', location='QC',
                                     datacenter_capacity_mw=1.0, facility_sizing_mode='legacy')
    assert legacy.sizing_metadata == {'mode': 'legacy'}


if __name__ == '__main__':
    sys.exit(1 if run_module_tests(globals()) else 0)
