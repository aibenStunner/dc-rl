"""Regression coverage for the paired capacity-scaling episode schedule."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests import _scaling_paired_check as paired


class CalendarMonthEnv:
    starts = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)

    def __init__(self):
        self.dc_env = type('DC', (), {'sizing_metadata': {'target_mw': 1}})()
        self.battery_config = {'observation_timing': 'current'}
        self.t_m = type('Time', (), {})()
        self.completed = []
        self.begun = False

    def reset(self, *, seed, options):
        day = options['init_day']
        if day not in self.starts or options['init_hour'] != 0:
            raise ValueError('calendar_month reset requires a month-start midnight')
        index = self.starts.index(day)
        self.t_m.simulated_total_timesteps = (self.starts[index + 1] if index < 11 else 365) * 96 - day * 96
        self.day = day
        self.steps = 0
        self.begun = False

    def begin_step(self, actions):
        assert actions == {'agent_ls': 1, 'agent_dc': 1}
        self.begun = True

    def step(self, action):
        assert self.begun
        self.begun = False
        self.steps += 1
        if self.steps > self.t_m.simulated_total_timesteps:
            raise AssertionError('stepped beyond the calendar month')
        if self.steps == self.t_m.simulated_total_timesteps:
            self.completed.append((self.day, self.steps))
        return None, None, None, None, {
            '__common__': {'total_price_cost_this_step_c': 0},
            'agent_dc': {'dc_total_power_kW': 0, 'dc_ITE_total_power_kW': 0,
                         'dc_HVAC_total_power_kW': 0},
            'agent_bat': {'bat_grid_import_kwh': 0, 'bat_charge_bus_KWh': 0,
                          'bat_discharge_bus_KWh': 0, 'pv_ac_kwh': 0,
                          'pv_curtailed_kwh': 0, 'bat_SOC': 0.5},
        }


class ScalingPairedCheckTests(unittest.TestCase):
    def test_build_explicitly_requests_calendar_month_even_if_yaml_differs(self):
        from harl.utils.configs_tools import get_defaults_yaml_args

        _algo, args = get_defaults_yaml_args('happo', 'sustaindc')
        args['episode_start_policy'] = 'valid_midnight'
        with patch.object(paired, 'get_defaults_yaml_args', return_value=(_algo, args)):
            env = paired.build(1.0)
        self.assertEqual(env.episode_start_policy, 'calendar_month')

    def test_paired_cases_use_ten_deterministic_complete_calendar_months(self):
        environments = []

        def build(_capacity):
            env = CalendarMonthEnv()
            environments.append(env)
            return env

        with patch.object(paired, 'build', side_effect=build):
            _, one_mw = paired.run(1.0)
            _, ten_mw = paired.run(10.0)

        expected_days = [31, 151, 243, 273, 334, 212, 120, 304, 90, 59]
        expected_steps = [2688, 2880, 2880, 2976, 2976,
                          2976, 2976, 2880, 2880, 2976]
        for env, rows in zip(environments, (one_mw, ten_mw)):
            self.assertEqual(env.completed, list(zip(expected_days, expected_steps)))
            self.assertEqual(len(rows), 10)


def test_scaling_paired_check():
    result = unittest.TextTestRunner(verbosity=0).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(ScalingPairedCheckTests)
    )
    assert result.wasSuccessful(), result.failures + result.errors


if __name__ == '__main__':
    unittest.main()
