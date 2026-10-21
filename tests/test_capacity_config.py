"""Configuration contracts for facility and dispatchable battery sizing."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests._harness import run_module_tests
from sustaindc_env import EnvConfig, resolve_battery_sizing


def test_defaults_preserve_legacy_sizing():
    config = EnvConfig({})
    assert config['facility_sizing']['mode'] == 'legacy'
    assert resolve_battery_sizing(config) is None


def test_facility_sizing_resolves_scaled_battery():
    for capacity in (1.0, 10.0):
        config = EnvConfig({'datacenter_capacity_mw': capacity,
                            'facility_sizing': {'mode': 'scaling'}})
        assert resolve_battery_sizing(config) == (capacity, 2 * capacity)


def test_absolute_battery_overrides_and_conflicts():
    config = EnvConfig({'facility_sizing': {'mode': 'scaling'},
                        'battery': {'rated_power_mw': 3.0, 'capacity_mwh': 4.0}})
    assert resolve_battery_sizing(config) == (3.0, 4.0)
    for values in ({'rated_power_mw': 1, 'power_fraction': 0.5},
                   {'capacity_mwh': 2, 'duration_hours': 1}):
        try:
            EnvConfig({'facility_sizing': {'mode': 'scaling'}, 'battery': values})
        except ValueError:
            continue
        raise AssertionError('conflicting sizing inputs must fail')


def test_invalid_capacity_and_battery_inputs_fail():
    for value in (0, -1, float('nan'), float('inf'), True):
        try:
            EnvConfig({'facility_sizing': {'mode': 'scaling'},
                       'datacenter_capacity_mw': value})
        except (ValueError, TypeError):
            continue
        raise AssertionError(f'invalid capacity accepted: {value}')
    for key in ('power_fraction', 'duration_hours', 'rated_power_mw', 'capacity_mwh'):
        try:
            EnvConfig({'facility_sizing': {'mode': 'scaling'}, 'battery': {key: 0}})
        except (ValueError, TypeError):
            continue
        raise AssertionError(f'invalid {key} accepted')


if __name__ == '__main__':
    sys.exit(1 if run_module_tests(globals()) else 0)
