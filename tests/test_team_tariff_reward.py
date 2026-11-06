"""Unit checks for Experiment 1's targeted tariff reward."""
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests._harness import run_module_tests
from utils import reward_creator
from sustaindc_env import EnvConfig


_PARAMS = {
    "total_price_cost_this_step_c": 250.0,
    "ls_overdue_penalty": 2.0,
    "ls_tasks_dropped": 3.0,
    "ls_norm_tasks_in_queue": 0.50,
    "dc_int_temperature": 24.0,
    "dc_thermal_limit_c": 22.0,
    "dc_constraint_violation": 1.0,
    "isterminal": False,
    "bat_terminal_soc_penalty_component": 0.0,
    "bat_degradation_cost_c": 999.0,
}

_CONFIG = {
    "scale": 0.01,
    "degradation_cost_weight": 0.0,
    "ls_overdue_penalty_weight": 0.50,
    "ls_dropped_penalty_weight": 0.25,
    "ls_backlog_penalty_weight": 0.20,
    "dc_thermal_penalty_weight": 0.75,
    "dc_constraint_penalty_weight": 0.40,
}


def test_legacy_energy_normalization_is_isolated_and_resets_per_environment():
    from sustaindc_env import SustainDC

    config = {
        'reward': {'mode': 'legacy'}, 'agents': ['agent_ls', 'agent_dc', 'agent_bat'],
        'location': 'montreal', 'cintensity_file': 'QC_NG_&_avgCI.csv',
        'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv', 'days_per_episode': 1,
        'sample_whole_year': False, 'month': 6,
    }
    first, second = SustainDC(config), SustainDC(config)
    params = lambda energy: {
        'bat_total_energy_with_battery_KWh': energy, 'norm_CI': 0.5,
        'ls_overdue_penalty': 0.0, 'ls_oldest_task_age': 0.0,
    }
    try:
        first.reset(seed=1, options={'init_day': 181, 'init_hour': 0})
        second.reset(seed=2, options={'init_day': 181, 'init_hour': 0})
        assert first.calculate_reward(params(100.0))[0] == 0.3
        assert second.calculate_reward(params(1000.0))[0] == 0.3
        a = first.calculate_reward(params(200.0))
        b = second.calculate_reward(params(1100.0))
        assert math.isclose(a[0], -0.7, abs_tol=1e-12)
        assert math.isclose(a[1], -1.0, abs_tol=1e-12)
        assert math.isclose(a[2], -1.0, abs_tol=1e-12)
        assert math.isclose(b[0], -0.7, abs_tol=1e-12)
        first.reset(seed=1, options={'init_day': 181, 'init_hour': 0})
        assert first.calculate_reward(params(200.0))[0] == 0.3
    finally:
        first.ls_env.close()
        second.ls_env.close()


def test_legacy_default_dc_and_battery_rewards_normalize_without_default_ls():
    from sustaindc_env import SustainDC

    env = SustainDC({
        'reward': {'mode': 'legacy'}, 'ls_reward': 'tou_reward',
        'location': 'montreal', 'cintensity_file': 'QC_NG_&_avgCI.csv',
        'workload_file': 'Alibaba_CPU_Data_Hourly_1.csv', 'days_per_episode': 1,
        'sample_whole_year': False, 'month': 6,
    })
    params = {
        'bat_total_energy_with_battery_KWh': 100.0, 'norm_CI': 0.5,
        'ls_overdue_penalty': 0.0, 'ls_oldest_task_age': 0.0, 'hour': 0,
    }
    env.reset(seed=1, options={'init_day': 181, 'init_hour': 0})
    try:
        assert env.calculate_reward(params)[1:] == (0.0, 0.0)
        params['bat_total_energy_with_battery_KWh'] = 200.0
        _, dc, battery = env.calculate_reward(params)
        assert math.isclose(dc, -1.0, abs_tol=1e-12)
        assert math.isclose(battery, -1.0, abs_tol=1e-12)
    finally:
        env.ls_env.close()


def test_targeted_reward_uses_shared_tariff_and_agent_safeguards():
    rewards, components = reward_creator.team_tariff_targeted_safeguards(
        _PARAMS, _CONFIG
    )

    assert rewards == {
        "agent_ls": -4.35,
        "agent_dc": -4.4,
        "agent_bat": -2.5,
    }
    assert components["shared_facility_cost_c"] == 250.0
    assert components["shared_tariff_cost_reward"] == -2.5
    assert components["ls_overdue_penalty_component"] == 1.0
    assert components["ls_dropped_penalty_component"] == 0.75
    assert components["ls_backlog_penalty_component"] == 0.1
    assert components["dc_thermal_penalty_component"] == 1.5
    assert components["dc_constraint_penalty_component"] == 0.4
    assert components["bat_degradation_penalty_component"] == 0.0


def test_feasible_empty_state_has_only_shared_tariff_component():
    params = dict(_PARAMS)
    params.update(
        ls_overdue_penalty=0.0,
        ls_tasks_dropped=0.0,
        ls_norm_tasks_in_queue=0.0,
        dc_int_temperature=22.0,
        dc_constraint_violation=0.0,
    )

    rewards, components = reward_creator.team_tariff_targeted_safeguards(
        params, _CONFIG
    )

    assert rewards == {
        "agent_ls": -2.5,
        "agent_dc": -2.5,
        "agent_bat": -2.5,
    }
    assert all(value == 0.0 for key, value in components.items()
               if key.endswith("penalty_component"))


def test_inlet_calibration_bound_replaces_outlet_setpoint_penalty_when_enabled():
    params = dict(_PARAMS, dc_int_temperature=36.0,
                  dc_thermal_limit_c=21.6, dc_max_rack_inlet_c=26.0)
    config = dict(_CONFIG, dc_max_rack_inlet_c=28.0)
    rewards, components = reward_creator.team_tariff_targeted_safeguards(params, config)
    assert components['dc_thermal_penalty_component'] == 0.0
    assert rewards['agent_dc'] == -2.9
    params['dc_max_rack_inlet_c'] = 29.0
    rewards, components = reward_creator.team_tariff_targeted_safeguards(params, config)
    assert components['dc_thermal_penalty_component'] == .75
    assert rewards['agent_dc'] == -3.65
    valid = EnvConfig({'reward': {'mode': 'team_tariff_targeted_safeguards',
                                  'dc_max_rack_inlet_c': 28.0},
                       'episode_start_policy': 'valid_midnight'})
    assert valid['reward']['dc_max_rack_inlet_c'] == 28.0


def test_positive_degradation_weight_charges_only_battery_reward():
    params = dict(_PARAMS, bat_degradation_cost_c=100.0)
    config = dict(_CONFIG, degradation_cost_weight=1.0)
    rewards, components = reward_creator.team_tariff_targeted_safeguards(
        params, config
    )
    assert components['bat_degradation_penalty_component'] == 1.0
    assert rewards['agent_bat'] == -3.5
    assert rewards['agent_ls'] == -4.35
    assert rewards['agent_dc'] == -4.4
    valid = EnvConfig({'reward': {'mode': 'team_tariff_targeted_safeguards',
                                  'degradation_cost_weight': 1.0},
                       'episode_start_policy': 'valid_midnight'})
    assert valid['reward']['degradation_cost_weight'] == 1.0


def test_zero_degradation_weight_ignores_reported_degradation_cost():
    low_cost = dict(_PARAMS, bat_degradation_cost_c=0.0)
    high_cost = dict(_PARAMS, bat_degradation_cost_c=1_000_000.0)

    low_rewards, low_components = reward_creator.team_tariff_targeted_safeguards(
        low_cost, _CONFIG
    )
    high_rewards, high_components = reward_creator.team_tariff_targeted_safeguards(
        high_cost, _CONFIG
    )

    assert low_rewards == high_rewards
    assert low_components["bat_degradation_penalty_component"] == 0.0
    assert high_components["bat_degradation_penalty_component"] == 0.0


def test_targeted_reward_rejects_nonfinite_or_negative_inputs():
    for key, value in (
        ("total_price_cost_this_step_c", -1.0),
        ("ls_overdue_penalty", math.nan),
        ("dc_int_temperature", math.inf),
    ):
        params = dict(_PARAMS, **{key: value})
        try:
            reward_creator.team_tariff_targeted_safeguards(params, _CONFIG)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected {key}={value!r} to be rejected")


def test_experiment_one_config_rejects_invalid_reward_settings():
    invalid = (
        {"reward": {"mode": "unknown"}},
        {"reward": {"mode": "team_tariff_targeted_safeguards", "scale": -0.01}},
        {
            "agents": ["agent_bat", "agent_dc", "agent_ls"],
            "episode_start_policy": "valid_midnight",
            "reward": {"mode": "team_tariff_targeted_safeguards"},
        },
    )
    for config in invalid:
        try:
            EnvConfig(config)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected {config!r} to be rejected")


def test_reward_scale_multiplies_bill_and_degradation_but_not_safeguards():
    """reward.scale converts cents to reward units. It must scale only the
    cent-denominated terms; the fixed-weight LS/DC safeguards do not change,
    so their weight relative to the bill falls as the bill grows."""
    params = dict(_PARAMS, bat_degradation_cost_c=100.0,
                  isterminal=True, ls_tasks_in_queue=4.0)
    base = dict(_CONFIG, degradation_cost_weight=1.0,
                ls_terminal_backlog_penalty_weight=0.5)
    _, low = reward_creator.team_tariff_targeted_safeguards(params, base)
    _, high = reward_creator.team_tariff_targeted_safeguards(
        params, dict(base, scale=2.0 * base["scale"])
    )
    for key in ("shared_tariff_cost_reward", "bat_degradation_penalty_component"):
        assert math.isclose(high[key], 2.0 * low[key]), key
        assert low[key] != 0.0, key
    for key in ("ls_overdue_penalty_component", "ls_dropped_penalty_component",
                "ls_backlog_penalty_component", "ls_terminal_backlog_penalty_component",
                "dc_thermal_penalty_component", "dc_constraint_penalty_component"):
        assert high[key] == low[key], key
        assert low[key] != 0.0, key
    # Battery terminal SOC is priced by the environment and passed through.
    assert high["bat_terminal_soc_penalty_component"] == low["bat_terminal_soc_penalty_component"]


def test_agent_rewards_reconcile_exactly_with_reported_components():
    params = dict(_PARAMS, bat_degradation_cost_c=100.0, isterminal=True,
                  ls_tasks_in_queue=4.0, bat_terminal_soc_penalty_component=1.25)
    config = dict(_CONFIG, degradation_cost_weight=1.0,
                  ls_terminal_backlog_penalty_weight=0.5)
    rewards, c = reward_creator.team_tariff_targeted_safeguards(params, config)
    assert math.isclose(c["shared_tariff_cost_reward"],
                        -config["scale"] * c["shared_facility_cost_c"])
    assert math.isclose(rewards["agent_ls"], c["shared_tariff_cost_reward"]
                        - c["ls_overdue_penalty_component"]
                        - c["ls_dropped_penalty_component"]
                        - c["ls_backlog_penalty_component"]
                        - c["ls_terminal_backlog_penalty_component"])
    assert math.isclose(rewards["agent_dc"], c["shared_tariff_cost_reward"]
                        - c["dc_thermal_penalty_component"]
                        - c["dc_constraint_penalty_component"])
    assert math.isclose(rewards["agent_bat"], c["shared_tariff_cost_reward"]
                        - c["bat_terminal_soc_penalty_component"]
                        - c["bat_degradation_penalty_component"])
    assert c["ls_terminal_backlog_tasks"] == 4.0
    assert math.isclose(c["ls_terminal_backlog_penalty_component"], 2.0)


def test_environment_terminal_soc_penalty_is_scaled_restoration_cost():
    from types import SimpleNamespace
    from sustaindc_env import SustainDC

    def settle(scale):
        env = object.__new__(SustainDC)
        env.reward_config = {"mode": "team_tariff_targeted_safeguards", "scale": scale}
        env._episode_start_soc = 0.5
        env._terminal_soc_settled = False
        env.bat_env = SimpleNamespace(
            get_battery_soc=lambda: 0.25, max_bat_cap=2.0,
            battery=SimpleNamespace(eff_c=0.9),
        )
        env.price_m = SimpleNamespace(get_current_price_denorm=lambda: 8.0)
        return env._terminal_soc_settlement(True)

    low, high = settle(0.01), settle(0.02)
    restore_kwh = 0.25 * 2.0 * 1000.0 / 0.9
    assert math.isclose(low["bat_terminal_soc_restore_grid_kwh"], restore_kwh)
    assert math.isclose(low["bat_terminal_soc_penalty_component"], 0.01 * restore_kwh * 8.0)
    assert math.isclose(high["bat_terminal_soc_penalty_component"],
                        2.0 * low["bat_terminal_soc_penalty_component"])


def test_legacy_price_reward_and_reward_registry_remain_available():
    assert reward_creator.default_price_reward(
        {"total_price_cost_this_step_c": 250.0}
    ) == -2.5
    for name in reward_creator.REWARD_METHOD_MAP:
        assert reward_creator.get_reward_method(name) is reward_creator.REWARD_METHOD_MAP[name]


if __name__ == "__main__":
    sys.exit(1 if run_module_tests(globals()) else 0)
