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
            "reward": {
                "mode": "team_tariff_targeted_safeguards",
                "degradation_cost_weight": 0.1,
            }
        },
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


def test_legacy_price_reward_and_reward_registry_remain_available():
    assert reward_creator.default_price_reward(
        {"total_price_cost_this_step_c": 250.0}
    ) == -2.5
    for name in reward_creator.REWARD_METHOD_MAP:
        assert reward_creator.get_reward_method(name) is reward_creator.REWARD_METHOD_MAP[name]


if __name__ == "__main__":
    sys.exit(1 if run_module_tests(globals()) else 0)
