"""Standalone integration checks for SustainDC's pluggable pricing path."""
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests._harness import run_module_tests
from sustaindc_env import EnvConfig, SustainDC


_ENV_CONFIG = {
    "agents": ["agent_ls", "agent_dc", "agent_bat"],
    "location": "ca",
    "cintensity_file": "CA_NG_&_avgCI.csv",
    "workload_file": "Alibaba_CPU_Data_Hourly_1.csv",
    "datacenter_capacity_mw": 1,
    "month": 6,
    "days_per_episode": 1,
    "sample_whole_year": False,
    "bat_reward": "default_price_reward",
    "pricing": {
        "model": "hydro_quebec",
        "config_file": "data/Pricing/hydro_quebec_2026.yaml",
        "options": {"tariff": "auto", "demand_floor_kw": 0.0},
    },
}


def _make_env():
    return SustainDC(dict(_ENV_CONFIG))


def _step_actions(env):
    return {
        agent: env.action_space[index].sample()
        for index, agent in enumerate(env.agents)
    }


def test_default_yaml_contains_nested_pricing_only():
    env_args = yaml.safe_load(
        (_REPO_ROOT / "harl" / "configs" / "envs_cfgs" / "sustaindc.yaml")
        .read_text(encoding="utf-8")
    )
    assert set(env_args["pricing"]) == {"model", "config_file", "options"}
    assert "tariff_rate_override" not in env_args
    assert "demand_floor_kw" not in env_args


def test_env_defaults_deep_copy_nested_pricing_options():
    first = EnvConfig({})
    second = EnvConfig({})
    first["pricing"]["options"]["tariff"] = "rate_l"
    assert second["pricing"]["options"]["tariff"] == "auto"


def test_env_rejects_removed_flat_pricing_keys():
    for key in ("tariff_rate_override", "demand_floor_kw"):
        try:
            EnvConfig({key: None})
        except ValueError as exc:
            assert key in str(exc)
        else:
            raise AssertionError(f"expected removed config key {key!r} to fail")


def test_env_constructs_resets_and_steps_with_nested_pricing():
    env = _make_env()
    observation = env.reset()
    assert set(observation) == set(_ENV_CONFIG["agents"])
    next_observation, rewards, terminated, truncated, info = env.step(_step_actions(env))
    assert set(next_observation) == set(_ENV_CONFIG["agents"])
    assert set(rewards) == set(_ENV_CONFIG["agents"])
    assert "__common__" in info
    assert all(agent in terminated for agent in _ENV_CONFIG["agents"])
    assert all(agent in truncated for agent in _ENV_CONFIG["agents"])


def test_info_uses_model_neutral_charge_keys_only():
    env = _make_env()
    env.reset()
    _, _, _, _, info = env.step(_step_actions(env))
    common = info["__common__"]
    assert {
        "energy_cost_this_step_c",
        "demand_charge_increment_c",
        "additional_charge_increment_c",
        "total_price_cost_this_step_c",
    } <= set(common)
    assert "optimization_charge_increment_c" not in common


def test_meter_reading_uses_pre_advance_pricing_clock_at_midnight():
    """Energy from the interval ending at midnight belongs to the clock state
    agents acted on, not the next-day clock produced by Time_Manager.step()."""
    env = _make_env()
    env.reset()
    env.t_m.day = 333
    env.t_m.hour = 23.75
    env.current_hour = 23.75
    captured = {}
    original_step = env.price_m.step

    def capture_step(**kwargs):
        captured.update(kwargs)
        return original_step(**kwargs)

    env.price_m.step = capture_step
    env.step(_step_actions(env))
    assert captured["day_of_year"] == 333
    assert captured["hour"] == 23.75
    assert env.t_m.day == 334
    assert env.t_m.hour == 0.0


def test_price_feature_block_remains_three_values():
    env = _make_env()
    env.reset()
    features = env._price_feature_block()
    assert features.shape == (3,)
    assert features.dtype.name == "float32"


if __name__ == "__main__":
    sys.exit(1 if run_module_tests(globals()) else 0)
