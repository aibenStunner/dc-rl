"""Fixed evaluation resets and wrapper forwarding."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

from tests._harness import run_module_tests
from tests.test_observation_layout import _get_env, _ENV_CONFIG
from harl.envs.sustaindc.harlsustaindc_env import HARLSustainDCEnv
from harl.envs.sustaindc.sustaindc_ptzoo import SustainDCPettingZooEnv


def test_fixed_reset_alignment_and_seed_zero():
    env = _get_env()
    options = {"init_day": 105, "init_hour": 6}
    first = env.reset(seed=0, options=options)
    index = (105 * 24 + 6) * 4
    assert env.t_m.current_timestep == index
    for manager in (env.workload_m, env.weather_m, env.ci_m, env.pv_m):
        assert manager.time_step == index
    soc = env.bat_env.get_battery_soc()
    second = env.reset(seed=0, options=options)
    for agent in first:
        np.testing.assert_array_equal(first[agent], second[agent])
    assert soc == env.bat_env.get_battery_soc()


def test_invalid_windows():
    env = _get_env()
    for options in ({"init_day": 336, "init_hour": 0},
                    {"init_day": 335, "init_hour": 0},
                    {"init_day": -1, "init_hour": 0},
                    {"init_day": 14}, {"init_day": True, "init_hour": 0},
                    {"init_day": 14, "init_hour": 24}, {"other": 1}):
        try:
            env.reset(seed=0, options=options)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid options {options}")


def test_exogenous_streams_independent_of_actions():
    env = _get_env()
    traces = []
    for actions in ({"agent_ls": 1, "agent_dc": 1, "agent_bat": 2},
                    {"agent_ls": 0, "agent_dc": 0, "agent_bat": 0}):
        env.reset(seed=0, options={"init_day": 196, "init_hour": 0})
        trace = []
        for _ in range(96):
            env.step(actions)
            trace.append((env.workload_m.get_current_workload(),
                          env.weather_m.get_current_temperature(),
                          env.ci_m.get_current_ci(), env.pv_m.get_current_ac_kwh()))
        traces.append(trace)
    np.testing.assert_array_equal(traces[0], traces[1])


def test_full_window_truncates_at_2880():
    env = _get_env()
    env.reset(seed=0, options={"init_day": 287, "init_hour": 0})
    for step in range(2880):
        _, _, terminated, truncated, _ = env.step(
            {"agent_ls": 1, "agent_dc": 1, "agent_bat": 2}
        )
        assert not terminated["__all__"]
        assert truncated["__all__"] == (step == 2879)


def test_wrappers_forward_seed_and_options():
    opts = {"init_day": 14, "init_hour": 0}
    config = dict(_ENV_CONFIG, partial_obs=True, nonoverlapping_shared_obs_space=True)
    for wrapper in (SustainDCPettingZooEnv(config), HARLSustainDCEnv(config)):
        first = wrapper.reset(seed=0, options=opts)
        second = wrapper.reset(seed=0, options=opts)
        if isinstance(first, tuple):
            np.testing.assert_array_equal(first[0], second[0])
        else:
            for agent in first:
                np.testing.assert_array_equal(first[agent], second[agent])


def test_harl_adapter_maps_truncation_to_done_without_bad_transition():
    class FakeParallelEnv:
        def step(self, actions):
            obs = {agent: np.array([index]) for index, agent in enumerate(AGENTS)}
            rewards = {agent: 0.0 for agent in AGENTS}
            terminated = {agent: False for agent in AGENTS}
            truncated = {agent: True for agent in AGENTS}
            info = {agent: {} for agent in AGENTS}
            return obs, rewards, terminated, truncated, info

    AGENTS = ("agent_ls", "agent_dc", "agent_bat")
    wrapper = object.__new__(HARLSustainDCEnv)
    wrapper.agents = list(AGENTS)
    wrapper.env = FakeParallelEnv()
    wrapper.wrap = lambda actions: dict(zip(AGENTS, actions))
    wrapper.unwrap = lambda value: [value[agent] for agent in AGENTS]
    wrapper._create_shared_observation = lambda observations: observations
    wrapper.get_avail_actions = lambda: None

    _, _, _, dones, infos, _ = wrapper.step(np.array([[1], [1], [2]]))
    assert dones == [True, True, True]
    assert all("bad_transition" not in info for info in infos)


if __name__ == "__main__":
    run_module_tests(globals())
