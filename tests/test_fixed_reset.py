"""Fixed evaluation resets and wrapper forwarding."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np

from tests._harness import run_module_tests
from tests.test_observation_layout import _get_env, _ENV_CONFIG
from sustaindc_env import SustainDC
from harl.envs.sustaindc.harlsustaindc_env import HARLSustainDCEnv
from harl.envs.sustaindc.sustaindc_ptzoo import SustainDCPettingZooEnv
from harl.common.buffers.on_policy_critic_buffer_fp import OnPolicyCriticBufferFP
from harl.runners.on_policy_base_runner import OnPolicyBaseRunner
from gymnasium.spaces import Box


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


def test_calendar_month_reset_uses_actual_month_lengths():
    config = dict(_ENV_CONFIG, episode_start_policy='calendar_month', days_per_episode=30)
    env = SustainDC(config)
    for start_day, days in ((0, 31), (31, 28), (59, 31), (334, 31)):
        env.reset(seed=0, options={'init_day': start_day, 'init_hour': 0})
        assert env.t_m.simulated_total_timesteps == days * 96
        assert env.t_m.total_timesteps == (start_day + days) * 96
    env.reset(seed=0, options={'init_day': 334, 'init_hour': 0})
    for step in range(31 * 96):
        _, _, terminated, truncated, _ = env.step(
            {'agent_ls': 1, 'agent_dc': 1, 'agent_bat': 2})
    assert not terminated['__all__']
    assert truncated['__all__']
    assert env.t_m.day == 365
    try:
        env.reset(seed=0, options={'init_day': 14, 'init_hour': 0})
    except ValueError:
        pass
    else:
        raise AssertionError('calendar month accepted a mid-month reset')


def test_calendar_month_random_reset_samples_all_seasons():
    config = dict(_ENV_CONFIG, episode_start_policy='calendar_month', days_per_episode=30)
    env = SustainDC(config)
    starts = set()
    for seed in range(120):
        env.reset(seed=seed)
        starts.add(env.t_m.day)
    assert 0 in starts and 334 in starts
    assert starts <= {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334}


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


def test_settled_episode_reward_stops_fp_critic_bootstrap():
    config = dict(
        _ENV_CONFIG, location="montreal", datacenter_capacity_mw=1.0,
        days_per_episode=30, episode_start_policy="valid_midnight",
        battery={"initial_soc": 0.5},
        reward={"mode": "team_tariff_targeted_safeguards",
                "ls_terminal_backlog_penalty_weight": 0.5},
        state_type="FP", partial_obs=True,
        nonoverlapping_shared_obs_space=True,
    )
    env = HARLSustainDCEnv(config)
    try:
        env.reset(seed=102, options={"init_day": 14, "init_hour": 0})
        for step in range(2880):
            _, shared, rewards, dones, infos, _ = env.step(
                np.array([0, 1, 1 if step == 2879 else 2])
            )
            if step < 2879:
                assert not any(dones)
        assert dones == [True, True, True]
        common = infos[0]
        assert common["isterminal"]
        assert common["ls_terminal_backlog_penalty_component"] > 0
        assert common["total_price_cost_this_step_c"] >= 0
        assert common["bat_terminal_soc_penalty_component"] > 0
        assert common["bat_terminal_soc_restore_grid_kwh"] > 0
        for agent, reward in zip(("agent_ls", "agent_dc", "agent_bat"), rewards):
            np.testing.assert_allclose(reward[0], common[f"{agent}_reward"])
        assert all("bad_transition" not in info for info in infos)

        buffer = OnPolicyCriticBufferFP({
            "episode_length": 1, "n_rollout_threads": 1,
            "hidden_sizes": [4], "recurrent_n": 1,
            "gamma": 0.99, "gae_lambda": 0.95,
            "use_gae": True, "use_proper_time_limits": True,
        }, Box(-10, 10, shape=(len(shared[0]),), dtype=np.float32), 3)
        reward_array = np.asarray(rewards, dtype=np.float32)[None, :, :]
        runner = object.__new__(OnPolicyBaseRunner)
        runner.num_agents = 3
        runner.state_type = "FP"
        runner.recurrent_n = 1
        runner.rnn_hidden_size = 4
        runner.algo_args = {"train": {"n_rollout_threads": 1}}
        runner.actor_buffer = []
        runner.critic_buffer = buffer
        class ActorBuffer:
            def insert(self, *args):
                pass
        runner.actor_buffer = [ActorBuffer() for _ in range(3)]
        runner.insert((
            np.zeros((1, 3, 40), dtype=np.float32),
            np.asarray(shared, dtype=np.float32)[None, :, :],
            reward_array,
            np.asarray(dones, dtype=bool)[None, :],
            [infos],
            [None],
            np.full((1, 3, 1), 123.0, dtype=np.float32),
            np.zeros((1, 3, 1), dtype=np.float32),
            np.zeros((1, 3, 1), dtype=np.float32),
            np.zeros((1, 3, 1, 4), dtype=np.float32),
            np.zeros((1, 3, 1, 4), dtype=np.float32),
        ))
        np.testing.assert_array_equal(buffer.masks[1], 0.0)
        np.testing.assert_array_equal(buffer.bad_masks[1], 1.0)
        buffer.compute_returns(np.full((1, 3, 1), 987.0, dtype=np.float32))
        np.testing.assert_allclose(buffer.returns[0], reward_array, atol=1e-3)
    finally:
        env.close()


if __name__ == "__main__":
    run_module_tests(globals())
