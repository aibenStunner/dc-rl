"""Reward routing checks for SustainDC's HARL integration."""
import inspect
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests._harness import run_module_tests
from harl.common.buffers.on_policy_critic_buffer_fp import OnPolicyCriticBufferFP
from harl.common.valuenorm import ValueNorm
from harl.envs.sustaindc.harlsustaindc_env import fp_critic_observations
from harl.runners.on_policy_base_runner import OnPolicyBaseRunner, select_ep_team_rewards
from harl.common.base_logger import BaseLogger


def test_ep_rejects_heterogeneous_rewards_and_accepts_equal_rewards():
    heterogeneous = np.array([[[1.0], [2.0], [3.0]]], dtype=np.float32)
    equal = np.array([[[2.0], [2.0], [2.0]]], dtype=np.float32)

    try:
        select_ep_team_rewards(heterogeneous)
    except ValueError as exc:
        assert "EP requires identical per-agent rewards" in str(exc)
    else:
        raise AssertionError("expected heterogeneous EP rewards to be rejected")

    np.testing.assert_array_equal(
        select_ep_team_rewards(equal),
        np.array([[2.0]], dtype=np.float32),
    )


def test_fp_critic_observations_append_one_hot_agent_identity():
    states = fp_critic_observations(np.array([10.0, 20.0], dtype=np.float32), 3)
    np.testing.assert_array_equal(
        states,
        np.array(
            [
                [10.0, 20.0, 1.0, 0.0, 0.0],
                [10.0, 20.0, 0.0, 1.0, 0.0],
                [10.0, 20.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )


class _Writer:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))


class _Logger(BaseLogger):
    def get_task_name(self):
        return "test"


class _CriticBuffer:
    def get_mean_rewards(self):
        return 0.0


def test_base_logger_keeps_completed_agent_returns_distinct():
    writer = _Writer()
    logger = _Logger(
        {"env": "sustaindc", "algo": "happo", "exp_name": "test"},
        {"train": {"n_rollout_threads": 1, "episode_length": 1, "num_env_steps": 1}},
        {}, 3, writer, ".",
    )
    logger.init(1)
    logger.episode_init(1)
    logger.per_step((
        None, None,
        np.array([[[-4.0], [-3.0], [-2.0]]], dtype=np.float32),
        np.ones((1, 3, 1), dtype=bool),
        None, None, None, None, None, None, None,
    ))
    logger.episode_log([{}, {}, {}], {}, None, _CriticBuffer())
    values = {tag: value for tag, value, _ in writer.scalars}
    assert values["train/episode/reward/agent_ls/return"] == -4.0
    assert values["train/episode/reward/agent_dc/return"] == -3.0
    assert values["train/episode/reward/agent_bat/return"] == -2.0
    assert values["train/episode/reward/mean_across_agents/return"] == -3.0
    logger.close()


def test_fp_buffer_preserves_distinct_agent_reward_columns():
    args = {
        "episode_length": 1,
        "n_rollout_threads": 1,
        "hidden_sizes": [4],
        "recurrent_n": 1,
        "gamma": 1.0,
        "gae_lambda": 1.0,
        "use_gae": False,
        "use_proper_time_limits": False,
    }
    from gymnasium import spaces

    buffer = OnPolicyCriticBufferFP(args, spaces.Box(-1.0, 1.0, shape=(5,)), 3)
    rewards = np.array([[[1.0], [2.0], [3.0]]], dtype=np.float32)
    buffer.insert(
        np.zeros((1, 3, 5), dtype=np.float32),
        np.zeros((1, 3, 1, 4), dtype=np.float32),
        np.zeros((1, 3, 1), dtype=np.float32),
        rewards,
        np.ones((1, 3, 1), dtype=np.float32),
        np.ones((1, 3, 1), dtype=np.float32),
    )
    np.testing.assert_array_equal(buffer.rewards[0], rewards)
    buffer.compute_returns(np.zeros((1, 3, 1), dtype=np.float32))
    np.testing.assert_array_equal(buffer.returns[0], rewards)


def test_value_normalizer_registers_statistics_on_non_cpu_device():
    normalizer = ValueNorm(1, device=torch.device("meta"))
    assert set(normalizer.state_dict()) == {
        "running_mean", "running_mean_sq", "debiasing_term"
    }


def test_runner_does_not_use_removed_numpy_inf():
    assert "np.Inf" not in inspect.getsource(OnPolicyBaseRunner.__init__)


def test_cpu_checkpoint_loader_uses_cpu_map_location():
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "value_normalizer.pt"
        torch.save(ValueNorm(1).state_dict(), checkpoint)
        runner = object.__new__(OnPolicyBaseRunner)
        runner.device = torch.device("cpu")
        runner.algo_args = {"train": {"model_dir": directory}}
        state = runner._load_checkpoint("value_normalizer.pt")
        assert set(state) == set(ValueNorm(1).state_dict())
        assert all(value.device.type == "cpu" for value in state.values())


def test_empty_value_normalizer_checkpoint_is_rejected():
    with tempfile.TemporaryDirectory() as directory:
        torch.save({}, Path(directory) / "value_normalizer.pt")
        runner = object.__new__(OnPolicyBaseRunner)
        runner.device = torch.device("cpu")
        runner.algo_args = {"train": {"model_dir": directory}}
        runner.value_normalizer = ValueNorm(1)
        try:
            runner._restore_value_normalizer()
        except ValueError as exc:
            assert "incomplete" in str(exc)
            assert "running_mean" in str(exc)
        else:
            raise AssertionError("empty normalizer checkpoint must not be silently reinitialized")


if __name__ == "__main__":
    sys.exit(1 if run_module_tests(globals()) else 0)
