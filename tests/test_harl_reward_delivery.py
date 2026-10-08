"""Reward routing checks for SustainDC's HARL integration."""
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests._harness import run_module_tests
from harl.common.buffers.on_policy_critic_buffer_fp import OnPolicyCriticBufferFP
from harl.envs.sustaindc.harlsustaindc_env import fp_critic_observations
from harl.runners.on_policy_base_runner import select_ep_team_rewards


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


if __name__ == "__main__":
    sys.exit(1 if run_module_tests(globals()) else 0)
