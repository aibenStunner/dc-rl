"""A closed ``HARLSustainDCEnv`` frees its simulator.

SuperSuit 3.7.0 memoizes the padding wrappers' ``action_space(self, agent)``
and ``observation_space(self, agent)`` with ``functools.lru_cache`` on the
class, so the cache holds every wrapper ever built, and through it the whole
``SustainDC`` underneath (about 33 MB at 10 MW). Fixed-month validation builds
a new environment per case, so a training run grew by one simulator per case
until the host ran out of memory. Closing an environment must release it, and
must leave any other open environment working.
"""
import gc
import sys
import weakref
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch

from tests._harness import run_module_tests
import harl.envs.sustaindc.fixed_month_validation as fixed_validation
from harl.envs.sustaindc.fixed_month_validation import ValidationCase, validate_fixed_months
from harl.envs.sustaindc.harlsustaindc_env import HARLSustainDCEnv
from harl.utils.configs_tools import get_defaults_yaml_args


def _shipped_env_args():
    return get_defaults_yaml_args('happo', 'sustaindc')[1]


def _simulator(env):
    """The ``SustainDC`` under the HARL adapter and its SuperSuit padding."""
    return env.env.unwrapped.env


def test_a_closed_environment_frees_its_simulator():
    env = HARLSustainDCEnv(_shipped_env_args())
    simulator = weakref.ref(_simulator(env))
    env.close()
    del env
    gc.collect()
    assert simulator() is None, gc.get_referrers(simulator())


class _HoldActor:
    def act(self, observations, state, masks, available_actions, deterministic):
        return torch.tensor([[1]]), torch.from_numpy(state)


def test_repeated_validation_cases_leave_no_simulator_alive():
    env_args = _shipped_env_args()
    simulators = []

    class Recording(HARLSustainDCEnv):
        def __init__(self, args):
            super().__init__(args)
            simulators.append(weakref.ref(_simulator(self)))

    cases = (ValidationCase('winter', 0), ValidationCase('spring', 90))
    with patch.object(fixed_validation, 'HARLSustainDCEnv', Recording), \
         patch.object(fixed_validation, 'month_steps', return_value=2):
        validate_fixed_months(env_args=env_args, actors=[_HoldActor() for _ in env_args['agents']],
                              recurrent_n=1, rnn_hidden_size=64, seed=102, event=1, cases=cases)
    gc.collect()
    assert len(simulators) == len(cases), len(simulators)
    assert [ref() is None for ref in simulators] == [True] * len(cases)


def test_closing_one_environment_leaves_another_working():
    kept = HARLSustainDCEnv(_shipped_env_args())
    spaces = (kept.observation_space, kept.action_space, kept.share_observation_space)
    closed = HARLSustainDCEnv(_shipped_env_args())
    closed.close()
    del closed
    gc.collect()
    observations, _, _ = kept.reset(seed=0, options={'init_day': 0, 'init_hour': 0})
    # The shipped battery sees the current interval: workload and cooling act first, battery last.
    kept.begin_step(np.ones(kept.n_agents - 1, dtype=np.int64))
    observations, _, _, _, _, _ = kept.step(np.ones(kept.n_agents, dtype=np.int64))
    for agent_id, agent in enumerate(kept.agents):
        assert kept.env.observation_space(agent) == spaces[0][agent_id]
        assert kept.env.action_space(agent) == spaces[1][agent_id]
        assert observations[agent_id].shape == spaces[0][agent_id].shape
    assert (kept.observation_space, kept.action_space, kept.share_observation_space) == spaces
    kept.close()


if __name__ == '__main__':
    sys.exit(1 if run_module_tests(globals()) else 0)
