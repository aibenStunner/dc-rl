"""Regression tests for standalone checkpoint evaluation timing."""
import json
import os
import runpy
import sys
import tempfile
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests._harness import run_module_tests
from harl.envs.sustaindc.sequential import sequential_battery_index


class _Actor:
    def __init__(self, agent_id, events):
        self.agent_id = agent_id
        self.events = events

    def act(self, obs, state, mask, available, deterministic=False):
        assert deterministic
        self.events.append(('act', self.agent_id, np.array(obs, copy=True)))
        return torch.tensor([[self.agent_id]]), torch.as_tensor(state)


class _EvalEnv:
    def __init__(self, events, current):
        self.events = events
        self.current = current
        self.pending = None

    def reset(self):
        return (np.zeros((1, 3, 4), dtype=np.float32),
                np.zeros((1, 3, 4), dtype=np.float32),
                np.ones((1, 3, 3), dtype=np.float32))

    def begin_step(self, actions):
        assert self.current
        assert self.pending is None
        assert actions.shape == (1, 2, 1)
        self.events.append(('begin', np.array(actions, copy=True)))
        self.pending = np.array(actions, copy=True)
        return np.full((1, 4), 7.0, dtype=np.float32)

    def step(self, actions):
        if self.current:
            if self.pending is None:
                raise RuntimeError('current timing requires begin_step before step')
            assert np.array_equal(actions[:, :2], self.pending)
            self.pending = None
        else:
            assert self.pending is None
        self.events.append(('step', np.array(actions, copy=True)))
        return (*self.reset()[:2], np.zeros((1, 3, 1), dtype=np.float32),
                np.ones((1, 3), dtype=bool), [[{}, {}, {}]], self.reset()[2])


def _run_checkpoint_eval(current):
    from harl.runners import RUNNER_REGISTRY

    events = []
    runners = []

    class Runner:
        def __init__(self, main_args, algo_args, env_args):
            self.sequential_battery_index = sequential_battery_index(env_args)
            self.num_agents = 3
            self.recurrent_n = self.rnn_hidden_size = 1
            self.algo_args = algo_args
            self.eval_envs = _EvalEnv(events, current)
            self.actor = [_Actor(i, events) for i in range(3)]
            self.dump_info = False
            self.closed = False
            runners.append(self)

        def prep_rollout(self):
            pass

        def close(self):
            self.closed = True

    config = {
        'algo_args': {'train': {}, 'eval': {}, 'logger': {}},
        'env_args': {'battery': {'observation_timing': 'current'}} if current else {},
        'main_args': {'algo': 'haa2c', 'env': 'sustaindc', 'exp_name': 'test'},
    }
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = (Path(tmp) / 'trained_models/sustaindc/az/haa2c/haa2c3agents/'
                      'seed-00001-2024-06-04-20-41-56')
        checkpoint.mkdir(parents=True)
        (checkpoint / 'config.json').write_text(json.dumps(config))
        with patch.dict(RUNNER_REGISTRY, {'haa2c': Runner}):
            # The script resolves its checkpoint relative to the working directory.
            previous_cwd = os.getcwd()
            try:
                os.chdir(tmp)
                with patch.object(sys, 'path', list(sys.path)), warnings.catch_warnings():
                    runpy.run_path(str(_REPO_ROOT / 'eval_sustaindc.py'), run_name='__main__')
            finally:
                os.chdir(previous_cwd)
    assert len(runners) == 1 and runners[0].closed
    return events


def test_standalone_eval_current_timing_uses_mid_step_battery_observation():
    events = _run_checkpoint_eval(True)
    assert [event[0] if event[0] != 'act' else event[1] for event in events] == [0, 1, 'begin', 2, 'step']
    assert np.array_equal(events[2][1], np.array([[[0], [1]]]))
    assert np.all(events[0][2] == 0.0) and np.all(events[1][2] == 0.0)
    assert np.all(events[3][2] == 7.0)
    assert np.array_equal(events[4][1], np.array([[[0], [1], [2]]]))


def test_standalone_eval_previous_timing_skips_begin_step():
    events = _run_checkpoint_eval(False)
    assert [event[0] if event[0] != 'act' else event[1] for event in events] == [0, 1, 2, 'step']
    assert np.all(events[2][2] == 0.0)


if __name__ == '__main__':
    sys.exit(run_module_tests(globals()))
