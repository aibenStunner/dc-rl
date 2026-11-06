"""Sequential battery observation: the battery sees this interval's gross load.

With ``battery.observation_timing: current`` the workload and cooling actions
are applied first (``begin_step``), and the battery then observes the facility
load they produced before choosing its own action (``step``). Physics, billing
and the post-step observations must be identical to ``previous`` timing; only
the battery's decision observation changes.
"""
import math
import sys
import threading
from multiprocessing import Pipe
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests._harness import run_module_tests
from sustaindc_env import EnvConfig, SustainDC

GROSS = SustainDC.BAT_IDX['previous_facility_load_ratio']
_ACTIONS = [(0, 0, 0), (1, 1, 1), (2, 2, 2), (1, 0, 2), (0, 2, 1), (2, 1, 0)]


def _expect(exc_type, fragment, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type as exc:
        assert fragment in str(exc), f'{fragment!r} not in {exc!r}'
        return exc
    raise AssertionError(f'{fn!r} did not raise {exc_type.__name__}')


def _env(timing='current'):
    # The shipped reward mode: the legacy default_* rewards share a
    # process-global energy normalizer, so two env instances in one process
    # would not reproduce each other's rewards regardless of timing.
    env = SustainDC(EnvConfig({
        'month': 6, 'episode_start_policy': 'valid_midnight',
        'reward': {'mode': 'team_tariff_targeted_safeguards'},
        'battery': {'initial_soc': 0.5, 'observation_timing': timing},
    }))
    env.reset(seed=102, options={'init_day': 14, 'init_hour': 0})
    return env


def _ls_dc(actions):
    return {'agent_ls': actions[0], 'agent_dc': actions[1]}


def _all(actions):
    return {'agent_ls': actions[0], 'agent_dc': actions[1], 'agent_bat': actions[2]}


def _harl_config(**battery):
    from tests.test_observation_layout import _ENV_CONFIG

    return dict(_ENV_CONFIG, state_type='FP', partial_obs=True,
                nonoverlapping_shared_obs_space=True,
                episode_start_policy='valid_midnight',
                battery={'initial_soc': 0.5, **battery})


# ---------------------------------------------------------------- configuration

def test_default_observation_timing_is_previous():
    assert EnvConfig({})['battery']['observation_timing'] == 'previous'


def test_invalid_observation_timing_is_rejected():
    for value in ('next', None, 1, '', True):
        _expect(ValueError, "observation_timing must be 'previous' or 'current'",
                EnvConfig, {'battery': {'observation_timing': value}})
    for agents in (['agent_ls', 'agent_bat', 'agent_dc'], ['agent_ls', 'agent_dc']):
        _expect(ValueError, 'requires agent_bat as the last agent', EnvConfig,
                {'agents': agents, 'battery': {'observation_timing': 'current'}})
    assert EnvConfig({'battery': {'observation_timing': 'current'}})['battery'][
        'observation_timing'] == 'current'


def test_shipped_training_config_uses_current_timing():
    config = yaml.safe_load((_REPO_ROOT / 'harl/configs/envs_cfgs/sustaindc.yaml').read_text())
    assert config['battery']['observation_timing'] == 'current'
    assert EnvConfig(config)['battery']['observation_timing'] == 'current'


def test_sequential_battery_index():
    from harl.envs.sustaindc.sequential import sequential_battery_index

    assert sequential_battery_index({}) is None
    assert sequential_battery_index({'ignored': True}) is None
    assert sequential_battery_index({'battery': {'observation_timing': 'previous'}}) is None
    assert sequential_battery_index({'battery': {'observation_timing': 'current'}}) == 2


# ------------------------------------------------------------------ environment

def test_previous_timing_rejects_begin_step():
    env = _env('previous')
    _expect(RuntimeError, "observation_timing 'current'", env.begin_step, _ls_dc((1, 1)))
    env.step(_all((1, 1, 2)))


def test_begin_step_reports_the_current_interval_gross_load():
    env = _env()
    cap_kw = env.datacenter_capacity_mw * 1000.0
    changed_steps = 0
    for step in range(8):
        actions = _ACTIONS[step % len(_ACTIONS)]
        before = np.array(env.bat_state, copy=True)
        clock = (env.t_m.day, env.current_hour)
        peak = env.price_m.state.running_peak_kw
        soc = env.bat_env.get_battery_soc()
        mid = env.begin_step(_ls_dc(actions))
        assert mid.shape == before.shape == (29,) and mid.dtype == np.float32
        assert mid is not env.bat_state
        changed = set(np.flatnonzero(mid != before).tolist())
        assert changed <= {GROSS}, changed
        changed_steps += bool(changed)
        gross_kw = env.dc_info['dc_total_power_kW']
        assert mid[GROSS] == np.float32(gross_kw / cap_kw)
        assert (env.t_m.day, env.current_hour) == clock
        assert env.price_m.state.running_peak_kw == peak
        assert env.bat_env.get_battery_soc() == soc
        assert np.array_equal(env.bat_state, before)
        obs, _, _, _, info = env.step(_all(actions))
        assert info['__common__']['dc_total_power_kW'] == gross_kw
        assert obs['agent_bat'][GROSS] == mid[GROSS]
    assert changed_steps > 0


def _same(a, b, path='root'):
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), path
        for key in a:
            _same(a[key], b[key], f'{path}.{key}')
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), path
        for index, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f'{path}[{index}]')
    elif isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray) and a.shape == b.shape and a.dtype == b.dtype, path
        if a.dtype.kind in 'fc':
            assert np.array_equal(a, b, equal_nan=True), path
        else:
            assert np.array_equal(a, b), path
    elif isinstance(a, float):
        assert isinstance(b, float) and (a == b or (math.isnan(a) and math.isnan(b))), path
    else:
        assert a == b, f'{path}: {a!r} != {b!r}'


def _rollout(timing, steps=96):
    env = _env(timing)
    out = []
    for step in range(steps):
        actions = _ACTIONS[step % len(_ACTIONS)]
        if timing == 'current':
            env.begin_step(_ls_dc(actions))
        out.append(env.step(_all(actions)))
    return out, env.bat_env.get_battery_soc(), env.price_m.state.running_peak_kw


def test_current_timing_matches_previous_timing_physics_and_billing():
    previous, previous_soc, previous_peak = _rollout('previous')
    control, _, _ = _rollout('previous')
    _same(previous, control)  # the comparison below is meaningful only if this holds
    current, current_soc, current_peak = _rollout('current')
    _same(previous, current)
    assert previous_soc == current_soc and previous_peak == current_peak
    charged = sum(r[4]['__common__']['bat_charge_bus_KWh'] > 0 for r in current)
    discharged = sum(r[4]['__common__']['bat_discharge_bus_KWh'] > 0 for r in current)
    assert charged > 0 and discharged > 0


def test_step_requires_begin_step_in_current_timing():
    env = _env()
    clock = (env.t_m.day, env.current_hour)
    _expect(RuntimeError, 'requires begin_step before step', env.step, _all((1, 1, 2)))
    assert (env.t_m.day, env.current_hour) == clock


def test_begin_step_cannot_be_repeated_before_step():
    env = _env()
    env.begin_step(_ls_dc((1, 1)))
    _expect(RuntimeError, 'already called', env.begin_step, _ls_dc((1, 1)))
    env.step(_all((1, 1, 2)))
    env.begin_step(_ls_dc((1, 1)))


def test_step_rejects_ls_dc_actions_that_differ_from_begin_step():
    env = _env()
    env.begin_step(_ls_dc((1, 1)))
    clock = (env.t_m.day, env.current_hour)
    soc = env.bat_env.get_battery_soc()
    _expect(ValueError, 'agent_ls differs from begin_step', env.step, _all((0, 1, 2)))
    _expect(ValueError, 'agent_dc differs from begin_step', env.step, _all((1, 2, 2)))
    _expect(ValueError, 'agent_ls differs from begin_step', env.step, {'agent_dc': 1, 'agent_bat': 2})
    assert (env.t_m.day, env.current_hour) == clock
    assert env.bat_env.get_battery_soc() == soc
    env.step(_all((1, 1, 2)))
    assert (env.t_m.day, env.current_hour) != clock


def test_begin_step_requires_exactly_the_non_battery_actions():
    env = _env()
    _expect(ValueError, 'expects actions for exactly', env.begin_step, {'agent_ls': 1})
    _expect(ValueError, 'expects actions for exactly', env.begin_step, _all((1, 1, 2)))
    env.begin_step(_ls_dc((1, 1)))
    env.step(_all((1, 1, 2)))


def test_reset_clears_a_pending_begin_step():
    env = _env()
    env.begin_step(_ls_dc((1, 1)))
    env.reset(seed=102, options={'init_day': 14, 'init_hour': 0})
    _expect(RuntimeError, 'requires begin_step before step', env.step, _all((1, 1, 2)))
    env.begin_step(_ls_dc((1, 1)))
    env.step(_all((1, 1, 2)))


# --------------------------------------------------------------------- wrappers

def test_harl_wrapper_begin_step_pads_the_battery_observation():
    from harl.envs.sustaindc.harlsustaindc_env import HARLSustainDCEnv

    env = HARLSustainDCEnv(_harl_config(observation_timing='current'))
    obs, _, _ = env.reset(seed=102, options={'init_day': 14, 'init_hour': 0})
    raw = env.env.unwrapped.env
    before = np.array(raw.bat_state, copy=True)
    _expect(ValueError, 'expects 2 actions', env.begin_step, np.array([[1], [1], [2]]))
    mid = env.begin_step(np.array([[1], [1]]))
    assert mid.shape == np.asarray(obs[2]).shape == (40,)
    assert mid.dtype == np.asarray(obs[2]).dtype
    expected = before.copy()
    expected[GROSS] = np.float32(raw.dc_info['dc_total_power_kW'] / (raw.datacenter_capacity_mw * 1000.0))
    assert np.array_equal(mid[:len(before)], expected)
    assert not np.any(mid[len(before):])
    next_obs = env.step(np.array([[1], [1], [2]]))[0]
    assert next_obs[2][GROSS] == mid[GROSS]


def test_dummy_vec_env_begin_step_with_the_real_environment():
    from harl.envs.env_wrappers import ShareDummyVecEnv
    from harl.envs.sustaindc.harlsustaindc_env import HARLSustainDCEnv

    vec = ShareDummyVecEnv([lambda: HARLSustainDCEnv(_harl_config(observation_timing='current'))])
    vec.reset()
    mid = vec.begin_step(np.array([[[1], [1]]]))
    assert mid.shape == (1, 40)
    obs = vec.step(np.array([[[1], [1], [2]]]))[0]
    assert obs[0][2][GROSS] == mid[0][GROSS]


class _FakeBeginEnv:
    def __init__(self, value):
        self.value, self.calls, self.closed = value, [], False

    def begin_step(self, actions):
        self.calls.append(np.array(actions, copy=True))
        return np.full(40, self.value, dtype=np.float32)

    def close(self):
        self.closed = True


def test_dummy_vec_env_begin_step_forwards_each_thread():
    from harl.envs.env_wrappers import ShareDummyVecEnv

    vec = object.__new__(ShareDummyVecEnv)
    vec.envs = [_FakeBeginEnv(1.0), _FakeBeginEnv(2.0)]
    actions = np.array([[[0], [1]], [[2], [0]]])
    out = vec.begin_step(actions)
    assert out.shape == (2, 40)
    assert np.all(out[0] == 1.0) and np.all(out[1] == 2.0)
    for env, thread_actions in zip(vec.envs, actions):
        assert len(env.calls) == 1 and np.array_equal(env.calls[0], thread_actions)


def test_subproc_vec_env_begin_step_sends_all_before_receiving():
    from harl.envs.env_wrappers import ShareSubprocVecEnv

    log = []

    class Remote:
        def __init__(self, value):
            self.value = value

        def send(self, message):
            log.append(('send', self.value, message))

        def recv(self):
            assert sum(entry[0] == 'send' for entry in log) == 2
            log.append(('recv', self.value))
            return np.full(40, self.value, dtype=np.float32)

    vec = object.__new__(ShareSubprocVecEnv)
    vec.remotes = [Remote(1.0), Remote(2.0)]
    actions = np.array([[[0], [1]], [[2], [0]]])
    out = vec.begin_step(actions)
    assert out.shape == (2, 40)
    assert np.all(out[0] == 1.0) and np.all(out[1] == 2.0)
    sends = [entry for entry in log if entry[0] == 'send']
    for (_, _, (command, data)), thread_actions in zip(sends, actions):
        assert command == 'begin_step' and np.array_equal(data, thread_actions)


def test_shareworker_dispatches_begin_step():
    from harl.envs.env_wrappers import shareworker

    class ParentEnd:
        def close(self):
            pass

    fake = _FakeBeginEnv(3.0)
    parent, child = Pipe()
    worker = threading.Thread(
        target=shareworker, args=(child, ParentEnd(), SimpleNamespace(x=lambda: fake)), daemon=True)
    worker.start()
    try:
        parent.send(('begin_step', np.array([[1], [0]])))
        assert parent.poll(5), 'worker did not answer begin_step'
        out = parent.recv()
        assert out.shape == (40,) and np.all(out == 3.0)
        assert np.array_equal(fake.calls[0], np.array([[1], [0]]))
    finally:
        if worker.is_alive():
            parent.send(('close', None))
            worker.join(5)
    assert fake.closed


# ---------------------------------------------------------------------- runners

class _CollectActor:
    def __init__(self, agent_id, action, log):
        self.agent_id, self.action, self.log = agent_id, action, log

    def get_actions(self, obs, rnn_states, masks, available_actions):
        self.log.append(('act', self.agent_id, np.array(obs, copy=True)))
        n = obs.shape[0]
        return (torch.full((n, 1), self.action, dtype=torch.int64), torch.zeros(n, 1),
                torch.zeros(n, 1, 4))


class _Critic:
    def get_values(self, share_obs, rnn_states, masks):
        n = share_obs.shape[0]
        return torch.zeros(n, 1), torch.zeros(n, 1, 4)


class _BeginVecEnv:
    def __init__(self, log, allowed=True):
        self.log, self.allowed = log, allowed

    def begin_step(self, actions):
        assert self.allowed, 'begin_step called in previous timing'
        self.log.append(('begin', np.array(actions, copy=True)))
        return np.full((actions.shape[0], 40), 7.0, dtype=np.float32)


def _collect_runner(sequential_index, log, threads=2, steps=3):
    from harl.runners.on_policy_base_runner import OnPolicyBaseRunner

    runner = object.__new__(OnPolicyBaseRunner)
    runner.num_agents = 3
    runner.actor = [_CollectActor(i, a, log) for i, a in enumerate((1, 0, 2))]
    runner.actor_buffer = [SimpleNamespace(
        obs=np.zeros((steps + 1, threads, 40), dtype=np.float32),
        rnn_states=np.zeros((steps + 1, threads, 1, 4), dtype=np.float32),
        masks=np.ones((steps + 1, threads, 1), dtype=np.float32),
        available_actions=None) for _ in range(3)]
    runner.critic = _Critic()
    runner.critic_buffer = SimpleNamespace(
        share_obs=np.zeros((steps + 1, threads, 5), dtype=np.float32),
        rnn_states_critic=np.zeros((steps + 1, threads, 1, 4), dtype=np.float32),
        masks=np.ones((steps + 1, threads, 1), dtype=np.float32))
    runner.state_type = 'EP'
    runner.algo_args = {'train': {'n_rollout_threads': threads}}
    runner.envs = _BeginVecEnv(log, allowed=sequential_index is not None)
    runner.sequential_battery_index = sequential_index
    return runner


def test_collect_feeds_the_battery_actor_the_mid_step_observation():
    log = []
    runner = _collect_runner(2, log)
    _, actions, _, _, _ = runner.collect(1)
    assert [entry[0] if entry[0] == 'begin' else entry[1] for entry in log] == [0, 1, 'begin', 2]
    begin_actions = log[2][1]
    assert begin_actions.shape == (2, 2, 1)
    assert np.array_equal(begin_actions, np.array([[[1], [0]], [[1], [0]]]))
    assert np.all(log[0][2] == 0.0) and np.all(log[1][2] == 0.0)
    assert np.all(log[3][2] == 7.0)
    assert np.all(runner.actor_buffer[2].obs[1] == 7.0)
    assert np.all(runner.actor_buffer[2].obs[0] == 0.0)
    assert np.all(runner.actor_buffer[0].obs[1] == 0.0)
    assert np.array_equal(actions[:, :, 0], np.array([[1, 0, 2], [1, 0, 2]]))


def test_collect_previous_timing_does_not_call_begin_step():
    log = []
    runner = _collect_runner(None, log)
    runner.collect(0)
    assert [entry[1] for entry in log] == [0, 1, 2]
    assert np.all(log[2][2] == 0.0)


class _EvalActor:
    def __init__(self, agent_id, action, log):
        self.agent_id, self.action, self.log = agent_id, action, log

    def act(self, obs, rnn_states, masks, available_actions, deterministic=False):
        assert deterministic
        self.log.append(('act', self.agent_id, np.array(obs, copy=True)))
        n = obs.shape[0]
        return torch.full((n, 1), self.action, dtype=torch.int64), torch.zeros(n, 1, 4)


def test_eval_feeds_the_battery_actor_the_mid_step_observation():
    import harl.runners.on_policy_base_runner as module

    log = []

    class EvalEnvs:
        def reset(self):
            return (np.zeros((1, 3, 40), dtype=np.float32), np.zeros((1, 3, 5), dtype=np.float32),
                    np.ones((1, 3, 3), dtype=np.float32))

        def begin_step(self, actions):
            log.append(('begin', np.array(actions, copy=True)))
            return np.full((1, 40), 7.0, dtype=np.float32)

        def step(self, actions):
            log.append(('step', np.array(actions, copy=True)))
            return (np.zeros((1, 3, 40), dtype=np.float32), np.zeros((1, 3, 5), dtype=np.float32),
                    np.zeros((1, 3, 1), dtype=np.float32), np.ones((1, 3), dtype=bool),
                    [[{}, {}, {}]], np.ones((1, 3, 3), dtype=np.float32))

    logger = SimpleNamespace(eval_init=lambda: None, eval_per_step=lambda data: None,
                             eval_thread_done=lambda tid: None, eval_log=lambda episode: None,
                             avg_eval_episode_reward=0.0)
    runner = object.__new__(module.OnPolicyBaseRunner)
    runner.num_agents, runner.recurrent_n, runner.rnn_hidden_size = 3, 1, 4
    runner.actor = [_EvalActor(i, a, log) for i, a in enumerate((1, 0, 2))]
    runner.algo_args = {'eval': {'n_eval_rollout_threads': 1, 'eval_episodes': 1}}
    runner.eval_event, runner.dump_info = 0, False
    runner.current_best_avg_eval_episode_reward = -np.inf
    runner.logger, runner.eval_envs = logger, EvalEnvs()
    runner.sequential_battery_index = 2
    with patch.object(module, 'build_trace_row', return_value={}):
        runner.eval()
    assert [entry[0] if entry[0] != 'act' else entry[1] for entry in log] == [0, 1, 'begin', 2, 'step']
    assert np.array_equal(log[2][1], np.array([[[1], [0]]]))
    assert np.all(log[3][2] == 7.0) and np.all(log[0][2] == 0.0)
    assert np.array_equal(log[4][1][:, :, 0], np.array([[1, 0, 2]]))


def test_render_feeds_the_battery_actor_the_mid_step_observation():
    from harl.runners.on_policy_base_runner import OnPolicyBaseRunner

    log = []

    class RenderEnv:
        render_episode = 1

        def reset(self):
            return [np.zeros(40, dtype=np.float32)] * 3, None, [np.ones(3)] * 3

        def begin_step(self, actions):
            log.append(('begin', np.array(actions, copy=True)))
            return np.full(40, 7.0, dtype=np.float32)

        def step(self, actions):
            log.append(('step', np.array(actions, copy=True)))
            return ([np.zeros(40, dtype=np.float32)] * 3, None, [[0.0]] * 3, [True] * 3,
                    [{}, {}, {}], [np.ones(3)] * 3)

    runner = object.__new__(OnPolicyBaseRunner)
    runner.num_agents, runner.recurrent_n, runner.rnn_hidden_size = 3, 1, 4
    runner.actor = [_EvalActor(i, a, log) for i, a in enumerate((1, 0, 2))]
    runner.algo_args = {'render': {'render_episodes': 1}}
    runner.env_num, runner.dump_render_info = 1, False
    runner.render_env = RenderEnv()
    runner.sequential_battery_index = 2
    runner.render()
    assert [entry[0] if entry[0] != 'act' else entry[1] for entry in log] == [0, 1, 'begin', 2, 'step']
    assert np.array_equal(log[2][1], np.array([[1], [0]]))
    assert np.all(log[3][2] == 7.0)
    assert np.array_equal(log[4][1][:, 0], np.array([1, 0, 2]))


def test_off_policy_runner_rejects_current_timing():
    from harl.runners.off_policy_base_runner import OffPolicyBaseRunner

    _expect(NotImplementedError, "observation_timing 'current'", OffPolicyBaseRunner,
            {'env': 'sustaindc', 'algo': 'hasac', 'exp_name': 'test'}, {},
            {'battery': {'observation_timing': 'current'}})


# ------------------------------------------------------------------- evaluators

class _RecordingActor:
    def __init__(self, action):
        self.action, self.seen = action, []

    def act(self, obs, state, mask, avail, deterministic):
        assert deterministic
        self.seen.append(np.array(obs, copy=True))
        return torch.tensor([[self.action]]), torch.as_tensor(state)


def _fixed_month_env(log):
    class FakeEnv:
        def __init__(self, env_args):
            self.steps = 0

        def reset(self, *, seed, options):
            return [np.zeros(4, dtype=np.float32)] * 3, None, [np.ones(3)] * 3

        def begin_step(self, actions):
            log.append(('begin', np.asarray(actions).tolist()))
            return np.full(4, 7.0, dtype=np.float32)

        def step(self, actions):
            log.append(('step', np.asarray(actions).tolist()))
            self.steps += 1
            done = self.steps == 2
            info = {'bat_grid_import_kwh': 0.0, 'bat_terminal_soc_restore_grid_kwh': 0.0,
                    '_harl_terminated': False, '_harl_truncated': done}
            return ([np.zeros(4, dtype=np.float32)] * 3, None, [[0.0]] * 3, [done] * 3,
                    [info] * 3, [np.ones(3)] * 3)

        def close(self):
            pass

    return FakeEnv


def test_fixed_month_validation_feeds_the_battery_the_mid_step_observation():
    import harl.envs.sustaindc.fixed_month_validation as fixed_validation

    for env_args, sequential in (({'battery': {'observation_timing': 'current'}}, True), ({}, False)):
        log = []
        actors = [_RecordingActor(1), _RecordingActor(0), _RecordingActor(2)]
        with patch.object(fixed_validation, 'HARLSustainDCEnv', _fixed_month_env(log)), \
             patch.object(fixed_validation, 'month_steps', return_value=2):
            fixed_validation.validate_fixed_months(
                env_args=env_args, actors=actors, recurrent_n=1, rnn_hidden_size=1,
                seed=102, event=1, cases=(fixed_validation.ValidationCase('test', 0),))
        if sequential:
            assert log == [('begin', [1, 0]), ('step', [1, 0, 2])] * 2
            assert all(np.all(seen == 7.0) for seen in actors[2].seen)
        else:
            assert log == [('step', [1, 0, 2])] * 2
            assert all(np.all(seen == 0.0) for seen in actors[2].seen)
        assert all(np.all(seen == 0.0) for seen in actors[0].seen + actors[1].seen)


def test_choose_actions_calls_begin_step_before_the_battery_actor():
    import evaluate_matched as evaluator

    for mode, executed_battery in (('checkpoint', 2), ('checkpoint-battery-idle', 2)):
        calls = []
        actors = [_RecordingActor(1), _RecordingActor(0), _RecordingActor(1 if mode != 'checkpoint' else 2)]

        def begin(actions):
            assert len(actors[2].seen) == 0
            calls.append(list(actions))
            return np.full(40, 7.0, dtype=np.float32)

        proposed, executed = evaluator.choose_actions(
            mode, actors, [np.zeros(40, dtype=np.float32)] * 3,
            [np.zeros((1, 1, 4), dtype=np.float32) for _ in range(3)],
            [np.ones((1, 1), dtype=np.float32)] * 3, [np.ones(3)] * 3, begin_step=begin)
        assert calls == [[1, 0]]
        assert np.all(actors[2].seen[0] == 7.0) and np.all(actors[0].seen[0] == 0.0)
        assert executed[:2] == [1, 0] and executed[2] == executed_battery
        assert proposed[:2] == [1, 0]
    actors = [_RecordingActor(1), _RecordingActor(0), _RecordingActor(2)]
    evaluator.choose_actions(
        'checkpoint', actors, [np.zeros(40, dtype=np.float32)] * 3,
        [np.zeros((1, 1, 4), dtype=np.float32) for _ in range(3)],
        [np.ones((1, 1), dtype=np.float32)] * 3, [np.ones(3)] * 3)
    assert np.all(actors[2].seen[0] == 0.0)


def _matched_config():
    from tests.test_observation_layout import _ENV_CONFIG

    return {'env_args': dict(_ENV_CONFIG, partial_obs=True, nonoverlapping_shared_obs_space=True,
                             battery={'initial_soc': 0.5, 'observation_timing': 'current'}),
            'algo_args': {'model': {'recurrent_n': 1, 'hidden_sizes': [4]}}}


def test_run_episode_rule_modes_work_in_current_timing():
    import evaluate_matched as evaluator

    for mode in ('noop', 'cold-noop', 'noop-trailing-peak-battery'):
        rows, summary, _ = evaluator.run_episode(_matched_config(), None, mode, 0, 102, max_steps=3)
        assert len(rows) == 3 and summary['steps'] == 3


def test_run_episode_checkpoint_battery_sees_the_current_gross_load():
    import evaluate_matched as evaluator

    config = _matched_config()
    actors = [_RecordingActor(1), _RecordingActor(1), _RecordingActor(2)]
    rows, _, _ = evaluator.run_episode(config, actors, 'checkpoint', 0, 102, max_steps=4)
    cap_kw = config['env_args']['datacenter_capacity_mw'] * 1000.0
    assert len(actors[2].seen) == 4
    for seen, row in zip(actors[2].seen, rows):
        assert seen.shape == (1, 40)
        assert seen[0, GROSS] == np.float32(row['power/dc_total_power_kW'] / cap_kw)


if __name__ == '__main__':
    sys.exit(1 if run_module_tests(globals()) else 0)
