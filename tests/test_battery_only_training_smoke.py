"""A battery-only launch trains end to end through ``train_sustaindc.main()``.

The driver launches the way the cluster does: a launch file built from the
battery-only variants (``happo_battery_only.yaml`` and
``sustaindc_battery_only.yaml``) goes to ``train_sustaindc.main()`` through
``--load_config``, with an experiment name and the menu-only overrides
(``--lr``, ``--critic_lr``, ``--entropy_coef``). Only the launch is shrunk, to
two updates of two subprocess environments (the vector environment a cluster
launch uses) and one fixed-month validation event after the second update.
Validation runs its unchanged code path on the first case (January) only, to
keep the test short.

Training runs in a fresh interpreter, so the runner's global seeding, torch
thread count and warning filters, and its worker processes, stay out of the
shared test process.
"""
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import torch

from tests._harness import run_module_tests
from tests.test_battery_only_config import _ALGO_VARIANT, _ENV_VARIANT
from harl.envs.sustaindc.fixed_month_validation import DEFAULT_CASES, month_steps, validate_fixed_months
from harl.utils.configs_tools import get_defaults_yaml_args, update_args

# Two updates of 2 threads x 16 steps; validation after the second update.
_TINY_LAUNCH = {
    'n_rollout_threads': 2,
    'episode_length': 16,
    'num_env_steps': 64,
    'eval_interval': 2,
    'log_interval': 1,
}
_EXP_NAME = 'battery-only-happo-s1'
# The overrides a launch may pass, at values that differ from the YAML defaults
# (lr and critic_lr 3e-4, entropy_coef 0.1), so a landed override is visible.
_OVERRIDES = {'lr': 0.0001, 'critic_lr': 0.0001, 'entropy_coef': 0.01}
_VALIDATION_CASE = DEFAULT_CASES[0]
_DRIVER = 'import sys; from tests.test_battery_only_training_smoke import _train; _train(sys.argv[1])'
_TIMEOUT_SECONDS = 600
_STDERR_TAIL_LINES = 60


def _leaves(node, key, path=()):
    """Return ``{path: value}`` for every non-dict entry named ``key`` under ``node``.

    ``update_args`` accepts a bare name only when exactly one entry carries it.
    """
    found = {}
    for name, value in node.items():
        if type(value) is dict:
            found.update(_leaves(value, key, path + (name,)))
        elif name == key:
            found[path + (name,)] = value
    return found


def _config_leaves(config, key):
    """``_leaves`` over both configuration mappings of a launch, each path led by its mapping's name."""
    return {(section, *path): value
            for section in ('algo_args', 'env_args')
            for path, value in _leaves(config[section], key).items()}


def _write_launch_file(path):
    """Write the launch file a cluster run loads: the battery-only variants with one training seed."""
    algo_args, env_args = get_defaults_yaml_args(_ALGO_VARIANT, _ENV_VARIANT)
    algo_args['train'].update(_TINY_LAUNCH)
    algo_args['seed']['seed'] = 1
    algo_args['logger']['log_dir'] = str(path.parent / 'results')
    launch = {'main_args': {'algo': 'happo', 'env': 'sustaindc'}, 'algo_args': algo_args, 'env_args': env_args}
    path.write_text(json.dumps(launch), encoding='utf-8')


def _train(root):
    """Launch through ``train_sustaindc.main()`` into ``root``; runs only in the driver interpreter."""
    import train_sustaindc
    from harl.runners import on_policy_base_runner
    from harl.utils.tracking import TensorboardWandbWriter

    root = Path(root)
    _write_launch_file(root / 'launch.json')
    argv = ['train_sustaindc.py', '--load_config', str(root / 'launch.json'), '--exp_name', _EXP_NAME]
    for key, value in _OVERRIDES.items():
        argv += [f'--{key}', str(value)]

    tags = []
    real_add_scalar = TensorboardWandbWriter.add_scalar
    real_run_and_close = train_sustaindc.run_and_close

    def record_tag(self, tag, scalar_value, global_step=None):
        tags.append(tag)
        return real_add_scalar(self, tag, scalar_value, global_step)

    def record_then_run(runner):
        torch.save(runner.actor[0].actor.state_dict(), root / 'initial_actor_agent0.pt')
        (root / 'run.json').write_text(json.dumps({'run_dir': str(runner.run_dir), 'save_dir': str(runner.save_dir)}))
        real_run_and_close(runner)

    def first_case_only(**kwargs):
        return validate_fixed_months(**dict(kwargs, cases=(_VALIDATION_CASE,)))

    with patch.object(sys, 'argv', argv), \
         patch.object(TensorboardWandbWriter, 'add_scalar', record_tag), \
         patch.object(train_sustaindc, 'run_and_close', record_then_run), \
         patch.object(on_policy_base_runner, 'validate_fixed_months', first_case_only):
        train_sustaindc.main()
    (root / 'tags.json').write_text(json.dumps(sorted(set(tags))))


def _check_run(root):
    """Assert what a completed battery-only launch leaves in ``root``."""
    run = json.loads((root / 'run.json').read_text())
    save_dir = Path(run['save_dir'])
    candidates = save_dir / 'candidates'
    steps = month_steps(_VALIDATION_CASE.init_day)

    validation = json.loads((candidates / 'event-1' / 'validation.json').read_text())
    assert validation['event'] == 1, validation['event']
    assert [case['name'] for case in validation['cases']] == [_VALIDATION_CASE.name], validation['cases']
    assert validation['cases'][0]['steps'] == steps, validation['cases'][0]['steps']
    # The battery cannot move the service, thermal or cooling-capacity gates, and
    # do-nothing base agents hold workload and cooling, so an ineligible event is a bug.
    assert validation['eligible'] is True, validation['aggregate']

    trace = candidates / 'traces' / f'event-1-{_VALIDATION_CASE.name}.csv'
    with trace.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert len(rows) == steps, len(rows)
    # One action column, the battery's: no column for an agent that is not in the run.
    action_columns = sorted(name for name in reader.fieldnames if name.startswith('action/'))
    assert action_columns == ['action/bat'], action_columns
    battery_actions = {float(row['action/bat']) for row in rows}
    assert battery_actions <= {0.0, 1.0, 2.0}, battery_actions

    last = json.loads((save_dir / 'last' / 'last.json').read_text())
    assert last['validation_event'] == 1, last
    initial = torch.load(root / 'initial_actor_agent0.pt', map_location='cpu', weights_only=True)
    trained = torch.load(save_dir / 'last' / 'actor_agent0.pt', map_location='cpu', weights_only=True)
    assert initial.keys() == trained.keys(), (sorted(initial), sorted(trained))
    assert any(not torch.equal(initial[key], trained[key]) for key in initial), 'training left the actor unchanged'

    # The eligible event is exported with exactly one actor.
    exported = sorted(path.name for path in save_dir.glob('*.pt'))
    assert exported == ['actor_agent0.pt', 'critic_agent.pt', 'value_normalizer.pt'], exported
    config = json.loads((Path(run['run_dir']) / 'config.json').read_text())
    assert config['env_args']['agents'] == ['agent_bat'], config['env_args']['agents']

    # Each override reached exactly one saved setting; both seeds are still the protocol's.
    for key, value in _OVERRIDES.items():
        assert list(_config_leaves(config, key).values()) == [value], (key, _config_leaves(config, key))
    assert _config_leaves(config, 'seed') == {
        ('algo_args', 'seed', 'seed'): 1,
        ('algo_args', 'eval', 'fixed_month_validation', 'seed'): 102,
    }, _config_leaves(config, 'seed')

    tags = json.loads((root / 'tags.json').read_text())
    # Per-update series carry the battery's label, never ls or dc.
    actor_series = sorted({tag.split('/')[3] for tag in tags if tag.startswith('train/update/actor/')})
    assert actor_series == ['bat'], actor_series
    # The pre-registered early-kill rule reads the battery policy's entropy.
    assert 'train/agent0/dist_entropy' in tags, tags
    assert 'train/update/actor/bat/dist_entropy' in tags, tags


def test_happo_trains_validates_and_exports_with_the_battery_as_the_only_agent():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        root = Path(directory)
        completed = subprocess.run(
            [sys.executable, '-c', _DRIVER, str(root)],
            cwd=_REPO_ROOT,
            env=dict(os.environ, WANDB_MODE='disabled'),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            encoding='utf-8',
            errors='replace',
            timeout=_TIMEOUT_SECONDS,
        )
        tail = '\n'.join(completed.stderr.splitlines()[-_STDERR_TAIL_LINES:])
        assert completed.returncode == 0, f'driver exited with {completed.returncode}; stderr tail:\n{tail}'
        _check_run(root)


def test_menu_overrides_reach_one_setting_each_and_the_shared_names_are_refused():
    algo_args, env_args = get_defaults_yaml_args(_ALGO_VARIANT, _ENV_VARIANT)
    config = {'algo_args': algo_args, 'env_args': env_args}
    keys = ('lr', 'critic_lr', 'entropy_coef', 'seed', 'enabled', 'mode')
    counts = {key: len(_config_leaves(config, key)) for key in keys}
    assert counts == {'lr': 1, 'critic_lr': 1, 'entropy_coef': 1, 'seed': 2, 'enabled': 3, 'mode': 3}, counts
    # A bare --seed would also reach the validation seed every comparison holds at 102, so it is refused.
    for key in ('seed', 'enabled', 'mode'):
        try:
            update_args({key: 7}, algo_args, env_args)
        except ValueError:
            continue
        raise AssertionError(f'--{key} was accepted')
    update_args({'seed.seed': 7}, algo_args, env_args)
    assert algo_args['seed']['seed'] == 7
    assert algo_args['eval']['fixed_month_validation']['seed'] == 102


if __name__ == '__main__':
    sys.exit(1 if run_module_tests(globals()) else 0)
