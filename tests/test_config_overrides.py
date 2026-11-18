"""Command-line ``--key value`` overrides reach exactly one setting or fail.

``update_args`` used to overwrite every leaf whose name matched, anywhere in
the configuration. ``--seed 1`` therefore also moved the fixed-month
validation seed off 102, and a typo was ignored without a word. An override
now names one setting: a bare name when it is unique, otherwise a dotted path
whose trailing components identify the setting. Anything else is an error,
and a failed override leaves the configuration untouched.
"""
import copy
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from tests._harness import run_module_tests
from harl.utils.configs_tools import get_defaults_yaml_args, update_args


def _shipped():
    return get_defaults_yaml_args('happo', 'sustaindc')


def _raises(overrides, algo_args, env_args):
    """Return the ValueError message ``update_args`` raises; fail if it does not raise."""
    try:
        update_args(overrides, algo_args, env_args)
    except ValueError as error:
        return str(error)
    raise AssertionError(f'{overrides} was accepted')


def test_a_bare_name_found_in_two_places_is_rejected_and_changes_nothing():
    algo_args, env_args = _shipped()
    before = copy.deepcopy((algo_args, env_args))
    message = _raises({'seed': 7}, algo_args, env_args)
    assert 'seed.seed' in message and 'eval.fixed_month_validation.seed' in message, message
    assert (algo_args, env_args) == before


def test_the_training_seed_path_leaves_the_validation_seed_alone():
    algo_args, env_args = _shipped()
    update_args({'seed.seed': 7}, algo_args, env_args)
    assert algo_args['seed']['seed'] == 7
    assert algo_args['eval']['fixed_month_validation']['seed'] == 102


def test_a_trailing_path_sets_the_validation_seed_alone():
    algo_args, env_args = _shipped()
    update_args({'fixed_month_validation.seed': 5}, algo_args, env_args)
    assert algo_args['eval']['fixed_month_validation']['seed'] == 5
    assert algo_args['seed']['seed'] == 1


def test_unique_bare_names_still_reach_their_setting():
    algo_args, env_args = _shipped()
    update_args({'lr': 1e-4, 'demand_floor_kw': 5000.0, 'num_env_steps': 64}, algo_args, env_args)
    assert algo_args['model']['lr'] == 1e-4
    assert env_args['pricing']['options']['demand_floor_kw'] == 5000.0
    assert algo_args['train']['num_env_steps'] == 64


def test_an_ambiguous_name_shared_by_algo_and_env_is_rejected():
    algo_args, env_args = _shipped()
    message = _raises({'mode': 'online'}, algo_args, env_args)
    assert 'reward.mode' in message and 'logger.wandb.mode' in message, message


def test_an_unknown_name_is_rejected_and_one_bad_key_blocks_the_rest():
    algo_args, env_args = _shipped()
    before = copy.deepcopy((algo_args, env_args))
    message = _raises({'lr': 1e-4, 'demand_flor_kw': 5000.0}, algo_args, env_args)
    assert 'demand_flor_kw' in message, message
    assert (algo_args, env_args) == before


def test_a_path_that_names_a_section_is_rejected():
    algo_args, env_args = _shipped()
    message = _raises({'fixed_month_validation': 1}, algo_args, env_args)
    assert 'fixed_month_validation' in message, message


if __name__ == '__main__':
    sys.exit(1 if run_module_tests(globals()) else 0)
