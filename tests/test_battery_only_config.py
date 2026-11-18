"""Battery-only training configuration.

Two configuration variants carry the battery-only study:

* ``harl/configs/envs_cfgs/sustaindc_battery_only.yaml`` is the shipped
  scenario with three changes and no others: the battery is the only learner
  (fixed base agents serve the workload and cooling), there is no hand-tuned
  discharge floor, and the reward is ``battery_tariff_delta``.
* ``harl/configs/algos_cfgs/happo_battery_only.yaml`` is the shipped HAPPO
  configuration with a pre-registered learning rate and a launch size whose
  arithmetic guarantees that monthly validation and weight saving run.

``test_environment_variant_is_the_reward_gate_scenario`` ties the training
environment to the scenario on which the reward reconciliation gate in
``test_battery_only_reward`` runs, so the gate certifies the exact reward path
that training uses.
"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests._harness import run_module_tests
from tests.test_battery_only_reward import _config
from harl.utils.configs_tools import get_defaults_yaml_args

_ALGO, _ENV = 'happo', 'sustaindc'
_ALGO_VARIANT, _ENV_VARIANT = 'happo_battery_only', 'sustaindc_battery_only'

# Training budget and configuration menu, protocol section 5.6.
_LEARNING_RATES = (0.0003, 0.0001)
_ENTROPY_COEFFICIENTS = (0.1, 0.01)
_MAX_UPDATES_PER_RUN = 120
_MAX_LAUNCHES = 6
_MAX_AGGREGATE_UPDATES = 600
_EARLY_KILL_UPDATE = 60          # the entropy kill rule is checked after 60 updates
_EPISODE_LENGTH = 1024           # launch arithmetic, protocol section 3
_VALIDATION_SEED = 102
_VALIDATION_GATES = (
    'require_zero_queue_overflow',
    'require_zero_overdue_steps',
    'require_zero_thermal_clamp_steps',
    'require_zero_capacity_exceeded_steps',
)

# The only leaves the HAPPO variant may change: the learning-rate menu entry
# and the launch size.
_ALGO_CHANGES = {
    'model.lr',
    'model.critic_lr',
    'train.n_rollout_threads',
    'train.num_env_steps',
    'train.eval_interval',
}

# Protocol sections 5.3 and 7: the actuator, reward and frozen scenario.
_PROTOCOL_PINS = {
    'agents': ['agent_bat'],
    'reward.mode': 'battery_tariff_delta',
    'reward.scale': 0.01,
    'reward.degradation_cost_weight': 1.0,
    'battery.discharge_mode': 'peak_limit',
    'battery.discharge_peak_floor_kw': None,
    'battery.peak_charge_headroom': True,
    'battery.peak_charge_headroom_basis': 'tariff',
    'battery.observation_timing': 'current',
    'battery.round_trip_efficiency': 0.9,
    'battery.degradation_cost_c_per_kwh': 5.78,
    'battery.initial_soc': 0.5,
    'datacenter_capacity_mw': 10,
    'location': 'montreal',
    'pricing.model': 'hydro_quebec',
    'pricing.config_file': 'data/Pricing/hydro_quebec_2026.yaml',
    'episode_start_policy': 'calendar_month',
    'dc_config_file': 'dc_config.json',
    'pv.enabled': True,
    'pv.capacity_fraction_of_datacenter': 0.05,
    'pv.surface_tilt_deg': 35.0,
    'pv.surface_azimuth_deg': 180.0,
}

_MISSING = object()


def _leaves(tree, path=''):
    """Every leaf of a nested mapping, keyed by its dotted path."""
    if isinstance(tree, dict) and tree:
        leaves = {}
        for key, value in tree.items():
            leaves.update(_leaves(value, f'{path}.{key}' if path else str(key)))
        return leaves
    return {path: tree}


def _changed(before, after):
    """Dotted paths whose values differ, or that exist on one side only."""
    old, new = _leaves(before), _leaves(after)
    return sorted(path for path in old.keys() | new.keys()
                  if old.get(path, _MISSING) != new.get(path, _MISSING))


def _shipped():
    return get_defaults_yaml_args(_ALGO, _ENV)


def _variants():
    return get_defaults_yaml_args(_ALGO_VARIANT, _ENV_VARIANT)


def test_environment_variant_is_the_reward_gate_scenario():
    _, env = _variants()
    gate = _config()
    assert env == gate, _changed(gate, env)


def test_environment_variant_changes_only_agents_floor_and_reward():
    _, shipped = _shipped()
    _, env = _variants()
    expected = copy.deepcopy(shipped)
    expected['agents'] = ['agent_bat']
    expected['battery']['discharge_peak_floor_kw'] = None
    expected['reward'] = {
        'mode': 'battery_tariff_delta',
        'scale': shipped['reward']['scale'],
        'degradation_cost_weight': shipped['reward']['degradation_cost_weight'],
    }
    assert env == expected, _changed(expected, env)


def test_environment_variant_pins_the_protocol_scenario():
    _, env = _variants()
    leaves = _leaves(env)
    wrong = {}
    for path, value in _PROTOCOL_PINS.items():
        actual = leaves.get(path, _MISSING)
        if type(actual) is not type(value) or actual != value:
            wrong[path] = actual
    assert not wrong, wrong


def test_happo_variant_changes_only_learning_rates_and_launch_size():
    shipped, _ = _shipped()
    algo, _ = _variants()
    changed = _changed(shipped, algo)
    assert set(changed) <= _ALGO_CHANGES, changed


def test_happo_variant_is_on_the_preregistered_menu():
    algo, _ = _variants()
    model = algo['model']
    assert type(model['lr']) is float and model['lr'] in _LEARNING_RATES, model['lr']
    assert type(model['critic_lr']) is float and model['critic_lr'] == model['lr'], model['critic_lr']
    entropy = algo['algo']['entropy_coef']
    assert type(entropy) is float and entropy in _ENTROPY_COEFFICIENTS, entropy


def test_happo_variant_launch_arithmetic_runs_validation_within_budget():
    algo, _ = _variants()
    train = algo['train']
    steps, length = train['num_env_steps'], train['episode_length']
    threads = train['n_rollout_threads']
    updates = steps // length // threads
    assert length == _EPISODE_LENGTH, length
    # Whole updates only, so the configured step count is exactly what runs.
    assert steps == updates * length * threads, (steps, updates, threads)
    assert 0 < updates <= _MAX_UPDATES_PER_RUN, updates
    # Validation and weight saving run only at multiples of eval_interval.
    assert updates % train['eval_interval'] == 0, (updates, train['eval_interval'])
    # Every pre-registered launch at this size fits the aggregate budget.
    assert _MAX_LAUNCHES * updates <= _MAX_AGGREGATE_UPDATES, updates
    # The entropy kill rule needs a logged entropy at update 60.
    assert _EARLY_KILL_UPDATE % train['log_interval'] == 0, train['log_interval']


def test_happo_variant_validates_on_fixed_months_and_stays_local():
    algo, _ = _variants()
    validation = algo['eval']['fixed_month_validation']
    assert validation['enabled'] is True, validation['enabled']
    assert validation['seed'] == _VALIDATION_SEED, validation['seed']
    assert all(validation[gate] is True for gate in _VALIDATION_GATES), validation
    wandb = algo['logger']['wandb']
    assert wandb['enabled'] is False and wandb['mode'] == 'disabled', wandb


if __name__ == '__main__':
    sys.exit(1 if run_module_tests(globals()) else 0)
