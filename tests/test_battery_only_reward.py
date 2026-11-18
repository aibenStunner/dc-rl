"""Battery-only reward mode ``battery_tariff_delta``.

With the workload and cooling actions held by fixed rules, battery actions
cannot change the facility load, so the bill the facility would pay with the
battery idle is computable exactly, step by step: meter the same load net of
PV (never below zero, as the real meter does) through a second, independent
pricing model. The battery's reward each step is the scaled bill saving
against that counterfactual (counterfactual minus actual charges), minus the
weighted, scaled degradation cost, minus the scaled cost of refilling the
battery to its starting charge, charged once at episode end.

Summed over an episode, the reward therefore reconciles exactly with the
refill-adjusted bill used to rank controllers.
``test_validation_months_reward_reconciles_with_refill_adjusted_bills`` is the
permanent reconciliation gate: the always-idle policy and the matched no-floor
causal rule, replayed through the training reward path on the four validation
months, must reconcile to within $1 per month-episode. The same replays must
reproduce, to the cent, the bills and battery wear that protocol section 4
reports for both policies, so the gate cannot pass on some other rule.
"""
import contextlib
import copy
import io
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests._harness import run_module_tests
from tests.test_base_agent_fallback import _SCENARIO
from harl.envs.sustaindc.fixed_month_validation import (
    DEFAULT_CASES,
    _refill_cost_c,
    month_steps,
    restoration_adjusted_bill_c,
)
from harl.envs.sustaindc.metrics import FacilityMetricAccumulator, build_trace_row
from sustaindc_env import EnvConfig, SustainDC
from utils import reward_creator

_SCALE = 0.01
_WEIGHT = 1.0
_REWARD = {
    'mode': 'battery_tariff_delta',
    'scale': _SCALE,
    'degradation_cost_weight': _WEIGHT,
}
_SEED = 102
_FEBRUARY_FIRST_DAY = 31        # zero-based day-of-year index of February 1
_GATE_TOLERANCE_C = 100.0       # protocol gate: $1 per month-episode
_IDENTITY_TOLERANCE_C = 0.01    # floating-point summation headroom only
_ANCHOR_TOLERANCE_C = 0.5       # section 4 reports cents
_RECHARGE_TARGET_SOC = 0.5      # the rule recharges toward the starting charge
_STEPS_PER_HOUR = 4             # 15-minute intervals

# Protocol section 4, per validation month: the idle battery's bill, and the
# matched no-floor rule's bill, cell-throughput degradation (cents) and number
# of discharge steps, as measured by the floor-calibration study's held-out
# run (seed 102). The rule ends every month above its starting charge, so no
# refill is due and its refill-adjusted bill is its bill.
_SECTION_4 = {
    'winter': {'idle_bill_c': 37148878.700761616, 'rule_bill_c': 37121022.054839335,
               'rule_degradation_c': 127723.1786890366, 'rule_discharge_steps': 57},
    'spring': {'idle_bill_c': 23438716.38793307, 'rule_bill_c': 23369160.799482636,
               'rule_degradation_c': 129410.13669911955, 'rule_discharge_steps': 50},
    'summer': {'idle_bill_c': 24260197.3582098, 'rule_bill_c': 24265621.349398863,
               'rule_degradation_c': 131886.99116926405, 'rule_discharge_steps': 53},
    'autumn': {'idle_bill_c': 24228416.70502253, 'rule_bill_c': 24147423.919921134,
               'rule_degradation_c': 130592.49951100507, 'rule_discharge_steps': 44},
}


def _config(**overrides):
    """The training scenario, battery-only, no discharge floor, this reward."""
    config = copy.deepcopy(_SCENARIO)
    config['agents'] = ['agent_bat']
    config['battery']['discharge_peak_floor_kw'] = None
    config['reward'] = dict(_REWARD)
    config.update(overrides)
    return config


def _expect(exc_type, fragment, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type as exc:
        assert fragment in str(exc), str(exc)
    else:
        raise AssertionError(f'expected {exc_type.__name__} containing {fragment!r}')


def _idle(env, step, peak, soc):
    return 2


def _scripted(env, step, peak, soc):
    # Discharge 4 h, charge 6 h, then idle: exercises both flow directions,
    # degradation and a terminal refill within one day.
    return 1 if step < 16 else (0 if step < 40 else 2)


def _no_floor_rule(env, step, peak, soc):
    """The matched no-floor causal rule of protocol section 4.

    Discharge when this interval's net import (facility load metered after
    workload and cooling act, minus this interval's PV) exceeds the billing
    running peak and charge remains; otherwise recharge while below the
    starting charge; otherwise idle.
    """
    load_kw = float(env.dc_info['dc_total_power_kW'])
    pv_kw = float(env.pv_m.get_current_ac_kwh()) * _STEPS_PER_HOUR
    armed = max(0.0, load_kw - pv_kw) > peak
    if armed and soc > 0:
        return 1
    if soc < _RECHARGE_TARGET_SOC:
        return 0
    return 2


def _run(config, init_day, policy, name):
    """Step one episode with stdout silenced; return rewards, bills and summary."""
    with contextlib.redirect_stdout(io.StringIO()):
        env = SustainDC(EnvConfig(config))
        env.reset(seed=_SEED, options={'init_day': init_day, 'init_hour': 0})
        accumulator = FacilityMetricAccumulator('episode')
        rewards, counterfactual, totals, actions = [], [], [], []
        truncated, step = {'__all__': False}, 0
        while not truncated['__all__']:
            assert step < 31 * 96, 'episode failed to truncate'
            # As in the evaluator: billing peak and charge are read before
            # begin_step, this interval's load and PV after it.
            peak = env.price_m.state.running_peak_kw
            soc = env.bat_env.get_battery_soc()
            env.begin_step({})
            action = policy(env, step, peak, soc)
            _, rew, term, truncated, info = env.step({'agent_bat': action})
            assert set(rew) == {'agent_bat'}, rew
            assert not any(term.values()), term
            common = info['__common__']
            assert common['agent_bat_reward'] == rew['agent_bat']
            rewards.append(rew['agent_bat'])
            counterfactual.append(common['counterfactual_cost_this_step_c'])
            totals.append(common['total_price_cost_this_step_c'])
            actions.append(action)
            accumulator.add(common)
            step += 1
    final_row = build_trace_row(info=common, actions=[1, 1, action], run_metadata={},
                                terminal=False, truncated=True)
    restore = float(final_row['battery/bat_terminal_soc_restore_grid_kwh'])
    summary = accumulator.snapshot()
    summary['episode/terminal/soc_restore_cost_c'] = _refill_cost_c(final_row, restore, name)
    return {
        'steps': step,
        'rewards': rewards,
        'counterfactual': counterfactual,
        'totals': totals,
        'actions': actions,
        'restore_kwh': restore,
        'summary': summary,
    }


def _residual(run, idle):
    """|reward-side minus bill-side| difference between two policies, in cents."""
    reward_side = -(math.fsum(run['rewards']) - math.fsum(idle['rewards'])) / _SCALE
    bill_side = (
        restoration_adjusted_bill_c(run['summary'])
        - restoration_adjusted_bill_c(idle['summary'])
        + _WEIGHT * (run['summary']['episode/battery/degradation_cost_c']
                     - idle['summary']['episode/battery/degradation_cost_c'])
    )
    return abs(reward_side - bill_side)


def test_reward_defaults_merge_and_reject_unknown_keys():
    config = EnvConfig(_config(reward={'mode': 'battery_tariff_delta'}))
    assert config['reward'] == {
        'mode': 'battery_tariff_delta', 'scale': 0.01, 'degradation_cost_weight': 1.0,
    }, config['reward']
    partial = EnvConfig(_config(reward={'mode': 'battery_tariff_delta', 'scale': 0.02}))
    assert partial['reward'] == {
        'mode': 'battery_tariff_delta', 'scale': 0.02, 'degradation_cost_weight': 1.0,
    }, partial['reward']
    _expect(ValueError, 'reward.ls_overdue_penalty_weight is unknown', EnvConfig,
            _config(reward={'mode': 'battery_tariff_delta',
                            'ls_overdue_penalty_weight': 0.5}))


def test_validator_requires_battery_only_agents_and_fixed_starts():
    _expect(ValueError, "battery_tariff_delta requires agents == ['agent_bat']", EnvConfig,
            _config(agents=['agent_ls', 'agent_dc', 'agent_bat']))
    _expect(ValueError, 'battery_tariff_delta requires valid_midnight or calendar_month',
            EnvConfig, _config(episode_start_policy='legacy_random'))
    _expect(ValueError, 'reward.scale must be positive', EnvConfig,
            _config(reward=dict(_REWARD, scale=0.0)))
    for key in ('scale', 'degradation_cost_weight'):
        for value in (-1.0, math.nan, True):
            _expect(ValueError, f'reward.{key} must be finite and non-negative', EnvConfig,
                    _config(reward=dict(_REWARD, **{key: value})))
    EnvConfig(_config())


def test_reward_function_is_bill_saving_minus_degradation_and_refill():
    params = {
        'total_price_cost_this_step_c': 100.0,
        'counterfactual_cost_this_step_c': 150.0,
        'bat_degradation_cost_c': 10.0,
        'bat_terminal_soc_penalty_component': 0.5,
    }
    reward_config = {'mode': 'battery_tariff_delta', 'scale': 0.01,
                     'degradation_cost_weight': 1.0}
    rewards, components = reward_creator.battery_tariff_delta(params, reward_config)
    # 0.01 * (150 - 100) - 1.0 * 0.01 * 10 - 0.5
    assert math.isclose(rewards['agent_bat'], -0.1, rel_tol=0.0, abs_tol=1e-12), rewards
    assert rewards['agent_ls'] == 0.0 and rewards['agent_dc'] == 0.0, rewards
    assert components['bat_bill_delta_c'] == -50.0, components
    assert math.isclose(components['bat_tariff_delta_reward_component'], 0.5,
                        rel_tol=0.0, abs_tol=1e-12), components
    assert math.isclose(components['bat_degradation_penalty_component'], 0.1,
                        rel_tol=0.0, abs_tol=1e-12), components
    assert components['bat_terminal_soc_penalty_component'] == 0.5, components
    for key, value in (('counterfactual_cost_this_step_c', -1.0),
                       ('total_price_cost_this_step_c', math.nan)):
        _expect(ValueError, f'{key} must be finite and non-negative',
                reward_creator.battery_tariff_delta, dict(params, **{key: value}),
                reward_config)
    _expect(ValueError, 'reward.scale must be positive', reward_creator.battery_tariff_delta,
            params, dict(reward_config, scale=0.0))


def test_idle_battery_reward_is_exactly_zero_every_step():
    config = _config(episode_start_policy='valid_midnight', days_per_episode=1)
    idle = _run(config, _FEBRUARY_FIRST_DAY, _idle, 'idle-day')
    assert idle['steps'] == 96, idle['steps']
    assert all(reward == 0.0 for reward in idle['rewards']), idle['rewards'][:5]
    # Idle metering is the counterfactual metering, bit for bit.
    assert idle['counterfactual'] == idle['totals']
    assert idle['restore_kwh'] == 0.0, idle['restore_kwh']


def test_scripted_day_reward_reconciles_with_bills():
    config = _config(episode_start_policy='valid_midnight', days_per_episode=1)
    scripted = _run(config, _FEBRUARY_FIRST_DAY, _scripted, 'scripted-day')
    idle = _run(config, _FEBRUARY_FIRST_DAY, _idle, 'idle-day')
    summary = scripted['summary']
    assert summary['episode/battery/charge_bus_kwh'] > 0.0, summary
    assert summary['episode/battery/discharge_bus_kwh'] > 0.0, summary
    assert summary['episode/battery/degradation_cost_c'] > 0.0, summary
    assert scripted['restore_kwh'] > 0.0, scripted['restore_kwh']
    assert (summary['episode/tariff/cost/total_c']
            != idle['summary']['episode/tariff/cost/total_c'])
    # Battery actions cannot move the counterfactual.
    assert scripted['counterfactual'] == idle['counterfactual']
    residual = _residual(scripted, idle)
    assert residual <= _IDENTITY_TOLERANCE_C, residual


def test_section_4_anchors_sum_to_protocol_totals():
    def total(key):
        return math.fsum(month[key] for month in _SECTION_4.values()) / 100

    assert set(_SECTION_4) == {case.name for case in DEFAULT_CASES}, _SECTION_4
    assert math.isclose(total('idle_bill_c'), 1_090_762.09, rel_tol=0.0, abs_tol=0.005)
    assert math.isclose(total('rule_bill_c'), 1_089_032.28, rel_tol=0.0, abs_tol=0.005)
    assert math.isclose(total('rule_degradation_c'), 5_196.13, rel_tol=0.0, abs_tol=0.005)


def test_validation_months_reward_reconciles_with_refill_adjusted_bills():
    """Protocol gate: idle vs the no-floor rule, four validation months."""
    assert _config()['battery']['initial_soc'] == _RECHARGE_TARGET_SOC
    residuals, anchor_errors = {}, {}
    for case in DEFAULT_CASES:
        anchor = _SECTION_4[case.name]
        idle = _run(_config(), case.init_day, _idle, f'{case.name}-idle')
        rule = _run(_config(), case.init_day, _no_floor_rule, f'{case.name}-rule')
        expected_steps = month_steps(case.init_day)
        assert idle['steps'] == expected_steps, (case.name, idle['steps'])
        assert rule['steps'] == expected_steps, (case.name, rule['steps'])
        assert all(reward == 0.0 for reward in idle['rewards']), case.name
        assert idle['counterfactual'] == idle['totals'], case.name
        assert rule['counterfactual'] == idle['counterfactual'], case.name
        idle_bill = idle['summary']['episode/tariff/cost/total_c']
        rule_bill = rule['summary']['episode/tariff/cost/total_c']
        assert math.isclose(math.fsum(idle['counterfactual']), idle_bill,
                            rel_tol=0.0, abs_tol=_IDENTITY_TOLERANCE_C), case.name
        # Non-vacuity: the rule must actually dispatch both ways and change the bill.
        assert rule['actions'].count(0) > 0, case.name
        assert rule['summary']['episode/battery/charge_bus_kwh'] > 0.0, case.name
        assert rule['summary']['episode/battery/discharge_bus_kwh'] > 0.0, case.name
        assert rule_bill != idle_bill, case.name
        # Section 4 anchors: this is the protocol's rule, measured as there.
        rule_discharges = rule['actions'].count(1)
        assert rule_discharges == anchor['rule_discharge_steps'], (case.name, rule_discharges)
        assert rule['restore_kwh'] == 0.0, (case.name, rule['restore_kwh'])
        errors = {
            'idle_bill_c': restoration_adjusted_bill_c(idle['summary']) - anchor['idle_bill_c'],
            'rule_bill_c': restoration_adjusted_bill_c(rule['summary']) - anchor['rule_bill_c'],
            'rule_degradation_c': (rule['summary']['episode/battery/degradation_cost_c']
                                   - anchor['rule_degradation_c']),
        }
        for key, error in errors.items():
            assert abs(error) <= _ANCHOR_TOLERANCE_C, (case.name, key, error)
        anchor_errors[case.name] = max(abs(error) for error in errors.values())
        residuals[case.name] = _residual(rule, idle)
    print('largest section 4 anchor error (cents) per month:',
          {name: f'{value:.3e}' for name, value in anchor_errors.items()})
    print('reconciliation residual (cents) per month:',
          {name: f'{value:.3e}' for name, value in residuals.items()})
    for name, residual in residuals.items():
        assert residual <= _GATE_TOLERANCE_C, (name, residual)
        assert residual <= _IDENTITY_TOLERANCE_C, (name, residual)


if __name__ == '__main__':
    sys.exit(1 if run_module_tests(globals()) else 0)
