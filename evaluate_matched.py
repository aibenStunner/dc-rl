"""Matched fixed-window checkpoint evaluation; never writes into a training run.

Defaults to the final window set (March, June, September, December), which lies
outside fixed-month checkpoint selection and discharge-floor calibration.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import tempfile

import numpy as np
import torch

from harl.algorithms.actors import ALGO_REGISTRY
from harl.algorithms.critics.v_critic import VCritic
from harl.envs.sustaindc.harlsustaindc_env import HARLSustainDCEnv
from harl.envs.sustaindc.metrics import FacilityMetricAccumulator, build_trace_row, write_trace_csv
from utils.managers.pricing.models.hydro_quebec import _MONTH_START_DAYS

AGENTS = ('agent_ls', 'agent_dc', 'agent_bat')
WINDOW_SETS = {
    # The fixed-month checkpoint-selection months (fixed_month_validation.DEFAULT_CASES).
    'validation': {'winter': 0, 'spring': 90, 'summer': 181, 'autumn': 273},
    # Final evaluation months: outside checkpoint selection and floor calibration.
    'final': {'march': 59, 'june': 151, 'september': 243, 'december': 334},
}
DEFAULT_WINDOW_SET = 'final'
WINDOW_SET_NOTES = {
    'validation': (
        'Validation months (January, April, July, October) are the fixed-month '
        'checkpoint-selection set and were used to check the 10 MW discharge floor, '
        'so results on them are not held-out evidence.'
    ),
    'final': (
        'Final months (March, June, September, December) are outside the fixed-month '
        'checkpoint-selection set and were not used to choose or check the 10 MW '
        'discharge floor. Training is not restricted to other months, so they are held '
        'out from selection and calibration, not from training.'
    ),
}
# Saved analysis scripts read WINDOWS as the validation months.
WINDOWS = WINDOW_SETS['validation']


def month_steps(start_day):
    index = _MONTH_START_DAYS.index(start_day)
    return ((*_MONTH_START_DAYS[1:], 365)[index] - start_day) * 96
MODES = (
    'checkpoint', 'noop', 'cold-noop', 'checkpoint-battery-idle',
    'noop-trailing-peak-battery', 'trailing-peak-workload-battery',
)
ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique_requested(values, available, option_name):
    """Return explicit selections once, or the complete ordered manifest."""
    selected = list(available) if values is None else list(values)
    duplicates = sorted({value for value in selected if selected.count(value) > 1})
    if duplicates:
        raise ValueError(f"duplicate {option_name} value: {duplicates[0]}")
    return selected


def select_windows(names, window_set):
    """Return requested windows of one window set, in request order, with start days."""
    available = WINDOW_SETS[window_set]
    selected = unique_requested(names, available, '--window')
    outside = [name for name in selected if name not in available]
    if outside:
        raise ValueError(
            f"--window {outside[0]} is not in window set {window_set!r} "
            f"(choose from {', '.join(available)})"
        )
    return {name: available[name] for name in selected}


def paired_difference(candidate, baseline):
    """Difference all comparable episode and physical endpoint metrics."""
    physical = {
        'grid_peak_kw', 'facility_peak_kw', 'initial_soc', 'final_soc',
        'terminal_soc_start', 'terminal_soc_end',
        'terminal_soc_restore_grid_kwh',
        'terminal_soc_reference_price_c_per_kwh',
    }
    return {
        key: value - baseline[key]
        for key, value in candidate.items()
        if (key.startswith('episode/') or key in physical)
        and isinstance(value, (int, float))
        and isinstance(baseline.get(key), (int, float))
        and math.isfinite(float(value))
        and math.isfinite(float(baseline[key]))
    }


def build_paired_comparisons(summaries, windows, modes):
    """Compare evaluated controllers against each available fixed baseline."""
    pairs = []
    for window in windows:
        group = {s['mode']: s for s in summaries if s['window'] == window}
        for baseline in ('noop', 'cold-noop'):
            if baseline not in group:
                continue
            for mode in modes:
                if mode == baseline or mode not in group:
                    continue
                if baseline == 'cold-noop' and mode == 'noop':
                    continue
                pairs.append({
                    'window': window,
                    'comparison': mode + ' minus ' + baseline,
                    'differences': paired_difference(group[mode], group[baseline]),
                })
    return pairs


def reconcile_summary_against_trace(rows, summary):
    """Verify stored episode totals and peaks against the emitted trace."""
    if not rows:
        raise ValueError('cannot reconcile an empty trace')
    sums = {
        'episode/tariff/cost/total_c': ('tariff/total_price_cost_this_step_c', sum),
        'episode/tariff/cost/energy_c': ('tariff/energy_cost_this_step_c', sum),
        'episode/tariff/cost/demand_increment_c': ('tariff/demand_charge_increment_c', sum),
        'episode/tariff/cost/additional_c': ('tariff/additional_charge_increment_c', sum),
        'episode/energy/grid_import_kwh': ('battery/bat_grid_import_kwh', sum),
        'grid_peak_kw': ('battery/bat_grid_import_kwh', lambda values: max(values) * 4),
        'facility_peak_kw': ('power/dc_total_power_kW', max),
    }
    for summary_key, (trace_key, reducer) in sums.items():
        values = [float(row[trace_key]) for row in rows]
        actual = reducer(values)
        expected = summary.get(summary_key)
        if expected is None or not math.isclose(float(expected), actual, rel_tol=1e-7, abs_tol=1e-5):
            raise ValueError(
                f'summary/trace mismatch for {summary_key}: summary={expected}, trace={actual}'
            )


def load_checkpoint(run, env):
    config_path = run / 'config.json'
    config = json.loads(config_path.read_text())
    a = config['algo_args']
    if config['main_args']['algo'] != 'happo' or a['algo']['share_param']:
        raise ValueError('Only independent HAPPO actors are supported')
    files = [run / 'models' / f'actor_agent{i}.pt' for i in range(3)]
    files += [run / 'models' / 'critic_agent.pt']
    if a['train']['use_valuenorm']:
        files += [run / 'models' / 'value_normalizer.pt']
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    args = {**a['model'], **a['algo']}
    actors = []
    for i in range(3):
        actor = ALGO_REGISTRY['happo'](args, env.observation_space[i], env.action_space[i], device=torch.device('cpu'))
        actor.actor.load_state_dict(torch.load(files[i], map_location='cpu', weights_only=True), strict=True)
        actor.actor.eval()
        actors.append(actor)
    critic = VCritic(args, env.share_observation_space[0], device=torch.device('cpu'))
    critic.critic.load_state_dict(torch.load(files[3], map_location='cpu', weights_only=True), strict=True)
    # Value normalization is irrelevant to actor inference. Historical checkpoints
    # may contain an empty normalizer; retain and hash it without restoring it.
    return actors, {str(p.resolve()): digest(p) for p in [config_path, *files]}


def choose_actions(mode, actors, observations, states, masks, avail,
                   dc_setpoint=None, dc_min_setpoint=None,
                   previous_facility_load_kw=None, running_peak_kw=None,
                   battery_soc=None, queue_depth=None, begin_step=None):
    """Return proposed and executed actions for one interval.

    With ``begin_step`` (battery.observation_timing 'current'), the battery
    actor sees the observation it returns after the earlier actors choose.
    """
    proposed = []
    if mode == 'noop':
        return [1, 1, 2], [1, 1, 2]
    if mode == 'cold-noop':
        if dc_setpoint is None or dc_min_setpoint is None:
            raise ValueError('cold-noop requires current and minimum DC setpoints')
        actions = [1, 0 if dc_setpoint > dc_min_setpoint else 1, 2]
        return actions, actions.copy()
    if mode == 'noop-trailing-peak-battery':
        if (
            running_peak_kw is not None and running_peak_kw > 0
            and battery_soc is not None and battery_soc > 0
            and previous_facility_load_kw is not None
            and previous_facility_load_kw >= running_peak_kw
        ):
            return [1, 1, 1], [1, 1, 1]
        return [1, 1, 2], [1, 1, 2]
    if mode == 'trailing-peak-workload-battery':
        has_prior_peak = running_peak_kw is not None and running_peak_kw > 0
        previous_at_peak = (
            previous_facility_load_kw is not None
            and has_prior_peak
            and previous_facility_load_kw >= running_peak_kw
        )
        if previous_at_peak:
            battery_action = 1 if battery_soc is not None and battery_soc > 0 else 2
            return [0, 1, battery_action], [0, 1, battery_action]
        if has_prior_peak and queue_depth is not None and queue_depth > 0:
            return [2, 1, 2], [2, 1, 2]
        return [1, 1, 2], [1, 1, 2]
    with torch.no_grad():
        for i, actor in enumerate(actors):
            if begin_step is not None and i == AGENTS.index('agent_bat'):
                obs_i = begin_step(list(proposed))
            else:
                obs_i = observations[i]
            action, state = actor.act(np.asarray(obs_i)[None], states[i], masks[i], np.asarray(avail[i])[None], deterministic=True)
            states[i] = state.cpu().numpy()
            proposed.append(int(action.cpu().numpy().reshape(-1)[0]))
    executed = proposed.copy()
    if mode == 'checkpoint-battery-idle':
        executed[2] = 2
    return proposed, executed


def run_episode(config, actors, mode, day, seed, max_steps=None):
    expected_steps = month_steps(day)
    if max_steps is None:
        max_steps = expected_steps
    env_args = dict(config['env_args'], episode_start_policy='calendar_month')
    env = HARLSustainDCEnv(env_args)
    try:
        obs, _, avail = env.reset(seed=seed, options={'init_day': day, 'init_hour': 0})
        raw = env.env.unwrapped.env
        initial_soc = float(raw.bat_env.get_battery_soc())
        model = config['algo_args']['model']
        states = [np.zeros((1, model['recurrent_n'], model['hidden_sizes'][-1]), dtype=np.float32) for _ in AGENTS]
        masks = [np.ones((1, 1), dtype=np.float32) for _ in AGENTS]
        rows, exogenous = [], []
        metrics = FacilityMetricAccumulator('episode')
        sequential = raw.battery_config['observation_timing'] == 'current'
        for step in range(max_steps):
            common = raw.infos['__common__']
            exo = (float(common['workload']), float(common['weather']), float(common['ci']), float(raw.pv_m.get_current_ac_kwh()))
            begun = []

            def begin_step(ls_dc_actions):
                begun.append(True)
                return env.begin_step(np.asarray(ls_dc_actions))

            proposed, executed = choose_actions(
                mode, actors, obs, states, masks, avail,
                dc_setpoint=raw.dc_env.raw_curr_stpt,
                dc_min_setpoint=raw.dc_env.min_temp,
                previous_facility_load_kw=raw._previous_facility_load_kw,
                running_peak_kw=raw.price_m.state.running_peak_kw,
                battery_soc=raw.bat_env.get_battery_soc(),
                queue_depth=len(raw.ls_env.tasks_queue),
                begin_step=begin_step if sequential else None,
            )
            if sequential and not begun:
                # Rule modes choose without the mid-step observation; the
                # environment still meters workload/cooling before dispatch.
                env.begin_step(np.asarray(executed[:2]))
            padded_obs, _, term, trunc, info = env.env.step(dict(zip(AGENTS, executed)))
            obs = [padded_obs[a] for a in AGENTS]
            values = info[AGENTS[0]]
            terminated = any(term[a] for a in AGENTS)
            truncated = any(trunc[a] for a in AGENTS)
            metrics.add(values)
            row = build_trace_row(info=values, actions=executed, run_metadata={'step': step, 'mode': mode, 'start_day': day, 'seed': seed}, terminal=terminated, truncated=truncated)
            for i, name in enumerate(('ls', 'dc', 'bat')):
                row['proposed_action/' + name] = proposed[i]
            for key, value in zip(('workload', 'weather', 'ci', 'pv_ac_kwh'), exo):
                row['exogenous/' + key] = value
            rows.append(row)
            exogenous.append(exo)
            if terminated or truncated:
                if step != expected_steps - 1 or terminated or not truncated:
                    raise RuntimeError(f'unexpected episode end at step {step + 1}: terminated={term}, truncated={trunc}')
                break
        if max_steps == expected_steps and (len(rows) != expected_steps or not rows[-1]['truncated']):
            raise RuntimeError(f'episode did not truncate exactly at {expected_steps} steps')
        summary = metrics.snapshot()
        for row in rows:
            def number(key):
                value = row.get(key)
                if value is None or not math.isfinite(float(value)):
                    raise ValueError(f'missing or nonfinite trace field {key}')
                return float(value)
            grid = number('battery/bat_grid_import_kwh')
            load = number('battery/bat_dc_load_kwh')
            pv = number('pv/pv_self_consumed_kwh')
            charge = number('battery/bat_charge_bus_KWh')
            discharge = number('battery/bat_discharge_bus_KWh')
            if not math.isclose(grid, load - pv + charge - discharge, rel_tol=1e-7, abs_tol=1e-5):
                raise ValueError('meter energy balance failed')
            total = number('tariff/total_price_cost_this_step_c')
            components = sum(number('tariff/' + key) for key in ('energy_cost_this_step_c', 'demand_charge_increment_c', 'additional_charge_increment_c'))
            if not math.isclose(total, components, rel_tol=1e-7, abs_tol=1e-5):
                raise ValueError('tariff charge balance failed')
        summary['grid_peak_kw'] = max(float(r['battery/bat_grid_import_kwh']) * 4 for r in rows)
        summary['facility_peak_kw'] = max(float(r['power/dc_total_power_kW']) for r in rows)
        terminal_fields = {
            'terminal_soc_start': 'battery/bat_terminal_soc_start',
            'terminal_soc_end': 'battery/bat_terminal_soc_end',
            'terminal_soc_restore_grid_kwh': 'battery/bat_terminal_soc_restore_grid_kwh',
            'terminal_soc_reference_price_c_per_kwh': 'battery/bat_terminal_soc_reference_price_c_per_kwh',
        }
        summary.update({
            'mode': mode,
            'start_day': day,
            'seed': seed,
            'initial_soc': initial_soc,
            'final_soc': float(raw.bat_env.get_battery_soc()),
            'steps': len(rows),
            'complete': len(rows) == expected_steps,
            'start_clock': [rows[0].get('billed_day'), rows[0].get('billed_hour')],
            'end_clock': [rows[-1].get('next_state_day'), rows[-1].get('next_state_hour')],
        })
        for summary_key, trace_key in terminal_fields.items():
            value = rows[-1].get(trace_key)
            if value is not None and math.isfinite(float(value)):
                summary[summary_key] = float(value)
        reconcile_summary_against_trace(rows, summary)
        return rows, summary, np.asarray(exogenous)
    finally:
        env.close()


def evaluate_scenarios(config, run, windows, modes, seed, smoke_steps):
    """Run selected controllers on fixed scenarios using one loaded checkpoint.

    ``windows`` maps each window name to its month-start day.
    """
    checkpoint_env = HARLSustainDCEnv(config['env_args'])
    try:
        actors, hashes = load_checkpoint(run, checkpoint_env)
    finally:
        checkpoint_env.close()
    summaries, traces = [], {}
    for window, day in windows.items():
        reference = None
        for mode in modes:
            rows, summary, exogenous = run_episode(
                config, actors, mode, day, seed, smoke_steps
            )
            if reference is None:
                reference = exogenous
            elif not np.array_equal(reference, exogenous):
                raise RuntimeError(f'exogenous mismatch in {window} for {mode}; results invalid')
            summary['window'] = window
            summaries.append(summary)
            traces[(window, mode)] = rows
    return summaries, hashes, traces


def validate_saved_scenario(env_args):
    """Check that matched windows represent the supported joint scenario."""
    if (env_args.get('agents') != list(AGENTS)
            or env_args.get('datacenter_capacity_mw') not in (1, 10)
            or env_args.get('days_per_episode') != 30
            or env_args.get('pricing', {}).get('model') != 'hydro_quebec'):
        raise ValueError('expected three agents, 1 or 10 MW, 30-day saved configuration and Hydro-Québec pricing')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path, help='saved run containing config.json and models/')
    p.add_argument('--output-root', type=Path, default=ROOT / 'results' / 'matched-evaluations')
    p.add_argument('--seed', type=int, default=102)
    p.add_argument('--window-set', choices=WINDOW_SETS, default=DEFAULT_WINDOW_SET,
                   help='final (default): months outside checkpoint selection and floor '
                        'calibration; validation: the checkpoint-selection months')
    p.add_argument('--window', choices=[name for names in WINDOW_SETS.values() for name in names],
                   action='append', help='restrict to these months of the chosen --window-set')
    p.add_argument('--mode', choices=MODES, action='append')
    p.add_argument('--smoke-steps', type=int, help='short diagnostic only; incomplete results are not evaluations')
    args = p.parse_args()
    if args.smoke_steps is not None and not 1 <= args.smoke_steps < 31 * 96:
        p.error('smoke steps must be in 1..2975')
    try:
        windows = select_windows(args.window, args.window_set)
        modes = unique_requested(args.mode, MODES, '--mode')
    except ValueError as error:
        p.error(str(error))
    run = args.run.resolve()
    output_root = args.output_root.resolve()
    if output_root == run or run in output_root.parents:
        p.error('output root must not be inside the saved run')
    config = json.loads((run / 'config.json').read_text())
    try:
        validate_saved_scenario(config['env_args'])
    except ValueError as error:
        p.error(str(error))
    output_root.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix='matched-', dir=output_root))
    summaries, hashes, traces = evaluate_scenarios(
        config, run, windows, modes, args.seed, args.smoke_steps
    )
    for (window, mode), rows in traces.items():
        write_trace_csv(rows, out / f'{window}-{mode}.csv')
    pairs = build_paired_comparisons(summaries, windows, modes)
    versions = {}
    for package in ('numpy', 'torch', 'gymnasium', 'pettingzoo', 'supersuit', 'pvlib'):
        try: versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package] = None
    def git(*params):
        return subprocess.check_output(['git', '-C', str(ROOT), *params], text=True).strip()
    provenance = {'checkpoint': str(run), 'hashes_sha256': hashes, 'checkout_commit': git('rev-parse', 'HEAD'), 'checkout_status': git('status', '--porcelain'), 'python': platform.python_version(), 'dependencies': versions, 'config': config, 'scenario_seed': args.seed, 'window_set': args.window_set, 'windows': windows, 'limitations': [WINDOW_SET_NOTES[args.window_set], 'Fixed representative windows are not statistical samples.', 'No-op is not a rule-based controller.', 'Checkpoint training code revision is not established by this evaluation.', 'Actor-only inference; critic weights are validated but not used, and the value normalizer is not restored.', 'Grid-minus-gross-load includes PV and is not a battery-only counterfactual.', 'Smoke results are incomplete when smoke_steps is shorter than the calendar month.']}
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2, allow_nan=False))
    (out / 'summaries.json').write_text(json.dumps(summaries, indent=2, allow_nan=False))
    (out / 'paired-differences.json').write_text(json.dumps(pairs, indent=2, allow_nan=False))
    print(out)


if __name__ == '__main__':
    os.chdir(ROOT)
    main()
