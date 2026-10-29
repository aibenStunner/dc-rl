"""Matched fixed-window checkpoint evaluation; never writes into a training run."""
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

AGENTS = ('agent_ls', 'agent_dc', 'agent_bat')
WINDOWS = {'winter': 14, 'spring': 105, 'summer': 196, 'autumn': 287}
MODES = ('checkpoint', 'noop', 'checkpoint-battery-idle')
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


def choose_actions(mode, actors, observations, states, masks, avail):
    proposed = []
    if mode == 'noop':
        return [1, 1, 2], [1, 1, 2]
    with torch.no_grad():
        for i, actor in enumerate(actors):
            action, state = actor.act(np.asarray(observations[i])[None], states[i], masks[i], np.asarray(avail[i])[None], deterministic=True)
            states[i] = state.cpu().numpy()
            proposed.append(int(action.cpu().numpy().reshape(-1)[0]))
    executed = proposed.copy()
    if mode == 'checkpoint-battery-idle':
        executed[2] = 2
    return proposed, executed


def run_episode(config, actors, mode, day, seed, max_steps=2880):
    env = HARLSustainDCEnv(config['env_args'])
    try:
        obs, _, avail = env.reset(seed=seed, options={'init_day': day, 'init_hour': 0})
        raw = env.env.unwrapped.env
        initial_soc = float(raw.bat_env.get_battery_soc())
        model = config['algo_args']['model']
        states = [np.zeros((1, model['recurrent_n'], model['hidden_sizes'][-1]), dtype=np.float32) for _ in AGENTS]
        masks = [np.ones((1, 1), dtype=np.float32) for _ in AGENTS]
        rows, exogenous = [], []
        metrics = FacilityMetricAccumulator('episode')
        for step in range(max_steps):
            common = raw.infos['__common__']
            exo = (float(common['workload']), float(common['weather']), float(common['ci']), float(raw.pv_m.get_current_ac_kwh()))
            proposed, executed = choose_actions(mode, actors, obs, states, masks, avail)
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
                if step != 2879 or terminated or not truncated:
                    raise RuntimeError(f'unexpected episode end at step {step + 1}: terminated={term}, truncated={trunc}')
                break
        if max_steps == 2880 and (len(rows) != 2880 or not rows[-1]['truncated']):
            raise RuntimeError('episode did not truncate exactly at 2880 steps')
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
            'complete': len(rows) == 2880,
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
    """Run selected controllers on fixed scenarios using one loaded checkpoint."""
    checkpoint_env = HARLSustainDCEnv(config['env_args'])
    try:
        actors, hashes = load_checkpoint(run, checkpoint_env)
    finally:
        checkpoint_env.close()
    summaries, traces = [], {}
    for window in windows:
        reference = None
        for mode in modes:
            rows, summary, exogenous = run_episode(
                config, actors, mode, WINDOWS[window], seed, smoke_steps
            )
            if reference is None:
                reference = exogenous
            elif not np.array_equal(reference, exogenous):
                raise RuntimeError(f'exogenous mismatch in {window} for {mode}; results invalid')
            summary['window'] = window
            summaries.append(summary)
            traces[(window, mode)] = rows
    return summaries, hashes, traces


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path, help='saved run containing config.json and models/')
    p.add_argument('--output-root', type=Path, default=ROOT / 'results' / 'matched-evaluations')
    p.add_argument('--seed', type=int, default=102)
    p.add_argument('--window', choices=WINDOWS, action='append')
    p.add_argument('--mode', choices=MODES, action='append')
    p.add_argument('--smoke-steps', type=int, default=2880, help='short diagnostic only; incomplete results are not evaluations')
    args = p.parse_args()
    if not 1 <= args.smoke_steps <= 2880:
        p.error('smoke steps must be in 1..2880')
    try:
        windows = unique_requested(args.window, WINDOWS, '--window')
        modes = unique_requested(args.mode, MODES, '--mode')
    except ValueError as error:
        p.error(str(error))
    run = args.run.resolve()
    output_root = args.output_root.resolve()
    if output_root == run or run in output_root.parents:
        p.error('output root must not be inside the saved run')
    config = json.loads((run / 'config.json').read_text())
    if config['env_args']['agents'] != list(AGENTS) or config['env_args']['datacenter_capacity_mw'] != 10 or config['env_args']['days_per_episode'] != 30:
        p.error('expected all three agents, 10 MW and 30-day saved configuration')
    output_root.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix='matched-', dir=output_root))
    summaries, hashes, traces = evaluate_scenarios(
        config, run, windows, modes, args.seed, args.smoke_steps
    )
    for (window, mode), rows in traces.items():
        write_trace_csv(rows, out / f'{window}-{mode}.csv')
    pairs = []
    for window in windows:
        group = {s['mode']: s for s in summaries if s['window'] == window}
        if 'noop' in group:
            for mode in modes:
                if mode != 'noop' and mode in group:
                    pairs.append({
                        'window': window,
                        'comparison': mode + ' minus noop',
                        'differences': paired_difference(group[mode], group['noop']),
                    })
    versions = {}
    for package in ('numpy', 'torch', 'gymnasium', 'pettingzoo', 'supersuit', 'pvlib'):
        try: versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package] = None
    def git(*params):
        return subprocess.check_output(['git', '-C', str(ROOT), *params], text=True).strip()
    provenance = {'checkpoint': str(run), 'hashes_sha256': hashes, 'checkout_commit': git('rev-parse', 'HEAD'), 'checkout_status': git('status', '--porcelain'), 'python': platform.python_version(), 'dependencies': versions, 'config': config, 'scenario_seed': args.seed, 'windows': WINDOWS, 'limitations': ['Fixed representative windows are not statistical samples.', 'No-op is not a rule-based controller.', 'Checkpoint training code revision is not established by this evaluation.', 'Actor-only inference; critic weights are validated but not used, and the value normalizer is not restored.', 'Grid-minus-gross-load includes PV and is not a battery-only counterfactual.', 'Smoke results are incomplete when smoke_steps < 2880.']}
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2, allow_nan=False))
    (out / 'summaries.json').write_text(json.dumps(summaries, indent=2, allow_nan=False))
    (out / 'paired-differences.json').write_text(json.dumps(pairs, indent=2, allow_nan=False))
    print(out)


if __name__ == '__main__':
    os.chdir(ROOT)
    main()
