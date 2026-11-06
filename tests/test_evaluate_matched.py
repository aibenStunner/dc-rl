"""Tests for standalone matched evaluator controller and checkpoint safeguards."""
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import evaluate_matched as evaluator


class Actor:
    def __init__(self, action):
        self.action = action

    def act(self, obs, state, mask, avail, deterministic):
        assert deterministic
        return torch.tensor([[self.action]]), torch.tensor(state + 1)


class MatchedTests(unittest.TestCase):
    def test_final_window_set_is_the_default(self):
        self.assertEqual(evaluator.DEFAULT_WINDOW_SET, 'final')
        final = evaluator.WINDOW_SETS['final']
        self.assertEqual(list(final.items()),
                         [('march', 59), ('june', 151), ('september', 243), ('december', 334)])
        self.assertEqual([evaluator.month_steps(day) for day in final.values()],
                         [31 * 96, 30 * 96, 30 * 96, 31 * 96])

    def test_validation_window_set_is_the_checkpoint_selection_months(self):
        from harl.envs.sustaindc.fixed_month_validation import DEFAULT_CASES

        validation = evaluator.WINDOW_SETS['validation']
        self.assertEqual(list(validation.items()),
                         [(case.name, case.init_day) for case in DEFAULT_CASES])
        self.assertEqual([evaluator.month_steps(day) for day in validation.values()],
                         [31 * 96, 30 * 96, 31 * 96, 31 * 96])
        # Saved analysis scripts read WINDOWS as the January/April/July/October set.
        self.assertIs(evaluator.WINDOWS, validation)

    def test_final_months_were_not_used_for_selection_or_floor_calibration(self):
        from harl.envs.sustaindc.fixed_month_validation import DEFAULT_CASES

        final_days = set(evaluator.WINDOW_SETS['final'].values())
        checkpoint_selection_days = {case.init_day for case in DEFAULT_CASES}
        # The 10 MW discharge floor was chosen on February/May/August/November.
        floor_selection_days = {31, 120, 212, 304}
        self.assertEqual(final_days & checkpoint_selection_days, set())
        self.assertEqual(final_days & floor_selection_days, set())

    def test_window_selection_stays_within_one_window_set(self):
        self.assertEqual(list(evaluator.select_windows(None, 'final').items()),
                         list(evaluator.WINDOW_SETS['final'].items()))
        self.assertEqual(list(evaluator.select_windows(['september', 'march'], 'final').items()),
                         [('september', 243), ('march', 59)])
        self.assertEqual(evaluator.select_windows(['spring'], 'validation'), {'spring': 90})
        with self.assertRaisesRegex(ValueError, "--window spring is not in window set 'final'"):
            evaluator.select_windows(['spring'], 'final')
        with self.assertRaisesRegex(ValueError, 'duplicate --window value: june'):
            evaluator.select_windows(['june', 'june'], 'final')

    def test_each_window_set_discloses_what_it_is_held_out_from(self):
        self.assertEqual(set(evaluator.WINDOW_SET_NOTES), set(evaluator.WINDOW_SETS))
        self.assertIn('not from training', evaluator.WINDOW_SET_NOTES['final'])
        self.assertIn('not held-out', evaluator.WINDOW_SET_NOTES['validation'])

    def _saved_run(self, root):
        run = root / 'run'
        run.mkdir()
        (run / 'config.json').write_text(json.dumps({'env_args': {
            'agents': list(evaluator.AGENTS), 'datacenter_capacity_mw': 10,
            'days_per_episode': 30, 'pricing': {'model': 'hydro_quebec'},
        }}))
        return run

    def test_cli_defaults_to_final_months_and_records_the_window_set(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = ['evaluate_matched.py', str(self._saved_run(root)),
                    '--output-root', str(root / 'out'), '--mode', 'noop']
            with patch.object(sys, 'argv', argv), \
                 patch.object(evaluator, 'evaluate_scenarios', return_value=([], {}, {})) as evaluate, \
                 patch('builtins.print'):
                evaluator.main()
            self.assertEqual(list(evaluate.call_args.args[2].items()),
                             list(evaluator.WINDOW_SETS['final'].items()))
            (out,) = (root / 'out').iterdir()
            provenance = json.loads((out / 'provenance.json').read_text())
        self.assertEqual(provenance['window_set'], 'final')
        self.assertEqual(provenance['windows'], evaluator.WINDOW_SETS['final'])
        self.assertIn(evaluator.WINDOW_SET_NOTES['final'], provenance['limitations'])

    def test_cli_rejects_a_window_outside_the_chosen_set(self):
        with tempfile.TemporaryDirectory() as directory:
            argv = ['evaluate_matched.py', str(Path(directory) / 'run'),
                    '--output-root', str(Path(directory) / 'out'), '--window', 'spring']
            with patch.object(sys, 'argv', argv), \
                 patch.object(evaluator, 'evaluate_scenarios') as evaluate, \
                 patch('sys.stderr', new_callable=io.StringIO) as stderr:
                with self.assertRaises(SystemExit):
                    evaluator.main()
            self.assertFalse((Path(directory) / 'out').exists())
        evaluate.assert_not_called()
        self.assertIn("not in window set 'final'", stderr.getvalue())

    def test_run_episode_handles_a_complete_28_day_calendar_month(self):
        from tests.test_observation_layout import _ENV_CONFIG

        config = {
            'env_args': dict(_ENV_CONFIG, days_per_episode=30, partial_obs=True,
                             nonoverlapping_shared_obs_space=True,
                             episode_start_policy='valid_midnight'),
            'algo_args': {'model': {'recurrent_n': 1, 'hidden_sizes': [64]}},
        }
        rows, summary, _ = evaluator.run_episode(config, [], 'noop', 31, 102)
        self.assertEqual(len(rows), 28 * 96)
        self.assertTrue(summary['complete'])
        self.assertEqual(summary['start_clock'], [31, 0])
        self.assertEqual(summary['end_clock'], [59, 0])
        self.assertIn('episode/tariff/cost/total_c', summary)
        self.assertIn('episode/energy/grid_import_kwh', summary)

    def test_controller_actions_and_recurrent_state(self):
        actors = [Actor(0), Actor(2), Actor(1)]
        obs = [np.zeros(2)] * 3
        states = [np.zeros((1, 1, 2)) for _ in range(3)]
        masks = [np.ones((1, 1))] * 3
        avail = [[1, 1, 1]] * 3
        proposed, executed = evaluator.choose_actions('checkpoint-battery-idle', actors, obs, states, masks, avail)
        self.assertEqual(proposed, [0, 2, 1])
        self.assertEqual(executed, [0, 2, 2])
        self.assertTrue(all(np.all(s == 1) for s in states))
        self.assertEqual(evaluator.choose_actions('noop', actors, obs, states, masks, avail), ([1, 1, 2], [1, 1, 2]))

    def test_cold_noop_reaches_lower_bound_without_clamping(self):
        self.assertIn('cold-noop', evaluator.MODES)
        for setpoint, expected in ((18.0, 0), (16.0, 0), (15.0, 1)):
            proposed, executed = evaluator.choose_actions(
                'cold-noop', [], [], [], [], [],
                dc_setpoint=setpoint, dc_min_setpoint=15.0,
            )
            self.assertEqual(proposed, executed)
            self.assertEqual(executed, [1, expected, 2])

    def test_trailing_peak_battery_is_causal_and_never_charges(self):
        self.assertIn('noop-trailing-peak-battery', evaluator.MODES)
        idle = evaluator.choose_actions(
            'noop-trailing-peak-battery', [], [], [], [], [],
            previous_facility_load_kw=900.0, running_peak_kw=1000.0,
            battery_soc=0.5,
        )
        discharge = evaluator.choose_actions(
            'noop-trailing-peak-battery', [], [], [], [], [],
            previous_facility_load_kw=1000.0, running_peak_kw=1000.0,
            battery_soc=0.5,
        )
        empty = evaluator.choose_actions(
            'noop-trailing-peak-battery', [], [], [], [], [],
            previous_facility_load_kw=1200.0, running_peak_kw=1000.0,
            battery_soc=0.0,
        )
        no_prior_peak = evaluator.choose_actions(
            'noop-trailing-peak-battery', [], [], [], [], [],
            previous_facility_load_kw=1200.0, running_peak_kw=0.0,
            battery_soc=0.5,
        )
        self.assertEqual(idle, ([1, 1, 2], [1, 1, 2]))
        self.assertEqual(discharge, ([1, 1, 1], [1, 1, 1]))
        self.assertEqual(empty, ([1, 1, 2], [1, 1, 2]))
        self.assertEqual(no_prior_peak, ([1, 1, 2], [1, 1, 2]))

    def test_trailing_peak_workload_battery_defers_then_recovers_queue(self):
        self.assertIn('trailing-peak-workload-battery', evaluator.MODES)
        defer_and_discharge = evaluator.choose_actions(
            'trailing-peak-workload-battery', [], [], [], [], [],
            previous_facility_load_kw=1000.0, running_peak_kw=1000.0,
            battery_soc=0.5, queue_depth=10,
        )
        recover_queue = evaluator.choose_actions(
            'trailing-peak-workload-battery', [], [], [], [], [],
            previous_facility_load_kw=900.0, running_peak_kw=1000.0,
            battery_soc=0.5, queue_depth=10,
        )
        empty_queue = evaluator.choose_actions(
            'trailing-peak-workload-battery', [], [], [], [], [],
            previous_facility_load_kw=900.0, running_peak_kw=1000.0,
            battery_soc=0.5, queue_depth=0,
        )
        no_prior_peak = evaluator.choose_actions(
            'trailing-peak-workload-battery', [], [], [], [], [],
            previous_facility_load_kw=1200.0, running_peak_kw=0.0,
            battery_soc=0.5, queue_depth=10,
        )
        self.assertEqual(defer_and_discharge, ([0, 1, 1], [0, 1, 1]))
        self.assertEqual(recover_queue, ([2, 1, 2], [2, 1, 2]))
        self.assertEqual(empty_queue, ([1, 1, 2], [1, 1, 2]))
        self.assertEqual(no_prior_peak, ([1, 1, 2], [1, 1, 2]))

    def test_saved_scenario_accepts_current_one_mw_and_ten_mw_runs(self):
        for capacity in (1, 10):
            evaluator.validate_saved_scenario({
                'agents': list(evaluator.AGENTS),
                'datacenter_capacity_mw': capacity,
                'days_per_episode': 30,
                'pricing': {'model': 'hydro_quebec'},
            })
        with self.assertRaisesRegex(ValueError, 'expected three agents'):
            evaluator.validate_saved_scenario({
                'agents': ['agent_ls'], 'datacenter_capacity_mw': 1,
                'days_per_episode': 30,
            })

    def test_missing_weights_fail_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / 'config.json').write_text(json.dumps({'main_args': {'algo': 'happo'}, 'algo_args': {'algo': {'share_param': False}, 'train': {'use_valuenorm': True}}}))
            with self.assertRaises(FileNotFoundError):
                evaluator.load_checkpoint(run, None)


    def test_requested_values_reject_duplicates(self):
        self.assertEqual(
            evaluator.unique_requested(['winter', 'summer'], evaluator.WINDOWS, '--window'),
            ['winter', 'summer'],
        )
        with self.assertRaisesRegex(ValueError, 'duplicate --window value: winter'):
            evaluator.unique_requested(['winter', 'winter'], evaluator.WINDOWS, '--window')

    def test_paired_comparisons_include_cold_baseline(self):
        summaries = [
            {'window': 'winter', 'mode': 'noop', 'episode/tariff/cost/total_c': 100.0},
            {'window': 'winter', 'mode': 'cold-noop', 'episode/tariff/cost/total_c': 90.0},
            {'window': 'winter', 'mode': 'checkpoint', 'episode/tariff/cost/total_c': 95.0},
        ]
        pairs = evaluator.build_paired_comparisons(summaries, ['winter'],
                                                    ['noop', 'cold-noop', 'checkpoint'])
        self.assertEqual([(p['comparison'], p['differences']['episode/tariff/cost/total_c'])
                          for p in pairs],
                         [('cold-noop minus noop', -10.0),
                          ('checkpoint minus noop', -5.0),
                          ('checkpoint minus cold-noop', 5.0)])

    def test_paired_difference_includes_meter_peaks_and_terminal_soc(self):
        baseline = {
            'episode/tariff/cost/total_c': 100.0,
            'grid_peak_kw': 11.0,
            'facility_peak_kw': 13.0,
            'initial_soc': 0.5,
            'final_soc': 0.6,
            'terminal_soc_start': 0.6,
            'terminal_soc_end': 0.5,
            'terminal_soc_restore_grid_kwh': 2.0,
        }
        candidate = {key: value + 1.0 for key, value in baseline.items()}
        differences = evaluator.paired_difference(candidate, baseline)
        self.assertEqual(differences['episode/tariff/cost/total_c'], 1.0)
        self.assertEqual(differences['grid_peak_kw'], 1.0)
        self.assertEqual(differences['terminal_soc_restore_grid_kwh'], 1.0)

    def test_reconciliation_rejects_trace_summary_mismatch(self):
        rows = [{
            'tariff/total_price_cost_this_step_c': 4.0,
            'tariff/energy_cost_this_step_c': 1.0,
            'tariff/demand_charge_increment_c': 2.0,
            'tariff/additional_charge_increment_c': 1.0,
            'battery/bat_grid_import_kwh': 2.0,
            'power/dc_total_power_kW': 8.0,
        }]
        summary = {
            'episode/tariff/cost/total_c': 4.0,
            'episode/tariff/cost/energy_c': 1.0,
            'episode/tariff/cost/demand_increment_c': 2.0,
            'episode/tariff/cost/additional_c': 1.0,
            'episode/energy/grid_import_kwh': 2.0,
            'grid_peak_kw': 8.0,
            'facility_peak_kw': 8.0,
        }
        evaluator.reconcile_summary_against_trace(rows, summary)
        summary['episode/energy/grid_import_kwh'] = 3.0
        with self.assertRaisesRegex(ValueError, 'summary/trace mismatch'):
            evaluator.reconcile_summary_against_trace(rows, summary)

    def test_evaluate_scenarios_loads_weights_once(self):
        config = {'env_args': {'unused': True}}
        fake_env = unittest.mock.MagicMock()
        rows = [{'marker': 1}]
        with patch.object(evaluator, 'HARLSustainDCEnv', return_value=fake_env), \
             patch.object(evaluator, 'load_checkpoint', return_value=(['actors'], {'a': 'hash'})) as load, \
             patch.object(evaluator, 'run_episode', return_value=(rows, {'complete': True}, np.array([[1.0]]))) as run:
            summaries, hashes, traces = evaluator.evaluate_scenarios(
                config, Path('/checkpoint'), {'june': 151}, ['checkpoint', 'noop'], 102, 2880
            )
        self.assertEqual(load.call_count, 1)
        self.assertEqual([call.args[2:5] for call in run.call_args_list],
                         [('checkpoint', 151, 102), ('noop', 151, 102)])
        self.assertEqual(hashes, {'a': 'hash'})
        self.assertEqual(len(summaries), 2)
        self.assertEqual(set(traces), {('june', 'checkpoint'), ('june', 'noop')})
        fake_env.close.assert_called_once()


def test_matched_evaluator():
    result = unittest.TextTestRunner(verbosity=0).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(MatchedTests)
    )
    assert result.wasSuccessful(), result.failures + result.errors


if __name__ == '__main__':
    unittest.main()
