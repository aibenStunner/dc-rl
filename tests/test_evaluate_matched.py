"""Tests for standalone matched evaluator controller and checkpoint safeguards."""
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
    def test_scenario_manifest(self):
        self.assertEqual(list(evaluator.WINDOWS.values()), [14, 105, 196, 287])
        self.assertTrue(all(b - a >= 30 for a, b in zip(evaluator.WINDOWS.values(), list(evaluator.WINDOWS.values())[1:])))

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
             patch.object(evaluator, 'run_episode', return_value=(rows, {'complete': True}, np.array([[1.0]]))):
            summaries, hashes, traces = evaluator.evaluate_scenarios(
                config, Path('/checkpoint'), ['winter'], ['checkpoint', 'noop'], 102, 2880
            )
        self.assertEqual(load.call_count, 1)
        self.assertEqual(hashes, {'a': 'hash'})
        self.assertEqual(len(summaries), 2)
        self.assertEqual(set(traces), {('winter', 'checkpoint'), ('winter', 'noop')})
        fake_env.close.assert_called_once()


def test_matched_evaluator():
    result = unittest.TextTestRunner(verbosity=0).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(MatchedTests)
    )
    assert result.wasSuccessful(), result.failures + result.errors


if __name__ == '__main__':
    unittest.main()
