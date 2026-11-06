"""Tests for objective-aligned fixed-calendar-month validation."""
import json
import math
import sys

import numpy as np
import torch
import tempfile
from pathlib import Path
from unittest.mock import patch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests._harness import run_module_tests
import harl.envs.sustaindc.fixed_month_validation as fixed_validation
from harl.envs.sustaindc.fixed_month_validation import (
    DEFAULT_CASES,
    ValidationCase,
    ValidationResult,
    eligible,
    select_best,
)


def _result(*, event=1, bill=100.0, peak=100.0, **metrics):
    return ValidationResult(
        event=event,
        cases=(),
        aggregate={
            "episode/tariff/cost/total_c": bill,
            "grid_peak_kw": peak,
            "episode/service/queue_overflow_forced_steps": 0.0,
            "episode/service/overdue_steps": 0.0,
            "episode/thermal/clamp_steps": 0.0,
            "episode/thermal/capacity_exceeded_steps": 0.0,
            "episode/terminal/soc_restore_grid_kwh": 0.0,
            "episode/terminal/soc_restore_cost_c": 0.0,
            **metrics,
        },
    )


def _drained(*, event=1, bill=100.0, peak=100.0, restore_kwh=500.0, refill_c=20.0):
    return _result(event=event, bill=bill, peak=peak, **{
        "episode/terminal/soc_restore_grid_kwh": restore_kwh,
        "episode/terminal/soc_restore_cost_c": refill_c,
    })


class _ConstantActor:
    def act(self, observations, state, masks, available_actions, deterministic):
        assert deterministic is True
        return torch.tensor([[1]]), torch.from_numpy(state)


def _two_step_month_env(final_info):
    """Fake env: a two-step month whose last step adds ``final_info``."""
    class FakeEnv:
        def __init__(self, env_args):
            self.steps = 0

        def reset(self, *, seed, options):
            return [np.array([0.0])] * 3, None, [np.array([1.0])] * 3

        def step(self, actions):
            self.steps += 1
            done = self.steps == 2
            info = {
                "bat_grid_import_kwh": 5.0,
                "total_price_cost_this_step_c": 2.0,
                "bat_terminal_soc_restore_grid_kwh": 0.0,
                "bat_terminal_soc_reference_price_c_per_kwh": 0.0,
                "_harl_terminated": False,
                "_harl_truncated": done,
            }
            if done:
                info = {**{k: v for k, v in info.items()
                           if k != "bat_terminal_soc_reference_price_c_per_kwh"}, **final_info}
            return [np.array([0.0])] * 3, None, [[0.0]] * 3, [done] * 3, [info] * 3, [np.array([1.0])] * 3

        def close(self):
            pass

    return FakeEnv


def test_default_cases_are_fixed_quarterly_calendar_month_starts():
    assert [(case.name, case.init_day) for case in DEFAULT_CASES] == [
        ("winter", 0), ("spring", 90), ("summer", 181), ("autumn", 273),
    ]


def test_eligibility_rejects_service_thermal_and_terminal_soc_failures():
    rules = {
        "require_zero_queue_overflow": True,
        "require_zero_overdue_steps": True,
        "require_zero_thermal_clamp_steps": True,
        "require_zero_capacity_exceeded_steps": True,
        "max_terminal_soc_restore_grid_kwh": 0.0,
    }
    assert eligible(_result(), rules)
    assert not eligible(_result(**{"episode/service/overdue_steps": 1.0}), rules)
    assert not eligible(_result(**{"episode/thermal/capacity_exceeded_steps": 1.0}), rules)
    assert not eligible(_result(**{"episode/terminal/soc_restore_grid_kwh": 0.01}), rules)


def test_ineligible_validation_evidence_is_retained_without_checkpoint():
    from harl.runners.on_policy_base_runner import OnPolicyBaseRunner

    with tempfile.TemporaryDirectory() as directory:
        candidates = Path(directory)
        (candidates / "event-1").mkdir()
        (candidates / "event-1" / "validation.json").write_text("{}")
        runner = object.__new__(OnPolicyBaseRunner)
        runner.fixed_validation = {"require_zero_overdue_steps": True, "retain_top_k": 1}
        retained = runner._retain_validation_candidates(candidates, [{
            "event": 1,
            "eligible": False,
            "cases": [],
            "aggregate": _result(**{"episode/service/overdue_steps": 1.0}).aggregate,
        }])
        assert retained[0]["event"] == 1
        assert (candidates / "event-1" / "validation.json").is_file()
        assert json.loads((candidates / "manifest.json").read_text())["candidates"][0]["eligible"] is False


def _validation_runner(save_dir):
    from types import SimpleNamespace
    from harl.runners.on_policy_base_runner import OnPolicyBaseRunner

    runner = object.__new__(OnPolicyBaseRunner)
    runner.save_dir = str(save_dir)
    Path(save_dir).mkdir()
    runner.validation_event = 0
    runner.fixed_validation = {"seed": 102, "require_zero_overdue_steps": True, "retain_top_k": 5}
    runner.algo_args = {"seed": {"seed": 1}}
    runner.env_args = {}
    runner.recurrent_n = 1
    runner.rnn_hidden_size = 1
    runner.num_agents = 3
    runner.actor = [SimpleNamespace(actor=torch.nn.Linear(1, 1)) for _ in range(3)]
    runner.critic = SimpleNamespace(critic=torch.nn.Linear(1, 1))
    runner.value_normalizer = torch.nn.Linear(1, 1)
    return runner


def test_every_validation_event_keeps_its_evaluated_weights_in_last_snapshot():
    import harl.runners.on_policy_base_runner as runner_module

    def ineligible_report(**kwargs):
        return _result(event=kwargs["event"], **{"episode/service/overdue_steps": 1.0})

    with tempfile.TemporaryDirectory() as directory:
        runner = _validation_runner(Path(directory) / "models")
        last = Path(runner.save_dir) / "last"
        with patch.object(runner_module, "validate_fixed_months", ineligible_report):
            runner._run_fixed_month_validation()
            assert (last / "actor_agent0.pt").is_file(), "ineligible event left no weights on disk"
            first = torch.load(last / "actor_agent0.pt", weights_only=True)["weight"].clone()
            with torch.no_grad():
                runner.actor[0].actor.weight.add_(1.0)
            runner._run_fixed_month_validation()

        assert sorted(path.name for path in last.iterdir()) == [
            "actor_agent0.pt", "actor_agent1.pt", "actor_agent2.pt",
            "critic_agent.pt", "last.json", "value_normalizer.pt",
        ]
        assert json.loads((last / "last.json").read_text())["validation_event"] == 2
        assert torch.equal(torch.load(last / "actor_agent0.pt", weights_only=True)["weight"], first + 1.0)
        assert not (Path(runner.save_dir) / "last.partial").exists()
        assert not list(Path(runner.save_dir).glob("*.pt")), "ineligible weights were exported as the winner"
        assert not list((Path(runner.save_dir) / "candidates" / "event-2").glob("*.pt"))


def test_last_snapshot_is_written_before_validation_can_fail():
    import harl.runners.on_policy_base_runner as runner_module

    def crashing_validator(**kwargs):
        raise RuntimeError("validator crashed")

    with tempfile.TemporaryDirectory() as directory:
        runner = _validation_runner(Path(directory) / "models")
        with patch.object(runner_module, "validate_fixed_months", crashing_validator):
            try:
                runner._run_fixed_month_validation()
            except RuntimeError:
                pass
            else:
                raise AssertionError("validator crash was swallowed")
        assert (Path(runner.save_dir) / "last" / "actor_agent0.pt").is_file()


def test_runner_uses_fixed_validation_instead_of_legacy_checkpoint_selection():
    from harl.runners.on_policy_base_runner import OnPolicyBaseRunner

    class Logger:
        def init(self, episodes):
            assert episodes == 1
        def episode_init(self, episode):
            pass
        def per_step(self, data):
            pass
        def episode_log(self, *args):
            pass

    runner = object.__new__(OnPolicyBaseRunner)
    runner.algo_args = {
        "train": {
            "num_env_steps": 1, "episode_length": 1, "n_rollout_threads": 1,
            "use_linear_lr_decay": False, "log_interval": 1, "eval_interval": 1,
            "render_interval": 1,
        },
        "eval": {"use_eval": True},
        "render": {"use_render": False},
    }
    runner.fixed_validation_enabled = True
    runner.actor_buffer = []
    runner.critic_buffer = None
    runner.logger = Logger()
    runner.warmup = lambda: None
    runner.prep_rollout = lambda: None
    runner.collect = lambda step: (None,) * 5
    runner.envs = type("Envs", (), {"step": lambda self, actions: (None,) * 6})()
    runner.insert = lambda data: None
    runner.compute = lambda: None
    runner.prep_training = lambda: None
    runner.train = lambda: ({}, {})
    runner.after_update = lambda: None
    runner._run_fixed_month_validation = lambda: setattr(runner, "fixed_called", True)
    runner.eval = lambda: (_ for _ in ()).throw(AssertionError("legacy eval called"))
    runner.save = lambda: (_ for _ in ()).throw(AssertionError("legacy save called"))
    runner.run()
    assert runner.fixed_called


def test_validator_rejects_partial_agent_completion():
    class Actor:
        def act(self, observations, state, masks, available_actions, deterministic):
            return torch.tensor([[1]]), torch.from_numpy(state)

    class FakeEnv:
        def __init__(self, env_args):
            pass

        def reset(self, *, seed, options):
            return [np.array([0.0])] * 3, None, [np.array([1.0])] * 3

        def step(self, actions):
            info = {
                "bat_grid_import_kwh": 0.0,
                "bat_terminal_soc_restore_grid_kwh": 0.0,
                "_harl_terminated": False,
                "_harl_truncated": False,
            }
            return [np.array([0.0])] * 3, None, [[0.0]] * 3, [True, False, False], [info] * 3, [np.array([1.0])] * 3

        def close(self):
            pass

    with patch.object(fixed_validation, "HARLSustainDCEnv", FakeEnv), \
         patch.object(fixed_validation, "month_steps", return_value=2):
        try:
            fixed_validation.validate_fixed_months(
                env_args={}, actors=[Actor(), Actor(), Actor()], recurrent_n=1,
                rnn_hidden_size=1, seed=1, event=1, cases=(ValidationCase("test", 0),),
            )
        except RuntimeError as exc:
            assert "ended inconsistently" in str(exc)
        else:
            raise AssertionError("partial agent completion was accepted")


def test_validator_rejects_final_step_termination_not_month_truncation():
    class Actor:
        def act(self, observations, state, masks, available_actions, deterministic):
            return torch.tensor([[1]]), torch.from_numpy(state)

    class FakeEnv:
        def __init__(self, env_args):
            pass

        def reset(self, *, seed, options):
            return [np.array([0.0])] * 3, None, [np.array([1.0])] * 3

        def step(self, actions):
            info = {
                "bat_grid_import_kwh": 0.0,
                "bat_terminal_soc_restore_grid_kwh": 0.0,
                "_harl_terminated": True,
                "_harl_truncated": False,
            }
            return [np.array([0.0])] * 3, None, [[0.0]] * 3, [True] * 3, [info] * 3, [np.array([1.0])] * 3

        def close(self):
            pass

    with patch.object(fixed_validation, "HARLSustainDCEnv", FakeEnv), \
         patch.object(fixed_validation, "month_steps", return_value=1):
        try:
            fixed_validation.validate_fixed_months(
                env_args={}, actors=[Actor(), Actor(), Actor()], recurrent_n=1,
                rnn_hidden_size=1, seed=1, event=1, cases=(ValidationCase("test", 0),),
            )
        except RuntimeError as exc:
            assert "terminated" in str(exc)
        else:
            raise AssertionError("final-step termination was accepted as a normal month end")


def test_candidate_retention_prunes_only_excess_eligible_snapshots_and_exports_winner():
    from harl.runners.on_policy_base_runner import OnPolicyBaseRunner

    with tempfile.TemporaryDirectory() as directory:
        candidates_dir = Path(directory) / "candidates"
        candidates_dir.mkdir()
        runner = object.__new__(OnPolicyBaseRunner)
        runner.fixed_validation = {"require_zero_overdue_steps": True, "retain_top_k": 1}
        runner.save_dir = Path(directory) / "models"
        runner.save_dir.mkdir()

        candidates = []
        for event, bill, overdue in ((1, 20.0, 0.0), (2, 10.0, 0.0), (3, 1.0, 1.0)):
            event_dir = candidates_dir / f"event-{event}"
            event_dir.mkdir()
            (event_dir / "validation.json").write_text("{}")
            torch.save({"event": event}, event_dir / "actor_agent0.pt")
            candidates.append({
                "event": event,
                "eligible": overdue == 0.0,
                "cases": [],
                "aggregate": _result(event=event, bill=bill, **{"episode/service/overdue_steps": overdue}).aggregate,
            })

        retained = runner._retain_validation_candidates(candidates_dir, candidates)
        assert [candidate["event"] for candidate in retained] == [2, 3]
        assert not (candidates_dir / "event-1").exists()
        assert (candidates_dir / "event-2" / "actor_agent0.pt").is_file()
        assert (candidates_dir / "event-3" / "validation.json").is_file()

        runner._export_validation_winner(candidates_dir / "event-2")
        assert torch.load(runner.save_dir / "actor_agent0.pt", weights_only=True) == {"event": 2}


def test_candidate_retention_rejects_nonpositive_limit():
    from harl.runners.on_policy_base_runner import OnPolicyBaseRunner

    with tempfile.TemporaryDirectory() as directory:
        runner = object.__new__(OnPolicyBaseRunner)
        runner.fixed_validation = {"retain_top_k": 0}
        try:
            runner._retain_validation_candidates(Path(directory), [])
        except ValueError as exc:
            assert "retain_top_k" in str(exc)
        else:
            raise AssertionError("nonpositive candidate retention was accepted")


def test_selection_is_eligibility_first_then_bill_peak_and_event():
    rules = {"require_zero_overdue_steps": True}
    ineligible_low_bill = _result(event=1, bill=1.0, peak=1.0, **{"episode/service/overdue_steps": 1.0})
    high_peak = _result(event=2, bill=10.0, peak=20.0)
    low_peak = _result(event=3, bill=10.0, peak=15.0)
    later_tie = _result(event=4, bill=10.0, peak=15.0)
    assert select_best([ineligible_low_bill, high_peak, later_tie, low_peak], rules) == low_peak


def test_selection_charges_the_terminal_soc_refill_to_the_bill():
    # Draining the battery at month end lowers the modeled bill; the energy
    # needed to refill it to the starting SOC is charged back before ranking.
    kept_soc = _result(event=2, bill=100.0)
    assert select_best([_drained(event=1, bill=90.0, refill_c=20.0), kept_soc], {}) == kept_soc
    assert select_best([_drained(event=1, bill=90.0, refill_c=5.0), kept_soc], {}).event == 1


def test_null_terminal_soc_cap_means_no_limit():
    drained = _drained(restore_kwh=1.0e6)
    assert eligible(drained, {"max_terminal_soc_restore_grid_kwh": None})
    assert not eligible(drained, {"max_terminal_soc_restore_grid_kwh": 10.0})


def test_shipped_rules_rank_drained_candidates_instead_of_rejecting_them():
    from harl.utils.configs_tools import get_defaults_yaml_args

    algo_args, _ = get_defaults_yaml_args("happo", "sustaindc")
    rules = algo_args["eval"]["fixed_month_validation"]
    assert rules["max_terminal_soc_restore_grid_kwh"] is None
    assert eligible(_drained(), rules)


def test_candidate_retention_keeps_the_restoration_adjusted_best():
    from harl.runners.on_policy_base_runner import OnPolicyBaseRunner

    with tempfile.TemporaryDirectory() as directory:
        candidates_dir = Path(directory)
        runner = object.__new__(OnPolicyBaseRunner)
        runner.fixed_validation = {"retain_top_k": 1}
        candidates = []
        for result in (_drained(event=1, bill=90.0, refill_c=20.0), _result(event=2, bill=100.0)):
            (candidates_dir / f"event-{result.event}").mkdir()
            candidates.append({"event": result.event, "eligible": True, "cases": [],
                               "aggregate": result.aggregate})
        retained = runner._retain_validation_candidates(candidates_dir, candidates)
        assert [candidate["event"] for candidate in retained] == [2]
        assert not (candidates_dir / "event-1").exists()
        assert (candidates_dir / "event-2").is_dir()


def test_validator_values_the_terminal_refill_at_the_reported_reference_price():
    final_info = {
        "bat_terminal_soc_restore_grid_kwh": 100.0,
        "bat_terminal_soc_reference_price_c_per_kwh": 3.821,
    }
    with patch.object(fixed_validation, "HARLSustainDCEnv", _two_step_month_env(final_info)), \
         patch.object(fixed_validation, "month_steps", return_value=2):
        result = fixed_validation.validate_fixed_months(
            env_args={}, actors=[_ConstantActor()] * 3, recurrent_n=1,
            rnn_hidden_size=1, seed=1, event=1,
            cases=(ValidationCase("first", 0), ValidationCase("second", 0)),
        )
    for case in result.cases:
        assert math.isclose(case["summary"]["episode/terminal/soc_restore_cost_c"], 382.1)
    aggregate = result.aggregate
    assert math.isclose(aggregate["episode/terminal/soc_restore_cost_c"], 764.2)
    assert math.isclose(aggregate["episode/tariff/cost/total_c"], 8.0)
    assert math.isclose(aggregate["restoration_adjusted_bill_c"], 772.2)


def test_validator_refuses_to_value_a_refill_without_its_reference_price():
    final_info = {"bat_terminal_soc_restore_grid_kwh": 100.0}
    with patch.object(fixed_validation, "HARLSustainDCEnv", _two_step_month_env(final_info)), \
         patch.object(fixed_validation, "month_steps", return_value=2):
        try:
            fixed_validation.validate_fixed_months(
                env_args={}, actors=[_ConstantActor()] * 3, recurrent_n=1,
                rnn_hidden_size=1, seed=1, event=1, cases=(ValidationCase("test", 0),),
            )
        except RuntimeError as exc:
            assert "reference price" in str(exc)
        else:
            raise AssertionError("a terminal refill was valued without its reference price")


def test_validator_forwards_fixed_reset_and_accepts_exact_final_done():
    class Actor:
        def act(self, observations, state, masks, available_actions, deterministic):
            assert deterministic is True
            return torch.tensor([[1]]), torch.from_numpy(state)

    class FakeEnv:
        instance = None

        def __init__(self, env_args):
            self.env_args = env_args
            self.reset_calls = []
            self.steps = 0
            FakeEnv.instance = self

        def reset(self, *, seed, options):
            self.reset_calls.append((seed, options))
            return [np.array([0.0])] * 3, None, [np.array([1.0])] * 3

        def step(self, actions):
            self.steps += 1
            done = self.steps == 2
            info = {
                "bat_grid_import_kwh": 5.0,
                "bat_terminal_soc_restore_grid_kwh": 0.0,
                "total_price_cost_this_step_c": 2.0,
                "_harl_terminated": False,
                "_harl_truncated": done,
            }
            return [np.array([0.0])] * 3, None, [[0.0]] * 3, [done] * 3, [info] * 3, [np.array([1.0])] * 3

        def close(self):
            pass

    with patch.object(fixed_validation, "HARLSustainDCEnv", FakeEnv), \
         patch.object(fixed_validation, "month_steps", return_value=2):
        result = fixed_validation.validate_fixed_months(
            env_args={"ignored": True}, actors=[Actor(), Actor(), Actor()],
            recurrent_n=1, rnn_hidden_size=1, seed=102, event=4,
            cases=(ValidationCase("test", 0),),
        )
    assert FakeEnv.instance.env_args["episode_start_policy"] == "calendar_month"
    assert FakeEnv.instance.reset_calls == [(102, {"init_day": 0, "init_hour": 0})]
    assert result.cases[0]["steps"] == 2
    assert result.aggregate["grid_peak_kw"] == 20.0


if __name__ == "__main__":
    sys.exit(1 if run_module_tests(globals()) else 0)
