"""Deterministic complete-month validation for SustainDC checkpoint selection.

This validator intentionally uses independent calendar months. Its aggregate is
not a contiguous annual utility bill and it does not itself compare a policy
against no-op or cold-noop baselines.

Eligible candidates are ranked by a restoration-adjusted bill: the modeled
tariff bill plus the energy cost of refilling the battery to its starting state
of charge. The refill is a selection adjustment, not a tariff charge.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from harl.envs.sustaindc.harlsustaindc_env import HARLSustainDCEnv
from harl.envs.sustaindc.metrics import FacilityMetricAccumulator, build_trace_row, write_trace_csv
from harl.envs.sustaindc.sequential import sequential_battery_index
from utils.managers.pricing.models.hydro_quebec import _MONTH_START_DAYS


@dataclass(frozen=True)
class ValidationCase:
    name: str
    init_day: int


DEFAULT_CASES = (
    ValidationCase("winter", 0),
    ValidationCase("spring", 90),
    ValidationCase("summer", 181),
    ValidationCase("autumn", 273),
)


@dataclass(frozen=True)
class ValidationResult:
    event: int
    cases: tuple[dict[str, object], ...]
    aggregate: dict[str, float]

    def to_json(self) -> dict[str, object]:
        return {"event": self.event, "cases": list(self.cases), "aggregate": self.aggregate}


def month_steps(init_day: int) -> int:
    """Return non-leap calendar-month length in SustainDC 15-minute steps."""
    try:
        index = _MONTH_START_DAYS.index(init_day)
    except ValueError as exc:
        raise ValueError(f"init_day must start a calendar month: {init_day}") from exc
    end_day = (*_MONTH_START_DAYS[1:], 365)[index]
    return (end_day - init_day) * 96


def eligible(result: ValidationResult, rules: Mapping[str, object]) -> bool:
    """Apply predeclared service, thermal, and terminal-SOC gates."""
    metrics = result.aggregate
    zero_gates = (
        ("require_zero_queue_overflow", "episode/service/queue_overflow_forced_steps"),
        ("require_zero_overdue_steps", "episode/service/overdue_steps"),
        ("require_zero_thermal_clamp_steps", "episode/thermal/clamp_steps"),
        ("require_zero_capacity_exceeded_steps", "episode/thermal/capacity_exceeded_steps"),
    )
    for option, metric in zero_gates:
        if rules.get(option, False) and metrics.get(metric, 0.0) > 0.0:
            return False
    maximum = rules.get("max_terminal_soc_restore_grid_kwh")
    if maximum is None:  # absent or null: no cap; ranking charges the refill instead
        return True
    return float(metrics.get("episode/terminal/soc_restore_grid_kwh", 0.0)) <= float(maximum)


def restoration_adjusted_bill_c(aggregate: Mapping[str, float]) -> float:
    """Return the modeled bill plus the cost of refilling the battery.

    The refill is valued as the environment's terminal-SOC settlement values it:
    grid energy needed to restore the starting SOC, at the energy price of the
    month's final interval. Any demand charge the refill might cause is excluded.
    """
    return (float(aggregate["episode/tariff/cost/total_c"])
            + float(aggregate["episode/terminal/soc_restore_cost_c"]))


def ranking_key(result: ValidationResult) -> tuple[float, float, int]:
    """Order candidates by restoration-adjusted bill, then peak, then earlier event."""
    return (restoration_adjusted_bill_c(result.aggregate),
            float(result.aggregate["grid_peak_kw"]), result.event)


def select_best(results: Iterable[ValidationResult], rules: Mapping[str, object]) -> ValidationResult | None:
    """Choose the eligible candidate with the lowest ``ranking_key``."""
    candidates = [result for result in results if eligible(result, rules)]
    if not candidates:
        return None
    return min(candidates, key=ranking_key)


def validate_fixed_months(
    *, env_args: Mapping[str, object], actors: Sequence[object], recurrent_n: int,
    rnn_hidden_size: int, seed: int, event: int, output_dir: Path | None = None,
    cases: Sequence[ValidationCase] = DEFAULT_CASES,
) -> ValidationResult:
    """Evaluate deterministic actors on direct, complete fixed-month episodes."""
    all_cases = []
    for case in cases:
        all_cases.append(_validate_case(
            env_args=dict(env_args), actors=actors, recurrent_n=recurrent_n,
            rnn_hidden_size=rnn_hidden_size, seed=seed, event=event, case=case,
            output_dir=output_dir,
        ))
    return ValidationResult(event=event, cases=tuple(all_cases), aggregate=_aggregate_cases(all_cases))


def _validate_case(*, env_args, actors, recurrent_n, rnn_hidden_size, seed, event, case, output_dir):
    expected_steps = month_steps(case.init_day)
    battery_index = sequential_battery_index(env_args)
    env = HARLSustainDCEnv(dict(env_args, episode_start_policy="calendar_month"))
    try:
        observations, _, available_actions = env.reset(seed=seed, options={"init_day": case.init_day, "init_hour": 0})
        states = [np.zeros((1, recurrent_n, rnn_hidden_size), dtype=np.float32) for _ in actors]
        masks = [np.ones((1, 1), dtype=np.float32) for _ in actors]
        metrics, rows = FacilityMetricAccumulator("episode"), []
        for step in range(expected_steps):
            actions = []
            with torch.no_grad():
                for agent_id, actor in enumerate(actors):
                    obs_i = env.begin_step(np.asarray(actions)) if agent_id == battery_index else observations[agent_id]
                    action, next_state = actor.act(np.asarray(obs_i)[None], states[agent_id], masks[agent_id], np.asarray(available_actions[agent_id])[None], deterministic=True)
                    states[agent_id] = next_state.cpu().numpy()
                    actions.append(int(action.cpu().numpy().reshape(-1)[0]))
            observations, _, _, dones, infos, available_actions = env.step(np.asarray(actions))
            values = infos[0]
            done_flags = np.asarray(dones, dtype=bool)
            terminated_flags = np.asarray(
                [info.get("_harl_terminated", False) for info in infos], dtype=bool
            )
            truncated_flags = np.asarray(
                [info.get("_harl_truncated", False) for info in infos], dtype=bool
            )
            episode_done = bool(np.all(done_flags))
            if np.any(done_flags) and not episode_done:
                raise RuntimeError(
                    f"validation case {case.name} ended inconsistently at step {step + 1}"
                )
            if np.any(terminated_flags):
                raise RuntimeError(
                    f"validation case {case.name} terminated at step {step + 1}"
                )
            metrics.add(values)
            rows.append(build_trace_row(info=values, actions=actions, run_metadata={"event": event, "case": case.name, "start_day": case.init_day, "seed": seed, "step": step}, terminal=False, truncated=episode_done))
            if episode_done:
                if step != expected_steps - 1 or not np.all(truncated_flags):
                    raise RuntimeError(f"validation case {case.name} ended unexpectedly at step {step + 1}")
                break
        if len(rows) != expected_steps:
            raise RuntimeError(f"validation case {case.name} has {len(rows)} steps; expected {expected_steps}")
        summary = metrics.snapshot()
        summary["grid_peak_kw"] = max(float(row["battery/bat_grid_import_kwh"]) * 4.0 for row in rows)
        restore_kwh = float(rows[-1]["battery/bat_terminal_soc_restore_grid_kwh"])
        summary["episode/terminal/soc_restore_grid_kwh"] = restore_kwh
        summary["episode/terminal/soc_restore_cost_c"] = _refill_cost_c(rows[-1], restore_kwh, case.name)
        summary["episode/service/queue_overflow_forced_steps"] = summary.get("episode/workload/queue_overflow_forced_tasks", 0.0)
        summary["episode/service/overdue_steps"] = summary.get("episode/sla/overdue_task_steps", 0.0)
        summary["episode/thermal/clamp_steps"] = summary.get("episode/thermal/crac_clamp_steps", 0.0)
        summary["episode/thermal/capacity_exceeded_steps"] = summary.get("episode/thermal/cooling_capacity_exceeded_steps", 0.0)
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            write_trace_csv(rows, output_dir / f"event-{event}-{case.name}.csv")
        return {"name": case.name, "init_day": case.init_day, "steps": len(rows), "summary": summary}
    finally:
        env.close()


def _refill_cost_c(final_row, restore_kwh, case_name):
    if restore_kwh == 0.0:
        return 0.0
    price = final_row.get("battery/bat_terminal_soc_reference_price_c_per_kwh")
    if price is None:
        raise RuntimeError(
            f"validation case {case_name} needs a battery refill but reports no reference price"
        )
    return restore_kwh * float(price)


def _aggregate_cases(cases: Sequence[dict[str, object]]) -> dict[str, float]:
    summaries = [case["summary"] for case in cases]
    bill_key = "episode/tariff/cost/total_c"
    aggregate = {bill_key: sum(float(summary.get(bill_key, 0.0)) for summary in summaries), "grid_peak_kw": max(float(summary["grid_peak_kw"]) for summary in summaries)}
    for key in ("episode/service/queue_overflow_forced_steps", "episode/service/overdue_steps", "episode/thermal/clamp_steps", "episode/thermal/capacity_exceeded_steps", "episode/terminal/soc_restore_grid_kwh", "episode/terminal/soc_restore_cost_c"):
        aggregate[key] = sum(float(summary.get(key, 0.0)) for summary in summaries)
    aggregate["restoration_adjusted_bill_c"] = restoration_adjusted_bill_c(aggregate)
    return aggregate
