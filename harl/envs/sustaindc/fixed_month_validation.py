"""Deterministic complete-month validation for SustainDC checkpoint selection.

This validator intentionally uses independent calendar months. Its aggregate is
not a contiguous annual utility bill and it does not itself compare a policy
against no-op or cold-noop baselines.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
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
    maximum = rules.get("max_terminal_soc_restore_grid_kwh", math.inf)
    return float(metrics.get("episode/terminal/soc_restore_grid_kwh", 0.0)) <= float(maximum)


def select_best(results: Iterable[ValidationResult], rules: Mapping[str, object]) -> ValidationResult | None:
    """Choose the eligible candidate by bill, peak, then earlier event."""
    candidates = [result for result in results if eligible(result, rules)]
    if not candidates:
        return None
    return min(candidates, key=lambda result: (
        result.aggregate["episode/tariff/cost/total_c"],
        result.aggregate["grid_peak_kw"],
        result.event,
    ))


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
        summary["episode/terminal/soc_restore_grid_kwh"] = float(rows[-1]["battery/bat_terminal_soc_restore_grid_kwh"])
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


def _aggregate_cases(cases: Sequence[dict[str, object]]) -> dict[str, float]:
    summaries = [case["summary"] for case in cases]
    bill_key = "episode/tariff/cost/total_c"
    aggregate = {bill_key: sum(float(summary.get(bill_key, 0.0)) for summary in summaries), "grid_peak_kw": max(float(summary["grid_peak_kw"]) for summary in summaries)}
    for key in ("episode/service/queue_overflow_forced_steps", "episode/service/overdue_steps", "episode/thermal/clamp_steps", "episode/thermal/capacity_exceeded_steps", "episode/terminal/soc_restore_grid_kwh"):
        aggregate[key] = sum(float(summary.get(key, 0.0)) for summary in summaries)
    return aggregate
