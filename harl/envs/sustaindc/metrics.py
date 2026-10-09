"""Facility-level SustainDC metric definitions and aggregation helpers."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Real
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MetricSpec:
    """Map one scalar info field to a canonical metric and reduction."""

    source_key: str
    canonical_suffix: str
    reduction: str
    required_in_targeted_mode: bool = False


_SUM_SPECS = (
    MetricSpec("energy_cost_this_step_c", "tariff/cost/energy_c", "sum"),
    MetricSpec("demand_charge_increment_c", "tariff/cost/demand_increment_c", "sum"),
    MetricSpec("additional_charge_increment_c", "tariff/cost/additional_c", "sum"),
    MetricSpec("total_price_cost_this_step_c", "tariff/cost/total_c", "sum"),
    MetricSpec("shared_tariff_cost_reward", "reward/shared_tariff_reward/sum", "sum", True),
    MetricSpec("agent_ls_reward", "reward/agent_ls/sum", "sum", True),
    MetricSpec("agent_dc_reward", "reward/agent_dc/sum", "sum", True),
    MetricSpec("agent_bat_reward", "reward/agent_bat/sum", "sum", True),
    MetricSpec("ls_overdue_penalty_component", "reward/penalty/ls_overdue/sum", "sum", True),
    MetricSpec("ls_dropped_penalty_component", "reward/penalty/ls_dropped/sum", "sum", True),
    MetricSpec("ls_backlog_penalty_component", "reward/penalty/ls_backlog/sum", "sum", True),
    MetricSpec("dc_thermal_penalty_component", "reward/penalty/dc_thermal/sum", "sum", True),
    MetricSpec("dc_constraint_penalty_component", "reward/penalty/dc_constraint/sum", "sum", True),
    MetricSpec("bat_terminal_soc_penalty_component", "reward/penalty/battery_terminal_soc/sum", "sum", True),
    MetricSpec("bat_degradation_penalty_component", "reward/penalty/battery_degradation/sum", "sum", True),
    MetricSpec("bat_grid_import_kwh", "energy/grid_import_kwh", "sum"),
    MetricSpec("bat_dc_load_kwh", "energy/dc_bus_load_kwh", "sum"),
    MetricSpec("bat_total_energy_without_battery_KWh", "energy/baseline_without_battery_kwh", "sum"),
    MetricSpec("pv_ac_kwh", "pv/generation_ac_kwh", "sum"),
    MetricSpec("pv_self_consumed_kwh", "pv/self_consumed_kwh", "sum"),
    MetricSpec("pv_curtailed_kwh", "pv/curtailed_kwh", "sum"),
    MetricSpec("bat_charge_bus_KWh", "battery/charge_bus_kwh", "sum"),
    MetricSpec("bat_discharge_bus_KWh", "battery/discharge_bus_kwh", "sum"),
    MetricSpec("bat_charge_cell_KWh", "battery/charge_cell_kwh", "sum"),
    MetricSpec("bat_discharge_cell_KWh", "battery/discharge_cell_kwh", "sum"),
    MetricSpec("bat_cell_throughput_KWh", "battery/cell_throughput_kwh", "sum"),
    MetricSpec("bat_degradation_cost_c", "battery/degradation_cost_c", "sum"),
    MetricSpec("bat_terminal_soc_restore_grid_kwh", "battery/terminal_soc_restore_grid_kwh", "sum"),
    MetricSpec("bat_CO2_footprint", "carbon/emissions", "sum"),
    MetricSpec("ls_original_workload", "workload/original_fraction/sum", "sum"),
    MetricSpec("ls_shifted_workload", "workload/shifted_fraction/sum", "sum"),
    MetricSpec("ls_computed_tasks", "workload/tasks/computed", "sum"),
    MetricSpec("ls_tasks_processed", "workload/tasks/processed", "sum"),
    MetricSpec("ls_tasks_dropped", "sla/dropped_tasks", "sum"),
    MetricSpec("ls_overdue_penalty", "sla/overdue_tasks/sum", "sum"),
    MetricSpec("ls_enforced", "sla/enforced_steps", "sum"),
    MetricSpec("dc_water_usage", "water/liters", "sum"),
    MetricSpec("dc_constraint_violation", "thermal/constraint", "sum"),
)
_MEAN_SPECS = (
    MetricSpec("price_denorm_c_per_kwh", "tariff/price_c_per_kwh", "mean"),
    MetricSpec("bat_avg_CI", "carbon/grid_intensity", "mean"),
    MetricSpec("ls_original_workload", "workload/original_fraction/mean", "mean"),
    MetricSpec("ls_shifted_workload", "workload/shifted_fraction/mean", "mean"),
    MetricSpec("ls_norm_tasks_in_queue", "sla/queue_depth_normalized/mean", "mean"),
    MetricSpec("ls_oldest_task_age", "sla/oldest_task_age_fraction/mean", "mean"),
    MetricSpec("ls_average_task_age", "sla/average_task_age_fraction/mean", "mean"),
    MetricSpec("dc_ITE_total_power_kW", "power/ite_kw/mean", "mean"),
    MetricSpec("dc_HVAC_total_power_kW", "power/hvac_kw/mean", "mean"),
    MetricSpec("dc_CT_total_power_kW", "power/cooling_tower_kw/mean", "mean"),
    MetricSpec("dc_Compressor_total_power_kW", "power/chiller_kw/mean", "mean"),
    MetricSpec("dc_total_power_kW", "power/facility_kw/mean", "mean"),
    MetricSpec("dc_crac_setpoint", "thermal/crac_setpoint_c/mean", "mean"),
    MetricSpec("dc_int_temperature", "thermal/internal_temperature_c/mean", "mean"),
    MetricSpec("dc_exterior_ambient_temp", "thermal/ambient_temperature_c/mean", "mean"),
)
_MAX_SPECS = (
    MetricSpec("ls_tasks_in_queue", "sla/queue_depth/max", "max"),
    MetricSpec("ls_norm_tasks_in_queue", "sla/queue_depth_normalized/max", "max"),
    MetricSpec("ls_oldest_task_age", "sla/oldest_task_age_fraction/max", "max"),
    MetricSpec("ls_overdue_penalty", "sla/overdue_tasks/max", "max"),
    MetricSpec("dc_int_temperature", "thermal/internal_temperature_c/max", "max"),
)
_LAST_SPECS = (
    MetricSpec("ls_tasks_in_queue", "sla/queue_depth/final", "last"),
    MetricSpec("ls_norm_tasks_in_queue", "sla/queue_depth_normalized/final", "last"),
    MetricSpec("bat_cumulative_cell_throughput_KWh", "battery/cumulative_cell_throughput_kwh", "last"),
    MetricSpec("bat_cumulative_degradation_cost_c", "battery/cumulative_degradation_cost_c", "last"),
    MetricSpec("bat_terminal_soc_reference_price_c_per_kwh", "battery/terminal_soc_reference_price_c_per_kwh", "last"),
    MetricSpec("billing_running_peak_kw", "tariff/billing/running_peak_kw", "last"),
    MetricSpec("billing_period_energy_kwh", "tariff/billing/energy_kwh", "last"),
    MetricSpec("billing_progress_fraction", "tariff/billing/progress_fraction", "last"),
    MetricSpec("dc_thermal_limit_c", "thermal/limit", "last"),
)


def _finite_real(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


class FacilityMetricAccumulator:
    """Aggregate one complete facility info dictionary per simulator step."""

    def __init__(self, namespace: str, *, targeted_mode: bool = False) -> None:
        self.namespace = namespace.rstrip("/")
        self.targeted_mode = targeted_mode
        self.reset()

    def reset(self) -> None:
        self._values: dict[str, list[float]] = defaultdict(list)
        self._missing = 0
        self._nonfinite = 0
        self._steps = 0
        self._actions = defaultdict(int)

    def _add_spec(self, info: Mapping[str, object], spec: MetricSpec) -> None:
        if spec.source_key not in info:
            if self.targeted_mode and spec.required_in_targeted_mode:
                raise KeyError(f"missing required targeted metric {spec.source_key}")
            self._missing += 1
            return
        value = info[spec.source_key]
        if not _finite_real(value):
            self._nonfinite += 1
            return
        self._values[spec.canonical_suffix].append(float(value))

    def add(self, info: Mapping[str, object]) -> None:
        """Add one facility-level record, never a duplicate per-agent record."""
        self._steps += 1
        for spec in _SUM_SPECS + _MEAN_SPECS + _MAX_SPECS + _LAST_SPECS:
            self._add_spec(info, spec)
        action = info.get("bat_a_t")
        if action in {"charge", "discharge", "idle"}:
            self._actions[str(action)] += 1
        elif action is not None:
            self._missing += 1
        if "bat_SOC" in info and _finite_real(info["bat_SOC"]):
            self._values["battery/soc"].append(float(info["bat_SOC"]))
        elif "bat_SOC" in info:
            self._nonfinite += 1
        else:
            self._missing += 1
        if "ls_tasks_in_queue" in info and _finite_real(info["ls_tasks_in_queue"]):
            self._values["sla/queue_depth"].append(float(info["ls_tasks_in_queue"]))

    def _metric(self, suffix: str, value: float) -> tuple[str, float]:
        return f"{self.namespace}/{suffix}", float(value)

    def snapshot(self) -> dict[str, float]:
        """Return reductions for all accumulated values without resetting them."""
        result: dict[str, float] = {}
        for specs, reduction in ((_SUM_SPECS, sum), (_MEAN_SPECS, np.mean), (_MAX_SPECS, max)):
            for spec in specs:
                values = self._values.get(spec.canonical_suffix)
                if values:
                    key, value = self._metric(spec.canonical_suffix, reduction(values))
                    result[key] = value
        for spec in _LAST_SPECS:
            values = self._values.get(spec.canonical_suffix)
            if values:
                key, value = self._metric(spec.canonical_suffix, values[-1])
                result[key] = value

        soc_values = self._values.get("battery/soc", [])
        if soc_values:
            for suffix, value in (("battery/soc/start", soc_values[0]), ("battery/soc/end", soc_values[-1]), ("battery/soc/min", min(soc_values)), ("battery/soc/max", max(soc_values)), ("battery/soc/mean", np.mean(soc_values))):
                key, numeric = self._metric(suffix, value)
                result[key] = numeric
        queue_values = self._values.get("sla/queue_depth", [])
        if queue_values:
            for suffix, value in (("sla/queue_depth/mean", np.mean(queue_values)), ("sla/queue_depth/max", max(queue_values)), ("sla/queue_depth/final", queue_values[-1]), ("sla/queue_task_steps", sum(queue_values))):
                key, numeric = self._metric(suffix, value)
                result[key] = numeric
        for action in ("charge", "discharge", "idle"):
            key, value = self._metric(f"battery/action/{action}_steps", self._actions[action])
            result[key] = value

        internal = self._values.get("thermal/internal_temperature_c/mean", [])
        limit = self._values.get("thermal/limit", [])
        if internal and limit:
            paired = list(zip(internal, limit))
            margins = [threshold - temperature for temperature, threshold in paired]
            for suffix, value in (("thermal/temperature_margin_c/mean", np.mean(margins)), ("thermal/temperature_margin_c/min", min(margins)), ("thermal/exceedance_degree_c_steps", sum(max(0.0, -margin) for margin in margins))):
                key, numeric = self._metric(suffix, value)
                result[key] = numeric
        constraint = self._values.get("thermal/constraint", [])
        if constraint:
            key, value = self._metric("thermal/constraint_violation_steps", sum(constraint))
            result[key] = value

        grid_import = sum(self._values.get("energy/grid_import_kwh", []))
        baseline = sum(self._values.get("energy/baseline_without_battery_kwh", []))
        if "energy/grid_import_kwh" in self._values:
            key, value = self._metric("energy/battery_delta_kwh", grid_import - baseline)
            result[key] = value
        pv_generation = sum(self._values.get("pv/generation_ac_kwh", []))
        if pv_generation > 0:
            for suffix, value in (("pv/self_consumption_fraction", sum(self._values.get("pv/self_consumed_kwh", [])) / pv_generation), ("pv/curtailment_fraction", sum(self._values.get("pv/curtailed_kwh", [])) / pv_generation)):
                key, numeric = self._metric(suffix, value)
                result[key] = numeric
        if grid_import > 0:
            for suffix, value in (("carbon/emissions_per_grid_import_kwh", sum(self._values.get("carbon/emissions", [])) / grid_import), ("water/liters_per_grid_import_kwh", sum(self._values.get("water/liters", [])) / grid_import)):
                key, numeric = self._metric(suffix, value)
                result[key] = numeric
        ite = self._values.get("power/ite_kw/mean", [])
        hvac = self._values.get("power/hvac_kw/mean", [])
        if ite and sum(ite) > 0:
            key, value = self._metric("efficiency/pue", 1.0 + sum(hvac) / sum(ite))
            result[key] = value

        agent_sums = []
        for agent in ("ls", "dc", "bat"):
            values = self._values.get(f"reward/agent_{agent}/sum", [])
            if values:
                total = sum(values)
                agent_sums.append(total)
                key, value = self._metric(f"reward/agent_{agent}/mean_per_step", np.mean(values))
                result[key] = value
        if agent_sums:
            key, value = self._metric("reward/mean_across_agents/sum", np.mean(agent_sums))
            result[key] = value
            if self._steps:
                key, value = self._metric("reward/mean_across_agents/mean_per_step", np.mean(agent_sums) / self._steps)
                result[key] = value
        values = self._values.get("reward/shared_tariff_reward/sum")
        if values:
            key, value = self._metric("reward/shared_tariff_reward/mean_per_step", np.mean(values))
            result[key] = value
        result[f"{self.namespace}/diagnostic/info_missing_key_count"] = float(self._missing)
        result[f"{self.namespace}/diagnostic/nonfinite_metric_count"] = float(self._nonfinite)
        result[f"{self.namespace}/diagnostic/step_count"] = float(self._steps)
        return result


_TRACE_PREFIXES = (
    ("reward/", (
        "shared_tariff_cost_reward", "agent_ls_reward", "agent_dc_reward",
        "agent_bat_reward", "ls_overdue_penalty_component",
        "ls_dropped_penalty_component", "ls_backlog_penalty_component",
        "dc_thermal_penalty_component", "dc_constraint_penalty_component",
        "bat_terminal_soc_penalty_component", "bat_degradation_penalty_component",
    )),
    ("tariff/", (
        "norm_price", "price_denorm_c_per_kwh", "energy_cost_this_step_c",
        "demand_charge_increment_c", "additional_charge_increment_c",
        "total_price_cost_this_step_c", "billing_running_peak_kw",
        "billing_period_energy_kwh", "billing_progress_fraction", "selected_tariff",
    )),
    ("battery/", (
        "bat_action", "bat_a_t", "bat_SOC", "bat_grid_import_kwh",
        "bat_dc_load_kwh", "bat_total_energy_without_battery_KWh",
        "bat_charge_bus_KWh", "bat_discharge_bus_KWh", "bat_charge_cell_KWh",
        "bat_discharge_cell_KWh", "bat_cell_throughput_KWh",
        "bat_cumulative_cell_throughput_KWh", "bat_degradation_cost_c",
        "bat_cumulative_degradation_cost_c", "bat_terminal_soc_start",
        "bat_terminal_soc_end", "bat_terminal_soc_restore_grid_kwh",
        "bat_terminal_soc_reference_price_c_per_kwh",
    )),
    ("pv/", ("pv_ac_kwh", "pv_self_consumed_kwh", "pv_curtailed_kwh")),
    ("workload/", (
        "ls_original_workload", "ls_shifted_workload", "ls_action",
        "ls_computed_tasks", "ls_tasks_processed", "ls_enforced",
    )),
    ("sla/", (
        "ls_tasks_in_queue", "ls_norm_tasks_in_queue", "ls_tasks_dropped",
        "ls_overdue_penalty", "ls_oldest_task_age", "ls_average_task_age",
        "ls_task_age_histogram",
    )),
    ("thermal/", (
        "dc_crac_setpoint", "dc_int_temperature", "dc_exterior_ambient_temp",
        "dc_thermal_limit_c", "dc_constraint_violation",
    )),
    ("power/", (
        "dc_ITE_total_power_kW", "dc_HVAC_total_power_kW", "dc_CT_total_power_kW",
        "dc_Compressor_total_power_kW", "dc_total_power_kW",
        "dc_CW_pump_power_kW", "dc_CT_pump_power_kW",
    )),
    ("water/", ("dc_water_usage",)),
    ("carbon/", ("bat_avg_CI", "bat_CO2_footprint")),
)


def _trace_value(value):
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist())
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value))
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_trace_row(*, info, actions, run_metadata, terminal, truncated):
    """Build one lossless facility trace record from merged SustainDC info."""
    row = dict(run_metadata)
    for key in ("billed_day", "billed_hour", "next_state_day", "next_state_hour"):
        row[key] = _trace_value(info.get(key))
    for prefix, keys in _TRACE_PREFIXES:
        for key in keys:
            if key in info:
                row[f"{prefix}{key}"] = _trace_value(info[key])
    action_names = ("ls", "dc", "bat")
    flattened = np.asarray(actions).reshape(len(action_names), -1)
    for index, name in enumerate(action_names):
        value = flattened[index][0]
        row[f"action/{name}"] = _trace_value(value)
    row["terminal"] = bool(terminal)
    row["truncated"] = bool(truncated)
    return row


def write_trace_csv(rows, path):
    """Write complete records with unioned columns and no numeric rounding."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def aggregate_metric_snapshots(snapshots: Sequence[Mapping[str, float]], namespace: str) -> dict[str, float]:
    """Aggregate completed same-schema episode snapshots by canonical suffix."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for snapshot in snapshots:
        for key, value in snapshot.items():
            if key.startswith("eval/episode/") and _finite_real(value):
                grouped[key.removeprefix("eval/episode/")].append(float(value))
    result = {f"{namespace}/episode_count": float(len(snapshots))}
    for suffix, values in grouped.items():
        result[f"{namespace}/{suffix}/mean"] = float(np.mean(values))
        result[f"{namespace}/{suffix}/std"] = float(np.std(values))
        result[f"{namespace}/{suffix}/min"] = float(min(values))
        result[f"{namespace}/{suffix}/max"] = float(max(values))
    return result
