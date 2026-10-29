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
    MetricSpec("bat_CO2_footprint", "carbon/emissions_g", "sum"),
    MetricSpec("ls_original_workload", "workload/original_fraction/sum", "sum"),
    MetricSpec("ls_shifted_workload", "workload/shifted_fraction/sum", "sum"),
    MetricSpec("ls_computed_tasks", "workload/tasks/computed", "sum"),
    MetricSpec("ls_tasks_processed", "workload/tasks/processed", "sum"),
    MetricSpec("ls_tasks_dropped", "workload/queue_overflow_forced_tasks", "sum"),
    MetricSpec("ls_overdue_penalty", "sla/overdue_task_steps", "sum"),
    MetricSpec("ls_enforced", "sla/enforced_steps", "sum"),
    MetricSpec("dc_water_usage", "water/liters", "sum"),
    MetricSpec("dc_constraint_violation", "thermal/crac_clamp_steps", "sum"),
)
_MEAN_SPECS = (
    MetricSpec("price_denorm_c_per_kwh", "tariff/price_c_per_kwh", "mean"),
    MetricSpec("bat_avg_CI", "carbon/grid_intensity_g_per_kwh", "mean"),
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
    MetricSpec("ls_overdue_penalty", "sla/overdue_tasks_per_step/max", "max"),
    MetricSpec("dc_int_temperature", "thermal/internal_temperature_c/max", "max"),
)
_LAST_SPECS = (
    
    
    MetricSpec("bat_cumulative_cell_throughput_KWh", "battery/cumulative_cell_throughput_kwh", "last"),
    MetricSpec("bat_cumulative_degradation_cost_c", "battery/cumulative_degradation_cost_c", "last"),
    MetricSpec("bat_terminal_soc_reference_price_c_per_kwh", "battery/terminal_soc_reference_price_c_per_kwh", "last"),
    MetricSpec("billing_running_peak_kw", "tariff/billing/running_peak_kw", "last"),
    MetricSpec("billing_period_energy_kwh", "tariff/billing/energy_kwh", "last"),
    MetricSpec("billing_progress_fraction", "tariff/billing/progress_fraction", "last"),
    MetricSpec("dc_thermal_limit_c", "thermal/crac_upper_bound_c", "last"),
)


def _finite_real(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


# ---------------------------------------------------------------------------
# Console summary
#
# The canonical metric set is deliberately broad (it is the tracking surface
# for W&B and TensorBoard). The console needs a small, stable, human-readable
# selection instead, so a run can be judged from its log without opening a
# dashboard. Each entry maps a canonical metric suffix to a short label and a
# unit; the suffix is matched against the snapshot *after* the namespace
# prefix is stripped.
# ---------------------------------------------------------------------------

CONSOLE_SUMMARY_GROUPS: tuple[tuple[str, str, tuple[tuple[str, str, str], ...]], ...] = (
    (
        "tariff",
        "Tariff",
        (
            ("tariff/cost/total_c", "cost", "c"),
            ("tariff/cost/demand_increment_c", "demand", "c"),
            ("tariff/billing/running_peak_kw", "peak", "kW"),
            ("tariff/price_c_per_kwh", "price", "c/kWh"),
        ),
    ),
    (
        "energy",
        "Energy",
        (
            ("energy/grid_import_kwh", "grid", "kWh"),
            ("energy/baseline_without_battery_kwh", "no-bat", "kWh"),
            ("energy/grid_minus_gross_load_kwh", "grid-gross", "kWh"),
            ("pv/generation_ac_kwh", "pv", "kWh"),
            ("pv/curtailment_fraction", "pv-curtailed", ""),
        ),
    ),
    (
        "workload",
        "Workload",
        (
            ("workload/original_fraction/mean", "orig", ""),
            ("workload/shifted_fraction/mean", "shifted", ""),
            ("sla/queue_depth/max", "queue-max", ""),
            ("sla/queue_depth/last_sample", "queue-last", ""),
            ("workload/queue_overflow_forced_tasks", "forced", ""),
            ("sla/overdue_task_steps", "overdue", ""),
        ),
    ),
    (
        "thermal",
        "Thermal",
        (
            ("thermal/internal_temperature_c/mean", "int", "C"),
            ("thermal/ambient_temperature_c/mean", "amb", "C"),
            ("thermal/outlet_minus_crac_upper_bound_c/max", "outlet-upper-bound-max", "C"),
            ("thermal/crac_clamp_steps", "clamp-steps", ""),
        ),
    ),
    (
        "battery",
        "Battery",
        (
            ("battery/soc/first_post_step", "soc-first-post-step", ""),
            ("battery/soc/last_post_step", "soc-last-post-step", ""),
            ("battery/action/charge_steps", "charge", "steps"),
            ("battery/action/discharge_steps", "discharge", "steps"),
            ("battery/cell_throughput_kwh", "throughput", "kWh"),
            ("battery/degradation_cost_c", "degradation", "c"),
        ),
    ),
    (
        "carbon",
        "Carbon/Water",
        (
            ("carbon/emissions_kg", "co2", "kg"),
            ("carbon/grid_intensity_g_per_kwh", "CI", "g/kWh"),
            ("water/liters", "water", "L"),
        ),
    ),
    (
        "reward",
        "Reward",
        (
            ("reward/shared_tariff_reward/sum", "tariff-r", ""),
            ("reward/agent_ls/sum", "ls", ""),
            ("reward/agent_dc/sum", "dc", ""),
            ("reward/agent_bat/sum", "bat", ""),
            ("reward/penalty/ls_overdue/sum", "ls-overdue", ""),
            ("reward/penalty/ls_dropped/sum", "ls-dropped", ""),
            ("reward/penalty/dc_thermal/sum", "dc-thermal", ""),
            ("reward/penalty/battery_terminal_soc/sum", "bat-soc", ""),
        ),
    ),
)


def strip_namespace(snapshot: Mapping[str, object], namespace: str) -> dict[str, float]:
    """Return the snapshot's numeric entries with `namespace` stripped.

    Keys outside the namespace, and non-finite values, are dropped so the
    result can be keyed by canonical suffix.
    """
    prefix = namespace.rstrip("/") + "/"
    result: dict[str, float] = {}
    for key, value in snapshot.items():
        if not key.startswith(prefix):
            continue
        if _finite_real(value):
            result[key[len(prefix):]] = float(value)
    return result


def _format_console_value(value: float, unit: str) -> str:
    if unit == "":
        if abs(value) >= 100:
            return f"{value:.0f}"
        if abs(value) >= 1:
            return f"{value:.2f}"
        return f"{value:.3f}"
    if abs(value) >= 100:
        return f"{value:.0f}{unit}"
    return f"{value:.2f}{unit}"


def format_console_summary(
    snapshot: Mapping[str, object],
    namespace: str,
    groups: Sequence[tuple[str, str, tuple[tuple[str, str, str], ...]]] = CONSOLE_SUMMARY_GROUPS,
) -> str:
    """Render a curated, human-readable summary of a metric snapshot.

    Only the configured suffixes are shown, and only when the snapshot
    actually contains them, so a summary never invents a value.
    """
    values = strip_namespace(snapshot, namespace)
    lines: list[str] = []
    for _, title, entries in groups:
        parts: list[str] = []
        for suffix, label, unit in entries:
            if suffix not in values:
                continue
            parts.append(f"{label}={_format_console_value(values[suffix], unit)}")
        if parts:
            lines.append(f"  {title:<13} " + "  ".join(parts))
    return "\n".join(lines)


class FacilityMetricAccumulator:
    """Online, bounded aggregation of one facility record per worker-step.

    Historical carbon/emissions was grams despite a kg console label; the new
    unit-suffixed metrics do not preserve that ambiguous tag. Historical
    train/episode values pooled workers and are now named train/rollout.
    SoC endpoints are post-step samples, not reset state or episode endpoints.
    """

    def __init__(self, namespace: str, *, targeted_mode: bool = False) -> None:
        self.namespace = namespace.rstrip("/")
        self.targeted_mode = targeted_mode
        self.reset()

    def reset(self) -> None:
        self._stats = {}
        self._missing = self._nonfinite = self._steps = 0
        self._actions = defaultdict(int)

    def _record(self, key, value):
        value = float(value)
        if key not in self._stats:
            self._stats[key] = [0.0, 0, value, value, value, value]
        stat = self._stats[key]
        stat[0] += value
        stat[1] += 1
        stat[2] = min(stat[2], value)
        stat[3] = max(stat[3], value)
        stat[5] = value

    def add(self, info: Mapping[str, object]) -> None:
        self._steps += 1
        for spec in _SUM_SPECS + _MEAN_SPECS + _MAX_SPECS + _LAST_SPECS:
            if spec.source_key not in info:
                if self.targeted_mode and spec.required_in_targeted_mode:
                    raise KeyError(f"missing required targeted metric {spec.source_key}")
                self._missing += 1
                continue
            value = info[spec.source_key]
            if isinstance(value, (bool, np.bool_)) and spec.source_key in ("ls_enforced", "dc_constraint_violation"):
                value = int(value)
            if not _finite_real(value):
                self._nonfinite += 1
                continue
            self._record(spec.canonical_suffix, value)
        action = info.get("bat_a_t")
        if isinstance(action, str) and action in {"charge", "discharge", "idle"}:
            self._actions[action] += 1
        for source, key in (("bat_SOC", "battery/soc"), ("ls_tasks_in_queue", "sla/queue_depth")):
            if _finite_real(info.get(source)):
                self._record(key, info[source])
        # Only same-step finite pairs may contribute to derived ratios/margins.
        for name, left, right in (("grid_gross", "bat_grid_import_kwh", "bat_total_energy_without_battery_KWh"),
                                  ("pue", "dc_ITE_total_power_kW", "dc_HVAC_total_power_kW"),
                                  ("outlet_setpoint", "dc_int_temperature", "dc_thermal_limit_c"),
                                  ("carbon_grid", "bat_CO2_footprint", "bat_grid_import_kwh"),
                                  ("water_grid", "dc_water_usage", "bat_grid_import_kwh"),
                                  ("pv_self", "pv_self_consumed_kwh", "pv_ac_kwh"),
                                  ("pv_curtail", "pv_curtailed_kwh", "pv_ac_kwh")):
            if _finite_real(info.get(left)) and _finite_real(info.get(right)):
                self._record("pair/" + name + "/left", info[left])
                self._record("pair/" + name + "/right", info[right])
                if name == "outlet_setpoint":
                    self._record("pair/outlet_setpoint/difference", info[left] - info[right])

    def snapshot(self) -> dict[str, float]:
        result = {}
        def put(key, value):
            result[f"{self.namespace}/{key}"] = float(value)
        last_specs = () if self.namespace.startswith("train/") else _LAST_SPECS
        for specs, index in ((_SUM_SPECS, 0), (_MAX_SPECS, 3), (last_specs, 5)):
            for spec in specs:
                if spec.canonical_suffix in self._stats:
                    put(spec.canonical_suffix, self._stats[spec.canonical_suffix][index])
        for spec in _MEAN_SPECS:
            st = self._stats.get(spec.canonical_suffix)
            if st: put(spec.canonical_suffix, st[0] / st[1])
        for key, prefix in (("battery/soc", "battery/soc"), ("sla/queue_depth", "sla/queue_depth")):
            st = self._stats.get(key)
            if st:
                put(prefix + "/mean", st[0] / st[1]); put(prefix + "/max", st[3])
                if key == "battery/soc":
                    put(prefix + "/min", st[2])
                    if not self.namespace.startswith("train/"):
                        put(prefix + "/first_post_step", st[4]); put(prefix + "/last_post_step", st[5])
                else:
                    if not self.namespace.startswith("train/"):
                        put(prefix + "/last_sample", st[5])
                    put("sla/queue_task_steps", st[0])
        if self._actions:
            for action in ("charge", "discharge", "idle"):
                put(f"battery/action/{action}_steps", self._actions[action])
        def pair(name):
            l = self._stats.get("pair/" + name + "/left")
            r = self._stats.get("pair/" + name + "/right")
            return (l[0], r[0]) if l and r else None
        p = pair("grid_gross")
        if p: put("energy/grid_minus_gross_load_kwh", p[0] - p[1])
        p = pair("pue")
        if p and p[0] > 0: put("efficiency/pue", 1 + p[1] / p[0])
        st = self._stats.get("pair/outlet_setpoint/difference")
        if st:
            put("thermal/outlet_minus_crac_upper_bound_c/mean", st[0] / st[1]); put("thermal/outlet_minus_crac_upper_bound_c/max", st[3])
        for name, suffix in (("carbon_grid", "carbon/emissions_g_per_grid_import_kwh"), ("water_grid", "water/liters_per_grid_import_kwh"), ("pv_self", "pv/self_consumption_fraction"), ("pv_curtail", "pv/curtailment_fraction")):
            p = pair(name)
            if p and p[1] > 0: put(suffix, p[0] / p[1])
        st = self._stats.get("carbon/emissions_g")
        if st: put("carbon/emissions_kg", st[0] / 1000)
        agent_sums = []
        for agent in ("ls", "dc", "bat"):
            st = self._stats.get(f"reward/agent_{agent}/sum")
            if st:
                agent_sums.append(st[0]); put(f"reward/agent_{agent}/mean_per_step", st[0] / st[1])
        if len(agent_sums) == 3 and all(
            self._stats[f"reward/agent_{agent}/sum"][1] == self._steps
            for agent in ("ls", "dc", "bat")
        ):
            put("reward/mean_across_agents/sum", np.mean(agent_sums))
            if self._steps: put("reward/mean_across_agents/mean_per_step", np.mean(agent_sums) / self._steps)
        st = self._stats.get("reward/shared_tariff_reward/sum")
        if st: put("reward/shared_tariff_reward/mean_per_step", st[0] / st[1])
        put("diagnostic/info_missing_key_count", self._missing)
        put("diagnostic/nonfinite_metric_count", self._nonfinite)
        put("diagnostic/step_count", self._steps)
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
