"""Tests for cadence-aware SustainDC facility metric aggregation."""
import csv
import sys
import tempfile
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests._harness import run_module_tests
from harl.envs.sustaindc.metrics import (
    FacilityMetricAccumulator,
    aggregate_metric_snapshots,
    build_trace_row,
    write_trace_csv,
)
from harl.envs.sustaindc.sustaindc_logger import SustainDCLogger


def _info(**overrides):
    info = {
        "total_price_cost_this_step_c": 0.0,
        "energy_cost_this_step_c": 0.0,
        "demand_charge_increment_c": 0.0,
        "additional_charge_increment_c": 0.0,
        "bat_grid_import_kwh": 0.0,
        "bat_total_energy_without_battery_KWh": 0.0,
        "bat_SOC": 0.5,
        "pv_ac_kwh": 0.0,
        "pv_self_consumed_kwh": 0.0,
        "pv_curtailed_kwh": 0.0,
        "bat_avg_CI": 0.0,
        "bat_CO2_footprint": 0.0,
        "bat_charge_bus_KWh": 0.0,
        "bat_discharge_bus_KWh": 0.0,
        "bat_charge_cell_KWh": 0.0,
        "bat_discharge_cell_KWh": 0.0,
        "bat_cell_throughput_KWh": 0.0,
        "bat_degradation_cost_c": 0.0,
        "bat_a_t": "idle",
        "ls_original_workload": 0.0,
        "ls_shifted_workload": 0.0,
        "ls_computed_tasks": 0.0,
        "ls_tasks_processed": 0.0,
        "ls_tasks_in_queue": 0.0,
        "ls_norm_tasks_in_queue": 0.0,
        "ls_tasks_dropped": 0.0,
        "ls_overdue_penalty": 0.0,
        "ls_oldest_task_age": 0.0,
        "ls_average_task_age": 0.0,
        "ls_enforced": 0.0,
        "dc_ITE_total_power_kW": 1.0,
        "dc_HVAC_total_power_kW": 1.0,
        "dc_CT_total_power_kW": 0.0,
        "dc_Compressor_total_power_kW": 0.0,
        "dc_total_power_kW": 2.0,
        "dc_crac_setpoint": 22.0,
        "dc_int_temperature": 22.0,
        "dc_exterior_ambient_temp": 10.0,
        "dc_thermal_limit_c": 23.0,
        "dc_constraint_violation": 0.0,
        "dc_water_usage": 0.0,
        "shared_tariff_cost_reward": 0.0,
        "agent_ls_reward": 0.0,
        "agent_dc_reward": 0.0,
        "agent_bat_reward": 0.0,
        "ls_overdue_penalty_component": 0.0,
        "ls_dropped_penalty_component": 0.0,
        "ls_backlog_penalty_component": 0.0,
        "dc_thermal_penalty_component": 0.0,
        "dc_constraint_penalty_component": 0.0,
        "bat_terminal_soc_penalty_component": 0.0,
        "bat_degradation_penalty_component": 0.0,
    }
    info.update(overrides)
    return info


def test_facility_cost_is_counted_once_per_environment_step():
    accumulator = FacilityMetricAccumulator("train/rollout")
    accumulator.add(_info(total_price_cost_this_step_c=10.0, bat_grid_import_kwh=2.0))
    accumulator.add(_info(total_price_cost_this_step_c=15.0, bat_grid_import_kwh=3.0))
    snapshot = accumulator.snapshot()
    assert snapshot["train/rollout/tariff/cost/total_c"] == 25.0
    assert snapshot["train/rollout/energy/grid_import_kwh"] == 5.0


def test_accumulator_preserves_targeted_rewards_and_penalties():
    accumulator = FacilityMetricAccumulator("eval/episode", targeted_mode=True)
    accumulator.add(_info(
        agent_ls_reward=-4.0,
        agent_dc_reward=-3.0,
        agent_bat_reward=-2.0,
        shared_tariff_cost_reward=-2.0,
        ls_overdue_penalty_component=1.0,
        dc_thermal_penalty_component=1.0,
    ))
    snapshot = accumulator.snapshot()
    assert snapshot["eval/episode/reward/agent_ls/sum"] == -4.0
    assert snapshot["eval/episode/reward/agent_dc/sum"] == -3.0
    assert snapshot["eval/episode/reward/agent_bat/sum"] == -2.0
    assert snapshot["eval/episode/reward/mean_across_agents/sum"] == -3.0
    assert snapshot["eval/episode/reward/penalty/ls_overdue/sum"] == 1.0


def test_accumulator_tracks_state_queue_actions_and_thermal_exceedance():
    accumulator = FacilityMetricAccumulator("train/rollout")
    accumulator.add(_info(
        bat_SOC=0.8, bat_a_t="charge", ls_tasks_in_queue=2.0,
        dc_int_temperature=25.0, dc_thermal_limit_c=23.0,
        dc_constraint_violation=1.0,
    ))
    accumulator.add(_info(
        bat_SOC=0.4, bat_a_t="discharge", ls_tasks_in_queue=4.0,
        dc_int_temperature=22.0, dc_thermal_limit_c=23.0,
    ))
    snapshot = accumulator.snapshot()
    assert snapshot["train/rollout/battery/soc/start"] == 0.8
    assert snapshot["train/rollout/battery/soc/end"] == 0.4
    assert snapshot["train/rollout/battery/soc/min"] == 0.4
    assert snapshot["train/rollout/battery/soc/max"] == 0.8
    assert abs(snapshot["train/rollout/battery/soc/mean"] - 0.6) < 1e-12
    assert snapshot["train/rollout/sla/queue_depth/mean"] == 3.0
    assert snapshot["train/rollout/sla/queue_depth/max"] == 4.0
    assert snapshot["train/rollout/sla/queue_depth/final"] == 4.0
    assert snapshot["train/rollout/sla/queue_task_steps"] == 6.0
    assert snapshot["train/rollout/battery/action/charge_steps"] == 1.0
    assert snapshot["train/rollout/battery/action/discharge_steps"] == 1.0
    assert snapshot["train/rollout/thermal/exceedance_degree_c_steps"] == 2.0
    assert snapshot["train/rollout/thermal/constraint_violation_steps"] == 1.0


def test_zero_denominator_ratios_are_omitted_and_nonfinite_is_diagnostic():
    accumulator = FacilityMetricAccumulator("train/rollout")
    accumulator.add(_info(
        dc_ITE_total_power_kW=0.0,
        dc_HVAC_total_power_kW=0.0,
        bat_grid_import_kwh=0.0,
    ))
    accumulator.add(_info(
        pv_ac_kwh=float("nan"),
        dc_ITE_total_power_kW=0.0,
        dc_HVAC_total_power_kW=0.0,
        bat_grid_import_kwh=0.0,
    ))
    snapshot = accumulator.snapshot()
    assert "train/rollout/pv/self_consumption_fraction" not in snapshot
    assert "train/rollout/efficiency/pue" not in snapshot
    assert snapshot["train/rollout/diagnostic/nonfinite_metric_count"] == 1.0


class FakeMetricWriter:
    def __init__(self):
        self.scalars = []
        self.metric_rows = []

    def add_scalar(self, tag, value, global_step=None):
        self.scalars.append((tag, value, global_step))

    def log_metrics(self, metrics, global_step=None):
        self.metric_rows.append((dict(metrics), global_step))


def _logger_with_fake_writer():
    return SustainDCLogger(
        {"env": "sustaindc", "algo": "happo", "exp_name": "test"},
        {
            "train": {
                "n_rollout_threads": 1,
                "episode_length": 1,
                "num_env_steps": 1,
            },
            "eval": {"n_eval_rollout_threads": 1},
        },
        {"location": "ca", "reward": {"mode": "team_tariff_targeted_safeguards"}},
        3,
        FakeMetricWriter(),
        ".",
    )


def _training_data(rewards, infos, dones):
    return (None, None, rewards, dones, infos, None, None, None, None, None, None)


class FakeCriticBuffer:
    def get_mean_rewards(self):
        return 0.0


def test_rollout_logging_keeps_targeted_returns_separate():
    logger = _logger_with_fake_writer()
    logger.init(episodes=1)
    logger.episode_init(1)
    logger.per_step(_training_data(
        rewards=np.array([[[-4.0], [-3.0], [-2.0]]], dtype=np.float32),
        infos=[[_info(agent_ls_reward=-4.0, agent_dc_reward=-3.0, agent_bat_reward=-2.0)] * 3],
        dones=np.zeros((1, 3, 1), dtype=bool),
    ))
    logger.episode_log([{}, {}, {}], {}, None, FakeCriticBuffer())
    logged = logger.writter.metric_rows[-1][0]
    assert logged["train/rollout/reward/agent_ls/sum"] == -4.0
    assert logged["train/rollout/reward/agent_dc/sum"] == -3.0
    assert logged["train/rollout/reward/agent_bat/sum"] == -2.0
    assert "train/episode/reward/agent_ls/sum" not in logged


def test_trace_row_retains_billing_and_next_state_clocks():
    row = build_trace_row(
        info=_info(
            billed_day=333,
            billed_hour=23.75,
            next_state_day=334,
            next_state_hour=0.0,
            total_price_cost_this_step_c=7.5,
            shared_tariff_cost_reward=-0.075,
        ),
        actions=np.array([[1], [2], [0]]),
        run_metadata={
            "seed": 1,
            "phase": "eval",
            "physical_episode_id": 2,
            "env_thread": 0,
            "step_in_episode": 95,
        },
        terminal=True,
        truncated=True,
    )
    assert row["billed_day"] == 333
    assert row["billed_hour"] == 23.75
    assert row["next_state_day"] == 334
    assert row["next_state_hour"] == 0.0
    assert row["reward/shared_tariff_cost_reward"] == -0.075
    assert row["action/ls"] == 1
    assert row["action/dc"] == 2
    assert row["action/bat"] == 0
    assert row["terminal"] is True


def test_trace_csv_retains_optional_columns_and_float_precision():
    rows = [
        {"env_thread": 0, "physical_episode_id": 1, "tariff/cost": 1.123456789},
        {"env_thread": 1, "physical_episode_id": 2, "optional": "value"},
    ]
    with tempfile.TemporaryDirectory() as directory:
        path = write_trace_csv(rows, Path(directory) / "trace.csv")
        with path.open(newline="", encoding="utf-8") as handle:
            result = list(csv.DictReader(handle))
    assert result[0]["tariff/cost"] == "1.123456789"
    assert result[1]["optional"] == "value"
    assert result[0]["env_thread"] == "0"
    assert result[1]["physical_episode_id"] == "2"


def test_evaluation_logs_completed_episode_and_aggregate_metrics():
    logger = _logger_with_fake_writer()
    logger.episode = 1
    logger.eval_init()
    logger.eval_per_step((
        None, None, np.array([[[-4.0], [-3.0], [-2.0]]]), None,
        [[_info(total_price_cost_this_step_c=10.0)] * 3], None,
    ))
    logger.one_episode_rewards[0].append(np.array([[-4.0], [-3.0], [-2.0]]))
    logger.eval_thread_done(0)
    logger.eval_log(1)
    episode_rows = [
        row for row, _ in logger.writter.metric_rows
        if "eval/episode/tariff/cost/total_c" in row
    ]
    aggregate_rows = [
        row for row, _ in logger.writter.metric_rows
        if "eval/aggregate/tariff/cost/total_c/mean" in row
    ]
    assert len(episode_rows) == 1
    assert episode_rows[0]["eval/episode/tariff/cost/total_c"] == 10.0
    assert len(aggregate_rows) == 1
    assert aggregate_rows[0]["eval/aggregate/tariff/cost/total_c/mean"] == 10.0


def test_aggregate_snapshots_reports_mean_standard_deviation_minimum_and_maximum():
    aggregates = aggregate_metric_snapshots(
        [
            {"eval/episode/tariff/cost/total_c": 10.0},
            {"eval/episode/tariff/cost/total_c": 30.0},
        ],
        "eval/aggregate",
    )
    prefix = "eval/aggregate/tariff/cost/total_c"
    assert aggregates[f"{prefix}/mean"] == 20.0
    assert aggregates[f"{prefix}/std"] == 10.0
    assert aggregates[f"{prefix}/min"] == 10.0
    assert aggregates[f"{prefix}/max"] == 30.0


if __name__ == "__main__":
    sys.exit(1 if run_module_tests(globals()) else 0)
