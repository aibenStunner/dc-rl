"""Task-flow and queue-service regressions for the load-shifting environment."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from envs.carbon_ls import CarbonLoadEnv
from sustaindc_env import SustainDC
from tests._harness import run_module_tests
from tests.test_observation_layout import _ENV_CONFIG


def test_task_ledger_conserves_work_across_deferral_overflow_and_deadlines():
    env = CarbonLoadEnv(flexible_workload_ratio=0.6, queue_max_len=10)
    expected_queue = 0
    for step, (day, hour, workload, action, forced) in enumerate((
        (1, 0, 0.5, 0, 20), (1, 0.25, 0.5, 0, 30),
        (1, 0.5, 0.0, 1, 0), (1, 0.75, 0.0, 2, 0),
        (2, 1.0, 0.5, 0, 20), (3, 2.0, 0.0, 1, 0),
    )):
        env.update_current_date(day, hour)
        env.update_workload(workload)
        _, _, _, _, info = env.step(action)
        arriving = round(workload * 100)
        executed = info['ls_computed_tasks']
        expected_queue += arriving - executed
        assert info['ls_tasks_in_queue'] == expected_queue, (step, info)
        assert 0 <= executed <= 90
        assert info['ls_tasks_dropped'] == forced
        if step == 4:
            assert info['ls_overdue_penalty'] == 0
        if step == 5:
            assert info['ls_overdue_penalty'] == 10
            assert info['ls_tasks_in_queue'] == 0


def test_varied_actions_conserve_the_task_ledger():
    env = CarbonLoadEnv(flexible_workload_ratio=0.6, queue_max_len=35)
    expected_queue = 0
    for step in range(144):
        env.update_current_date(1 + step // 96, (step % 96) / 4)
        workload = (0.0, 0.23, 0.5, 0.83, 1.0)[step % 5]
        env.update_workload(workload)
        action = (0, 1, 2, 0, 2, 1, 2)[step % 7]
        _, _, _, _, info = env.step(action)
        arrivals = math.ceil(workload * 0.4 * 100) + math.floor(workload * 0.6 * 100)
        expected_queue += arrivals - info['ls_computed_tasks']
        assert expected_queue == info['ls_tasks_in_queue'], (step, info)
        assert 0 <= info['ls_tasks_in_queue'] <= 35
        assert 0 <= info['ls_shifted_workload'] <= 1


def test_joint_step_passes_serviced_queue_load_to_dc_and_terminal_reward():
    config = dict(_ENV_CONFIG, days_per_episode=1,
                  episode_start_policy='valid_midnight',
                  reward={'mode': 'team_tariff_targeted_safeguards',
                          'ls_terminal_backlog_penalty_weight': 0.5})
    env = SustainDC(config)
    env.reset(seed=102, options={'init_day': 14, 'init_hour': 0})
    _, _, _, _, first = env.step({'agent_ls': 0, 'agent_dc': 1, 'agent_bat': 2})
    assert first['agent_ls']['ls_tasks_in_queue'] > 0
    previous_queue = first['agent_ls']['ls_tasks_in_queue']
    _, _, _, _, second = env.step({'agent_ls': 2, 'agent_dc': 1, 'agent_bat': 2})
    ls = second['agent_ls']
    assert ls['ls_tasks_processed'] > 0
    assert ls['ls_tasks_in_queue'] < previous_queue
    assert second['agent_dc']['dc_total_power_kW'] > 0
    assert env.dc_env.cpu_load_frac == ls['ls_shifted_workload']
    for _ in range(94):
        _, _, _, truncated, final = env.step(
            {'agent_ls': 0, 'agent_dc': 1, 'agent_bat': 2})
    assert truncated['__all__']
    assert final['__common__']['ls_tasks_in_queue'] > 0
    assert final['__common__']['ls_terminal_backlog_penalty_component'] > 0


def test_process_queue_uses_spare_capacity_without_new_shiftable_work():
    env = CarbonLoadEnv(flexible_workload_ratio=0.6, queue_max_len=100)
    env.update_current_date(1, 0)
    env.update_workload(0.5)
    _, _, _, _, deferred = env.step(0)
    assert deferred['ls_tasks_in_queue'] == 30
    assert deferred['ls_shifted_workload'] == 0.2

    env.update_current_date(1, 0.25)
    env.update_workload(0.0)
    _, _, _, _, serviced = env.step(2)
    assert serviced['ls_tasks_processed'] == 30
    assert serviced['ls_tasks_in_queue'] == 0
    assert serviced['ls_shifted_workload'] == 0.3


if __name__ == '__main__':
    sys.exit(1 if run_module_tests(globals()) else 0)
