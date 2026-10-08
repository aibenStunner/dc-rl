"""Behavioral tests for the simulation clock."""
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tests._harness import run_module_tests
from utils.managers.time_manager import Time_Manager, sc_obs


def test_reset_uses_timezone_hour_only_when_explicit_hour_is_omitted():
    manager = Time_Manager(init_day=10, days_per_episode=2, timezone_shift=-5)
    assert np.allclose(manager.reset(), sc_obs(-5, 10))
    assert manager.day == 10
    assert manager.hour == -5
    assert manager.current_timestep == 10 * 96 - 5 * 4
    assert manager.total_timesteps == manager.current_timestep + 2 * 96

    assert np.allclose(manager.reset(init_day=12, init_hour=3), sc_obs(3, 12))
    assert manager.day == 12
    assert manager.hour == 3


def test_step_advances_a_quarter_hour_and_crosses_midnight():
    manager = Time_Manager(init_day=333, days_per_episode=2)
    manager.reset(init_day=333, init_hour=23.75)
    day, hour, observation, terminal = manager.step()
    assert day == 334
    assert hour == 0.0
    assert np.allclose(observation, sc_obs(0.0, 334))
    assert terminal is False


def test_terminal_is_true_on_exact_episode_step_count():
    manager = Time_Manager(init_day=10, days_per_episode=1)
    manager.reset(init_day=10, init_hour=0)
    for _ in range(95):
        assert manager.step()[-1] is False
    assert manager.step()[-1] is True
    assert manager.current_timestep == manager.total_timesteps


if __name__ == "__main__":
    sys.exit(1 if run_module_tests(globals()) else 0)
