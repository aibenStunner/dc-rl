"""Regression tests for utilities shared by the manager package."""
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tests._harness import run_module_tests
from utils.managers.common import CoherentNoise, _cyc, _cyc_at, normalize
from utils.managers.time_manager import sc_obs


def test_package_reexports_public_manager_symbols_and_helpers():
    from utils.managers import (
        CI_Manager,
        CoherentNoise as ExportedNoise,
        Time_Manager,
        Weather_Manager,
        Workload_Manager,
        _cyc as exported_cyc,
        _cyc_at as exported_cyc_at,
        normalize as exported_normalize,
        sc_obs as exported_sc_obs,
    )
    from utils.managers.carbon_intensity_manager import CI_Manager as CarbonClass
    from utils.managers.time_manager import Time_Manager as TimeClass
    from utils.managers.weather_manager import Weather_Manager as WeatherClass
    from utils.managers.workload_manager import Workload_Manager as WorkloadClass

    assert CI_Manager is CarbonClass
    assert Time_Manager is TimeClass
    assert Weather_Manager is WeatherClass
    assert Workload_Manager is WorkloadClass
    assert ExportedNoise is CoherentNoise
    assert exported_cyc is _cyc
    assert exported_cyc_at is _cyc_at
    assert exported_normalize is normalize
    assert exported_sc_obs is sc_obs


def test_normalize_is_unclipped_affine_for_scalars_and_arrays():
    assert normalize(15.0, 10.0, 20.0) == 0.5
    assert np.array_equal(
        normalize(np.array([0.0, 10.0, 20.0]), 0.0, 10.0),
        np.array([0.0, 1.0, 2.0]),
    )


def test_cyclic_helpers_wrap_forward_and_backward():
    values = np.array([10, 20, 30])
    assert np.array_equal(_cyc(values, 2, 4), np.array([30, 10, 20, 30]))
    assert np.array_equal(_cyc(values, -2, 3), np.array([20, 30, 10]))
    assert _cyc_at(values, 3) == 10
    assert _cyc_at(values, -1) == 30


def test_sc_obs_uses_rounded_cyclic_phases_in_unit_interval():
    assert np.allclose(sc_obs(0, 0), [1.0, 0.5, 1.0, 0.5])
    assert np.allclose(sc_obs(6, 0)[:2], [0.5, 1.0])
    assert np.allclose(sc_obs(0, 365)[2:], [1.0, 0.5])

    hour, day = 1, 1
    expected = np.array([
        np.cos(round(hour / 24, 3) * 2 * np.pi) * 0.5 + 0.5,
        np.sin(round(hour / 24, 3) * 2 * np.pi) * 0.5 + 0.5,
        np.cos(round(day / 365, 3) * 2 * np.pi) * 0.5 + 0.5,
        np.sin(round(day / 365, 3) * 2 * np.pi) * 0.5 + 0.5,
    ])
    actual = np.array(sc_obs(hour, day))
    assert np.allclose(actual, expected)
    assert np.all((actual >= 0.0) & (actual <= 1.0))


def test_coherent_noise_is_seeded_and_has_requested_standard_deviation():
    baseline = np.zeros(32)
    rng_state = np.random.get_state()
    try:
        np.random.seed(123)
        first = CoherentNoise(baseline, weight=0.2, desired_std_dev=0.7).generate(32)
        np.random.seed(123)
        second = CoherentNoise(baseline, weight=0.2, desired_std_dev=0.7).generate(32)
    finally:
        np.random.set_state(rng_state)

    assert first.shape == baseline.shape
    assert np.array_equal(baseline, np.zeros(32))
    assert np.allclose(first, second)
    assert np.isclose(np.std(first - baseline), 0.7)


if __name__ == "__main__":
    sys.exit(1 if run_module_tests(globals()) else 0)
