from __future__ import annotations

import csv
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tests._harness import run_module_tests
from utils.pricing.contracts import (
    MeterReading,
    PricingClock,
    PricingContext,
    PricingState,
)
from utils.pricing.models.time_series import TimeSeriesPricingModel


def _context(future_steps=8, timestep_minutes=15):
    return PricingContext(datacenter_capacity_mw=12.5,
                          timestep_minutes=timestep_minutes,
                          future_steps=future_steps)


def _clock(day=0, hour=0.0):
    return PricingClock(day_of_year=day, hour=hour)


def _write_csv(path: Path, values, column="price_c_per_kwh"):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[column])
        writer.writeheader()
        for value in values:
            writer.writerow({column: value})


def _model(path, **overrides):
    parameters = {
        "schema_version": 1,
        "model": "time_series",
        "csv_file": str(path),
        "price_column": "price_c_per_kwh",
        "source_interval_minutes": 60,
        "wrap": True,
    }
    parameters.update(overrides)
    return TimeSeriesPricingModel(parameters)


def test_named_column_and_zero_order_hold_expand_hourly_input():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "prices.csv"
        _write_csv(path, [1.0, 2.0, 4.0])
        model = _model(path)
        observation = model.observe(
            context=_context(future_steps=5, timestep_minutes=15),
            state=PricingState(),
            clock=_clock(day=0, hour=0.0),
        )

    assert observation.current_price_c_per_kwh == 1.0
    # Forecast slot k is the price in the next simulation interval.
    # The hourly source is held for four 15-minute slots, so slots one
    # through three remain at source hour 0 before slot four enters hour 1.
    assert np.allclose(observation.forecast_normalized,
                       [0.25, 0.25, 0.25, 0.5, 0.5])
    assert observation.forecast_normalized.shape == (5,)
    charges = model.step(context=_context(), state=PricingState(), clock=_clock(),
                         reading=MeterReading(energy_kwh=2.0, demand_kw=5.0))
    assert charges.energy_cost_c == 2.0


def test_missing_column_is_named_in_error():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "prices.csv"
        _write_csv(path, [1.0], column="other")
        try:
            _model(path)
        except ValueError as exc:
            assert "price_c_per_kwh" in str(exc)
        else:
            raise AssertionError("expected missing price column to be rejected")


def test_nan_and_negative_values_are_rejected():
    for values in ([math.nan, 1.0], [-1.0, 1.0]):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prices.csv"
            _write_csv(path, values)
            try:
                _model(path)
            except ValueError as exc:
                assert "price" in str(exc).lower()
            else:
                raise AssertionError("expected invalid time-series prices to fail")


def test_forecast_wraps_at_end_when_enabled():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "prices.csv"
        _write_csv(path, [2.0, 4.0])
        model = _model(path, wrap=True)
        observation = model.observe(
            context=_context(future_steps=3, timestep_minutes=60),
            state=PricingState(),
            clock=_clock(day=0, hour=1.0),
        )

    # Forecasting starts one simulation step after the current hour.  At
    # hour 1, the first next-hour slot is source value 4, then wrapping
    # returns to source value 2 for the following two slots.
    assert np.allclose(observation.forecast_normalized, [0.5, 1.0, 0.5])


def test_source_index_advances_across_episode_days():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "prices.csv"
        _write_csv(path, [float(index) for index in range(48)])
        model = _model(path, wrap=False)
        context = _context(future_steps=1, timestep_minutes=60)
        day_zero = model.observe(context=context, state=PricingState(),
                                 clock=_clock(day=0, hour=1.0))
        day_one = model.observe(context=context, state=PricingState(),
                                clock=_clock(day=1, hour=1.0))

    # The source index is based on absolute episode time, not just hour of day:
    # day 0 hour 1 selects source row 1, while day 1 hour 1 selects row 25.
    assert day_zero.current_price_c_per_kwh == 1.0
    assert day_one.current_price_c_per_kwh == 25.0


def test_forecast_without_wrap_fails_before_reading_past_data():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "prices.csv"
        _write_csv(path, [2.0, 4.0])
        model = _model(path, wrap=False)
        try:
            model.observe(context=_context(future_steps=2, timestep_minutes=60),
                          state=PricingState(), clock=_clock(day=0, hour=1.0))
        except ValueError as exc:
            assert "wrap" in str(exc).lower() or "forecast" in str(exc).lower()
        else:
            raise AssertionError("expected non-wrapping forecast to fail")


def test_non_integer_source_to_simulation_ratio_is_rejected():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "prices.csv"
        _write_csv(path, [2.0, 4.0])
        model = _model(path, source_interval_minutes=50)
        try:
            model.observe(context=_context(future_steps=1, timestep_minutes=15),
                          state=PricingState(), clock=_clock(day=0, hour=0.0))
        except ValueError as exc:
            assert "integer" in str(exc).lower()
        else:
            raise AssertionError("expected non-integer interval ratio to fail")


if __name__ == "__main__":
    sys.exit(run_module_tests(globals()))
