from __future__ import annotations

import math
import sys
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
from utils.pricing.models.time_of_use import TimeOfUsePricingModel


HOURLY_PRICES = [float(index + 1) for index in range(24)]


def _context(future_steps=8, timestep_minutes=15):
    return PricingContext(datacenter_capacity_mw=12.5,
                          timestep_minutes=timestep_minutes,
                          future_steps=future_steps)


def _clock(hour=13.5):
    return PricingClock(day_of_year=100, hour=hour)


def test_tou_selects_fractional_hour_and_charges_current_price():
    model = TimeOfUsePricingModel({
        "schema_version": 1,
        "model": "time_of_use",
        "hourly_prices_c_per_kwh": HOURLY_PRICES,
    })
    charges = model.step(context=_context(), state=PricingState(), clock=_clock(2.75),
                         reading=MeterReading(energy_kwh=2.0, demand_kw=4.0))

    assert charges.energy_cost_c == 6.0
    assert charges.demand_cost_increment_c == 0.0


def test_tou_forecast_wraps_at_midnight_and_is_normalized():
    model = TimeOfUsePricingModel({
        "schema_version": 1,
        "model": "time_of_use",
        "hourly_prices_c_per_kwh": HOURLY_PRICES,
    })
    observation = model.observe(
        context=_context(future_steps=4, timestep_minutes=15),
        state=PricingState(),
        clock=_clock(23.75),
    )

    assert observation.current_price_c_per_kwh == 24.0
    assert observation.forecast_normalized.shape == (4,)
    assert np.all((observation.forecast_normalized >= 0.0) &
                  (observation.forecast_normalized <= 1.0))
    assert np.array_equal(observation.forecast_normalized,
                          np.full(4, 1.0 / 24.0, dtype=np.float32))
    assert math.isclose(observation.billing_progress, 23.75 / 24.0)


def test_tou_requires_exactly_24_finite_nonnegative_prices():
    for prices in ([1.0] * 23, [1.0] * 24 + [1.0], [1.0] * 23 + [-1.0],
                   [1.0] * 23 + [math.nan]):
        try:
            TimeOfUsePricingModel({
                "schema_version": 1,
                "model": "time_of_use",
                "hourly_prices_c_per_kwh": prices,
            })
        except ValueError as exc:
            assert "hourly_prices_c_per_kwh" in str(exc)
        else:
            raise AssertionError("expected invalid TOU schedule to be rejected")


if __name__ == "__main__":
    sys.exit(run_module_tests(globals()))
