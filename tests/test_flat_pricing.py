from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tests._harness import run_module_tests
from utils.managers.pricing.contracts import (
    MeterReading,
    PricingCharges,
    PricingClock,
    PricingContext,
    PricingState,
)
from utils.managers.pricing.loader import PricingConfigError, create_pricing_model, load_pricing_config
from utils.managers.pricing.models.flat import FlatPricingModel


def _context(future_steps=8):
    return PricingContext(datacenter_capacity_mw=12.5, timestep_minutes=15,
                          future_steps=future_steps)


def _clock(hour=13.5):
    return PricingClock(day_of_year=100, hour=hour)


def test_flat_charges_and_observation_use_configured_rate():
    model = FlatPricingModel({"schema_version": 1, "model": "flat",
                              "energy_price_c_per_kwh": 5.0})
    charges = model.step(context=_context(), state=PricingState(), clock=_clock(),
                         reading=MeterReading(energy_kwh=25.0, demand_kw=100.0))

    assert charges == PricingCharges(energy_cost_c=125.0,
                                     demand_cost_increment_c=0.0,
                                     additional_cost_increment_c=0.0)
    observation = model.observe(context=_context(), state=PricingState(),
                                clock=_clock())
    assert observation.forecast_normalized.shape == (8,)
    assert np.array_equal(observation.forecast_normalized, np.ones(8, dtype=np.float32))
    assert observation.current_price_c_per_kwh == 5.0
    assert observation.normalized_price == 1.0
    assert observation.billing_progress == 0.0


def test_zero_flat_rate_has_zero_normalized_outputs():
    model = FlatPricingModel({"schema_version": 1, "model": "flat",
                              "energy_price_c_per_kwh": 0.0})
    observation = model.observe(context=_context(3), state=PricingState(),
                                clock=_clock())

    assert observation.normalized_price == 0.0
    assert np.array_equal(observation.forecast_normalized, np.zeros(3, dtype=np.float32))


def test_flat_loader_rejects_negative_and_nonfinite_rates():
    for invalid_rate in (-1.0, math.nan, math.inf):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parameter_file = root / "flat.yaml"
            parameter_file.write_text(yaml.safe_dump({
                "schema_version": 1,
                "model": "flat",
                "energy_price_c_per_kwh": invalid_rate,
            }), encoding="utf-8")
            try:
                load_pricing_config({"pricing": {
                    "model": "flat", "config_file": "flat.yaml",
                }}, root)
            except PricingConfigError as exc:
                assert "energy_price_c_per_kwh" in str(exc)
            else:
                raise AssertionError("expected invalid flat rate to be rejected")


def test_flat_parameter_schema_rejects_unknown_fields():
    try:
        FlatPricingModel({"schema_version": 1, "model": "flat",
                          "energy_price_c_per_kwh": 5.0, "unexpected": 1})
    except ValueError as exc:
        assert "unexpected" in str(exc)
    else:
        raise AssertionError("expected unknown flat parameter to be rejected")


def test_flat_rejects_invalid_meter_readings():
    model = FlatPricingModel({"schema_version": 1, "model": "flat",
                              "energy_price_c_per_kwh": 5.0})
    for reading in (MeterReading(energy_kwh=-1.0, demand_kw=1.0),
                    MeterReading(energy_kwh=1.0, demand_kw=float("nan"))):
        try:
            model.step(context=_context(), state=PricingState(), clock=_clock(),
                       reading=reading)
        except ValueError as exc:
            assert "reading." in str(exc)
        else:
            raise AssertionError("expected invalid meter reading to be rejected")


if __name__ == "__main__":
    sys.exit(run_module_tests(globals()))
