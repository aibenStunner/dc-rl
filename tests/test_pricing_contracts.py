from __future__ import annotations

import inspect
import sys
from dataclasses import fields
from pathlib import Path
from typing import Mapping, get_type_hints

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tests._harness import run_module_tests
from utils.pricing.contracts import (
    MeterReading, PricingCharges, PricingClock, PricingContext,
    PricingModel, PricingObservation, PricingState,
)


def test_context_fields_and_values_are_pinned():
    context = PricingContext(datacenter_capacity_mw=12.5,
                             timestep_minutes=15,
                             future_steps=8)
    assert [field.name for field in fields(PricingContext)] == [
        "datacenter_capacity_mw", "timestep_minutes", "future_steps"
    ]
    assert context.datacenter_capacity_mw == 12.5
    assert context.timestep_minutes == 15
    assert context.future_steps == 8


def test_clock_fields_and_values_are_pinned():
    clock = PricingClock(day_of_year=274, hour=13.5)
    assert [field.name for field in fields(PricingClock)] == [
        "day_of_year", "hour"
    ]
    assert clock.day_of_year == 274
    assert clock.hour == 13.5


def test_meter_reading_fields_and_values_are_pinned():
    reading = MeterReading(energy_kwh=42.25, demand_kw=7.5)
    assert [field.name for field in fields(MeterReading)] == [
        "energy_kwh", "demand_kw"
    ]
    assert reading.energy_kwh == 42.25
    assert reading.demand_kw == 7.5


def test_state_fields_and_defaults_are_pinned():
    state = PricingState()
    assert [field.name for field in fields(PricingState)] == [
        "billing_period_energy_kwh", "running_peak_kw", "model_state"
    ]
    assert state.billing_period_energy_kwh == 0.0
    assert state.running_peak_kw == 0.0
    assert state.model_state == {}
    assert state.model_state is not PricingState().model_state

    custom = PricingState(billing_period_energy_kwh=3.0,
                          running_peak_kw=4.0,
                          model_state={"period": 2})
    assert custom.billing_period_energy_kwh == 3.0
    assert custom.running_peak_kw == 4.0
    assert custom.model_state == {"period": 2}


def test_observation_fields_and_values_are_pinned():
    forecast = np.array([0.5, 0.75], dtype=np.float32)
    observation = PricingObservation(current_price_c_per_kwh=5.0,
                                     normalized_price=0.5,
                                     forecast_normalized=forecast,
                                     billing_progress=0.25,
                                     normalized_peak=0.4)
    assert [field.name for field in fields(PricingObservation)] == [
        "current_price_c_per_kwh", "normalized_price", "forecast_normalized",
        "billing_progress", "normalized_peak"
    ]
    assert observation.current_price_c_per_kwh == 5.0
    assert observation.normalized_price == 0.5
    assert np.array_equal(observation.forecast_normalized, forecast)
    assert observation.billing_progress == 0.25
    assert observation.normalized_peak == 0.4


def test_charges_fields_and_values_are_pinned():
    charges = PricingCharges(energy_cost_c=10.0,
                             demand_cost_increment_c=3.0,
                             additional_cost_increment_c=2.0)
    assert [field.name for field in fields(PricingCharges)] == [
        "energy_cost_c", "demand_cost_increment_c",
        "additional_cost_increment_c"
    ]
    assert charges.energy_cost_c == 10.0
    assert charges.demand_cost_increment_c == 3.0
    assert charges.additional_cost_increment_c == 2.0


def test_charges_total_is_sum_of_standard_categories():
    charges = PricingCharges(energy_cost_c=10.0,
                             demand_cost_increment_c=3.0,
                             additional_cost_increment_c=2.0)
    assert charges.total_cost_c == 15.0


def test_observation_requires_exact_forecast_shape_at_manager_boundary():
    obs = PricingObservation(current_price_c_per_kwh=5.0,
                             normalized_price=0.5,
                             forecast_normalized=np.array([0.5, 0.5]),
                             billing_progress=0.25,
                             normalized_peak=0.4)
    assert obs.forecast_normalized.shape == (2,)


def test_protocol_accepts_structural_implementation_with_exact_signatures():
    class MinimalModel:
        name = "minimal"

        def reset(self, *, context: PricingContext, state: PricingState,
                  clock: PricingClock,
                  carry_state: Mapping[str, object] | None,
                  options: Mapping[str, object]) -> None:
            pass

        def step(self, *, context: PricingContext, state: PricingState,
                 clock: PricingClock,
                 reading: MeterReading) -> PricingCharges:
            return PricingCharges(0.0, 0.0, 0.0)

        def observe(self, *, context: PricingContext, state: PricingState,
                    clock: PricingClock) -> PricingObservation:
            return PricingObservation(
                0.0, 0.0, np.zeros(1, dtype=np.float32), 0.0, 0.0)

        def export_carry_state(
            self, state: PricingState,
        ) -> Mapping[str, object]:
            return {}

    expected_annotations = {
        "reset": {
            "context": PricingContext,
            "state": PricingState,
            "clock": PricingClock,
            "carry_state": Mapping[str, object] | None,
            "options": Mapping[str, object],
            "return": type(None),
        },
        "step": {
            "context": PricingContext,
            "state": PricingState,
            "clock": PricingClock,
            "reading": MeterReading,
            "return": PricingCharges,
        },
        "observe": {
            "context": PricingContext,
            "state": PricingState,
            "clock": PricingClock,
            "return": PricingObservation,
        },
        "export_carry_state": {
            "state": PricingState,
            "return": Mapping[str, object],
        },
    }
    expected_parameters = {
        "reset": ["self", "context", "state", "clock", "carry_state", "options"],
        "step": ["self", "context", "state", "clock", "reading"],
        "observe": ["self", "context", "state", "clock"],
        "export_carry_state": ["self", "state"],
    }
    for method_name, parameter_names in expected_parameters.items():
        method = getattr(MinimalModel, method_name)
        signature = inspect.signature(method)
        assert list(signature.parameters) == parameter_names
        if method_name == "export_carry_state":
            assert signature.parameters["state"].kind is (
                inspect.Parameter.POSITIONAL_OR_KEYWORD
            )
        else:
            assert all(parameter.kind is inspect.Parameter.KEYWORD_ONLY
                       for name, parameter in signature.parameters.items()
                       if name != "self")

        annotations = get_type_hints(method)
        expected = expected_annotations[method_name]
        assert {name: annotations[name] for name in expected if name != "return"} == {
            name: annotation for name, annotation in expected.items()
            if name != "return"
        }
        assert annotations["return"] is expected["return"]

    assert isinstance(MinimalModel(), PricingModel)


def test_observation_forecast_is_copy_safe_and_intrinsically_immutable():
    source = np.array([0.5, 0.5])
    obs = PricingObservation(current_price_c_per_kwh=5.0,
                             normalized_price=0.5,
                             forecast_normalized=source,
                             billing_progress=0.25,
                             normalized_peak=0.4)

    source[0] = 1.0
    assert obs.forecast_normalized[0] == 0.5

    try:
        obs.forecast_normalized[0] = 1.0
    except ValueError:
        pass
    else:
        raise AssertionError("forecast_normalized must be immutable")

    try:
        obs.forecast_normalized.setflags(write=True)
    except ValueError:
        pass
    else:
        raise AssertionError("forecast_normalized must not re-enable writes")


if __name__ == "__main__":
    sys.exit(1 if run_module_tests(globals()) else 0)
