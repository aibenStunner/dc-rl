from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any

import numpy as np

from .contracts import (
    JSONLikeValue,
    MeterReading,
    PricingCharges,
    PricingClock,
    PricingContext,
    PricingObservation,
    PricingState,
)
from .loader import PricingConfig, create_pricing_model


class PriceManager:
    """Orchestrate a configured pricing model and its shared billing state."""

    def __init__(
        self,
        *,
        pricing_config: PricingConfig,
        model: Any,
        context: PricingContext,
    ) -> None:
        self.pricing_config = pricing_config
        self.model = model
        self.context = context
        self.options = dict(pricing_config.options)
        self.state = PricingState()
        self._clock: PricingClock | None = None
        self._observation: PricingObservation | None = None
        self._charges = PricingCharges(0.0, 0.0, 0.0)

    @classmethod
    def from_config(
        cls,
        pricing_config: PricingConfig,
        *,
        datacenter_capacity_mw: float,
        future_steps: int,
        timestep_minutes: int = 15,
    ) -> "PriceManager":
        context = PricingContext(
            datacenter_capacity_mw=_finite_nonnegative(
                datacenter_capacity_mw, "datacenter_capacity_mw"
            ),
            timestep_minutes=_positive_divisor(timestep_minutes),
            future_steps=_nonnegative_integer(future_steps, "future_steps"),
        )
        model = create_pricing_model(pricing_config)
        return cls(pricing_config=pricing_config, model=model, context=context)

    def reset(
        self,
        *,
        init_day: int,
        init_hour: float,
        carry_state: Mapping[str, JSONLikeValue] | None = None,
    ) -> PricingObservation:
        clock = _clock(init_day, init_hour)
        # A reset starts an independent sampled episode by default. Callers
        # running one chronological billing timeline must explicitly pass the
        # previous `export_carry_state()` result; implicitly carrying a peak
        # between randomly dated episodes would suppress later demand charges.
        _validate_carry_state(carry_state)
        state = PricingState()
        self.model.reset(
            context=self.context,
            state=state,
            clock=clock,
            carry_state=carry_state,
            options=self.options,
        )
        _validate_model_state(state)
        observation = self._validate_observation(
            self.model.observe(context=self.context, state=state, clock=clock)
        )
        _validate_model_state(state)
        self.state = state
        self._clock = clock
        self._observation = observation
        self._charges = PricingCharges(0.0, 0.0, 0.0)
        return observation

    def step(
        self,
        *,
        metered_energy_kwh: float,
        day_of_year: int,
        hour: float,
    ) -> PricingCharges:
        self._require_reset()
        energy_kwh = _finite_nonnegative(metered_energy_kwh, "metered_energy_kwh")
        clock = _clock(day_of_year, hour)
        timestep_hours = self.context.timestep_minutes / 60.0
        reading = MeterReading(
            energy_kwh=energy_kwh,
            demand_kw=energy_kwh / timestep_hours,
        )
        state_before = _snapshot_state(self.state)
        try:
            charges_result = self.model.step(
                context=self.context,
                state=self.state,
                clock=clock,
                reading=reading,
            )
            _validate_model_state(self.state)
            charges = self._validate_charges(charges_result)
            billing_energy_before = state_before.billing_period_energy_kwh
            # Billing-period energy is common manager state.  Assign from the
            # pre-step value so a model cannot double-count it accidentally.
            self.state.billing_period_energy_kwh = billing_energy_before + energy_kwh
            observation = self._validate_observation(
                self.model.observe(context=self.context, state=self.state, clock=clock)
            )
            _validate_model_state(self.state)
        except Exception:
            _restore_state(self.state, state_before)
            raise
        self._charges = charges
        self._observation = observation
        self._clock = clock
        return charges

    def settle_pending_period(self) -> PricingCharges:
        """Settle an optional model-specific pending billing period charge."""
        self._require_reset()
        state_before = _snapshot_state(self.state)
        charges_before = self._charges
        observation_before = self._observation
        clock_before = self._clock
        assert clock_before is not None
        try:
            settlement_method = getattr(self.model, "settle_pending_period", None)
            if settlement_method is None:
                settlement_method = getattr(self.model, "settle_pending_day", None)
            if settlement_method is None:
                charges = PricingCharges(0.0, 0.0, 0.0)
            else:
                charges_result = settlement_method(state=self.state)
                _validate_model_state(self.state)
                charges = self._validate_charges(charges_result)
            observation = self._validate_observation(
                self.model.observe(
                    context=self.context, state=self.state, clock=clock_before
                )
            )
            _validate_model_state(self.state)
        except Exception:
            _restore_state(self.state, state_before)
            self._charges = charges_before
            self._observation = observation_before
            self._clock = clock_before
            raise
        self._charges = charges
        self._observation = observation
        return charges

    def get_current_price(self) -> float:
        return self._require_observation().normalized_price

    def get_forecast_price(self) -> np.ndarray:
        return np.array(self._require_observation().forecast_normalized, copy=True)

    def get_current_price_denorm(self) -> float:
        return self._require_observation().current_price_c_per_kwh

    def get_billing_progress_fraction(self) -> float:
        return self._require_observation().billing_progress

    def get_normalized_peak(self) -> float:
        return self._require_observation().normalized_peak

    def get_energy_cost_this_step_c(self) -> float:
        return self._charges.energy_cost_c

    def get_demand_charge_increment_c(self) -> float:
        return self._charges.demand_cost_increment_c

    def get_additional_charge_increment_c(self) -> float:
        return self._charges.additional_cost_increment_c

    def get_total_cost_this_step_c(self) -> float:
        return self._charges.total_cost_c

    def export_carry_state(self) -> Mapping[str, JSONLikeValue]:
        self._require_reset()
        state_before = _snapshot_state(self.state)
        try:
            carry_state = self.model.export_carry_state(self.state)
            _validate_carry_state(carry_state)
            return _copy_json_like(dict(carry_state))
        except Exception:
            _restore_state(self.state, state_before)
            raise

    def _require_reset(self) -> None:
        if self._clock is None or self._observation is None:
            raise RuntimeError("price manager must be reset before use")

    def _require_observation(self) -> PricingObservation:
        self._require_reset()
        assert self._observation is not None
        return self._observation

    def _validate_charges(self, charges: object) -> PricingCharges:
        if not isinstance(charges, PricingCharges):
            raise ValueError("pricing model must return PricingCharges")
        for field in (
            "energy_cost_c",
            "demand_cost_increment_c",
            "additional_cost_increment_c",
        ):
            _finite_number(getattr(charges, field), f"charge.{field}")
        _finite_number(charges.total_cost_c, "charge.total_cost_c")
        return charges

    def _validate_observation(self, observation: object) -> PricingObservation:
        if not isinstance(observation, PricingObservation):
            raise ValueError("pricing model must return PricingObservation")
        for field in (
            "current_price_c_per_kwh",
            "normalized_price",
            "billing_progress",
            "normalized_peak",
        ):
            _finite_number(getattr(observation, field), f"observation.{field}")
        if not 0.0 <= observation.normalized_price <= 1.0:
            raise ValueError("observation.normalized_price must be in [0, 1]")
        if not 0.0 <= observation.billing_progress <= 1.0:
            raise ValueError("observation.billing_progress must be in [0, 1]")
        if not 0.0 <= observation.normalized_peak <= 1.0:
            raise ValueError("observation.normalized_peak must be in [0, 1]")
        forecast = np.asarray(observation.forecast_normalized)
        if forecast.shape != (self.context.future_steps,):
            raise ValueError(
                "observation.forecast_normalized has wrong length: "
                f"expected {self.context.future_steps}, got {forecast.size}"
            )
        if not np.all(np.isfinite(forecast)):
            raise ValueError("observation.forecast_normalized must be finite")
        if np.any((forecast < 0.0) | (forecast > 1.0)):
            raise ValueError("observation.forecast_normalized must be in [0, 1]")
        return observation


def _snapshot_state(state: PricingState) -> PricingState:
    _validate_model_state(state)
    return PricingState(
        billing_period_energy_kwh=state.billing_period_energy_kwh,
        running_peak_kw=state.running_peak_kw,
        model_state=_copy_json_like(state.model_state),
    )


def _restore_state(state: PricingState, snapshot: PricingState) -> None:
    state.billing_period_energy_kwh = snapshot.billing_period_energy_kwh
    state.running_peak_kw = snapshot.running_peak_kw
    state.model_state = _copy_json_like(snapshot.model_state)


def _copy_json_like(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_copy_json_like(item) for item in value]
    if isinstance(value, dict):
        return {key: _copy_json_like(item) for key, item in value.items()}
    raise AssertionError("JSON-like values must be validated before copying")


def _validate_model_state(state: PricingState) -> None:
    _finite_nonnegative_numeric(
        state.billing_period_energy_kwh, "state.billing_period_energy_kwh"
    )
    _finite_nonnegative_numeric(state.running_peak_kw, "state.running_peak_kw")
    if not isinstance(state.model_state, dict):
        raise ValueError("state.model_state must be a dictionary with string keys")
    _validate_json_like(state.model_state, "state.model_state")


def _validate_carry_state(carry_state: object) -> None:
    if carry_state is None:
        return
    if not isinstance(carry_state, Mapping):
        raise ValueError("carry_state must be a mapping with string keys")
    _validate_json_like(dict(carry_state), "carry_state")


def _validate_json_like(value: object, field: str) -> None:
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{field} must be finite")
        if isinstance(value, float):
            return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_like(item, f"{field}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{field} has unsupported non-string key {key!r}")
            _validate_json_like(item, f"{field}.{key}")
        return
    raise ValueError(
        f"{field} has unsupported value of type {type(value).__name__}; "
        "expected JSON-like data"
    )


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _finite_nonnegative(value: object, field: str) -> float:
    number = _finite_number(value, field)
    if number < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")
    return number


def _finite_nonnegative_numeric(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (Real, np.integer, np.floating)
    ):
        raise ValueError(f"{field} must be a finite non-negative numeric value")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{field} must be a finite non-negative numeric value")
    return number


def _nonnegative_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _positive_divisor(value: object) -> int:
    if isinstance(value, bool) or type(value) is not int or value <= 0 or 1440 % value:
        raise ValueError("timestep_minutes must be a positive divisor of 1440")
    return value


def _clock(day_of_year: object, hour: object) -> PricingClock:
    if isinstance(day_of_year, bool) or type(day_of_year) is not int:
        raise ValueError("day_of_year must be an integer")
    hour_number = _finite_number(hour, "hour")
    if hour_number < 0.0 or hour_number >= 24.0:
        raise ValueError("hour must be finite in the range [0, 24)")
    return PricingClock(day_of_year=day_of_year, hour=hour_number)


__all__ = ["PriceManager"]
