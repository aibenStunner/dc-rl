from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from ..contracts import (
    MeterReading,
    PricingCharges,
    PricingClock,
    PricingContext,
    PricingObservation,
    PricingState,
)


_ALLOWED_KEYS = {"schema_version", "model", "metadata", "hourly_prices_c_per_kwh"}


def _validate_metadata(metadata: object) -> None:
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping")
    unknown = set(metadata) - {"name", "currency"}
    if unknown:
        key = min(unknown, key=repr)
        raise ValueError(f"metadata.{key} is unknown")
    for field in ("name", "currency"):
        if field in metadata and (not isinstance(metadata[field], str)
                                  or not metadata[field]):
            raise ValueError(f"metadata.{field} must be a non-empty string")


def _validate_reading(reading: MeterReading) -> None:
    for field in ("energy_kwh", "demand_kw"):
        value = getattr(reading, field)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"reading.{field} must be finite and non-negative")


def _validate_parameters(parameters: Mapping[str, object]) -> np.ndarray:
    if not isinstance(parameters, Mapping):
        raise ValueError("time_of_use parameters must be a mapping")
    unknown = set(parameters) - _ALLOWED_KEYS
    if unknown:
        key = min(unknown, key=repr)
        raise ValueError(f"time_of_use parameter {key!r} is unknown")
    if type(parameters.get("schema_version")) is not int or parameters["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    if parameters.get("model") != "time_of_use":
        raise ValueError("model must be 'time_of_use'")
    if "metadata" in parameters:
        _validate_metadata(parameters["metadata"])
    values = parameters.get("hourly_prices_c_per_kwh")
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
        raise ValueError("hourly_prices_c_per_kwh must be a list")
    if len(values) != 24:
        raise ValueError("hourly_prices_c_per_kwh must contain exactly 24 values")
    prices = []
    for index, value in enumerate(values):
        if isinstance(value, bool):
            raise ValueError(f"hourly_prices_c_per_kwh[{index}] must be finite and non-negative")
        try:
            price = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"hourly_prices_c_per_kwh[{index}] must be finite and non-negative"
            ) from exc
        if not math.isfinite(price) or price < 0:
            raise ValueError(f"hourly_prices_c_per_kwh[{index}] must be finite and non-negative")
        prices.append(price)
    return np.asarray(prices, dtype=np.float64)


class TimeOfUsePricingModel:
    name = "time_of_use"

    def __init__(self, parameters: Mapping[str, object]):
        self.hourly_prices_c_per_kwh = _validate_parameters(parameters)
        self._price_denominator = float(np.max(self.hourly_prices_c_per_kwh))

    def reset(
        self,
        *,
        context: PricingContext,
        state: PricingState,
        clock: PricingClock,
        carry_state: Mapping[str, object] | None,
        options: Mapping[str, object],
    ) -> None:
        pass

    def _price_at(self, hour: float) -> float:
        return float(self.hourly_prices_c_per_kwh[math.floor(hour) % 24])

    def step(
        self,
        *,
        context: PricingContext,
        state: PricingState,
        clock: PricingClock,
        reading: MeterReading,
    ) -> PricingCharges:
        _validate_reading(reading)
        return PricingCharges(reading.energy_kwh * self._price_at(clock.hour), 0.0, 0.0)

    def observe(
        self,
        *,
        context: PricingContext,
        state: PricingState,
        clock: PricingClock,
    ) -> PricingObservation:
        current = self._price_at(clock.hour)
        step_hours = context.timestep_minutes / 60.0
        forecast = np.asarray([
            self._price_at(clock.hour + (index + 1) * step_hours)
            for index in range(context.future_steps)
        ], dtype=np.float32)
        denominator = self._price_denominator
        if denominator == 0.0:
            normalized = 0.0
            normalized_forecast = np.zeros(context.future_steps, dtype=np.float32)
        else:
            normalized = current / denominator
            normalized_forecast = forecast / denominator
        return PricingObservation(
            current,
            normalized,
            normalized_forecast,
            (clock.hour % 24.0) / 24.0,
            0.0,
        )

    def export_carry_state(self, state: PricingState) -> Mapping[str, object]:
        return {}
