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


_ALLOWED_KEYS = {"schema_version", "model", "metadata", "energy_price_c_per_kwh"}


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


def _validate_parameters(parameters: Mapping[str, object]) -> float:
    if not isinstance(parameters, Mapping):
        raise ValueError("flat parameters must be a mapping")
    unknown = set(parameters) - _ALLOWED_KEYS
    if unknown:
        key = min(unknown, key=repr)
        raise ValueError(f"flat parameter {key!r} is unknown")
    if type(parameters.get("schema_version")) is not int:
        raise ValueError("schema_version must be 1")
    if parameters["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    if parameters.get("model") != "flat":
        raise ValueError("model must be 'flat'")
    if "metadata" in parameters:
        _validate_metadata(parameters["metadata"])
    if "energy_price_c_per_kwh" not in parameters:
        raise ValueError("energy_price_c_per_kwh is required")
    value = parameters["energy_price_c_per_kwh"]
    if isinstance(value, bool):
        raise ValueError("energy_price_c_per_kwh must be a finite non-negative number")
    try:
        price = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "energy_price_c_per_kwh must be a finite non-negative number"
        ) from exc
    if not math.isfinite(price) or price < 0:
        raise ValueError("energy_price_c_per_kwh must be a finite non-negative number")
    return price


def _validate_reading(reading: MeterReading) -> None:
    for field in ("energy_kwh", "demand_kw"):
        value = getattr(reading, field)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"reading.{field} must be finite and non-negative")


class FlatPricingModel:
    name = "flat"

    def __init__(self, parameters: Mapping[str, object]):
        self.price_c_per_kwh = _validate_parameters(parameters)

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

    def step(
        self,
        *,
        context: PricingContext,
        state: PricingState,
        clock: PricingClock,
        reading: MeterReading,
    ) -> PricingCharges:
        _validate_reading(reading)
        return PricingCharges(reading.energy_kwh * self.price_c_per_kwh, 0.0, 0.0)

    def observe(
        self,
        *,
        context: PricingContext,
        state: PricingState,
        clock: PricingClock,
    ) -> PricingObservation:
        normalized = 0.0 if self.price_c_per_kwh == 0 else 1.0
        return PricingObservation(
            self.price_c_per_kwh,
            normalized,
            np.full(context.future_steps, normalized, dtype=np.float32),
            0.0,
            0.0,
        )

    def export_carry_state(self, state: PricingState) -> Mapping[str, object]:
        return {}
