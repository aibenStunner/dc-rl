from __future__ import annotations

import csv
import math
from pathlib import Path
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


_ALLOWED_KEYS = {
    "schema_version", "model", "metadata", "csv_file", "price_column",
    "source_interval_minutes", "wrap",
}


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


def _positive_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _validate_reading(reading: MeterReading) -> None:
    for field in ("energy_kwh", "demand_kw"):
        value = getattr(reading, field)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"reading.{field} must be finite and non-negative")


def _validate_parameters(parameters: Mapping[str, object]) -> tuple[Path, str, int, bool]:
    if not isinstance(parameters, Mapping):
        raise ValueError("time_series parameters must be a mapping")
    unknown = set(parameters) - _ALLOWED_KEYS
    if unknown:
        key = min(unknown, key=repr)
        raise ValueError(f"time_series parameter {key!r} is unknown")
    if type(parameters.get("schema_version")) is not int or parameters["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    if parameters.get("model") != "time_series":
        raise ValueError("model must be 'time_series'")
    if "metadata" in parameters:
        _validate_metadata(parameters["metadata"])
    csv_file = parameters.get("csv_file")
    if not isinstance(csv_file, str) or not csv_file:
        raise ValueError("csv_file must be a non-empty string")
    price_column = parameters.get("price_column")
    if not isinstance(price_column, str) or not price_column:
        raise ValueError("price_column must be a non-empty string")
    source_interval = _positive_integer(parameters.get("source_interval_minutes"),
                                        "source_interval_minutes")
    wrap = parameters.get("wrap")
    if type(wrap) is not bool:
        raise ValueError("wrap must be a boolean")
    return Path(csv_file), price_column, source_interval, wrap


def _load_prices(csv_path: Path, price_column: str) -> np.ndarray:
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or price_column not in reader.fieldnames:
                raise ValueError(f"CSV price column {price_column!r} is missing")
            values = []
            for row_number, row in enumerate(reader, start=2):
                raw = row.get(price_column)
                try:
                    value = float(raw) if raw is not None else math.nan
                except (TypeError, ValueError):
                    value = math.nan
                if not math.isfinite(value) or value < 0:
                    raise ValueError(
                        f"CSV price {price_column!r} at row {row_number} "
                        "must be finite and non-negative"
                    )
                values.append(value)
    except ValueError:
        raise
    except OSError as exc:
        raise ValueError(f"CSV file {csv_path} could not be read: {exc}") from exc
    if not values:
        raise ValueError(f"CSV price column {price_column!r} contains no data")
    return np.asarray(values, dtype=np.float64)


class TimeSeriesPricingModel:
    name = "time_series"

    def __init__(self, parameters: Mapping[str, object]):
        csv_file, self.price_column, self.source_interval_minutes, self.wrap = \
            _validate_parameters(parameters)
        self.csv_file = csv_file
        self.prices = _load_prices(csv_file, self.price_column)
        self._price_denominator = float(np.max(self.prices))

    def reset(
        self,
        *,
        context: PricingContext,
        state: PricingState,
        clock: PricingClock,
        carry_state: Mapping[str, object] | None,
        options: Mapping[str, object],
    ) -> None:
        self._validate_ratio(context)

    def _validate_ratio(self, context: PricingContext) -> int:
        simulation_interval = context.timestep_minutes
        if isinstance(simulation_interval, bool) or type(simulation_interval) is not int \
                or simulation_interval <= 0:
            raise ValueError("context.timestep_minutes must be a positive integer")
        ratio, remainder = divmod(self.source_interval_minutes, simulation_interval)
        if remainder or ratio <= 0:
            raise ValueError(
                "source_interval_minutes must be an integer multiple of "
                "context.timestep_minutes"
            )
        return ratio

    def _source_index(self, clock: PricingClock, context: PricingContext) -> int:
        self._validate_ratio(context)
        absolute_minutes = (
            (clock.day_of_year % 365) * 24 * 60 + clock.hour * 60.0
        )
        source_position = absolute_minutes / self.source_interval_minutes
        return math.floor(source_position)

    def _price_at_index(self, index: int) -> float:
        if self.wrap:
            return float(self.prices[index % len(self.prices)])
        if index < 0 or index >= len(self.prices):
            raise ValueError("forecast exceeds time-series data; set wrap: true")
        return float(self.prices[index])

    def step(
        self,
        *,
        context: PricingContext,
        state: PricingState,
        clock: PricingClock,
        reading: MeterReading,
    ) -> PricingCharges:
        _validate_reading(reading)
        return PricingCharges(
            reading.energy_kwh * self._price_at_index(self._source_index(clock, context)),
            0.0,
            0.0,
        )

    def observe(
        self,
        *,
        context: PricingContext,
        state: PricingState,
        clock: PricingClock,
    ) -> PricingObservation:
        ratio = self._validate_ratio(context)
        current = self._price_at_index(self._source_index(clock, context))
        absolute_minutes = (
            (clock.day_of_year % 365) * 24 * 60 + clock.hour * 60.0
        )
        current_simulation_step = math.floor(
            absolute_minutes / context.timestep_minutes
        )
        forecast = np.asarray([
            self._price_at_index(
                (current_simulation_step + index + 1) // ratio
            )
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
