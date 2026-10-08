from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, TypeAlias, runtime_checkable

import numpy as np


JSONLikeValue: TypeAlias = (
    None | bool | int | float | str | list["JSONLikeValue"] | dict[str, "JSONLikeValue"]
)


@dataclass(frozen=True)
class PricingContext:
    datacenter_capacity_mw: float
    timestep_minutes: int
    future_steps: int


@dataclass(frozen=True)
class PricingClock:
    day_of_year: int
    hour: float


@dataclass(frozen=True)
class MeterReading:
    energy_kwh: float
    demand_kw: float


@dataclass
class PricingState:
    """Shared pricing state; ``model_state`` contains JSON-like plugin data only.

    Pricing models may store only ``None``, booleans, integers, floats, strings,
    lists, and string-keyed dictionaries recursively in ``model_state``. The
    manager validates this contract after model lifecycle mutations so state can
    be safely snapshotted and restored transactionally.
    """

    billing_period_energy_kwh: float = 0.0
    running_peak_kw: float = 0.0
    model_state: dict[str, JSONLikeValue] = field(default_factory=dict)


@dataclass(frozen=True)
class PricingObservation:
    current_price_c_per_kwh: float
    normalized_price: float
    forecast_normalized: np.ndarray
    billing_progress: float
    normalized_peak: float

    def __post_init__(self) -> None:
        forecast = np.array(self.forecast_normalized, copy=True)
        immutable_buffer = memoryview(forecast.tobytes())
        immutable_forecast = np.frombuffer(
            immutable_buffer, dtype=forecast.dtype, count=forecast.size
        ).reshape(forecast.shape)
        object.__setattr__(self, "forecast_normalized", immutable_forecast)


@dataclass(frozen=True)
class PricingCharges:
    energy_cost_c: float
    demand_cost_increment_c: float
    additional_cost_increment_c: float

    @property
    def total_cost_c(self) -> float:
        return (self.energy_cost_c + self.demand_cost_increment_c
                + self.additional_cost_increment_c)


@runtime_checkable
class PricingModel(Protocol):
    name: str

    def reset(self, *, context: PricingContext, state: PricingState,
              clock: PricingClock,
              carry_state: Mapping[str, JSONLikeValue] | None,
              options: Mapping[str, object]) -> None: ...

    def step(self, *, context: PricingContext, state: PricingState,
             clock: PricingClock, reading: MeterReading) -> PricingCharges: ...

    def observe(self, *, context: PricingContext, state: PricingState,
                clock: PricingClock) -> PricingObservation: ...

    def export_carry_state(
        self, state: PricingState
    ) -> Mapping[str, JSONLikeValue]: ...
