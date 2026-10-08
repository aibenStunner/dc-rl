from typing import Mapping

import numpy as np

from utils.pricing.contracts import (
    MeterReading,
    PricingCharges,
    PricingClock,
    PricingContext,
    PricingObservation,
    PricingState,
)


class ValidPlugin:
    def __init__(self, parameters: Mapping[str, object]) -> None:
        self.name = "fixture_plugin"
        self.parameters = parameters

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
        return PricingCharges(0.0, 0.0, 0.0)

    def observe(
        self,
        *,
        context: PricingContext,
        state: PricingState,
        clock: PricingClock,
    ) -> PricingObservation:
        return PricingObservation(
            1.0,
            1.0,
            np.ones(context.future_steps, dtype=np.float32),
            0.0,
            0.0,
        )

    def export_carry_state(self, state: PricingState) -> Mapping[str, object]:
        return {}


class InvalidSignaturePlugin:
    name = "invalid_signature"

    def __init__(self, parameters: Mapping[str, object]) -> None:
        self.parameters = parameters

    def reset(
        self,
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
        return PricingCharges(0.0, 0.0, 0.0)

    def observe(
        self,
        *,
        context: PricingContext,
        state: PricingState,
        clock: PricingClock,
    ) -> PricingObservation:
        return PricingObservation(
            1.0,
            1.0,
            np.ones(context.future_steps, dtype=np.float32),
            0.0,
            0.0,
        )

    def export_carry_state(self, state: PricingState) -> Mapping[str, object]:
        return {}


class InvalidPlugin:
    def __init__(self, parameters: Mapping[str, object]) -> None:
        self.parameters = parameters
