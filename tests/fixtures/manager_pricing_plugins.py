from __future__ import annotations

from threading import Lock
from typing import Mapping

import numpy as np

from utils.managers.pricing.contracts import (
    MeterReading,
    PricingCharges,
    PricingClock,
    PricingContext,
    PricingObservation,
    PricingState,
)


class _Base:
    name = "manager_fixture"

    def __init__(self, parameters: Mapping[str, object]) -> None:
        pass

    def reset(self, *, context: PricingContext, state: PricingState,
              clock: PricingClock, carry_state: Mapping[str, object] | None,
              options: Mapping[str, object]) -> None:
        pass

    def step(self, *, context: PricingContext, state: PricingState,
             clock: PricingClock, reading: MeterReading) -> PricingCharges:
        return PricingCharges(0.0, 0.0, 0.0)

    def observe(self, *, context: PricingContext, state: PricingState,
                clock: PricingClock) -> PricingObservation:
        return PricingObservation(
            1.0, 1.0, np.ones(context.future_steps, dtype=np.float32), 0.0, 0.0,
        )

    def export_carry_state(self, state: PricingState) -> Mapping[str, object]:
        return {}


class ValidPlugin(_Base):
    pass


class NonSerializableStatePlugin(_Base):
    def reset(self, *, context: PricingContext, state: PricingState,
              clock: PricingClock, carry_state: Mapping[str, object] | None,
              options: Mapping[str, object]) -> None:
        state.model_state["resource"] = Lock()


class NestedRollbackPlugin(_Base):
    def reset(self, *, context: PricingContext, state: PricingState,
              clock: PricingClock, carry_state: Mapping[str, object] | None,
              options: Mapping[str, object]) -> None:
        state.model_state["schedule"] = {
            "segments": [{"label": "original", "values": [1, 2]}, None]
        }

    def step(self, *, context: PricingContext, state: PricingState,
             clock: PricingClock, reading: MeterReading) -> PricingCharges:
        state.model_state["schedule"]["segments"][0]["label"] = "mutated"
        state.model_state["schedule"]["segments"][0]["values"].append(3)
        return PricingCharges(np.nan, 0.0, 0.0)


class WrongForecastPlugin(_Base):
    def observe(self, *, context: PricingContext, state: PricingState,
                clock: PricingClock) -> PricingObservation:
        return PricingObservation(1.0, 1.0, np.ones(1, dtype=np.float32), 0.0, 0.0)


class NonfiniteChargePlugin(_Base):
    def step(self, *, context: PricingContext, state: PricingState,
             clock: PricingClock, reading: MeterReading) -> PricingCharges:
        return PricingCharges(np.nan, 0.0, 0.0)


class OutOfRangeObservationPlugin(_Base):
    def observe(self, *, context: PricingContext, state: PricingState,
                clock: PricingClock) -> PricingObservation:
        return PricingObservation(
            1.0, 1.1, np.ones(context.future_steps, dtype=np.float32), 0.0, 0.0,
        )


class CostPlugin(_Base):
    def step(self, *, context: PricingContext, state: PricingState,
             clock: PricingClock, reading: MeterReading) -> PricingCharges:
        return PricingCharges(2.0, 3.0, 4.0)


class MutatingNonfiniteChargePlugin(_Base):
    def step(self, *, context: PricingContext, state: PricingState,
             clock: PricingClock, reading: MeterReading) -> PricingCharges:
        state.billing_period_energy_kwh = 999.0
        state.running_peak_kw = 888.0
        state.model_state["nested"] = {"value": ["mutated"]}
        return PricingCharges(np.nan, 0.0, 0.0)


class MutatingInvalidObservationPlugin(_Base):
    def step(self, *, context: PricingContext, state: PricingState,
             clock: PricingClock, reading: MeterReading) -> PricingCharges:
        state.billing_period_energy_kwh = 999.0
        state.running_peak_kw = 888.0
        state.model_state["nested"] = {"value": ["mutated"]}
        state.model_state["return_invalid_observation"] = True
        return PricingCharges(2.0, 3.0, 4.0)

    def observe(self, *, context: PricingContext, state: PricingState,
                clock: PricingClock) -> PricingObservation:
        if state.model_state.get("return_invalid_observation"):
            return PricingObservation(
                1.0, 1.1, np.ones(context.future_steps, dtype=np.float32), 0.0, 0.0,
            )
        return super().observe(context=context, state=state, clock=clock)


class ReplacingModelStatePlugin(_Base):
    def step(self, *, context: PricingContext, state: PricingState,
             clock: PricingClock, reading: MeterReading) -> PricingCharges:
        state.model_state = ["invalid root"]
        return PricingCharges(0.0, 0.0, 0.0)


class ObserveMutatesInvalidStatePlugin(_Base):
    def __init__(self, parameters: Mapping[str, object]) -> None:
        self._observations = 0

    def observe(self, *, context: PricingContext, state: PricingState,
                clock: PricingClock) -> PricingObservation:
        self._observations += 1
        if self._observations > 1:
            state.model_state["resource"] = Lock()
        return super().observe(context=context, state=state, clock=clock)


class MutatingInvalidSettlementPlugin(_Base):
    def settle_pending_period(self, *, state: PricingState) -> PricingCharges:
        state.billing_period_energy_kwh = 999.0
        state.running_peak_kw = 888.0
        state.model_state["nested"] = {"value": ["mutated"]}
        return PricingCharges(np.nan, 0.0, 0.0)


class SettlementChangesObservationPlugin(_Base):
    def reset(self, *, context: PricingContext, state: PricingState,
              clock: PricingClock, carry_state: Mapping[str, object] | None,
              options: Mapping[str, object]) -> None:
        state.model_state["settled"] = False

    def settle_pending_period(self, *, state: PricingState) -> PricingCharges:
        state.model_state["settled"] = True
        return PricingCharges(0.0, 0.0, 7.0)

    def observe(self, *, context: PricingContext, state: PricingState,
                clock: PricingClock) -> PricingObservation:
        normalized_price = 0.5 if state.model_state["settled"] else 1.0
        return PricingObservation(
            normalized_price, normalized_price,
            np.full(context.future_steps, normalized_price, dtype=np.float32),
            0.0, 0.0,
        )


class MutatingInvalidCarryExportPlugin(_Base):
    def export_carry_state(self, state: PricingState) -> Mapping[str, object]:
        state.billing_period_energy_kwh = 999.0
        state.running_peak_kw = 888.0
        state.model_state["nested"] = {"value": ["mutated"]}
        return {"nested": {"peak_kw": np.nan}}


class MutatingRaisingCarryExportPlugin(_Base):
    def export_carry_state(self, state: PricingState) -> Mapping[str, object]:
        state.billing_period_energy_kwh = 999.0
        state.running_peak_kw = 888.0
        state.model_state["nested"] = {"value": ["mutated"]}
        raise RuntimeError("export failed")


class NestedNonfiniteStatePlugin(_Base):
    def reset(self, *, context: PricingContext, state: PricingState,
              clock: PricingClock, carry_state: Mapping[str, object] | None,
              options: Mapping[str, object]) -> None:
        state.model_state["period"] = {"peak_kw": np.nan}


class _SharedStateMutationPlugin(_Base):
    field: str
    invalid_value: object

    def _mutate_shared_state(self, state: PricingState) -> None:
        setattr(state, self.field, self.invalid_value)


class _StepSharedStateMutationPlugin(_SharedStateMutationPlugin):
    def step(self, *, context: PricingContext, state: PricingState,
             clock: PricingClock, reading: MeterReading) -> PricingCharges:
        self._mutate_shared_state(state)
        return PricingCharges(2.0, 3.0, 4.0)


class _ResetSharedStateMutationPlugin(_SharedStateMutationPlugin):
    def __init__(self, parameters: Mapping[str, object]) -> None:
        self._resets = 0

    def reset(self, *, context: PricingContext, state: PricingState,
              clock: PricingClock, carry_state: Mapping[str, object] | None,
              options: Mapping[str, object]) -> None:
        self._resets += 1
        if self._resets > 1:
            self._mutate_shared_state(state)


class StepNaNBillingEnergyPlugin(_StepSharedStateMutationPlugin):
    field = "billing_period_energy_kwh"
    invalid_value = np.nan


class StepNegativeBillingEnergyPlugin(_StepSharedStateMutationPlugin):
    field = "billing_period_energy_kwh"
    invalid_value = -1.0


class StepNaNRunningPeakPlugin(_StepSharedStateMutationPlugin):
    field = "running_peak_kw"
    invalid_value = np.nan


class StepNegativeRunningPeakPlugin(_StepSharedStateMutationPlugin):
    field = "running_peak_kw"
    invalid_value = -1.0


class ResetNaNBillingEnergyPlugin(_ResetSharedStateMutationPlugin):
    field = "billing_period_energy_kwh"
    invalid_value = np.nan


class ResetNegativeBillingEnergyPlugin(_ResetSharedStateMutationPlugin):
    field = "billing_period_energy_kwh"
    invalid_value = -1.0


class ResetNaNRunningPeakPlugin(_ResetSharedStateMutationPlugin):
    field = "running_peak_kw"
    invalid_value = np.nan


class ResetNegativeRunningPeakPlugin(_ResetSharedStateMutationPlugin):
    field = "running_peak_kw"
    invalid_value = -1.0


class StepBooleanBillingEnergyPlugin(_StepSharedStateMutationPlugin):
    field = "billing_period_energy_kwh"
    invalid_value = True


class ResetBooleanRunningPeakPlugin(_ResetSharedStateMutationPlugin):
    field = "running_peak_kw"
    invalid_value = True
