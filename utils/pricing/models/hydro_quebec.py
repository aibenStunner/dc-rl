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


_ALLOWED_KEYS = {"schema_version", "model", "metadata", "selection", "rates"}
_RATE_NAMES = ("rate_m", "rate_l", "new_dc_rate")
_ALLOWED_OPTIONS = {"tariff", "demand_floor_kw"}
_ALLOWED_TARIFFS = ("auto",) + _RATE_NAMES
_COMMON_RATE_FIELDS = {"approved", "approximate", "citation"}
_RATE_ALLOWED_FIELDS = {
    "rate_m": _COMMON_RATE_FIELDS | {
        "energy_tier1_c_per_kwh", "energy_tier2_c_per_kwh", "tier2_threshold_kwh",
        "demand_charge_c_per_kw_month", "minimum_billing_demand_kw",
        "winter_ratchet_fraction", "minimum_bill_c_1phase", "minimum_bill_c_3phase",
    },
    "rate_l": _COMMON_RATE_FIELDS | {
        "energy_price_c_per_kwh", "demand_charge_c_per_kw_month",
        "minimum_billing_demand_kw", "optimization_charge_c_per_kw_day",
        "optimization_charge_cap_c_per_kw_month", "optimization_overrun_fraction",
    },
    "new_dc_rate": _COMMON_RATE_FIELDS | {
        "energy_price_c_per_kwh", "demand_charge_c_per_kw_month",
        "minimum_billing_demand_kw",
    },
}
_RATE_REQUIRED_FIELDS = {
    "rate_m": {
        "energy_tier1_c_per_kwh", "energy_tier2_c_per_kwh", "tier2_threshold_kwh",
        "demand_charge_c_per_kw_month", "minimum_billing_demand_kw",
        "winter_ratchet_fraction", "minimum_bill_c_1phase", "minimum_bill_c_3phase",
    },
    "rate_l": {
        "energy_price_c_per_kwh", "demand_charge_c_per_kw_month",
        "minimum_billing_demand_kw", "optimization_charge_c_per_kw_day",
        "optimization_charge_cap_c_per_kw_month", "optimization_overrun_fraction",
    },
    "new_dc_rate": {
        "energy_price_c_per_kwh", "demand_charge_c_per_kw_month",
        "minimum_billing_demand_kw",
    },
}


def _field_error(field: str, message: str) -> ValueError:
    return ValueError(f"{field} {message}")


def _finite_nonnegative(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise _field_error(field, "must be a finite non-negative number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise _field_error(field, "must be a finite non-negative number") from exc
    if not math.isfinite(number) or number < 0.0:
        raise _field_error(field, "must be a finite non-negative number")
    return number


def _finite_positive(value: object, field: str) -> float:
    number = _finite_nonnegative(value, field)
    if number <= 0.0:
        raise _field_error(field, "must be greater than zero")
    return number


def _fraction(value: object, field: str) -> float:
    number = _finite_nonnegative(value, field)
    if number > 1.0:
        raise _field_error(field, "must be between zero and one")
    return number


def _overrun_fraction(value: object, field: str) -> float:
    number = _finite_nonnegative(value, field)
    if number < 1.0:
        raise _field_error(field, "must be at least one")
    return number


def _required_mapping(parameters: Mapping[str, object], field: str) -> Mapping[str, object]:
    value = parameters.get(field)
    if not isinstance(value, Mapping):
        raise _field_error(field, "must be a mapping")
    return value


def _validate_metadata(metadata: object) -> None:
    if not isinstance(metadata, Mapping):
        raise _field_error("metadata", "must be a mapping")
    allowed = {"name", "currency", "effective_date", "source", "winter"}
    unknown = set(metadata) - allowed
    if unknown:
        key = min(unknown, key=repr)
        raise _field_error(f"metadata.{key}", "is unknown")
    for field in ("name", "currency", "effective_date", "source"):
        if field in metadata and (not isinstance(metadata[field], str) or not metadata[field]):
            raise _field_error(f"metadata.{field}", "must be a non-empty string")
    if "winter" not in metadata:
        raise _field_error("metadata.winter", "is required")
    winter = metadata["winter"]
    if not isinstance(winter, Mapping):
        raise _field_error("metadata.winter", "must be a mapping")
    unknown = set(winter) - {"start_day_of_year", "end_day_of_year"}
    if unknown:
        key = min(unknown, key=repr)
        raise _field_error(f"metadata.winter.{key}", "is unknown")
    for field in ("start_day_of_year", "end_day_of_year"):
        value = winter.get(field)
        if isinstance(value, bool) or type(value) is not int or not 0 <= value < 365:
            raise _field_error(
                f"metadata.winter.{field}",
                "must be an integer in the range 0 through 364",
            )


def _validate_rate(rate_name: str, rate: object) -> dict[str, object]:
    if not isinstance(rate, Mapping):
        raise _field_error(f"rates.{rate_name}", "must be a mapping")

    unknown = set(rate) - _RATE_ALLOWED_FIELDS[rate_name]
    if unknown:
        key = min(unknown, key=repr)
        raise _field_error(f"rates.{rate_name}.{key}", "is unknown")

    approved = rate.get("approved")
    if type(approved) is not bool:
        raise _field_error(f"rates.{rate_name}.approved", "must be a boolean")
    if "approximate" in rate and type(rate["approximate"]) is not bool:
        raise _field_error(f"rates.{rate_name}.approximate", "must be a boolean")
    if "citation" not in rate or not isinstance(rate["citation"], str) or not rate["citation"]:
        raise _field_error(f"rates.{rate_name}.citation", "must be a non-empty string")

    result = dict(rate)
    required = tuple(_RATE_REQUIRED_FIELDS[rate_name])
    for field in required:
        if field not in rate:
            raise _field_error(f"rates.{rate_name}.{field}", "is required")

    numeric_fields = set(required) | {
        "energy_price_c_per_kwh", "energy_tier1_c_per_kwh", "energy_tier2_c_per_kwh",
        "tier2_threshold_kwh", "demand_charge_c_per_kw_month",
        "minimum_billing_demand_kw", "winter_ratchet_fraction",
        "minimum_bill_c_1phase", "minimum_bill_c_3phase",
        "optimization_charge_c_per_kw_day", "optimization_charge_cap_c_per_kw_month",
        "optimization_overrun_fraction",
    }
    for field in numeric_fields:
        if field not in rate:
            continue
        if field == "winter_ratchet_fraction":
            result[field] = _fraction(rate[field], f"rates.{rate_name}.{field}")
        elif field == "optimization_overrun_fraction":
            result[field] = _overrun_fraction(rate[field], f"rates.{rate_name}.{field}")
        else:
            result[field] = _finite_nonnegative(rate[field], f"rates.{rate_name}.{field}")

    if rate_name == "rate_m" and result["energy_tier2_c_per_kwh"] is None:
        raise _field_error("rates.rate_m.energy_tier2_c_per_kwh", "must be provided")
    if rate_name == "rate_m" and result["tier2_threshold_kwh"] <= 0.0:
        raise _field_error("rates.rate_m.tier2_threshold_kwh", "must be greater than zero")
    return result


def _validate_parameters(parameters: Mapping[str, object]) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    if not isinstance(parameters, Mapping):
        raise ValueError("hydro_quebec parameters must be a mapping")
    unknown = set(parameters) - _ALLOWED_KEYS
    if unknown:
        key = min(unknown, key=repr)
        raise _field_error(f"parameter {key!r}", "is unknown")
    if type(parameters.get("schema_version")) is not int or parameters["schema_version"] != 1:
        raise _field_error("schema_version", "must be 1")
    if parameters.get("model") != "hydro_quebec":
        raise _field_error("model", "must be 'hydro_quebec'")
    if "metadata" in parameters:
        _validate_metadata(parameters["metadata"])

    selection = _required_mapping(parameters, "selection")
    unknown = set(selection) - {"rate_l_threshold_kw"}
    if unknown:
        key = min(unknown, key=repr)
        raise _field_error(f"selection.{key}", "is unknown")
    threshold = _finite_positive(selection.get("rate_l_threshold_kw"),
                                 "selection.rate_l_threshold_kw")

    rates = _required_mapping(parameters, "rates")
    missing = [name for name in _RATE_NAMES if name not in rates]
    if missing:
        raise _field_error("rates", f"is missing {missing[0]!r}")
    unknown = set(rates) - set(_RATE_NAMES)
    if unknown:
        key = min(unknown, key=repr)
        raise _field_error(f"rates.{key}", "is unknown")
    validated_rates = {
        name: _validate_rate(name, rates[name]) for name in _RATE_NAMES
    }
    return {"rate_l_threshold_kw": threshold}, validated_rates


def validate_options(options: Mapping[str, object]) -> None:
    if not isinstance(options, Mapping):
        raise ValueError("pricing.options must be a mapping")
    unknown = set(options) - _ALLOWED_OPTIONS
    if unknown:
        key = min(unknown, key=repr)
        raise ValueError(f"pricing.options.{key} is unknown")
    tariff = options.get("tariff", "auto")
    if not isinstance(tariff, str) or tariff not in _ALLOWED_TARIFFS:
        raise ValueError(
            "pricing.options.tariff must be one of "
            + ", ".join(_ALLOWED_TARIFFS)
        )
    if "demand_floor_kw" in options:
        _finite_nonnegative(options["demand_floor_kw"], "pricing.options.demand_floor_kw")


def _validate_reading(reading: MeterReading) -> None:
    for field in ("energy_kwh", "demand_kw"):
        value = getattr(reading, field)
        if isinstance(value, bool) or not math.isfinite(value) or value < 0.0:
            raise ValueError(f"reading.{field} must be finite and non-negative")


def _validate_context(context: PricingContext) -> None:
    if isinstance(context.timestep_minutes, bool) or type(context.timestep_minutes) is not int \
            or context.timestep_minutes <= 0 or 1440 % context.timestep_minutes:
        raise ValueError("context.timestep_minutes must be a positive divisor of 1440")
    if isinstance(context.future_steps, bool) or type(context.future_steps) is not int \
            or context.future_steps < 0:
        raise ValueError("context.future_steps must be a non-negative integer")
    if isinstance(context.datacenter_capacity_mw, bool) \
            or not math.isfinite(context.datacenter_capacity_mw) \
            or context.datacenter_capacity_mw < 0.0:
        raise ValueError("context.datacenter_capacity_mw must be finite and non-negative")


def _validate_clock(clock: PricingClock) -> None:
    if isinstance(clock.day_of_year, bool) or type(clock.day_of_year) is not int:
        raise ValueError("clock.day_of_year must be an integer")
    if isinstance(clock.hour, bool) or not math.isfinite(clock.hour) \
            or clock.hour < 0.0 or clock.hour >= 24.0:
        raise ValueError("clock.hour must be finite in the range [0, 24)")


class HydroQuebecPricingModel:
    name = "hydro_quebec"

    def __init__(self, parameters: Mapping[str, object]):
        selection, rates = _validate_parameters(parameters)
        self.rate_l_threshold_kw = selection["rate_l_threshold_kw"]
        self.rates = rates
        winter = parameters["metadata"]["winter"]
        self.winter_start_day = winter["start_day_of_year"]
        self.winter_end_day = winter["end_day_of_year"]
        self._parameters = parameters

    def _select_tariff(self, context: PricingContext, options: Mapping[str, object]) -> str:
        validate_options(options)
        tariff = options.get("tariff", "auto")
        if tariff != "auto":
            return tariff
        capacity_kw = context.datacenter_capacity_mw * 1000.0
        return "rate_l" if capacity_kw >= self.rate_l_threshold_kw else "rate_m"

    def _winter_day(self, day_of_year: int) -> bool:
        day = day_of_year % 365
        if self.winter_start_day <= self.winter_end_day:
            return self.winter_start_day <= day <= self.winter_end_day
        return day >= self.winter_start_day or day <= self.winter_end_day

    def _rate(self, state: PricingState) -> Mapping[str, object]:
        return self.rates[state.model_state["tariff"]]

    def _energy_price(self, state: PricingState) -> float:
        rate = self._rate(state)
        if state.model_state["tariff"] == "rate_m":
            if state.billing_period_energy_kwh >= rate["tier2_threshold_kwh"]:
                return rate["energy_tier2_c_per_kwh"]
            return rate["energy_tier1_c_per_kwh"]
        return rate["energy_price_c_per_kwh"]

    def _energy_cost(self, state: PricingState, energy_kwh: float) -> float:
        rate = self._rate(state)
        if state.model_state["tariff"] != "rate_m":
            return energy_kwh * rate["energy_price_c_per_kwh"]
        tier1_remaining = max(
            0.0,
            rate["tier2_threshold_kwh"] - state.billing_period_energy_kwh,
        )
        tier1_kwh = min(energy_kwh, tier1_remaining)
        return (tier1_kwh * rate["energy_tier1_c_per_kwh"]
                + (energy_kwh - tier1_kwh) * rate["energy_tier2_c_per_kwh"])

    def _normalizer(self, state: PricingState) -> float:
        rate = self._rate(state)
        if state.model_state["tariff"] == "rate_m":
            return max(rate["energy_tier1_c_per_kwh"], rate["energy_tier2_c_per_kwh"])
        return rate["energy_price_c_per_kwh"]

    def _settle_pending_day(self, state: PricingState) -> float:
        if state.model_state["tariff"] != "rate_l":
            return 0.0
        if state.model_state["pending_day_key"] is None:
            return 0.0

        day_peak = state.model_state["day_peak_kw"]
        was_winter = state.model_state["pending_day_is_winter"]
        state.model_state["pending_day_key"] = None
        state.model_state["pending_day_is_winter"] = False
        state.model_state["pending_day_started_at_midnight"] = False
        state.model_state["day_peak_kw"] = 0.0
        if not was_winter:
            return 0.0

        rate = self._rate(state)
        contract_power_kw = state.model_state["contract_power_kw"]
        overrun_kw = max(
            0.0,
            day_peak - rate["optimization_overrun_fraction"] * contract_power_kw,
        )
        if overrun_kw <= 0.0:
            return 0.0
        charge = overrun_kw * rate["optimization_charge_c_per_kw_day"]
        cap = rate["optimization_charge_cap_c_per_kw_month"] * contract_power_kw
        room = max(0.0, cap - state.model_state["month_optimization_charge_c"])
        charge = min(charge, room)
        state.model_state["month_optimization_charge_c"] += charge
        return charge

    def _record_daily_peak(
        self,
        state: PricingState,
        clock: PricingClock,
        reading: MeterReading,
        context: PricingContext,
    ) -> float:
        if state.model_state["tariff"] != "rate_l":
            return 0.0

        day_key = clock.day_of_year % 365
        pending_day_key = state.model_state["pending_day_key"]
        additional = 0.0
        if pending_day_key is not None and pending_day_key != day_key:
            additional = self._settle_pending_day(state)

        if state.model_state["pending_day_key"] is None:
            state.model_state["pending_day_key"] = day_key
            state.model_state["pending_day_is_winter"] = self._winter_day(clock.day_of_year)
            state.model_state["pending_day_started_at_midnight"] = clock.hour == 0.0
            state.model_state["day_peak_kw"] = reading.demand_kw
        else:
            state.model_state["day_peak_kw"] = max(
                state.model_state["day_peak_kw"], reading.demand_kw
            )

        # Preserve the established per-step settlement behavior for complete
        # calendar days, while keeping a partial reset day pending until its
        # calendar boundary or an explicit settlement.
        if (state.model_state["pending_day_started_at_midnight"]
                and clock.hour + context.timestep_minutes / 60.0 >= 24.0):
            additional += self._settle_pending_day(state)
        return additional

    def settle_pending_day(self, *, state: PricingState) -> PricingCharges:
        if "tariff" not in state.model_state:
            raise ValueError("model must be reset before settling a pending day")
        return PricingCharges(0.0, 0.0, self._settle_pending_day(state))

    def settle_pending_period(self, *, state: PricingState) -> PricingCharges:
        return self.settle_pending_day(state=state)

    def reset(
        self,
        *,
        context: PricingContext,
        state: PricingState,
        clock: PricingClock,
        carry_state: Mapping[str, object] | None,
        options: Mapping[str, object],
    ) -> None:
        _validate_context(context)
        _validate_clock(clock)
        validate_options(options)
        tariff = self._select_tariff(context, options)
        demand_floor = _finite_nonnegative(
            options.get("demand_floor_kw", 0.0),
            "pricing.options.demand_floor_kw",
        )
        carry = {} if carry_state is None else carry_state
        if not isinstance(carry, Mapping):
            raise ValueError("carry_state must be a mapping")
        unknown = set(carry) - {"winter_peak_kw", "ending_peak_kw"}
        if unknown:
            key = min(unknown, key=repr)
            raise _field_error(f"carry_state.{key}", "is unknown")
        winter_peak = _finite_nonnegative(carry.get("winter_peak_kw", 0.0),
                                          "carry_state.winter_peak_kw")
        ending_peak = _finite_nonnegative(carry.get("ending_peak_kw", 0.0),
                                          "carry_state.ending_peak_kw")
        rate = self.rates[tariff]
        ratchet = rate.get("winter_ratchet_fraction")
        ratcheted_floor = (ratchet * winter_peak) if ratchet is not None else 0.0
        minimum_billing_demand = rate.get("minimum_billing_demand_kw", 0.0)
        state.billing_period_energy_kwh = 0.0
        state.running_peak_kw = max(
            ending_peak,
            demand_floor,
            ratcheted_floor,
            minimum_billing_demand,
        )
        state.model_state.clear()
        state.model_state.update({
            "tariff": tariff,
            "winter_peak_kw": winter_peak,
            "ending_peak_kw": state.running_peak_kw,
            "ratcheted_floor_kw": ratcheted_floor,
            "demand_floor_kw": demand_floor,
            "contract_power_kw": max(state.running_peak_kw,
                                      rate.get("minimum_billing_demand_kw", 0.0)),
            "day_peak_kw": 0.0,
            "pending_day_key": None,
            "pending_day_is_winter": False,
            "pending_day_started_at_midnight": False,
            "month_optimization_charge_c": 0.0,
        })

    def step(
        self,
        *,
        context: PricingContext,
        state: PricingState,
        clock: PricingClock,
        reading: MeterReading,
    ) -> PricingCharges:
        _validate_context(context)
        _validate_clock(clock)
        _validate_reading(reading)
        if "tariff" not in state.model_state:
            raise ValueError("model must be reset before step")
        is_winter = self._winter_day(clock.day_of_year)
        if is_winter:
            state.model_state["winter_peak_kw"] = max(
                state.model_state["winter_peak_kw"], reading.demand_kw
            )
        previous_peak = state.running_peak_kw
        state.running_peak_kw = max(state.running_peak_kw, reading.demand_kw)
        state.model_state["ending_peak_kw"] = state.running_peak_kw
        demand_increment = self._rate(state)["demand_charge_c_per_kw_month"] * (
            state.running_peak_kw - previous_peak
        )
        energy_cost = self._energy_cost(state, reading.energy_kwh)
        additional = self._record_daily_peak(state, clock, reading, context)
        return PricingCharges(energy_cost, demand_increment, additional)

    def observe(
        self,
        *,
        context: PricingContext,
        state: PricingState,
        clock: PricingClock,
    ) -> PricingObservation:
        _validate_context(context)
        _validate_clock(clock)
        if "tariff" not in state.model_state:
            raise ValueError("model must be reset before observe")
        denominator = self._normalizer(state)
        current = self._energy_price(state)
        normalized = 0.0 if denominator == 0.0 else current / denominator
        forecast = np.full(context.future_steps, normalized, dtype=np.float32)
        rate = self._rate(state)
        threshold = rate.get("tier2_threshold_kwh")
        progress = (0.0 if threshold is None else
                    min(1.0, state.billing_period_energy_kwh / threshold))
        normalized_peak = min(
            1.0,
            state.running_peak_kw / max(context.datacenter_capacity_mw * 1000.0, 1.0),
        )
        return PricingObservation(
            current_price_c_per_kwh=float(current),
            normalized_price=float(np.clip(normalized, 0.0, 1.0)),
            forecast_normalized=forecast,
            billing_progress=float(progress),
            normalized_peak=float(normalized_peak),
        )

    def export_carry_state(self, state: PricingState) -> Mapping[str, object]:
        if "tariff" not in state.model_state:
            raise ValueError("model must be reset before exporting carry state")
        return {
            "winter_peak_kw": float(state.model_state["winter_peak_kw"]),
            "ending_peak_kw": float(state.model_state["ending_peak_kw"]),
        }


__all__ = ["HydroQuebecPricingModel", "validate_options"]
