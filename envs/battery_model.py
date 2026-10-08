# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC-BY-NC license found in the
# LICENSE file in the root directory of this source tree.

"""Reduced-form battery cell model with explicit bus and cell boundaries."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class BatteryFlow:
    """One operation's energy flow across the battery boundaries, in MWh."""

    charge_bus_mwh: float = 0.0
    discharge_bus_mwh: float = 0.0
    charge_cell_mwh: float = 0.0
    discharge_cell_mwh: float = 0.0


class Battery2:
    """Battery cell-energy state with C/L/C rate constraints.

    ``capacity`` and ``current_load`` are cell-side MWh. Charge and discharge
    requests are bus-side MW over ``duration_hours``. A symmetric one-way
    efficiency derived from ``round_trip_efficiency`` connects the boundaries.
    """

    def __init__(
        self,
        capacity,
        current_load=0,
        round_trip_efficiency=1.0,
        c_lim=0.1,
        d_lim=1,
        upper_u=-0.04,
        upper_v=1,
        lower_u=0.01,
        lower_v=0,
    ):
        if not _is_finite_positive(round_trip_efficiency) or round_trip_efficiency > 1.0:
            raise ValueError("round_trip_efficiency must be finite in (0, 1]")
        self.capacity = float(capacity)
        self.current_load = float(current_load)
        self.round_trip_efficiency = float(round_trip_efficiency)
        self.eff_c = math.sqrt(self.round_trip_efficiency)
        self.eff_d = math.sqrt(self.round_trip_efficiency)
        self.c_lim = float(c_lim)
        self.d_lim = float(d_lim)
        self.upper_lim_u = float(upper_u)
        self.upper_lim_v = float(upper_v)
        self.lower_lim_u = float(lower_u)
        self.lower_lim_v = float(lower_v)
        self.last_flow = BatteryFlow()

    def reset(self, initial_soc=0.0):
        """Reset cell energy to a validated fraction of usable capacity."""
        if not _is_finite_nonnegative(initial_soc) or float(initial_soc) > 1.0:
            raise ValueError("initial_soc must be finite in [0, 1]")
        self.current_load = self.capacity * float(initial_soc)
        self.last_flow = BatteryFlow()

    def calc_max_charge_bus_power(self, duration_hours):
        """Maximum bus-side charge power in MW at the current cell SoC."""
        max_cell_rate_power = self.capacity * self.c_lim / self.eff_c
        # SoC ceiling is a cell-energy bound. Convert the free cell energy to
        # bus energy through eta_charge; C/L/C upper_u remains a legacy rate
        # curve term and must not prevent a physically full battery.
        max_cell_capacity_power = (
            (self.capacity - self.current_load)
            / (self.eff_c * duration_hours)
        )
        return max(0.0, min(max_cell_rate_power, max_cell_capacity_power))

    def calc_max_discharge_bus_power(self, duration_hours, dc_bus_limit_mw):
        """Maximum bus-side discharge power in MW at the current cell SoC."""
        max_cell_rate_power = self.capacity * self.d_lim * self.eff_d
        # SoC floor is a cell-energy bound. Convert available cell energy to
        # bus delivery through eta_discharge; do not leave stranded energy at
        # an implicit lower floor when no minimum SoC is configured.
        max_cell_capacity_power = (
            self.current_load * self.eff_d / duration_hours
        )
        return max(0.0, min(max_cell_rate_power, max_cell_capacity_power,
                            max(0.0, dc_bus_limit_mw)))

    def charge(self, requested_bus_power_mw, duration_hours):
        """Charge from the bus and return the realized flow."""
        _validate_operation_inputs(requested_bus_power_mw, duration_hours)
        bus_power = min(requested_bus_power_mw,
                        self.calc_max_charge_bus_power(duration_hours))
        charge_bus_mwh = bus_power * duration_hours
        charge_cell_mwh = charge_bus_mwh * self.eff_c
        self.current_load = round(min(self.capacity, self.current_load + charge_cell_mwh), 8)
        self.last_flow = BatteryFlow(
            charge_bus_mwh=charge_bus_mwh,
            charge_cell_mwh=charge_cell_mwh,
        )
        return self.last_flow

    def discharge(self, requested_bus_power_mw, duration_hours, dc_bus_limit_mw):
        """Discharge to the DC bus and return the realized flow."""
        _validate_operation_inputs(requested_bus_power_mw, duration_hours)
        if not _is_finite_nonnegative(dc_bus_limit_mw):
            raise ValueError("dc_bus_limit_mw must be finite and non-negative")
        bus_power = min(requested_bus_power_mw,
                        self.calc_max_discharge_bus_power(duration_hours, dc_bus_limit_mw))
        discharge_bus_mwh = bus_power * duration_hours
        discharge_cell_mwh = discharge_bus_mwh / self.eff_d
        self.current_load = round(max(0.0, self.current_load - discharge_cell_mwh), 8)
        self.last_flow = BatteryFlow(
            discharge_bus_mwh=discharge_bus_mwh,
            discharge_cell_mwh=discharge_cell_mwh,
        )
        return self.last_flow

    def is_full(self):
        return self.capacity == self.current_load

    def get_battery_soc(self):
        """Return state of charge in [0, 1]."""
        return self.current_load / self.capacity


def _is_finite_positive(value):
    return not isinstance(value, bool) and math.isfinite(float(value)) and float(value) > 0.0


def _is_finite_nonnegative(value):
    return not isinstance(value, bool) and math.isfinite(float(value)) and float(value) >= 0.0


def _validate_operation_inputs(requested_bus_power_mw, duration_hours):
    if not _is_finite_nonnegative(requested_bus_power_mw):
        raise ValueError("requested_bus_power_mw must be finite and non-negative")
    if not _is_finite_positive(duration_hours):
        raise ValueError("duration_hours must be finite and positive")
