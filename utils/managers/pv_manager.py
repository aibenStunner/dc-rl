"""Fixed-tilt on-site PV production manager driven by an EPW weather file."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

from .common import REPO_ROOT, STEPS_PER_HOUR, _cyc, _cyc_at


class PV_Manager:
    """Generate fixed-tilt PV AC energy at the simulator's 15-minute cadence.

    The EPW is hourly. Its modeled AC power is zero-order-held across four
    simulator intervals, conserving hourly PV energy without fabricating
    sub-hour irradiance variability.
    """

    _ALLOWED_TEMPERATURE_MODELS = {
        "close_mount_glass_glass",
        "open_rack_glass_glass",
        "open_rack_glass_polymer",
        "insulated_back_glass_polymer",
    }

    def __init__(
        self,
        *,
        epw_filename: str,
        datacenter_capacity_mw: float,
        enabled: bool = True,
        capacity_fraction_of_datacenter: float = 0.05,
        surface_tilt_deg: float = 35.0,
        surface_azimuth_deg: float = 180.0,
        dc_ac_ratio: float = 1.20,
        gamma_pdc_per_deg_c: float = -0.0037,
        system_losses_fraction: float = 0.1408,
        inverter_efficiency: float = 0.96,
        temperature_model: str = "close_mount_glass_glass",
        export_policy: str = "curtail",
        timezone_shift: int = 0,
        init_day: int = 0,
    ):
        self.enabled = _validate_bool(enabled, "pv.enabled")
        self.capacity_fraction_of_datacenter = _finite_nonnegative(
            capacity_fraction_of_datacenter, "pv.capacity_fraction_of_datacenter"
        )
        self.surface_tilt_deg = _bounded(
            surface_tilt_deg, "pv.surface_tilt_deg", 0.0, 90.0
        )
        self.surface_azimuth_deg = _bounded(
            surface_azimuth_deg, "pv.surface_azimuth_deg", 0.0, 360.0
        )
        self.dc_ac_ratio = _finite_positive(dc_ac_ratio, "pv.dc_ac_ratio")
        self.gamma_pdc_per_deg_c = _finite_number(
            gamma_pdc_per_deg_c, "pv.gamma_pdc_per_deg_c"
        )
        self.system_losses_fraction = _bounded(
            system_losses_fraction, "pv.system_losses_fraction", 0.0, 1.0
        )
        self.inverter_efficiency = _bounded(
            inverter_efficiency, "pv.inverter_efficiency", 0.0, 1.0,
            strict_lower=True,
        )
        if temperature_model not in self._ALLOWED_TEMPERATURE_MODELS:
            raise ValueError(
                "pv.temperature_model must be one of "
                f"{sorted(self._ALLOWED_TEMPERATURE_MODELS)}"
            )
        if export_policy != "curtail":
            raise ValueError("pv.export_policy must be 'curtail'")
        self.temperature_model = temperature_model
        self.export_policy = export_policy
        self.timezone_shift = _integer(timezone_shift, "timezone_shift")
        self.init_day = _integer(init_day, "init_day")
        self.capacity_dc_kw = self.capacity_fraction_of_datacenter * datacenter_capacity_mw * 1000.0
        self.capacity_dc_w = self.capacity_dc_kw * 1000.0
        self.capacity_ac_kw = self.capacity_dc_kw / self.dc_ac_ratio if self.dc_ac_ratio else 0.0

        epw_path = Path(epw_filename)
        if not epw_path.is_absolute():
            epw_path = REPO_ROOT / "data" / "Weather" / epw_filename
        if not epw_path.is_file():
            raise ValueError(f"pv EPW file does not exist: {epw_path}")
        self.epw_path = epw_path
        self.hourly_weather, self.metadata = pvlib.iotools.read_epw(
            epw_path, coerce_year=2025
        )
        if len(self.hourly_weather) != 8760:
            raise ValueError(f"pv EPW must contain 8760 hourly records, got {len(self.hourly_weather)}")
        self._validate_epw_inputs()
        self.hourly_ac_kw = self._calculate_hourly_ac_kw()
        self.ac_kw = np.repeat(self.hourly_ac_kw, STEPS_PER_HOUR)
        if self.timezone_shift:
            self.ac_kw = np.roll(self.ac_kw, -self.timezone_shift * STEPS_PER_HOUR)
        self.original_ac_kw = self.ac_kw.copy()
        self.time_steps_day = 24 * STEPS_PER_HOUR
        self.time_step = 0
        self._current_ac_kwh = 0.0
        self._forecast_ac_kwh = np.zeros(8, dtype=np.float32)

    def _validate_epw_inputs(self):
        required = ["ghi", "dni", "dhi", "temp_air", "wind_speed", "atmospheric_pressure"]
        missing = set(required) - set(self.hourly_weather)
        if missing:
            raise ValueError(f"pv EPW missing required columns: {sorted(missing)}")
        for column in required:
            values = self.hourly_weather[column].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"pv EPW column {column} contains non-finite values")
        if (self.hourly_weather[["ghi", "dni", "dhi"]] < 0).any().any():
            raise ValueError("pv EPW irradiance values must be non-negative")

    def _calculate_hourly_ac_kw(self):
        if not self.enabled or self.capacity_dc_w == 0.0:
            return np.zeros(len(self.hourly_weather), dtype=float)
        location = pvlib.location.Location(
            latitude=self.metadata["latitude"],
            longitude=self.metadata["longitude"],
            tz=float(self.metadata["TZ"]),
            altitude=self.metadata["altitude"],
        )
        solar_position = location.get_solarposition(self.hourly_weather.index)
        airmass = pvlib.atmosphere.get_relative_airmass(solar_position["apparent_zenith"])
        poa = pvlib.irradiance.get_total_irradiance(
            surface_tilt=self.surface_tilt_deg,
            surface_azimuth=self.surface_azimuth_deg,
            solar_zenith=solar_position["apparent_zenith"],
            solar_azimuth=solar_position["azimuth"],
            dni=self.hourly_weather["dni"],
            ghi=self.hourly_weather["ghi"],
            dhi=self.hourly_weather["dhi"],
            dni_extra=pvlib.irradiance.get_extra_radiation(self.hourly_weather.index),
            airmass=airmass,
            model="perez",
        )["poa_global"].clip(lower=0.0)
        temperature_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS[
            "sapm"
        ][self.temperature_model]
        cell_temperature = pvlib.temperature.sapm_cell(
            poa, self.hourly_weather["temp_air"], self.hourly_weather["wind_speed"],
            **temperature_parameters,
        )
        # The EPW TMY may carry missing irradiance/meteorological fields in
        # individual intervals. Missing physical inputs become zero AC output
        # rather than contaminating the annual trace with NaN.
        poa = poa.fillna(0.0)
        cell_temperature = cell_temperature.fillna(self.hourly_weather["temp_air"])
        pdc_w = pvlib.pvsystem.pvwatts_dc(
            poa, cell_temperature, self.capacity_dc_w, self.gamma_pdc_per_deg_c
        ).fillna(0.0).clip(lower=0.0)
        pdc_after_losses_w = pdc_w * (1.0 - self.system_losses_fraction)
        pac_w = pvlib.inverter.pvwatts(
            pdc_after_losses_w,
            pdc0=self.capacity_dc_w / self.dc_ac_ratio,
            eta_inv_nom=self.inverter_efficiency,
        ).clip(lower=0.0)
        return pac_w.to_numpy(dtype=float) / 1000.0

    def reset(self, init_day=None, init_hour=None, future_steps=8):
        self.time_step = (
            (init_day if init_day is not None else self.init_day) * self.time_steps_day
            + (init_hour if init_hour is not None else 0) * STEPS_PER_HOUR
        )
        self._cache(future_steps)
        return self._current_ac_kwh, self._forecast_ac_kwh

    def step(self, future_steps=8):
        self.time_step += 1
        if self.time_step >= len(self.ac_kw):
            self.time_step = self.init_day * self.time_steps_day
        self._cache(future_steps)
        return self._current_ac_kwh, self._forecast_ac_kwh

    def _cache(self, future_steps):
        self._current_ac_kwh = float(self.ac_kw[self.time_step] * 0.25)
        self._forecast_ac_kwh = (_cyc(self.ac_kw, self.time_step + 1, future_steps) * 0.25).astype(np.float32)

    def get_current_ac_kwh(self):
        return self._current_ac_kwh

    def get_forecast_ac_kwh(self):
        return self._forecast_ac_kwh.copy()

    def get_current_ac_norm(self):
        return 0.0 if self.capacity_ac_kw == 0.0 else float(
            np.clip(self._current_ac_kwh / (self.capacity_ac_kw * 0.25), 0.0, 1.0)
        )

    def get_forecast_ac_norm(self):
        if self.capacity_ac_kw == 0.0:
            return np.zeros_like(self._forecast_ac_kwh)
        return np.clip(
            self._forecast_ac_kwh / (self.capacity_ac_kw * 0.25), 0.0, 1.0
        ).astype(np.float32)


def _finite_number(value, field):
    if isinstance(value, bool):
        raise ValueError(f"{field} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _finite_nonnegative(value, field):
    number = _finite_number(value, field)
    if number < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return number


def _finite_positive(value, field):
    number = _finite_nonnegative(value, field)
    if number <= 0:
        raise ValueError(f"{field} must be positive")
    return number


def _bounded(value, field, lower, upper, strict_lower=False):
    number = _finite_number(value, field)
    if number > upper or number < lower or (strict_lower and number <= lower):
        raise ValueError(f"{field} must be in the configured physical range")
    return number


def _integer(value, field):
    if isinstance(value, bool) or type(value) is not int:
        raise ValueError(f"{field} must be an integer")
    return value


def _validate_bool(value, field):
    if type(value) is not bool:
        raise ValueError(f"{field} must be a boolean")
    return value
