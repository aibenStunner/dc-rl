"""Manager of the Hydro-Quebec electricity tariff applicable to the
simulated facility.

NUMBERS: extracted directly from Hydro-Quebec's official "2026 Electricity
Rates" tariff booklet (effective April 1, 2026; the source PDF is
hq-electricity-rates.pdf).

ONE IMPORTANT DEVIATION FROM CI_Manager: carbon intensity is a purely
EXOGENOUS time series -- CI_Manager.step() takes no arguments and could in
principle be advanced before any agent acts. A tariff is not: which price
tier applies, and how large the demand-charge increment is, depends on the
facility's OWN metered energy this step. So PriceManager.step() takes the
step's metered energy as an argument, and MUST be called after the battery
agent has acted (i.e. after bat_env.step()), once
bat_info['bat_total_energy_with_battery_KWh'] is known for the step just
taken -- the same point in sustaindc_env.py's step() where CI_Manager.step()
is already called (right after _perform_actions()).

KNOWN LIMITATIONS (deliberately out of scope, not silently approximated):
  - The medium/high-voltage supply credit and transformation-loss
    adjustment (articles 12.2/12.4) are not modelled: both require a
    customer's specific supply voltage, which SustainDC has no concept of.
  - Rate M's winter ratchet (minimum billing demand = 65% of the max demand
    during the winter portion of the trailing 12 months) is tracked across
    episodes via `winter_peak_kw`, which accumulates on every winter step and
    is NOT cleared by reset(). The floor applied at reset is
    max(peak_carry_kw, demand_floor_kw, 0.65 * winter_peak_kw). This is a
    trailing-maximum approximation of the real 12-month window: the real rule
    ages peaks out after 12 months, whereas `winter_peak_kw` here only ever
    climbs. For training that is the conservative direction (the floor never
    silently drops), but it is not a faithful 12-month rolling window.
  - Episodes must actually VISIT winter for any of this to engage. See
    SustainDC's `sample_whole_year` config key. Consequently
    `TariffSpec.winter_ratchet_fraction` (0.65) is RECORDED BUT NOT APPLIED
    by any computation here -- the caller applies it when choosing
    `demand_floor_kw`.
  - `TariffSpec.minimum_bill_c_1phase` / `minimum_bill_c_3phase` ($15.426 /
    $46.278, article 4.2) are likewise RECORDED BUT NOT APPLIED: enforcing a
    minimum monthly bill needs an end-of-billing-period settlement step
    (compare the period's accumulated total against the minimum, charge the
    difference), which has no natural home in a per-step reward and is
    irrelevant at any realistic data-centre scale -- a 1 MW facility clears
    $46 of billing within the first few minutes of a month.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class TariffSpec:
    """One Hydro-Quebec rate's billable structure. All prices in CENTS
    (kept in the same unit CI_Manager-adjacent code favours -- avoids a
    silent $/cents mixup at the reward boundary)."""

    name: str
    citation: str                                  # human-readable article ref
    is_approved: bool                               # False only for the pending new rate

    # --- energy price, cents per kWh ---
    energy_tier1_c_per_kwh: float
    energy_tier2_c_per_kwh: Optional[float] = None   # None => no second tier (flat)
    tier2_threshold_kwh: Optional[float] = None      # monthly cumulative kWh where tier 2 starts

    # --- demand charge, cents per kW of billing demand, per month ---
    demand_charge_c_per_kw_month: float = 0.0
    min_billing_demand_kw: float = 0.0               # Rate L's "contract power" floor (>=5000)
    winter_ratchet_fraction: Optional[float] = None   # Rate M's 65% rule; None if N/A

    # --- minimum monthly bill, cents ---
    minimum_bill_c_1phase: float = 0.0
    minimum_bill_c_3phase: float = 0.0

    # --- Rate L's winter daily-overrun optimization charge (article 5.6) ---
    optimization_charge_c_per_kw_day: float = 0.0     # 0.0 => not applicable (Rate M / new rate)
    optimization_charge_cap_c_per_kw_month: float = 0.0
    optimization_overrun_fraction: float = 1.10       # overrun = demand exceeding 110% of contract power


# Article 4.2. Applies 50 kW <= min. billing demand < 5,000 kW (article 4.1).
RATE_M = TariffSpec(
    name="rate_m",
    citation="2026 Electricity Rates, Chapter 4, Section 1 (articles 4.1-4.8)",
    is_approved=True,
    energy_tier1_c_per_kwh=6.292,
    energy_tier2_c_per_kwh=4.666,
    tier2_threshold_kwh=210_000.0,
    demand_charge_c_per_kw_month=1824.2,             # $18.242/kW
    min_billing_demand_kw=0.0,                        # floor is the ratchet, not a fixed kW
    winter_ratchet_fraction=0.65,
    minimum_bill_c_1phase=1542.6,                     # $15.426
    minimum_bill_c_3phase=4627.8,                     # $46.278
)

# Article 5.2. Applies to an annual contract with min. billing demand >= 5,000 kW
# (article 5.1), principally industrial -- the rate this thesis's "large power"
# facilities sit on today, before/absent the proposed new data-centre rate below.
RATE_L = TariffSpec(
    name="rate_l",
    citation="2026 Electricity Rates, Chapter 5, Section 1 (articles 5.1-5.9)",
    is_approved=True,
    energy_tier1_c_per_kwh=3.821,                     # flat, no tier 2
    demand_charge_c_per_kw_month=1502.7,              # $15.027/kW
    min_billing_demand_kw=5_000.0,                    # "contract power" floor
    optimization_charge_c_per_kw_day=880.8,           # $8.808/kW, winter days only
    optimization_charge_cap_c_per_kw_month=2642.0,    # $26.420/kW monthly-equivalent cap
    optimization_overrun_fraction=1.10,
)

# Hydro-Quebec press release (news.hydroquebec.com, "Hydro-Quebec proposing to
# the Regie de l'energie a new rate for large data centres..."), fetched
# 2026-09-24. PENDING REGULATORY APPROVAL -- NOT YET IN FORCE, and Hydro-Quebec
# has not published a detailed rate structure (no confirmed demand-charge
# component, no confirmed tiers): only a rough "~13 cents/kWh average, roughly
# double current large-power rates" figure and a >=5 MW threshold. This is
# therefore an EXPLICIT APPROXIMATION -- a flat energy-only placeholder with
# zero demand charge -- not a citation-grade rate. Never selected by default;
# only reachable via an explicit `tariff_rate_override='new_dc_rate'`. Replace
# with the real structure the moment the Regie approves one.
NEW_DC_RATE_PLACEHOLDER = TariffSpec(
    name="new_dc_rate_placeholder",
    citation="Hydro-Quebec press release (proposed, pending Regie de l'energie "
             "approval, effective H2 2026) -- APPROXIMATE, not an official tariff",
    is_approved=False,
    energy_tier1_c_per_kwh=13.0,
    demand_charge_c_per_kw_month=0.0,
    min_billing_demand_kw=5_000.0,
)


def is_hq_winter_day(day_of_year: int) -> bool:
    """True if `day_of_year` (0-indexed, Jan 1 = 0) falls in HQ's winter
    period: Dec 1 - Mar 31 inclusive (definitions, "winter period"). Uses a
    plain 365-day non-leap calendar, consistent with the rest of this
    codebase's day-of-year handling (e.g. utils/utils_cf.py's get_init_day).
    Dec 1 = day 334; Mar 31 = day 89 (0-indexed)."""
    day_of_year = day_of_year % 365
    return day_of_year >= 334 or day_of_year <= 89


def select_tariff(datacenter_capacity_mw: float,
                   tariff_rate_override: Optional[str] = None) -> TariffSpec:
    """Pick a TariffSpec by the facility's configured power rating, mirroring
    Hydro-Quebec's own eligibility rule (article "power": medium power = min.
    billing demand < 5,000 kW -> Rate M; large power >= 5,000 kW -> Rate L),
    unless `tariff_rate_override` names a specific rate explicitly.

    `tariff_rate_override`: None (auto-select, default) | 'rate_m' | 'rate_l'
    | 'new_dc_rate' (the pending placeholder above -- never auto-selected).
    """
    by_name = {"rate_m": RATE_M, "rate_l": RATE_L, "new_dc_rate": NEW_DC_RATE_PLACEHOLDER}
    if tariff_rate_override is not None:
        if tariff_rate_override not in by_name:
            raise ValueError(
                f"tariff_rate_override={tariff_rate_override!r} must be one of "
                f"{sorted(by_name)} or None (auto-select)")
        return by_name[tariff_rate_override]

    capacity_kw = datacenter_capacity_mw * 1000.0
    if capacity_kw < 50.0:
        # HQ's small-power carve-out (demand billed only above 50 kW) has no
        # real analogue for a simulated data centre; Rate M is the closest
        # sane default and its own tier-1 energy price still applies.
        return RATE_M
    if capacity_kw < 5_000.0:
        return RATE_M
    return RATE_L


class PriceManager:
    """See module docstring for the exogenous-CI vs endogenous-tariff
    deviation from CI_Manager's contract.

    Args:
        datacenter_capacity_mw: facility's configured power rating (MW);
            drives rate auto-selection via select_tariff().
        tariff_rate_override: forces a specific rate; see select_tariff().
        init_day: kept for CI_Manager-call-site symmetry; unused (a tariff
            has no day-indexed data file to seek into).
        future_steps: length of the price forecast vector returned by
            reset()/step() -- Rate M/L have no intra-day time-of-use
            pattern (unlike the old Rate D TOU model), so the forecast is
            honestly flat at the current marginal price until the next
            tier crossing, not a placeholder.
        demand_floor_kw: minimum billing demand floor to seed the running
            peak with at reset() if the caller does not supply a
            `peak_carry_kw` (e.g. a config constant standing in for "last
            winter's ratcheted demand" -- see module docstring's KNOWN
            LIMITATIONS on the multi-month ratchet).
    """

    def __init__(self, datacenter_capacity_mw: float = 1.0,
                 tariff_rate_override: Optional[str] = None,
                 init_day: int = 0, future_steps: int = 8,
                 demand_floor_kw: float = 0.0, debug: bool = False):
        self.tariff = select_tariff(datacenter_capacity_mw, tariff_rate_override)
        self.datacenter_capacity_mw = datacenter_capacity_mw
        self.capacity_kw = datacenter_capacity_mw * 1000.0
        self.init_day = init_day
        self.future_steps = future_steps
        self.demand_floor_kw = demand_floor_kw
        self.debug = debug

        # 15-minute steps: matches HQ's own "maximum power demand" integration
        # period exactly (article 1.2/"maximum power demand" definition), so
        # no sub-stepping/averaging is needed -- one env step IS one window.
        self.timestep_per_hour = 4
        self.dt_hours = 1.0 / self.timestep_per_hour

        # A generous normalizer for the observation: the highest price this
        # tariff can ever actually charge per kWh (tier 2 if it has one, else
        # tier 1), so norm_price in [0, 1] under all reachable conditions.
        self._price_norm_denominator = (
            self.tariff.energy_tier2_c_per_kwh
            if self.tariff.energy_tier2_c_per_kwh is not None
            else self.tariff.energy_tier1_c_per_kwh)
        # A generous normalizer for the running-peak observation feature.
        self._peak_norm_denominator = max(self.capacity_kw, 1.0)

        # Winter-day optimization-charge bookkeeping (Rate L only; harmless
        # no-ops for Rate M / the new-rate placeholder, whose
        # optimization_charge_c_per_kw_day is 0.0).
        self._day_peak_kw = 0.0
        self._step_in_day = 0
        self._month_optimization_charge_c = 0.0
        # Survives reset(): Rate M's winter ratchet looks back over the
        # trailing 12 months, which spans many episodes.
        self.winter_peak_kw = 0.0

        self.reset(init_day=init_day, init_hour=0, peak_carry_kw=demand_floor_kw)

    # ------------------------------------------------------------------ #
    # internal helpers
    # ------------------------------------------------------------------ #

    def _energy_cost_c(self, prev_month_kwh: float, energy_kwh: float) -> float:
        """Exact cost (cents) of `energy_kwh` given `prev_month_kwh` already
        consumed this billing period -- correctly split across Rate M's
        210,000 kWh tier boundary if this step's energy straddles it. Flat
        (Rate L / new-rate placeholder) tariffs take the single-price path."""
        if self.tariff.tier2_threshold_kwh is None:
            return energy_kwh * self.tariff.energy_tier1_c_per_kwh
        tier1_remaining = max(0.0, self.tariff.tier2_threshold_kwh - prev_month_kwh)
        tier1_kwh = min(energy_kwh, tier1_remaining)
        tier2_kwh = energy_kwh - tier1_kwh
        return (tier1_kwh * self.tariff.energy_tier1_c_per_kwh
                + tier2_kwh * self.tariff.energy_tier2_c_per_kwh)

    def _marginal_price_c_per_kwh(self, month_kwh: float) -> float:
        """The price the NEXT kWh would be billed at, given `month_kwh`
        already consumed -- what the observation/forecast expose (a single
        scalar price signal), as distinct from _energy_cost_c's exact
        step-cost accounting used for the reward."""
        if (self.tariff.tier2_threshold_kwh is not None
                and month_kwh >= self.tariff.tier2_threshold_kwh):
            return self.tariff.energy_tier2_c_per_kwh
        return self.tariff.energy_tier1_c_per_kwh

    def _normalize_price(self, price_c_per_kwh: float) -> float:
        return float(np.clip(price_c_per_kwh / self._price_norm_denominator, 0.0, 1.0))

    def _optimization_charge_increment_c(self, metered_power_kw: float,
                                          is_winter: bool) -> float:
        """Rate L's article 5.6 winter daily-overrun charge, billed as a
        one-off spike at each day's close (24h = 96 steps at this dt), not
        smoothly per-step -- HQ bills it per calendar day, not per 15-min
        window. Monthly total is capped at the monthly-equivalent rate
        applied to the day's overrun, same as HQ's own cap wording. A no-op
        (returns 0.0) for Rate M / the new-rate placeholder, whose
        optimization_charge_c_per_kw_day is 0.0."""
        if self.tariff.optimization_charge_c_per_kw_day == 0.0:
            return 0.0
        self._day_peak_kw = max(self._day_peak_kw, metered_power_kw)
        self._step_in_day += 1
        steps_per_day = 24 * self.timestep_per_hour
        if self._step_in_day < steps_per_day:
            return 0.0
        # Day boundary: settle today's overrun charge, if any, then reset.
        self._step_in_day = 0
        day_peak, self._day_peak_kw = self._day_peak_kw, 0.0
        if not is_winter:
            return 0.0
        overrun_kw = max(0.0, day_peak - self.tariff.optimization_overrun_fraction * self._contract_power_kw)
        if overrun_kw <= 0.0:
            return 0.0
        charge_c = overrun_kw * self.tariff.optimization_charge_c_per_kw_day
        cap_c = self.tariff.optimization_charge_cap_c_per_kw_month * self._contract_power_kw
        room_left_c = max(0.0, cap_c - self._month_optimization_charge_c)
        charge_c = min(charge_c, room_left_c)
        self._month_optimization_charge_c += charge_c
        return charge_c

    # ------------------------------------------------------------------ #
    # public interface
    # ------------------------------------------------------------------ #

    def reset(self, init_day=None, init_hour=None, peak_carry_kw: float = 0.0):
        """Start a new billing period (episode). `peak_carry_kw`: the
        running peak to carry forward from the previous episode (see module
        docstring's KNOWN LIMITATIONS on the multi-month winter ratchet) --
        the caller is responsible for threading the previous episode's
        `ending_peak_kw` back in here if cross-episode ratchet behaviour is
        wanted; defaults to 0.0 (i.e. `demand_floor_kw` alone applies).

        Returns (current_price_norm, forecast_price_norm, current_price_denorm),
        matching CI_Manager.reset()'s 3-tuple shape.
        """
        self.month_energy_kwh = 0.0
        # Rate M's minimum billing demand (article 4.4): 65% of the maximum
        # demand recorded during a winter-period consumption period in the
        # trailing 12 months. `winter_peak_kw` accumulates across episodes
        # (it is NOT cleared here) precisely so this floor can be derived from
        # demand the facility actually drew in winter, rather than supplied by
        # hand. Before any winter step has been seen it is 0.0 and the floor
        # falls back to the configured `demand_floor_kw`.
        ratcheted_floor_kw = 0.0
        if self.tariff.winter_ratchet_fraction is not None:
            ratcheted_floor_kw = self.tariff.winter_ratchet_fraction * self.winter_peak_kw
        self.ratcheted_floor_kw = ratcheted_floor_kw
        self.peak_kw = max(peak_carry_kw, self.demand_floor_kw, ratcheted_floor_kw)
        # Fixed for the whole period: Rate L's "contract power" (article 5.3)
        # is a pre-set, negotiated quantity, not the running peak this same
        # period's own overrun is busy ratcheting up -- using self.peak_kw
        # here would let a day's overrun inflate the very threshold it's
        # measured against, silently erasing itself. self.peak_kw remains
        # correct for the BASE demand charge (article 5.4: billing demand is
        # never less than contract power, but IS the actual peak if higher).
        self._contract_power_kw = max(self.peak_kw, self.tariff.min_billing_demand_kw)
        self._demand_charge_increment_c = 0.0
        self._optimization_increment_c = 0.0
        self._day_peak_kw = 0.0
        self._step_in_day = 0
        self._month_optimization_charge_c = 0.0

        price_denorm = self._marginal_price_c_per_kwh(self.month_energy_kwh)
        price_norm = self._normalize_price(price_denorm)
        forecast = np.full(self.future_steps, price_norm, dtype=np.float32)
        self._current_price_norm = price_norm
        self._forecast_price_norm = forecast
        self._current_price_denorm = price_denorm
        return price_norm, forecast, price_denorm

    def step(self, metered_energy_kwh: float = 0.0, is_winter: bool = False):
        """Advance one 15-minute step given the facility's ACTUAL metered
        energy this step (kWh, always >= 0 -- see module docstring; pass
        bat_info['bat_total_energy_with_battery_KWh'], called after
        bat_env.step()). `is_winter`: whether this step falls in HQ's winter
        period (Dec 1 - Mar 31). Drives BOTH Rate L's optimization charge
        and Rate M's winter-ratchet tracking (article 4.4).

        Returns the same 3-tuple shape as reset()/CI_Manager.step().
        """
        metered_power_kw = metered_energy_kwh / self.dt_hours

        # Feed Rate M's winter ratchet (article 4.4). Tracked on every winter
        # step and deliberately NOT reset between episodes, so next period's
        # minimum billing demand reflects this winter's actual maximum.
        if is_winter:
            self.winter_peak_kw = max(self.winter_peak_kw, metered_power_kw)

        prev_peak = self.peak_kw
        self.peak_kw = max(self.peak_kw, metered_power_kw)
        self._demand_charge_increment_c = (
            self.tariff.demand_charge_c_per_kw_month * (self.peak_kw - prev_peak))

        self._optimization_increment_c = self._optimization_charge_increment_c(
            metered_power_kw, is_winter)

        self._energy_cost_this_step_c = self._energy_cost_c(
            self.month_energy_kwh, metered_energy_kwh)
        self.month_energy_kwh += metered_energy_kwh

        price_denorm = self._marginal_price_c_per_kwh(self.month_energy_kwh)
        price_norm = self._normalize_price(price_denorm)
        forecast = np.full(self.future_steps, price_norm, dtype=np.float32)
        self._current_price_norm = price_norm
        self._forecast_price_norm = forecast
        self._current_price_denorm = price_denorm
        return price_norm, forecast, price_denorm

    def get_current_price(self) -> float:
        return self._current_price_norm

    def get_forecast_price(self) -> np.ndarray:
        return self._forecast_price_norm

    def get_current_price_denorm(self) -> float:
        """The marginal price (cents/kWh) the NEXT kWh would be billed at."""
        return self._current_price_denorm

    def get_tier2_progress_fraction(self) -> float:
        """How far this billing period's cumulative energy is toward the
        tier-2 threshold, in [0, 1] -- lets an agent anticipate a coming
        price drop (Rate M's cheaper tier). 0.0 for flat tariffs (Rate L /
        the new-rate placeholder), which have nothing to anticipate."""
        if self.tariff.tier2_threshold_kwh is None:
            return 0.0
        return float(np.clip(self.month_energy_kwh / self.tariff.tier2_threshold_kwh, 0.0, 1.0))

    def get_normalized_peak(self) -> float:
        """This period's running peak demand so far, normalized by the
        facility's configured capacity -- roadmap item 3's "expose monthly
        peak demand as separate state from day 1"."""
        return float(np.clip(self.peak_kw / self._peak_norm_denominator, 0.0, 1.0))

    def get_demand_charge_increment_c(self) -> float:
        """This step's demand-charge cost (cents) -- 0.0 unless this step's
        power set a new running-peak record, in which case it's
        `demand_charge_c_per_kw_month * (new_peak - old_peak)`. Sums to the
        full monthly demand charge exactly once (at whichever step set the
        final peak)."""
        return self._demand_charge_increment_c

    def get_optimization_charge_increment_c(self) -> float:
        """Rate L's winter daily-overrun charge, billed as a spike at each
        day's close; 0.0 every other step and always 0.0 for Rate M / the
        new-rate placeholder."""
        return self._optimization_increment_c

    def get_energy_cost_this_step_c(self) -> float:
        """Exact tier-aware cost (cents) of the energy metered THIS step --
        the quantity a cost-based reward should actually use, distinct from
        `get_current_price()`'s single marginal-price signal."""
        return self._energy_cost_this_step_c

    @property
    def ending_peak_kw(self) -> float:
        """What the caller should persist forward as the next episode's
        `peak_carry_kw`, if cross-episode ratchet carry-over is wanted."""
        return self.peak_kw
