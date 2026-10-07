"""Correctness tests for utils/price_manager.py -- Hydro-Quebec Rate M,
Rate L, and the proposed large-data-centre rate placeholder.

Run:  python3 tests/test_price_manager.py     (this file alone)
      python3 tests/run_all.py                 (the whole suite)
Works under pytest too if it is ever installed.
"""
import sys
from pathlib import Path

# Make the repo root importable no matter how this file is invoked
# (`python3 tests/test_price_manager.py` puts tests/ on sys.path, not the
# repo root, so `from utils...` would fail without this).
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from tests._harness import run_module_tests
from utils.price_manager import (
    RATE_L, RATE_M, NEW_DC_RATE_PLACEHOLDER, PriceManager, is_hq_winter_day,
    select_tariff,
)


def test_is_hq_winter_day_boundaries():
    assert is_hq_winter_day(334) is True     # Dec 1
    assert is_hq_winter_day(364) is True     # Dec 31
    assert is_hq_winter_day(0) is True        # Jan 1
    assert is_hq_winter_day(89) is True       # Mar 31
    assert is_hq_winter_day(333) is False      # Nov 30
    assert is_hq_winter_day(90) is False       # Apr 1
    assert is_hq_winter_day(180) is False       # mid-summer


# ---------------------------------------------------------- rate selection --

def test_rate_numbers_match_the_official_tariff_pdf():
    """Pin the exact cited figures so a future edit can't silently drift."""
    assert RATE_M.demand_charge_c_per_kw_month == 1824.2       # $18.242/kW
    assert RATE_M.energy_tier1_c_per_kwh == 6.292
    assert RATE_M.energy_tier2_c_per_kwh == 4.666
    assert RATE_M.tier2_threshold_kwh == 210_000.0
    assert RATE_M.minimum_bill_c_1phase == 1542.6               # $15.426
    assert RATE_M.minimum_bill_c_3phase == 4627.8               # $46.278
    assert RATE_M.winter_ratchet_fraction == 0.65

    assert RATE_L.demand_charge_c_per_kw_month == 1502.7        # $15.027/kW
    assert RATE_L.energy_tier1_c_per_kwh == 3.821
    assert RATE_L.energy_tier2_c_per_kwh is None                # flat, no tiers
    assert RATE_L.min_billing_demand_kw == 5_000.0
    assert RATE_L.optimization_charge_c_per_kw_day == 880.8     # $8.808/kW-day
    assert RATE_L.optimization_charge_cap_c_per_kw_month == 2642.0  # $26.420/kW

    assert NEW_DC_RATE_PLACEHOLDER.is_approved is False
    assert RATE_M.is_approved and RATE_L.is_approved


def test_auto_selection_thresholds():
    assert select_tariff(datacenter_capacity_mw=0.001).name == "rate_m"   # 1 kW
    assert select_tariff(datacenter_capacity_mw=0.049).name == "rate_m"   # 49 kW
    assert select_tariff(datacenter_capacity_mw=0.050).name == "rate_m"   # exactly 50 kW
    assert select_tariff(datacenter_capacity_mw=1.0).name == "rate_m"     # 1 MW
    assert select_tariff(datacenter_capacity_mw=4.999).name == "rate_m"   # just under 5 MW
    assert select_tariff(datacenter_capacity_mw=5.0).name == "rate_l"     # exactly 5 MW
    assert select_tariff(datacenter_capacity_mw=20.0).name == "rate_l"    # thesis's target scale


def test_new_dc_rate_never_auto_selected():
    for mw in (0.01, 1.0, 5.0, 20.0, 1000.0):
        assert select_tariff(datacenter_capacity_mw=mw).name != "new_dc_rate_placeholder"
    assert select_tariff(datacenter_capacity_mw=20.0,
                         tariff_rate_override="new_dc_rate").name == "new_dc_rate_placeholder"


def test_override_rejects_unknown_name():
    try:
        select_tariff(datacenter_capacity_mw=1.0, tariff_rate_override="rate_x")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown tariff_rate_override should raise ValueError")


# -------------------------------------------------------------- energy tiers --

def test_rate_m_tier_boundary_hand_computed():
    """A single step whose energy exactly straddles the 210,000 kWh
    boundary must split cost precisely across both tiers."""
    pm = PriceManager(datacenter_capacity_mw=5.0, tariff_rate_override="rate_m")
    pm.reset(peak_carry_kw=0.0)
    # Get to 209,900 kWh consumed so far this month (100 kWh short of tier 2).
    pm.month_energy_kwh = 209_900.0
    pm.step(metered_energy_kwh=200.0)   # 100 kWh at tier 1, 100 kWh at tier 2
    expected_c = 100.0 * RATE_M.energy_tier1_c_per_kwh + 100.0 * RATE_M.energy_tier2_c_per_kwh
    assert abs(pm.get_energy_cost_this_step_c() - expected_c) < 1e-9
    assert pm.month_energy_kwh == 210_100.0


def test_rate_m_tier2_is_reached_at_thesis_scale():
    """At the roadmap's target 5-20 MW scale, tier 2 must be reached WITHIN
    a single ~30-day episode -- confirms tier 2 is exercised at the
    configured thesis-scale facility rather than remaining dead code."""
    pm = PriceManager(datacenter_capacity_mw=5.0, tariff_rate_override="rate_m")
    pm.reset()
    steps_per_day = 96
    for _ in range(10 * steps_per_day):    # 10 days at a steady 5 MW draw
        pm.step(metered_energy_kwh=5_000.0 * 0.25)   # 1,250 kWh/step
    assert pm.month_energy_kwh > RATE_M.tier2_threshold_kwh
    assert pm.get_current_price() == pm._normalize_price(RATE_M.energy_tier2_c_per_kwh)


def test_rate_l_is_flat_no_tier_crossing_effect():
    pm = PriceManager(datacenter_capacity_mw=20.0, tariff_rate_override="rate_l")
    pm.reset()
    pm.step(metered_energy_kwh=1_000_000.0)   # absurdly large, would cross any tier
    expected_c = 1_000_000.0 * RATE_L.energy_tier1_c_per_kwh
    assert abs(pm.get_energy_cost_this_step_c() - expected_c) < 1e-6


# ------------------------------------------------------------- demand charge --

def test_demand_charge_bills_only_the_running_peak_increment():
    """Feed a non-monotonic power trace; total demand charge must equal
    rate * final_peak, NOT rate * sum-of-every-step's-power."""
    pm = PriceManager(datacenter_capacity_mw=1.0, tariff_rate_override="rate_m")
    pm.reset(peak_carry_kw=0.0)
    dt_h = 0.25
    power_trace_kw = [100.0, 400.0, 250.0, 900.0, 300.0, 900.0, 50.0]  # peak hit twice
    total_increment_c = 0.0
    for p_kw in power_trace_kw:
        pm.step(metered_energy_kwh=p_kw * dt_h)
        total_increment_c += pm.get_demand_charge_increment_c()
    expected_c = RATE_M.demand_charge_c_per_kw_month * max(power_trace_kw)
    assert abs(total_increment_c - expected_c) < 1e-6
    assert pm.ending_peak_kw == max(power_trace_kw)
    # re-hitting the same peak (two 900s) must NOT double-bill it
    assert total_increment_c < RATE_M.demand_charge_c_per_kw_month * sum(power_trace_kw)


def test_winter_ratchet_floor_is_65pct_of_supplied_winter_max():
    winter_max_kw = 2_000.0
    floor_kw = RATE_M.winter_ratchet_fraction * winter_max_kw
    pm = PriceManager(datacenter_capacity_mw=3.0, tariff_rate_override="rate_m",
                      demand_floor_kw=floor_kw)
    pm.reset(peak_carry_kw=0.0)   # no carry-over; floor alone must still apply
    assert pm.peak_kw == floor_kw
    # drawing below the floor must not lower the billed peak
    pm.step(metered_energy_kwh=10.0 * 0.25)   # 10 kW, far below the floor
    assert pm.peak_kw == floor_kw
    assert pm.get_demand_charge_increment_c() == 0.0


def test_peak_carries_over_when_caller_threads_it_through():
    pm = PriceManager(datacenter_capacity_mw=1.0, tariff_rate_override="rate_m")
    pm.reset(peak_carry_kw=0.0)
    pm.step(metered_energy_kwh=800.0 * 0.25)
    ending_peak = pm.ending_peak_kw
    assert ending_peak == 800.0
    pm.reset(peak_carry_kw=ending_peak)   # simulating the next episode
    assert pm.peak_kw == ending_peak
    pm.step(metered_energy_kwh=10.0 * 0.25)   # a quiet step in the new episode
    assert pm.get_demand_charge_increment_c() == 0.0   # peak already stood


# ----------------------------------------------------- Rate L optimization --

def test_rate_l_optimization_charge_only_fires_on_winter_overrun():
    pm = PriceManager(datacenter_capacity_mw=20.0, tariff_rate_override="rate_l")
    pm.reset(peak_carry_kw=6_000.0)   # contract power floor for this test
    steps_per_day = 96
    total_opt_c = 0.0
    # A summer day well above 110% of contract power must NOT be billed.
    for i in range(steps_per_day):
        pm.step(metered_energy_kwh=7_000.0 * 0.25, is_winter=False)
        total_opt_c += pm.get_optimization_charge_increment_c()
    assert total_opt_c == 0.0
    # The same overrun on a winter day must be billed at day's close.
    for i in range(steps_per_day):
        pm.step(metered_energy_kwh=7_000.0 * 0.25, is_winter=True)
        total_opt_c += pm.get_optimization_charge_increment_c()
    overrun_kw = 7_000.0 - 1.10 * 6_000.0
    expected_c = overrun_kw * RATE_L.optimization_charge_c_per_kw_day
    assert abs(total_opt_c - expected_c) < 1e-6


def test_winter_ratchet_tracks_observed_winter_peak():
    """Rate M article 4.4: the minimum billing demand for a period is 65% of
    the max demand recorded in winter. That floor must be DERIVED from winter
    demand the facility actually drew, not supplied by hand."""
    pm = PriceManager(datacenter_capacity_mw=2.0, tariff_rate_override="rate_m")
    pm.reset(peak_carry_kw=0.0)
    assert pm.peak_kw == 0.0                      # nothing observed yet
    for _ in range(96):                            # one winter day at 1200 kW
        pm.step(metered_energy_kwh=1200.0 * 0.25, is_winter=True)
    assert pm.winter_peak_kw == 1200.0

    pm.reset(peak_carry_kw=0.0)                    # next period, nothing carried
    assert abs(pm.peak_kw - 0.65 * 1200.0) < 1e-9
    assert abs(pm.ratcheted_floor_kw - 780.0) < 1e-9
    # a quiet step must be billed against the floor, not against zero
    pm.step(metered_energy_kwh=10.0 * 0.25, is_winter=False)
    assert pm.get_demand_charge_increment_c() == 0.0
    assert pm.peak_kw == 780.0


def test_summer_demand_does_not_feed_the_winter_ratchet():
    """Only winter steps may raise winter_peak_kw (article 4.4 is explicitly
    scoped to the winter period)."""
    pm = PriceManager(datacenter_capacity_mw=2.0, tariff_rate_override="rate_m")
    pm.reset(peak_carry_kw=0.0)
    for _ in range(96):
        pm.step(metered_energy_kwh=1800.0 * 0.25, is_winter=False)
    assert pm.winter_peak_kw == 0.0
    pm.reset(peak_carry_kw=0.0)
    assert pm.peak_kw == 0.0                       # no winter -> no ratchet floor


def test_winter_peak_survives_reset_and_only_climbs():
    """The trailing-12-month window spans episodes, so winter_peak_kw must
    NOT be cleared by reset(), and a milder winter must not lower it."""
    pm = PriceManager(datacenter_capacity_mw=2.0, tariff_rate_override="rate_m")
    pm.reset(peak_carry_kw=0.0)
    pm.step(metered_energy_kwh=1500.0 * 0.25, is_winter=True)
    assert pm.winter_peak_kw == 1500.0
    pm.reset(peak_carry_kw=0.0)
    assert pm.winter_peak_kw == 1500.0             # survived the reset
    pm.step(metered_energy_kwh=400.0 * 0.25, is_winter=True)   # milder winter
    assert pm.winter_peak_kw == 1500.0             # ratchets never fall


def test_rate_l_has_no_winter_ratchet():
    """Rate L's floor is its negotiated contract power, not a 65% ratchet."""
    pm = PriceManager(datacenter_capacity_mw=20.0, tariff_rate_override="rate_l")
    pm.reset(peak_carry_kw=0.0)
    for _ in range(96):
        pm.step(metered_energy_kwh=9000.0 * 0.25, is_winter=True)
    pm.reset(peak_carry_kw=0.0)
    assert pm.ratcheted_floor_kw == 0.0
    assert RATE_L.winter_ratchet_fraction is None


def test_rate_m_never_bills_optimization_charge():
    pm = PriceManager(datacenter_capacity_mw=1.0, tariff_rate_override="rate_m")
    pm.reset(peak_carry_kw=0.0)
    for _ in range(96):
        pm.step(metered_energy_kwh=900.0 * 0.25, is_winter=True)
        assert pm.get_optimization_charge_increment_c() == 0.0


# ------------------------------------------------------------------ shapes --

def test_reset_and_step_return_shapes_match_ci_manager_contract():
    pm = PriceManager(datacenter_capacity_mw=1.0, future_steps=8)
    norm, forecast, denorm = pm.reset()
    assert isinstance(norm, float) and 0.0 <= norm <= 1.0
    assert isinstance(forecast, np.ndarray) and forecast.shape == (8,)
    assert isinstance(denorm, float)
    norm2, forecast2, denorm2 = pm.step(metered_energy_kwh=50.0)
    assert forecast2.shape == (8,)
    assert pm.get_current_price() == norm2
    assert np.array_equal(pm.get_forecast_price(), forecast2)


if __name__ == "__main__":
    sys.exit(1 if run_module_tests(globals()) else 0)
