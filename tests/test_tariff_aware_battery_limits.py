"""Opt-in tariff-aware battery limits: recharge cap and declared discharge floor.

The recharge cap is the grid-import level up to which battery charging adds
no tariff charge in the interval about to be billed. Under winter Rate L it
also respects the daily 110 % contract-power threshold until the month's
optimization charge reaches its cap. The discharge floor is a user-declared
expected monthly peak; it is never inferred.
"""
import copy
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from harl.utils.configs_tools import get_defaults_yaml_args
from sustaindc_env import EnvConfig, SustainDC
from tests._harness import run_module_tests
from utils.managers.pricing.contracts import (
    MeterReading,
    PricingClock,
    PricingContext,
    PricingState,
)
from utils.managers.pricing.loader import create_pricing_model, load_pricing_config
from utils.managers.pricing.manager import PriceManager
from utils.managers.pricing.models.hydro_quebec import HydroQuebecPricingModel


_HYDRO_CONFIG = _REPO_ROOT / "data" / "Pricing" / "hydro_quebec_2026.yaml"
_FLAT_CONFIG = _REPO_ROOT / "data" / "Pricing" / "flat_example.yaml"
_RATE_L_CAP_C_PER_KW_MONTH = 2642.0
_RATE_L_DAILY_C_PER_KW = 880.8


def _pricing_config(tariff):
    return load_pricing_config(
        {"pricing": {"model": "hydro_quebec", "config_file": str(_HYDRO_CONFIG),
                     "options": {"tariff": tariff, "demand_floor_kw": 0.0}}},
        _REPO_ROOT,
    )


def _context(capacity_mw):
    return PricingContext(datacenter_capacity_mw=capacity_mw,
                          timestep_minutes=15, future_steps=8)


def _model(tariff, capacity_mw):
    return create_pricing_model(_pricing_config(tariff)), _context(capacity_mw)


def _reset(model, context, tariff, day):
    state = PricingState()
    model.reset(context=context, state=state, clock=PricingClock(day, 0.0),
                carry_state=None, options={"tariff": tariff, "demand_floor_kw": 0.0})
    return state


def _feed(model, context, state, day, hour, kw):
    model.step(context=context, state=state, clock=PricingClock(day, hour),
               reading=MeterReading(energy_kwh=kw * 0.25, demand_kw=kw))


def _feed_day(model, context, state, day, kw):
    for quarter in range(96):
        _feed(model, context, state, day, quarter * 0.25, kw)


def _billing(state):
    return copy.deepcopy((state.billing_period_energy_kwh, state.running_peak_kw,
                          state.model_state))


def _limit(model, context, state, day, hour):
    before = _billing(state)
    limit = model.charge_import_limit_kw(context=context, state=state,
                                         clock=PricingClock(day, hour))
    assert _billing(state) == before, "charge_import_limit_kw mutated billing state"
    return limit


# --- Pricing model -----------------------------------------------------------

def test_winter_rate_l_cap_is_the_daily_threshold_until_the_day_peak_exceeds_it():
    model, context = _model("rate_l", 10.0)
    state = _reset(model, context, "rate_l", day=0)
    assert state.model_state["contract_power_kw"] == 5000.0
    _feed_day(model, context, state, 0, 7000.0)       # settles at its last interval
    assert state.running_peak_kw == 7000.0
    assert state.model_state["pending_day_key"] is None
    assert math.isclose(_limit(model, context, state, 1, 0.0), 5500.0)
    _feed(model, context, state, 1, 0.0, 5200.0)
    assert math.isclose(_limit(model, context, state, 1, 0.25), 5500.0)
    _feed(model, context, state, 1, 0.25, 6200.0)
    assert math.isclose(_limit(model, context, state, 1, 0.5), 6200.0)
    assert state.running_peak_kw == 7000.0


def test_rate_l_cap_at_reset_is_the_minimum_billing_demand():
    model, context = _model("rate_l", 10.0)
    state = _reset(model, context, "rate_l", day=0)
    assert math.isclose(_limit(model, context, state, 0, 0.0), 5000.0)


def test_non_winter_rate_l_cap_is_the_running_peak():
    model, context = _model("rate_l", 10.0)
    state = _reset(model, context, "rate_l", day=100)
    _feed_day(model, context, state, 100, 7000.0)
    assert math.isclose(_limit(model, context, state, 101, 0.0), 7000.0)


def test_rate_m_cap_is_the_running_peak():
    model, context = _model("rate_m", 2.0)
    state = _reset(model, context, "rate_m", day=0)
    assert _limit(model, context, state, 0, 0.0) == 0.0
    _feed(model, context, state, 0, 0.0, 800.0)
    assert math.isclose(_limit(model, context, state, 0, 0.25), 800.0)


def test_rate_l_cap_is_the_running_peak_once_the_monthly_cap_is_reached():
    model, context = _model("rate_l", 30.0)
    state = _reset(model, context, "rate_l", day=0)
    _feed_day(model, context, state, 0, 25000.0)
    assert math.isclose(state.model_state["month_optimization_charge_c"],
                        _RATE_L_CAP_C_PER_KW_MONTH * 5000.0)
    assert math.isclose(_limit(model, context, state, 1, 0.0), 25000.0)


def test_rate_l_cap_counts_the_pending_day_toward_the_monthly_cap():
    model, context = _model("rate_l", 15.0)
    state = _reset(model, context, "rate_l", day=0)
    _feed_day(model, context, state, 0, 12000.0)
    _feed_day(model, context, state, 1, 12000.0)
    settled = state.model_state["month_optimization_charge_c"]
    assert math.isclose(settled, 2 * 6500.0 * _RATE_L_DAILY_C_PER_KW)
    cap_c = _RATE_L_CAP_C_PER_KW_MONTH * 5000.0
    _feed(model, context, state, 2, 0.0, 7000.0)
    # Settled plus today's 1500 kW overrun is still below the monthly cap.
    assert settled + 1500.0 * _RATE_L_DAILY_C_PER_KW < cap_c
    assert math.isclose(_limit(model, context, state, 2, 0.25), 7000.0)
    _feed(model, context, state, 2, 0.25, 8000.0)
    # Today's 2500 kW overrun exhausts the cap, so more overrun today is free.
    assert settled + 2500.0 * _RATE_L_DAILY_C_PER_KW >= cap_c
    assert math.isclose(_limit(model, context, state, 2, 0.5), 12000.0)


def _assert_cap_is_next_month_opening_peak(tariff, capacity_mw, kw, expected):
    model, context = _model(tariff, capacity_mw)
    state = _reset(model, context, tariff, day=30)
    _feed_day(model, context, state, 30, kw)
    limit = _limit(model, context, state, 31, 0.0)
    opened = copy.deepcopy(state)
    _feed(model, context, opened, 31, 0.0, 0.0)
    assert math.isclose(limit, opened.running_peak_kw), (tariff, limit, opened.running_peak_kw)
    assert math.isclose(limit, expected), (tariff, limit, expected)


def test_month_boundary_cap_uses_the_next_month_opening_peak():
    _assert_cap_is_next_month_opening_peak("rate_l", 10.0, 7000.0, 5000.0)
    _assert_cap_is_next_month_opening_peak("rate_m", 2.0, 800.0, 0.65 * 800.0)


def test_cap_rejects_an_unreset_model_and_an_invalid_clock():
    model, context = _model("rate_l", 10.0)
    for call in (
        lambda: model.charge_import_limit_kw(context=context, state=PricingState(),
                                             clock=PricingClock(0, 0.0)),
        lambda: model.charge_import_limit_kw(
            context=context, state=_reset(model, context, "rate_l", 0),
            clock=PricingClock(0, 24.0)),
    ):
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError("accepted an invalid charge-limit call")


# --- Price manager -----------------------------------------------------------

def _manager(model=None):
    config = _pricing_config("rate_l")
    if model is None:
        return PriceManager.from_config(config, datacenter_capacity_mw=10.0,
                                        future_steps=8, timestep_minutes=15)
    return PriceManager(pricing_config=config, model=model, context=_context(10.0))


def _run_partial_day(manager, kw=7000.0):
    """Reset at noon on day 0 and bill to midnight as the environment does."""
    manager.reset(init_day=0, init_hour=12.0)
    for quarter in range(48, 96):
        manager.step(metered_energy_kwh=kw * 0.25, day_of_year=0, hour=quarter * 0.25)
        day, hour = (1, 0.0) if quarter == 95 else (0, (quarter + 1) * 0.25)
        manager.observe_decision_clock(day_of_year=day, hour=hour)


def test_manager_cap_uses_the_decision_clock():
    manager = _manager()
    _run_partial_day(manager)
    # The noon-started day 0 is still pending. At its billed clock its 7000 kW
    # would count as today's peak, but the next decision interval is day 1.
    assert manager.state.model_state["pending_day_key"] == 0
    before = _billing(manager.state)
    assert math.isclose(manager.get_charge_import_limit_kw(), 5500.0)
    assert _billing(manager.state) == before


def test_manager_cap_requires_reset():
    try:
        _manager().get_charge_import_limit_kw()
    except RuntimeError:
        pass
    else:
        raise AssertionError("charge limit available before reset")


class _NoLimitModel(HydroQuebecPricingModel):
    charge_import_limit_kw = None


def test_manager_falls_back_to_the_running_peak_without_a_model_method():
    manager = _manager(model=_NoLimitModel(_pricing_config("rate_l").parameters))
    _run_partial_day(manager)
    assert manager.get_charge_import_limit_kw() == manager.state.running_peak_kw == 7000.0
    flat = PriceManager.from_config(
        load_pricing_config({"pricing": {"model": "flat",
                                         "config_file": str(_FLAT_CONFIG)}}, _REPO_ROOT),
        datacenter_capacity_mw=1.0, future_steps=8, timestep_minutes=15)
    flat.reset(init_day=0, init_hour=0.0)
    assert flat.get_charge_import_limit_kw() == flat.state.running_peak_kw


def _returning(value):
    class _FixedValue(HydroQuebecPricingModel):
        def charge_import_limit_kw(self, *, context, state, clock):
            return value
    return _FixedValue(_pricing_config("rate_l").parameters)


class _MutatingModel(HydroQuebecPricingModel):
    def charge_import_limit_kw(self, *, context, state, clock):
        state.running_peak_kw += 1.0
        state.model_state["day_peak_kw"] = 1.0
        return 5500.0


def test_manager_rejects_invalid_or_mutating_cap_models():
    for value in (-1.0, float("nan"), float("inf"), True, "5500", None):
        manager = _manager(model=_returning(value))
        manager.reset(init_day=0, init_hour=0.0)
        try:
            manager.get_charge_import_limit_kw()
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted charge limit {value!r}")
    manager = _manager(model=_MutatingModel(_pricing_config("rate_l").parameters))
    _run_partial_day(manager)
    before = _billing(manager.state)
    try:
        manager.get_charge_import_limit_kw()
    except ValueError:
        pass
    else:
        raise AssertionError("accepted a charge-limit call that mutated billing state")
    assert _billing(manager.state) == before


# --- Environment configuration -----------------------------------------------

def _rejects(battery, message):
    try:
        EnvConfig({"battery": battery})
    except ValueError:
        return
    raise AssertionError(message)


def test_env_config_defaults_keep_existing_behavior():
    battery = EnvConfig({})["battery"]
    assert battery["peak_charge_headroom_basis"] == "running_peak"
    assert battery["discharge_peak_floor_kw"] is None


def test_env_config_validates_the_headroom_basis():
    for basis in (None, 1, True, "tariff_aware", ""):
        _rejects({"peak_charge_headroom": True, "peak_charge_headroom_basis": basis},
                 f"accepted peak_charge_headroom_basis {basis!r}")
    _rejects({"peak_charge_headroom": False, "peak_charge_headroom_basis": "tariff"},
             "accepted a tariff basis with peak_charge_headroom off")
    config = EnvConfig({"battery": {"peak_charge_headroom": True,
                                    "peak_charge_headroom_basis": "tariff"}})
    assert config["battery"]["peak_charge_headroom_basis"] == "tariff"


def test_env_config_validates_the_discharge_floor():
    for floor in (-1.0, float("nan"), float("inf"), True, "600"):
        _rejects({"discharge_mode": "peak_limit", "discharge_peak_floor_kw": floor},
                 f"accepted discharge_peak_floor_kw {floor!r}")
    _rejects({"discharge_mode": "fixed_rate", "discharge_peak_floor_kw": 600.0},
             "accepted a discharge floor without peak_limit")
    for floor in (0, 600, 600.0):
        config = EnvConfig({"battery": {"discharge_mode": "peak_limit",
                                        "discharge_peak_floor_kw": floor}})
        assert config["battery"]["discharge_peak_floor_kw"] == floor


def test_shipped_yaml_selects_the_validated_10mw_pilot_scenario():
    algo_args, env_args = get_defaults_yaml_args("happo", "sustaindc")
    assert env_args["datacenter_capacity_mw"] == 10
    assert env_args["battery"]["peak_charge_headroom_basis"] == "tariff"
    assert env_args["battery"]["discharge_mode"] == "peak_limit"
    assert env_args["battery"]["discharge_peak_floor_kw"] == 6310.0
    assert algo_args["eval"]["fixed_month_validation"]["enabled"] is True
    assert algo_args["train"]["num_env_steps"] == 100000


# --- Environment routing -----------------------------------------------------

def test_env_routes_the_tariff_cap_to_charging():
    # 20 MW January on Rate L: idle on day 0 sets a running peak above the
    # 5500 kW threshold, then charging on day 1 meets the daily cap.
    env = SustainDC(EnvConfig({
        "month": 0, "episode_start_policy": "valid_midnight",
        "datacenter_capacity_mw": 20, "facility_sizing": {"mode": "scaling"},
        "battery": {"initial_soc": 0.0, "peak_charge_headroom": True,
                    "peak_charge_headroom_basis": "tariff"},
    }))
    env.reset(seed=102, options={"init_day": 0, "init_hour": 0})
    assert env.price_m.state.model_state["tariff"] == "rate_l"
    below_peak = 0
    for step in range(192):
        expected = env.price_m.get_charge_import_limit_kw()
        prior_peak = env.price_m.state.running_peak_kw
        battery_action = 2 if step < 96 else 0
        _, _, _, _, info = env.step({"agent_ls": 1, "agent_dc": 1,
                                     "agent_bat": battery_action})
        common = info["__common__"]
        assert env.bat_env.grid_import_peak_kw == expected
        assert expected <= prior_peak
        below_peak += expected < prior_peak
        base_net_kw = common["dc_total_power_kW"] - 4 * common["pv_ac_kwh"]
        if base_net_kw <= expected:
            assert 4 * common["bat_grid_import_kwh"] <= expected + 1e-6
    assert below_peak > 0


def test_env_default_basis_keeps_the_running_peak():
    env = SustainDC(EnvConfig({
        "month": 0, "episode_start_policy": "valid_midnight",
        "datacenter_capacity_mw": 10, "facility_sizing": {"mode": "scaling"},
        "battery": {"initial_soc": 0.0, "peak_charge_headroom": True},
    }))
    env.reset(seed=102, options={"init_day": 0, "init_hour": 0})
    for _ in range(8):
        prior_peak = env.price_m.state.running_peak_kw
        env.step({"agent_ls": 1, "agent_dc": 1, "agent_bat": 0})
        assert env.bat_env.grid_import_peak_kw == prior_peak


def test_env_applies_the_declared_discharge_floor():
    env = SustainDC(EnvConfig({
        "month": 6, "episode_start_policy": "valid_midnight",
        "battery": {"initial_soc": 0.5, "discharge_mode": "peak_limit",
                    "discharge_peak_floor_kw": 600.0},
    }))
    env.reset(seed=102, options={"init_day": 14, "init_hour": 0})
    floor_bound = 0
    for _ in range(96):
        prior_peak = env.price_m.state.running_peak_kw
        rate_cap = env.bat_env.discharging_rate_modifier(env.bat_env.battery)
        _, _, _, _, info = env.step({"agent_ls": 1, "agent_dc": 1, "agent_bat": 1})
        common = info["__common__"]
        target_kw = max(prior_peak, 600.0)
        assert env.bat_env.discharge_peak_target_kw == target_kw
        floor_bound += prior_peak < 600.0
        net_kw = common["dc_total_power_kW"] - 4 * common["pv_ac_kwh"]
        excess_kw = max(0.0, net_kw - target_kw)
        assert math.isclose(common["bat_requested_power_mw"],
                            min(rate_cap, excess_kw / 1000.0), abs_tol=1e-9)
    assert floor_bound > 0


if __name__ == "__main__":
    sys.exit(run_module_tests(globals()))
