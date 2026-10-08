from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests._harness import run_module_tests
from utils.pricing.contracts import (
    MeterReading,
    PricingClock,
    PricingContext,
    PricingState,
)
from utils.pricing.loader import (
    PricingConfigError,
    create_pricing_model,
    load_pricing_config,
)
from utils.pricing.manager import PriceManager
from utils.pricing.models.hydro_quebec import HydroQuebecPricingModel


_CONFIG_FILE = _REPO_ROOT / "data" / "Pricing" / "hydro_quebec_2026.yaml"


def _context(capacity_mw=2.0, future_steps=8, timestep_minutes=15):
    return PricingContext(
        datacenter_capacity_mw=capacity_mw,
        timestep_minutes=timestep_minutes,
        future_steps=future_steps,
    )


def _clock(day=100, hour=0.0):
    return PricingClock(day_of_year=day, hour=hour)


def _model(tariff="rate_m", capacity_mw=2.0, demand_floor_kw=0.0):
    config = load_pricing_config(
        {
            "pricing": {
                "model": "hydro_quebec",
                "config_file": str(_CONFIG_FILE),
                "options": {
                    "tariff": tariff,
                    "demand_floor_kw": demand_floor_kw,
                },
            }
        },
        _REPO_ROOT,
    )
    return create_pricing_model(config), _context(capacity_mw=capacity_mw)


def _reset(model, context, state=None, day=100, carry=None, tariff="rate_m",
           demand_floor_kw=0.0):
    state = state or PricingState()
    model.reset(
        context=context,
        state=state,
        clock=_clock(day=day),
        carry_state=carry,
        options={"tariff": tariff, "demand_floor_kw": demand_floor_kw},
    )
    return state


def test_yaml_contains_cited_hydro_quebec_figures():
    parameters = yaml.safe_load(_CONFIG_FILE.read_text(encoding="utf-8"))
    assert parameters["metadata"]["effective_date"] == "2026-04-01"
    assert isinstance(parameters["metadata"]["effective_date"], str)
    assert parameters["selection"]["rate_l_threshold_kw"] == 5000.0
    assert parameters["rates"]["rate_m"]["energy_tier1_c_per_kwh"] == 6.292
    assert parameters["rates"]["rate_m"]["energy_tier2_c_per_kwh"] == 4.666
    assert parameters["rates"]["rate_m"]["tier2_threshold_kwh"] == 210000.0
    assert parameters["rates"]["rate_m"]["demand_charge_c_per_kw_month"] == 1824.2
    assert parameters["rates"]["rate_m"]["winter_ratchet_fraction"] == 0.65
    assert parameters["rates"]["rate_l"]["energy_price_c_per_kwh"] == 3.821
    assert parameters["rates"]["rate_l"]["demand_charge_c_per_kw_month"] == 1502.7
    assert parameters["rates"]["rate_l"]["optimization_charge_c_per_kw_day"] == 880.8
    assert parameters["rates"]["rate_l"]["optimization_overrun_fraction"] == 1.10
    assert parameters["rates"]["new_dc_rate"]["approved"] is False

    config = load_pricing_config(
        {
            "pricing": {
                "model": "hydro_quebec",
                "config_file": str(_CONFIG_FILE),
            }
        },
        _REPO_ROOT,
    )
    assert config.parameters["rates"]["rate_l"]["optimization_overrun_fraction"] == 1.10
    assert config.parameters["metadata"]["effective_date"] == "2026-04-01"
    assert config.parameters["metadata"]["source"] == "hq-electricity-rates.pdf"


def test_loader_rejects_invalid_rate_l_overrun_fraction_with_field_error():
    for value, expected_message in (
        (0.99, "must be at least one"),
        ("invalid", "must be a finite non-negative number"),
        (math.inf, "must be a finite non-negative number"),
    ):
        parameters = yaml.safe_load(_CONFIG_FILE.read_text(encoding="utf-8"))
        parameters["rates"]["rate_l"]["optimization_overrun_fraction"] = value
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "bad.yaml"
            path.write_text(yaml.safe_dump(parameters), encoding="utf-8")
            try:
                load_pricing_config(
                    {"pricing": {"model": "hydro_quebec", "config_file": str(path)}},
                    root,
                )
            except PricingConfigError as exc:
                message = str(exc)
                assert "rates.rate_l.optimization_overrun_fraction" in message
                assert expected_message in message
            else:
                raise AssertionError("expected invalid Rate L overrun fraction to fail")


def test_rate_schemas_reject_fields_from_other_rates_with_field_paths():
    cases = (
        ("rate_l", "energy_tier1_c_per_kwh", 1.0),
        ("rate_m", "optimization_charge_c_per_kw_day", 1.0),
        ("new_dc_rate", "winter_ratchet_fraction", 0.65),
        ("new_dc_rate", "unexpected_field", 1.0),
    )
    for rate_name, field, value in cases:
        parameters = yaml.safe_load(_CONFIG_FILE.read_text(encoding="utf-8"))
        parameters["rates"][rate_name][field] = value
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "bad.yaml"
            path.write_text(yaml.safe_dump(parameters), encoding="utf-8")
            try:
                load_pricing_config(
                    {"pricing": {"model": "hydro_quebec", "config_file": str(path)}},
                    root,
                )
            except PricingConfigError as exc:
                message = str(exc)
                assert f"rates.{rate_name}.{field}" in message, message
            else:
                raise AssertionError("expected rate-specific field to be rejected")


def test_yaml_winter_bounds_control_wrapped_and_nonwrapped_seasons():
    parameters = yaml.safe_load(_CONFIG_FILE.read_text(encoding="utf-8"))
    parameters["metadata"]["winter"] = {
        "start_day_of_year": 100,
        "end_day_of_year": 120,
    }
    model = HydroQuebecPricingModel(parameters)
    context = _context()

    in_window = _reset(model, context, day=100)
    model.step(
        context=context,
        state=in_window,
        clock=_clock(day=100),
        reading=MeterReading(energy_kwh=300.0, demand_kw=1200.0),
    )
    assert in_window.model_state["winter_peak_kw"] == 1200.0

    outside_window = _reset(model, context, day=334)
    model.step(
        context=context,
        state=outside_window,
        clock=_clock(day=334),
        reading=MeterReading(energy_kwh=300.0, demand_kw=1200.0),
    )
    assert outside_window.model_state["winter_peak_kw"] == 0.0


def test_carry_state_rejects_unknown_keys_with_field_path():
    model, context = _model()
    try:
        _reset(model, context, carry={"winter_peak_kw": 1.0, "surprise": 2.0})
    except ValueError as exc:
        assert "carry_state.surprise" in str(exc), str(exc)
    else:
        raise AssertionError("expected unknown carry_state key to fail")


def test_auto_selection_uses_threshold_and_never_selects_pending_rate():
    model, context = _model(tariff="auto", capacity_mw=4.999)
    state = _reset(model, context, tariff="auto")
    assert state.model_state["tariff"] == "rate_m"

    model, context = _model(tariff="auto", capacity_mw=5.0)
    state = _reset(model, context, tariff="auto")
    assert state.model_state["tariff"] == "rate_l"

    model, context = _model(tariff="auto", capacity_mw=1000.0)
    state = _reset(model, context, tariff="auto")
    assert state.model_state["tariff"] == "rate_l"


def test_explicit_pending_rate_is_available_but_not_auto_selected():
    model, context = _model(tariff="new_dc_rate", capacity_mw=20.0)
    state = _reset(model, context, tariff="new_dc_rate")
    assert state.model_state["tariff"] == "new_dc_rate"
    charges = model.step(
        context=context,
        state=state,
        clock=_clock(day=100, hour=0.0),
        reading=MeterReading(energy_kwh=10.0, demand_kw=40.0),
    )
    assert charges.energy_cost_c == 130.0
    assert charges.demand_cost_increment_c == 0.0


def test_rate_m_tier_boundary_splits_one_step_exactly():
    config = load_pricing_config(
        {"pricing": {"model": "hydro_quebec", "config_file": str(_CONFIG_FILE),
                     "options": {"tariff": "rate_m"}}},
        _REPO_ROOT,
    )
    manager = PriceManager.from_config(
        config, datacenter_capacity_mw=2.0, future_steps=8, timestep_minutes=15,
    )
    manager.reset(init_day=100, init_hour=0.0)
    manager.step(metered_energy_kwh=209900.0, day_of_year=100, hour=0.0)
    charges = manager.step(metered_energy_kwh=200.0, day_of_year=100, hour=0.25)
    assert math.isclose(charges.energy_cost_c, 100.0 * 6.292 + 100.0 * 4.666)
    assert manager.state.billing_period_energy_kwh == 210100.0


def test_rate_l_is_flat_even_after_large_energy_step():
    model, context = _model(tariff="rate_l", capacity_mw=20.0)
    state = _reset(model, context, tariff="rate_l")
    charges = model.step(
        context=context,
        state=state,
        clock=_clock(hour=0.0),
        reading=MeterReading(energy_kwh=1_000_000.0, demand_kw=4_000_000.0),
    )
    assert math.isclose(charges.energy_cost_c, 1_000_000.0 * 3.821)


def test_demand_charge_bills_only_running_peak_increments():
    model, context = _model()
    state = _reset(model, context)
    total = 0.0
    for demand_kw in [100.0, 400.0, 250.0, 900.0, 300.0, 900.0, 50.0]:
        charges = model.step(
            context=context,
            state=state,
            clock=_clock(hour=0.0),
            reading=MeterReading(energy_kwh=demand_kw * 0.25, demand_kw=demand_kw),
        )
        total += charges.demand_cost_increment_c
    assert math.isclose(total, 900.0 * 1824.2)
    assert state.running_peak_kw == 900.0
    assert state.model_state["ending_peak_kw"] == 900.0


def test_winter_ratchet_tracks_winter_peak_and_carries_to_next_period():
    model, context = _model()
    state = _reset(model, context, day=334)
    for index in range(96):
        model.step(
            context=context,
            state=state,
            clock=_clock(day=334, hour=index * 0.25),
            reading=MeterReading(energy_kwh=1200.0 * 0.25, demand_kw=1200.0),
        )
    assert state.model_state["winter_peak_kw"] == 1200.0
    carry = model.export_carry_state(state)
    assert carry == {"winter_peak_kw": 1200.0, "ending_peak_kw": 1200.0}

    next_state = PricingState()
    model.reset(
        context=context,
        state=next_state,
        clock=_clock(day=100),
        carry_state=carry,
        options={"tariff": "rate_m", "demand_floor_kw": 0.0},
    )
    # Existing PriceManager semantics carry the ending peak across episodes;
    # the winter ratchet is only a fallback when no ending peak is carried.
    assert next_state.running_peak_kw == 1200.0
    assert next_state.model_state["winter_peak_kw"] == 1200.0


def test_winter_ratchet_floor_applies_without_ending_peak_carry():
    model, context = _model()
    state = PricingState()
    model.reset(
        context=context,
        state=state,
        clock=_clock(day=100),
        carry_state={"winter_peak_kw": 1200.0, "ending_peak_kw": 0.0},
        options={"tariff": "rate_m", "demand_floor_kw": 0.0},
    )
    assert state.running_peak_kw == 780.0
    assert state.model_state["winter_peak_kw"] == 1200.0


def test_summer_demand_does_not_feed_winter_ratchet():
    model, context = _model()
    state = _reset(model, context, day=100)
    model.step(
        context=context,
        state=state,
        clock=_clock(day=100, hour=0.0),
        reading=MeterReading(energy_kwh=1800.0 * 0.25, demand_kw=1800.0),
    )
    assert state.model_state["winter_peak_kw"] == 0.0

    carry = model.export_carry_state(state)
    assert carry == {"winter_peak_kw": 0.0, "ending_peak_kw": 1800.0}

    next_state = PricingState()
    model.reset(
        context=context,
        state=next_state,
        clock=_clock(day=334),
        carry_state=carry,
        options={"tariff": "rate_m", "demand_floor_kw": 0.0},
    )
    # Summer demand does not raise winter_peak_kw, but the observed ending
    # peak still carries across the episode boundary.
    assert next_state.running_peak_kw == 1800.0
    assert next_state.model_state["winter_peak_kw"] == 0.0


def test_rate_l_optimization_charge_is_winter_daily_overrun_only():
    model, context = _model(tariff="rate_l", capacity_mw=20.0)
    state = _reset(model, context, day=100, tariff="rate_l", carry={"ending_peak_kw": 6000.0})
    summer_total = 0.0
    for index in range(96):
        charges = model.step(
            context=context,
            state=state,
            clock=_clock(day=100, hour=index * 0.25),
            reading=MeterReading(energy_kwh=7000.0 * 0.25, demand_kw=7000.0),
        )
        summer_total += charges.additional_cost_increment_c
    assert summer_total == 0.0

    winter_total = 0.0
    for index in range(96):
        charges = model.step(
            context=context,
            state=state,
            clock=_clock(day=334, hour=index * 0.25),
            reading=MeterReading(energy_kwh=7000.0 * 0.25, demand_kw=7000.0),
        )
        winter_total += charges.additional_cost_increment_c
    assert math.isclose(winter_total, (7000.0 - 1.10 * 6000.0) * 880.8)


def test_rate_l_noon_reset_settles_only_that_calendar_days_peak():
    model, context = _model(tariff="rate_l", capacity_mw=20.0)
    state = _reset(
        model,
        context,
        day=334,
        tariff="rate_l",
        carry={"ending_peak_kw": 6000.0},
    )
    expected_charge = (7000.0 - 1.10 * 6000.0) * 880.8

    for hour in range(12, 24):
        charges = model.step(
            context=context,
            state=state,
            clock=_clock(day=334, hour=float(hour)),
            reading=MeterReading(energy_kwh=7000.0, demand_kw=7000.0),
        )
        assert charges.additional_cost_increment_c == 0.0

    charges = model.step(
        context=context,
        state=state,
        clock=_clock(day=335, hour=0.0),
        reading=MeterReading(energy_kwh=6500.0, demand_kw=6500.0),
    )
    assert math.isclose(charges.additional_cost_increment_c, expected_charge)
    assert state.model_state["day_peak_kw"] == 6500.0


def test_rate_l_settlement_uses_the_previous_days_winter_status():
    model, context = _model(tariff="rate_l", capacity_mw=20.0)
    expected_charge = (7000.0 - 1.10 * 6000.0) * 880.8

    december_state = _reset(
        model,
        context,
        day=333,
        tariff="rate_l",
        carry={"ending_peak_kw": 6000.0},
    )
    model.step(
        context=context,
        state=december_state,
        clock=_clock(day=333, hour=23.0),
        reading=MeterReading(energy_kwh=7000.0, demand_kw=7000.0),
    )
    dec_1_entry = model.step(
        context=context,
        state=december_state,
        clock=_clock(day=334, hour=0.0),
        reading=MeterReading(energy_kwh=7000.0, demand_kw=7000.0),
    )
    assert dec_1_entry.additional_cost_increment_c == 0.0
    after_dec_1 = model.step(
        context=context,
        state=december_state,
        clock=_clock(day=335, hour=0.0),
        reading=MeterReading(energy_kwh=6500.0, demand_kw=6500.0),
    )
    assert math.isclose(after_dec_1.additional_cost_increment_c, expected_charge)

    march_state = _reset(
        model,
        context,
        day=89,
        tariff="rate_l",
        carry={"ending_peak_kw": 6000.0},
    )
    model.step(
        context=context,
        state=march_state,
        clock=_clock(day=89, hour=23.0),
        reading=MeterReading(energy_kwh=7000.0, demand_kw=7000.0),
    )
    april_1_entry = model.step(
        context=context,
        state=march_state,
        clock=_clock(day=90, hour=0.0),
        reading=MeterReading(energy_kwh=6500.0, demand_kw=6500.0),
    )
    assert math.isclose(april_1_entry.additional_cost_increment_c, expected_charge)
    after_april_1 = model.step(
        context=context,
        state=march_state,
        clock=_clock(day=91, hour=0.0),
        reading=MeterReading(energy_kwh=7000.0, demand_kw=7000.0),
    )
    assert after_april_1.additional_cost_increment_c == 0.0


def test_rate_l_settles_december_31_when_clock_wraps_to_january_1():
    model, context = _model(tariff="rate_l", capacity_mw=20.0)
    state = _reset(
        model,
        context,
        day=364,
        tariff="rate_l",
        carry={"ending_peak_kw": 6000.0},
    )
    expected_charge = (7000.0 - 1.10 * 6000.0) * 880.8
    model.step(
        context=context,
        state=state,
        clock=_clock(day=364, hour=23.0),
        reading=MeterReading(energy_kwh=7000.0, demand_kw=7000.0),
    )

    charges = model.step(
        context=context,
        state=state,
        clock=_clock(day=365, hour=0.0),
        reading=MeterReading(energy_kwh=6500.0, demand_kw=6500.0),
    )
    assert math.isclose(charges.additional_cost_increment_c, expected_charge)
    assert state.model_state["pending_day_key"] == 0
    assert state.model_state["day_peak_kw"] == 6500.0


def test_rate_l_explicit_boundary_settles_final_partial_day_once():
    model, context = _model(tariff="rate_l", capacity_mw=20.0)
    state = _reset(
        model,
        context,
        day=334,
        tariff="rate_l",
        carry={"ending_peak_kw": 6000.0},
    )
    expected_charge = (7000.0 - 1.10 * 6000.0) * 880.8
    model.step(
        context=context,
        state=state,
        clock=_clock(day=334, hour=12.0),
        reading=MeterReading(energy_kwh=7000.0, demand_kw=7000.0),
    )

    settlement = model.settle_pending_day(state=state)
    assert settlement.energy_cost_c == 0.0
    assert settlement.demand_cost_increment_c == 0.0
    assert math.isclose(settlement.additional_cost_increment_c, expected_charge)
    assert model.settle_pending_day(state=state).additional_cost_increment_c == 0.0


def test_winter_calendar_boundaries_are_configured_and_inclusive():
    model, context = _model()
    for day, expected in ((333, False), (334, True), (364, True), (0, True),
                          (89, True), (90, False)):
        state = PricingState()
        _reset(model, context, state=state, day=day)
        model.step(
            context=context,
            state=state,
            clock=_clock(day=day, hour=0.0),
            reading=MeterReading(energy_kwh=1.0, demand_kw=100.0),
        )
        assert (state.model_state["winter_peak_kw"] > 0.0) is expected


def test_winter_interval_can_be_nonwrapped():
    parameters = yaml.safe_load(_CONFIG_FILE.read_text(encoding="utf-8"))
    parameters["metadata"]["winter"] = {
        "start_day_of_year": 100,
        "end_day_of_year": 120,
    }
    model = HydroQuebecPricingModel(parameters)
    context = _context()
    for day, expected in ((99, False), (100, True), (120, True), (121, False)):
        state = _reset(model, context, state=PricingState(), day=day)
        model.step(
            context=context,
            state=state,
            clock=_clock(day=day, hour=0.0),
            reading=MeterReading(energy_kwh=1.0, demand_kw=100.0),
        )
        assert (state.model_state["winter_peak_kw"] > 0.0) is expected


def test_winter_bounds_are_required_in_yaml_metadata():
    parameters = yaml.safe_load(_CONFIG_FILE.read_text(encoding="utf-8"))
    del parameters["metadata"]["winter"]
    try:
        HydroQuebecPricingModel(parameters)
    except ValueError as exc:
        assert "metadata.winter" in str(exc), str(exc)
    else:
        raise AssertionError("expected missing winter metadata to fail")


def test_observation_has_contract_shape_and_peak_progress():
    model, context = _model()
    state = _reset(model, context)
    state.billing_period_energy_kwh = 100.0
    state.running_peak_kw = 1000.0
    observation = model.observe(
        context=context,
        state=state,
        clock=_clock(hour=13.5),
    )
    assert isinstance(observation.normalized_price, float)
    assert observation.forecast_normalized.shape == (8,)
    assert np.all((observation.forecast_normalized >= 0.0) &
                  (observation.forecast_normalized <= 1.0))
    assert observation.billing_progress > 0.0
    assert observation.normalized_peak == 0.5



def test_manager_cost_trace_is_exact_and_legacy_independent():
    """A deterministic Rate M trace stays self-contained after removal of the
    old module: the package manager owns common billing energy exactly once
    and exposes the sum of all three standardized charge categories."""
    config = load_pricing_config(
        {"pricing": {
            "model": "hydro_quebec",
            "config_file": str(_CONFIG_FILE),
            "options": {"tariff": "rate_m", "demand_floor_kw": 0.0},
        }},
        _REPO_ROOT,
    )
    manager = PriceManager.from_config(
        config, datacenter_capacity_mw=2.0, future_steps=8, timestep_minutes=15
    )
    manager.reset(init_day=100, init_hour=0.0)
    trace = [100.0, 400.0, 250.0, 900.0]
    total = 0.0
    for index, demand_kw in enumerate(trace):
        charges = manager.step(
            metered_energy_kwh=demand_kw * 0.25,
            day_of_year=100,
            hour=index * 0.25,
        )
        assert math.isclose(
            manager.get_total_cost_this_step_c(),
            charges.energy_cost_c + charges.demand_cost_increment_c
            + charges.additional_cost_increment_c,
        )
        total += charges.demand_cost_increment_c
    assert math.isclose(total, 900.0 * 1824.2)


def test_loader_rejects_hydro_options_with_source_and_field():
    for options, expected in (
        ({"tariff": "rate_x"}, "pricing.options.tariff"),
        ({"demand_floor_kw": -1.0}, "pricing.options.demand_floor_kw"),
        ({"surprise": 1}, "pricing.options.surprise"),
    ):
        try:
            load_pricing_config(
                {"pricing": {
                    "model": "hydro_quebec",
                    "config_file": str(_CONFIG_FILE),
                    "options": options,
                }},
                _REPO_ROOT,
            )
        except PricingConfigError as exc:
            assert expected in str(exc), str(exc)
        else:
            raise AssertionError("expected invalid Hydro-Québec option to fail")


def test_loader_rejects_missing_rates_and_nonfinite_figures_with_field():
    metadata = {
        "effective_date": "2026-04-01",
        "source": "test-source",
        "winter": {"start_day_of_year": 334, "end_day_of_year": 89},
    }
    selection = {"rate_l_threshold_kw": 5000.0}
    for payload, expected in (
        ({"schema_version": 1, "model": "hydro_quebec",
          "metadata": metadata, "selection": selection}, "rates"),
        ({"schema_version": 1, "model": "hydro_quebec",
          "metadata": metadata, "selection": selection, "rates": {
              "rate_m": {
                  "approved": True,
                  "citation": "test",
                  "energy_tier1_c_per_kwh": math.nan,
                  "energy_tier2_c_per_kwh": 1.0,
                  "tier2_threshold_kwh": 1.0,
                  "demand_charge_c_per_kw_month": 0.0,
                  "minimum_billing_demand_kw": 0.0,
                  "winter_ratchet_fraction": 0.0,
                  "minimum_bill_c_1phase": 0.0,
                  "minimum_bill_c_3phase": 0.0,
              },
              "rate_l": {
                  "approved": True,
                  "citation": "test",
                  "energy_price_c_per_kwh": 1.0,
                  "demand_charge_c_per_kw_month": 0.0,
                  "minimum_billing_demand_kw": 0.0,
                  "optimization_charge_c_per_kw_day": 0.0,
                  "optimization_charge_cap_c_per_kw_month": 0.0,
                  "optimization_overrun_fraction": 1.0,
              },
              "new_dc_rate": {
                  "approved": True,
                  "citation": "test",
                  "energy_price_c_per_kwh": 1.0,
                  "demand_charge_c_per_kw_month": 0.0,
                  "minimum_billing_demand_kw": 0.0,
              },
          }}, "rates.rate_m"),
    ):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "bad.yaml"
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            try:
                load_pricing_config(
                    {"pricing": {"model": "hydro_quebec", "config_file": str(path)}},
                    root,
                )
            except PricingConfigError as exc:
                assert expected in str(exc), str(exc)
            else:
                raise AssertionError("expected invalid Hydro-Québec YAML to fail")


def test_model_rejects_invalid_reading():
    model, context = _model()
    state = _reset(model, context)
    try:
        model.step(
            context=context,
            state=state,
            clock=_clock(),
            reading=MeterReading(energy_kwh=-1.0, demand_kw=1.0),
        )
    except ValueError as exc:
        assert "reading.energy_kwh" in str(exc)
    else:
        raise AssertionError("expected invalid reading to fail")


def test_rate_l_minimum_billing_demand_initializes_running_peak():
    model, context = _model(tariff="rate_l", capacity_mw=20.0)
    state = _reset(model, context, tariff="rate_l")

    assert state.running_peak_kw == 5000.0
    charges = model.step(
        context=context,
        state=state,
        clock=_clock(),
        reading=MeterReading(energy_kwh=1000.0, demand_kw=4000.0),
    )
    assert charges.demand_cost_increment_c == 0.0


def test_rate_m_has_no_minimum_demand_without_carry_or_ratchet():
    model, context = _model(tariff="rate_m")
    state = _reset(model, context, tariff="rate_m")

    assert state.running_peak_kw == 0.0


if __name__ == "__main__":
    sys.exit(run_module_tests(globals()))
