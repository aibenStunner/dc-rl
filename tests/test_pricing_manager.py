from __future__ import annotations

import math
import sys
from pathlib import Path
from types import MappingProxyType

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests._harness import run_module_tests
from utils.pricing.loader import PricingConfig, load_pricing_config
from utils.pricing.manager import PriceManager


_HYDRO_CONFIG = _REPO_ROOT / "data" / "Pricing" / "hydro_quebec_2026.yaml"


def _config(selector: str = "tests.fixtures.manager_pricing_plugins:ValidPlugin"):
    if selector == "hydro_quebec":
        return load_pricing_config(
            {
                "pricing": {
                    "model": "hydro_quebec",
                    "config_file": str(_HYDRO_CONFIG),
                    "options": {"tariff": "rate_m", "demand_floor_kw": 0.0},
                }
            },
            _REPO_ROOT,
        )
    return PricingConfig(
        model=selector,
        config_file=Path("fixture.yaml"),
        options={},
        parameters=MappingProxyType({"schema_version": 1, "model": "fixture_plugin"}),
    )


def _manager(selector: str = "tests.fixtures.custom_pricing_plugin:ValidPlugin", *,
             future_steps: int = 4, timestep_minutes: int = 15):
    return PriceManager.from_config(
        _config(selector),
        datacenter_capacity_mw=2.0,
        future_steps=future_steps,
        timestep_minutes=timestep_minutes,
    )


def test_reset_does_not_implicitly_carry_billing_state():
    """Independent episode resets must start a fresh billing period unless a
    chronological caller explicitly supplies carry state."""
    manager = _manager("hydro_quebec")
    manager.reset(init_day=100, init_hour=0.0)
    manager.step(metered_energy_kwh=900.0 * 0.25, day_of_year=100, hour=0.0)
    carry = manager.export_carry_state()
    assert manager.state.running_peak_kw == 900.0

    manager.reset(init_day=200, init_hour=0.0)
    assert manager.state.running_peak_kw == 0.0

    manager.reset(init_day=200, init_hour=0.0, carry_state=carry)
    assert manager.state.running_peak_kw == 900.0


def test_negative_meter_energy_is_rejected_before_state_mutates():
    manager = _manager()
    manager.reset(init_day=10, init_hour=0.0)
    before = manager.state.billing_period_energy_kwh
    try:
        manager.step(metered_energy_kwh=-1.0, day_of_year=10, hour=0.0)
    except ValueError as exc:
        assert "metered_energy_kwh" in str(exc)
    else:
        raise AssertionError("expected negative meter energy to fail")
    assert manager.state.billing_period_energy_kwh == before


def test_nan_meter_energy_is_rejected_before_state_mutates():
    manager = _manager()
    manager.reset(init_day=10, init_hour=0.0)
    before = manager.state.billing_period_energy_kwh
    try:
        manager.step(metered_energy_kwh=math.nan, day_of_year=10, hour=0.0)
    except ValueError as exc:
        assert "metered_energy_kwh" in str(exc)
    else:
        raise AssertionError("expected NaN meter energy to fail")
    assert manager.state.billing_period_energy_kwh == before


def test_plugin_nonserializable_model_state_is_rejected_at_reset_boundary():
    manager = _manager(
        "tests.fixtures.manager_pricing_plugins:NonSerializableStatePlugin"
    )
    try:
        manager.reset(init_day=10, init_hour=0.0)
    except ValueError as exc:
        assert "state.model_state.resource" in str(exc)
        assert "unsupported" in str(exc)
    else:
        raise AssertionError("expected non-serializable model state to fail")


def test_nested_model_state_rolls_back_after_model_failure():
    manager = _manager("tests.fixtures.manager_pricing_plugins:NestedRollbackPlugin")
    manager.reset(init_day=10, init_hour=0.0)
    state = manager.state
    original = {"schedule": {"segments": [{"label": "original", "values": [1, 2]}, None]}}

    try:
        manager.step(metered_energy_kwh=1.0, day_of_year=10, hour=0.0)
    except ValueError as exc:
        assert "charge" in str(exc)
    else:
        raise AssertionError("expected nonfinite charge to fail")

    assert state.model_state == original


def test_plugin_wrong_forecast_length_is_rejected():
    manager = _manager("tests.fixtures.manager_pricing_plugins:WrongForecastPlugin")
    try:
        manager.reset(init_day=10, init_hour=0.0)
    except ValueError as exc:
        assert "forecast_normalized" in str(exc)
    else:
        raise AssertionError("expected wrong forecast length to fail")


def test_plugin_nonfinite_charge_is_rejected():
    manager = _manager("tests.fixtures.manager_pricing_plugins:NonfiniteChargePlugin")
    manager.reset(init_day=10, init_hour=0.0)
    try:
        manager.step(metered_energy_kwh=1.0, day_of_year=10, hour=0.0)
    except ValueError as exc:
        assert "charge" in str(exc)
    else:
        raise AssertionError("expected nonfinite charge to fail")
    assert manager.state.billing_period_energy_kwh == 0.0


def test_plugin_failure_restores_all_shared_state_in_place():
    manager = _manager(
        "tests.fixtures.manager_pricing_plugins:MutatingNonfiniteChargePlugin"
    )
    manager.reset(init_day=10, init_hour=0.0)
    state = manager.state
    state.billing_period_energy_kwh = 12.0
    state.running_peak_kw = 34.0
    state.model_state["nested"] = {"value": ["original"]}

    try:
        manager.step(metered_energy_kwh=1.0, day_of_year=10, hour=0.0)
    except ValueError as exc:
        assert "charge" in str(exc)
    else:
        raise AssertionError("expected nonfinite charge to fail")

    assert manager.state is state
    assert state.billing_period_energy_kwh == 12.0
    assert state.running_peak_kw == 34.0
    assert state.model_state == {"nested": {"value": ["original"]}}


def test_invalid_observation_restores_state_and_manager_outputs_in_place():
    manager = _manager(
        "tests.fixtures.manager_pricing_plugins:MutatingInvalidObservationPlugin"
    )
    initial_observation = manager.reset(init_day=10, init_hour=0.0)
    initial_charges = manager._charges
    initial_clock = manager._clock
    state = manager.state
    state.billing_period_energy_kwh = 12.0
    state.running_peak_kw = 34.0
    state.model_state["nested"] = {"value": ["original"]}

    try:
        manager.step(metered_energy_kwh=1.0, day_of_year=10, hour=0.25)
    except ValueError as exc:
        assert "normalized_price" in str(exc)
    else:
        raise AssertionError("expected invalid observation to fail")

    assert manager.state is state
    assert state.billing_period_energy_kwh == 12.0
    assert state.running_peak_kw == 34.0
    assert state.model_state == {"nested": {"value": ["original"]}}
    assert manager._charges is initial_charges
    assert manager._observation is initial_observation
    assert manager._clock is initial_clock


def test_plugin_normalized_values_outside_unit_interval_are_rejected():
    manager = _manager("tests.fixtures.manager_pricing_plugins:OutOfRangeObservationPlugin")
    try:
        manager.reset(init_day=10, init_hour=0.0)
    except ValueError as exc:
        assert "normalized" in str(exc)
    else:
        raise AssertionError("expected out-of-range normalized output to fail")


def test_total_cost_equals_standard_categories():
    manager = _manager("tests.fixtures.manager_pricing_plugins:CostPlugin")
    manager.reset(init_day=10, init_hour=0.0)
    manager.step(metered_energy_kwh=1.0, day_of_year=10, hour=0.0)
    assert manager.get_energy_cost_this_step_c() == 2.0
    assert manager.get_demand_charge_increment_c() == 3.0
    assert manager.get_additional_charge_increment_c() == 4.0
    assert manager.get_total_cost_this_step_c() == 9.0


def test_exported_carry_state_round_trips():
    manager = PriceManager.from_config(
        _config("hydro_quebec"), datacenter_capacity_mw=2.0,
        future_steps=4, timestep_minutes=15,
    )
    manager.reset(init_day=100, init_hour=0.0)
    manager.step(metered_energy_kwh=100.0, day_of_year=100, hour=0.0)
    carry = manager.export_carry_state()

    restored = PriceManager.from_config(
        _config("hydro_quebec"), datacenter_capacity_mw=2.0,
        future_steps=4, timestep_minutes=15,
    )
    restored.reset(init_day=100, init_hour=0.0, carry_state=carry)
    assert restored.export_carry_state() == carry
    assert restored.get_normalized_peak() == manager.get_normalized_peak()


def test_rate_m_threshold_straddling_is_counted_once_by_manager():
    manager = PriceManager.from_config(
        _config("hydro_quebec"), datacenter_capacity_mw=2.0,
        future_steps=4, timestep_minutes=15,
    )
    manager.reset(init_day=100, init_hour=0.0)
    manager.step(metered_energy_kwh=209900.0, day_of_year=100, hour=0.0)
    charges = manager.step(metered_energy_kwh=200.0, day_of_year=100, hour=0.25)
    assert math.isclose(charges.energy_cost_c, 100.0 * 6.292 + 100.0 * 4.666)
    assert manager.state.billing_period_energy_kwh == 210100.0


def test_optional_pending_period_settlement_updates_standardized_getters():
    config = load_pricing_config(
        {
            "pricing": {
                "model": "hydro_quebec",
                "config_file": str(_HYDRO_CONFIG),
                "options": {"tariff": "rate_l", "demand_floor_kw": 0.0},
            }
        },
        _REPO_ROOT,
    )
    manager = PriceManager.from_config(
        config, datacenter_capacity_mw=20.0, future_steps=4, timestep_minutes=15,
    )
    manager.reset(init_day=334, init_hour=12.0, carry_state={"ending_peak_kw": 6000.0})
    manager.step(metered_energy_kwh=7000.0 * 0.25, day_of_year=334, hour=12.0)
    charges = manager.settle_pending_period()
    assert charges.energy_cost_c == 0.0
    assert charges.demand_cost_increment_c == 0.0
    assert charges.additional_cost_increment_c > 0.0
    assert manager.get_total_cost_this_step_c() == charges.total_cost_c
    assert manager.settle_pending_period().total_cost_c == 0.0


def test_forecast_getter_returns_a_copy():
    manager = _manager()
    manager.reset(init_day=10, init_hour=0.0)
    forecast = manager.get_forecast_price()
    forecast[0] = 0.0
    assert manager.get_forecast_price()[0] != 0.0


def test_non_mapping_model_state_root_is_rejected_and_step_rolls_back():
    manager = _manager(
        "tests.fixtures.manager_pricing_plugins:ReplacingModelStatePlugin"
    )
    manager.reset(init_day=10, init_hour=0.0)
    state = manager.state
    initial_observation = manager._observation
    initial_charges = manager._charges
    initial_clock = manager._clock

    try:
        manager.step(metered_energy_kwh=1.0, day_of_year=10, hour=0.25)
    except ValueError as exc:
        assert "state.model_state" in str(exc)
        assert "dictionary" in str(exc)
    else:
        raise AssertionError("expected non-mapping model state root to fail")

    assert manager.state is state
    assert state.model_state == {}
    assert manager._observation is initial_observation
    assert manager._charges is initial_charges
    assert manager._clock is initial_clock


def test_reset_observe_invalid_state_leaves_prior_manager_state_untouched():
    manager = _manager(
        "tests.fixtures.manager_pricing_plugins:ObserveMutatesInvalidStatePlugin"
    )
    initial_observation = manager.reset(init_day=10, init_hour=0.0)
    state = manager.state
    initial_charges = manager._charges
    initial_clock = manager._clock

    try:
        manager.reset(init_day=11, init_hour=0.0)
    except ValueError as exc:
        assert "state.model_state.resource" in str(exc)
    else:
        raise AssertionError("expected observe mutation to fail reset")

    assert manager.state is state
    assert manager._observation is initial_observation
    assert manager._charges is initial_charges
    assert manager._clock is initial_clock


def test_invalid_pending_settlement_restores_state_and_manager_outputs():
    manager = _manager(
        "tests.fixtures.manager_pricing_plugins:MutatingInvalidSettlementPlugin"
    )
    initial_observation = manager.reset(init_day=10, init_hour=0.0)
    initial_charges = manager._charges
    initial_clock = manager._clock
    state = manager.state
    state.billing_period_energy_kwh = 12.0
    state.running_peak_kw = 34.0
    state.model_state["nested"] = {"value": ["original"]}

    try:
        manager.settle_pending_period()
    except ValueError as exc:
        assert "charge" in str(exc)
    else:
        raise AssertionError("expected invalid pending settlement to fail")

    assert manager.state is state
    assert state.billing_period_energy_kwh == 12.0
    assert state.running_peak_kw == 34.0
    assert state.model_state == {"nested": {"value": ["original"]}}
    assert manager._observation is initial_observation
    assert manager._charges is initial_charges
    assert manager._clock is initial_clock


def test_successful_pending_settlement_refreshes_observation_at_current_clock():
    manager = _manager(
        "tests.fixtures.manager_pricing_plugins:SettlementChangesObservationPlugin"
    )
    manager.reset(init_day=10, init_hour=0.0)
    manager.step(metered_energy_kwh=1.0, day_of_year=10, hour=0.25)

    assert manager.get_current_price() == 1.0
    assert manager._clock is not None
    clock_before = manager._clock

    charges = manager.settle_pending_period()

    assert charges.total_cost_c == 7.0
    assert manager.get_current_price() == 0.5
    assert manager._clock is clock_before


def test_invalid_export_carry_state_restores_state_in_place():
    manager = _manager(
        "tests.fixtures.manager_pricing_plugins:MutatingInvalidCarryExportPlugin"
    )
    manager.reset(init_day=10, init_hour=0.0)
    state = manager.state
    state.billing_period_energy_kwh = 12.0
    state.running_peak_kw = 34.0
    state.model_state["nested"] = {"value": ["original"]}

    try:
        manager.export_carry_state()
    except ValueError as exc:
        assert "carry_state.nested.peak_kw" in str(exc)
        assert "finite" in str(exc)
    else:
        raise AssertionError("expected invalid carry-state export to fail")

    assert manager.state is state
    assert state.billing_period_energy_kwh == 12.0
    assert state.running_peak_kw == 34.0
    assert state.model_state == {"nested": {"value": ["original"]}}


def test_raising_export_carry_state_restores_state_in_place():
    manager = _manager(
        "tests.fixtures.manager_pricing_plugins:MutatingRaisingCarryExportPlugin"
    )
    manager.reset(init_day=10, init_hour=0.0)
    state = manager.state
    state.billing_period_energy_kwh = 12.0
    state.running_peak_kw = 34.0
    state.model_state["nested"] = {"value": ["original"]}

    try:
        manager.export_carry_state()
    except RuntimeError as exc:
        assert "export failed" in str(exc)
    else:
        raise AssertionError("expected carry-state export to fail")

    assert manager.state is state
    assert state.billing_period_energy_kwh == 12.0
    assert state.running_peak_kw == 34.0
    assert state.model_state == {"nested": {"value": ["original"]}}


def test_opaque_carry_state_is_rejected_with_field_specific_error():
    manager = _manager()

    try:
        manager.reset(init_day=10, init_hour=0.0, carry_state={"opaque": object()})
    except ValueError as exc:
        assert "carry_state.opaque" in str(exc)
        assert "unsupported" in str(exc)
    else:
        raise AssertionError("expected opaque carry state to fail")


def test_nested_nonfinite_carry_state_is_rejected_with_field_specific_error():
    manager = _manager()

    try:
        manager.reset(
            init_day=10,
            init_hour=0.0,
            carry_state={"period": {"peak_kw": math.nan}},
        )
    except ValueError as exc:
        assert "carry_state.period.peak_kw" in str(exc)
        assert "finite" in str(exc)
    else:
        raise AssertionError("expected nested non-finite carry state to fail")


def test_nested_nonfinite_model_state_is_rejected_at_reset_boundary():
    manager = _manager(
        "tests.fixtures.manager_pricing_plugins:NestedNonfiniteStatePlugin"
    )

    try:
        manager.reset(init_day=10, init_hour=0.0)
    except ValueError as exc:
        assert "state.model_state.period.peak_kw" in str(exc)
        assert "finite" in str(exc)
    else:
        raise AssertionError("expected nested non-finite model state to fail")


def _assert_invalid_shared_state_step_rolls_back(selector: str, field: str) -> None:
    manager = _manager(selector)
    initial_observation = manager.reset(init_day=10, init_hour=0.0)
    initial_charges = manager._charges
    initial_clock = manager._clock
    state = manager.state
    state.billing_period_energy_kwh = 12.0
    state.running_peak_kw = 34.0

    try:
        manager.step(metered_energy_kwh=1.0, day_of_year=10, hour=0.25)
    except ValueError as exc:
        assert f"state.{field}" in str(exc)
    else:
        raise AssertionError(f"expected invalid {field} to fail step")

    assert manager.state is state
    assert state.billing_period_energy_kwh == 12.0
    assert state.running_peak_kw == 34.0
    assert manager._observation is initial_observation
    assert manager._charges is initial_charges
    assert manager._clock is initial_clock


def test_step_nan_billing_period_energy_rolls_back_state_and_outputs():
    _assert_invalid_shared_state_step_rolls_back(
        "tests.fixtures.manager_pricing_plugins:StepNaNBillingEnergyPlugin",
        "billing_period_energy_kwh",
    )


def test_step_negative_billing_period_energy_rolls_back_state_and_outputs():
    _assert_invalid_shared_state_step_rolls_back(
        "tests.fixtures.manager_pricing_plugins:StepNegativeBillingEnergyPlugin",
        "billing_period_energy_kwh",
    )


def test_step_boolean_billing_period_energy_rolls_back_state_and_outputs():
    _assert_invalid_shared_state_step_rolls_back(
        "tests.fixtures.manager_pricing_plugins:StepBooleanBillingEnergyPlugin",
        "billing_period_energy_kwh",
    )


def test_step_nan_running_peak_rolls_back_state_and_outputs():
    _assert_invalid_shared_state_step_rolls_back(
        "tests.fixtures.manager_pricing_plugins:StepNaNRunningPeakPlugin",
        "running_peak_kw",
    )


def test_step_negative_running_peak_rolls_back_state_and_outputs():
    _assert_invalid_shared_state_step_rolls_back(
        "tests.fixtures.manager_pricing_plugins:StepNegativeRunningPeakPlugin",
        "running_peak_kw",
    )


def _assert_invalid_shared_state_reset_preserves_prior_state(
        selector: str, field: str) -> None:
    manager = _manager(selector)
    initial_observation = manager.reset(init_day=10, init_hour=0.0)
    initial_charges = manager._charges
    initial_clock = manager._clock
    state = manager.state

    try:
        manager.reset(init_day=11, init_hour=0.0)
    except ValueError as exc:
        assert f"state.{field}" in str(exc)
    else:
        raise AssertionError(f"expected invalid {field} to fail reset")

    assert manager.state is state
    assert state.billing_period_energy_kwh == 0.0
    assert state.running_peak_kw == 0.0
    assert manager._observation is initial_observation
    assert manager._charges is initial_charges
    assert manager._clock is initial_clock


def test_reset_nan_billing_period_energy_preserves_prior_state_and_outputs():
    _assert_invalid_shared_state_reset_preserves_prior_state(
        "tests.fixtures.manager_pricing_plugins:ResetNaNBillingEnergyPlugin",
        "billing_period_energy_kwh",
    )


def test_reset_negative_billing_period_energy_preserves_prior_state_and_outputs():
    _assert_invalid_shared_state_reset_preserves_prior_state(
        "tests.fixtures.manager_pricing_plugins:ResetNegativeBillingEnergyPlugin",
        "billing_period_energy_kwh",
    )


def test_reset_nan_running_peak_preserves_prior_state_and_outputs():
    _assert_invalid_shared_state_reset_preserves_prior_state(
        "tests.fixtures.manager_pricing_plugins:ResetNaNRunningPeakPlugin",
        "running_peak_kw",
    )


def test_reset_negative_running_peak_preserves_prior_state_and_outputs():
    _assert_invalid_shared_state_reset_preserves_prior_state(
        "tests.fixtures.manager_pricing_plugins:ResetNegativeRunningPeakPlugin",
        "running_peak_kw",
    )


def test_reset_boolean_running_peak_preserves_prior_state_and_outputs():
    _assert_invalid_shared_state_reset_preserves_prior_state(
        "tests.fixtures.manager_pricing_plugins:ResetBooleanRunningPeakPlugin",
        "running_peak_kw",
    )


if __name__ == "__main__":
    sys.exit(run_module_tests(globals()))
