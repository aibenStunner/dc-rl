from __future__ import annotations

import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import yaml

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tests._harness import run_module_tests
from utils.managers.pricing.contracts import PricingModel
from utils.managers.pricing.loader import (
    PricingConfigError,
    create_pricing_model,
    load_parameter_file,
    load_pricing_config,
    resolve_pricing_model,
)


def _assert_error(raw, expected_source, repo_root=None):
    try:
        load_pricing_config(raw, repo_root or Path.cwd())
    except PricingConfigError as exc:
        assert expected_source in str(exc), str(exc)
    else:
        raise AssertionError("expected PricingConfigError")


def _write_parameter_file(path, model="flat", **extra):
    payload = {"schema_version": 1, "model": model, **extra}
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


@contextmanager
def _temporary_repo_root():
    with tempfile.TemporaryDirectory() as directory:
        yield Path(directory)


def test_missing_pricing_mapping_is_rejected():
    _assert_error({}, "pricing")


def test_old_flat_keys_are_rejected_with_migration_message():
    _assert_error(
        {"pricing": {"model": "flat", "config_file": "x.yaml"},
         "tariff_rate_override": "rate_l"},
        "pricing.options.tariff",
    )


def test_unknown_pricing_keys_are_rejected():
    _assert_error(
        {"pricing": {"model": "flat", "config_file": "x.yaml",
                      "surprise": 1}},
        "surprise",
    )


def test_valid_plugin_can_define_name_during_construction():
    with _temporary_repo_root() as tmp_path:
        parameter_path = tmp_path / "plugin.yaml"
        _write_parameter_file(parameter_path, model="fixture_plugin", value=3)
        config = load_pricing_config(
            {"pricing": {"model": "tests.fixtures.custom_pricing_plugin:ValidPlugin",
                          "config_file": str(parameter_path)}},
            tmp_path,
        )
        model = create_pricing_model(config)
        assert isinstance(model, PricingModel)
        assert model.name == "fixture_plugin"


def test_empty_yaml_is_rejected():
    with _temporary_repo_root() as tmp_path:
        parameter_path = tmp_path / "empty.yaml"
        parameter_path.write_text("", encoding="utf-8")
        _assert_error(
            {"pricing": {"model": "flat", "config_file": str(parameter_path)}},
            str(parameter_path),
        )


def test_unsupported_schema_version_is_rejected():
    with _temporary_repo_root() as tmp_path:
        parameter_path = tmp_path / "unsupported.yaml"
        _write_parameter_file(parameter_path, schema_version=2)
        _assert_error(
            {"pricing": {"model": "flat", "config_file": str(parameter_path)}},
            "schema_version",
        )


def test_schema_version_must_be_builtin_integer_one():
    for invalid_version in (True, 1.0):
        with _temporary_repo_root() as tmp_path:
            parameter_path = tmp_path / "invalid-version.yaml"
            _write_parameter_file(parameter_path, schema_version=invalid_version)
            _assert_error(
                {"pricing": {"model": "flat", "config_file": str(parameter_path)}},
                "schema_version",
            )


def test_removed_flat_key_is_reported_without_pricing_mapping():
    _assert_error(
        {"tariff_rate_override": "rate_l"},
        "tariff_rate_override",
    )
    _assert_error(
        {"demand_floor_kw": 5},
        "pricing.options.demand_floor_kw",
    )


def test_mixed_type_unknown_pricing_keys_raise_config_error():
    _assert_error(
        {"pricing": {"model": "flat", "config_file": "x.yaml",
                      "surprise": 1, 7: "also unknown"}},
        "pricing.",
    )


def test_unknown_registry_name_lists_builtins():
    with _temporary_repo_root() as tmp_path:
        parameter_path = tmp_path / "unknown.yaml"
        _write_parameter_file(parameter_path, model="mystery")
        config = load_pricing_config(
            {"pricing": {"model": "mystery", "config_file": str(parameter_path)}},
            tmp_path,
        )
        try:
            resolve_pricing_model(config.model)
        except PricingConfigError as exc:
            assert "mystery" in str(exc)
            assert "flat" in str(exc)
        else:
            raise AssertionError("expected unknown registry name to fail")


def test_malformed_import_path_names_the_path():
    with _temporary_repo_root() as tmp_path:
        parameter_path = tmp_path / "plugin.yaml"
        _write_parameter_file(parameter_path, model="pkg:Plugin")
        config = load_pricing_config(
            {"pricing": {"model": ":Plugin", "config_file": str(parameter_path)}},
            tmp_path,
        )
        try:
            resolve_pricing_model(config.model)
        except PricingConfigError as exc:
            assert ":Plugin" in str(exc)
        else:
            raise AssertionError("expected malformed import path to fail")


def test_import_failure_names_the_path():
    with _temporary_repo_root() as tmp_path:
        parameter_path = tmp_path / "plugin.yaml"
        _write_parameter_file(parameter_path, model="pkg:Plugin")
        config = load_pricing_config(
            {"pricing": {"model": "missing_package:Plugin",
                          "config_file": str(parameter_path)}},
            tmp_path,
        )
        try:
            resolve_pricing_model(config.model)
        except PricingConfigError as exc:
            assert "missing_package:Plugin" in str(exc)
        else:
            raise AssertionError("expected import failure")


def test_incompatible_plugin_is_rejected():
    with _temporary_repo_root() as tmp_path:
        parameter_path = tmp_path / "plugin.yaml"
        _write_parameter_file(parameter_path, model="fixture_plugin")
        config = load_pricing_config(
            {"pricing": {"model": "tests.fixtures.custom_pricing_plugin:InvalidPlugin",
                          "config_file": str(parameter_path)}},
            tmp_path,
        )
        try:
            create_pricing_model(config)
        except PricingConfigError as exc:
            assert "tests.fixtures.custom_pricing_plugin:InvalidPlugin" in str(exc)
            assert "PricingModel" in str(exc)
        else:
            raise AssertionError("expected incompatible plugin to fail")


def test_malformed_plugin_signature_is_rejected_at_construction():
    with _temporary_repo_root() as tmp_path:
        parameter_path = tmp_path / "plugin.yaml"
        _write_parameter_file(parameter_path, model="fixture_plugin")
        config = load_pricing_config(
            {"pricing": {"model": "tests.fixtures.custom_pricing_plugin:InvalidSignaturePlugin",
                          "config_file": str(parameter_path)}},
            tmp_path,
        )
        try:
            create_pricing_model(config)
        except PricingConfigError as exc:
            message = str(exc)
            assert "InvalidSignaturePlugin" in message
            assert "reset" in message
        else:
            raise AssertionError("expected malformed plugin signature to fail")


def test_valid_import_path_plugin_loads():
    with _temporary_repo_root() as tmp_path:
        parameter_path = tmp_path / "plugin.yaml"
        _write_parameter_file(parameter_path, model="fixture_plugin", value=3)
        config = load_pricing_config(
            {"pricing": {"model": "tests.fixtures.custom_pricing_plugin:ValidPlugin",
                          "config_file": str(parameter_path)}},
            tmp_path,
        )
        model = create_pricing_model(config)
        assert isinstance(model, PricingModel)
        assert model.parameters["value"] == 3


def test_relative_config_path_resolves_from_repo_root():
    with _temporary_repo_root() as tmp_path:
        parameter_path = tmp_path / "nested" / "flat.yaml"
        parameter_path.parent.mkdir()
        _write_parameter_file(parameter_path, energy_price_c_per_kwh=5.0)
        config = load_pricing_config(
            {"pricing": {"model": "flat", "config_file": "nested/flat.yaml"}},
            tmp_path,
        )
        assert config.config_file == parameter_path.resolve()
        assert load_parameter_file(config.config_file, config.model)["model"] == "flat"


def test_create_model_uses_parameters_validated_during_config_load():
    with _temporary_repo_root() as tmp_path:
        parameter_path = tmp_path / "flat.yaml"
        _write_parameter_file(parameter_path, energy_price_c_per_kwh=5.0)
        config = load_pricing_config(
            {"pricing": {"model": "flat", "config_file": str(parameter_path)}},
            tmp_path,
        )
        _write_parameter_file(parameter_path, energy_price_c_per_kwh=99.0)
        model = create_pricing_model(config)
        assert model.price_c_per_kwh == 5.0


def test_flat_builtin_resolves_end_to_end():
    with _temporary_repo_root() as tmp_path:
        parameter_path = tmp_path / "flat.yaml"
        _write_parameter_file(parameter_path, energy_price_c_per_kwh=5.0)
        config = load_pricing_config(
            {"pricing": {"model": "flat", "config_file": str(parameter_path)}},
            tmp_path,
        )
        model = create_pricing_model(config)
        assert model.name == "flat"
        assert model.price_c_per_kwh == 5.0


if __name__ == "__main__":
    sys.exit(run_module_tests(globals()))
