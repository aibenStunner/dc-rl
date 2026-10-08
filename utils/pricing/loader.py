from __future__ import annotations

import copy
import importlib
import inspect
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import yaml

from .contracts import PricingModel
from .models import BUILTIN_PRICING_MODELS
from .models.hydro_quebec import validate_options


@dataclass(frozen=True)
class PricingConfig:
    model: str
    config_file: Path
    options: Mapping[str, object]
    parameters: Mapping[str, object]


class PricingConfigError(ValueError):
    pass


def _source(path: Path) -> str:
    return f"pricing config file {path}"


def load_pricing_config(raw: Mapping[str, object], repo_root: Path) -> PricingConfig:
    if not isinstance(raw, Mapping):
        raise PricingConfigError("pricing must be a mapping")
    for removed in ("tariff_rate_override", "demand_floor_kw"):
        if removed in raw:
            replacement = "pricing.options.tariff" if removed == "tariff_rate_override" else f"pricing.options.{removed}"
            raise PricingConfigError(
                f"{removed} is no longer supported; use {replacement}"
            )
    if "pricing" not in raw:
        raise PricingConfigError("pricing mapping is required")

    pricing = raw["pricing"]
    if not isinstance(pricing, Mapping):
        raise PricingConfigError("pricing must be a mapping")

    allowed = {"model", "config_file", "options"}
    unknown = set(pricing) - allowed
    if unknown:
        key = min(unknown, key=repr)
        raise PricingConfigError(f"pricing.{key} is unknown")
    missing = [key for key in ("model", "config_file") if key not in pricing]
    if missing:
        raise PricingConfigError(f"pricing.{missing[0]} is required")

    model = pricing["model"]
    if not isinstance(model, str) or not model:
        raise PricingConfigError("pricing.model must be a non-empty string")
    config_file_value = pricing["config_file"]
    if not isinstance(config_file_value, str) or not config_file_value:
        raise PricingConfigError("pricing.config_file must be a non-empty string")
    options = pricing.get("options", {})
    if not isinstance(options, Mapping):
        raise PricingConfigError("pricing.options must be a mapping")

    config_file = Path(config_file_value)
    if not config_file.is_absolute():
        config_file = Path(repo_root) / config_file
    config_file = config_file.resolve()
    parameters = dict(load_parameter_file(config_file, model))
    if model == "time_series":
        csv_file = parameters.get("csv_file")
        if isinstance(csv_file, str) and not Path(csv_file).is_absolute():
            parameters["csv_file"] = str((Path(repo_root) / csv_file).resolve())
    if model in BUILTIN_PRICING_MODELS:
        try:
            BUILTIN_PRICING_MODELS[model](copy.deepcopy(parameters))
            if model == "hydro_quebec":
                validate_options(options)
        except Exception as exc:
            if model == "hydro_quebec" and str(exc).startswith("pricing.options."):
                message = str(exc)
            else:
                message = f"{_source(config_file)} has invalid {model} parameters: {exc}"
            raise PricingConfigError(message) from exc
    frozen_parameters = MappingProxyType(copy.deepcopy(parameters))
    return PricingConfig(model, config_file, dict(options), frozen_parameters)


def load_parameter_file(path: Path, expected_model: str) -> Mapping[str, object]:
    try:
        with Path(path).open("r", encoding="utf-8") as parameter_file:
            parameters = yaml.safe_load(parameter_file)
    except Exception as exc:
        raise PricingConfigError(f"{_source(Path(path))} could not be read: {exc}") from exc
    if not isinstance(parameters, Mapping):
        raise PricingConfigError(f"{_source(Path(path))} must contain a mapping")
    if type(parameters.get("schema_version")) is not int or parameters["schema_version"] != 1:
        raise PricingConfigError(f"{_source(Path(path))}.schema_version must be 1")
    parameter_model = parameters.get("model")
    if ":" not in expected_model and parameter_model != expected_model:
        raise PricingConfigError(
            f"{_source(Path(path))}.model must match pricing.model {expected_model!r}"
        )
    return parameters


def resolve_pricing_model(selector: str) -> type[PricingModel]:
    if selector in BUILTIN_PRICING_MODELS:
        return BUILTIN_PRICING_MODELS[selector]
    if ":" not in selector:
        names = ", ".join(sorted(BUILTIN_PRICING_MODELS))
        raise PricingConfigError(
            f"unknown pricing model {selector!r}; built-ins: {names}"
        )
    module_name, attribute_name = selector.split(":", 1)
    if not module_name or not attribute_name:
        raise PricingConfigError(f"malformed pricing import path {selector!r}")
    try:
        module = importlib.import_module(module_name)
        model_class = getattr(module, attribute_name)
    except Exception as exc:
        raise PricingConfigError(
            f"could not import pricing model {selector!r}: {exc}"
        ) from exc
    return model_class


_METHOD_SIGNATURES = {
    "reset": ("context", "state", "clock", "carry_state", "options"),
    "step": ("context", "state", "clock", "reading"),
    "observe": ("context", "state", "clock"),
    "export_carry_state": ("state",),
}


def _validate_model_methods(model: PricingModel, selector: str) -> None:
    for method_name, expected_names in _METHOD_SIGNATURES.items():
        method = getattr(model, method_name, None)
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError) as exc:
            raise PricingConfigError(
                f"pricing model {selector!r} has invalid {method_name} method signature: {exc}"
            ) from exc
        parameters = list(signature.parameters.values())
        if method_name == "export_carry_state":
            valid = (
                len(parameters) == 1
                and parameters[0].name == "state"
                and parameters[0].kind
                in (inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD)
            )
        else:
            valid = (
                len(parameters) == len(expected_names)
                and tuple(parameter.name for parameter in parameters) == expected_names
                and all(parameter.kind is inspect.Parameter.KEYWORD_ONLY
                        for parameter in parameters)
            )
        if not valid:
            raise PricingConfigError(
                f"pricing model {selector!r} has invalid {method_name} method signature"
            )


def create_pricing_model(pricing_config: PricingConfig) -> PricingModel:
    parameters = pricing_config.parameters
    model_class = resolve_pricing_model(pricing_config.model)
    try:
        model = model_class(copy.deepcopy(dict(parameters)))
    except Exception as exc:
        raise PricingConfigError(
            f"could not instantiate pricing model {pricing_config.model!r}: {exc}"
        ) from exc
    if not isinstance(model, PricingModel):
        raise PricingConfigError(
            f"pricing model {pricing_config.model!r} does not implement PricingModel"
        )
    _validate_model_methods(model, pricing_config.model)
    return model
