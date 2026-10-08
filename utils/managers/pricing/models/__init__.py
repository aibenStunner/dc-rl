from __future__ import annotations

from typing import Mapping

from ..contracts import PricingModel
from .flat import FlatPricingModel
from .hydro_quebec import HydroQuebecPricingModel
from .time_of_use import TimeOfUsePricingModel
from .time_series import TimeSeriesPricingModel


BUILTIN_PRICING_MODELS: Mapping[str, type[PricingModel]] = {
    "flat": FlatPricingModel,
    "hydro_quebec": HydroQuebecPricingModel,
    "time_of_use": TimeOfUsePricingModel,
    "time_series": TimeSeriesPricingModel,
}

__all__ = [
    "BUILTIN_PRICING_MODELS",
    "FlatPricingModel",
    "HydroQuebecPricingModel",
    "TimeOfUsePricingModel",
    "TimeSeriesPricingModel",
]
