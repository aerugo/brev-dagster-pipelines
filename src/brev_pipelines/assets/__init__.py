"""Dagster assets for Brev Data Platform."""

from brev_pipelines.assets.central_bank_speeches import central_bank_speeches_assets
from brev_pipelines.assets.demo import demo_assets
from brev_pipelines.assets.health import health_assets
from brev_pipelines.assets.synthetic_speeches import synthetic_speeches_assets
from brev_pipelines.assets.validation import validation_assets

__all__ = [
    "central_bank_speeches_assets",
    "demo_assets",
    "health_assets",
    "synthetic_speeches_assets",
    "validation_assets",
]
