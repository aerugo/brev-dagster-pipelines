"""Dagster jobs for Brev Data Platform.

Provides pre-configured jobs for common pipeline operations:
- speeches_full_run: Process all records
- speeches_trial_run: Process only 10 records for testing
- synthetic_trial_run: Generate synthetic data for 10 records
"""

from dagster import AssetSelection, define_asset_job

from brev_pipelines.config import TRIAL_RUN_CONFIG

# Asset selections
SPEECHES_ASSETS = AssetSelection.groups("central_bank_speeches")
SYNTHETIC_ASSETS = AssetSelection.groups("synthetic_speeches")
ALL_SPEECHES_ASSETS = SPEECHES_ASSETS | SYNTHETIC_ASSETS

# =============================================================================
# Full Pipeline Jobs (all records)
# =============================================================================

speeches_full_run = define_asset_job(
    name="speeches_full_run",
    description="Run the full Central Bank Speeches pipeline (all records)",
    selection=SPEECHES_ASSETS,
)

synthetic_full_run = define_asset_job(
    name="synthetic_full_run",
    description="Generate synthetic speeches for all records (Phase 4)",
    selection=SYNTHETIC_ASSETS,
)

# =============================================================================
# Trial Run Jobs (limited records for testing)
# =============================================================================

speeches_trial_run = define_asset_job(
    name="speeches_trial_run",
    description="Trial run: Process only 10 speeches for testing",
    selection=SPEECHES_ASSETS,
    config={
        "ops": {
            "raw_speeches": {
                "config": TRIAL_RUN_CONFIG,
            },
        },
    },
)

synthetic_trial_run = define_asset_job(
    name="synthetic_trial_run",
    description="Trial run: Generate synthetic data for 10 speeches",
    selection=SYNTHETIC_ASSETS,
    config={
        "ops": {
            "raw_speeches": {
                "config": TRIAL_RUN_CONFIG,
            },
        },
    },
)

# Full pipeline trial run (real + synthetic)
full_pipeline_trial_run = define_asset_job(
    name="full_pipeline_trial_run",
    description="Trial run: Complete pipeline (real + synthetic) with 10 records",
    selection=ALL_SPEECHES_ASSETS,
    config={
        "ops": {
            "raw_speeches": {
                "config": TRIAL_RUN_CONFIG,
            },
        },
    },
)

# Export all jobs
all_jobs = [
    speeches_full_run,
    speeches_trial_run,
    synthetic_full_run,
    synthetic_trial_run,
    full_pipeline_trial_run,
]
