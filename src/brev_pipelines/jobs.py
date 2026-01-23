"""Dagster jobs for Brev Data Platform.

Provides pre-configured jobs for common pipeline operations:
- speeches_full_run: Process all records
- speeches_trial_run: Process only 10 records for testing
- synthetic_full_run: Generate synthetic data for all records
- full_pipeline_trial_run: Complete pipeline with 10 records
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

# Full pipeline (real + synthetic) with all records
# Must run all assets together since K8s pods don't share intermediate storage
full_pipeline_full_run = define_asset_job(
    name="full_pipeline_full_run",
    description="Complete pipeline: Process all speeches and generate synthetic data",
    selection=ALL_SPEECHES_ASSETS,
)

# =============================================================================
# Trial Run Jobs (limited records for testing)
# =============================================================================

# Trial run for speeches pipeline only (10 records)
# Config is passed to all assets that need it (raw_speeches, speeches_data_product, weaviate_index)
speeches_trial_run = define_asset_job(
    name="speeches_trial_run",
    description="Trial run: Process only 10 speeches for testing (separate collections/paths)",
    selection=SPEECHES_ASSETS,
    config={
        "ops": {
            "raw_speeches": {"config": TRIAL_RUN_CONFIG},
            "speeches_data_product": {"config": TRIAL_RUN_CONFIG},
            "weaviate_index": {"config": TRIAL_RUN_CONFIG},
        },
    },
)

# Full pipeline trial run (real + synthetic, 10 records)
# Must include all assets since synthetic depends on enriched_speeches
full_pipeline_trial_run = define_asset_job(
    name="full_pipeline_trial_run",
    description="Trial run: Complete pipeline (real + synthetic) with 10 records (separate collections/paths)",
    selection=ALL_SPEECHES_ASSETS,
    config={
        "ops": {
            "raw_speeches": {"config": TRIAL_RUN_CONFIG},
            "speeches_data_product": {"config": TRIAL_RUN_CONFIG},
            "weaviate_index": {"config": TRIAL_RUN_CONFIG},
            "synthetic_data_product": {"config": TRIAL_RUN_CONFIG},
            "synthetic_weaviate_index": {"config": TRIAL_RUN_CONFIG},
        },
    },
)

# Export all jobs
all_jobs = [
    speeches_full_run,
    speeches_trial_run,
    synthetic_full_run,
    full_pipeline_full_run,
    full_pipeline_trial_run,
]
