"""Dagster jobs for Brev Data Platform.

Provides pre-configured jobs for common pipeline operations:

ETL Pipeline (speeches):
- speeches_full_run: Process all records with GPT-OSS summaries
- speeches_trial_run: Process only 10 records for testing

Synthesis Pipeline (decoupled, loads from LakeFS):
- synthetic_full_run: Generate synthetic data from enriched speeches
- synthetic_trial_run: Synthesis trial run with 10 records

Combined Pipelines:
- full_pipeline_full_run: Complete pipeline (ETL + synthesis) all records
- full_pipeline_trial_run: Complete pipeline with 10 records
"""

from dagster import AssetSelection, define_asset_job

from brev_pipelines.config import TRIAL_RUN_CONFIG

# Asset selections
SPEECHES_ASSETS = AssetSelection.groups("central_bank_speeches")
SYNTHETIC_ASSETS = AssetSelection.groups("synthetic_speeches")
ALL_SPEECHES_ASSETS = SPEECHES_ASSETS | SYNTHETIC_ASSETS

# =============================================================================
# ETL Pipeline Jobs
# =============================================================================

speeches_full_run = define_asset_job(
    name="speeches_full_run",
    description=(
        "ETL pipeline with GPT-OSS summaries. "
        "Processes all speeches, generates summaries, and stores in LakeFS."
    ),
    selection=SPEECHES_ASSETS,
)

speeches_trial_run = define_asset_job(
    name="speeches_trial_run",
    description=(
        "ETL trial: Process only 10 speeches for testing. "
        "Uses separate collections/paths to avoid affecting production data."
    ),
    selection=SPEECHES_ASSETS,
    config={
        # Note: Dagster uses "ops" as the config key for asset configurations.
        # This is expected behavior - assets use the same config namespace as ops.
        "ops": {
            "raw_speeches": {"config": TRIAL_RUN_CONFIG},
            "speeches_data_product": {"config": TRIAL_RUN_CONFIG},
            "weaviate_index": {"config": TRIAL_RUN_CONFIG},
        },
    },
)

# =============================================================================
# Synthesis Pipeline Jobs (decoupled - loads from LakeFS)
# =============================================================================

synthetic_full_run = define_asset_job(
    name="synthetic_full_run",
    description=(
        "Synthesis pipeline using Safe Synthesizer. "
        "Loads enriched data from LakeFS (decoupled from ETL pipeline), "
        "trains on summaries and classifications."
    ),
    selection=SYNTHETIC_ASSETS,
)

synthetic_trial_run = define_asset_job(
    name="synthetic_trial_run",
    description=(
        "Synthesis trial: Generate synthetic data with limited records for testing. "
        "Loads from trial LakeFS path, uses separate Weaviate collections."
    ),
    selection=SYNTHETIC_ASSETS,
    config={
        "ops": {
            "enriched_data_for_synthesis": {"config": TRIAL_RUN_CONFIG},
            "synthetic_data_product": {"config": TRIAL_RUN_CONFIG},
            "synthetic_weaviate_index": {"config": TRIAL_RUN_CONFIG},
        },
    },
)

# =============================================================================
# Combined Pipeline Jobs (ETL + Synthesis)
# =============================================================================

full_pipeline_full_run = define_asset_job(
    name="full_pipeline_full_run",
    description=(
        "Complete pipeline: ETL + summaries followed by synthesis. "
        "Processes all speeches and generates synthetic dataset."
    ),
    selection=ALL_SPEECHES_ASSETS,
)

full_pipeline_trial_run = define_asset_job(
    name="full_pipeline_trial_run",
    description=(
        "Complete pipeline trial: ETL + synthesis with 10 records. "
        "Uses separate collections/paths for testing."
    ),
    selection=ALL_SPEECHES_ASSETS,
    config={
        "ops": {
            # ETL pipeline trial config
            "raw_speeches": {"config": TRIAL_RUN_CONFIG},
            "speeches_data_product": {"config": TRIAL_RUN_CONFIG},
            "weaviate_index": {"config": TRIAL_RUN_CONFIG},
            # Synthesis pipeline trial config
            "enriched_data_for_synthesis": {"config": TRIAL_RUN_CONFIG},
            "synthetic_data_product": {"config": TRIAL_RUN_CONFIG},
            "synthetic_weaviate_index": {"config": TRIAL_RUN_CONFIG},
        },
    },
)

# Export all jobs
all_jobs = [
    # ETL pipeline
    speeches_full_run,
    speeches_trial_run,
    # Synthesis pipeline
    synthetic_full_run,
    synthetic_trial_run,
    # Combined
    full_pipeline_full_run,
    full_pipeline_trial_run,
]
