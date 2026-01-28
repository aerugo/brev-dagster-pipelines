"""Dagster jobs for Brev Data Platform.

Provides pre-configured jobs for common pipeline operations:

ETL Pipeline (speeches):
- speeches_full_run: Process all records with GPT-OSS summaries (production assets)
- speeches_trial_run: Process only 10 records for testing (trial assets)

The trial and production pipelines use completely separate asset graphs:
- Production: central_bank_speeches group, lakefs_parquet IO manager
- Trial: cbs_trial group, lakefs_parquet_trial IO manager

This prevents trial runs from overwriting production data.

Synthesis Pipeline (depends on speeches pipeline completion):
- synthetic_full_run: Generate synthetic data from enriched speeches
- synthetic_trial_run: Synthesis trial run with 10 records
"""

from dagster import AssetSelection, define_asset_job

from brev_pipelines.config import TRIAL_RUN_CONFIG

# Asset selections - separate groups for trial and production
SPEECHES_PRODUCTION_ASSETS = AssetSelection.groups("central_bank_speeches")
SPEECHES_TRIAL_ASSETS = AssetSelection.groups("cbs_trial")
SYNTHETIC_ASSETS = AssetSelection.groups("synthetic_speeches")

# =============================================================================
# ETL Pipeline Jobs
# =============================================================================

speeches_full_run = define_asset_job(
    name="speeches_full_run",
    description=(
        "ETL pipeline with GPT-OSS summaries. "
        "Processes all speeches, generates summaries, and stores in LakeFS. "
        "Uses production asset graph (central_bank_speeches group)."
    ),
    selection=SPEECHES_PRODUCTION_ASSETS,
)

speeches_trial_run = define_asset_job(
    name="speeches_trial_run",
    description=(
        "ETL trial: Process only 10 speeches for testing. "
        "Uses completely separate trial asset graph (cbs_trial group) "
        "with trial/ LakeFS paths and separate Weaviate collection."
    ),
    selection=SPEECHES_TRIAL_ASSETS,
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

# Export all jobs
all_jobs = [
    # ETL pipeline
    speeches_full_run,
    speeches_trial_run,
    # Synthesis pipeline (requires speeches pipeline to complete first)
    synthetic_full_run,
    synthetic_trial_run,
]
