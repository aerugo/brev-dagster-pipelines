"""Dagster jobs for Brev Data Platform.

Provides pre-configured jobs for common pipeline operations:

Phase 3 (ETL + Summarization):
- speeches_full_run: Process all records with GPT-OSS summaries
- speeches_trial_run: Process only 10 records for testing

Phase 4 (Two-Stage Synthesis - DECOUPLED):
- synthetic_full_run: Generate synthetic data (loads from LakeFS, independent of Phase 3)
- synthetic_trial_run: Synthesis trial run with 10 records

Combined Pipelines:
- full_pipeline_full_run: Complete pipeline (real + synthetic) all records
- full_pipeline_trial_run: Complete pipeline with 10 records
"""

from dagster import AssetSelection, define_asset_job

from brev_pipelines.config import TRIAL_RUN_CONFIG

# Asset selections
SPEECHES_ASSETS = AssetSelection.groups("central_bank_speeches")
SYNTHETIC_ASSETS = AssetSelection.groups("synthetic_speeches")
ALL_SPEECHES_ASSETS = SPEECHES_ASSETS | SYNTHETIC_ASSETS

# =============================================================================
# Phase 3: ETL + Summarization Jobs
# =============================================================================

speeches_full_run = define_asset_job(
    name="speeches_full_run",
    description=(
        "Phase 3: ETL pipeline with GPT-OSS summaries. "
        "Processes all speeches, generates summaries, and stores in LakeFS."
    ),
    selection=SPEECHES_ASSETS,
)

speeches_trial_run = define_asset_job(
    name="speeches_trial_run",
    description=(
        "Phase 3 Trial: Process only 10 speeches for testing. "
        "Uses separate collections/paths to avoid affecting production data."
    ),
    selection=SPEECHES_ASSETS,
    config={
        "ops": {
            "raw_speeches": {"config": TRIAL_RUN_CONFIG},
            "speeches_data_product": {"config": TRIAL_RUN_CONFIG},
            "weaviate_index": {"config": TRIAL_RUN_CONFIG},
        },
    },
)

# =============================================================================
# Phase 4: Two-Stage Synthesis Jobs (DECOUPLED - loads from LakeFS)
# =============================================================================

synthetic_full_run = define_asset_job(
    name="synthetic_full_run",
    description=(
        "Phase 4: Two-stage synthesis pipeline. "
        "Loads enriched data from LakeFS (decoupled from Phase 3), "
        "trains Safe Synthesizer on summaries, expands with GPT-OSS 120B."
    ),
    selection=SYNTHETIC_ASSETS,
)

synthetic_trial_run = define_asset_job(
    name="synthetic_trial_run",
    description=(
        "Phase 4 Trial: Synthesis with limited records for testing. "
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
# Combined Pipeline Jobs (Phase 3 + Phase 4)
# =============================================================================

full_pipeline_full_run = define_asset_job(
    name="full_pipeline_full_run",
    description=(
        "Complete pipeline: Phase 3 (ETL + summaries) + Phase 4 (synthesis + expansion). "
        "Processes all speeches and generates synthetic dataset."
    ),
    selection=ALL_SPEECHES_ASSETS,
)

full_pipeline_trial_run = define_asset_job(
    name="full_pipeline_trial_run",
    description=(
        "Complete pipeline trial: All phases with 10 records. "
        "Uses separate collections/paths for testing."
    ),
    selection=ALL_SPEECHES_ASSETS,
    config={
        "ops": {
            # Phase 3 trial config
            "raw_speeches": {"config": TRIAL_RUN_CONFIG},
            "speeches_data_product": {"config": TRIAL_RUN_CONFIG},
            "weaviate_index": {"config": TRIAL_RUN_CONFIG},
            # Phase 4 trial config
            "enriched_data_for_synthesis": {"config": TRIAL_RUN_CONFIG},
            "synthetic_data_product": {"config": TRIAL_RUN_CONFIG},
            "synthetic_weaviate_index": {"config": TRIAL_RUN_CONFIG},
        },
    },
)

# Export all jobs
all_jobs = [
    # Phase 3
    speeches_full_run,
    speeches_trial_run,
    # Phase 4
    synthetic_full_run,
    synthetic_trial_run,
    # Combined
    full_pipeline_full_run,
    full_pipeline_trial_run,
]
