"""Pipeline configuration for Brev Data Platform.

Provides configurable options for pipeline runs including:
- sample_size: Limit number of records for trial runs
- is_trial: Whether to use trial collections/paths (separate from full data)
"""

from dagster import Config


class PipelineConfig(Config):
    """Configuration for pipeline runs.

    Attributes:
        sample_size: Maximum number of records to process. Set to 0 for no limit.
                    Use small values (e.g., 10) for trial runs.
        is_trial: If True, write to trial-specific collections and paths.
                 This keeps trial data separate from full production data.
    """

    sample_size: int = 0  # 0 means no limit
    is_trial: bool = False  # If True, use trial collections/paths


# Preset configurations for common use cases
TRIAL_RUN_CONFIG = {"sample_size": 10, "is_trial": True}
SMALL_RUN_CONFIG = {"sample_size": 100}
MEDIUM_RUN_CONFIG = {"sample_size": 1000}
