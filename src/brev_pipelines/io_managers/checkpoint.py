"""LLM Checkpoint Manager for robust partial result persistence.

Provides row-level checkpointing for expensive LLM operations, allowing:
- Resume from partial progress after failures
- Batch saving to minimize I/O overhead
- Automatic cleanup on successful completion

Usage:
    checkpoint = LLMCheckpointManager(minio, "speech_classification", run_id)

    # Load any existing checkpoint
    existing = checkpoint.load()
    processed_ids = set(existing["reference"]) if existing else set()

    # Process rows with checkpointing
    for batch in batches:
        results = process_batch(batch)
        checkpoint.save_batch(results)

    # Get final results and cleanup
    final_df = checkpoint.finalize()
"""

from __future__ import annotations

import contextlib
import io
from typing import TYPE_CHECKING, Any

import polars as pl
from minio.error import S3Error
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from brev_pipelines.resources.minio import MinIOResource

if TYPE_CHECKING:
    from collections.abc import Callable

    from dagster import DagsterLogManager

# Type alias for checkpoint records - callers should use specific TypedDicts
# from brev_pipelines.types (ClassificationCheckpointRecord, etc.)
CheckpointRecord = dict[str, Any]


class LLMCheckpointManager(BaseModel):
    """Manages checkpoints for LLM processing with partial result persistence.

    Saves intermediate results to MinIO during long-running LLM operations,
    allowing recovery from failures without reprocessing completed rows.

    Attributes:
        minio: MinIO resource for checkpoint storage.
        asset_name: Name of the asset being processed.
        run_id: Dagster run ID for namespacing checkpoints.
        bucket: MinIO bucket for checkpoints.
        checkpoint_interval: Number of rows between checkpoint saves.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    minio: MinIOResource
    asset_name: str
    run_id: str
    bucket: str = Field(default="dagster-checkpoints")
    checkpoint_interval: int = Field(default=10)

    # Internal state (private attributes in Pydantic v2)
    _accumulated_results: list[CheckpointRecord] = PrivateAttr(default_factory=list)
    _total_saved: int = PrivateAttr(default=0)

    def __init__(self, **data: Any) -> None:
        """Initialize checkpoint manager with given configuration."""
        super().__init__(**data)

    @property
    def checkpoint_path(self) -> str:
        """Path to checkpoint file in MinIO."""
        return f"checkpoints/{self.asset_name}/{self.run_id}.parquet"

    def load(self) -> pl.DataFrame | None:
        """Load existing checkpoint if available.

        Returns:
            DataFrame with previously processed results, or None if no checkpoint exists.

        Raises:
            RuntimeError: If checkpoint load fails for reasons other than missing file.
        """
        self.minio.ensure_bucket(self.bucket)
        client = self.minio.get_client()

        try:
            response = client.get_object(self.bucket, self.checkpoint_path)
            try:
                data = response.read()
            finally:
                response.close()
                response.release_conn()

            df = pl.read_parquet(io.BytesIO(data))
            self._total_saved = len(df)
            return df
        except S3Error as e:
            if e.code == "NoSuchKey":
                # No checkpoint exists - this is expected on first run
                return None
            # Re-raise other S3 errors (permissions, network, etc.)
            raise RuntimeError(f"Failed to load checkpoint: {e}") from e
        except Exception as e:
            # Handle mock exceptions and other cases where S3Error isn't raised
            if "NoSuchKey" in str(e):
                return None
            raise RuntimeError(f"Failed to load checkpoint: {e}") from e

    def save_batch(self, results: list[CheckpointRecord], force: bool = False) -> None:
        """Accumulate results and save checkpoint when interval is reached.

        Args:
            results: List of checkpoint records to accumulate. Should use TypedDict
                    types from brev_pipelines.types (ClassificationCheckpointRecord,
                    SummaryCheckpointRecord, etc.) for type safety.
            force: Force save even if interval not reached.
        """
        self._accumulated_results.extend(results)

        if force or len(self._accumulated_results) >= self.checkpoint_interval:
            self._flush_checkpoint()

    def _flush_checkpoint(self) -> None:
        """Write accumulated results to checkpoint file.

        Loads existing checkpoint data directly (not via self.load()) to avoid
        recursive state updates, then combines with new results and writes.
        """
        if not self._accumulated_results:
            return

        self.minio.ensure_bucket(self.bucket)
        client = self.minio.get_client()

        # Load existing checkpoint data directly (don't use self.load() to avoid state updates)
        existing_data: list[CheckpointRecord] = []
        try:
            response = client.get_object(self.bucket, self.checkpoint_path)
            try:
                data = response.read()
            finally:
                response.close()
                response.release_conn()
            existing_df = pl.read_parquet(io.BytesIO(data))
            existing_data = existing_df.to_dicts()
        except S3Error as e:
            if e.code != "NoSuchKey":
                raise RuntimeError(f"Failed to load existing checkpoint: {e}") from e
            # NoSuchKey is fine - no existing checkpoint
        except Exception as e:
            # Handle mock exceptions and other cases where S3Error isn't raised
            if "NoSuchKey" not in str(e):
                raise RuntimeError(f"Failed to load existing checkpoint: {e}") from e
            # NoSuchKey is fine - no existing checkpoint

        # Combine existing + new
        all_data = existing_data + list(self._accumulated_results)

        # Create DataFrame and serialize
        combined_df = pl.DataFrame(all_data)

        buffer = io.BytesIO()
        combined_df.write_parquet(buffer)
        parquet_bytes = buffer.getvalue()

        # Upload to MinIO
        client.put_object(
            bucket_name=self.bucket,
            object_name=self.checkpoint_path,
            data=io.BytesIO(parquet_bytes),
            length=len(parquet_bytes),
            content_type="application/octet-stream",
        )

        self._total_saved = len(combined_df)
        self._accumulated_results = []

    def finalize(self) -> pl.DataFrame | None:
        """Flush any remaining results and return final DataFrame.

        Returns:
            Complete DataFrame with all processed results, or None if no data.
        """
        # Flush any remaining accumulated results
        if self._accumulated_results:
            self._flush_checkpoint()

        # Load and return final results
        return self.load()

    def cleanup(self) -> None:
        """Delete checkpoint file after successful completion."""
        client = self.minio.get_client()
        with contextlib.suppress(Exception):
            client.remove_object(self.bucket, self.checkpoint_path)

    @property
    def processed_count(self) -> int:
        """Number of rows saved to checkpoint."""
        return self._total_saved


def process_with_checkpoint(
    df: pl.DataFrame,
    id_column: str,
    process_fn: Callable[[dict[str, Any]], dict[str, Any]],
    checkpoint_manager: LLMCheckpointManager,
    batch_size: int = 10,
    logger: DagsterLogManager | None = None,
    progress_log_interval: int = 100,
) -> pl.DataFrame | None:
    """Process DataFrame rows with checkpointing for LLM operations.

    Generic helper that handles checkpoint loading, resumption, and saving.
    Logs progress at intervals rather than per-item for clean output.

    Args:
        df: Input DataFrame to process.
        id_column: Column name to use as unique identifier.
        process_fn: Function that takes a row dict and returns a checkpoint record.
                   Should handle errors internally and return error info.
                   Use TypedDict types from brev_pipelines.types for type safety.
        checkpoint_manager: Checkpoint manager for persistence.
        batch_size: Number of rows to process before saving checkpoint.
        logger: Optional logger for progress updates.
        progress_log_interval: Log progress every N items (default 100).

    Returns:
        DataFrame with all processing results.
    """
    # Load existing checkpoint
    existing = checkpoint_manager.load()
    processed_ids = set()
    if existing is not None:
        processed_ids = set(existing[id_column].to_list())
        if logger:
            logger.info(f"Resuming from checkpoint: {len(processed_ids)} rows already complete")

    # Filter to unprocessed rows
    to_process = df.filter(~pl.col(id_column).is_in(list(processed_ids)))
    total_remaining = len(to_process)

    if total_remaining == 0:
        if logger:
            logger.info("All rows already processed, nothing to do")
        return existing

    if logger:
        logger.info(f"Processing {total_remaining} rows")

    # Process in batches with progress and failure tracking
    rows = to_process.to_dicts()
    batch_results: list[dict[str, Any]] = []
    processed_count = 0
    success_count = 0
    failure_count = 0
    last_log_count = 0
    recent_errors: list[str] = []  # Track last few errors for summary

    for row in rows:
        result = process_fn(row)
        batch_results.append(result)
        processed_count += 1

        # Track success/failure based on _llm_status column
        status = result.get("_llm_status", "success")
        if status == "failed":
            failure_count += 1
            error_msg = result.get("_llm_error", "Unknown error")
            record_id = result.get("reference", result.get("id", "unknown"))
            # Keep last 5 unique errors for summary
            error_summary = f"{record_id}: {error_msg[:100]}"
            if error_summary not in recent_errors:
                recent_errors.append(error_summary)
                if len(recent_errors) > 5:
                    recent_errors.pop(0)
            # Log individual failures with warning level
            if logger:
                logger.warning(f"LLM call failed for {record_id}: {error_msg[:200]}")
        else:
            success_count += 1

        # Save checkpoint at batch boundaries
        if len(batch_results) >= batch_size:
            checkpoint_manager.save_batch(batch_results, force=True)
            batch_results = []

        # Log progress at intervals (including failure rate)
        if logger and processed_count - last_log_count >= progress_log_interval:
            pct = (processed_count / total_remaining) * 100
            fail_rate = (failure_count / processed_count * 100) if processed_count > 0 else 0
            logger.info(
                f"Progress: {processed_count}/{total_remaining} ({pct:.0f}%) - "
                f"Success: {success_count}, Failed: {failure_count} ({fail_rate:.1f}%)"
            )
            last_log_count = processed_count

    # Save any remaining results
    if batch_results:
        checkpoint_manager.save_batch(batch_results, force=True)

    # Log final summary with failure details
    if logger:
        fail_rate = (failure_count / processed_count * 100) if processed_count > 0 else 0
        logger.info(
            f"Completed: {processed_count}/{total_remaining} rows - "
            f"Success: {success_count}, Failed: {failure_count} ({fail_rate:.1f}%)"
        )
        if failure_count > 0:
            logger.warning(f"LLM failures detected: {failure_count} rows failed")
            if recent_errors:
                logger.warning("Recent errors:\n  " + "\n  ".join(recent_errors))

    return checkpoint_manager.finalize()
