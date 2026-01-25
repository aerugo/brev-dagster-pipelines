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

import io
from typing import Any

import polars as pl
from pydantic import BaseModel, Field

from brev_pipelines.resources.minio import MinIOResource


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

    minio: MinIOResource
    asset_name: str
    run_id: str
    bucket: str = Field(default="dagster-checkpoints")
    checkpoint_interval: int = Field(default=10)

    # Internal state
    _accumulated_results: list = []
    _total_saved: int = 0

    class Config:
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True

    def __init__(self, **data):
        super().__init__(**data)
        object.__setattr__(self, "_accumulated_results", [])
        object.__setattr__(self, "_total_saved", 0)

    @property
    def checkpoint_path(self) -> str:
        """Path to checkpoint file in MinIO."""
        return f"checkpoints/{self.asset_name}/{self.run_id}.parquet"

    def load(self) -> pl.DataFrame | None:
        """Load existing checkpoint if available.

        Returns:
            DataFrame with previously processed results, or None if no checkpoint exists.
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
            object.__setattr__(self, "_total_saved", len(df))
            return df
        except Exception:
            # No checkpoint exists
            return None

    def save_batch(self, results: list[dict], force: bool = False) -> None:
        """Accumulate results and save checkpoint when interval is reached.

        Args:
            results: List of result dictionaries to accumulate.
            force: Force save even if interval not reached.
        """
        self._accumulated_results.extend(results)

        if force or len(self._accumulated_results) >= self.checkpoint_interval:
            self._flush_checkpoint()

    def _flush_checkpoint(self) -> None:
        """Write accumulated results to checkpoint file."""
        if not self._accumulated_results:
            return

        self.minio.ensure_bucket(self.bucket)
        client = self.minio.get_client()

        # Load existing checkpoint
        existing_df = self.load()

        # Create new DataFrame from accumulated results
        new_df = pl.DataFrame(self._accumulated_results)

        # Combine with existing
        if existing_df is not None:
            combined_df = pl.concat([existing_df, new_df])
        else:
            combined_df = new_df

        # Serialize to Parquet
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

        object.__setattr__(self, "_total_saved", len(combined_df))
        object.__setattr__(self, "_accumulated_results", [])

    def finalize(self) -> pl.DataFrame:
        """Flush any remaining results and return final DataFrame.

        Returns:
            Complete DataFrame with all processed results.
        """
        # Flush any remaining accumulated results
        if self._accumulated_results:
            self._flush_checkpoint()

        # Load and return final results
        return self.load()

    def cleanup(self) -> None:
        """Delete checkpoint file after successful completion."""
        client = self.minio.get_client()
        try:
            client.remove_object(self.bucket, self.checkpoint_path)
        except Exception:
            pass  # Ignore if doesn't exist

    @property
    def processed_count(self) -> int:
        """Number of rows saved to checkpoint."""
        return self._total_saved


def process_with_checkpoint(
    df: pl.DataFrame,
    id_column: str,
    process_fn: callable,
    checkpoint_manager: LLMCheckpointManager,
    batch_size: int = 10,
    logger=None,
) -> pl.DataFrame:
    """Process DataFrame rows with checkpointing for LLM operations.

    Generic helper that handles checkpoint loading, resumption, and saving.

    Args:
        df: Input DataFrame to process.
        id_column: Column name to use as unique identifier.
        process_fn: Function that takes a row dict and returns a result dict.
                   Should handle errors internally and return error info.
        checkpoint_manager: Checkpoint manager for persistence.
        batch_size: Number of rows to process before saving checkpoint.
        logger: Optional logger for progress updates.

    Returns:
        DataFrame with all processing results.
    """
    # Load existing checkpoint
    existing = checkpoint_manager.load()
    processed_ids = set()
    if existing is not None:
        processed_ids = set(existing[id_column].to_list())
        if logger:
            logger.info(f"Loaded checkpoint with {len(processed_ids)} processed rows")

    # Filter to unprocessed rows
    to_process = df.filter(~pl.col(id_column).is_in(list(processed_ids)))
    if logger:
        logger.info(f"Processing {len(to_process)} remaining rows (skipping {len(processed_ids)} already done)")

    if len(to_process) == 0:
        return existing

    # Process in batches
    rows = to_process.to_dicts()
    batch_results = []

    for i, row in enumerate(rows):
        result = process_fn(row)
        batch_results.append(result)

        # Save checkpoint at batch boundaries
        if len(batch_results) >= batch_size:
            checkpoint_manager.save_batch(batch_results, force=True)
            batch_results = []
            if logger:
                logger.info(f"Checkpoint saved: {checkpoint_manager.processed_count} rows complete")

    # Save any remaining results
    if batch_results:
        checkpoint_manager.save_batch(batch_results, force=True)
        if logger:
            logger.info(f"Final checkpoint: {checkpoint_manager.processed_count} rows complete")

    return checkpoint_manager.finalize()
