"""MinIO I/O Manager for Polars DataFrames.

Stores DataFrames as Parquet files in MinIO for fast intermediate persistence.
Used for checkpointing and intermediate asset storage where LakeFS versioning
is not needed.

Follows INV-D003 (Parquet for Structured Data).
"""

from __future__ import annotations

import io

import polars as pl
from dagster import ConfigurableIOManager, InputContext, OutputContext
from minio.error import S3Error
from pydantic import Field

# Import at runtime - Dagster ConfigurableIOManager needs actual type for validation
from brev_pipelines.resources.minio import MinIOResource


class MinIOPolarsIOManager(ConfigurableIOManager):
    """I/O Manager for storing Polars DataFrames in MinIO as Parquet.

    Provides fast intermediate storage without the overhead of LakeFS versioning.
    Ideal for:
    - Intermediate pipeline assets
    - LLM processing checkpoints
    - Temporary data that doesn't need versioning

    Attributes:
        minio: MinIO resource for object storage.
        bucket: MinIO bucket name.
        base_path: Base path prefix for all objects.
    """

    minio: MinIOResource = Field(description="MinIO resource for storage")
    bucket: str = Field(default="dagster-io", description="MinIO bucket name")
    base_path: str = Field(default="assets", description="Base path prefix")

    def handle_output(self, context: OutputContext, obj: pl.DataFrame) -> None:
        """Store a Polars DataFrame to MinIO as Parquet.

        Args:
            context: Dagster output context with asset metadata.
            obj: Polars DataFrame to store.
        """
        if obj is None:
            context.log.warning("Received None object, skipping output")
            return

        # Ensure bucket exists
        self.minio.ensure_bucket(self.bucket)

        # Determine output path from asset key
        asset_key = context.asset_key.path[-1] if context.asset_key else "output"
        path = f"{self.base_path}/{asset_key}.parquet"

        # Serialize DataFrame to Parquet bytes
        buffer = io.BytesIO()
        obj.write_parquet(buffer)
        parquet_bytes = buffer.getvalue()

        # Upload to MinIO
        client = self.minio.get_client()
        client.put_object(
            bucket_name=self.bucket,
            object_name=path,
            data=io.BytesIO(parquet_bytes),
            length=len(parquet_bytes),
            content_type="application/octet-stream",
        )

        context.log.info(f"Stored {len(obj)} rows to minio://{self.bucket}/{path}")

    def load_input(self, context: InputContext) -> pl.DataFrame:
        """Load a Polars DataFrame from MinIO Parquet file.

        Args:
            context: Dagster input context with asset metadata.

        Returns:
            Loaded Polars DataFrame.

        Raises:
            FileNotFoundError: If the object does not exist in MinIO.
            RuntimeError: If MinIO returns an unexpected error.
        """
        # Determine input path from asset key
        asset_key = context.asset_key.path[-1] if context.asset_key else "input"
        path = f"{self.base_path}/{asset_key}.parquet"

        # Download from MinIO with error handling
        client = self.minio.get_client()
        try:
            response = client.get_object(self.bucket, path)
            try:
                data = response.read()
            finally:
                response.close()
                response.release_conn()
        except S3Error as e:
            if e.code == "NoSuchKey":
                raise FileNotFoundError(f"Object not found in MinIO: {self.bucket}/{path}") from e
            raise RuntimeError(f"MinIO error loading {path}: {e}") from e

        # Parse Parquet to DataFrame
        df = pl.read_parquet(io.BytesIO(data))
        context.log.info(f"Loaded {len(df)} rows from minio://{self.bucket}/{path}")
        return df
