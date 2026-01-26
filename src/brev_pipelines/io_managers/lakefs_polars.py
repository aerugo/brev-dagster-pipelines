"""LakeFS I/O Manager for Polars DataFrames.

Stores DataFrames as Parquet files in LakeFS with automatic versioning.
Follows INV-D002 (LakeFS for Data Versioning) and INV-D003 (Parquet for Structured Data).
"""

from __future__ import annotations

import io

import polars as pl
from dagster import ConfigurableIOManager, InputContext, OutputContext
from lakefs_sdk.models import CommitCreation  # type: ignore[attr-defined]
from pydantic import Field

# Import at runtime - Dagster ConfigurableIOManager needs actual type for validation
from brev_pipelines.resources.lakefs import LakeFSResource
from brev_pipelines.resources.minio import MinIOResource


class LakeFSPolarsIOManager(ConfigurableIOManager):
    """I/O Manager for storing Polars DataFrames in LakeFS as Parquet.

    Automatically handles:
    - Parquet serialization/deserialization
    - LakeFS branch management
    - Commit creation with metadata

    Attributes:
        lakefs: LakeFS resource for versioning operations.
        minio: MinIO resource for underlying storage.
        repository: LakeFS repository name.
        branch: LakeFS branch name.
        base_path: Base path in repository for data products.
    """

    lakefs: LakeFSResource = Field(description="LakeFS resource")
    minio: MinIOResource = Field(description="MinIO resource for underlying storage")
    repository: str = Field(default="data", description="LakeFS repository name")
    branch: str = Field(default="main", description="LakeFS branch name")
    base_path: str = Field(default="data-products", description="Base path in repository")

    def handle_output(self, context: OutputContext, obj: pl.DataFrame) -> None:
        """Store a Polars DataFrame to LakeFS as Parquet.

        Args:
            context: Dagster output context with asset metadata.
            obj: Polars DataFrame to store.
        """
        if obj is None:
            context.log.warning("Received None object, skipping output")
            return

        # Determine output path from asset key
        asset_key = context.asset_key.path[-1] if context.asset_key else "output"
        path = f"{self.base_path}/{asset_key}.parquet"

        # Serialize DataFrame to Parquet bytes
        buffer = io.BytesIO()
        obj.write_parquet(buffer)
        parquet_bytes = buffer.getvalue()

        # Get LakeFS client
        lakefs_client = self.lakefs.get_client()

        # Upload to LakeFS
        lakefs_client.objects_api.upload_object(
            repository=self.repository,
            branch=self.branch,
            path=path,
            content=parquet_bytes,
        )

        # Create commit - handle case where data hasn't changed
        commit_message = f"Update {asset_key} data product"
        if context.run_id:
            commit_message += f" (Dagster run: {context.run_id[:8]})"

        try:
            lakefs_client.commits_api.commit(
                repository=self.repository,
                branch=self.branch,
                commit_creation=CommitCreation(
                    message=commit_message,
                    metadata={
                        "dagster_run_id": context.run_id or "",
                        "asset_key": asset_key,
                        "num_rows": str(len(obj)),
                        "num_columns": str(len(obj.columns)),
                    },
                    date=None,
                    allow_empty=False,
                ),
            )
            context.log.info(
                f"Stored {len(obj)} rows to lakefs://{self.repository}/{self.branch}/{path}"
            )
        except Exception as e:
            # LakeFS returns error when there are no changes to commit
            if "no changes" in str(e).lower():
                context.log.info(f"No changes to commit for {asset_key} (data unchanged)")
            else:
                raise

    def load_input(self, context: InputContext) -> pl.DataFrame:
        """Load a Polars DataFrame from LakeFS Parquet file.

        Args:
            context: Dagster input context with asset metadata.

        Returns:
            Loaded Polars DataFrame.
        """
        # Determine input path from asset key
        asset_key = context.asset_key.path[-1] if context.asset_key else "input"
        path = f"{self.base_path}/{asset_key}.parquet"

        # Get LakeFS client
        lakefs_client = self.lakefs.get_client()

        # Download from LakeFS
        response = lakefs_client.objects_api.get_object(
            repository=self.repository,
            ref=self.branch,
            path=path,
        )

        # Parse Parquet to DataFrame (response is a bytearray)
        df = pl.read_parquet(io.BytesIO(response))
        context.log.info(
            f"Loaded {len(df)} rows from lakefs://{self.repository}/{self.branch}/{path}"
        )
        return df
