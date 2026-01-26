"""Unit tests for LakeFS Polars I/O Manager.

Tests the LakeFSPolarsIOManager for storing and loading Polars DataFrames
as Parquet files in LakeFS with automatic versioning.
All external LakeFS calls are mocked per INV-P010.
"""

from __future__ import annotations

import io
from typing import get_type_hints
from unittest.mock import MagicMock, patch

import polars as pl
from dagster import AssetKey, build_input_context, build_output_context

from brev_pipelines.io_managers.lakefs_polars import LakeFSPolarsIOManager
from brev_pipelines.resources.lakefs import LakeFSResource
from brev_pipelines.resources.minio import MinIOResource


class TestLakeFSPolarsIOManagerInit:
    """Tests for LakeFSPolarsIOManager initialization."""

    @patch("brev_pipelines.resources.minio.Minio")
    @patch("lakefs_sdk.client.LakeFSClient")
    def test_initialization_with_defaults(
        self, mock_lakefs_class: MagicMock, mock_minio_class: MagicMock
    ):
        """Test manager initializes with default values."""
        mock_lakefs_class.return_value = MagicMock()
        mock_minio_class.return_value = MagicMock()

        lakefs_resource = LakeFSResource(
            endpoint="http://localhost:8000",
            access_key="test",
            secret_key="test",
        )
        minio_resource = MinIOResource(
            endpoint="localhost:9000",
            access_key="test",
            secret_key="test",
        )

        manager = LakeFSPolarsIOManager(
            lakefs=lakefs_resource,
            minio=minio_resource,
        )

        assert manager.lakefs is lakefs_resource
        assert manager.minio is minio_resource
        assert manager.repository == "data"
        assert manager.branch == "main"
        assert manager.base_path == "data-products"

    @patch("brev_pipelines.resources.minio.Minio")
    @patch("lakefs_sdk.client.LakeFSClient")
    def test_initialization_with_custom_config(
        self, mock_lakefs_class: MagicMock, mock_minio_class: MagicMock
    ):
        """Test manager initializes with custom configuration."""
        mock_lakefs_class.return_value = MagicMock()
        mock_minio_class.return_value = MagicMock()

        lakefs_resource = LakeFSResource(
            endpoint="http://localhost:8000",
            access_key="test",
            secret_key="test",
        )
        minio_resource = MinIOResource(
            endpoint="localhost:9000",
            access_key="test",
            secret_key="test",
        )

        manager = LakeFSPolarsIOManager(
            lakefs=lakefs_resource,
            minio=minio_resource,
            repository="custom-repo",
            branch="feature-branch",
            base_path="custom/path",
        )

        assert manager.repository == "custom-repo"
        assert manager.branch == "feature-branch"
        assert manager.base_path == "custom/path"


class TestLakeFSPolarsIOManagerHandleOutput:
    """Tests for handle_output method."""

    @patch("brev_pipelines.io_managers.lakefs_polars.CommitCreation")
    @patch("brev_pipelines.resources.minio.Minio")
    @patch("lakefs_sdk.client.LakeFSClient")
    def test_handle_output_stores_and_commits(
        self,
        mock_lakefs_class: MagicMock,
        mock_minio_class: MagicMock,
        mock_commit_creation: MagicMock,
        sample_speeches_df: pl.DataFrame,
    ):
        """Test handle_output uploads to LakeFS and creates commit."""
        mock_lakefs_client = MagicMock()
        mock_lakefs_class.return_value = mock_lakefs_client
        mock_minio_class.return_value = MagicMock()

        lakefs_resource = LakeFSResource(
            endpoint="http://localhost:8000",
            access_key="test",
            secret_key="test",
        )
        minio_resource = MinIOResource(
            endpoint="localhost:9000",
            access_key="test",
            secret_key="test",
        )

        manager = LakeFSPolarsIOManager(
            lakefs=lakefs_resource,
            minio=minio_resource,
        )

        context = build_output_context(
            asset_key=AssetKey(["test_asset"]),
            run_id="run-12345678",
        )

        manager.handle_output(context, sample_speeches_df)

        # Verify upload was called
        mock_lakefs_client.objects_api.upload_object.assert_called_once()
        upload_call = mock_lakefs_client.objects_api.upload_object.call_args
        assert upload_call.kwargs["repository"] == "data"
        assert upload_call.kwargs["branch"] == "main"
        assert upload_call.kwargs["path"] == "data-products/test_asset.parquet"

        # Verify commit was created
        mock_lakefs_client.commits_api.commit.assert_called_once()

    @patch("brev_pipelines.io_managers.lakefs_polars.CommitCreation")
    @patch("brev_pipelines.resources.minio.Minio")
    @patch("lakefs_sdk.client.LakeFSClient")
    def test_handle_output_skips_none(
        self,
        mock_lakefs_class: MagicMock,
        mock_minio_class: MagicMock,
        mock_commit_creation: MagicMock,
    ):
        """Test handle_output skips when obj is None."""
        mock_lakefs_client = MagicMock()
        mock_lakefs_class.return_value = mock_lakefs_client
        mock_minio_class.return_value = MagicMock()

        lakefs_resource = LakeFSResource(
            endpoint="http://localhost:8000",
            access_key="test",
            secret_key="test",
        )
        minio_resource = MinIOResource(
            endpoint="localhost:9000",
            access_key="test",
            secret_key="test",
        )

        manager = LakeFSPolarsIOManager(
            lakefs=lakefs_resource,
            minio=minio_resource,
        )

        context = build_output_context(asset_key=AssetKey(["test_asset"]))

        manager.handle_output(context, None)

        mock_lakefs_client.objects_api.upload_object.assert_not_called()
        mock_lakefs_client.commits_api.commit.assert_not_called()

    @patch("brev_pipelines.io_managers.lakefs_polars.CommitCreation")
    @patch("brev_pipelines.resources.minio.Minio")
    @patch("lakefs_sdk.client.LakeFSClient")
    def test_handle_output_uses_asset_key_for_path(
        self,
        mock_lakefs_class: MagicMock,
        mock_minio_class: MagicMock,
        mock_commit_creation: MagicMock,
        sample_speeches_df: pl.DataFrame,
    ):
        """Test handle_output derives path from asset key."""
        mock_lakefs_client = MagicMock()
        mock_lakefs_class.return_value = mock_lakefs_client
        mock_minio_class.return_value = MagicMock()

        lakefs_resource = LakeFSResource(
            endpoint="http://localhost:8000",
            access_key="test",
            secret_key="test",
        )
        minio_resource = MinIOResource(
            endpoint="localhost:9000",
            access_key="test",
            secret_key="test",
        )

        manager = LakeFSPolarsIOManager(
            lakefs=lakefs_resource,
            minio=minio_resource,
            base_path="products",
        )

        context = build_output_context(
            asset_key=AssetKey(["namespace", "speeches_data"]),
        )

        manager.handle_output(context, sample_speeches_df)

        upload_call = mock_lakefs_client.objects_api.upload_object.call_args
        assert upload_call.kwargs["path"] == "products/speeches_data.parquet"

    @patch("brev_pipelines.io_managers.lakefs_polars.CommitCreation")
    @patch("brev_pipelines.resources.minio.Minio")
    @patch("lakefs_sdk.client.LakeFSClient")
    def test_handle_output_commit_includes_metadata(
        self,
        mock_lakefs_class: MagicMock,
        mock_minio_class: MagicMock,
        mock_commit_creation: MagicMock,
        sample_speeches_df: pl.DataFrame,
    ):
        """Test commit includes metadata about the data."""
        mock_lakefs_client = MagicMock()
        mock_lakefs_class.return_value = mock_lakefs_client
        mock_minio_class.return_value = MagicMock()

        lakefs_resource = LakeFSResource(
            endpoint="http://localhost:8000",
            access_key="test",
            secret_key="test",
        )
        minio_resource = MinIOResource(
            endpoint="localhost:9000",
            access_key="test",
            secret_key="test",
        )

        manager = LakeFSPolarsIOManager(
            lakefs=lakefs_resource,
            minio=minio_resource,
        )

        context = build_output_context(
            asset_key=AssetKey(["speeches"]),
            run_id="run-abc12345",
        )

        manager.handle_output(context, sample_speeches_df)

        # Verify CommitCreation was called with metadata
        mock_commit_creation.assert_called_once()
        call_kwargs = mock_commit_creation.call_args.kwargs
        assert "metadata" in call_kwargs
        assert call_kwargs["metadata"]["asset_key"] == "speeches"
        assert call_kwargs["metadata"]["num_rows"] == "3"
        assert call_kwargs["metadata"]["dagster_run_id"] == "run-abc12345"


class TestLakeFSPolarsIOManagerLoadInput:
    """Tests for load_input method."""

    @patch("brev_pipelines.resources.minio.Minio")
    @patch("lakefs_sdk.client.LakeFSClient")
    def test_load_input_retrieves_dataframe(
        self,
        mock_lakefs_class: MagicMock,
        mock_minio_class: MagicMock,
        sample_speeches_df: pl.DataFrame,
    ):
        """Test load_input retrieves DataFrame from LakeFS."""
        buffer = io.BytesIO()
        sample_speeches_df.write_parquet(buffer)
        parquet_bytes = bytearray(buffer.getvalue())

        # LakeFS SDK returns bytearray directly from get_object
        mock_lakefs_client = MagicMock()
        mock_lakefs_client.objects_api.get_object.return_value = parquet_bytes
        mock_lakefs_class.return_value = mock_lakefs_client
        mock_minio_class.return_value = MagicMock()

        lakefs_resource = LakeFSResource(
            endpoint="http://localhost:8000",
            access_key="test",
            secret_key="test",
        )
        minio_resource = MinIOResource(
            endpoint="localhost:9000",
            access_key="test",
            secret_key="test",
        )

        manager = LakeFSPolarsIOManager(
            lakefs=lakefs_resource,
            minio=minio_resource,
        )

        context = build_input_context(asset_key=AssetKey(["test_asset"]))

        result = manager.load_input(context)

        assert len(result) == len(sample_speeches_df)
        assert result.columns == sample_speeches_df.columns

        # Verify get_object was called correctly
        mock_lakefs_client.objects_api.get_object.assert_called_once_with(
            repository="data",
            ref="main",
            path="data-products/test_asset.parquet",
        )

    @patch("brev_pipelines.resources.minio.Minio")
    @patch("lakefs_sdk.client.LakeFSClient")
    def test_load_input_uses_asset_key_for_path(
        self,
        mock_lakefs_class: MagicMock,
        mock_minio_class: MagicMock,
        sample_speeches_df: pl.DataFrame,
    ):
        """Test load_input derives path from asset key."""
        buffer = io.BytesIO()
        sample_speeches_df.write_parquet(buffer)
        parquet_bytes = bytearray(buffer.getvalue())

        # LakeFS SDK returns bytearray directly from get_object
        mock_lakefs_client = MagicMock()
        mock_lakefs_client.objects_api.get_object.return_value = parquet_bytes
        mock_lakefs_class.return_value = mock_lakefs_client
        mock_minio_class.return_value = MagicMock()

        lakefs_resource = LakeFSResource(
            endpoint="http://localhost:8000",
            access_key="test",
            secret_key="test",
        )
        minio_resource = MinIOResource(
            endpoint="localhost:9000",
            access_key="test",
            secret_key="test",
        )

        manager = LakeFSPolarsIOManager(
            lakefs=lakefs_resource,
            minio=minio_resource,
            repository="speeches-repo",
            branch="production",
            base_path="outputs",
        )

        context = build_input_context(asset_key=AssetKey(["namespace", "my_data"]))

        manager.load_input(context)

        mock_lakefs_client.objects_api.get_object.assert_called_once_with(
            repository="speeches-repo",
            ref="production",
            path="outputs/my_data.parquet",
        )


class TestLakeFSPolarsIOManagerTypeAnnotations:
    """Tests verifying type annotations follow standards."""

    def test_handle_output_has_type_annotations(self):
        """Test handle_output has proper type annotations."""
        hints = get_type_hints(LakeFSPolarsIOManager.handle_output)
        assert "context" in hints
        assert "obj" in hints
        assert "return" in hints

    def test_load_input_has_type_annotations(self):
        """Test load_input has proper type annotations."""
        hints = get_type_hints(LakeFSPolarsIOManager.load_input)
        assert "context" in hints
        assert "return" in hints

    def test_is_configurable_io_manager(self):
        """Test LakeFSPolarsIOManager extends ConfigurableIOManager."""
        from dagster import ConfigurableIOManager

        assert issubclass(LakeFSPolarsIOManager, ConfigurableIOManager)

    def test_field_descriptions_present(self):
        """Test all fields have descriptions."""
        # Check that the model has field descriptions
        assert LakeFSPolarsIOManager.model_fields["repository"].description is not None
        assert LakeFSPolarsIOManager.model_fields["branch"].description is not None
        assert LakeFSPolarsIOManager.model_fields["base_path"].description is not None
