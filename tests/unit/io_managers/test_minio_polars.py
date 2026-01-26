"""Unit tests for MinIO Polars I/O Manager.

Tests the MinIOPolarsIOManager for storing and loading Polars DataFrames
as Parquet files in MinIO. All external MinIO calls are mocked per INV-P010.

Note: Due to Dagster's ConfigurableIOManager requiring runtime type resolution,
and the I/O managers using TYPE_CHECKING imports, imports must happen inside
patch contexts to avoid forward reference resolution errors.
"""

from __future__ import annotations

import io
from typing import get_type_hints
from unittest.mock import MagicMock, patch

import polars as pl
from dagster import AssetKey, build_input_context, build_output_context


class TestMinIOPolarsIOManagerInit:
    """Tests for MinIOPolarsIOManager initialization."""

    @patch("brev_pipelines.resources.minio.Minio")
    def test_initialization_with_defaults(self, mock_minio_class: MagicMock):
        """Test manager initializes with default values."""
        mock_minio_class.return_value = MagicMock()

        # Import inside patch context to avoid forward reference issues
        from brev_pipelines.io_managers.minio_polars import MinIOPolarsIOManager
        from brev_pipelines.resources.minio import MinIOResource

        minio_resource = MinIOResource(
            endpoint="localhost:9000",
            access_key="test",
            secret_key="test",
        )

        manager = MinIOPolarsIOManager(minio=minio_resource)

        assert manager.minio is minio_resource
        assert manager.bucket == "dagster-io"
        assert manager.base_path == "assets"

    @patch("brev_pipelines.resources.minio.Minio")
    def test_initialization_with_custom_config(self, mock_minio_class: MagicMock):
        """Test manager initializes with custom configuration."""
        mock_minio_class.return_value = MagicMock()

        from brev_pipelines.io_managers.minio_polars import MinIOPolarsIOManager
        from brev_pipelines.resources.minio import MinIOResource

        minio_resource = MinIOResource(
            endpoint="localhost:9000",
            access_key="test",
            secret_key="test",
        )

        manager = MinIOPolarsIOManager(
            minio=minio_resource,
            bucket="custom-bucket",
            base_path="custom/path",
        )

        assert manager.bucket == "custom-bucket"
        assert manager.base_path == "custom/path"


class TestMinIOPolarsIOManagerHandleOutput:
    """Tests for handle_output method."""

    @patch("brev_pipelines.resources.minio.Minio")
    def test_handle_output_stores_dataframe(
        self, mock_minio_class: MagicMock, sample_speeches_df: pl.DataFrame
    ):
        """Test handle_output stores DataFrame as Parquet in MinIO."""
        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_minio_class.return_value = mock_client

        from brev_pipelines.io_managers.minio_polars import MinIOPolarsIOManager
        from brev_pipelines.resources.minio import MinIOResource

        minio_resource = MinIOResource(
            endpoint="localhost:9000",
            access_key="test",
            secret_key="test",
        )

        manager = MinIOPolarsIOManager(minio=minio_resource)

        context = build_output_context(asset_key=AssetKey(["test_asset"]))

        manager.handle_output(context, sample_speeches_df)

        # Verify put_object was called with correct arguments
        mock_client.put_object.assert_called_once()
        call_args = mock_client.put_object.call_args
        assert call_args.kwargs["bucket_name"] == "dagster-io"
        assert call_args.kwargs["object_name"] == "assets/test_asset.parquet"
        assert call_args.kwargs["content_type"] == "application/octet-stream"

    @patch("brev_pipelines.resources.minio.Minio")
    def test_handle_output_skips_none(self, mock_minio_class: MagicMock):
        """Test handle_output skips when obj is None."""
        mock_client = MagicMock()
        mock_minio_class.return_value = mock_client

        from brev_pipelines.io_managers.minio_polars import MinIOPolarsIOManager
        from brev_pipelines.resources.minio import MinIOResource

        minio_resource = MinIOResource(
            endpoint="localhost:9000",
            access_key="test",
            secret_key="test",
        )

        manager = MinIOPolarsIOManager(minio=minio_resource)
        context = build_output_context(asset_key=AssetKey(["test_asset"]))

        manager.handle_output(context, None)

        mock_client.put_object.assert_not_called()

    @patch("brev_pipelines.resources.minio.Minio")
    def test_handle_output_uses_asset_key_for_path(
        self, mock_minio_class: MagicMock, sample_speeches_df: pl.DataFrame
    ):
        """Test handle_output derives path from asset key."""
        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_minio_class.return_value = mock_client

        from brev_pipelines.io_managers.minio_polars import MinIOPolarsIOManager
        from brev_pipelines.resources.minio import MinIOResource

        minio_resource = MinIOResource(
            endpoint="localhost:9000",
            access_key="test",
            secret_key="test",
        )

        manager = MinIOPolarsIOManager(
            minio=minio_resource,
            base_path="data",
        )

        context = build_output_context(asset_key=AssetKey(["namespace", "my_asset"]))

        manager.handle_output(context, sample_speeches_df)

        call_args = mock_client.put_object.call_args
        assert call_args.kwargs["object_name"] == "data/my_asset.parquet"

    @patch("brev_pipelines.resources.minio.Minio")
    def test_handle_output_serializes_valid_parquet(
        self, mock_minio_class: MagicMock, sample_speeches_df: pl.DataFrame
    ):
        """Test handle_output creates valid Parquet data."""
        captured_data = {}

        def capture_put_object(**kwargs):
            captured_data["data"] = kwargs["data"].read()
            captured_data["length"] = kwargs["length"]

        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_client.put_object.side_effect = capture_put_object
        mock_minio_class.return_value = mock_client

        from brev_pipelines.io_managers.minio_polars import MinIOPolarsIOManager
        from brev_pipelines.resources.minio import MinIOResource

        minio_resource = MinIOResource(
            endpoint="localhost:9000",
            access_key="test",
            secret_key="test",
        )

        manager = MinIOPolarsIOManager(minio=minio_resource)
        context = build_output_context(asset_key=AssetKey(["test"]))

        manager.handle_output(context, sample_speeches_df)

        # Verify the captured data is valid Parquet
        df = pl.read_parquet(io.BytesIO(captured_data["data"]))
        assert len(df) == len(sample_speeches_df)
        assert df.columns == sample_speeches_df.columns


class TestMinIOPolarsIOManagerLoadInput:
    """Tests for load_input method."""

    @patch("brev_pipelines.resources.minio.Minio")
    def test_load_input_retrieves_dataframe(
        self, mock_minio_class: MagicMock, sample_speeches_df: pl.DataFrame
    ):
        """Test load_input retrieves DataFrame from MinIO."""
        # Create parquet bytes
        buffer = io.BytesIO()
        sample_speeches_df.write_parquet(buffer)
        parquet_bytes = buffer.getvalue()

        mock_response = MagicMock()
        mock_response.read.return_value = parquet_bytes
        mock_response.close.return_value = None
        mock_response.release_conn.return_value = None

        mock_client = MagicMock()
        mock_client.get_object.return_value = mock_response
        mock_minio_class.return_value = mock_client

        from brev_pipelines.io_managers.minio_polars import MinIOPolarsIOManager
        from brev_pipelines.resources.minio import MinIOResource

        minio_resource = MinIOResource(
            endpoint="localhost:9000",
            access_key="test",
            secret_key="test",
        )

        manager = MinIOPolarsIOManager(minio=minio_resource)
        context = build_input_context(asset_key=AssetKey(["test_asset"]))

        result = manager.load_input(context)

        assert len(result) == len(sample_speeches_df)
        assert result.columns == sample_speeches_df.columns

        # Verify get_object was called correctly
        mock_client.get_object.assert_called_once_with("dagster-io", "assets/test_asset.parquet")

    @patch("brev_pipelines.resources.minio.Minio")
    def test_load_input_uses_asset_key_for_path(
        self, mock_minio_class: MagicMock, sample_speeches_df: pl.DataFrame
    ):
        """Test load_input derives path from asset key."""
        buffer = io.BytesIO()
        sample_speeches_df.write_parquet(buffer)

        mock_response = MagicMock()
        mock_response.read.return_value = buffer.getvalue()
        mock_response.close.return_value = None
        mock_response.release_conn.return_value = None

        mock_client = MagicMock()
        mock_client.get_object.return_value = mock_response
        mock_minio_class.return_value = mock_client

        from brev_pipelines.io_managers.minio_polars import MinIOPolarsIOManager
        from brev_pipelines.resources.minio import MinIOResource

        minio_resource = MinIOResource(
            endpoint="localhost:9000",
            access_key="test",
            secret_key="test",
        )

        manager = MinIOPolarsIOManager(
            minio=minio_resource,
            base_path="custom",
        )
        context = build_input_context(asset_key=AssetKey(["namespace", "my_data"]))

        manager.load_input(context)

        mock_client.get_object.assert_called_once_with("dagster-io", "custom/my_data.parquet")

    @patch("brev_pipelines.resources.minio.Minio")
    def test_load_input_closes_response(
        self, mock_minio_class: MagicMock, sample_speeches_df: pl.DataFrame
    ):
        """Test load_input properly closes response connection."""
        buffer = io.BytesIO()
        sample_speeches_df.write_parquet(buffer)

        mock_response = MagicMock()
        mock_response.read.return_value = buffer.getvalue()

        mock_client = MagicMock()
        mock_client.get_object.return_value = mock_response
        mock_minio_class.return_value = mock_client

        from brev_pipelines.io_managers.minio_polars import MinIOPolarsIOManager
        from brev_pipelines.resources.minio import MinIOResource

        minio_resource = MinIOResource(
            endpoint="localhost:9000",
            access_key="test",
            secret_key="test",
        )

        manager = MinIOPolarsIOManager(minio=minio_resource)
        context = build_input_context(asset_key=AssetKey(["test"]))

        manager.load_input(context)

        mock_response.close.assert_called_once()
        mock_response.release_conn.assert_called_once()


class TestMinIOPolarsIOManagerIntegration:
    """Integration-style tests for round-trip storage."""

    @patch("brev_pipelines.resources.minio.Minio")
    def test_roundtrip_preserves_data(
        self, mock_minio_class: MagicMock, sample_speeches_df: pl.DataFrame
    ):
        """Test storing and loading preserves data integrity."""
        stored_data = {}

        def mock_put_object(**kwargs):
            stored_data["parquet"] = kwargs["data"].read()

        def mock_get_object(bucket, path):
            response = MagicMock()
            response.read.return_value = stored_data["parquet"]
            response.close.return_value = None
            response.release_conn.return_value = None
            return response

        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_client.put_object.side_effect = mock_put_object
        mock_client.get_object.side_effect = mock_get_object
        mock_minio_class.return_value = mock_client

        from brev_pipelines.io_managers.minio_polars import MinIOPolarsIOManager
        from brev_pipelines.resources.minio import MinIOResource

        minio_resource = MinIOResource(
            endpoint="localhost:9000",
            access_key="test",
            secret_key="test",
        )

        manager = MinIOPolarsIOManager(minio=minio_resource)

        # Store
        output_context = build_output_context(asset_key=AssetKey(["roundtrip_test"]))
        manager.handle_output(output_context, sample_speeches_df)

        # Load
        input_context = build_input_context(asset_key=AssetKey(["roundtrip_test"]))
        loaded_df = manager.load_input(input_context)

        # Verify data integrity
        assert loaded_df.shape == sample_speeches_df.shape
        assert loaded_df.columns == sample_speeches_df.columns
        assert loaded_df["reference"].to_list() == sample_speeches_df["reference"].to_list()

    @patch("brev_pipelines.resources.minio.Minio")
    def test_handles_complex_data_types(self, mock_minio_class: MagicMock):
        """Test handles DataFrames with various data types."""
        stored_data = {}

        def mock_put_object(**kwargs):
            stored_data["parquet"] = kwargs["data"].read()

        def mock_get_object(bucket, path):
            response = MagicMock()
            response.read.return_value = stored_data["parquet"]
            response.close.return_value = None
            response.release_conn.return_value = None
            return response

        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_client.put_object.side_effect = mock_put_object
        mock_client.get_object.side_effect = mock_get_object
        mock_minio_class.return_value = mock_client

        from brev_pipelines.io_managers.minio_polars import MinIOPolarsIOManager
        from brev_pipelines.resources.minio import MinIOResource

        minio_resource = MinIOResource(
            endpoint="localhost:9000",
            access_key="test",
            secret_key="test",
        )

        # DataFrame with various types
        complex_df = pl.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "str_col": ["a", "b", "c"],
                "bool_col": [True, False, True],
                "list_col": [[1, 2], [3, 4], [5, 6]],
            }
        )

        manager = MinIOPolarsIOManager(minio=minio_resource)

        output_context = build_output_context(asset_key=AssetKey(["complex"]))
        manager.handle_output(output_context, complex_df)

        input_context = build_input_context(asset_key=AssetKey(["complex"]))
        loaded_df = manager.load_input(input_context)

        assert loaded_df["int_col"].to_list() == [1, 2, 3]
        assert loaded_df["bool_col"].to_list() == [True, False, True]


class TestMinIOPolarsIOManagerTypeAnnotations:
    """Tests verifying type annotations follow standards."""

    def test_handle_output_has_type_annotations(self):
        """Test handle_output has proper type annotations."""
        with patch("brev_pipelines.resources.minio.Minio"):
            from brev_pipelines.io_managers.minio_polars import MinIOPolarsIOManager

            hints = get_type_hints(MinIOPolarsIOManager.handle_output)
            assert "context" in hints
            assert "obj" in hints
            assert "return" in hints

    def test_load_input_has_type_annotations(self):
        """Test load_input has proper type annotations."""
        with patch("brev_pipelines.resources.minio.Minio"):
            from brev_pipelines.io_managers.minio_polars import MinIOPolarsIOManager

            hints = get_type_hints(MinIOPolarsIOManager.load_input)
            assert "context" in hints
            assert "return" in hints

    def test_is_configurable_io_manager(self):
        """Test MinIOPolarsIOManager extends ConfigurableIOManager."""
        from dagster import ConfigurableIOManager

        with patch("brev_pipelines.resources.minio.Minio"):
            from brev_pipelines.io_managers.minio_polars import MinIOPolarsIOManager

            assert issubclass(MinIOPolarsIOManager, ConfigurableIOManager)
