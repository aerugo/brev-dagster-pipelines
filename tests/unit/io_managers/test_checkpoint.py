"""Unit tests for LLM Checkpoint Manager.

Tests the LLMCheckpointManager and process_with_checkpoint function
for partial result persistence during expensive LLM operations.
All external MinIO calls are mocked per INV-P010.
"""

from __future__ import annotations

import io
from typing import Any, get_type_hints
from unittest.mock import MagicMock, patch

import polars as pl

from brev_pipelines.io_managers.checkpoint import (
    LLMCheckpointManager,
    process_with_checkpoint,
)


class TestLLMCheckpointManagerInit:
    """Tests for LLMCheckpointManager initialization."""

    @patch("brev_pipelines.resources.minio.Minio")
    def test_initialization_with_required_args(self, mock_minio_class: MagicMock):
        """Test manager initializes with required arguments."""
        from brev_pipelines.resources.minio import MinIOResource

        mock_minio_class.return_value = MagicMock()

        minio_resource = MinIOResource(
            endpoint="localhost:9000",
            access_key="test",
            secret_key="test",
        )

        manager = LLMCheckpointManager(
            minio=minio_resource,
            asset_name="test_asset",
            run_id="run-123",
        )

        assert manager.minio is minio_resource
        assert manager.asset_name == "test_asset"
        assert manager.run_id == "run-123"
        assert manager.bucket == "dagster-checkpoints"  # default
        assert manager.checkpoint_interval == 10  # default

    @patch("brev_pipelines.resources.minio.Minio")
    def test_initialization_with_custom_config(self, mock_minio_class: MagicMock):
        """Test manager initializes with custom configuration."""
        from brev_pipelines.resources.minio import MinIOResource

        mock_minio_class.return_value = MagicMock()

        minio_resource = MinIOResource(
            endpoint="localhost:9000",
            access_key="test",
            secret_key="test",
        )

        manager = LLMCheckpointManager(
            minio=minio_resource,
            asset_name="classification",
            run_id="run-456",
            bucket="custom-checkpoints",
            checkpoint_interval=50,
        )

        assert manager.bucket == "custom-checkpoints"
        assert manager.checkpoint_interval == 50

    @patch("brev_pipelines.resources.minio.Minio")
    def test_checkpoint_path_property(self, mock_minio_class: MagicMock):
        """Test checkpoint_path returns correct path."""
        from brev_pipelines.resources.minio import MinIOResource

        mock_minio_class.return_value = MagicMock()

        minio_resource = MinIOResource(
            endpoint="localhost:9000",
            access_key="test",
            secret_key="test",
        )

        manager = LLMCheckpointManager(
            minio=minio_resource,
            asset_name="speech_classification",
            run_id="run-abc123",
        )

        expected_path = "checkpoints/speech_classification/run-abc123.parquet"
        assert manager.checkpoint_path == expected_path

    @patch("brev_pipelines.resources.minio.Minio")
    def test_private_attributes_initialized(self, mock_minio_class: MagicMock):
        """Test private attributes are properly initialized."""
        from brev_pipelines.resources.minio import MinIOResource

        mock_minio_class.return_value = MagicMock()

        minio_resource = MinIOResource(
            endpoint="localhost:9000",
            access_key="test",
            secret_key="test",
        )

        manager = LLMCheckpointManager(
            minio=minio_resource,
            asset_name="test",
            run_id="run-123",
        )

        # Access private attributes (Pydantic v2 PrivateAttr)
        assert manager._accumulated_results == []
        assert manager._total_saved == 0


class TestLLMCheckpointManagerLoad:
    """Tests for checkpoint loading functionality."""

    @patch("brev_pipelines.resources.minio.Minio")
    def test_load_returns_none_when_no_checkpoint(self, mock_minio_class: MagicMock):
        """Test load returns None when no checkpoint exists."""
        from brev_pipelines.resources.minio import MinIOResource

        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_client.get_object.side_effect = Exception("NoSuchKey")
        mock_minio_class.return_value = mock_client

        minio_resource = MinIOResource(
            endpoint="localhost:9000",
            access_key="test",
            secret_key="test",
        )

        manager = LLMCheckpointManager(
            minio=minio_resource,
            asset_name="test",
            run_id="run-123",
        )

        result = manager.load()

        assert result is None

    @patch("brev_pipelines.resources.minio.Minio")
    def test_load_returns_dataframe_when_checkpoint_exists(
        self, mock_minio_class: MagicMock, sample_speeches_df: pl.DataFrame
    ):
        """Test load returns DataFrame when checkpoint exists."""
        from brev_pipelines.resources.minio import MinIOResource

        # Create parquet bytes from sample DataFrame
        buffer = io.BytesIO()
        sample_speeches_df.write_parquet(buffer)
        parquet_bytes = buffer.getvalue()

        mock_response = MagicMock()
        mock_response.read.return_value = parquet_bytes
        mock_response.close.return_value = None
        mock_response.release_conn.return_value = None

        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_client.get_object.return_value = mock_response
        mock_minio_class.return_value = mock_client

        minio_resource = MinIOResource(
            endpoint="localhost:9000",
            access_key="test",
            secret_key="test",
        )

        manager = LLMCheckpointManager(
            minio=minio_resource,
            asset_name="test",
            run_id="run-123",
        )

        result = manager.load()

        assert result is not None
        assert len(result) == len(sample_speeches_df)
        assert manager._total_saved == len(sample_speeches_df)

    @patch("brev_pipelines.resources.minio.Minio")
    def test_load_updates_total_saved_count(
        self, mock_minio_class: MagicMock, sample_speeches_df: pl.DataFrame
    ):
        """Test load updates _total_saved count."""
        from brev_pipelines.resources.minio import MinIOResource

        buffer = io.BytesIO()
        sample_speeches_df.write_parquet(buffer)

        mock_response = MagicMock()
        mock_response.read.return_value = buffer.getvalue()
        mock_response.close.return_value = None
        mock_response.release_conn.return_value = None

        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_client.get_object.return_value = mock_response
        mock_minio_class.return_value = mock_client

        minio_resource = MinIOResource(
            endpoint="localhost:9000",
            access_key="test",
            secret_key="test",
        )

        manager = LLMCheckpointManager(
            minio=minio_resource,
            asset_name="test",
            run_id="run-123",
        )

        manager.load()

        assert manager.processed_count == 3  # sample_speeches_df has 3 rows


class TestLLMCheckpointManagerSaveBatch:
    """Tests for batch saving functionality."""

    @patch("brev_pipelines.resources.minio.Minio")
    def test_save_batch_accumulates_results(self, mock_minio_class: MagicMock):
        """Test save_batch accumulates results without saving when under interval."""
        from brev_pipelines.resources.minio import MinIOResource

        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_minio_class.return_value = mock_client

        minio_resource = MinIOResource(
            endpoint="localhost:9000",
            access_key="test",
            secret_key="test",
        )

        manager = LLMCheckpointManager(
            minio=minio_resource,
            asset_name="test",
            run_id="run-123",
            checkpoint_interval=10,
        )

        results = [{"id": "1", "value": "a"}, {"id": "2", "value": "b"}]
        manager.save_batch(results)

        assert len(manager._accumulated_results) == 2
        # Should not have called put_object yet
        mock_client.put_object.assert_not_called()

    @patch("brev_pipelines.resources.minio.Minio")
    def test_save_batch_flushes_at_interval(self, mock_minio_class: MagicMock):
        """Test save_batch flushes when checkpoint_interval is reached."""
        from brev_pipelines.resources.minio import MinIOResource

        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_client.get_object.side_effect = Exception("NoSuchKey")
        mock_minio_class.return_value = mock_client

        minio_resource = MinIOResource(
            endpoint="localhost:9000",
            access_key="test",
            secret_key="test",
        )

        manager = LLMCheckpointManager(
            minio=minio_resource,
            asset_name="test",
            run_id="run-123",
            checkpoint_interval=5,
        )

        # Add 5 results to trigger flush
        results = [{"id": str(i), "value": f"val{i}"} for i in range(5)]
        manager.save_batch(results)

        # Should have flushed
        mock_client.put_object.assert_called_once()
        assert manager._accumulated_results == []

    @patch("brev_pipelines.resources.minio.Minio")
    def test_save_batch_force_flushes(self, mock_minio_class: MagicMock):
        """Test save_batch with force=True flushes immediately."""
        from brev_pipelines.resources.minio import MinIOResource

        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_client.get_object.side_effect = Exception("NoSuchKey")
        mock_minio_class.return_value = mock_client

        minio_resource = MinIOResource(
            endpoint="localhost:9000",
            access_key="test",
            secret_key="test",
        )

        manager = LLMCheckpointManager(
            minio=minio_resource,
            asset_name="test",
            run_id="run-123",
            checkpoint_interval=100,  # High interval
        )

        results = [{"id": "1", "value": "a"}]
        manager.save_batch(results, force=True)

        mock_client.put_object.assert_called_once()
        assert manager._accumulated_results == []


class TestLLMCheckpointManagerFlush:
    """Tests for _flush_checkpoint functionality."""

    @patch("brev_pipelines.resources.minio.Minio")
    def test_flush_does_nothing_when_empty(self, mock_minio_class: MagicMock):
        """Test _flush_checkpoint does nothing with no accumulated results."""
        from brev_pipelines.resources.minio import MinIOResource

        mock_client = MagicMock()
        mock_minio_class.return_value = mock_client

        minio_resource = MinIOResource(
            endpoint="localhost:9000",
            access_key="test",
            secret_key="test",
        )

        manager = LLMCheckpointManager(
            minio=minio_resource,
            asset_name="test",
            run_id="run-123",
        )

        manager._flush_checkpoint()

        mock_client.put_object.assert_not_called()

    @patch("brev_pipelines.resources.minio.Minio")
    def test_flush_combines_with_existing_checkpoint(self, mock_minio_class: MagicMock):
        """Test _flush_checkpoint combines new results with existing checkpoint."""
        from brev_pipelines.resources.minio import MinIOResource

        # Setup existing checkpoint data
        existing_df = pl.DataFrame({"id": ["1", "2"], "value": ["a", "b"]})
        buffer = io.BytesIO()
        existing_df.write_parquet(buffer)

        mock_response = MagicMock()
        mock_response.read.return_value = buffer.getvalue()
        mock_response.close.return_value = None
        mock_response.release_conn.return_value = None

        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_client.get_object.return_value = mock_response
        mock_minio_class.return_value = mock_client

        minio_resource = MinIOResource(
            endpoint="localhost:9000",
            access_key="test",
            secret_key="test",
        )

        manager = LLMCheckpointManager(
            minio=minio_resource,
            asset_name="test",
            run_id="run-123",
        )

        # Add new results
        manager._accumulated_results = [{"id": "3", "value": "c"}]
        manager._flush_checkpoint()

        # Verify put_object was called
        mock_client.put_object.assert_called_once()

        # Verify combined count
        assert manager._total_saved == 3


class TestLLMCheckpointManagerFinalize:
    """Tests for finalize functionality."""

    @patch("brev_pipelines.resources.minio.Minio")
    def test_finalize_flushes_and_returns_df(self, mock_minio_class: MagicMock):
        """Test finalize flushes remaining results and returns final DataFrame."""
        from brev_pipelines.resources.minio import MinIOResource

        # Setup so load returns data after flush
        final_df = pl.DataFrame({"id": ["1", "2"], "value": ["a", "b"]})
        buffer = io.BytesIO()
        final_df.write_parquet(buffer)

        # First call fails (no existing), second call returns data after flush
        mock_response = MagicMock()
        mock_response.read.return_value = buffer.getvalue()
        mock_response.close.return_value = None
        mock_response.release_conn.return_value = None

        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        # First call for flush (no existing), second for load
        mock_client.get_object.side_effect = [
            Exception("NoSuchKey"),
            mock_response,
        ]
        mock_minio_class.return_value = mock_client

        minio_resource = MinIOResource(
            endpoint="localhost:9000",
            access_key="test",
            secret_key="test",
        )

        manager = LLMCheckpointManager(
            minio=minio_resource,
            asset_name="test",
            run_id="run-123",
        )

        manager._accumulated_results = [{"id": "1", "value": "a"}, {"id": "2", "value": "b"}]
        result = manager.finalize()

        assert result is not None
        assert len(result) == 2


class TestLLMCheckpointManagerCleanup:
    """Tests for cleanup functionality."""

    @patch("brev_pipelines.resources.minio.Minio")
    def test_cleanup_removes_checkpoint_file(self, mock_minio_class: MagicMock):
        """Test cleanup removes checkpoint file from MinIO."""
        from brev_pipelines.resources.minio import MinIOResource

        mock_client = MagicMock()
        mock_minio_class.return_value = mock_client

        minio_resource = MinIOResource(
            endpoint="localhost:9000",
            access_key="test",
            secret_key="test",
        )

        manager = LLMCheckpointManager(
            minio=minio_resource,
            asset_name="test",
            run_id="run-123",
        )

        manager.cleanup()

        mock_client.remove_object.assert_called_once_with(
            "dagster-checkpoints",
            "checkpoints/test/run-123.parquet",
        )

    @patch("brev_pipelines.resources.minio.Minio")
    def test_cleanup_ignores_nonexistent_file(self, mock_minio_class: MagicMock):
        """Test cleanup ignores errors when file doesn't exist."""
        from brev_pipelines.resources.minio import MinIOResource

        mock_client = MagicMock()
        mock_client.remove_object.side_effect = Exception("NoSuchKey")
        mock_minio_class.return_value = mock_client

        minio_resource = MinIOResource(
            endpoint="localhost:9000",
            access_key="test",
            secret_key="test",
        )

        manager = LLMCheckpointManager(
            minio=minio_resource,
            asset_name="test",
            run_id="run-123",
        )

        # Should not raise
        manager.cleanup()


class TestProcessWithCheckpoint:
    """Tests for process_with_checkpoint helper function."""

    @patch("brev_pipelines.resources.minio.Minio")
    def test_process_all_rows_when_no_checkpoint(
        self,
        mock_minio_class: MagicMock,
        sample_speeches_df: pl.DataFrame,
    ):
        """Test all rows are processed when no checkpoint exists."""
        from brev_pipelines.resources.minio import MinIOResource

        # Stateful mock to track stored data
        stored_data: dict[str, bytes] = {}

        def mock_put_object(**kwargs):
            stored_data[kwargs["object_name"]] = kwargs["data"].read()

        def mock_get_object(bucket, path):
            if path not in stored_data:
                raise Exception("NoSuchKey")
            response = MagicMock()
            response.read.return_value = stored_data[path]
            response.close.return_value = None
            response.release_conn.return_value = None
            return response

        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_client.put_object.side_effect = mock_put_object
        mock_client.get_object.side_effect = mock_get_object
        mock_minio_class.return_value = mock_client

        minio_resource = MinIOResource(
            endpoint="localhost:9000",
            access_key="test",
            secret_key="test",
        )

        processed_rows: list[str] = []

        def process_fn(row: dict[str, Any]) -> dict[str, Any]:
            processed_rows.append(row["reference"])
            return {"reference": row["reference"], "processed": True}

        manager = LLMCheckpointManager(
            minio=minio_resource,
            asset_name="test",
            run_id="run-123",
            checkpoint_interval=100,
        )

        result = process_with_checkpoint(
            df=sample_speeches_df,
            id_column="reference",
            process_fn=process_fn,
            checkpoint_manager=manager,
            batch_size=10,
        )

        # Verify all rows were processed
        assert len(processed_rows) == 3
        assert "BIS_2024_001" in processed_rows
        assert "BIS_2024_002" in processed_rows
        assert "ECB_2024_001" in processed_rows
        # Result should contain processed data
        assert len(result) == 3

    @patch("brev_pipelines.resources.minio.Minio")
    def test_process_skips_already_processed_rows(
        self, mock_minio_class: MagicMock, sample_speeches_df: pl.DataFrame
    ):
        """Test already processed rows are skipped when checkpoint exists."""
        from brev_pipelines.resources.minio import MinIOResource

        # Setup existing checkpoint with 2 rows
        existing_df = pl.DataFrame(
            {"reference": ["BIS_2024_001", "BIS_2024_002"], "processed": [True, True]}
        )
        buffer = io.BytesIO()
        existing_df.write_parquet(buffer)
        initial_checkpoint = buffer.getvalue()

        # Stateful mock to track stored data
        stored_data: dict[str, bytes] = {"checkpoints/test/run-123.parquet": initial_checkpoint}

        def mock_put_object(**kwargs):
            stored_data[kwargs["object_name"]] = kwargs["data"].read()

        def mock_get_object(bucket, path):
            if path not in stored_data:
                raise Exception("NoSuchKey")
            response = MagicMock()
            response.read.return_value = stored_data[path]
            response.close.return_value = None
            response.release_conn.return_value = None
            return response

        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_client.put_object.side_effect = mock_put_object
        mock_client.get_object.side_effect = mock_get_object
        mock_minio_class.return_value = mock_client

        minio_resource = MinIOResource(
            endpoint="localhost:9000",
            access_key="test",
            secret_key="test",
        )

        processed_rows: list[str] = []

        def process_fn(row: dict[str, Any]) -> dict[str, Any]:
            processed_rows.append(row["reference"])
            return {"reference": row["reference"], "processed": True}

        manager = LLMCheckpointManager(
            minio=minio_resource,
            asset_name="test",
            run_id="run-123",
            checkpoint_interval=100,
        )

        result = process_with_checkpoint(
            df=sample_speeches_df,
            id_column="reference",
            process_fn=process_fn,
            checkpoint_manager=manager,
            batch_size=10,
        )

        # Only the third row should be processed
        assert len(processed_rows) == 1
        assert processed_rows[0] == "ECB_2024_001"
        # Result should include all 3 rows (2 from checkpoint + 1 new)
        assert len(result) == 3

    @patch("brev_pipelines.resources.minio.Minio")
    def test_process_returns_existing_when_all_processed(
        self, mock_minio_class: MagicMock, sample_speeches_df: pl.DataFrame
    ):
        """Test returns existing checkpoint when all rows already processed."""
        from brev_pipelines.resources.minio import MinIOResource

        # All 3 rows already in checkpoint
        existing_df = pl.DataFrame(
            {
                "reference": ["BIS_2024_001", "BIS_2024_002", "ECB_2024_001"],
                "processed": [True, True, True],
            }
        )
        buffer = io.BytesIO()
        existing_df.write_parquet(buffer)

        mock_response = MagicMock()
        mock_response.read.return_value = buffer.getvalue()
        mock_response.close.return_value = None
        mock_response.release_conn.return_value = None

        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_client.get_object.return_value = mock_response
        mock_minio_class.return_value = mock_client

        minio_resource = MinIOResource(
            endpoint="localhost:9000",
            access_key="test",
            secret_key="test",
        )

        process_fn = MagicMock()

        manager = LLMCheckpointManager(
            minio=minio_resource,
            asset_name="test",
            run_id="run-123",
        )

        result = process_with_checkpoint(
            df=sample_speeches_df,
            id_column="reference",
            process_fn=process_fn,
            checkpoint_manager=manager,
        )

        # process_fn should never be called
        process_fn.assert_not_called()
        assert len(result) == 3

    @patch("brev_pipelines.resources.minio.Minio")
    def test_process_saves_checkpoints_at_batch_boundaries(self, mock_minio_class: MagicMock):
        """Test checkpoints are saved at batch boundaries."""
        from brev_pipelines.resources.minio import MinIOResource

        # Stateful mock to track stored data
        stored_data: dict[str, bytes] = {}
        put_call_count = 0

        def mock_put_object(**kwargs):
            nonlocal put_call_count
            put_call_count += 1
            stored_data[kwargs["object_name"]] = kwargs["data"].read()

        def mock_get_object(bucket, path):
            if path not in stored_data:
                raise Exception("NoSuchKey")
            response = MagicMock()
            response.read.return_value = stored_data[path]
            response.close.return_value = None
            response.release_conn.return_value = None
            return response

        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_client.put_object.side_effect = mock_put_object
        mock_client.get_object.side_effect = mock_get_object
        mock_minio_class.return_value = mock_client

        minio_resource = MinIOResource(
            endpoint="localhost:9000",
            access_key="test",
            secret_key="test",
        )

        df = pl.DataFrame({"id": [str(i) for i in range(5)]})

        def process_fn(row: dict[str, Any]) -> dict[str, Any]:
            return {"id": row["id"], "processed": True}

        manager = LLMCheckpointManager(
            minio=minio_resource,
            asset_name="test",
            run_id="run-123",
            checkpoint_interval=100,  # High to not auto-flush
        )

        result = process_with_checkpoint(
            df=df,
            id_column="id",
            process_fn=process_fn,
            checkpoint_manager=manager,
            batch_size=2,  # Save every 2 rows
        )

        # Verify all rows processed and final result correct
        assert len(result) == 5
        assert result["id"].to_list() == ["0", "1", "2", "3", "4"]
        # Should have saved 3 times: batches of 2, 2, and 1
        assert put_call_count == 3

    @patch("brev_pipelines.resources.minio.Minio")
    def test_process_with_logger(
        self, mock_minio_class: MagicMock, sample_speeches_df: pl.DataFrame
    ):
        """Test process_with_checkpoint logs progress when logger provided."""
        from brev_pipelines.resources.minio import MinIOResource

        # Stateful mock to track stored data
        stored_data: dict[str, bytes] = {}

        def mock_put_object(**kwargs):
            stored_data[kwargs["object_name"]] = kwargs["data"].read()

        def mock_get_object(bucket, path):
            if path not in stored_data:
                raise Exception("NoSuchKey")
            response = MagicMock()
            response.read.return_value = stored_data[path]
            response.close.return_value = None
            response.release_conn.return_value = None
            return response

        mock_client = MagicMock()
        mock_client.bucket_exists.return_value = True
        mock_client.put_object.side_effect = mock_put_object
        mock_client.get_object.side_effect = mock_get_object
        mock_minio_class.return_value = mock_client

        minio_resource = MinIOResource(
            endpoint="localhost:9000",
            access_key="test",
            secret_key="test",
        )

        mock_logger = MagicMock()

        def process_fn(row: dict[str, Any]) -> dict[str, Any]:
            return {"reference": row["reference"], "processed": True}

        manager = LLMCheckpointManager(
            minio=minio_resource,
            asset_name="test",
            run_id="run-123",
            checkpoint_interval=100,
        )

        result = process_with_checkpoint(
            df=sample_speeches_df,
            id_column="reference",
            process_fn=process_fn,
            checkpoint_manager=manager,
            batch_size=10,
            logger=mock_logger,
        )

        # Logger should have been called
        mock_logger.info.assert_called()
        # Result should be valid
        assert len(result) == 3


class TestCheckpointTypeAnnotations:
    """Tests verifying type annotations follow standards."""

    def test_llm_checkpoint_manager_has_complete_types(self):
        """Test LLMCheckpointManager has all type annotations."""
        hints = get_type_hints(LLMCheckpointManager.load)
        assert "return" in hints

        hints = get_type_hints(LLMCheckpointManager.save_batch)
        assert "results" in hints
        assert "force" in hints

        hints = get_type_hints(LLMCheckpointManager.finalize)
        assert "return" in hints

    def test_process_with_checkpoint_has_proper_callable_type(self):
        """Test process_fn parameter has proper Callable type annotation."""
        from collections.abc import Callable

        from dagster import DagsterLogManager

        # Need all types used in the function signature
        hints = get_type_hints(
            process_with_checkpoint,
            globalns={
                "Callable": Callable,
                "Any": Any,
                "pl": pl,
                "DagsterLogManager": DagsterLogManager,
                "LLMCheckpointManager": LLMCheckpointManager,
            },
        )
        assert "process_fn" in hints
        assert "df" in hints
        assert "id_column" in hints
        assert "return" in hints

    def test_pydantic_v2_model_config(self):
        """Test LLMCheckpointManager uses Pydantic v2 model_config."""
        assert hasattr(LLMCheckpointManager, "model_config")
        config = LLMCheckpointManager.model_config
        assert config.get("arbitrary_types_allowed") is True
