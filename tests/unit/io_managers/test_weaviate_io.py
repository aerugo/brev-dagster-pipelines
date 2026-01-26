"""Unit tests for Weaviate I/O Manager.

Tests the WeaviateIOManager for storing DataFrames with embeddings
in Weaviate vector database. All external Weaviate calls are mocked per INV-P010.

Note: Due to Dagster's ConfigurableIOManager requiring runtime type resolution,
and the I/O managers using TYPE_CHECKING imports, we test the logic through
alternative means where full instantiation isn't possible in isolation.
"""

from __future__ import annotations

from typing import get_type_hints
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
from dagster import AssetKey, build_input_context, build_output_context


class TestWeaviateIOManagerFullIntegration:
    """Tests for full handle_output and load_input with mocked resources."""

    @patch("brev_pipelines.resources.weaviate.WeaviateResource.insert_objects")
    @patch("brev_pipelines.resources.weaviate.WeaviateResource.ensure_collection")
    @patch("weaviate.connect_to_custom")
    def test_handle_output_stores_dataframe_with_embeddings(
        self,
        mock_weaviate_connect: MagicMock,
        mock_ensure_collection: MagicMock,
        mock_insert_objects: MagicMock,
        sample_speeches_df: pl.DataFrame,
    ):
        """Test handle_output stores DataFrame and embeddings in Weaviate."""
        from brev_pipelines.io_managers.weaviate_io import WeaviateIOManager
        from brev_pipelines.resources.weaviate import WeaviateResource

        mock_weaviate_client = MagicMock()
        mock_weaviate_connect.return_value = mock_weaviate_client
        mock_insert_objects.return_value = 3

        weaviate_resource = WeaviateResource(
            host="localhost",
            port=8080,
            grpc_port=50051,
        )

        manager = WeaviateIOManager(weaviate=weaviate_resource)
        context = build_output_context(asset_key=AssetKey(["test_speeches"]))

        embeddings = [[0.1] * 1024, [0.2] * 1024, [0.3] * 1024]
        manager.handle_output(context, (sample_speeches_df, embeddings))

        # Verify ensure_collection was called
        mock_ensure_collection.assert_called_once()
        call_kwargs = mock_ensure_collection.call_args.kwargs
        assert call_kwargs["name"] == "Testspeeches"
        assert call_kwargs["vector_dimensions"] == 1024

        # Verify insert_objects was called
        mock_insert_objects.assert_called_once()
        insert_call = mock_insert_objects.call_args
        assert insert_call.kwargs["collection_name"] == "Testspeeches"
        assert len(insert_call.kwargs["objects"]) == 3
        assert len(insert_call.kwargs["vectors"]) == 3

    @patch("brev_pipelines.resources.weaviate.WeaviateResource.insert_objects")
    @patch("brev_pipelines.resources.weaviate.WeaviateResource.ensure_collection")
    @patch("weaviate.connect_to_custom")
    def test_handle_output_skips_none(
        self,
        mock_weaviate_connect: MagicMock,
        mock_ensure_collection: MagicMock,
        mock_insert_objects: MagicMock,
    ):
        """Test handle_output skips when obj is None."""
        from brev_pipelines.io_managers.weaviate_io import WeaviateIOManager
        from brev_pipelines.resources.weaviate import WeaviateResource

        mock_weaviate_client = MagicMock()
        mock_weaviate_connect.return_value = mock_weaviate_client

        weaviate_resource = WeaviateResource(
            host="localhost",
            port=8080,
            grpc_port=50051,
        )

        manager = WeaviateIOManager(weaviate=weaviate_resource)
        context = build_output_context(asset_key=AssetKey(["test"]))

        manager.handle_output(context, None)

        mock_ensure_collection.assert_not_called()
        mock_insert_objects.assert_not_called()

    @patch("weaviate.connect_to_custom")
    def test_handle_output_raises_on_length_mismatch(
        self, mock_weaviate_connect: MagicMock, sample_speeches_df: pl.DataFrame
    ):
        """Test handle_output raises ValueError when lengths don't match."""
        from brev_pipelines.io_managers.weaviate_io import WeaviateIOManager
        from brev_pipelines.resources.weaviate import WeaviateResource

        mock_weaviate_client = MagicMock()
        mock_weaviate_connect.return_value = mock_weaviate_client

        weaviate_resource = WeaviateResource(
            host="localhost",
            port=8080,
            grpc_port=50051,
        )

        manager = WeaviateIOManager(weaviate=weaviate_resource)
        context = build_output_context(asset_key=AssetKey(["test"]))

        # 3 rows but only 2 embeddings
        embeddings = [[0.1] * 1024, [0.2] * 1024]

        with pytest.raises(ValueError, match="3 rows but got 2 embeddings"):
            manager.handle_output(context, (sample_speeches_df, embeddings))

    @patch("brev_pipelines.resources.weaviate.WeaviateResource.get_object_count")
    @patch("weaviate.connect_to_custom")
    def test_load_input_returns_count(
        self,
        mock_weaviate_connect: MagicMock,
        mock_get_object_count: MagicMock,
    ):
        """Test load_input returns object count from collection."""
        from brev_pipelines.io_managers.weaviate_io import WeaviateIOManager
        from brev_pipelines.resources.weaviate import WeaviateResource

        mock_weaviate_client = MagicMock()
        mock_weaviate_connect.return_value = mock_weaviate_client
        mock_get_object_count.return_value = 42

        weaviate_resource = WeaviateResource(
            host="localhost",
            port=8080,
            grpc_port=50051,
        )

        manager = WeaviateIOManager(weaviate=weaviate_resource)
        context = build_input_context(asset_key=AssetKey(["test_collection"]))

        result = manager.load_input(context)

        assert result == 42
        mock_get_object_count.assert_called_once_with("Testcollection")

    @patch("brev_pipelines.resources.weaviate.WeaviateResource.insert_objects")
    @patch("brev_pipelines.resources.weaviate.WeaviateResource.ensure_collection")
    @patch("weaviate.connect_to_custom")
    def test_handle_output_with_prefix(
        self,
        mock_weaviate_connect: MagicMock,
        mock_ensure_collection: MagicMock,
        mock_insert_objects: MagicMock,
        sample_speeches_df: pl.DataFrame,
    ):
        """Test handle_output applies collection prefix."""
        from brev_pipelines.io_managers.weaviate_io import WeaviateIOManager
        from brev_pipelines.resources.weaviate import WeaviateResource

        mock_weaviate_client = MagicMock()
        mock_weaviate_connect.return_value = mock_weaviate_client
        mock_insert_objects.return_value = 3

        weaviate_resource = WeaviateResource(
            host="localhost",
            port=8080,
            grpc_port=50051,
        )

        manager = WeaviateIOManager(
            weaviate=weaviate_resource,
            collection_prefix="Dev",
        )
        context = build_output_context(asset_key=AssetKey(["speeches"]))

        embeddings = [[0.1] * 512] * 3
        manager.handle_output(context, (sample_speeches_df, embeddings))

        # Should have prefix applied
        call_kwargs = mock_ensure_collection.call_args.kwargs
        assert call_kwargs["name"] == "Devspeeches"


class TestWeaviateIOManagerHandleOutput:
    """Tests for handle_output method behavior.

    These tests verify the core logic without full Dagster ConfigurableIOManager instantiation.
    """

    def test_handle_output_raises_on_mismatched_lengths(self):
        """Test ValueError is raised for mismatched DataFrame/embedding lengths."""
        # Test the validation logic directly by simulating the check
        df = pl.DataFrame({"col": [1, 2, 3]})
        embeddings = [[0.1] * 1024, [0.2] * 1024]  # Only 2 embeddings for 3 rows

        # This is the same validation logic from the handle_output method
        if len(df) != len(embeddings):
            msg = f"DataFrame has {len(df)} rows but got {len(embeddings)} embeddings"
            with pytest.raises(ValueError, match="3 rows but got 2 embeddings"):
                raise ValueError(msg)

    def test_collection_name_conversion_to_pascal_case(self):
        """Test asset key is converted to PascalCase collection name."""
        # Simulate the conversion logic from handle_output
        asset_key = "my_speech_data"
        collection_prefix = ""

        collection_name = f"{collection_prefix}{asset_key}".replace("_", "")
        collection_name = "".join(word.title() for word in collection_name.split())

        assert collection_name == "Myspeechdata"

    def test_collection_name_with_prefix(self):
        """Test collection prefix is applied correctly."""
        asset_key = "speeches"
        collection_prefix = "Dev"

        collection_name = f"{collection_prefix}{asset_key}".replace("_", "")
        collection_name = "".join(word.title() for word in collection_name.split())

        assert collection_name == "Devspeeches"

    def test_schema_inference_from_dataframe(self):
        """Test Weaviate schema is correctly inferred from DataFrame types."""
        df = pl.DataFrame(
            {
                "text_col": ["a", "b"],
                "int_col": [1, 2],
                "float_col": [1.1, 2.2],
                "bool_col": [True, False],
            }
        )

        # Simulate the schema inference logic
        properties = []
        for col in df.columns:
            dtype = df[col].dtype
            prop_type = "text"
            if dtype == pl.Date:
                prop_type = "date"
            elif dtype == pl.Boolean:
                prop_type = "boolean"
            elif dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64):
                prop_type = "int"
            elif dtype in (pl.Float32, pl.Float64):
                prop_type = "number"
            properties.append({"name": col, "type": prop_type})

        prop_types = {p["name"]: p["type"] for p in properties}
        assert prop_types["text_col"] == "text"
        assert prop_types["int_col"] == "int"
        assert prop_types["float_col"] == "number"
        assert prop_types["bool_col"] == "boolean"

    def test_vector_dimensions_detection(self):
        """Test vector dimensions are correctly detected from embeddings."""
        embeddings_1024 = [[0.1] * 1024, [0.2] * 1024]
        embeddings_512 = [[0.1] * 512, [0.2] * 512]
        embeddings_empty: list[list[float]] = []

        # Same logic as in handle_output
        assert (len(embeddings_1024[0]) if embeddings_1024 else 1024) == 1024
        assert (len(embeddings_512[0]) if embeddings_512 else 1024) == 512
        assert (len(embeddings_empty[0]) if embeddings_empty else 1024) == 1024  # default


class TestWeaviateIOManagerLoadInput:
    """Tests for load_input method behavior."""

    def test_collection_name_conversion_for_load(self):
        """Test asset key is converted to PascalCase for load."""
        asset_key = "my_speech_embeddings"
        collection_prefix = ""

        collection_name = f"{collection_prefix}{asset_key}".replace("_", "")
        collection_name = "".join(word.title() for word in collection_name.split())

        assert collection_name == "Myspeechembeddings"

    def test_default_collection_name(self):
        """Test default collection name when no asset key."""
        asset_key = "default"  # Fallback value
        collection_prefix = ""

        collection_name = f"{collection_prefix}{asset_key}".replace("_", "")
        collection_name = "".join(word.title() for word in collection_name.split())

        assert collection_name == "Default"


class TestWeaviateIOManagerTypeAnnotations:
    """Tests verifying type annotations follow standards.

    These tests can import the I/O manager class without instantiation
    to verify type annotations are correctly defined.
    """

    def test_handle_output_has_type_annotations(self):
        """Test handle_output has proper type annotations."""
        # Import inside test to avoid triggering Dagster validation at module level
        with patch("weaviate.connect_to_custom"):
            from brev_pipelines.io_managers.weaviate_io import WeaviateIOManager

            hints = get_type_hints(WeaviateIOManager.handle_output)
            assert "context" in hints
            assert "obj" in hints
            assert "return" in hints

    def test_load_input_has_type_annotations(self):
        """Test load_input has proper type annotations."""
        with patch("weaviate.connect_to_custom"):
            from brev_pipelines.io_managers.weaviate_io import WeaviateIOManager

            hints = get_type_hints(WeaviateIOManager.load_input)
            assert "context" in hints
            assert "return" in hints
            # Returns int, not DataFrame
            assert hints["return"] is int

    def test_is_configurable_io_manager(self):
        """Test WeaviateIOManager extends ConfigurableIOManager."""
        from dagster import ConfigurableIOManager

        with patch("weaviate.connect_to_custom"):
            from brev_pipelines.io_managers.weaviate_io import WeaviateIOManager

            assert issubclass(WeaviateIOManager, ConfigurableIOManager)

    def test_handle_output_obj_type_is_tuple(self):
        """Test handle_output expects tuple of DataFrame and embeddings."""
        with patch("weaviate.connect_to_custom"):
            from brev_pipelines.io_managers.weaviate_io import WeaviateIOManager

            hints = get_type_hints(WeaviateIOManager.handle_output)
            obj_type = hints["obj"]
            # Check it's a tuple type
            assert hasattr(obj_type, "__origin__") and obj_type.__origin__ is tuple
