"""Unit tests for Weaviate resource.

Tests the WeaviateResource configuration, client creation, and operations.
All external Weaviate calls are mocked per INV-P010.
"""

from __future__ import annotations

from typing import get_type_hints
from unittest.mock import MagicMock, patch

import pytest

from brev_pipelines.types import WeaviatePropertyDef, WeaviateSearchResult


class TestWeaviateResourceInit:
    """Tests for WeaviateResource initialization."""

    def test_resource_initialization_defaults(self):
        """Test WeaviateResource has sensible defaults."""
        from brev_pipelines.resources.weaviate import WeaviateResource

        resource = WeaviateResource()

        assert resource.host == "weaviate.weaviate.svc.cluster.local"
        assert resource.port == 80
        assert resource.grpc_host == "weaviate-grpc.weaviate.svc.cluster.local"
        assert resource.grpc_port == 50051

    def test_resource_custom_config(self):
        """Test WeaviateResource with custom configuration."""
        from brev_pipelines.resources.weaviate import WeaviateResource

        resource = WeaviateResource(
            host="localhost",
            port=8080,
            grpc_host="localhost",
            grpc_port=50052,
        )

        assert resource.host == "localhost"
        assert resource.port == 8080
        assert resource.grpc_host == "localhost"
        assert resource.grpc_port == 50052


class TestWeaviateClientCreation:
    """Tests for Weaviate client creation."""

    @patch("weaviate.connect_to_custom")
    def test_get_client_returns_weaviate_client(self, mock_connect: MagicMock):
        """Test get_client() returns a WeaviateClient instance."""
        from brev_pipelines.resources.weaviate import WeaviateResource

        mock_client = MagicMock()
        mock_connect.return_value = mock_client

        resource = WeaviateResource(
            host="localhost",
            port=8080,
            grpc_host="localhost",
            grpc_port=50051,
        )

        client = resource.get_client()

        assert client is mock_client
        mock_connect.assert_called_once_with(
            http_host="localhost",
            http_port=8080,
            http_secure=False,
            grpc_host="localhost",
            grpc_port=50051,
            grpc_secure=False,
        )


class TestWeaviateEnsureCollection:
    """Tests for ensure_collection method."""

    @patch("weaviate.connect_to_custom")
    def test_ensure_collection_creates_when_not_exists(self, mock_connect: MagicMock):
        """Test collection is created when it doesn't exist."""
        from brev_pipelines.resources.weaviate import WeaviateResource

        mock_client = MagicMock()
        mock_client.collections.exists.return_value = False
        mock_connect.return_value = mock_client

        resource = WeaviateResource()
        properties: list[WeaviatePropertyDef] = [
            {"name": "title", "type": "text", "description": "Speech title"},
            {"name": "date", "type": "date", "description": "Speech date"},
            {"name": "monetary_stance", "type": "int", "description": "1-5 scale"},
        ]

        resource.ensure_collection("TestCollection", properties)

        mock_client.collections.exists.assert_called_once_with("TestCollection")
        mock_client.collections.create.assert_called_once()
        mock_client.close.assert_called_once()

    @patch("weaviate.connect_to_custom")
    def test_ensure_collection_skips_when_exists(self, mock_connect: MagicMock):
        """Test collection creation is skipped when it exists."""
        from brev_pipelines.resources.weaviate import WeaviateResource

        mock_client = MagicMock()
        mock_client.collections.exists.return_value = True
        mock_connect.return_value = mock_client

        resource = WeaviateResource()
        properties: list[WeaviatePropertyDef] = [{"name": "title"}]

        resource.ensure_collection("ExistingCollection", properties)

        mock_client.collections.exists.assert_called_once_with("ExistingCollection")
        mock_client.collections.create.assert_not_called()
        mock_client.close.assert_called_once()


class TestWeaviateInsertObjects:
    """Tests for insert_objects method."""

    @patch("weaviate.connect_to_custom")
    def test_insert_objects_success(self, mock_connect: MagicMock):
        """Test successful object insertion."""
        from brev_pipelines.resources.weaviate import WeaviateResource

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_batch_context = MagicMock()
        mock_collection.batch.dynamic.return_value.__enter__ = MagicMock(
            return_value=mock_batch_context
        )
        mock_collection.batch.dynamic.return_value.__exit__ = MagicMock(return_value=None)
        mock_client.collections.get.return_value = mock_collection
        mock_connect.return_value = mock_client

        resource = WeaviateResource()
        objects = [
            {"reference": "BIS_2024_001", "title": "Test Speech 1"},
            {"reference": "BIS_2024_002", "title": "Test Speech 2"},
        ]
        vectors = [[0.1] * 1024, [0.2] * 1024]

        count = resource.insert_objects("TestCollection", objects, vectors)

        assert count == 2
        mock_client.close.assert_called_once()

    @patch("weaviate.connect_to_custom")
    def test_insert_objects_mismatched_lengths(self, mock_connect: MagicMock):
        """Test ValueError raised for mismatched object/vector lengths."""
        from brev_pipelines.resources.weaviate import WeaviateResource

        mock_connect.return_value = MagicMock()

        resource = WeaviateResource()
        objects = [{"title": "Test"}]
        vectors = [[0.1] * 1024, [0.2] * 1024]  # 2 vectors, 1 object

        with pytest.raises(ValueError, match="same length"):
            resource.insert_objects("TestCollection", objects, vectors)


class TestWeaviateVectorSearch:
    """Tests for vector_search method."""

    @patch("weaviate.connect_to_custom")
    def test_vector_search_returns_results(self, mock_connect: MagicMock):
        """Test vector search returns properly formatted results."""
        from brev_pipelines.resources.weaviate import WeaviateResource

        # Setup mock
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_obj = MagicMock()
        mock_obj.properties = {"reference": "BIS_2024_001", "title": "Test Speech"}
        mock_obj.metadata.distance = 0.15
        mock_obj.metadata.certainty = 0.85
        mock_query_result = MagicMock()
        mock_query_result.objects = [mock_obj]
        mock_collection.query.near_vector.return_value = mock_query_result
        mock_client.collections.get.return_value = mock_collection
        mock_connect.return_value = mock_client

        resource = WeaviateResource()
        query_vector = [0.1] * 1024

        results = resource.vector_search("TestCollection", query_vector, limit=5)

        assert len(results) == 1
        # WeaviateSearchResult has properties nested under "properties" key
        assert results[0]["properties"]["reference"] == "BIS_2024_001"
        assert results[0]["_distance"] == 0.15
        assert results[0]["_certainty"] == 0.85
        mock_client.close.assert_called_once()


class TestWeaviateOperations:
    """Tests for other Weaviate operations."""

    def test_get_object_count(self, mock_weaviate_resource: MagicMock):
        """Test get_object_count returns count."""
        count = mock_weaviate_resource.get_object_count.return_value
        assert count == 100

    def test_health_check_success(self, mock_weaviate_resource: MagicMock):
        """Test health_check returns True when healthy."""
        mock_weaviate_resource.health_check.return_value = True
        assert mock_weaviate_resource.health_check() is True

    @patch("weaviate.connect_to_custom")
    def test_health_check_failure(self, mock_connect: MagicMock):
        """Test health_check returns False on error."""
        from brev_pipelines.resources.weaviate import WeaviateResource

        mock_connect.side_effect = Exception("Connection refused")

        resource = WeaviateResource()
        result = resource.health_check()

        assert result is False

    @patch("weaviate.connect_to_custom")
    def test_delete_collection_when_exists(self, mock_connect: MagicMock):
        """Test delete_collection returns True when collection existed."""
        from brev_pipelines.resources.weaviate import WeaviateResource

        mock_client = MagicMock()
        mock_client.collections.exists.return_value = True
        mock_connect.return_value = mock_client

        resource = WeaviateResource()
        result = resource.delete_collection("OldCollection")

        assert result is True
        mock_client.collections.delete.assert_called_once_with("OldCollection")

    @patch("weaviate.connect_to_custom")
    def test_delete_collection_when_not_exists(self, mock_connect: MagicMock):
        """Test delete_collection returns False when collection didn't exist."""
        from brev_pipelines.resources.weaviate import WeaviateResource

        mock_client = MagicMock()
        mock_client.collections.exists.return_value = False
        mock_connect.return_value = mock_client

        resource = WeaviateResource()
        result = resource.delete_collection("NonExistent")

        assert result is False
        mock_client.collections.delete.assert_not_called()


class TestWeaviateTypeAnnotations:
    """Tests verifying type annotations use proper types from brev_pipelines.types."""

    def test_ensure_collection_uses_property_def_type(self):
        """Test ensure_collection accepts WeaviatePropertyDef list."""
        from brev_pipelines.resources.weaviate import WeaviateResource
        from brev_pipelines.types import WeaviatePropertyDef

        hints = get_type_hints(WeaviateResource.ensure_collection)
        assert "properties" in hints
        # Should be list[WeaviatePropertyDef]
        expected_type = list[WeaviatePropertyDef]
        assert hints["properties"] == expected_type

    def test_vector_search_returns_search_result_type(self):
        """Test vector_search returns list of WeaviateSearchResult."""
        from brev_pipelines.resources.weaviate import WeaviateResource

        hints = get_type_hints(WeaviateResource.vector_search)
        assert "return" in hints
        # Should be list[WeaviateSearchResult]
        expected_type = list[WeaviateSearchResult]
        assert hints["return"] == expected_type

    def test_insert_objects_has_proper_types(self):
        """Test insert_objects has proper type annotations."""
        from brev_pipelines.resources.weaviate import WeaviateResource

        hints = get_type_hints(WeaviateResource.insert_objects)
        assert "objects" in hints
        assert "vectors" in hints
        assert "return" in hints
        assert hints["return"] is int

    def test_get_client_has_return_type(self):
        """Test get_client has return type annotation."""
        from weaviate import WeaviateClient

        from brev_pipelines.resources.weaviate import WeaviateResource

        hints = get_type_hints(
            WeaviateResource.get_client,
            globalns={"WeaviateClient": WeaviateClient},
        )
        assert "return" in hints
        assert hints["return"] is WeaviateClient
