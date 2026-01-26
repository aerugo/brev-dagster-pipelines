"""Tests for Weaviate resource exception handling.

TDD RED phase: These tests define the expected exception behavior
for WeaviateResource before implementation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from brev_pipelines.resources.weaviate import (
    WeaviateCollectionError,
    WeaviateConnectionError,
    WeaviateError,
    WeaviateResource,
)


class TestWeaviateExceptionTypes:
    """Test Weaviate exception type hierarchy."""

    def test_weaviate_error_is_base_exception(self) -> None:
        """WeaviateError should be the base exception."""
        error = WeaviateError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"

    def test_weaviate_connection_error_inherits_from_base(self) -> None:
        """WeaviateConnectionError should inherit from WeaviateError."""
        error = WeaviateConnectionError("Connection failed")
        assert isinstance(error, WeaviateError)
        assert isinstance(error, Exception)
        assert "Connection failed" in str(error)

    def test_weaviate_collection_error_inherits_from_base(self) -> None:
        """WeaviateCollectionError should inherit from WeaviateError."""
        error = WeaviateCollectionError("Collection operation failed")
        assert isinstance(error, WeaviateError)
        assert isinstance(error, Exception)
        assert "Collection operation failed" in str(error)


class TestWeaviateResourceConnectionExceptions:
    """Test that WeaviateResource raises proper connection exceptions."""

    @pytest.fixture
    def resource(self) -> WeaviateResource:
        """Create Weaviate resource for testing."""
        return WeaviateResource(
            host="weaviate.example.com",
            port=8080,
            grpc_host="weaviate-grpc.example.com",
            grpc_port=50051,
        )

    def test_get_client_raises_connection_error_on_failure(
        self,
        resource: WeaviateResource,
    ) -> None:
        """get_client should raise WeaviateConnectionError when connection fails."""
        with (
            patch(
                "weaviate.connect_to_custom",
                side_effect=Exception("Connection refused"),
            ),
            pytest.raises(WeaviateConnectionError) as exc_info,
        ):
            resource.get_client()

        assert "Connection refused" in str(exc_info.value)
        assert "weaviate.example.com" in str(exc_info.value)

    def test_health_check_returns_false_on_connection_error(
        self,
        resource: WeaviateResource,
    ) -> None:
        """health_check should return False on connection errors (not raise)."""
        with patch(
            "weaviate.connect_to_custom",
            side_effect=Exception("Connection refused"),
        ):
            result = resource.health_check()
            assert result is False

    def test_health_check_returns_true_on_success(
        self,
        resource: WeaviateResource,
    ) -> None:
        """health_check should return True when Weaviate is ready."""
        mock_client = MagicMock()
        mock_client.is_ready.return_value = True

        with patch("weaviate.connect_to_custom", return_value=mock_client):
            result = resource.health_check()
            assert result is True


class TestWeaviateResourceCollectionExceptions:
    """Test collection operation exception handling."""

    @pytest.fixture
    def resource(self) -> WeaviateResource:
        """Create Weaviate resource for testing."""
        return WeaviateResource(
            host="weaviate.example.com",
            port=8080,
            grpc_host="weaviate-grpc.example.com",
            grpc_port=50051,
        )

    def test_ensure_collection_raises_connection_error_on_network_failure(
        self,
        resource: WeaviateResource,
    ) -> None:
        """ensure_collection should raise WeaviateConnectionError on network failure."""
        with (
            patch(
                "weaviate.connect_to_custom",
                side_effect=Exception("Network unreachable"),
            ),
            pytest.raises(WeaviateConnectionError),
        ):
            resource.ensure_collection(
                name="TestCollection",
                properties=[{"name": "title", "type": "text"}],
            )

    def test_ensure_collection_raises_collection_error_on_schema_error(
        self,
        resource: WeaviateResource,
    ) -> None:
        """ensure_collection should raise WeaviateCollectionError on schema errors."""
        mock_client = MagicMock()
        mock_client.collections.exists.return_value = False
        mock_client.collections.create.side_effect = Exception("Invalid schema definition")

        with (
            patch("weaviate.connect_to_custom", return_value=mock_client),
            pytest.raises(WeaviateCollectionError) as exc_info,
        ):
            resource.ensure_collection(
                name="TestCollection",
                properties=[{"name": "title", "type": "text"}],
            )

        assert "Invalid schema" in str(exc_info.value) or "TestCollection" in str(exc_info.value)

    def test_insert_objects_raises_connection_error_on_network_failure(
        self,
        resource: WeaviateResource,
    ) -> None:
        """insert_objects should raise WeaviateConnectionError on network failure."""
        with (
            patch(
                "weaviate.connect_to_custom",
                side_effect=Exception("Connection timed out"),
            ),
            pytest.raises(WeaviateConnectionError),
        ):
            resource.insert_objects(
                collection_name="TestCollection",
                objects=[{"title": "test"}],
                vectors=[[0.1, 0.2, 0.3]],
            )

    def test_insert_objects_raises_collection_error_on_batch_failure(
        self,
        resource: WeaviateResource,
    ) -> None:
        """insert_objects should raise WeaviateCollectionError on batch insert failure."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.collections.get.return_value = mock_collection
        mock_collection.batch.dynamic.side_effect = Exception("Batch insert failed")

        with (
            patch("weaviate.connect_to_custom", return_value=mock_client),
            pytest.raises(WeaviateCollectionError),
        ):
            resource.insert_objects(
                collection_name="TestCollection",
                objects=[{"title": "test"}],
                vectors=[[0.1, 0.2, 0.3]],
            )

    def test_vector_search_raises_connection_error_on_network_failure(
        self,
        resource: WeaviateResource,
    ) -> None:
        """vector_search should raise WeaviateConnectionError on network failure."""
        with (
            patch(
                "weaviate.connect_to_custom",
                side_effect=Exception("Connection refused"),
            ),
            pytest.raises(WeaviateConnectionError),
        ):
            resource.vector_search(
                collection_name="TestCollection",
                query_vector=[0.1, 0.2, 0.3],
            )

    def test_vector_search_raises_collection_error_on_query_failure(
        self,
        resource: WeaviateResource,
    ) -> None:
        """vector_search should raise WeaviateCollectionError on query failure."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.collections.get.return_value = mock_collection
        mock_collection.query.near_vector.side_effect = Exception("Query execution failed")

        with (
            patch("weaviate.connect_to_custom", return_value=mock_client),
            pytest.raises(WeaviateCollectionError),
        ):
            resource.vector_search(
                collection_name="TestCollection",
                query_vector=[0.1, 0.2, 0.3],
            )

    def test_get_object_count_raises_connection_error_on_network_failure(
        self,
        resource: WeaviateResource,
    ) -> None:
        """get_object_count should raise WeaviateConnectionError on network failure."""
        with (
            patch(
                "weaviate.connect_to_custom",
                side_effect=Exception("Network error"),
            ),
            pytest.raises(WeaviateConnectionError),
        ):
            resource.get_object_count("TestCollection")

    def test_delete_collection_raises_connection_error_on_network_failure(
        self,
        resource: WeaviateResource,
    ) -> None:
        """delete_collection should raise WeaviateConnectionError on network failure."""
        with (
            patch(
                "weaviate.connect_to_custom",
                side_effect=Exception("Connection timed out"),
            ),
            pytest.raises(WeaviateConnectionError),
        ):
            resource.delete_collection("TestCollection")

    def test_delete_collection_raises_collection_error_on_delete_failure(
        self,
        resource: WeaviateResource,
    ) -> None:
        """delete_collection should raise WeaviateCollectionError when delete fails."""
        mock_client = MagicMock()
        mock_client.collections.exists.return_value = True
        mock_client.collections.delete.side_effect = Exception("Permission denied")

        with (
            patch("weaviate.connect_to_custom", return_value=mock_client),
            pytest.raises(WeaviateCollectionError),
        ):
            resource.delete_collection("TestCollection")


class TestWeaviateResourceSuccessScenarios:
    """Test successful operation scenarios to ensure exceptions aren't raised inappropriately."""

    @pytest.fixture
    def resource(self) -> WeaviateResource:
        """Create Weaviate resource for testing."""
        return WeaviateResource(
            host="weaviate.example.com",
            port=8080,
            grpc_host="weaviate-grpc.example.com",
            grpc_port=50051,
        )

    def test_ensure_collection_succeeds_when_collection_exists(
        self,
        resource: WeaviateResource,
    ) -> None:
        """ensure_collection should not raise when collection already exists."""
        mock_client = MagicMock()
        mock_client.collections.exists.return_value = True

        with patch("weaviate.connect_to_custom", return_value=mock_client):
            # Should not raise
            resource.ensure_collection(
                name="ExistingCollection",
                properties=[{"name": "title", "type": "text"}],
            )

        mock_client.collections.create.assert_not_called()

    def test_get_object_count_returns_count_on_success(
        self,
        resource: WeaviateResource,
    ) -> None:
        """get_object_count should return count when successful."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_response = MagicMock()
        mock_response.total_count = 42
        mock_client.collections.get.return_value = mock_collection
        mock_collection.aggregate.over_all.return_value = mock_response

        with patch("weaviate.connect_to_custom", return_value=mock_client):
            result = resource.get_object_count("TestCollection")
            assert result == 42

    def test_delete_collection_returns_false_when_not_exists(
        self,
        resource: WeaviateResource,
    ) -> None:
        """delete_collection should return False when collection doesn't exist."""
        mock_client = MagicMock()
        mock_client.collections.exists.return_value = False

        with patch("weaviate.connect_to_custom", return_value=mock_client):
            result = resource.delete_collection("NonExistentCollection")
            assert result is False

        mock_client.collections.delete.assert_not_called()
