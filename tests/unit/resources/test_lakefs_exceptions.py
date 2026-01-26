"""Tests for LakeFS resource exception handling.

TDD RED phase: These tests define the expected exception behavior
for LakeFSResource before implementation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from brev_pipelines.resources.lakefs import (
    LakeFSConnectionError,
    LakeFSError,
    LakeFSNotFoundError,
    LakeFSResource,
)


class TestLakeFSExceptionTypes:
    """Test LakeFS exception type hierarchy."""

    def test_lakefs_error_is_base_exception(self) -> None:
        """LakeFSError should be the base exception."""
        error = LakeFSError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"

    def test_lakefs_connection_error_inherits_from_base(self) -> None:
        """LakeFSConnectionError should inherit from LakeFSError."""
        error = LakeFSConnectionError("Connection failed")
        assert isinstance(error, LakeFSError)
        assert isinstance(error, Exception)
        assert "Connection failed" in str(error)

    def test_lakefs_not_found_error_inherits_from_base(self) -> None:
        """LakeFSNotFoundError should inherit from LakeFSError."""
        error = LakeFSNotFoundError("Object not found")
        assert isinstance(error, LakeFSError)
        assert isinstance(error, Exception)
        assert "Object not found" in str(error)


class TestLakeFSResourceExceptions:
    """Test that LakeFSResource raises proper exceptions."""

    @pytest.fixture
    def resource(self) -> LakeFSResource:
        """Create LakeFS resource for testing."""
        return LakeFSResource(
            endpoint="lakefs.example.com:8000",
            access_key="test_key",
            secret_key="test_secret",
        )

    def test_get_client_raises_connection_error_on_failure(
        self,
        resource: LakeFSResource,
    ) -> None:
        """get_client should raise LakeFSConnectionError when connection fails."""
        with (
            patch(
                "lakefs_sdk.Configuration",
                side_effect=Exception("Connection refused"),
            ),
            pytest.raises(LakeFSConnectionError) as exc_info,
        ):
            resource.get_client()

        assert "Connection refused" in str(exc_info.value)
        assert "lakefs.example.com" in str(exc_info.value)

    def test_list_repositories_raises_connection_error(
        self,
        resource: LakeFSResource,
    ) -> None:
        """list_repositories should raise LakeFSConnectionError on network failure."""
        mock_client = MagicMock()
        mock_client.repositories_api.list_repositories.side_effect = Exception("Network error")

        with (
            patch(
                "brev_pipelines.resources.lakefs.LakeFSResource.get_client",
                return_value=mock_client,
            ),
            pytest.raises(LakeFSConnectionError) as exc_info,
        ):
            resource.list_repositories()

        assert (
            "Network error" in str(exc_info.value)
            or "list repositories" in str(exc_info.value).lower()
        )

    def test_list_repositories_raises_lakefs_error_on_api_error(
        self,
        resource: LakeFSResource,
    ) -> None:
        """list_repositories should raise LakeFSError on API errors."""
        mock_client = MagicMock()
        mock_client.repositories_api.list_repositories.side_effect = Exception(
            "API rate limit exceeded"
        )

        with (
            patch(
                "brev_pipelines.resources.lakefs.LakeFSResource.get_client",
                return_value=mock_client,
            ),
            pytest.raises(LakeFSError),
        ):
            resource.list_repositories()

    def test_health_check_returns_false_on_error(
        self,
        resource: LakeFSResource,
    ) -> None:
        """health_check should return False on connection errors (not raise)."""
        with patch(
            "brev_pipelines.resources.lakefs.LakeFSResource.get_client",
            side_effect=LakeFSConnectionError("Failed"),
        ):
            result = resource.health_check()
            assert result is False

    def test_health_check_returns_true_on_success(
        self,
        resource: LakeFSResource,
    ) -> None:
        """health_check should return True when connection succeeds."""
        mock_client = MagicMock()
        mock_client.repositories_api.list_repositories.return_value = MagicMock(results=[])

        with patch(
            "brev_pipelines.resources.lakefs.LakeFSResource.get_client",
            return_value=mock_client,
        ):
            result = resource.health_check()
            assert result is True


class TestLakeFSResourceNotFoundError:
    """Test LakeFSNotFoundError scenarios."""

    @pytest.fixture
    def resource(self) -> LakeFSResource:
        """Create LakeFS resource for testing."""
        return LakeFSResource(
            endpoint="lakefs.example.com:8000",
            access_key="test_key",
            secret_key="test_secret",
        )

    def test_get_object_raises_not_found_error(
        self,
        resource: LakeFSResource,
    ) -> None:
        """get_object should raise LakeFSNotFoundError when object doesn't exist."""
        mock_client = MagicMock()
        mock_client.objects_api.get_object.side_effect = Exception("404: Not Found")

        with (
            patch(
                "brev_pipelines.resources.lakefs.LakeFSResource.get_client",
                return_value=mock_client,
            ),
            pytest.raises(LakeFSNotFoundError) as exc_info,
        ):
            resource.get_object(
                repository="test-repo",
                ref="main",
                path="missing/file.parquet",
            )

        assert "not found" in str(exc_info.value).lower()

    def test_get_object_returns_bytes_on_success(
        self,
        resource: LakeFSResource,
    ) -> None:
        """get_object should return bytes when object exists."""
        mock_client = MagicMock()
        mock_client.objects_api.get_object.return_value = b"test content"

        with patch(
            "brev_pipelines.resources.lakefs.LakeFSResource.get_client",
            return_value=mock_client,
        ):
            result = resource.get_object(
                repository="test-repo",
                ref="main",
                path="existing/file.txt",
            )

            assert result == b"test content"

    def test_get_object_raises_lakefs_error_on_other_errors(
        self,
        resource: LakeFSResource,
    ) -> None:
        """get_object should raise LakeFSError on non-404 errors."""
        mock_client = MagicMock()
        mock_client.objects_api.get_object.side_effect = Exception("500: Internal Server Error")

        with (
            patch(
                "brev_pipelines.resources.lakefs.LakeFSResource.get_client",
                return_value=mock_client,
            ),
            pytest.raises(LakeFSError) as exc_info,
        ):
            resource.get_object(
                repository="test-repo",
                ref="main",
                path="path/to/file.txt",
            )

        # Should be LakeFSError but NOT LakeFSNotFoundError
        assert not isinstance(exc_info.value, LakeFSNotFoundError)


class TestLakeFSResourcePutObject:
    """Test put_object exception handling."""

    @pytest.fixture
    def resource(self) -> LakeFSResource:
        """Create LakeFS resource for testing."""
        return LakeFSResource(
            endpoint="lakefs.example.com:8000",
            access_key="test_key",
            secret_key="test_secret",
        )

    def test_put_object_raises_connection_error_on_network_failure(
        self,
        resource: LakeFSResource,
    ) -> None:
        """put_object should raise LakeFSConnectionError on network failure."""
        mock_client = MagicMock()
        mock_client.objects_api.upload_object.side_effect = Exception("Connection timed out")

        with (
            patch(
                "brev_pipelines.resources.lakefs.LakeFSResource.get_client",
                return_value=mock_client,
            ),
            pytest.raises(LakeFSConnectionError),
        ):
            resource.put_object(
                repository="test-repo",
                branch="main",
                path="path/to/file.txt",
                content=b"test content",
            )

    def test_put_object_raises_lakefs_error_on_api_error(
        self,
        resource: LakeFSResource,
    ) -> None:
        """put_object should raise LakeFSError on API errors."""
        mock_client = MagicMock()
        mock_client.objects_api.upload_object.side_effect = Exception("Permission denied")

        with (
            patch(
                "brev_pipelines.resources.lakefs.LakeFSResource.get_client",
                return_value=mock_client,
            ),
            pytest.raises(LakeFSError),
        ):
            resource.put_object(
                repository="test-repo",
                branch="main",
                path="path/to/file.txt",
                content=b"test content",
            )

    def test_put_object_succeeds_on_success(
        self,
        resource: LakeFSResource,
    ) -> None:
        """put_object should succeed when upload works."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.physical_address = "s3://bucket/path"
        mock_client.objects_api.upload_object.return_value = mock_response

        with patch(
            "brev_pipelines.resources.lakefs.LakeFSResource.get_client",
            return_value=mock_client,
        ):
            # Should not raise
            resource.put_object(
                repository="test-repo",
                branch="main",
                path="path/to/file.txt",
                content=b"test content",
            )

            mock_client.objects_api.upload_object.assert_called_once()
