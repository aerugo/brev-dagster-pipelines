"""Unit tests for LakeFS resource.

Tests the LakeFSResource configuration and client creation.
All external LakeFS calls are mocked per INV-P010.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestLakeFSResourceInit:
    """Tests for LakeFSResource initialization."""

    def test_resource_initialization(self):
        """Test LakeFSResource can be initialized with required fields."""
        from brev_pipelines.resources.lakefs import LakeFSResource

        resource = LakeFSResource(
            endpoint="localhost:8000",
            access_key="test-access-key",
            secret_key="test-secret-key",
        )

        assert resource.endpoint == "localhost:8000"
        assert resource.access_key == "test-access-key"
        assert resource.secret_key == "test-secret-key"

    def test_resource_with_http_endpoint(self):
        """Test resource handles http:// prefixed endpoint."""
        from brev_pipelines.resources.lakefs import LakeFSResource

        resource = LakeFSResource(
            endpoint="http://localhost:8000",
            access_key="key",
            secret_key="secret",
        )

        assert resource.endpoint == "http://localhost:8000"

    def test_resource_with_https_endpoint(self):
        """Test resource handles https:// prefixed endpoint."""
        from brev_pipelines.resources.lakefs import LakeFSResource

        resource = LakeFSResource(
            endpoint="https://lakefs.example.com",
            access_key="key",
            secret_key="secret",
        )

        assert resource.endpoint == "https://lakefs.example.com"


class TestLakeFSClientCreation:
    """Tests for LakeFS client creation."""

    @patch("lakefs_sdk.Configuration")
    @patch("lakefs_sdk.client.LakeFSClient")
    def test_get_client_returns_lakefs_client(
        self, mock_client_class: MagicMock, mock_config_class: MagicMock
    ):
        """Test get_client() returns a LakeFSClient instance."""
        from brev_pipelines.resources.lakefs import LakeFSResource

        mock_config = MagicMock()
        mock_config_class.return_value = mock_config
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        resource = LakeFSResource(
            endpoint="localhost:8000",
            access_key="test-key",
            secret_key="test-secret",
        )

        client = resource.get_client()

        assert client is mock_client
        mock_config_class.assert_called_once_with(
            host="http://localhost:8000",
            username="test-key",
            password="test-secret",
        )
        mock_client_class.assert_called_once_with(mock_config)

    @patch("lakefs_sdk.Configuration")
    @patch("lakefs_sdk.client.LakeFSClient")
    def test_get_client_preserves_http_prefix(
        self, mock_client_class: MagicMock, mock_config_class: MagicMock
    ):
        """Test get_client() preserves http:// prefix in endpoint."""
        from brev_pipelines.resources.lakefs import LakeFSResource

        mock_config = MagicMock()
        mock_config_class.return_value = mock_config

        resource = LakeFSResource(
            endpoint="http://custom-host:8000",
            access_key="key",
            secret_key="secret",
        )

        resource.get_client()

        mock_config_class.assert_called_once()
        call_kwargs = mock_config_class.call_args
        assert call_kwargs[1]["host"] == "http://custom-host:8000"

    @patch("lakefs_sdk.Configuration")
    @patch("lakefs_sdk.client.LakeFSClient")
    def test_get_client_preserves_https_prefix(
        self, mock_client_class: MagicMock, mock_config_class: MagicMock
    ):
        """Test get_client() preserves https:// prefix in endpoint."""
        from brev_pipelines.resources.lakefs import LakeFSResource

        mock_config = MagicMock()
        mock_config_class.return_value = mock_config

        resource = LakeFSResource(
            endpoint="https://lakefs.prod.example.com",
            access_key="key",
            secret_key="secret",
        )

        resource.get_client()

        mock_config_class.assert_called_once()
        call_kwargs = mock_config_class.call_args
        assert call_kwargs[1]["host"] == "https://lakefs.prod.example.com"


class TestLakeFSOperations:
    """Tests for LakeFS operations."""

    def test_list_repositories(self, mock_lakefs_resource: MagicMock):
        """Test list_repositories returns repository IDs."""
        repos = mock_lakefs_resource.list_repositories()

        assert repos == ["central-bank-speeches"]

    def test_health_check_success(self, mock_lakefs_resource: MagicMock):
        """Test health_check returns True when LakeFS is accessible."""
        result = mock_lakefs_resource.health_check()

        assert result is True

    @patch("lakefs_sdk.Configuration")
    @patch("lakefs_sdk.client.LakeFSClient")
    def test_health_check_failure(self, mock_client_class: MagicMock, mock_config_class: MagicMock):
        """Test health_check returns False when LakeFS is not accessible."""
        from brev_pipelines.resources.lakefs import LakeFSResource

        mock_client = MagicMock()
        mock_client.repositories_api.list_repositories.side_effect = Exception("Connection refused")
        mock_client_class.return_value = mock_client

        resource = LakeFSResource(
            endpoint="localhost:8000",
            access_key="key",
            secret_key="secret",
        )

        result = resource.health_check()

        assert result is False


class TestLakeFSTypeAnnotations:
    """Tests verifying type annotations are present and correct.

    Uses typing.get_type_hints() to resolve stringified annotations
    from PEP 563 (from __future__ import annotations).
    """

    def test_get_client_has_return_type(self):
        """Test get_client() has proper return type annotation."""
        from typing import get_type_hints

        from lakefs_sdk.client import LakeFSClient

        from brev_pipelines.resources.lakefs import LakeFSResource

        # Get resolved type hints with namespace that includes LakeFSClient
        # This is needed because the import is under TYPE_CHECKING
        hints = get_type_hints(
            LakeFSResource.get_client,
            globalns={"LakeFSClient": LakeFSClient},
        )
        assert "return" in hints
        assert hints["return"] is LakeFSClient

    def test_list_repositories_has_return_type(self):
        """Test list_repositories() has proper return type annotation."""
        from typing import get_type_hints

        from brev_pipelines.resources.lakefs import LakeFSResource

        hints = get_type_hints(LakeFSResource.list_repositories)
        assert "return" in hints
        assert hints["return"] == list[str]

    def test_health_check_has_return_type(self):
        """Test health_check() has proper return type annotation."""
        from typing import get_type_hints

        from brev_pipelines.resources.lakefs import LakeFSResource

        hints = get_type_hints(LakeFSResource.health_check)
        assert "return" in hints
        assert hints["return"] is bool
