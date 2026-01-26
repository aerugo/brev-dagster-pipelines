"""Tests for Weaviate validation asset.

TDD tests for Weaviate validation functionality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, Mock

import pytest

from brev_pipelines.assets.validation import validate_weaviate

if TYPE_CHECKING:
    from dagster import AssetExecutionContext


class TestValidateWeaviate:
    """Test Weaviate validation asset."""

    @pytest.fixture
    def mock_weaviate_healthy(self) -> MagicMock:
        """Create mock Weaviate resource that is healthy."""
        weaviate = MagicMock()
        client = MagicMock()
        client.is_ready.return_value = True
        client.collections.list_all.return_value = {"TestCollection": Mock()}
        weaviate.get_client.return_value = client
        return weaviate

    @pytest.fixture
    def mock_weaviate_unhealthy(self) -> MagicMock:
        """Create mock Weaviate resource that is not ready."""
        weaviate = MagicMock()
        client = MagicMock()
        client.is_ready.return_value = False
        weaviate.get_client.return_value = client
        return weaviate

    @pytest.fixture
    def mock_weaviate_connection_error(self) -> MagicMock:
        """Create mock Weaviate resource that fails to connect."""
        weaviate = MagicMock()
        weaviate.get_client.side_effect = Exception("Connection refused")
        return weaviate

    def test_validate_weaviate_exists(self) -> None:
        """validate_weaviate asset should exist."""
        assert validate_weaviate is not None
        assert callable(validate_weaviate)

    def test_validate_weaviate_returns_dict(
        self, asset_context: AssetExecutionContext, mock_weaviate_healthy: MagicMock
    ) -> None:
        """validate_weaviate should return a dictionary."""
        result = validate_weaviate(context=asset_context, weaviate=mock_weaviate_healthy)
        assert isinstance(result, dict)

    def test_validate_weaviate_has_required_fields(
        self, asset_context: AssetExecutionContext, mock_weaviate_healthy: MagicMock
    ) -> None:
        """Result should have component, passed, tests fields."""
        result = validate_weaviate(context=asset_context, weaviate=mock_weaviate_healthy)

        assert "component" in result
        assert result["component"] == "weaviate"
        assert "passed" in result
        assert "tests" in result
        assert isinstance(result["tests"], list)

    def test_validate_weaviate_healthy_passes(
        self, asset_context: AssetExecutionContext, mock_weaviate_healthy: MagicMock
    ) -> None:
        """Should return passed=True when Weaviate is healthy."""
        result = validate_weaviate(context=asset_context, weaviate=mock_weaviate_healthy)

        assert result["passed"] is True

    def test_validate_weaviate_unhealthy_fails(
        self, asset_context: AssetExecutionContext, mock_weaviate_unhealthy: MagicMock
    ) -> None:
        """Should return passed=False when Weaviate is not ready."""
        result = validate_weaviate(context=asset_context, weaviate=mock_weaviate_unhealthy)

        assert result["passed"] is False

    def test_validate_weaviate_connection_error_fails(
        self, asset_context: AssetExecutionContext, mock_weaviate_connection_error: MagicMock
    ) -> None:
        """Should return passed=False with error on connection failure."""
        result = validate_weaviate(context=asset_context, weaviate=mock_weaviate_connection_error)

        assert result["passed"] is False
        assert result["error"] is not None
        assert "Connection refused" in result["error"]

    def test_validate_weaviate_tests_connection(
        self, asset_context: AssetExecutionContext, mock_weaviate_healthy: MagicMock
    ) -> None:
        """Should include connection test in results."""
        result = validate_weaviate(context=asset_context, weaviate=mock_weaviate_healthy)

        test_names = [t["name"] for t in result["tests"]]
        assert "connection" in test_names

    def test_validate_weaviate_tests_schema_access(
        self, asset_context: AssetExecutionContext, mock_weaviate_healthy: MagicMock
    ) -> None:
        """Should include schema access test in results."""
        result = validate_weaviate(context=asset_context, weaviate=mock_weaviate_healthy)

        test_names = [t["name"] for t in result["tests"]]
        assert "schema_access" in test_names

    def test_validate_weaviate_has_duration(
        self, asset_context: AssetExecutionContext, mock_weaviate_healthy: MagicMock
    ) -> None:
        """Result should include duration_ms."""
        result = validate_weaviate(context=asset_context, weaviate=mock_weaviate_healthy)

        assert "duration_ms" in result
        assert isinstance(result["duration_ms"], float)
        assert result["duration_ms"] >= 0


class TestValidatePlatformIncludesWeaviate:
    """Test that validate_platform includes Weaviate."""

    def test_validate_platform_deps_include_weaviate(self) -> None:
        """validate_platform should depend on validate_weaviate."""
        from brev_pipelines.assets.validation import validate_platform

        # Get asset dependencies from the underlying assets definition
        asset_deps = validate_platform.asset_deps
        # asset_deps is a dict of AssetKey -> set of AssetKeys
        all_deps = set()
        for deps_set in asset_deps.values():
            for dep in deps_set:
                all_deps.add(str(dep))

        assert any("weaviate" in name.lower() for name in all_deps), (
            f"validate_platform should depend on validate_weaviate. Current deps: {all_deps}"
        )
