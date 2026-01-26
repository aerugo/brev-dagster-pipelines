"""Tests for NIM exception handling.

TDD RED phase: These tests define the expected exception behavior
for NIMResource before implementation.
"""

from unittest.mock import Mock, patch

import pytest
import requests

from brev_pipelines.resources.nim import (
    NIMError,
    NIMRateLimitError,
    NIMResource,
    NIMServerError,
    NIMTimeoutError,
)


class TestNIMExceptionTypes:
    """Test NIM exception type hierarchy."""

    def test_nim_error_is_base_exception(self) -> None:
        """NIMError should be the base exception for all NIM errors."""
        error = NIMError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"

    def test_nim_timeout_error_inherits_from_nim_error(self) -> None:
        """NIMTimeoutError should inherit from NIMError."""
        error = NIMTimeoutError("Request timed out")
        assert isinstance(error, NIMError)
        assert isinstance(error, Exception)

    def test_nim_server_error_includes_status_code(self) -> None:
        """NIMServerError should include status code."""
        error = NIMServerError(503, "Service unavailable")
        assert isinstance(error, NIMError)
        assert error.status_code == 503
        assert "503" in str(error)
        assert "Service unavailable" in str(error)

    def test_nim_rate_limit_error_inherits_from_nim_error(self) -> None:
        """NIMRateLimitError should inherit from NIMError."""
        error = NIMRateLimitError("Rate limited")
        assert isinstance(error, NIMError)


class TestNIMResourceExceptionBehavior:
    """Test NIMResource raises exceptions instead of returning error strings."""

    @pytest.fixture
    def nim_resource(self) -> NIMResource:
        """Create NIM resource for testing."""
        return NIMResource(endpoint="http://test-nim:8000", model="test-model")

    def test_generate_raises_timeout_error(self, nim_resource: NIMResource) -> None:
        """generate() should raise NIMTimeoutError on timeout."""
        with patch("requests.post") as mock_post:
            mock_post.side_effect = requests.exceptions.Timeout("Connection timed out")

            with pytest.raises(NIMTimeoutError) as exc_info:
                nim_resource.generate("test prompt")

            assert "timed out" in str(exc_info.value).lower()

    def test_generate_raises_server_error_on_5xx(self, nim_resource: NIMResource) -> None:
        """generate() should raise NIMServerError on 5xx responses."""
        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 503
            mock_response.text = "Service Unavailable"
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
                response=mock_response
            )
            mock_post.return_value = mock_response

            with pytest.raises(NIMServerError) as exc_info:
                nim_resource.generate("test prompt")

            assert exc_info.value.status_code == 503

    def test_generate_raises_rate_limit_error_on_429(self, nim_resource: NIMResource) -> None:
        """generate() should raise NIMRateLimitError on 429."""
        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 429
            mock_response.text = "Rate limited"
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
                response=mock_response
            )
            mock_post.return_value = mock_response

            with pytest.raises(NIMRateLimitError):
                nim_resource.generate("test prompt")

    def test_generate_raises_nim_error_on_connection_failure(
        self, nim_resource: NIMResource
    ) -> None:
        """generate() should raise NIMError on connection failure."""
        with patch("requests.post") as mock_post:
            mock_post.side_effect = requests.exceptions.ConnectionError("Connection refused")

            with pytest.raises(NIMError) as exc_info:
                nim_resource.generate("test prompt")

            assert "Connection refused" in str(exc_info.value)

    def test_generate_returns_string_on_success(self, nim_resource: NIMResource) -> None:
        """generate() should return string content on success."""
        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status = Mock()
            mock_response.json.return_value = {
                "choices": [{"message": {"content": "Test response"}}]
            }
            mock_post.return_value = mock_response

            result = nim_resource.generate("test prompt")

            assert result == "Test response"
            # Ensure we're NOT returning error strings anymore
            assert not result.startswith("LLM error:")

    def test_generate_does_not_return_error_strings(self, nim_resource: NIMResource) -> None:
        """generate() should NOT return 'LLM error:' strings - it should raise."""
        with patch("requests.post") as mock_post:
            mock_post.side_effect = requests.exceptions.Timeout("Timeout")

            # Old behavior would return "LLM error: ..."
            # New behavior should raise exception
            with pytest.raises(NIMError):
                result = nim_resource.generate("test prompt")
                # If we get here, the old behavior is still in place
                assert not result.startswith("LLM error:"), (
                    "generate() should raise exceptions, not return error strings"
                )


class TestNIMResourceHealthCheck:
    """Test NIMResource health_check behavior."""

    @pytest.fixture
    def nim_resource(self) -> NIMResource:
        """Create NIM resource for testing."""
        return NIMResource(endpoint="http://test-nim:8000", model="test-model")

    def test_health_check_returns_true_when_healthy(self, nim_resource: NIMResource) -> None:
        """health_check() should return True when NIM is healthy."""
        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            assert nim_resource.health_check() is True

    def test_health_check_returns_false_when_unhealthy(self, nim_resource: NIMResource) -> None:
        """health_check() should return False when NIM is not ready."""
        with patch("requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 503
            mock_get.return_value = mock_response

            assert nim_resource.health_check() is False

    def test_health_check_returns_false_on_connection_error(
        self, nim_resource: NIMResource
    ) -> None:
        """health_check() should return False on connection error."""
        with patch("requests.get") as mock_get:
            mock_get.side_effect = requests.exceptions.ConnectionError()

            assert nim_resource.health_check() is False
