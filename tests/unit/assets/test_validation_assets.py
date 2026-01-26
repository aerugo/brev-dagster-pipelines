"""Unit tests for validation assets.

Tests the validation asset functions for MinIO, LakeFS, NIM, and platform validation.
All external service calls are mocked per INV-P010.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

from brev_pipelines.assets.validation import (
    ValidationResult,
    quick_health_check,
    validate_lakefs,
    validate_minio,
    validate_nim,
    validate_platform,
)

if TYPE_CHECKING:
    from dagster import AssetExecutionContext


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_validation_result_to_dict(self) -> None:
        """Test ValidationResult.to_dict() returns correct structure."""
        result = ValidationResult(
            component="test",
            passed=True,
            tests=[{"name": "test1", "passed": True}],
            error=None,
            duration_ms=100.5,
        )
        d = result.to_dict()

        assert d["component"] == "test"
        assert d["passed"] is True
        assert len(d["tests"]) == 1
        assert d["tests"][0]["name"] == "test1"
        assert d["error"] is None
        assert d["duration_ms"] == 100.5

    def test_validation_result_with_error(self) -> None:
        """Test ValidationResult with error message."""
        result = ValidationResult(
            component="test",
            passed=False,
            error="Connection failed",
        )
        d = result.to_dict()

        assert d["passed"] is False
        assert d["error"] == "Connection failed"

    def test_validation_result_default_values(self) -> None:
        """Test ValidationResult default values."""
        result = ValidationResult(component="test", passed=True)
        d = result.to_dict()

        assert d["tests"] == []
        assert d["error"] is None
        assert d["duration_ms"] == 0.0


class TestValidateMinio:
    """Tests for validate_minio asset."""

    def test_validate_minio_all_tests_pass(
        self,
        asset_context: AssetExecutionContext,
        mock_minio_resource: MagicMock,
    ) -> None:
        """Test validate_minio when all tests pass."""
        mock_client = mock_minio_resource.get_client.return_value

        # Configure mock for list_buckets
        mock_client.list_buckets.return_value = [
            MagicMock(name="bucket1"),
            MagicMock(name="bucket2"),
        ]

        # Configure mock for bucket operations
        mock_client.bucket_exists.return_value = False  # Bucket doesn't exist yet

        # Capture written data to return in get_object
        captured_data: list[bytes] = []

        def capture_put(**kwargs: Any) -> None:
            data = kwargs.get("data")
            if data and hasattr(data, "read"):
                captured_data.append(data.read())

        # Use side_effect to capture both positional and keyword args
        def capture_put_object(
            bucket: str, key: str, data: Any, length: int, **kwargs: Any
        ) -> None:
            if hasattr(data, "read"):
                captured_data.append(data.read())

        mock_client.put_object.side_effect = capture_put_object

        # Configure mock for read to return captured data
        def get_object_mock(bucket: str, key: str) -> MagicMock:
            mock_response = MagicMock()
            # Return the captured data or default
            if captured_data:
                mock_response.read.return_value = captured_data[0]
            else:
                mock_response.read.return_value = b'{"test": true}'
            mock_response.close.return_value = None
            mock_response.release_conn.return_value = None
            return mock_response

        mock_client.get_object.side_effect = get_object_mock

        result = validate_minio(asset_context, mock_minio_resource)

        assert result["component"] == "minio"
        assert result["passed"] is True
        assert len(result["tests"]) == 7  # 7 tests in validate_minio
        assert result["duration_ms"] > 0

        # Verify all tests passed
        for test in result["tests"]:
            assert test["passed"] is True, f"Test {test['name']} failed"

    def test_validate_minio_connection_failure(
        self,
        asset_context: AssetExecutionContext,
        mock_minio_resource: MagicMock,
    ) -> None:
        """Test validate_minio when connection fails."""
        mock_client = mock_minio_resource.get_client.return_value
        mock_client.list_buckets.side_effect = ConnectionError("Connection refused")

        result = validate_minio(asset_context, mock_minio_resource)

        assert result["component"] == "minio"
        assert result["passed"] is False
        assert result["error"] == "Connection refused"

    def test_validate_minio_create_bucket_failure(
        self,
        asset_context: AssetExecutionContext,
        mock_minio_resource: MagicMock,
    ) -> None:
        """Test validate_minio when bucket creation fails."""
        mock_client = mock_minio_resource.get_client.return_value
        mock_client.list_buckets.return_value = []
        mock_client.bucket_exists.return_value = False
        mock_client.make_bucket.side_effect = Exception("Permission denied")

        result = validate_minio(asset_context, mock_minio_resource)

        assert result["passed"] is False
        # Find the create_bucket test
        create_test = next(t for t in result["tests"] if t["name"] == "create_bucket")
        assert create_test["passed"] is False
        assert "Permission denied" in create_test["error"]

    def test_validate_minio_data_mismatch(
        self,
        asset_context: AssetExecutionContext,
        mock_minio_resource: MagicMock,
    ) -> None:
        """Test validate_minio when read data doesn't match written data."""
        mock_client = mock_minio_resource.get_client.return_value
        mock_client.list_buckets.return_value = []
        mock_client.bucket_exists.return_value = False

        # Return different data than what was written
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"different": "data"}).encode()
        mock_response.close.return_value = None
        mock_response.release_conn.return_value = None
        mock_client.get_object.return_value = mock_response

        result = validate_minio(asset_context, mock_minio_resource)

        assert result["passed"] is False
        read_test = next(t for t in result["tests"] if t["name"] == "read_object")
        assert read_test["passed"] is False


class TestValidateLakefs:
    """Tests for validate_lakefs asset."""

    def test_validate_lakefs_all_tests_pass(
        self,
        asset_context: AssetExecutionContext,
        mock_lakefs_resource: MagicMock,
    ) -> None:
        """Test validate_lakefs when all tests pass."""
        result = validate_lakefs(asset_context, mock_lakefs_resource)

        assert result["component"] == "lakefs"
        assert result["passed"] is True
        assert len(result["tests"]) == 3
        assert result["duration_ms"] > 0

    def test_validate_lakefs_connection_failure(
        self,
        asset_context: AssetExecutionContext,
        mock_lakefs_resource: MagicMock,
    ) -> None:
        """Test validate_lakefs when health check fails."""
        mock_lakefs_resource.health_check.side_effect = ConnectionError("Cannot connect")

        result = validate_lakefs(asset_context, mock_lakefs_resource)

        assert result["passed"] is False
        assert result["error"] == "Cannot connect"

    def test_validate_lakefs_unhealthy(
        self,
        asset_context: AssetExecutionContext,
        mock_lakefs_resource: MagicMock,
    ) -> None:
        """Test validate_lakefs when service is unhealthy."""
        mock_lakefs_resource.health_check.return_value = False

        result = validate_lakefs(asset_context, mock_lakefs_resource)

        assert result["passed"] is False
        connection_test = next(t for t in result["tests"] if t["name"] == "connection")
        assert connection_test["passed"] is False

    def test_validate_lakefs_list_repos_failure(
        self,
        asset_context: AssetExecutionContext,
        mock_lakefs_resource: MagicMock,
    ) -> None:
        """Test validate_lakefs when list_repositories fails."""
        mock_lakefs_resource.list_repositories.side_effect = Exception("API error")

        result = validate_lakefs(asset_context, mock_lakefs_resource)

        assert result["passed"] is False
        repos_test = next(t for t in result["tests"] if t["name"] == "list_repositories")
        assert repos_test["passed"] is False


class TestValidateNim:
    """Tests for validate_nim asset."""

    def test_validate_nim_all_tests_pass(
        self,
        asset_context: AssetExecutionContext,
        mock_nim_resource: MagicMock,
    ) -> None:
        """Test validate_nim when all tests pass."""
        # Configure mock for generate
        mock_nim_resource.generate.return_value = "blue and full of stars."

        result = validate_nim(asset_context, mock_nim_resource)

        assert result["component"] == "nim"
        assert result["passed"] is True
        assert len(result["tests"]) == 3  # health, simple_completion, structured

    def test_validate_nim_health_check_failure(
        self,
        asset_context: AssetExecutionContext,
        mock_nim_resource: MagicMock,
    ) -> None:
        """Test validate_nim when health check fails."""
        mock_nim_resource.health_check.side_effect = ConnectionError("Service unavailable")

        result = validate_nim(asset_context, mock_nim_resource)

        assert result["passed"] is False
        assert result["error"] == "Service unavailable"

    def test_validate_nim_unhealthy_returns_early(
        self,
        asset_context: AssetExecutionContext,
        mock_nim_resource: MagicMock,
    ) -> None:
        """Test validate_nim returns early when service unhealthy."""
        mock_nim_resource.health_check.return_value = False

        result = validate_nim(asset_context, mock_nim_resource)

        assert result["passed"] is False
        assert result["error"] == "NIM service not healthy"
        # Should only have health check test
        assert len(result["tests"]) == 1

    def test_validate_nim_empty_response(
        self,
        asset_context: AssetExecutionContext,
        mock_nim_resource: MagicMock,
    ) -> None:
        """Test validate_nim when LLM returns empty response."""
        mock_nim_resource.generate.return_value = ""

        result = validate_nim(asset_context, mock_nim_resource)

        assert result["passed"] is False
        completion_test = next(t for t in result["tests"] if t["name"] == "simple_completion")
        assert completion_test["passed"] is False

    def test_validate_nim_error_in_response(
        self,
        asset_context: AssetExecutionContext,
        mock_nim_resource: MagicMock,
    ) -> None:
        """Test validate_nim when LLM returns error in response."""
        mock_nim_resource.generate.return_value = "Error: rate limit exceeded"

        result = validate_nim(asset_context, mock_nim_resource)

        assert result["passed"] is False
        completion_test = next(t for t in result["tests"] if t["name"] == "simple_completion")
        assert completion_test["passed"] is False

    def test_validate_nim_json_response(
        self,
        asset_context: AssetExecutionContext,
        mock_nim_resource: MagicMock,
    ) -> None:
        """Test validate_nim with valid JSON-like response."""

        def mock_generate(prompt: str, max_tokens: int) -> str:
            if "JSON" in prompt:
                return '{"status": "ok", "message": "working"}'
            return "valid response"

        mock_nim_resource.generate.side_effect = mock_generate

        result = validate_nim(asset_context, mock_nim_resource)

        assert result["passed"] is True
        structured_test = next(t for t in result["tests"] if t["name"] == "structured_response")
        assert structured_test["passed"] is True


class TestValidatePlatform:
    """Tests for validate_platform asset."""

    def test_validate_platform_all_pass(
        self,
        asset_context: AssetExecutionContext,
        mock_minio_resource: MagicMock,
    ) -> None:
        """Test validate_platform when all components pass."""
        minio_result = {"component": "minio", "passed": True, "tests": [], "error": None}
        lakefs_result = {"component": "lakefs", "passed": True, "tests": [], "error": None}
        nim_result = {"component": "nim", "passed": True, "tests": [], "error": None}

        result = validate_platform(
            asset_context, minio_result, lakefs_result, nim_result, mock_minio_resource
        )

        assert result["validation_run"]["overall_status"] == "PASSED"
        assert result["validation_run"]["passed_components"] == 3
        assert result["validation_run"]["total_components"] == 3
        assert "timestamp" in result["validation_run"]

    def test_validate_platform_one_failure(
        self,
        asset_context: AssetExecutionContext,
        mock_minio_resource: MagicMock,
    ) -> None:
        """Test validate_platform when one component fails."""
        minio_result = {"component": "minio", "passed": True, "tests": [], "error": None}
        lakefs_result = {"component": "lakefs", "passed": False, "tests": [], "error": "Failed"}
        nim_result = {"component": "nim", "passed": True, "tests": [], "error": None}

        result = validate_platform(
            asset_context, minio_result, lakefs_result, nim_result, mock_minio_resource
        )

        assert result["validation_run"]["overall_status"] == "FAILED"
        assert result["validation_run"]["passed_components"] == 2

    def test_validate_platform_all_fail(
        self,
        asset_context: AssetExecutionContext,
        mock_minio_resource: MagicMock,
    ) -> None:
        """Test validate_platform when all components fail."""
        minio_result = {"component": "minio", "passed": False, "tests": [], "error": "Err1"}
        lakefs_result = {"component": "lakefs", "passed": False, "tests": [], "error": "Err2"}
        nim_result = {"component": "nim", "passed": False, "tests": [], "error": "Err3"}

        result = validate_platform(
            asset_context, minio_result, lakefs_result, nim_result, mock_minio_resource
        )

        assert result["validation_run"]["overall_status"] == "FAILED"
        assert result["validation_run"]["passed_components"] == 0

    def test_validate_platform_stores_report(
        self,
        asset_context: AssetExecutionContext,
        mock_minio_resource: MagicMock,
    ) -> None:
        """Test validate_platform stores report to MinIO."""
        mock_client = mock_minio_resource.get_client.return_value

        minio_result = {"component": "minio", "passed": True, "tests": [], "error": None}
        lakefs_result = {"component": "lakefs", "passed": True, "tests": [], "error": None}
        nim_result = {"component": "nim", "passed": True, "tests": [], "error": None}

        result = validate_platform(
            asset_context, minio_result, lakefs_result, nim_result, mock_minio_resource
        )

        # Verify bucket was ensured
        mock_minio_resource.ensure_bucket.assert_called_once_with("data-products")

        # Verify put_object was called twice (timestamped + latest)
        assert mock_client.put_object.call_count == 2
        assert "report_location" in result

    def test_validate_platform_handles_minio_error(
        self,
        asset_context: AssetExecutionContext,
        mock_minio_resource: MagicMock,
    ) -> None:
        """Test validate_platform handles MinIO storage error gracefully."""
        mock_minio_resource.ensure_bucket.side_effect = Exception("Storage error")

        minio_result = {"component": "minio", "passed": True, "tests": [], "error": None}
        lakefs_result = {"component": "lakefs", "passed": True, "tests": [], "error": None}
        nim_result = {"component": "nim", "passed": True, "tests": [], "error": None}

        result = validate_platform(
            asset_context, minio_result, lakefs_result, nim_result, mock_minio_resource
        )

        # Should still succeed but note storage failure
        assert result["report_location"] == "not_stored"

    def test_validate_platform_summary_format(
        self,
        asset_context: AssetExecutionContext,
        mock_minio_resource: MagicMock,
    ) -> None:
        """Test validate_platform summary uses correct format."""
        minio_result = {"component": "minio", "passed": True, "tests": [], "error": None}
        lakefs_result = {"component": "lakefs", "passed": False, "tests": [], "error": None}
        nim_result = {"component": "nim", "passed": True, "tests": [], "error": None}

        result = validate_platform(
            asset_context, minio_result, lakefs_result, nim_result, mock_minio_resource
        )

        summary = result["summary"]
        assert "PASSED" in summary["minio"]
        assert "FAILED" in summary["lakefs"]
        assert "PASSED" in summary["nim"]


class TestQuickHealthCheck:
    """Tests for quick_health_check asset."""

    def test_quick_health_check_all_healthy(
        self,
        asset_context: AssetExecutionContext,
        mock_minio_resource: MagicMock,
        mock_lakefs_resource: MagicMock,
        mock_nim_resource: MagicMock,
    ) -> None:
        """Test quick_health_check when all services are healthy."""
        result = quick_health_check(
            asset_context,
            mock_minio_resource,
            mock_lakefs_resource,
            mock_nim_resource,
        )

        assert result["overall_status"] == "healthy"
        assert result["services"]["minio"]["status"] == "healthy"
        assert result["services"]["lakefs"]["status"] == "healthy"
        assert result["services"]["nim"]["status"] == "healthy"
        assert "timestamp" in result
        assert result["duration_ms"] > 0

    def test_quick_health_check_one_unhealthy(
        self,
        asset_context: AssetExecutionContext,
        mock_minio_resource: MagicMock,
        mock_lakefs_resource: MagicMock,
        mock_nim_resource: MagicMock,
    ) -> None:
        """Test quick_health_check when one service is unhealthy."""
        mock_nim_resource.health_check.return_value = False

        result = quick_health_check(
            asset_context,
            mock_minio_resource,
            mock_lakefs_resource,
            mock_nim_resource,
        )

        assert result["overall_status"] == "unhealthy"
        assert result["services"]["minio"]["status"] == "healthy"
        assert result["services"]["lakefs"]["status"] == "healthy"
        assert result["services"]["nim"]["status"] == "unhealthy"

    def test_quick_health_check_minio_error(
        self,
        asset_context: AssetExecutionContext,
        mock_minio_resource: MagicMock,
        mock_lakefs_resource: MagicMock,
        mock_nim_resource: MagicMock,
    ) -> None:
        """Test quick_health_check handles MinIO connection error."""
        mock_client = mock_minio_resource.get_client.return_value
        mock_client.list_buckets.side_effect = ConnectionError("Connection refused")

        result = quick_health_check(
            asset_context,
            mock_minio_resource,
            mock_lakefs_resource,
            mock_nim_resource,
        )

        assert result["overall_status"] == "unhealthy"
        assert result["services"]["minio"]["status"] == "unhealthy"
        assert result["services"]["minio"]["error"] is not None
        assert "Connection refused" in result["services"]["minio"]["error"]

    def test_quick_health_check_lakefs_error(
        self,
        asset_context: AssetExecutionContext,
        mock_minio_resource: MagicMock,
        mock_lakefs_resource: MagicMock,
        mock_nim_resource: MagicMock,
    ) -> None:
        """Test quick_health_check handles LakeFS error."""
        mock_lakefs_resource.health_check.side_effect = TimeoutError("Request timed out")

        result = quick_health_check(
            asset_context,
            mock_minio_resource,
            mock_lakefs_resource,
            mock_nim_resource,
        )

        assert result["overall_status"] == "unhealthy"
        assert result["services"]["lakefs"]["status"] == "unhealthy"
        assert "Request timed out" in result["services"]["lakefs"]["error"]

    def test_quick_health_check_all_unhealthy(
        self,
        asset_context: AssetExecutionContext,
        mock_minio_resource: MagicMock,
        mock_lakefs_resource: MagicMock,
        mock_nim_resource: MagicMock,
    ) -> None:
        """Test quick_health_check when all services fail."""
        mock_client = mock_minio_resource.get_client.return_value
        mock_client.list_buckets.side_effect = Exception("MinIO error")
        mock_lakefs_resource.health_check.side_effect = Exception("LakeFS error")
        mock_nim_resource.health_check.side_effect = Exception("NIM error")

        result = quick_health_check(
            asset_context,
            mock_minio_resource,
            mock_lakefs_resource,
            mock_nim_resource,
        )

        assert result["overall_status"] == "unhealthy"
        for service in result["services"].values():
            assert service["status"] == "unhealthy"
            assert service["error"] is not None

    def test_quick_health_check_error_truncation(
        self,
        asset_context: AssetExecutionContext,
        mock_minio_resource: MagicMock,
        mock_lakefs_resource: MagicMock,
        mock_nim_resource: MagicMock,
    ) -> None:
        """Test quick_health_check truncates long error messages."""
        long_error = "A" * 200  # Longer than 100 char limit
        mock_client = mock_minio_resource.get_client.return_value
        mock_client.list_buckets.side_effect = Exception(long_error)

        result = quick_health_check(
            asset_context,
            mock_minio_resource,
            mock_lakefs_resource,
            mock_nim_resource,
        )

        # Error should be truncated to 100 chars
        assert len(result["services"]["minio"]["error"]) == 100
