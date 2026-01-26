"""Tests for Safe Synthesizer retry wrapper.

TDD RED phase: These tests define the expected retry behavior
for Safe Synthesizer calls before implementation.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

# These imports will fail until we implement the module
from brev_pipelines.resources.safe_synth_retry import (
    SafeSynthError,
    SafeSynthJobFailedError,
    SafeSynthRetryConfig,
    SafeSynthServerError,
    SafeSynthTimeoutError,
    retry_safe_synth_call,
)


class TestSafeSynthRetryConfig:
    """Test SafeSynthRetryConfig dataclass."""

    def test_default_values(self) -> None:
        """Should have sensible defaults for Safe Synth jobs."""
        config = SafeSynthRetryConfig()

        assert config.max_retries == 3
        assert config.initial_delay == 10.0  # Safe Synth jobs take longer
        assert config.max_delay == 120.0
        assert config.exponential_base == 2.0
        assert config.jitter_factor == 0.2

    def test_custom_values(self) -> None:
        """Should accept custom configuration values."""
        config = SafeSynthRetryConfig(
            max_retries=5,
            initial_delay=30.0,
            max_delay=300.0,
            exponential_base=3.0,
            jitter_factor=0.1,
        )

        assert config.max_retries == 5
        assert config.initial_delay == 30.0
        assert config.max_delay == 300.0
        assert config.exponential_base == 3.0
        assert config.jitter_factor == 0.1


class TestSafeSynthExceptions:
    """Test Safe Synth exception type hierarchy."""

    def test_safe_synth_error_is_base_exception(self) -> None:
        """SafeSynthError should be the base exception."""
        error = SafeSynthError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"

    def test_safe_synth_timeout_error_inherits_from_base(self) -> None:
        """SafeSynthTimeoutError should inherit from SafeSynthError."""
        error = SafeSynthTimeoutError("Job timed out after 30 minutes")
        assert isinstance(error, SafeSynthError)
        assert isinstance(error, Exception)

    def test_safe_synth_server_error_includes_status_code(self) -> None:
        """SafeSynthServerError should include status code."""
        error = SafeSynthServerError(503, "Service unavailable")
        assert isinstance(error, SafeSynthError)
        assert error.status_code == 503
        assert "503" in str(error)
        assert "Service unavailable" in str(error)

    def test_safe_synth_job_failed_error_includes_job_id(self) -> None:
        """SafeSynthJobFailedError should include job ID and reason."""
        error = SafeSynthJobFailedError("synth-job-123", "Privacy check failed")
        assert isinstance(error, SafeSynthError)
        assert error.job_id == "synth-job-123"
        assert error.reason == "Privacy check failed"
        assert "synth-job-123" in str(error)
        assert "Privacy check failed" in str(error)


class TestRetrySafeSynthCall:
    """Test retry_safe_synth_call function."""

    def test_succeeds_on_first_try(self) -> None:
        """Should return result immediately on success."""
        mock_fn = Mock(return_value={"synthetic": "data"})

        result = retry_safe_synth_call(mock_fn, run_id="test-001")

        assert result == {"synthetic": "data"}
        assert mock_fn.call_count == 1

    def test_retries_on_server_error(self) -> None:
        """Should retry on server errors."""
        mock_fn = Mock(
            side_effect=[
                SafeSynthServerError(503, "Temporarily unavailable"),
                SafeSynthServerError(503, "Still unavailable"),
                {"synthetic": "data"},  # Success on 3rd try
            ]
        )

        with patch("time.sleep"):  # Don't actually sleep in tests
            result = retry_safe_synth_call(
                mock_fn,
                run_id="test-002",
                config=SafeSynthRetryConfig(max_retries=3),
            )

        assert result == {"synthetic": "data"}
        assert mock_fn.call_count == 3

    def test_retries_on_timeout_error(self) -> None:
        """Should retry on timeout errors."""
        mock_fn = Mock(
            side_effect=[
                SafeSynthTimeoutError("Job timed out"),
                {"synthetic": "data"},  # Success on 2nd try
            ]
        )

        with patch("time.sleep"):
            result = retry_safe_synth_call(
                mock_fn,
                run_id="test-003",
                config=SafeSynthRetryConfig(max_retries=3),
            )

        assert result == {"synthetic": "data"}
        assert mock_fn.call_count == 2

    def test_retries_on_connection_error(self) -> None:
        """Should retry on connection errors."""
        mock_fn = Mock(
            side_effect=[
                ConnectionError("Connection refused"),
                {"synthetic": "data"},
            ]
        )

        with patch("time.sleep"):
            result = retry_safe_synth_call(
                mock_fn,
                run_id="test-004",
                config=SafeSynthRetryConfig(max_retries=3),
            )

        assert result == {"synthetic": "data"}
        assert mock_fn.call_count == 2

    def test_exhausts_retries_and_raises(self) -> None:
        """Should raise after max retries exhausted."""
        mock_fn = Mock(side_effect=SafeSynthServerError(503, "Always failing"))

        with patch("time.sleep"), pytest.raises(SafeSynthServerError) as exc_info:
            retry_safe_synth_call(
                mock_fn,
                run_id="test-005",
                config=SafeSynthRetryConfig(max_retries=3),
            )

        assert mock_fn.call_count == 3
        assert exc_info.value.status_code == 503

    def test_no_retry_on_validation_error(self) -> None:
        """Should not retry on validation errors (client's fault)."""
        mock_fn = Mock(side_effect=ValueError("Invalid input data"))

        with pytest.raises(ValueError):
            retry_safe_synth_call(mock_fn, run_id="test-006")

        assert mock_fn.call_count == 1  # No retries

    def test_no_retry_on_type_error(self) -> None:
        """Should not retry on type errors."""
        mock_fn = Mock(side_effect=TypeError("Expected list, got dict"))

        with pytest.raises(TypeError):
            retry_safe_synth_call(mock_fn, run_id="test-007")

        assert mock_fn.call_count == 1

    def test_no_retry_on_key_error(self) -> None:
        """Should not retry on key errors."""
        mock_fn = Mock(side_effect=KeyError("missing_key"))

        with pytest.raises(KeyError):
            retry_safe_synth_call(mock_fn, run_id="test-008")

        assert mock_fn.call_count == 1

    def test_exponential_backoff_delays(self) -> None:
        """Should use exponential backoff between retries."""
        mock_fn = Mock(
            side_effect=[
                SafeSynthTimeoutError("Timeout 1"),
                SafeSynthTimeoutError("Timeout 2"),
                {"data": "success"},
            ]
        )

        sleep_calls: list[float] = []
        with patch("time.sleep", side_effect=lambda x: sleep_calls.append(x)):
            retry_safe_synth_call(
                mock_fn,
                run_id="test-009",
                config=SafeSynthRetryConfig(
                    max_retries=3,
                    initial_delay=10.0,
                    exponential_base=2.0,
                    jitter_factor=0.0,  # No jitter for predictable test
                ),
            )

        # First retry: 10s, Second retry: 20s (10 * 2^1)
        assert len(sleep_calls) == 2
        assert sleep_calls[0] == 10.0
        assert sleep_calls[1] == 20.0

    def test_respects_max_delay(self) -> None:
        """Should cap delay at max_delay."""
        mock_fn = Mock(
            side_effect=[
                SafeSynthTimeoutError("Timeout"),
                SafeSynthTimeoutError("Timeout"),
                SafeSynthTimeoutError("Timeout"),
                {"data": "success"},
            ]
        )

        sleep_calls: list[float] = []
        with patch("time.sleep", side_effect=lambda x: sleep_calls.append(x)):
            retry_safe_synth_call(
                mock_fn,
                run_id="test-010",
                config=SafeSynthRetryConfig(
                    max_retries=4,
                    initial_delay=50.0,
                    max_delay=60.0,  # Cap at 60s
                    exponential_base=2.0,
                    jitter_factor=0.0,
                ),
            )

        # Delays: 50s, 100s->60s (capped), 200s->60s (capped)
        assert len(sleep_calls) == 3
        assert sleep_calls[0] == 50.0
        assert sleep_calls[1] == 60.0  # Capped
        assert sleep_calls[2] == 60.0  # Capped

    def test_calls_logger_on_retry(self) -> None:
        """Should log retry attempts if logger provided."""
        mock_fn = Mock(
            side_effect=[
                SafeSynthServerError(503, "Retry me"),
                {"data": "success"},
            ]
        )
        mock_logger = Mock()

        with patch("time.sleep"):
            retry_safe_synth_call(
                mock_fn,
                run_id="test-011",
                logger=mock_logger,
            )

        # Should have logged the retry warning
        assert mock_logger.warning.called
        warning_call = str(mock_logger.warning.call_args)
        assert "1" in warning_call  # Attempt number
        assert "test-011" in warning_call or "Retry" in warning_call

    def test_calls_logger_on_final_failure(self) -> None:
        """Should log error when all retries exhausted."""
        mock_fn = Mock(side_effect=SafeSynthServerError(503, "Always failing"))
        mock_logger = Mock()

        with patch("time.sleep"), pytest.raises(SafeSynthServerError):
            retry_safe_synth_call(
                mock_fn,
                run_id="test-012",
                config=SafeSynthRetryConfig(max_retries=2),
                logger=mock_logger,
            )

        # Should have logged final error
        assert mock_logger.error.called

    def test_uses_default_config_when_none_provided(self) -> None:
        """Should use default config when none provided."""
        mock_fn = Mock(return_value={"data": "success"})

        # Should not raise - uses default config
        result = retry_safe_synth_call(mock_fn, run_id="test-013")

        assert result == {"data": "success"}
