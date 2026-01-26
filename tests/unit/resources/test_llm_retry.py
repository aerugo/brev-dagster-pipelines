"""Unit tests for LLM retry wrapper with exponential backoff.

Tests written BEFORE implementation (TDD - INV-P010).

This module tests:
- Backoff calculation with jitter
- Retry logic for transient failures
- Error type classification
- Validation functions for LLM responses
- LLMCallResult dataclass structure

All tests mock external dependencies per INV-P010.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    pass


# =============================================================================
# Test RetryConfig
# =============================================================================


class TestRetryConfig:
    """Tests for RetryConfig Pydantic model."""

    def test_default_values(self) -> None:
        """Test RetryConfig has sensible defaults."""
        from brev_pipelines.resources.llm_retry import RetryConfig

        config = RetryConfig()

        assert config.max_retries == 5
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter == 0.2

    def test_custom_values(self) -> None:
        """Test RetryConfig accepts custom values."""
        from brev_pipelines.resources.llm_retry import RetryConfig

        config = RetryConfig(
            max_retries=3,
            base_delay=0.5,
            max_delay=30.0,
            exponential_base=3.0,
            jitter=0.1,
        )

        assert config.max_retries == 3
        assert config.base_delay == 0.5
        assert config.max_delay == 30.0
        assert config.exponential_base == 3.0
        assert config.jitter == 0.1

    def test_config_is_frozen(self) -> None:
        """Test RetryConfig is immutable (frozen)."""
        from pydantic import ValidationError

        from brev_pipelines.resources.llm_retry import RetryConfig

        config = RetryConfig()

        with pytest.raises(ValidationError):
            config.max_retries = 10  # type: ignore[misc]


# =============================================================================
# Test calculate_backoff
# =============================================================================


class TestCalculateBackoff:
    """Tests for backoff calculation."""

    def test_first_retry_uses_base_delay(self) -> None:
        """First retry (attempt=0) should use base delay."""
        from brev_pipelines.resources.llm_retry import RetryConfig, calculate_backoff

        config = RetryConfig(base_delay=1.0, jitter=0.0)
        delay = calculate_backoff(attempt=0, config=config)

        assert delay == 1.0

    def test_exponential_increase(self) -> None:
        """Delay should increase exponentially."""
        from brev_pipelines.resources.llm_retry import RetryConfig, calculate_backoff

        config = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter=0.0)

        assert calculate_backoff(0, config) == 1.0
        assert calculate_backoff(1, config) == 2.0
        assert calculate_backoff(2, config) == 4.0
        assert calculate_backoff(3, config) == 8.0
        assert calculate_backoff(4, config) == 16.0

    def test_max_delay_cap(self) -> None:
        """Delay should be capped at max_delay."""
        from brev_pipelines.resources.llm_retry import RetryConfig, calculate_backoff

        config = RetryConfig(base_delay=1.0, max_delay=5.0, jitter=0.0)
        delay = calculate_backoff(attempt=10, config=config)

        assert delay == 5.0

    def test_jitter_applied_within_bounds(self) -> None:
        """Jitter should add randomness within bounds."""
        from brev_pipelines.resources.llm_retry import RetryConfig, calculate_backoff

        config = RetryConfig(base_delay=10.0, jitter=0.2)  # ±20% jitter
        delays = [calculate_backoff(0, config) for _ in range(100)]

        # With 20% jitter on 10s, range is 8-12
        assert all(8.0 <= d <= 12.0 for d in delays)
        # Should have some variation (not all identical)
        assert len({round(d, 2) for d in delays}) > 1

    def test_zero_jitter_produces_deterministic_results(self) -> None:
        """Zero jitter should produce identical delays."""
        from brev_pipelines.resources.llm_retry import RetryConfig, calculate_backoff

        config = RetryConfig(base_delay=5.0, jitter=0.0)
        delays = [calculate_backoff(1, config) for _ in range(10)]

        assert all(d == delays[0] for d in delays)

    def test_negative_delay_not_possible(self) -> None:
        """Delay should never be negative even with jitter."""
        from brev_pipelines.resources.llm_retry import RetryConfig, calculate_backoff

        config = RetryConfig(base_delay=0.1, jitter=0.5)  # Aggressive jitter
        delays = [calculate_backoff(0, config) for _ in range(100)]

        assert all(d >= 0.0 for d in delays)


# =============================================================================
# Test Error Types
# =============================================================================


class TestErrorTypes:
    """Tests for retryable error types."""

    def test_retryable_error_is_exception(self) -> None:
        """RetryableError should be an Exception."""
        from brev_pipelines.resources.llm_retry import RetryableError

        assert issubclass(RetryableError, Exception)
        error = RetryableError("test error")
        assert str(error) == "test error"

    def test_validation_error_is_retryable(self) -> None:
        """ValidationError should be a RetryableError."""
        from brev_pipelines.resources.llm_retry import RetryableError, ValidationError

        assert issubclass(ValidationError, RetryableError)
        error = ValidationError("invalid json")
        assert isinstance(error, RetryableError)

    def test_timeout_error_is_retryable(self) -> None:
        """LLMTimeoutError should be a RetryableError."""
        from brev_pipelines.resources.llm_retry import LLMTimeoutError, RetryableError

        assert issubclass(LLMTimeoutError, RetryableError)

    def test_rate_limit_error_is_retryable(self) -> None:
        """LLMRateLimitError should be a RetryableError."""
        from brev_pipelines.resources.llm_retry import LLMRateLimitError, RetryableError

        assert issubclass(LLMRateLimitError, RetryableError)

    def test_server_error_is_retryable(self) -> None:
        """LLMServerError should be a RetryableError."""
        from brev_pipelines.resources.llm_retry import LLMServerError, RetryableError

        assert issubclass(LLMServerError, RetryableError)


# =============================================================================
# Test LLMCallResult
# =============================================================================


class TestLLMCallResult:
    """Tests for LLMCallResult dataclass."""

    def test_success_result_structure(self) -> None:
        """Success result should have correct fields."""
        from brev_pipelines.resources.llm_retry import LLMCallResult

        result: LLMCallResult[dict[str, int]] = LLMCallResult(
            record_id="test_001",
            status="success",
            response='{"count": 5}',
            parsed_data={"count": 5},
            attempts=1,
            duration_ms=100,
        )

        assert result.record_id == "test_001"
        assert result.status == "success"
        assert result.response == '{"count": 5}'
        assert result.parsed_data == {"count": 5}
        assert result.attempts == 1
        assert result.duration_ms == 100
        assert result.fallback_used is False
        assert result.error_type is None
        assert result.error_message is None

    def test_failed_result_structure(self) -> None:
        """Failed result should have error details."""
        from brev_pipelines.resources.llm_retry import LLMCallResult

        result: LLMCallResult[dict[str, int]] = LLMCallResult(
            record_id="test_002",
            status="failed",
            error_type="LLMTimeoutError",
            error_message="Connection timed out after 30s",
            attempts=5,
            fallback_used=True,
            fallback_values={"count": 0},
            duration_ms=31000,
        )

        assert result.status == "failed"
        assert result.error_type == "LLMTimeoutError"
        assert result.error_message == "Connection timed out after 30s"
        assert result.attempts == 5
        assert result.fallback_used is True
        assert result.fallback_values == {"count": 0}
        assert result.parsed_data is None

    def test_result_is_generic(self) -> None:
        """LLMCallResult should be generic over parsed data type."""
        from brev_pipelines.resources.llm_retry import LLMCallResult
        from brev_pipelines.types import SpeechClassification  # noqa: F401

        # Should type-check with SpeechClassification
        result: LLMCallResult[SpeechClassification] = LLMCallResult(
            record_id="test",
            status="success",
            parsed_data={
                "monetary_stance": 3,
                "trade_stance": 3,
                "tariff_mention": 0,
                "economic_outlook": 3,
            },
            attempts=1,
            duration_ms=100,
        )

        assert result.parsed_data is not None
        assert result.parsed_data["monetary_stance"] == 3


# =============================================================================
# Test retry_with_backoff
# =============================================================================


class TestRetryWithBackoff:
    """Tests for retry_with_backoff function."""

    def test_success_on_first_try(self) -> None:
        """Successful call returns immediately without retries."""
        from brev_pipelines.resources.llm_retry import RetryConfig, retry_with_backoff

        call_count = 0

        def llm_call() -> str:
            nonlocal call_count
            call_count += 1
            return '{"monetary_stance": "neutral"}'

        def validate(response: str) -> dict[str, str]:
            return {"monetary_stance": "neutral"}

        def fallback() -> dict[str, str]:
            return {"monetary_stance": "neutral"}

        result = retry_with_backoff(
            fn=llm_call,
            validate_fn=validate,
            record_id="test_001",
            fallback_fn=fallback,
            config=RetryConfig(max_retries=5),
        )

        assert result.status == "success"
        assert result.attempts == 1
        assert result.fallback_used is False
        assert call_count == 1

    def test_retry_on_nim_exception(self) -> None:
        """NIM exceptions should trigger retry."""
        from brev_pipelines.resources.llm_retry import RetryConfig, retry_with_backoff
        from brev_pipelines.resources.nim import NIMTimeoutError

        call_count = 0

        def llm_call() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise NIMTimeoutError("Request timed out")
            return '{"result": "ok"}'

        def validate(response: str) -> dict[str, str]:
            return {"result": "ok"}

        def fallback() -> dict[str, str]:
            return {"result": "fallback"}

        config = RetryConfig(max_retries=5, base_delay=0.01, jitter=0.0)
        result = retry_with_backoff(
            fn=llm_call,
            validate_fn=validate,
            record_id="test_002",
            fallback_fn=fallback,
            config=config,
        )

        assert result.status == "success"
        assert result.attempts == 3
        assert result.fallback_used is False
        assert call_count == 3

    def test_retry_on_validation_error(self) -> None:
        """Validation failures should trigger retry."""
        from brev_pipelines.resources.llm_retry import (
            RetryConfig,
            ValidationError,
            retry_with_backoff,
        )

        call_count = 0

        def llm_call() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                return "invalid json response"
            return '{"valid": "json"}'

        def validate(response: str) -> dict[str, str]:
            if "invalid" in response:
                raise ValidationError("Invalid JSON")
            return {"valid": "json"}

        def fallback() -> dict[str, str]:
            return {"valid": "fallback"}

        config = RetryConfig(max_retries=5, base_delay=0.01, jitter=0.0)
        result = retry_with_backoff(
            fn=llm_call,
            validate_fn=validate,
            record_id="test_003",
            fallback_fn=fallback,
            config=config,
        )

        assert result.status == "success"
        assert result.attempts == 2
        assert call_count == 2

    def test_fallback_after_max_retries(self) -> None:
        """After max retries, fallback values should be used."""
        from brev_pipelines.resources.llm_retry import RetryConfig, retry_with_backoff
        from brev_pipelines.resources.nim import NIMServerError

        def llm_call() -> str:
            raise NIMServerError(503, "service unavailable")

        def validate(response: str) -> dict[str, str]:
            return {"result": "ok"}

        fallback_values = {"result": "fallback", "is_fallback": "true"}

        def fallback() -> dict[str, str]:
            return fallback_values

        config = RetryConfig(max_retries=3, base_delay=0.01, jitter=0.0)
        result = retry_with_backoff(
            fn=llm_call,
            validate_fn=validate,
            record_id="test_004",
            fallback_fn=fallback,
            config=config,
        )

        assert result.status == "failed"
        assert result.attempts == 3
        assert result.fallback_used is True
        assert result.fallback_values == fallback_values
        assert result.error_type is not None

    def test_error_type_classification_timeout(self) -> None:
        """Timeout errors should be classified as LLMTimeoutError."""
        from brev_pipelines.resources.llm_retry import RetryConfig, retry_with_backoff
        from brev_pipelines.resources.nim import NIMTimeoutError

        def llm_call() -> str:
            raise NIMTimeoutError("timeout exceeded")

        def validate(response: str) -> dict[str, str]:
            return {}

        def fallback() -> dict[str, str]:
            return {}

        config = RetryConfig(max_retries=1, base_delay=0.01, jitter=0.0)
        result = retry_with_backoff(
            fn=llm_call,
            validate_fn=validate,
            record_id="test_timeout",
            fallback_fn=fallback,
            config=config,
        )

        assert result.error_type == "LLMTimeoutError"

    def test_error_type_classification_rate_limit(self) -> None:
        """Rate limit errors should be classified as LLMRateLimitError."""
        from brev_pipelines.resources.llm_retry import RetryConfig, retry_with_backoff
        from brev_pipelines.resources.nim import NIMRateLimitError

        def llm_call() -> str:
            raise NIMRateLimitError("429 rate limit exceeded")

        def validate(response: str) -> dict[str, str]:
            return {}

        def fallback() -> dict[str, str]:
            return {}

        config = RetryConfig(max_retries=1, base_delay=0.01, jitter=0.0)
        result = retry_with_backoff(
            fn=llm_call,
            validate_fn=validate,
            record_id="test_rate_limit",
            fallback_fn=fallback,
            config=config,
        )

        assert result.error_type == "LLMRateLimitError"

    def test_error_type_classification_server_error(self) -> None:
        """Server errors should be classified as LLMServerError."""
        from brev_pipelines.resources.llm_retry import RetryConfig, retry_with_backoff
        from brev_pipelines.resources.nim import NIMServerError

        def llm_call() -> str:
            raise NIMServerError(503, "service unavailable")

        def validate(response: str) -> dict[str, str]:
            return {}

        def fallback() -> dict[str, str]:
            return {}

        config = RetryConfig(max_retries=1, base_delay=0.01, jitter=0.0)
        result = retry_with_backoff(
            fn=llm_call,
            validate_fn=validate,
            record_id="test_server_error",
            fallback_fn=fallback,
            config=config,
        )

        assert result.error_type == "LLMServerError"

    def test_duration_tracking(self) -> None:
        """Duration should be tracked in milliseconds."""
        from brev_pipelines.resources.llm_retry import RetryConfig, retry_with_backoff

        def llm_call() -> str:
            time.sleep(0.05)  # 50ms
            return '{"ok": true}'

        def validate(response: str) -> dict[str, bool]:
            return {"ok": True}

        def fallback() -> dict[str, bool]:
            return {"ok": False}

        config = RetryConfig(max_retries=1)
        result = retry_with_backoff(
            fn=llm_call,
            validate_fn=validate,
            record_id="test_timing",
            fallback_fn=fallback,
            config=config,
        )

        assert result.duration_ms >= 50
        assert result.duration_ms < 500  # Reasonable upper bound

    def test_non_retryable_error_breaks_immediately(self) -> None:
        """Non-retryable errors should break out of retry loop."""
        from brev_pipelines.resources.llm_retry import RetryConfig, retry_with_backoff

        call_count = 0

        def llm_call() -> str:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Unexpected error")

        def validate(response: str) -> dict[str, str]:
            return {}

        def fallback() -> dict[str, str]:
            return {"fallback": "true"}

        config = RetryConfig(max_retries=5, base_delay=0.01, jitter=0.0)
        result = retry_with_backoff(
            fn=llm_call,
            validate_fn=validate,
            record_id="test_non_retryable",
            fallback_fn=fallback,
            config=config,
        )

        assert result.status == "failed"
        assert result.fallback_used is True
        assert call_count == 1  # Should not retry
        assert result.error_type == "unexpected_error"

    def test_logger_receives_warnings(self) -> None:
        """Logger should receive warning messages on retry."""
        from brev_pipelines.resources.llm_retry import RetryConfig, retry_with_backoff
        from brev_pipelines.resources.nim import NIMTimeoutError

        call_count = 0

        def llm_call() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise NIMTimeoutError("Request timed out")
            return '{"ok": true}'

        def validate(response: str) -> dict[str, bool]:
            return {"ok": True}

        def fallback() -> dict[str, bool]:
            return {"ok": False}

        mock_logger = MagicMock()
        config = RetryConfig(max_retries=3, base_delay=0.01, jitter=0.0)

        retry_with_backoff(
            fn=llm_call,
            validate_fn=validate,
            record_id="test_logging",
            fallback_fn=fallback,
            config=config,
            logger=mock_logger,
        )

        # Logger should have received warning for the retry
        mock_logger.warning.assert_called()

    def test_logger_receives_error_on_permanent_failure(self) -> None:
        """Logger should receive error message on permanent failure."""
        from brev_pipelines.resources.llm_retry import RetryConfig, retry_with_backoff
        from brev_pipelines.resources.nim import NIMServerError

        def llm_call() -> str:
            raise NIMServerError(500, "permanent failure")

        def validate(response: str) -> dict[str, str]:
            return {}

        def fallback() -> dict[str, str]:
            return {}

        mock_logger = MagicMock()
        config = RetryConfig(max_retries=2, base_delay=0.01, jitter=0.0)

        retry_with_backoff(
            fn=llm_call,
            validate_fn=validate,
            record_id="test_error_logging",
            fallback_fn=fallback,
            config=config,
            logger=mock_logger,
        )

        mock_logger.error.assert_called()

    def test_uses_default_config_when_none_provided(self) -> None:
        """Should use default RetryConfig when none provided."""
        from brev_pipelines.resources.llm_retry import retry_with_backoff

        def llm_call() -> str:
            return '{"ok": true}'

        def validate(response: str) -> dict[str, bool]:
            return {"ok": True}

        def fallback() -> dict[str, bool]:
            return {"ok": False}

        # Should not raise when config is None
        result = retry_with_backoff(
            fn=llm_call,
            validate_fn=validate,
            record_id="test_default_config",
            fallback_fn=fallback,
            config=None,
        )

        assert result.status == "success"


# =============================================================================
# Test Validation Functions
# =============================================================================


class TestValidateClassificationResponse:
    """Tests for validate_classification_response function."""

    def test_valid_classification_response(self) -> None:
        """Valid classification JSON should parse correctly."""
        from brev_pipelines.resources.llm_retry import validate_classification_response

        response = """{
            "monetary_stance": "hawkish",
            "trade_stance": "neutral",
            "tariff_mention": 1,
            "economic_outlook": "positive"
        }"""

        result = validate_classification_response(response)

        assert result["monetary_stance"] == 4  # hawkish -> 4
        assert result["trade_stance"] == 3  # neutral -> 3
        assert result["tariff_mention"] == 1
        assert result["economic_outlook"] == 4  # positive -> 4

    def test_extracts_json_from_surrounding_text(self) -> None:
        """Should extract JSON from response with surrounding text."""
        from brev_pipelines.resources.llm_retry import validate_classification_response

        response = """Based on my analysis, here is the classification:
        {"monetary_stance": "dovish", "trade_stance": "globalist", "tariff_mention": 0, "economic_outlook": "neutral"}
        This reflects the speech's focus on economic growth."""

        result = validate_classification_response(response)

        assert result["monetary_stance"] == 2  # dovish -> 2
        assert result["trade_stance"] == 4  # globalist -> 4

    def test_raises_on_no_json(self) -> None:
        """Should raise ValidationError when no JSON found."""
        from brev_pipelines.resources.llm_retry import (
            ValidationError,
            validate_classification_response,
        )

        response = "This is a plain text response with no JSON."

        with pytest.raises(ValidationError, match="No JSON found"):
            validate_classification_response(response)

    def test_raises_on_invalid_json(self) -> None:
        """Should raise ValidationError on malformed JSON."""
        from brev_pipelines.resources.llm_retry import (
            ValidationError,
            validate_classification_response,
        )

        response = '{"monetary_stance": "hawkish", trade_stance: neutral}'

        with pytest.raises(ValidationError, match="Invalid JSON"):
            validate_classification_response(response)

    def test_raises_on_missing_fields(self) -> None:
        """Should raise ValidationError when required fields missing."""
        from brev_pipelines.resources.llm_retry import (
            ValidationError,
            validate_classification_response,
        )

        response = '{"monetary_stance": "neutral"}'  # Missing other fields

        with pytest.raises(ValidationError, match="Missing required fields"):
            validate_classification_response(response)

    def test_raises_on_invalid_monetary_stance(self) -> None:
        """Should raise ValidationError for invalid monetary_stance value."""
        from brev_pipelines.resources.llm_retry import (
            ValidationError,
            validate_classification_response,
        )

        response = """{
            "monetary_stance": "invalid_value",
            "trade_stance": "neutral",
            "tariff_mention": 0,
            "economic_outlook": "neutral"
        }"""

        with pytest.raises(ValidationError, match="Invalid monetary_stance"):
            validate_classification_response(response)

    def test_raises_on_invalid_trade_stance(self) -> None:
        """Should raise ValidationError for invalid trade_stance value."""
        from brev_pipelines.resources.llm_retry import (
            ValidationError,
            validate_classification_response,
        )

        response = """{
            "monetary_stance": "neutral",
            "trade_stance": "invalid",
            "tariff_mention": 0,
            "economic_outlook": "neutral"
        }"""

        with pytest.raises(ValidationError, match="Invalid trade_stance"):
            validate_classification_response(response)

    def test_raises_on_invalid_outlook(self) -> None:
        """Should raise ValidationError for invalid economic_outlook value."""
        from brev_pipelines.resources.llm_retry import (
            ValidationError,
            validate_classification_response,
        )

        response = """{
            "monetary_stance": "neutral",
            "trade_stance": "neutral",
            "tariff_mention": 0,
            "economic_outlook": "invalid"
        }"""

        with pytest.raises(ValidationError, match="Invalid economic_outlook"):
            validate_classification_response(response)

    def test_raises_on_invalid_tariff_mention(self) -> None:
        """Should raise ValidationError for invalid tariff_mention value."""
        from brev_pipelines.resources.llm_retry import (
            ValidationError,
            validate_classification_response,
        )

        response = """{
            "monetary_stance": "neutral",
            "trade_stance": "neutral",
            "tariff_mention": 5,
            "economic_outlook": "neutral"
        }"""

        with pytest.raises(ValidationError, match="Invalid tariff_mention"):
            validate_classification_response(response)

    def test_accepts_string_tariff_mention(self) -> None:
        """Should accept tariff_mention as string '0' or '1'."""
        from brev_pipelines.resources.llm_retry import validate_classification_response

        response = """{
            "monetary_stance": "neutral",
            "trade_stance": "neutral",
            "tariff_mention": "1",
            "economic_outlook": "neutral"
        }"""

        result = validate_classification_response(response)
        assert result["tariff_mention"] == 1

    def test_all_monetary_stance_values(self) -> None:
        """Should correctly map all monetary stance values."""
        from brev_pipelines.resources.llm_retry import validate_classification_response

        stances = [
            ("very_dovish", 1),
            ("dovish", 2),
            ("neutral", 3),
            ("hawkish", 4),
            ("very_hawkish", 5),
        ]

        for stance_str, expected_int in stances:
            response = f"""{{
                "monetary_stance": "{stance_str}",
                "trade_stance": "neutral",
                "tariff_mention": 0,
                "economic_outlook": "neutral"
            }}"""

            result = validate_classification_response(response)
            assert result["monetary_stance"] == expected_int

    def test_all_trade_stance_values(self) -> None:
        """Should correctly map all trade stance values."""
        from brev_pipelines.resources.llm_retry import validate_classification_response

        stances = [
            ("very_protectionist", 1),
            ("protectionist", 2),
            ("neutral", 3),
            ("globalist", 4),
            ("very_globalist", 5),
        ]

        for stance_str, expected_int in stances:
            response = f"""{{
                "monetary_stance": "neutral",
                "trade_stance": "{stance_str}",
                "tariff_mention": 0,
                "economic_outlook": "neutral"
            }}"""

            result = validate_classification_response(response)
            assert result["trade_stance"] == expected_int

    def test_all_outlook_values(self) -> None:
        """Should correctly map all economic outlook values."""
        from brev_pipelines.resources.llm_retry import validate_classification_response

        outlooks = [
            ("very_negative", 1),
            ("negative", 2),
            ("neutral", 3),
            ("positive", 4),
            ("very_positive", 5),
        ]

        for outlook_str, expected_int in outlooks:
            response = f"""{{
                "monetary_stance": "neutral",
                "trade_stance": "neutral",
                "tariff_mention": 0,
                "economic_outlook": "{outlook_str}"
            }}"""

            result = validate_classification_response(response)
            assert result["economic_outlook"] == expected_int


class TestValidateSummaryResponse:
    """Tests for validate_summary_response function."""

    def test_valid_summary_response(self) -> None:
        """Valid summary should be returned as-is."""
        from brev_pipelines.resources.llm_retry import validate_summary_response

        summary = "This speech discusses monetary policy and economic growth. " * 3

        result = validate_summary_response(summary)

        assert result == summary.strip()

    def test_raises_on_llm_error(self) -> None:
        """Should raise ValidationError on LLM error response."""
        from brev_pipelines.resources.llm_retry import (
            ValidationError,
            validate_summary_response,
        )

        response = "LLM error: timeout exceeded"

        with pytest.raises(ValidationError):
            validate_summary_response(response)

    def test_raises_on_too_short_summary(self) -> None:
        """Should raise ValidationError for too short summary."""
        from brev_pipelines.resources.llm_retry import (
            ValidationError,
            validate_summary_response,
        )

        response = "Too short."

        with pytest.raises(ValidationError, match="too short"):
            validate_summary_response(response)

    def test_truncates_too_long_summary(self) -> None:
        """Should truncate summary that exceeds max length."""
        from brev_pipelines.resources.llm_retry import validate_summary_response

        long_summary = "x" * 2000  # Exceeds 1500 char limit

        result = validate_summary_response(long_summary)

        assert len(result) == 1503  # 1500 + "..."
        assert result.endswith("...")

    def test_strips_whitespace(self) -> None:
        """Should strip leading/trailing whitespace."""
        from brev_pipelines.resources.llm_retry import validate_summary_response

        summary = "   This is a valid summary with enough characters to pass validation.   "

        result = validate_summary_response(summary)

        assert not result.startswith(" ")
        assert not result.endswith(" ")

    def test_minimum_length_boundary(self) -> None:
        """Summary at exactly minimum length should pass."""
        from brev_pipelines.resources.llm_retry import validate_summary_response

        # Exactly 50 characters
        summary = "x" * 50

        result = validate_summary_response(summary)
        assert result == summary


# =============================================================================
# Test Type Annotations
# =============================================================================


class TestTypeAnnotations:
    """Tests verifying type annotations follow invariants."""

    def test_retry_config_uses_pydantic_v2(self) -> None:
        """RetryConfig should be a Pydantic BaseModel."""
        from pydantic import BaseModel

        from brev_pipelines.resources.llm_retry import RetryConfig  # noqa: I001

        assert issubclass(RetryConfig, BaseModel)

    def test_llm_call_result_is_generic(self) -> None:
        """LLMCallResult should be generic."""
        from brev_pipelines.resources.llm_retry import LLMCallResult

        # Check it's defined with Generic (has __orig_bases__ attribute)
        assert hasattr(LLMCallResult, "__orig_bases__")

    def test_calculate_backoff_has_return_type(self) -> None:
        """calculate_backoff should have return type annotation."""
        from typing import get_type_hints

        from brev_pipelines.resources.llm_retry import calculate_backoff

        hints = get_type_hints(calculate_backoff)
        assert "return" in hints
        assert hints["return"] is float

    def test_retry_with_backoff_has_full_annotations(self) -> None:
        """retry_with_backoff should have all parameters annotated."""
        import inspect

        from brev_pipelines.resources.llm_retry import retry_with_backoff

        sig = inspect.signature(retry_with_backoff)

        # Should have annotations for all parameters
        expected_params = ["fn", "validate_fn", "record_id", "fallback_fn", "config", "logger"]
        for param in expected_params:
            assert param in sig.parameters, f"Missing parameter: {param}"
            assert sig.parameters[param].annotation != inspect.Parameter.empty, (
                f"Missing annotation for: {param}"
            )

        # Should have return annotation
        assert sig.return_annotation != inspect.Signature.empty
