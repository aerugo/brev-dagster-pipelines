"""LLM retry wrapper with exponential backoff and dead letter pattern.

Provides robust error handling for LLM calls in data pipelines:
- Exponential backoff with jitter for transient failures
- Typed results with success/failure tracking
- Validation functions for structured LLM responses

Usage:
    result = retry_with_backoff(
        fn=lambda: nim.generate(prompt),
        validate_fn=validate_classification_response,
        record_id=record["reference"],
        fallback_fn=lambda: SpeechClassification(monetary_stance=3, ...),
    )

    if result.status == "success":
        data = result.parsed_data
    else:
        data = result.fallback_values
        log.warning(f"Failed: {result.error_message}")

Follows invariants:
- INV-P004: Complete type annotations
- INV-P005: No Any types (uses Generic[T])
- INV-P006: Modern Python 3.11+ syntax
- INV-P007: Pydantic v2 for RetryConfig
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from brev_pipelines.resources.nim import (
    NIMError,
    NIMServiceUnavailableError,
)
from brev_pipelines.resources.nim import (
    NIMRateLimitError as NIMRateLimitException,
)
from brev_pipelines.resources.nim import (
    NIMServerError as NIMServerException,
)
from brev_pipelines.resources.nim import (
    NIMTimeoutError as NIMTimeoutException,
)
from brev_pipelines.types import (
    MONETARY_STANCE_SCALE,
    OUTLOOK_SCALE,
    TRADE_STANCE_SCALE,
    SpeechClassification,
)

if TYPE_CHECKING:
    from collections.abc import Callable

T = TypeVar("T")


# =============================================================================
# Error Types
# =============================================================================


class RetryableError(Exception):
    """Base class for errors that should trigger a retry."""

    pass


class ValidationError(RetryableError):
    """LLM response failed validation (invalid JSON, missing fields)."""

    pass


class LLMTimeoutError(RetryableError):
    """LLM request timed out."""

    pass


class LLMRateLimitError(RetryableError):
    """LLM service returned rate limit error (429)."""

    pass


class LLMServerError(RetryableError):
    """LLM service returned server error (5xx)."""

    pass


# =============================================================================
# Configuration
# =============================================================================


class RetryConfig(BaseModel):
    """Configuration for LLM call retry behavior.

    Uses Pydantic v2 syntax (INV-P007).

    Attributes:
        max_retries: Maximum number of retry attempts (1-10).
        base_delay: Initial delay in seconds before first retry.
        max_delay: Maximum delay cap in seconds.
        exponential_base: Multiplier for exponential backoff.
        jitter: Random variation range (0.0-0.5), as fraction of delay.
    """

    model_config = ConfigDict(frozen=True)

    max_retries: int = Field(default=5, ge=1, le=10)
    base_delay: float = Field(default=1.0, ge=0.0)
    max_delay: float = Field(default=60.0, ge=1.0)
    exponential_base: float = Field(default=2.0, ge=1.0)
    jitter: float = Field(default=0.2, ge=0.0, le=0.5)


# =============================================================================
# Result Type
# =============================================================================


@dataclass
class LLMCallResult(Generic[T]):
    """Result of an LLM call with full metadata.

    Generic over the parsed data type T (INV-P005: no Any).

    Attributes:
        record_id: Identifier for the record being processed.
        status: "success" or "failed".
        response: Raw LLM response string (on success).
        parsed_data: Validated/parsed data (on success).
        error_type: Error class name (on failure).
        error_message: Error details (on failure).
        attempts: Number of attempts made.
        fallback_used: Whether fallback values were used.
        fallback_values: Fallback data (if fallback_used).
        duration_ms: Total processing time in milliseconds.
    """

    record_id: str
    status: str  # "success" or "failed"

    # Success fields
    response: str | None = None
    parsed_data: T | None = None

    # Failure fields
    error_type: str | None = None
    error_message: str | None = None
    attempts: int = 0

    # Fallback info
    fallback_used: bool = False
    fallback_values: T | None = None

    # Timing
    duration_ms: int = 0


# =============================================================================
# Backoff Calculation
# =============================================================================


def calculate_backoff(attempt: int, config: RetryConfig) -> float:
    """Calculate delay with exponential backoff and jitter.

    Args:
        attempt: Zero-based attempt number (0 = first retry).
        config: Retry configuration.

    Returns:
        Delay in seconds before next retry.
    """
    delay = config.base_delay * (config.exponential_base**attempt)
    delay = min(delay, config.max_delay)

    # Add jitter (±config.jitter percent)
    if config.jitter > 0:
        jitter_range = delay * config.jitter
        delay += random.uniform(-jitter_range, jitter_range)

    return max(0.0, delay)


# =============================================================================
# Error Classification
# =============================================================================


def _classify_llm_error(error_msg: str) -> type[RetryableError]:
    """Classify error message into specific error type.

    Args:
        error_msg: Error message from LLM response.

    Returns:
        Appropriate error class.
    """
    error_lower = error_msg.lower()

    if "timeout" in error_lower:
        return LLMTimeoutError
    if "429" in error_msg or "rate" in error_lower:
        return LLMRateLimitError
    if any(code in error_msg for code in ("500", "502", "503", "504")):
        return LLMServerError

    return RetryableError


# =============================================================================
# Main Retry Function
# =============================================================================


def retry_with_backoff(
    fn: Callable[[], str],
    validate_fn: Callable[[str], T],
    record_id: str,
    fallback_fn: Callable[[], T],
    config: RetryConfig | None = None,
) -> LLMCallResult[T]:
    """Execute LLM call with retry logic and validation.

    This is a pure function that returns structured results without logging.
    Use LLMProgressTracker for aggregated progress logging in batch operations.

    Args:
        fn: Function that makes the LLM call and returns raw response.
        validate_fn: Function that validates/parses response, raises ValidationError on failure.
        record_id: Identifier for the record being processed.
        fallback_fn: Function that returns fallback values if all retries fail.
        config: Retry configuration (defaults to RetryConfig()).

    Returns:
        LLMCallResult with either parsed data or fallback values.
    """
    if config is None:
        config = RetryConfig()

    start_time = time.time()
    last_error: Exception | None = None
    last_error_type: str | None = None

    for attempt in range(config.max_retries):
        try:
            # Make the LLM call
            raw_response = fn()

            # Validate and parse the response
            parsed_data = validate_fn(raw_response)

            # Success!
            duration_ms = int((time.time() - start_time) * 1000)
            return LLMCallResult(
                record_id=record_id,
                status="success",
                response=raw_response,
                parsed_data=parsed_data,
                attempts=attempt + 1,
                duration_ms=duration_ms,
            )

        except NIMServiceUnavailableError as e:
            # Service unavailable - skip retries, use fallback immediately
            last_error = e
            last_error_type = "NIMServiceUnavailableError"
            break

        except NIMTimeoutException as e:
            last_error = LLMTimeoutError(str(e))
            last_error_type = "LLMTimeoutError"

        except NIMRateLimitException as e:
            last_error = LLMRateLimitError(str(e))
            last_error_type = "LLMRateLimitError"

        except NIMServerException as e:
            last_error = LLMServerError(str(e))
            last_error_type = "LLMServerError"

        except NIMError as e:
            last_error = RetryableError(str(e))
            last_error_type = "RetryableError"

        except RetryableError as e:
            last_error = e
            last_error_type = type(e).__name__

        except Exception as e:
            # Non-retryable error - break immediately
            last_error = e
            last_error_type = "unexpected_error"
            break

        # Wait before retry (no logging - caller handles progress)
        if attempt < config.max_retries - 1:
            delay = calculate_backoff(attempt, config)
            time.sleep(delay)

    # All retries exhausted or early exit - use fallback
    duration_ms = int((time.time() - start_time) * 1000)
    fallback_values = fallback_fn()
    actual_attempts = attempt + 1

    return LLMCallResult(
        record_id=record_id,
        status="failed",
        error_type=last_error_type,
        error_message=str(last_error) if last_error else None,
        attempts=actual_attempts,
        fallback_used=True,
        fallback_values=fallback_values,
        duration_ms=duration_ms,
    )


# =============================================================================
# Validation Functions
# =============================================================================


def validate_classification_response(response: str) -> SpeechClassification:
    """Validate and parse classification LLM response.

    Args:
        response: Raw LLM response string.

    Returns:
        Validated SpeechClassification.

    Raises:
        ValidationError: If response cannot be parsed as valid classification.
    """
    # Try to extract JSON from response
    json_match = re.search(r"\{[^}]+\}", response)
    if not json_match:
        raise ValidationError(f"No JSON found in response: {response[:100]}...")

    try:
        result = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON: {e}") from e

    # Validate required fields
    required_fields = ["monetary_stance", "trade_stance", "tariff_mention", "economic_outlook"]
    missing = [f for f in required_fields if f not in result]
    if missing:
        raise ValidationError(f"Missing required fields: {missing}")

    # Map string values to numeric scales
    monetary_raw = result.get("monetary_stance", "")
    monetary = MONETARY_STANCE_SCALE.get(monetary_raw)
    if monetary is None:
        raise ValidationError(f"Invalid monetary_stance: {monetary_raw}")

    trade_raw = result.get("trade_stance", "")
    trade = TRADE_STANCE_SCALE.get(trade_raw)
    if trade is None:
        raise ValidationError(f"Invalid trade_stance: {trade_raw}")

    outlook_raw = result.get("economic_outlook", "")
    outlook = OUTLOOK_SCALE.get(outlook_raw)
    if outlook is None:
        raise ValidationError(f"Invalid economic_outlook: {outlook_raw}")

    tariff = result.get("tariff_mention")
    if tariff not in (0, 1, "0", "1"):
        raise ValidationError(f"Invalid tariff_mention: {tariff}")

    return SpeechClassification(
        monetary_stance=monetary,
        trade_stance=trade,
        tariff_mention=int(tariff),
        economic_outlook=outlook,
    )


def validate_summary_response(response: str) -> str:
    """Validate summary LLM response.

    Args:
        response: Raw LLM response string.

    Returns:
        Validated summary string.

    Raises:
        ValidationError: If response is invalid.
    """
    stripped = response.strip()
    if len(stripped) < 50:
        raise ValidationError(f"Summary too short ({len(stripped)} chars)")

    # Truncate if too long
    if len(stripped) > 1500:
        return stripped[:1500] + "..."

    return stripped
