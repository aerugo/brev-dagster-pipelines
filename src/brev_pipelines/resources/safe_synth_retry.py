"""Retry wrapper for Safe Synthesizer API calls.

Provides exponential backoff with jitter for transient failures during
synthetic data generation. Safe Synthesizer jobs can take 10-30+ minutes,
so retry delays are longer than typical HTTP retries.

Usage:
    from brev_pipelines.resources.safe_synth_retry import (
        retry_safe_synth_call,
        SafeSynthRetryConfig,
    )

    result = retry_safe_synth_call(
        lambda: safe_synth.synthesize(data, run_id, config),
        run_id=run_id,
        config=SafeSynthRetryConfig(max_retries=3),
        logger=context.log,
    )
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

    from dagster import DagsterLogManager

T = TypeVar("T")


# =============================================================================
# Exception Types
# =============================================================================


class SafeSynthError(Exception):
    """Base exception for Safe Synthesizer errors."""

    pass


class SafeSynthTimeoutError(SafeSynthError):
    """Raised when Safe Synthesizer job times out."""

    pass


class SafeSynthServerError(SafeSynthError):
    """Raised when Safe Synthesizer returns server error."""

    def __init__(self, status_code: int, message: str) -> None:
        """Initialize with status code and message."""
        self.status_code = status_code
        super().__init__(f"Safe Synthesizer error {status_code}: {message}")


class SafeSynthJobFailedError(SafeSynthError):
    """Raised when Safe Synthesizer job fails."""

    def __init__(self, job_id: str, reason: str) -> None:
        """Initialize with job ID and failure reason."""
        self.job_id = job_id
        self.reason = reason
        super().__init__(f"Safe Synthesizer job {job_id} failed: {reason}")


# Retryable errors - server issues that may be transient
_BASE_RETRYABLE: tuple[type[Exception], ...] = (
    SafeSynthTimeoutError,
    SafeSynthServerError,
    ConnectionError,
    TimeoutError,
)

# Add NeMo SDK exceptions if available (SDK may not be installed in local dev)
try:
    from nemo_microservices import APIError, APIStatusError

    RETRYABLE_ERRORS: tuple[type[Exception], ...] = (*_BASE_RETRYABLE, APIError, APIStatusError)
except ImportError:
    RETRYABLE_ERRORS = _BASE_RETRYABLE


# =============================================================================
# Retry Configuration
# =============================================================================


@dataclass
class SafeSynthRetryConfig:
    """Configuration for Safe Synthesizer retry behavior.

    Safe Synthesizer jobs take longer than typical API calls (10-30+ minutes),
    so delays are longer to allow the service to recover.

    Attributes:
        max_retries: Maximum number of retry attempts.
        initial_delay: Initial delay in seconds before first retry.
        max_delay: Maximum delay between retries.
        exponential_base: Base for exponential backoff calculation.
        jitter_factor: Random jitter factor (0.2 = +/-20%).
    """

    max_retries: int = 3
    initial_delay: float = 10.0  # Longer initial delay for heavy jobs
    max_delay: float = 120.0  # Max 2 minutes between retries
    exponential_base: float = 2.0
    jitter_factor: float = 0.2


# =============================================================================
# Retry Logic
# =============================================================================


def calculate_backoff_delay(
    attempt: int,
    config: SafeSynthRetryConfig,
) -> float:
    """Calculate delay before next retry attempt.

    Uses exponential backoff with optional jitter.

    Args:
        attempt: Current attempt number (1-indexed).
        config: Retry configuration.

    Returns:
        Delay in seconds with jitter applied.
    """
    # Exponential backoff: initial_delay * (base ^ (attempt - 1))
    delay = config.initial_delay * (config.exponential_base ** (attempt - 1))

    # Cap at max delay
    delay = min(delay, config.max_delay)

    # Apply jitter (+/-jitter_factor)
    if config.jitter_factor > 0:
        jitter = delay * config.jitter_factor
        delay = delay + random.uniform(-jitter, jitter)

    return max(0, delay)


def retry_safe_synth_call(
    fn: Callable[[], T],
    run_id: str,
    config: SafeSynthRetryConfig | None = None,
    logger: DagsterLogManager | None = None,
) -> T:
    """Execute Safe Synthesizer call with retry logic.

    Retries on transient server errors with exponential backoff.
    Does NOT retry on validation errors or other client-side issues.

    Args:
        fn: Callable that performs the Safe Synthesizer operation.
        run_id: Run ID for logging context.
        config: Retry configuration. Uses defaults if not provided.
        logger: Optional Dagster logger for status updates.

    Returns:
        Result from successful function call.

    Raises:
        SafeSynthError: If all retries exhausted.
        ValueError: If input validation fails (no retry).
        TypeError: If type validation fails (no retry).
        KeyError: If key validation fails (no retry).
        Other: Non-retryable exceptions propagate immediately.
    """
    if config is None:
        config = SafeSynthRetryConfig()

    last_error: Exception | None = None

    for attempt in range(1, config.max_retries + 1):
        try:
            return fn()

        except RETRYABLE_ERRORS as e:
            last_error = e

            if attempt < config.max_retries:
                delay = calculate_backoff_delay(attempt, config)

                if logger:
                    logger.warning(
                        f"Safe Synth call failed for {run_id} "
                        f"(attempt {attempt}/{config.max_retries}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )

                time.sleep(delay)
            else:
                if logger:
                    logger.error(
                        f"Safe Synth call failed permanently for {run_id} "
                        f"after {config.max_retries} attempts. Last error: {e}"
                    )

        except (ValueError, TypeError, KeyError) as e:
            # Client-side errors - don't retry
            if logger:
                logger.error(f"Safe Synth validation error for {run_id} (not retrying): {e}")
            raise

    # All retries exhausted
    if last_error:
        raise last_error
    raise SafeSynthError(f"Safe Synth call failed for {run_id} after {config.max_retries} attempts")


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "SafeSynthError",
    "SafeSynthTimeoutError",
    "SafeSynthServerError",
    "SafeSynthJobFailedError",
    "SafeSynthRetryConfig",
    "RETRYABLE_ERRORS",
    "calculate_backoff_delay",
    "retry_safe_synth_call",
]
