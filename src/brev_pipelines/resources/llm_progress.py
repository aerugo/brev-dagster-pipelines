"""LLM Progress Tracker for aggregated logging in batch operations.

Provides clean progress reporting for LLM batch processing:
- Periodic progress logs (not every call)
- Aggregated success/failure counts
- Distinguishes expected mock fallback from unexpected errors
- Summary logging at completion

Usage:
    tracker = LLMProgressTracker(
        total_items=len(df),
        logger=context.log,
        mock_mode=nim.use_mock_fallback,
    )

    for row in rows:
        result = retry_with_backoff(...)
        tracker.record(result)

    tracker.finalize()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dagster import DagsterLogManager

    from brev_pipelines.resources.llm_retry import LLMCallResult


@dataclass
class LLMProgressTracker:
    """Aggregates LLM call results and logs progress periodically.

    Attributes:
        total_items: Total number of items to process.
        logger: Dagster logger for output.
        log_interval: Number of items between progress logs.
        mock_mode: If True, NIMServiceUnavailableError is expected (local dev).
    """

    total_items: int
    logger: DagsterLogManager
    log_interval: int = 100
    mock_mode: bool = False

    # Counters
    processed: int = field(default=0, init=False)
    success_count: int = field(default=0, init=False)
    fallback_count: int = field(default=0, init=False)
    mock_fallback_count: int = field(default=0, init=False)

    # Track unexpected errors for final summary
    unexpected_errors: list[tuple[str, str]] = field(default_factory=list, init=False)

    def record(self, result: LLMCallResult[object]) -> None:
        """Record a result and log progress if needed.

        Args:
            result: The LLM call result to record.
        """
        self.processed += 1

        if result.status == "success":
            self.success_count += 1
        else:
            self.fallback_count += 1

            # Distinguish expected mock fallback from unexpected errors
            is_mock_fallback = (
                self.mock_mode and result.error_type == "NIMServiceUnavailableError"
            )

            if is_mock_fallback:
                self.mock_fallback_count += 1
            else:
                # Track unexpected errors for summary
                self.unexpected_errors.append(
                    (result.record_id, result.error_message or "Unknown error")
                )

        # Log progress at intervals
        if self.processed % self.log_interval == 0:
            self._log_progress()

    def _log_progress(self) -> None:
        """Log current progress."""
        pct = (self.processed / self.total_items) * 100
        msg = f"Progress: {self.processed}/{self.total_items} ({pct:.0f}%)"

        if self.fallback_count > 0:
            if self.mock_mode and self.mock_fallback_count == self.fallback_count:
                # All fallbacks are expected mock mode
                msg += f" [mock mode: {self.mock_fallback_count} using fallback]"
            else:
                # Some real errors
                real_errors = self.fallback_count - self.mock_fallback_count
                msg += f" [{self.success_count} success, {real_errors} errors]"

        self.logger.info(msg)

    def finalize(self) -> None:
        """Log final summary."""
        real_errors = len(self.unexpected_errors)

        # Always log completion summary
        if self.mock_mode and self.mock_fallback_count > 0:
            self.logger.info(
                f"Completed: {self.processed}/{self.total_items} - "
                f"mock mode enabled, {self.mock_fallback_count} items used fallback values"
            )
        elif real_errors == 0:
            self.logger.info(
                f"Completed: {self.processed}/{self.total_items} - "
                f"all items processed successfully"
            )
        else:
            self.logger.info(
                f"Completed: {self.processed}/{self.total_items} - "
                f"{self.success_count} success, {real_errors} failures"
            )

        # Log unexpected errors (not mock fallback)
        if real_errors > 0:
            # Group errors by type for cleaner output
            error_sample = self.unexpected_errors[:5]
            self.logger.error(
                f"{real_errors} items failed with unexpected errors. "
                f"Sample: {[f'{rid}: {err[:50]}...' for rid, err in error_sample]}"
            )
            if real_errors > 5:
                self.logger.error(f"... and {real_errors - 5} more errors")
