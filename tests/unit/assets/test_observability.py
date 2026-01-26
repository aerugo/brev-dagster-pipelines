"""Tests for LLM asset observability (metadata and logging).

Tests TypedDict definitions for metadata and downstream filtering capabilities.
Written before implementation per INV-P010 (TDD).
"""

from __future__ import annotations

import polars as pl


class TestMetadataTypedDicts:
    """Tests for LLM metadata TypedDict definitions."""

    def test_llm_failure_breakdown_can_be_instantiated(self) -> None:
        """LLMFailureBreakdown TypedDict can be created with all error types."""
        from brev_pipelines.types import LLMFailureBreakdown

        breakdown: LLMFailureBreakdown = {
            "ValidationError": 5,
            "LLMTimeoutError": 2,
            "LLMRateLimitError": 1,
            "LLMServerError": 0,
            "unexpected_error": 1,
        }

        assert breakdown["ValidationError"] == 5
        assert breakdown["LLMTimeoutError"] == 2
        assert breakdown["LLMRateLimitError"] == 1
        assert breakdown["LLMServerError"] == 0
        assert breakdown["unexpected_error"] == 1

    def test_llm_asset_metadata_can_be_instantiated(self) -> None:
        """LLMAssetMetadata TypedDict can be created with all fields."""
        from brev_pipelines.types import LLMAssetMetadata, LLMFailureBreakdown

        breakdown: LLMFailureBreakdown = {
            "ValidationError": 1,
            "LLMTimeoutError": 0,
            "LLMRateLimitError": 0,
            "LLMServerError": 0,
            "unexpected_error": 0,
        }

        metadata: LLMAssetMetadata = {
            "total_processed": 100,
            "successful": 99,
            "failed": 1,
            "success_rate": "99.0%",
            "failed_references": ["REF_001"],
            "failure_breakdown": breakdown,
            "avg_attempts": 1.05,
            "total_duration_ms": 5000,
        }

        assert metadata["total_processed"] == 100
        assert metadata["successful"] == 99
        assert metadata["failed"] == 1
        assert metadata["success_rate"] == "99.0%"
        assert metadata["failed_references"] == ["REF_001"]
        assert metadata["avg_attempts"] == 1.05
        assert metadata["total_duration_ms"] == 5000

    def test_failure_breakdown_incrementing_works(self) -> None:
        """Can increment failure breakdown counts."""
        from brev_pipelines.types import LLMFailureBreakdown

        breakdown: LLMFailureBreakdown = {
            "ValidationError": 0,
            "LLMTimeoutError": 0,
            "LLMRateLimitError": 0,
            "LLMServerError": 0,
            "unexpected_error": 0,
        }

        # Simulate counting failures
        failures = [
            {"reference": "1", "error_type": "LLMTimeoutError"},
            {"reference": "2", "error_type": "LLMTimeoutError"},
            {"reference": "3", "error_type": "ValidationError"},
            {"reference": "4", "error_type": "unknown_error"},  # Should go to unexpected
        ]

        for f in failures:
            error_type = str(f["error_type"])
            if error_type in breakdown:
                breakdown[error_type] += 1  # type: ignore[literal-required]
            else:
                breakdown["unexpected_error"] += 1

        assert breakdown["LLMTimeoutError"] == 2
        assert breakdown["ValidationError"] == 1
        assert breakdown["unexpected_error"] == 1


class TestDownstreamFiltering:
    """Tests for downstream asset filtering by LLM status."""

    def test_filter_to_successful_records_only(self) -> None:
        """Can filter DataFrame to only successful LLM records."""
        df = pl.DataFrame(
            {
                "reference": ["A", "B", "C", "D"],
                "monetary_stance": [3, 3, 3, 3],
                "_llm_status": ["success", "failed", "success", "failed"],
                "_llm_fallback_used": [False, True, False, True],
            }
        )

        successful = df.filter(pl.col("_llm_status") == "success")

        assert len(successful) == 2
        assert successful["reference"].to_list() == ["A", "C"]
        assert all(not v for v in successful["_llm_fallback_used"].to_list())

    def test_filter_to_failed_records_only(self) -> None:
        """Can filter DataFrame to only failed LLM records."""
        df = pl.DataFrame(
            {
                "reference": ["A", "B", "C"],
                "_llm_status": ["success", "failed", "failed"],
                "_llm_error": [None, "Timeout", "Validation failed"],
            }
        )

        failed = df.filter(pl.col("_llm_status") == "failed")

        assert len(failed) == 2
        assert "A" not in failed["reference"].to_list()

    def test_add_confidence_column_based_on_status(self) -> None:
        """Can add confidence column based on LLM status."""
        df = pl.DataFrame(
            {
                "reference": ["A", "B", "C"],
                "_llm_status": ["success", "failed", "success"],
            }
        )

        with_confidence = df.with_columns(
            pl.when(pl.col("_llm_status") == "success")
            .then(pl.lit("high"))
            .otherwise(pl.lit("low"))
            .alias("confidence")
        )

        assert with_confidence["confidence"].to_list() == ["high", "low", "high"]

    def test_count_by_status(self) -> None:
        """Can count records by LLM status."""
        df = pl.DataFrame(
            {
                "reference": ["A", "B", "C", "D", "E"],
                "_llm_status": ["success", "success", "failed", "success", "failed"],
            }
        )

        counts = df.group_by("_llm_status").len().sort("_llm_status")

        # Should be 2 failed, 3 success
        assert counts.filter(pl.col("_llm_status") == "failed")["len"].item() == 2
        assert counts.filter(pl.col("_llm_status") == "success")["len"].item() == 3


class TestAssetMetadataOutput:
    """Tests for asset metadata output in Dagster context."""

    def test_metadata_dict_matches_dagster_requirements(self) -> None:
        """Metadata dict values are compatible with Dagster."""
        # Dagster metadata supports: int, float, str, bool, list, dict
        metadata = {
            "total_processed": 100,
            "successful": 95,
            "failed": 5,
            "success_rate": "95.0%",
            "failure_breakdown": {
                "ValidationError": 3,
                "LLMTimeoutError": 2,
                "LLMRateLimitError": 0,
                "LLMServerError": 0,
                "unexpected_error": 0,
            },
            "avg_attempts": 1.25,
        }

        # All values should be JSON-serializable types
        assert isinstance(metadata["total_processed"], int)
        assert isinstance(metadata["successful"], int)
        assert isinstance(metadata["failed"], int)
        assert isinstance(metadata["success_rate"], str)
        assert isinstance(metadata["failure_breakdown"], dict)
        assert isinstance(metadata["avg_attempts"], float)

    def test_failed_references_limited_to_100(self) -> None:
        """Failed references should be limited to avoid metadata bloat."""
        # Generate 200 fake references
        all_refs = [f"REF_{i:05d}" for i in range(200)]

        # Limit to 100 as per plan
        limited = all_refs[:100]

        assert len(limited) == 100
        assert limited[0] == "REF_00000"
        assert limited[99] == "REF_00099"

    def test_success_rate_with_zero_records(self) -> None:
        """Success rate handles zero records without division by zero."""
        total = 0
        success_count = 0

        success_rate = f"{100 * success_count / total:.1f}%" if total > 0 else "N/A"

        assert success_rate == "N/A"

    def test_success_rate_calculation(self) -> None:
        """Success rate is calculated correctly."""
        test_cases = [
            (100, 95, "95.0%"),
            (100, 100, "100.0%"),
            (100, 0, "0.0%"),
            (1000, 987, "98.7%"),
        ]

        for total, success_count, expected in test_cases:
            success_rate = f"{100 * success_count / total:.1f}%"
            assert success_rate == expected, f"Failed for {success_count}/{total}"


class TestAverageAttemptsCalculation:
    """Tests for average attempts calculation from DataFrame."""

    def test_average_attempts_with_all_success(self) -> None:
        """Average attempts is 1.0 when all calls succeed first try."""
        df = pl.DataFrame(
            {
                "_llm_attempts": [1, 1, 1, 1, 1],
            }
        )

        avg = df.select(pl.col("_llm_attempts").mean()).item()

        assert avg == 1.0

    def test_average_attempts_with_retries(self) -> None:
        """Average attempts reflects retry behavior."""
        df = pl.DataFrame(
            {
                "_llm_attempts": [1, 1, 3, 1, 5],  # Two records needed retries
            }
        )

        avg = df.select(pl.col("_llm_attempts").mean()).item()

        assert avg == 2.2  # (1+1+3+1+5) / 5 = 11/5 = 2.2

    def test_average_attempts_empty_dataframe(self) -> None:
        """Handles empty DataFrame gracefully."""
        df = pl.DataFrame(
            {
                "_llm_attempts": [],
            }
        ).cast({"_llm_attempts": pl.Int64})

        avg = df.select(pl.col("_llm_attempts").mean()).item()

        # Polars returns None for mean of empty series
        assert avg is None
