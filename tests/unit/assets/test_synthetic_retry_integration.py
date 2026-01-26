"""Integration tests for synthesis retry behavior.

Tests that the retry wrapper is properly integrated with synthetic_summaries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, Mock, patch

import polars as pl
import pytest
from dagster import build_asset_context

from brev_pipelines.assets.synthetic_speeches import synthetic_summaries
from brev_pipelines.resources.safe_synth_retry import SafeSynthServerError

if TYPE_CHECKING:
    from dagster import AssetExecutionContext


class TestSyntheticSummariesRetry:
    """Test retry behavior in synthetic_summaries asset."""

    @pytest.fixture
    def asset_context(self) -> AssetExecutionContext:
        """Create Dagster asset context."""
        return build_asset_context()

    @pytest.fixture
    def sample_input_df(self) -> pl.DataFrame:
        """Create sample input DataFrame."""
        return pl.DataFrame(
            {
                "reference": ["BIS_001"],
                "date": ["2024-01-15"],
                "central_bank": ["Test Bank"],
                "speaker": ["Test Speaker"],
                "title": ["Test Speech"],
                "summary": ["This is a test summary with enough content to be valid."],
                "monetary_stance": [3],
                "trade_stance": [3],
                "economic_outlook": [3],
                "tariff_mention": [0],
                "is_governor": [True],
            }
        )

    @pytest.fixture
    def mock_safe_synth(self) -> MagicMock:
        """Create mock Safe Synth resource."""
        return MagicMock()

    def _make_successful_response(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Create a successful Safe Synth response."""
        return (
            [{"reference": "SYNTH-000001", "summary": "Synthetic summary"}],
            {"mia_score": 0.95, "privacy_passed": True},
        )

    def test_retries_on_safe_synth_server_error(
        self,
        asset_context: AssetExecutionContext,
        sample_input_df: pl.DataFrame,
        mock_safe_synth: MagicMock,
    ) -> None:
        """Should retry when Safe Synth fails with server error."""
        # Fail twice, succeed on third
        mock_safe_synth.synthesize.side_effect = [
            SafeSynthServerError(503, "Temporarily unavailable"),
            SafeSynthServerError(503, "Still unavailable"),
            self._make_successful_response(),
        ]

        # Spy on context log
        asset_context.log.warning = Mock()

        with patch("time.sleep"):  # Don't actually sleep in tests
            result_df, evaluation = synthetic_summaries(
                asset_context,
                sample_input_df,
                mock_safe_synth,
            )

        # Should have called synthesize 3 times
        assert mock_safe_synth.synthesize.call_count == 3

        # Should have logged retry warnings
        assert asset_context.log.warning.called
        warning_calls = " ".join(str(c) for c in asset_context.log.warning.call_args_list)
        assert "attempt" in warning_calls.lower() or "failed" in warning_calls.lower()

    def test_succeeds_on_first_try(
        self,
        asset_context: AssetExecutionContext,
        sample_input_df: pl.DataFrame,
        mock_safe_synth: MagicMock,
    ) -> None:
        """Should return immediately on success without retry."""
        mock_safe_synth.synthesize.return_value = self._make_successful_response()

        result_df, evaluation = synthetic_summaries(
            asset_context,
            sample_input_df,
            mock_safe_synth,
        )

        # Should only call synthesize once
        assert mock_safe_synth.synthesize.call_count == 1

        # Should have synthetic data
        assert len(result_df) == 1

    def test_raises_after_exhausting_retries(
        self,
        asset_context: AssetExecutionContext,
        sample_input_df: pl.DataFrame,
        mock_safe_synth: MagicMock,
    ) -> None:
        """Should raise after all retries exhausted."""
        # Always fail
        mock_safe_synth.synthesize.side_effect = SafeSynthServerError(503, "Always failing")

        with patch("time.sleep"), pytest.raises(SafeSynthServerError):
            synthetic_summaries(
                asset_context,
                sample_input_df,
                mock_safe_synth,
            )

        # Should have tried max_retries times (3)
        assert mock_safe_synth.synthesize.call_count == 3

    def test_does_not_retry_on_validation_error(
        self,
        asset_context: AssetExecutionContext,
        sample_input_df: pl.DataFrame,
        mock_safe_synth: MagicMock,
    ) -> None:
        """Should not retry on validation errors (client's fault)."""
        mock_safe_synth.synthesize.side_effect = ValueError("Invalid input data")

        with pytest.raises(ValueError):
            synthetic_summaries(
                asset_context,
                sample_input_df,
                mock_safe_synth,
            )

        # Should only try once - no retries for validation errors
        assert mock_safe_synth.synthesize.call_count == 1

    def test_result_has_synthetic_markers(
        self,
        asset_context: AssetExecutionContext,
        sample_input_df: pl.DataFrame,
        mock_safe_synth: MagicMock,
    ) -> None:
        """Successful result should have proper synthetic markers."""
        mock_safe_synth.synthesize.return_value = (
            [
                {
                    "reference": "original-001",  # This will be replaced
                    "summary": "Synthetic summary content",
                    "monetary_stance": 4,
                }
            ],
            {"mia_score": 0.95, "privacy_passed": True},
        )

        result_df, evaluation = synthetic_summaries(
            asset_context,
            sample_input_df,
            mock_safe_synth,
        )

        # Should have synthetic reference
        assert result_df["reference"][0].startswith("SYNTH-")

        # Should have is_synthetic flag
        assert "is_synthetic" in result_df.columns
        assert result_df["is_synthetic"][0] is True
