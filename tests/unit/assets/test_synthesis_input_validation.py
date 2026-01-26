"""Tests for synthesis pipeline input validation.

TDD RED phase: These tests define the expected input validation behavior
for enriched_data_for_synthesis before implementation.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, Mock

import polars as pl
import pytest
from dagster import build_asset_context

from brev_pipelines.assets.synthetic_speeches import enriched_data_for_synthesis
from brev_pipelines.config import PipelineConfig

if TYPE_CHECKING:
    from dagster import AssetExecutionContext


class TestEnrichedDataInputValidation:
    """Test input validation for enriched_data_for_synthesis."""

    @pytest.fixture
    def asset_context(self) -> AssetExecutionContext:
        """Create Dagster asset context."""
        return build_asset_context()

    @pytest.fixture
    def mock_lakefs(self) -> MagicMock:
        """Create mock LakeFS resource."""
        return MagicMock()

    def _setup_lakefs_mock(self, mock_lakefs: MagicMock, df: pl.DataFrame) -> None:
        """Configure mock LakeFS to return the given DataFrame."""
        buffer = io.BytesIO()
        df.write_parquet(buffer)
        buffer.seek(0)
        mock_lakefs.get_client.return_value.objects_api.get_object.return_value = buffer.getvalue()

    def test_warns_on_failed_classification_records(
        self, asset_context: AssetExecutionContext, mock_lakefs: MagicMock
    ) -> None:
        """Should log warning when input contains failed classification records."""
        df_with_failures = pl.DataFrame(
            {
                "reference": ["BIS_001", "BIS_002", "BIS_003"],
                "summary": ["summary1", "summary2", "summary3"],
                "monetary_stance": [3, 3, 3],
                "trade_stance": [3, 3, 3],
                "economic_outlook": [3, 3, 3],
                "_llm_status_class": ["success", "failed", "success"],
            }
        )
        self._setup_lakefs_mock(mock_lakefs, df_with_failures)

        # Spy on the context log
        asset_context.log.warning = Mock()

        _result = enriched_data_for_synthesis(asset_context, PipelineConfig(), mock_lakefs)

        # Should have logged warning about failed records
        assert asset_context.log.warning.called
        warning_calls = [str(c) for c in asset_context.log.warning.call_args_list]
        assert any("failed" in call.lower() for call in warning_calls)

    def test_warns_on_failed_summary_records(
        self, asset_context: AssetExecutionContext, mock_lakefs: MagicMock
    ) -> None:
        """Should log warning when input contains failed summary records."""
        df_with_failures = pl.DataFrame(
            {
                "reference": ["BIS_001", "BIS_002", "BIS_003"],
                "summary": ["summary1", "summary2", "summary3"],
                "monetary_stance": [3, 3, 3],
                "trade_stance": [3, 3, 3],
                "economic_outlook": [3, 3, 3],
                "_llm_status_summary": ["success", "success", "failed"],
            }
        )
        self._setup_lakefs_mock(mock_lakefs, df_with_failures)

        asset_context.log.warning = Mock()

        _result = enriched_data_for_synthesis(asset_context, PipelineConfig(), mock_lakefs)

        # Should have logged warning about failed records
        assert asset_context.log.warning.called
        warning_calls = [str(c) for c in asset_context.log.warning.call_args_list]
        assert any("failed" in call.lower() for call in warning_calls)

    def test_reports_failure_statistics(
        self, asset_context: AssetExecutionContext, mock_lakefs: MagicMock
    ) -> None:
        """Should log failure statistics when dead letter columns present."""
        df_with_failures = pl.DataFrame(
            {
                "reference": ["BIS_001", "BIS_002", "BIS_003", "BIS_004"],
                "summary": ["s1", "s2", "s3", "s4"],
                "monetary_stance": [3, 3, 3, 3],
                "trade_stance": [3, 3, 3, 3],
                "economic_outlook": [3, 3, 3, 3],
                "_llm_status_class": ["success", "failed", "success", "failed"],
            }
        )
        self._setup_lakefs_mock(mock_lakefs, df_with_failures)

        asset_context.log.info = Mock()
        asset_context.log.warning = Mock()

        _result = enriched_data_for_synthesis(asset_context, PipelineConfig(), mock_lakefs)

        # Should log statistics about failures - either percentage or count
        all_log_calls = " ".join(str(c) for c in asset_context.log.info.call_args_list)
        # Check for 50% or 2 failed
        assert "50" in all_log_calls or "2" in all_log_calls or "failed" in all_log_calls.lower()

    def test_accepts_clean_data_without_failure_warning(
        self, asset_context: AssetExecutionContext, mock_lakefs: MagicMock
    ) -> None:
        """Should not warn when all records successful."""
        df_clean = pl.DataFrame(
            {
                "reference": ["BIS_001", "BIS_002"],
                "summary": ["s1", "s2"],
                "monetary_stance": [3, 4],
                "trade_stance": [3, 4],
                "economic_outlook": [3, 4],
                "_llm_status_class": ["success", "success"],
                "_llm_status_summary": ["success", "success"],
            }
        )
        self._setup_lakefs_mock(mock_lakefs, df_clean)

        asset_context.log.warning = Mock()

        _result = enriched_data_for_synthesis(asset_context, PipelineConfig(), mock_lakefs)

        # Should NOT have logged warning about failures
        if asset_context.log.warning.called:
            warning_calls = [str(c) for c in asset_context.log.warning.call_args_list]
            # Ensure no warning about "failed" records
            assert not any("failed" in call.lower() for call in warning_calls)

    def test_works_without_dead_letter_columns(
        self, asset_context: AssetExecutionContext, mock_lakefs: MagicMock
    ) -> None:
        """Should work with legacy data missing dead letter columns."""
        df_legacy = pl.DataFrame(
            {
                "reference": ["BIS_001", "BIS_002"],
                "summary": ["s1", "s2"],
                "monetary_stance": [3, 4],
                "trade_stance": [3, 4],
                "economic_outlook": [3, 4],
                # No _llm_* columns
            }
        )
        self._setup_lakefs_mock(mock_lakefs, df_legacy)

        # Should not raise
        result = enriched_data_for_synthesis(asset_context, PipelineConfig(), mock_lakefs)

        assert len(result) == 2

    def test_logs_info_about_missing_dead_letter_columns(
        self, asset_context: AssetExecutionContext, mock_lakefs: MagicMock
    ) -> None:
        """Should log info when dead letter columns are absent."""
        df_legacy = pl.DataFrame(
            {
                "reference": ["BIS_001"],
                "summary": ["s1"],
                "monetary_stance": [3],
                "trade_stance": [3],
                "economic_outlook": [3],
            }
        )
        self._setup_lakefs_mock(mock_lakefs, df_legacy)

        asset_context.log.info = Mock()

        _result = enriched_data_for_synthesis(asset_context, PipelineConfig(), mock_lakefs)

        # Should log about missing columns or legacy data
        info_calls = " ".join(str(c) for c in asset_context.log.info.call_args_list)
        assert "dead letter" in info_calls.lower() or "legacy" in info_calls.lower()

    def test_warns_on_high_failure_rate(
        self, asset_context: AssetExecutionContext, mock_lakefs: MagicMock
    ) -> None:
        """Should warn when failure rate exceeds threshold (>10%)."""
        # Create data with >10% failures
        df_high_failure = pl.DataFrame(
            {
                "reference": [f"BIS_{i:03d}" for i in range(10)],
                "summary": [f"s{i}" for i in range(10)],
                "monetary_stance": [3] * 10,
                "trade_stance": [3] * 10,
                "economic_outlook": [3] * 10,
                "_llm_status_class": [
                    "failed",
                    "failed",
                    "success",
                    "success",
                    "success",
                    "success",
                    "success",
                    "success",
                    "success",
                    "success",
                ],  # 20% failed
            }
        )
        self._setup_lakefs_mock(mock_lakefs, df_high_failure)

        asset_context.log.warning = Mock()

        _result = enriched_data_for_synthesis(asset_context, PipelineConfig(), mock_lakefs)

        # Should warn about high failure rate
        assert asset_context.log.warning.called
        warning_calls = " ".join(str(c) for c in asset_context.log.warning.call_args_list)
        assert (
            "high" in warning_calls.lower()
            or "20" in warning_calls
            or "rate" in warning_calls.lower()
        )
