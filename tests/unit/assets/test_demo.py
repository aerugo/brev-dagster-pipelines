"""Unit tests for demo pipeline assets.

Tests the demo asset functions for data generation, cleaning, enrichment, and storage.
All external service calls are mocked per INV-P010.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pandas as pd
import pytest

from brev_pipelines.assets.demo import (
    cleaned_data,
    data_summary,
    nim_enriched_data,
    raw_sample_data,
)

if TYPE_CHECKING:
    from dagster import AssetExecutionContext


class TestRawSampleData:
    """Tests for raw_sample_data asset."""

    def test_generates_100_records(
        self,
        asset_context: AssetExecutionContext,
    ) -> None:
        """Test raw_sample_data generates exactly 100 records."""
        df = raw_sample_data(asset_context)

        assert len(df) == 100

    def test_has_required_columns(
        self,
        asset_context: AssetExecutionContext,
    ) -> None:
        """Test raw_sample_data has all required columns."""
        df = raw_sample_data(asset_context)

        expected_cols = ["id", "name", "age", "region", "category", "spend", "active"]
        assert list(df.columns) == expected_cols

    def test_id_format(
        self,
        asset_context: AssetExecutionContext,
    ) -> None:
        """Test raw_sample_data generates IDs in correct format."""
        df = raw_sample_data(asset_context)

        # All IDs should match CUST-XXXX format
        assert all(df["id"].str.match(r"CUST-\d{4}"))
        assert df["id"].iloc[0] == "CUST-0000"
        assert df["id"].iloc[99] == "CUST-0099"

    def test_region_values(
        self,
        asset_context: AssetExecutionContext,
    ) -> None:
        """Test raw_sample_data uses valid regions."""
        df = raw_sample_data(asset_context)

        valid_regions = {"North", "South", "East", "West", "Central"}
        actual_regions = set(df["region"].unique())
        assert actual_regions.issubset(valid_regions)

    def test_category_values(
        self,
        asset_context: AssetExecutionContext,
    ) -> None:
        """Test raw_sample_data uses valid categories."""
        df = raw_sample_data(asset_context)

        valid_categories = {"Premium", "Standard", "Basic"}
        actual_categories = set(df["category"].unique())
        assert actual_categories.issubset(valid_categories)

    def test_age_range(
        self,
        asset_context: AssetExecutionContext,
    ) -> None:
        """Test raw_sample_data generates ages in valid range."""
        df = raw_sample_data(asset_context)

        assert df["age"].min() >= 18
        assert df["age"].max() <= 75

    def test_spend_range(
        self,
        asset_context: AssetExecutionContext,
    ) -> None:
        """Test raw_sample_data generates spend in valid range."""
        df = raw_sample_data(asset_context)

        assert df["spend"].min() >= 100
        assert df["spend"].max() <= 10000

    def test_active_is_boolean(
        self,
        asset_context: AssetExecutionContext,
    ) -> None:
        """Test raw_sample_data active column is boolean."""
        df = raw_sample_data(asset_context)

        assert df["active"].dtype == bool
        # Should have both True and False values
        assert df["active"].any()
        assert not df["active"].all()

    def test_deterministic_with_seed(
        self,
        asset_context: AssetExecutionContext,
    ) -> None:
        """Test raw_sample_data is deterministic due to seed=42."""
        df1 = raw_sample_data(asset_context)
        df2 = raw_sample_data(asset_context)

        pd.testing.assert_frame_equal(df1, df2)


class TestCleanedData:
    """Tests for cleaned_data asset."""

    @pytest.fixture
    def sample_raw_data(self) -> pd.DataFrame:
        """Create sample raw data for testing."""
        return pd.DataFrame(
            {
                "id": ["CUST-0001", "CUST-0002", "CUST-0003"],
                "name": ["Customer 1", "Customer 2", "Customer 3"],
                "age": [25, 45, 65],
                "region": ["north", "SOUTH", "West"],  # Mixed case
                "category": ["Premium", "Standard", "Basic"],
                "spend": [500.0, 5000.0, 15000.0],  # One extreme value
                "active": [True, True, False],
            }
        )

    def test_normalizes_region_names(
        self,
        asset_context: AssetExecutionContext,
        sample_raw_data: pd.DataFrame,
    ) -> None:
        """Test cleaned_data normalizes region names to title case."""
        df = cleaned_data(asset_context, sample_raw_data)

        assert df["region"].iloc[0] == "North"
        assert df["region"].iloc[1] == "South"
        assert df["region"].iloc[2] == "West"

    def test_caps_extreme_spend(
        self,
        asset_context: AssetExecutionContext,
        sample_raw_data: pd.DataFrame,
    ) -> None:
        """Test cleaned_data caps spend at 9000."""
        df = cleaned_data(asset_context, sample_raw_data)

        assert df["spend"].max() == 9000
        assert df["spend"].iloc[2] == 9000  # Was 15000, now capped

    def test_adds_tier_column(
        self,
        asset_context: AssetExecutionContext,
        sample_raw_data: pd.DataFrame,
    ) -> None:
        """Test cleaned_data adds tier classification column."""
        df = cleaned_data(asset_context, sample_raw_data)

        assert "tier" in df.columns

    def test_tier_classification(
        self,
        asset_context: AssetExecutionContext,
        sample_raw_data: pd.DataFrame,
    ) -> None:
        """Test cleaned_data assigns correct tiers based on spend."""
        df = cleaned_data(asset_context, sample_raw_data)

        # 500 -> Low Value
        assert df["tier"].iloc[0] == "Low Value"
        # 5000 -> Medium Value (1000-5000]
        assert df["tier"].iloc[1] == "Medium Value"
        # 9000 (capped from 15000) -> High Value (5000+)
        assert df["tier"].iloc[2] == "High Value"

    def test_preserves_original_columns(
        self,
        asset_context: AssetExecutionContext,
        sample_raw_data: pd.DataFrame,
    ) -> None:
        """Test cleaned_data preserves all original columns."""
        df = cleaned_data(asset_context, sample_raw_data)

        original_cols = list(sample_raw_data.columns)
        for col in original_cols:
            assert col in df.columns

    def test_preserves_row_count(
        self,
        asset_context: AssetExecutionContext,
        sample_raw_data: pd.DataFrame,
    ) -> None:
        """Test cleaned_data preserves row count."""
        df = cleaned_data(asset_context, sample_raw_data)

        assert len(df) == len(sample_raw_data)


class TestNimEnrichedData:
    """Tests for nim_enriched_data asset."""

    @pytest.fixture
    def sample_cleaned_data(self) -> pd.DataFrame:
        """Create sample cleaned data for testing."""
        return pd.DataFrame(
            {
                "id": [f"CUST-{i:04d}" for i in range(15)],
                "name": [f"Customer {i}" for i in range(15)],
                "age": [25 + i * 3 for i in range(15)],
                "region": ["North"] * 5 + ["South"] * 5 + ["East"] * 5,
                "category": ["Premium"] * 5 + ["Standard"] * 5 + ["Basic"] * 5,
                "spend": [1000 + i * 100 for i in range(15)],
                "active": [True] * 15,
                "tier": ["Low Value"] * 5 + ["Medium Value"] * 5 + ["High Value"] * 5,
            }
        )

    def test_enriches_sample_of_10(
        self,
        asset_context: AssetExecutionContext,
        sample_cleaned_data: pd.DataFrame,
        mock_nim_resource: MagicMock,
    ) -> None:
        """Test nim_enriched_data enriches exactly 10 records."""
        mock_nim_resource.generate.return_value = "A tech-savvy customer with premium needs."

        df = nim_enriched_data(asset_context, sample_cleaned_data, mock_nim_resource)

        # Count enriched records (not "Not enriched")
        enriched_count = sum(1 for p in df["ai_profile"] if "Not enriched" not in p)
        assert enriched_count == 10

    def test_adds_ai_profile_column(
        self,
        asset_context: AssetExecutionContext,
        sample_cleaned_data: pd.DataFrame,
        mock_nim_resource: MagicMock,
    ) -> None:
        """Test nim_enriched_data adds ai_profile column."""
        mock_nim_resource.generate.return_value = "Profile text"

        df = nim_enriched_data(asset_context, sample_cleaned_data, mock_nim_resource)

        assert "ai_profile" in df.columns

    def test_non_enriched_message(
        self,
        asset_context: AssetExecutionContext,
        sample_cleaned_data: pd.DataFrame,
        mock_nim_resource: MagicMock,
    ) -> None:
        """Test nim_enriched_data marks non-enriched records."""
        mock_nim_resource.generate.return_value = "Profile text"

        df = nim_enriched_data(asset_context, sample_cleaned_data, mock_nim_resource)

        # 15 records - 10 enriched = 5 not enriched
        not_enriched = df[df["ai_profile"] == "Not enriched (sample limit)"]
        assert len(not_enriched) == 5

    def test_calls_nim_generate(
        self,
        asset_context: AssetExecutionContext,
        sample_cleaned_data: pd.DataFrame,
        mock_nim_resource: MagicMock,
    ) -> None:
        """Test nim_enriched_data calls NIM for each sample."""
        mock_nim_resource.generate.return_value = "Profile text"

        nim_enriched_data(asset_context, sample_cleaned_data, mock_nim_resource)

        assert mock_nim_resource.generate.call_count == 10

    def test_handles_nim_error_response(
        self,
        asset_context: AssetExecutionContext,
        sample_cleaned_data: pd.DataFrame,
        mock_nim_resource: MagicMock,
    ) -> None:
        """Test nim_enriched_data handles error in response."""
        mock_nim_resource.generate.return_value = "Error: rate limit exceeded"

        df = nim_enriched_data(asset_context, sample_cleaned_data, mock_nim_resource)

        # Should count as not enriched due to "error" in response
        # The function stores the response but doesn't count it as enriched
        assert "ai_profile" in df.columns

    def test_small_dataset(
        self,
        asset_context: AssetExecutionContext,
        mock_nim_resource: MagicMock,
    ) -> None:
        """Test nim_enriched_data with fewer than 10 records."""
        small_data = pd.DataFrame(
            {
                "id": ["CUST-0001", "CUST-0002", "CUST-0003"],
                "age": [25, 35, 45],
                "region": ["North", "South", "East"],
                "category": ["Premium", "Standard", "Basic"],
                "tier": ["Low Value", "Medium Value", "High Value"],
            }
        )
        mock_nim_resource.generate.return_value = "Profile"

        nim_enriched_data(asset_context, small_data, mock_nim_resource)

        # Should enrich all 3 records
        assert mock_nim_resource.generate.call_count == 3

    def test_preserves_columns(
        self,
        asset_context: AssetExecutionContext,
        sample_cleaned_data: pd.DataFrame,
        mock_nim_resource: MagicMock,
    ) -> None:
        """Test nim_enriched_data preserves existing columns."""
        mock_nim_resource.generate.return_value = "Profile"

        df = nim_enriched_data(asset_context, sample_cleaned_data, mock_nim_resource)

        for col in sample_cleaned_data.columns:
            assert col in df.columns


class TestDataSummary:
    """Tests for data_summary asset."""

    @pytest.fixture
    def sample_enriched_data(self) -> pd.DataFrame:
        """Create sample enriched data for testing."""
        return pd.DataFrame(
            {
                "id": ["CUST-0001", "CUST-0002", "CUST-0003", "CUST-0004"],
                "region": ["North", "North", "South", "East"],
                "tier": ["Low Value", "High Value", "Medium Value", "High Value"],
                "spend": [500.0, 8000.0, 3000.0, 7000.0],
                "ai_profile": [
                    "Tech savvy customer",
                    "High value customer",
                    "Not enriched (sample limit)",
                    "Premium customer",
                ],
            }
        )

    def test_returns_summary_dict(
        self,
        asset_context: AssetExecutionContext,
        sample_enriched_data: pd.DataFrame,
        mock_minio_resource: MagicMock,
    ) -> None:
        """Test data_summary returns dictionary."""
        result = data_summary(asset_context, sample_enriched_data, mock_minio_resource)

        assert isinstance(result, dict)

    def test_summary_contains_required_fields(
        self,
        asset_context: AssetExecutionContext,
        sample_enriched_data: pd.DataFrame,
        mock_minio_resource: MagicMock,
    ) -> None:
        """Test data_summary contains all required fields."""
        result = data_summary(asset_context, sample_enriched_data, mock_minio_resource)

        assert "total_records" in result
        assert "region_distribution" in result
        assert "tier_distribution" in result
        assert "average_spend" in result
        assert "enriched_count" in result

    def test_total_records(
        self,
        asset_context: AssetExecutionContext,
        sample_enriched_data: pd.DataFrame,
        mock_minio_resource: MagicMock,
    ) -> None:
        """Test data_summary calculates total records."""
        result = data_summary(asset_context, sample_enriched_data, mock_minio_resource)

        assert result["total_records"] == 4

    def test_region_distribution(
        self,
        asset_context: AssetExecutionContext,
        sample_enriched_data: pd.DataFrame,
        mock_minio_resource: MagicMock,
    ) -> None:
        """Test data_summary calculates region distribution."""
        result = data_summary(asset_context, sample_enriched_data, mock_minio_resource)

        assert result["region_distribution"]["North"] == 2
        assert result["region_distribution"]["South"] == 1
        assert result["region_distribution"]["East"] == 1

    def test_tier_distribution(
        self,
        asset_context: AssetExecutionContext,
        sample_enriched_data: pd.DataFrame,
        mock_minio_resource: MagicMock,
    ) -> None:
        """Test data_summary calculates tier distribution."""
        result = data_summary(asset_context, sample_enriched_data, mock_minio_resource)

        assert result["tier_distribution"]["High Value"] == 2
        assert result["tier_distribution"]["Low Value"] == 1
        assert result["tier_distribution"]["Medium Value"] == 1

    def test_average_spend(
        self,
        asset_context: AssetExecutionContext,
        sample_enriched_data: pd.DataFrame,
        mock_minio_resource: MagicMock,
    ) -> None:
        """Test data_summary calculates average spend."""
        result = data_summary(asset_context, sample_enriched_data, mock_minio_resource)

        expected_avg = round((500 + 8000 + 3000 + 7000) / 4, 2)
        assert result["average_spend"] == expected_avg

    def test_enriched_count(
        self,
        asset_context: AssetExecutionContext,
        sample_enriched_data: pd.DataFrame,
        mock_minio_resource: MagicMock,
    ) -> None:
        """Test data_summary counts enriched records."""
        result = data_summary(asset_context, sample_enriched_data, mock_minio_resource)

        # 3 enriched, 1 not enriched
        assert result["enriched_count"] == 3

    def test_stores_to_minio(
        self,
        asset_context: AssetExecutionContext,
        sample_enriched_data: pd.DataFrame,
        mock_minio_resource: MagicMock,
    ) -> None:
        """Test data_summary stores result to MinIO."""
        mock_client = mock_minio_resource.get_client.return_value

        data_summary(asset_context, sample_enriched_data, mock_minio_resource)

        mock_minio_resource.ensure_bucket.assert_called_once_with("data-products")
        mock_client.put_object.assert_called_once()

        # Check put_object call args
        call_args = mock_client.put_object.call_args
        assert call_args[0][0] == "data-products"  # bucket
        assert call_args[0][1] == "demo/summary.json"  # key

    def test_minio_data_is_valid_json(
        self,
        asset_context: AssetExecutionContext,
        sample_enriched_data: pd.DataFrame,
        mock_minio_resource: MagicMock,
    ) -> None:
        """Test data_summary stores valid JSON to MinIO."""
        mock_client = mock_minio_resource.get_client.return_value
        captured_data = []

        def capture_put(bucket, key, data, length, content_type):
            captured_data.append(data.read())

        mock_client.put_object.side_effect = capture_put

        result = data_summary(asset_context, sample_enriched_data, mock_minio_resource)

        # Verify stored data is valid JSON matching result
        assert len(captured_data) == 1
        stored = json.loads(captured_data[0].decode())
        assert stored["total_records"] == result["total_records"]
