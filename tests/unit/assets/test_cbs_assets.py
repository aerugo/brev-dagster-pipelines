"""Unit tests for central bank speeches pipeline assets.

Tests the CBS (Central Bank Speeches) asset functions including:
- cleaned_speeches: Data cleaning with null fills and filtering
- enriched_speeches: Data combination from multiple sources
- classification_snapshot: LakeFS persistence

Kaggle-dependent assets (raw_speeches) and LLM-dependent assets
(speech_classification, speech_summaries, speech_embeddings) are tested
for their helper functions and core logic, with external calls mocked.
"""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from brev_pipelines.assets.central_bank_speeches import (
    MONETARY_STANCE_SCALE,
    OUTLOOK_SCALE,
    SPEECHES_SCHEMA,
    TRADE_STANCE_SCALE,
    classification_snapshot,
    cleaned_speeches,
    embeddings_snapshot,
    enriched_speeches,
    speech_classification,
    speech_embeddings,
    speech_summaries,
    speeches_data_product,
    summaries_snapshot,
    weaviate_index,
)
from brev_pipelines.config import PipelineConfig

if TYPE_CHECKING:
    from dagster import AssetExecutionContext


class TestSpeechesSchema:
    """Tests for SPEECHES_SCHEMA constant."""

    def test_schema_has_correct_fields(self) -> None:
        """Test schema contains all required fields."""
        field_names = [prop["name"] for prop in SPEECHES_SCHEMA]

        assert "reference" in field_names
        assert "date" in field_names
        assert "central_bank" in field_names
        assert "speaker" in field_names
        assert "title" in field_names
        assert "text" in field_names
        assert "monetary_stance" in field_names
        assert "trade_stance" in field_names
        assert "tariff_mention" in field_names
        assert "economic_outlook" in field_names
        assert "is_governor" in field_names

    def test_schema_field_types(self) -> None:
        """Test schema fields have correct types."""
        schema_dict = {prop["name"]: prop["type"] for prop in SPEECHES_SCHEMA}

        assert schema_dict["reference"] == "text"
        assert schema_dict["monetary_stance"] == "int"
        assert schema_dict["tariff_mention"] == "boolean"
        assert schema_dict["is_governor"] == "boolean"


class TestClassificationScales:
    """Tests for classification scale mappings."""

    def test_monetary_stance_scale(self) -> None:
        """Test monetary stance scale values."""
        assert MONETARY_STANCE_SCALE["very_dovish"] == 1
        assert MONETARY_STANCE_SCALE["neutral"] == 3
        assert MONETARY_STANCE_SCALE["very_hawkish"] == 5
        assert len(MONETARY_STANCE_SCALE) == 5

    def test_trade_stance_scale(self) -> None:
        """Test trade stance scale values."""
        assert TRADE_STANCE_SCALE["very_protectionist"] == 1
        assert TRADE_STANCE_SCALE["neutral"] == 3
        assert TRADE_STANCE_SCALE["very_globalist"] == 5
        assert len(TRADE_STANCE_SCALE) == 5

    def test_outlook_scale(self) -> None:
        """Test economic outlook scale values."""
        assert OUTLOOK_SCALE["very_negative"] == 1
        assert OUTLOOK_SCALE["neutral"] == 3
        assert OUTLOOK_SCALE["very_positive"] == 5
        assert len(OUTLOOK_SCALE) == 5


class TestCleanedSpeeches:
    """Tests for cleaned_speeches asset."""

    @pytest.fixture
    def raw_speeches_df(self) -> pl.DataFrame:
        """Create sample raw speeches data with various null scenarios."""
        return pl.DataFrame(
            {
                "reference": ["BIS_001", "BIS_002", "BIS_003", "BIS_004"],
                "title": ["Speech 1", None, "Speech 3", "Short"],
                "speaker": ["Powell", "Lagarde", None, "Bailey"],
                "central_bank": ["FED", "ECB", "BOE", "BOE"],
                "text": [
                    "A" * 200,  # Long enough
                    "B" * 150,  # Long enough
                    "C" * 50,  # Too short (< 100)
                    "D" * 200,  # Long enough
                ],
                "is_gov": [True, True, False, None],
            }
        )

    def test_fills_null_strings(
        self,
        asset_context: AssetExecutionContext,
        raw_speeches_df: pl.DataFrame,
    ) -> None:
        """Test cleaned_speeches fills null string values."""
        df = cleaned_speeches(asset_context, raw_speeches_df)

        # Null title should be filled with empty string
        assert df.filter(pl.col("reference") == "BIS_002")["title"][0] == ""

        # BIS_001 speaker should be preserved
        assert df.filter(pl.col("reference") == "BIS_001")["speaker"][0] == "Powell"

    def test_fills_null_booleans(
        self,
        asset_context: AssetExecutionContext,
        raw_speeches_df: pl.DataFrame,
    ) -> None:
        """Test cleaned_speeches fills null boolean values."""
        df = cleaned_speeches(asset_context, raw_speeches_df)

        # Null is_gov should be filled with False
        bis_004 = df.filter(pl.col("reference") == "BIS_004")
        assert bis_004["is_gov"][0] is False

    def test_filters_short_speeches(
        self,
        asset_context: AssetExecutionContext,
        raw_speeches_df: pl.DataFrame,
    ) -> None:
        """Test cleaned_speeches filters speeches < 100 chars."""
        df = cleaned_speeches(asset_context, raw_speeches_df)

        # BIS_003 has only 50 chars, should be filtered out
        assert len(df) == 3
        assert "BIS_003" not in df["reference"].to_list()

    def test_preserves_valid_data(
        self,
        asset_context: AssetExecutionContext,
        raw_speeches_df: pl.DataFrame,
    ) -> None:
        """Test cleaned_speeches preserves valid, non-null data."""
        df = cleaned_speeches(asset_context, raw_speeches_df)

        bis_001 = df.filter(pl.col("reference") == "BIS_001")
        assert bis_001["title"][0] == "Speech 1"
        assert bis_001["speaker"][0] == "Powell"
        assert bis_001["central_bank"][0] == "FED"


class TestEnrichedSpeeches:
    """Tests for enriched_speeches asset."""

    @pytest.fixture
    def sample_embeddings_tuple(self) -> tuple[pl.DataFrame, list[list[float]]]:
        """Create sample embeddings tuple."""
        df = pl.DataFrame(
            {
                "reference": ["BIS_001", "BIS_002", "BIS_003"],
                "title": ["Speech 1", "Speech 2", "Speech 3"],
                "text": ["Text 1", "Text 2", "Text 3"],
                "is_gov": [True, True, False],
            }
        )
        embeddings = [[0.1] * 1024, [0.2] * 1024, [0.3] * 1024]
        return df, embeddings

    @pytest.fixture
    def sample_classification_df(self) -> pl.DataFrame:
        """Create sample classification results."""
        return pl.DataFrame(
            {
                "reference": ["BIS_001", "BIS_002", "BIS_003"],
                "monetary_stance": [3, 4, 2],
                "trade_stance": [3, 5, 1],
                "tariff_mention": [0, 1, 0],
                "economic_outlook": [4, 3, 2],
            }
        )

    @pytest.fixture
    def sample_summaries_df(self) -> pl.DataFrame:
        """Create sample summaries results."""
        return pl.DataFrame(
            {
                "reference": ["BIS_001", "BIS_002", "BIS_003"],
                "summary": [
                    "Summary of speech 1",
                    "Summary of speech 2",
                    "Summary of speech 3",
                ],
            }
        )

    def test_combines_all_sources(
        self,
        asset_context: AssetExecutionContext,
        sample_embeddings_tuple: tuple[pl.DataFrame, list[list[float]]],
        sample_classification_df: pl.DataFrame,
        sample_summaries_df: pl.DataFrame,
    ) -> None:
        """Test enriched_speeches combines all data sources."""
        df = enriched_speeches(
            asset_context,
            sample_embeddings_tuple,
            sample_summaries_df,
            sample_classification_df,
        )

        # Check all sources are combined
        assert "reference" in df.columns
        assert "title" in df.columns
        assert "monetary_stance" in df.columns
        assert "summary" in df.columns
        assert "processed_at" in df.columns

    def test_joins_classifications(
        self,
        asset_context: AssetExecutionContext,
        sample_embeddings_tuple: tuple[pl.DataFrame, list[list[float]]],
        sample_classification_df: pl.DataFrame,
        sample_summaries_df: pl.DataFrame,
    ) -> None:
        """Test enriched_speeches joins classification correctly."""
        df = enriched_speeches(
            asset_context,
            sample_embeddings_tuple,
            sample_summaries_df,
            sample_classification_df,
        )

        bis_002 = df.filter(pl.col("reference") == "BIS_002")
        assert bis_002["monetary_stance"][0] == 4
        assert bis_002["trade_stance"][0] == 5
        assert bis_002["tariff_mention"][0] == 1

    def test_joins_summaries(
        self,
        asset_context: AssetExecutionContext,
        sample_embeddings_tuple: tuple[pl.DataFrame, list[list[float]]],
        sample_classification_df: pl.DataFrame,
        sample_summaries_df: pl.DataFrame,
    ) -> None:
        """Test enriched_speeches joins summaries correctly."""
        df = enriched_speeches(
            asset_context,
            sample_embeddings_tuple,
            sample_summaries_df,
            sample_classification_df,
        )

        bis_001 = df.filter(pl.col("reference") == "BIS_001")
        assert bis_001["summary"][0] == "Summary of speech 1"

    def test_adds_processed_at_timestamp(
        self,
        asset_context: AssetExecutionContext,
        sample_embeddings_tuple: tuple[pl.DataFrame, list[list[float]]],
        sample_classification_df: pl.DataFrame,
        sample_summaries_df: pl.DataFrame,
    ) -> None:
        """Test enriched_speeches adds processed_at timestamp."""
        df = enriched_speeches(
            asset_context,
            sample_embeddings_tuple,
            sample_summaries_df,
            sample_classification_df,
        )

        assert df["processed_at"][0] is not None
        # Should be ISO format
        assert "T" in df["processed_at"][0]


class TestSpeechesDataProduct:
    """Tests for speeches_data_product asset."""

    @pytest.fixture
    def sample_enriched_df(self) -> pl.DataFrame:
        """Create sample enriched DataFrame."""
        return pl.DataFrame(
            {
                "reference": ["BIS_001", "BIS_002", "BIS_003"],
                "title": ["Speech 1", "Speech 2", "Speech 3"],
                "text": ["Text 1", "Text 2", "Text 3"],
                "monetary_stance": [3, 4, 2],
                "trade_stance": [3, 5, 1],
                "tariff_mention": [0, 1, 0],
                "economic_outlook": [4, 3, 2],
            }
        )

    @pytest.fixture
    def pipeline_config(self) -> PipelineConfig:
        """Create pipeline config."""
        return PipelineConfig(is_trial=False, sample_size=0)

    @pytest.fixture
    def mock_lakefs_with_client(self) -> MagicMock:
        """Create mock LakeFS resource with properly structured client."""
        resource = MagicMock()
        client = MagicMock()

        # Set up objects_api
        client.objects_api = MagicMock()
        client.objects_api.upload_object = MagicMock()

        # Set up commits_api
        client.commits_api = MagicMock()
        mock_commit = MagicMock()
        mock_commit.id = "commit123"
        client.commits_api.commit = MagicMock(return_value=mock_commit)

        resource.get_client = MagicMock(return_value=client)
        return resource

    def test_uploads_to_lakefs(
        self,
        asset_context: AssetExecutionContext,
        sample_enriched_df: pl.DataFrame,
        pipeline_config: PipelineConfig,
        mock_lakefs_with_client: MagicMock,
    ) -> None:
        """Test speeches_data_product uploads to LakeFS."""
        speeches_data_product(
            asset_context,
            pipeline_config,
            sample_enriched_df,
            mock_lakefs_with_client,
        )

        mock_client = mock_lakefs_with_client.get_client.return_value
        mock_client.objects_api.upload_object.assert_called_once()
        call_kwargs = mock_client.objects_api.upload_object.call_args.kwargs
        assert call_kwargs["repository"] == "data"
        assert call_kwargs["branch"] == "main"
        assert "speeches.parquet" in call_kwargs["path"]

    def test_returns_metadata(
        self,
        asset_context: AssetExecutionContext,
        sample_enriched_df: pl.DataFrame,
        pipeline_config: PipelineConfig,
        mock_lakefs_with_client: MagicMock,
    ) -> None:
        """Test speeches_data_product returns correct metadata."""
        result = speeches_data_product(
            asset_context,
            pipeline_config,
            sample_enriched_df,
            mock_lakefs_with_client,
        )

        assert result["num_records"] == 3
        assert result["tariff_mentions"] == 1  # BIS_002 has tariff_mention=1
        assert result["commit_id"] == "commit123"
        assert "lakefs://" in result["path"]

    def test_uses_trial_path(
        self,
        asset_context: AssetExecutionContext,
        sample_enriched_df: pl.DataFrame,
        mock_lakefs_with_client: MagicMock,
    ) -> None:
        """Test speeches_data_product uses trial path when is_trial=True."""
        config = PipelineConfig(is_trial=True)

        speeches_data_product(
            asset_context,
            config,
            sample_enriched_df,
            mock_lakefs_with_client,
        )

        mock_client = mock_lakefs_with_client.get_client.return_value
        call_kwargs = mock_client.objects_api.upload_object.call_args.kwargs
        assert "trial" in call_kwargs["path"]

    def test_handles_no_changes_error(
        self,
        asset_context: AssetExecutionContext,
        sample_enriched_df: pl.DataFrame,
        pipeline_config: PipelineConfig,
        mock_lakefs_with_client: MagicMock,
    ) -> None:
        """Test speeches_data_product handles 'no changes' error gracefully."""
        mock_client = mock_lakefs_with_client.get_client.return_value
        mock_client.commits_api.commit.side_effect = Exception("no changes to commit")

        result = speeches_data_product(
            asset_context,
            pipeline_config,
            sample_enriched_df,
            mock_lakefs_with_client,
        )

        # Should still return result with None commit_id
        assert result["commit_id"] is None


class TestWeaviateIndex:
    """Tests for weaviate_index asset."""

    @pytest.fixture
    def sample_embeddings_tuple(self) -> tuple[pl.DataFrame, list[list[float]]]:
        """Create sample embeddings tuple."""
        df = pl.DataFrame(
            {
                "reference": ["BIS_001", "BIS_002"],
                "date": ["2024-01-15", "2024-01-20"],
                "central_bank": ["FED", "ECB"],
                "speaker": ["Powell", "Lagarde"],
                "title": ["Speech 1", "Speech 2"],
                "text": ["Full text 1" * 100, "Full text 2" * 100],
                "is_governor": [1, 1],
            }
        )
        embeddings = [[0.1] * 1024, [0.2] * 1024]
        return df, embeddings

    @pytest.fixture
    def sample_classification_df(self) -> pl.DataFrame:
        """Create sample classification results."""
        return pl.DataFrame(
            {
                "reference": ["BIS_001", "BIS_002"],
                "monetary_stance": [4, 3],
                "trade_stance": [3, 4],
                "tariff_mention": [0, 1],
                "economic_outlook": [4, 3],
            }
        )

    @pytest.fixture
    def pipeline_config(self) -> PipelineConfig:
        """Create pipeline config."""
        return PipelineConfig(is_trial=False)

    @pytest.fixture
    def mock_weaviate(self) -> MagicMock:
        """Create mock Weaviate resource."""
        resource = MagicMock()
        resource.ensure_collection = MagicMock()
        resource.insert_objects = MagicMock(return_value=2)
        return resource

    def test_ensures_collection_exists(
        self,
        asset_context: AssetExecutionContext,
        sample_embeddings_tuple: tuple[pl.DataFrame, list[list[float]]],
        sample_classification_df: pl.DataFrame,
        pipeline_config: PipelineConfig,
        mock_weaviate: MagicMock,
    ) -> None:
        """Test weaviate_index ensures collection exists."""
        weaviate_index(
            asset_context,
            pipeline_config,
            sample_embeddings_tuple,
            sample_classification_df,
            mock_weaviate,
        )

        mock_weaviate.ensure_collection.assert_called_once()
        call_kwargs = mock_weaviate.ensure_collection.call_args.kwargs
        assert call_kwargs["name"] == "CentralBankSpeeches"
        assert call_kwargs["vector_dimensions"] == 1024

    def test_inserts_objects(
        self,
        asset_context: AssetExecutionContext,
        sample_embeddings_tuple: tuple[pl.DataFrame, list[list[float]]],
        sample_classification_df: pl.DataFrame,
        pipeline_config: PipelineConfig,
        mock_weaviate: MagicMock,
    ) -> None:
        """Test weaviate_index inserts objects correctly."""
        result = weaviate_index(
            asset_context,
            pipeline_config,
            sample_embeddings_tuple,
            sample_classification_df,
            mock_weaviate,
        )

        mock_weaviate.insert_objects.assert_called_once()
        assert result["object_count"] == 2

    def test_uses_trial_collection(
        self,
        asset_context: AssetExecutionContext,
        sample_embeddings_tuple: tuple[pl.DataFrame, list[list[float]]],
        sample_classification_df: pl.DataFrame,
        mock_weaviate: MagicMock,
    ) -> None:
        """Test weaviate_index uses trial collection when is_trial=True."""
        config = PipelineConfig(is_trial=True)

        result = weaviate_index(
            asset_context,
            config,
            sample_embeddings_tuple,
            sample_classification_df,
            mock_weaviate,
        )

        call_kwargs = mock_weaviate.ensure_collection.call_args.kwargs
        assert "Trial" in call_kwargs["name"]
        assert result["collection"] == "CentralBankSpeechesTrial"

    def test_returns_correct_metadata(
        self,
        asset_context: AssetExecutionContext,
        sample_embeddings_tuple: tuple[pl.DataFrame, list[list[float]]],
        sample_classification_df: pl.DataFrame,
        pipeline_config: PipelineConfig,
        mock_weaviate: MagicMock,
    ) -> None:
        """Test weaviate_index returns correct metadata."""
        result = weaviate_index(
            asset_context,
            pipeline_config,
            sample_embeddings_tuple,
            sample_classification_df,
            mock_weaviate,
        )

        assert result["collection"] == "CentralBankSpeeches"
        assert result["object_count"] == 2
        assert result["vector_dimensions"] == 1024


class TestClassificationSnapshot:
    """Tests for classification_snapshot asset."""

    @pytest.fixture
    def sample_classification_df(self) -> pl.DataFrame:
        """Create sample classification results."""
        return pl.DataFrame(
            {
                "reference": ["BIS_001", "BIS_002", "BIS_003"],
                "title": ["Speech 1", "Speech 2", "Speech 3"],
                "monetary_stance": [3, 4, 2],
                "trade_stance": [3, 5, 1],
                "tariff_mention": [0, 1, 1],
                "economic_outlook": [4, 3, 2],
            }
        )

    @pytest.fixture
    def pipeline_config(self) -> PipelineConfig:
        """Create pipeline config."""
        return PipelineConfig(is_trial=False)

    @pytest.fixture
    def mock_lakefs_with_client(self) -> MagicMock:
        """Create mock LakeFS resource with properly structured client."""
        resource = MagicMock()
        client = MagicMock()
        client.objects_api = MagicMock()
        client.commits_api = MagicMock()
        mock_commit = MagicMock()
        mock_commit.id = "snap123"
        client.commits_api.commit = MagicMock(return_value=mock_commit)
        resource.get_client = MagicMock(return_value=client)
        return resource

    def test_uploads_classification_columns_only(
        self,
        asset_context: AssetExecutionContext,
        sample_classification_df: pl.DataFrame,
        pipeline_config: PipelineConfig,
        mock_lakefs_with_client: MagicMock,
    ) -> None:
        """Test classification_snapshot only uploads relevant columns."""
        mock_client = mock_lakefs_with_client.get_client.return_value

        # Capture uploaded content
        captured_content: list[bytes] = []

        def capture_upload(**kwargs: Any) -> None:
            captured_content.append(kwargs["content"])

        mock_client.objects_api.upload_object.side_effect = capture_upload

        classification_snapshot(
            asset_context,
            pipeline_config,
            sample_classification_df,
            mock_lakefs_with_client,
        )

        # Parse uploaded parquet and check columns
        uploaded_df = pl.read_parquet(io.BytesIO(captured_content[0]))
        assert "reference" in uploaded_df.columns
        assert "monetary_stance" in uploaded_df.columns
        assert "title" not in uploaded_df.columns  # Should be excluded

    def test_returns_correct_metadata(
        self,
        asset_context: AssetExecutionContext,
        sample_classification_df: pl.DataFrame,
        pipeline_config: PipelineConfig,
        mock_lakefs_with_client: MagicMock,
    ) -> None:
        """Test classification_snapshot returns correct metadata."""
        result = classification_snapshot(
            asset_context,
            pipeline_config,
            sample_classification_df,
            mock_lakefs_with_client,
        )

        assert result["num_records"] == 3
        assert result["commit_id"] == "snap123"
        assert "classifications.parquet" in result["path"]


class TestSummariesSnapshot:
    """Tests for summaries_snapshot asset."""

    @pytest.fixture
    def sample_summaries_df(self) -> pl.DataFrame:
        """Create sample summaries results."""
        return pl.DataFrame(
            {
                "reference": ["BIS_001", "BIS_002", "BIS_003"],
                "summary": [
                    "Summary with specific metrics: 2.5% inflation",
                    "Summary discussing trade policy and tariffs",
                    None,  # One null summary
                ],
            }
        )

    @pytest.fixture
    def pipeline_config(self) -> PipelineConfig:
        """Create pipeline config."""
        return PipelineConfig(is_trial=False)

    @pytest.fixture
    def mock_lakefs_with_client(self) -> MagicMock:
        """Create mock LakeFS resource with properly structured client."""
        resource = MagicMock()
        client = MagicMock()
        client.objects_api = MagicMock()
        client.commits_api = MagicMock()
        mock_commit = MagicMock()
        mock_commit.id = "sum123"
        client.commits_api.commit = MagicMock(return_value=mock_commit)
        resource.get_client = MagicMock(return_value=client)
        return resource

    def test_counts_non_null_summaries(
        self,
        asset_context: AssetExecutionContext,
        sample_summaries_df: pl.DataFrame,
        pipeline_config: PipelineConfig,
        mock_lakefs_with_client: MagicMock,
    ) -> None:
        """Test summaries_snapshot counts non-null summaries."""
        result = summaries_snapshot(
            asset_context,
            pipeline_config,
            sample_summaries_df,
            mock_lakefs_with_client,
        )

        assert result["num_records"] == 3
        assert result["summaries_with_content"] == 2  # One is null


class TestEmbeddingsSnapshot:
    """Tests for embeddings_snapshot asset."""

    @pytest.fixture
    def sample_embeddings_tuple(self) -> tuple[pl.DataFrame, list[list[float]]]:
        """Create sample embeddings tuple."""
        df = pl.DataFrame(
            {
                "reference": ["BIS_001", "BIS_002"],
            }
        )
        embeddings = [[0.1] * 1024, [0.2] * 1024]
        return df, embeddings

    @pytest.fixture
    def pipeline_config(self) -> PipelineConfig:
        """Create pipeline config."""
        return PipelineConfig(is_trial=False)

    @pytest.fixture
    def mock_lakefs_with_client(self) -> MagicMock:
        """Create mock LakeFS resource with properly structured client."""
        resource = MagicMock()
        client = MagicMock()
        client.objects_api = MagicMock()
        client.commits_api = MagicMock()
        mock_commit = MagicMock()
        mock_commit.id = "emb123"
        client.commits_api.commit = MagicMock(return_value=mock_commit)
        resource.get_client = MagicMock(return_value=client)
        return resource

    def test_stores_embeddings_with_references(
        self,
        asset_context: AssetExecutionContext,
        sample_embeddings_tuple: tuple[pl.DataFrame, list[list[float]]],
        pipeline_config: PipelineConfig,
        mock_lakefs_with_client: MagicMock,
    ) -> None:
        """Test embeddings_snapshot stores embeddings with references."""
        result = embeddings_snapshot(
            asset_context,
            pipeline_config,
            sample_embeddings_tuple,
            mock_lakefs_with_client,
        )

        assert result["num_records"] == 2
        assert result["dimensions"] == 1024
        assert "embeddings.parquet" in result["path"]

    def test_reports_size_mb(
        self,
        asset_context: AssetExecutionContext,
        sample_embeddings_tuple: tuple[pl.DataFrame, list[list[float]]],
        pipeline_config: PipelineConfig,
        mock_lakefs_with_client: MagicMock,
    ) -> None:
        """Test embeddings_snapshot reports size in MB."""
        result = embeddings_snapshot(
            asset_context,
            pipeline_config,
            sample_embeddings_tuple,
            mock_lakefs_with_client,
        )

        assert "size_mb" in result
        assert result["size_mb"] > 0


class TestSpeechEmbeddings:
    """Tests for speech_embeddings asset with checkpointing."""

    @pytest.fixture
    def sample_cleaned_df(self) -> pl.DataFrame:
        """Create sample cleaned speeches."""
        return pl.DataFrame(
            {
                "reference": ["BIS_001", "BIS_002"],
                "title": ["Speech 1", "Speech 2"],
                "text": ["Full text of speech 1" * 50, "Full text of speech 2" * 50],
            }
        )

    @pytest.fixture
    def mock_minio_for_checkpoint(self) -> MagicMock:
        """Create mock MinIO resource for checkpointing."""
        resource = MagicMock()
        resource.ensure_bucket = MagicMock()
        client = MagicMock()
        # Make get_object raise exception to simulate no checkpoint
        client.get_object.side_effect = Exception("Not found")
        client.put_object = MagicMock()
        client.remove_object = MagicMock()
        resource.get_client = MagicMock(return_value=client)
        return resource

    @pytest.fixture
    def mock_nim_embedding(self) -> MagicMock:
        """Create mock NIM embedding resource."""
        resource = MagicMock()

        def mock_embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
            return [[0.1] * 1024 for _ in texts]

        resource.embed_texts = MagicMock(side_effect=mock_embed_texts)
        resource.embed_text = MagicMock(return_value=[0.1] * 1024)
        return resource

    def test_generates_embeddings_for_all_rows(
        self,
        asset_context: AssetExecutionContext,
        sample_cleaned_df: pl.DataFrame,
        mock_nim_embedding: MagicMock,
        mock_minio_for_checkpoint: MagicMock,
    ) -> None:
        """Test speech_embeddings generates embeddings for all rows."""

        # Create expected results DataFrame with embedding JSON strings
        results_df = pl.DataFrame(
            {
                "reference": ["BIS_001", "BIS_002"],
                "embedding": [
                    json.dumps([0.1] * 1024),
                    json.dumps([0.2] * 1024),
                ],
            }
        )

        with (
            patch(
                "brev_pipelines.assets.central_bank_speeches.LLMCheckpointManager",
            ) as mock_checkpoint_cls,
            patch(
                "brev_pipelines.assets.central_bank_speeches.process_with_checkpoint",
                return_value=results_df,
            ),
        ):
            mock_checkpoint_cls.return_value.cleanup = MagicMock()
            df, embeddings = speech_embeddings(
                asset_context,
                sample_cleaned_df,
                mock_nim_embedding,
                mock_minio_for_checkpoint,
            )

        assert len(embeddings) == 2
        assert len(embeddings[0]) == 1024
        assert len(df) == 2


class TestSpeechClassification:
    """Tests for speech_classification asset with checkpointing."""

    @pytest.fixture
    def sample_cleaned_df(self) -> pl.DataFrame:
        """Create sample cleaned speeches."""
        return pl.DataFrame(
            {
                "reference": ["BIS_001", "BIS_002"],
                "title": ["Monetary Policy", "Trade Relations"],
                "speaker": ["Powell", "Lagarde"],
                "central_bank": ["FED", "ECB"],
                "text": ["Inflation is our primary concern" * 50, "Global trade benefits all" * 50],
            }
        )

    @pytest.fixture
    def mock_minio_for_checkpoint(self) -> MagicMock:
        """Create mock MinIO resource for checkpointing."""
        resource = MagicMock()
        resource.ensure_bucket = MagicMock()
        client = MagicMock()
        client.get_object.side_effect = Exception("Not found")
        client.put_object = MagicMock()
        client.remove_object = MagicMock()
        resource.get_client = MagicMock(return_value=client)
        return resource

    @pytest.fixture
    def mock_nim_reasoning(self) -> MagicMock:
        """Create mock NIM reasoning resource."""
        resource = MagicMock()
        resource.generate = MagicMock(
            return_value='{"monetary_stance": "neutral", "trade_stance": "neutral", "tariff_mention": 0, "economic_outlook": "neutral"}'
        )
        return resource

    def test_classifies_speeches(
        self,
        asset_context: AssetExecutionContext,
        sample_cleaned_df: pl.DataFrame,
        mock_nim_reasoning: MagicMock,
        mock_minio_for_checkpoint: MagicMock,
    ) -> None:
        """Test speech_classification classifies all speeches."""
        # Create expected results DataFrame
        results_df = pl.DataFrame(
            {
                "reference": ["BIS_001", "BIS_002"],
                "monetary_stance": [3, 3],
                "trade_stance": [3, 3],
                "tariff_mention": [0, 0],
                "economic_outlook": [3, 3],
            }
        )

        with (
            patch(
                "brev_pipelines.assets.central_bank_speeches.LLMCheckpointManager",
            ) as mock_checkpoint_cls,
            patch(
                "brev_pipelines.assets.central_bank_speeches.process_with_checkpoint",
                return_value=results_df,
            ),
        ):
            mock_checkpoint_cls.return_value.cleanup = MagicMock()
            df = speech_classification(
                asset_context,
                sample_cleaned_df,
                mock_nim_reasoning,
                mock_minio_for_checkpoint,
            )

        assert "monetary_stance" in df.columns
        assert "trade_stance" in df.columns
        assert "tariff_mention" in df.columns
        assert "economic_outlook" in df.columns
        assert len(df) == 2


class TestSpeechSummaries:
    """Tests for speech_summaries asset with checkpointing."""

    @pytest.fixture
    def sample_cleaned_df(self) -> pl.DataFrame:
        """Create sample cleaned speeches."""
        return pl.DataFrame(
            {
                "reference": ["BIS_001", "BIS_002"],
                "title": ["Economic Outlook 2024", "Trade Policy Update"],
                "speaker": ["Powell", "Lagarde"],
                "central_bank": ["FED", "ECB"],
                "text": ["Detailed economic analysis" * 100, "Trade policy discussion" * 100],
            }
        )

    @pytest.fixture
    def mock_minio_for_checkpoint(self) -> MagicMock:
        """Create mock MinIO resource for checkpointing."""
        resource = MagicMock()
        resource.ensure_bucket = MagicMock()
        client = MagicMock()
        client.get_object.side_effect = Exception("Not found")
        client.put_object = MagicMock()
        client.remove_object = MagicMock()
        resource.get_client = MagicMock(return_value=client)
        return resource

    @pytest.fixture
    def mock_nim_reasoning(self) -> MagicMock:
        """Create mock NIM reasoning resource."""
        resource = MagicMock()
        resource.generate = MagicMock(
            return_value="* Inflation: 2.5% target\n* GDP growth: 2.1%\n* Key risk: supply chain"
        )
        return resource

    def test_generates_summaries(
        self,
        asset_context: AssetExecutionContext,
        sample_cleaned_df: pl.DataFrame,
        mock_nim_reasoning: MagicMock,
        mock_minio_for_checkpoint: MagicMock,
    ) -> None:
        """Test speech_summaries generates summaries for all speeches."""
        # Create expected results DataFrame
        results_df = pl.DataFrame(
            {
                "reference": ["BIS_001", "BIS_002"],
                "summary": [
                    "* Inflation: 2.5% target\n* GDP growth: 2.1%",
                    "* Trade policy: balanced approach\n* Key risk: tariffs",
                ],
            }
        )

        with (
            patch(
                "brev_pipelines.assets.central_bank_speeches.LLMCheckpointManager",
            ) as mock_checkpoint_cls,
            patch(
                "brev_pipelines.assets.central_bank_speeches.process_with_checkpoint",
                return_value=results_df,
            ),
        ):
            mock_checkpoint_cls.return_value.cleanup = MagicMock()
            df = speech_summaries(
                asset_context,
                sample_cleaned_df,
                mock_nim_reasoning,
                mock_minio_for_checkpoint,
            )

        assert "reference" in df.columns
        assert "summary" in df.columns
        assert len(df) == 2
