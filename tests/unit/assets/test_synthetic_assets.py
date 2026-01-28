"""Unit tests for synthetic speeches pipeline assets.

Tests the synthetic pipeline asset functions including:
- enriched_data_for_synthesis: Loading enriched data from LakeFS
- synthetic_summaries: Generating synthetic data with Safe Synthesizer
- synthetic_validation_report: Storing privacy validation in LakeFS
- synthetic_embeddings: Generating embeddings for synthetic data
- synthetic_data_product: Storing synthetic data in LakeFS
- synthetic_weaviate_index: Indexing in Weaviate

All external service calls are mocked per INV-P010.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import polars as pl
import pytest

from brev_pipelines.assets.synthetic_speeches import (
    SYNTHETIC_SCHEMA,
    enriched_data_for_synthesis,
    synthetic_data_product,
    synthetic_embeddings,
    synthetic_embeddings_snapshot,
    synthetic_summaries,
    synthetic_summaries_snapshot,
    synthetic_validation_report,
    synthetic_weaviate_index,
)
from brev_pipelines.config import PipelineConfig

if TYPE_CHECKING:
    from dagster import AssetExecutionContext


class TestSyntheticSchema:
    """Tests for SYNTHETIC_SCHEMA constant."""

    def test_schema_has_required_fields(self) -> None:
        """Test schema contains all required fields."""
        field_names = [prop["name"] for prop in SYNTHETIC_SCHEMA]

        assert "reference" in field_names
        assert "date" in field_names
        assert "central_bank" in field_names
        assert "speaker" in field_names
        assert "title" in field_names
        assert "summary" in field_names  # Summaries, not full text
        assert "monetary_stance" in field_names
        assert "trade_stance" in field_names
        assert "tariff_mention" in field_names
        assert "economic_outlook" in field_names
        assert "is_governor" in field_names
        assert "is_synthetic" in field_names  # Synthetic marker

    def test_schema_has_is_synthetic_marker(self) -> None:
        """Test schema includes is_synthetic boolean field."""
        schema_dict = {prop["name"]: prop["type"] for prop in SYNTHETIC_SCHEMA}

        assert schema_dict["is_synthetic"] == "boolean"

    def test_schema_uses_summary_not_text(self) -> None:
        """Test schema uses summary field instead of full text."""
        field_names = [prop["name"] for prop in SYNTHETIC_SCHEMA]

        assert "summary" in field_names
        assert "text" not in field_names  # Full text excluded


class TestEnrichedDataForSynthesis:
    """Tests for enriched_data_for_synthesis asset."""

    @pytest.fixture
    def pipeline_config(self) -> PipelineConfig:
        """Create pipeline config."""
        return PipelineConfig(is_trial=False)

    @pytest.fixture
    def sample_enriched_parquet(self) -> bytes:
        """Create sample enriched data as parquet bytes."""
        df = pl.DataFrame(
            {
                "reference": ["BIS_001", "BIS_002"],
                "summary": ["Summary 1", "Summary 2"],
                "monetary_stance": [3, 4],
                "trade_stance": [3, 5],
                "economic_outlook": [4, 3],
            }
        )
        buffer = io.BytesIO()
        df.write_parquet(buffer)
        return buffer.getvalue()

    @pytest.fixture
    def mock_lakefs_with_client(self, sample_enriched_parquet: bytes) -> MagicMock:
        """Create mock LakeFS resource with properly structured client."""
        resource = MagicMock()
        client = MagicMock()
        client.objects_api = MagicMock()
        client.objects_api.get_object = MagicMock(return_value=sample_enriched_parquet)
        resource.get_client = MagicMock(return_value=client)
        return resource

    def test_loads_from_lakefs(
        self,
        asset_context: AssetExecutionContext,
        pipeline_config: PipelineConfig,
        mock_lakefs_with_client: MagicMock,
    ) -> None:
        """Test enriched_data_for_synthesis loads data from LakeFS."""
        df = enriched_data_for_synthesis(
            asset_context,
            pipeline_config,
            mock_lakefs_with_client,
        )

        assert len(df) == 2
        assert "reference" in df.columns
        assert "summary" in df.columns

    def test_uses_trial_path_when_is_trial(
        self,
        asset_context: AssetExecutionContext,
        mock_lakefs_with_client: MagicMock,
    ) -> None:
        """Test enriched_data_for_synthesis uses trial path when is_trial=True."""
        config = PipelineConfig(is_trial=True)

        enriched_data_for_synthesis(
            asset_context,
            config,
            mock_lakefs_with_client,
        )

        mock_client = mock_lakefs_with_client.get_client.return_value
        call_kwargs = mock_client.objects_api.get_object.call_args.kwargs
        assert "trial" in call_kwargs["path"]

    def test_raises_on_missing_columns(
        self,
        asset_context: AssetExecutionContext,
        pipeline_config: PipelineConfig,
    ) -> None:
        """Test enriched_data_for_synthesis raises if required columns missing."""
        # Create data missing required columns
        df = pl.DataFrame(
            {
                "reference": ["BIS_001"],
                "title": ["Speech 1"],
                # Missing: summary, monetary_stance, trade_stance, economic_outlook
            }
        )
        buffer = io.BytesIO()
        df.write_parquet(buffer)

        resource = MagicMock()
        client = MagicMock()
        client.objects_api = MagicMock()
        client.objects_api.get_object = MagicMock(return_value=buffer.getvalue())
        resource.get_client = MagicMock(return_value=client)

        with pytest.raises(ValueError, match="Missing required columns"):
            enriched_data_for_synthesis(
                asset_context,
                pipeline_config,
                resource,
            )


class TestSyntheticSummaries:
    """Tests for synthetic_summaries asset."""

    @pytest.fixture
    def sample_enriched_df(self) -> pl.DataFrame:
        """Create sample enriched DataFrame."""
        return pl.DataFrame(
            {
                "reference": ["BIS_001", "BIS_002", "BIS_003"],
                "date": ["2024-01-15", "2024-01-20", "2024-02-01"],
                "central_bank": ["FED", "ECB", "BOE"],
                "speaker": ["Powell", "Lagarde", "Bailey"],
                "title": ["Outlook 1", "Outlook 2", "Outlook 3"],
                "monetary_stance": [3, 4, 2],
                "trade_stance": [3, 5, 1],
                "economic_outlook": [4, 3, 2],
                "tariff_mention": [0, 1, 0],
                "is_governor": [1, 1, 1],
                "summary": [
                    "Inflation metrics: 2.5%",
                    "Trade policy focus on EU",
                    "Banking sector risks",
                ],
            }
        )

    @pytest.fixture
    def mock_k8s_scaler(self) -> MagicMock:
        """Create mock K8s scaler resource."""
        return MagicMock()

    def test_calls_safe_synth(
        self,
        asset_context: AssetExecutionContext,
        sample_enriched_df: pl.DataFrame,
        mock_safe_synth_resource: MagicMock,
        mock_k8s_scaler: MagicMock,
    ) -> None:
        """Test synthetic_summaries calls Safe Synthesizer."""
        mock_safe_synth_resource.synthesize.return_value = (
            sample_enriched_df.to_dicts(),
            {"mia_score": 0.85, "privacy_passed": True, "job_id": "test-job"},
        )

        df, evaluation = synthetic_summaries(
            asset_context,
            sample_enriched_df,
            mock_safe_synth_resource,
            mock_k8s_scaler,
        )

        mock_safe_synth_resource.synthesize.assert_called_once()

    def test_adds_synthetic_reference_ids(
        self,
        asset_context: AssetExecutionContext,
        sample_enriched_df: pl.DataFrame,
        mock_safe_synth_resource: MagicMock,
        mock_k8s_scaler: MagicMock,
    ) -> None:
        """Test synthetic_summaries generates SYNTH-XXXXXX IDs."""
        mock_safe_synth_resource.synthesize.return_value = (
            sample_enriched_df.to_dicts(),
            {"mia_score": 0.85, "privacy_passed": True, "job_id": "test-job"},
        )

        df, _ = synthetic_summaries(
            asset_context,
            sample_enriched_df,
            mock_safe_synth_resource,
            mock_k8s_scaler,
        )

        # All references should be SYNTH-XXXXXX format
        assert all(ref.startswith("SYNTH-") for ref in df["reference"].to_list())

    def test_adds_is_synthetic_flag(
        self,
        asset_context: AssetExecutionContext,
        sample_enriched_df: pl.DataFrame,
        mock_safe_synth_resource: MagicMock,
        mock_k8s_scaler: MagicMock,
    ) -> None:
        """Test synthetic_summaries adds is_synthetic=True flag."""
        mock_safe_synth_resource.synthesize.return_value = (
            sample_enriched_df.to_dicts(),
            {"mia_score": 0.85, "privacy_passed": True, "job_id": "test-job"},
        )

        df, _ = synthetic_summaries(
            asset_context,
            sample_enriched_df,
            mock_safe_synth_resource,
            mock_k8s_scaler,
        )

        assert "is_synthetic" in df.columns
        assert all(df["is_synthetic"].to_list())

    def test_returns_evaluation_metadata(
        self,
        asset_context: AssetExecutionContext,
        sample_enriched_df: pl.DataFrame,
        mock_safe_synth_resource: MagicMock,
        mock_k8s_scaler: MagicMock,
    ) -> None:
        """Test synthetic_summaries returns evaluation metadata."""
        mock_safe_synth_resource.synthesize.return_value = (
            sample_enriched_df.to_dicts(),
            {"mia_score": 0.85, "aia_score": 0.90, "privacy_passed": True, "job_id": "test-123"},
        )

        _, evaluation = synthetic_summaries(
            asset_context,
            sample_enriched_df,
            mock_safe_synth_resource,
            mock_k8s_scaler,
        )

        assert "total_records" in evaluation
        assert "mia_score" in evaluation
        assert "aia_score" in evaluation
        assert "privacy_passed" in evaluation
        assert "generated_at" in evaluation
        assert "synthesis_type" in evaluation

    def test_disables_holdout_for_small_datasets(
        self,
        asset_context: AssetExecutionContext,
        mock_safe_synth_resource: MagicMock,
        mock_k8s_scaler: MagicMock,
    ) -> None:
        """Test synthetic_summaries disables holdout for <500 records."""
        # Small dataset (< 500)
        small_df = pl.DataFrame(
            {
                "reference": [f"BIS_{i:03d}" for i in range(100)],
                "date": ["2024-01-15"] * 100,
                "central_bank": ["FED"] * 100,
                "speaker": ["Powell"] * 100,
                "title": ["Speech"] * 100,
                "monetary_stance": [3] * 100,
                "trade_stance": [3] * 100,
                "economic_outlook": [3] * 100,
                "tariff_mention": [0] * 100,
                "is_governor": [1] * 100,
                "summary": ["Summary"] * 100,
            }
        )

        mock_safe_synth_resource.synthesize.return_value = (
            small_df.to_dicts(),
            {"mia_score": 0.85, "privacy_passed": True, "job_id": "test-job"},
        )

        _, evaluation = synthetic_summaries(
            asset_context,
            small_df,
            mock_safe_synth_resource,
            mock_k8s_scaler,
        )

        # Check config passed to synthesize
        call_kwargs = mock_safe_synth_resource.synthesize.call_args.kwargs
        assert call_kwargs["config"]["data"]["holdout"] == 0


class TestSyntheticValidationReport:
    """Tests for synthetic_validation_report asset."""

    @pytest.fixture
    def sample_synthetic_tuple(self) -> tuple[pl.DataFrame, dict[str, Any]]:
        """Create sample synthetic data tuple."""
        df = pl.DataFrame(
            {
                "reference": ["SYNTH-000001", "SYNTH-000002"],
                "summary": ["Synthetic summary 1", "Synthetic summary 2"],
            }
        )
        evaluation = {
            "mia_score": 0.85,
            "aia_score": 0.90,
            "privacy_passed": True,
            "job_id": "test-job-123",
            "html_report_bytes": b"<html>Report</html>",
        }
        return df, evaluation

    @pytest.fixture
    def mock_lakefs_with_client(self) -> MagicMock:
        """Create mock LakeFS resource with properly structured client."""
        resource = MagicMock()
        client = MagicMock()
        client.objects_api = MagicMock()
        client.commits_api = MagicMock()
        mock_commit = MagicMock()
        mock_commit.id = "val123"
        client.commits_api.commit = MagicMock(return_value=mock_commit)
        resource.get_client = MagicMock(return_value=client)
        return resource

    def test_stores_json_report(
        self,
        asset_context: AssetExecutionContext,
        sample_synthetic_tuple: tuple[pl.DataFrame, dict[str, Any]],
        mock_lakefs_with_client: MagicMock,
    ) -> None:
        """Test synthetic_validation_report stores JSON report in LakeFS."""
        report = synthetic_validation_report(
            asset_context,
            sample_synthetic_tuple,
            mock_lakefs_with_client,
        )

        mock_client = mock_lakefs_with_client.get_client.return_value
        # Should upload at least twice (JSON + HTML)
        assert mock_client.objects_api.upload_object.call_count >= 2

        assert report["mia_score"] == 0.85
        assert report["html_report_available"] is True

    def test_stores_html_report_when_available(
        self,
        asset_context: AssetExecutionContext,
        sample_synthetic_tuple: tuple[pl.DataFrame, dict[str, Any]],
        mock_lakefs_with_client: MagicMock,
    ) -> None:
        """Test synthetic_validation_report stores HTML report when available."""
        mock_client = mock_lakefs_with_client.get_client.return_value

        uploaded_paths: list[str] = []

        def capture_upload(**kwargs: Any) -> None:
            uploaded_paths.append(kwargs["path"])

        mock_client.objects_api.upload_object.side_effect = capture_upload

        synthetic_validation_report(
            asset_context,
            sample_synthetic_tuple,
            mock_lakefs_with_client,
        )

        # Should have both JSON and HTML
        assert any("validation_report.json" in p for p in uploaded_paths)
        assert any("evaluation_report.html" in p for p in uploaded_paths)


class TestSyntheticEmbeddings:
    """Tests for synthetic_embeddings asset."""

    @pytest.fixture
    def sample_synthetic_tuple(self) -> tuple[pl.DataFrame, dict[str, Any]]:
        """Create sample synthetic data tuple."""
        df = pl.DataFrame(
            {
                "reference": ["SYNTH-000001", "SYNTH-000002"],
                "title": ["Title 1", "Title 2"],
                "summary": ["Summary of speech 1", "Summary of speech 2"],
            }
        )
        evaluation = {"mia_score": 0.85, "privacy_passed": True}
        return df, evaluation

    @pytest.fixture
    def mock_minio_resource(self) -> MagicMock:
        """Create mock MinIO resource."""
        resource = MagicMock()
        resource.ensure_bucket = MagicMock()
        client = MagicMock()
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

    @pytest.fixture
    def mock_checkpoint_manager(self) -> MagicMock:
        """Create mock checkpoint manager."""
        mgr = MagicMock()
        # Return empty list (no checkpoints)
        mgr.get_processed_ids.return_value = set()
        mgr.checkpoint_batch = MagicMock()
        mgr.clear_checkpoints = MagicMock()
        return mgr

    def test_generates_embeddings(
        self,
        asset_context: AssetExecutionContext,
        sample_synthetic_tuple: tuple[pl.DataFrame, dict[str, Any]],
        mock_nim_embedding: MagicMock,
        mock_minio_resource: MagicMock,
        mock_checkpoint_manager: MagicMock,
    ) -> None:
        """Test synthetic_embeddings generates embeddings for all rows."""
        from unittest.mock import patch

        with patch(
            "brev_pipelines.assets.synthetic_speeches.LLMCheckpointManager",
            return_value=mock_checkpoint_manager,
        ):
            df, embeddings = synthetic_embeddings(
                asset_context,
                sample_synthetic_tuple,
                mock_nim_embedding,
                mock_minio_resource,
            )

        assert len(embeddings) == 2
        assert len(embeddings[0]) == 1024  # Expected dimension

    def test_uses_title_and_summary_for_embedding(
        self,
        asset_context: AssetExecutionContext,
        sample_synthetic_tuple: tuple[pl.DataFrame, dict[str, Any]],
        mock_minio_resource: MagicMock,
        mock_checkpoint_manager: MagicMock,
    ) -> None:
        """Test synthetic_embeddings uses title + summary for embedding text."""
        from unittest.mock import patch

        # Create mock with captured texts
        captured_texts: list[str] = []
        resource = MagicMock()

        def capture_embed(texts: list[str], batch_size: int = 32) -> list[list[float]]:
            captured_texts.extend(texts)
            return [[0.1] * 1024 for _ in texts]

        resource.embed_texts = MagicMock(side_effect=capture_embed)
        resource.embed_text = MagicMock(return_value=[0.1] * 1024)

        with patch(
            "brev_pipelines.assets.synthetic_speeches.LLMCheckpointManager",
            return_value=mock_checkpoint_manager,
        ):
            synthetic_embeddings(
                asset_context,
                sample_synthetic_tuple,
                resource,
                mock_minio_resource,
            )

        # Check texts contain both title and summary
        assert len(captured_texts) == 2
        assert "Title 1" in captured_texts[0]
        assert "Summary of speech 1" in captured_texts[0]


class TestSyntheticDataProduct:
    """Tests for synthetic_data_product asset."""

    @pytest.fixture
    def sample_synthetic_tuple(self) -> tuple[pl.DataFrame, dict[str, Any]]:
        """Create sample synthetic data tuple."""
        df = pl.DataFrame(
            {
                "reference": ["SYNTH-000001", "SYNTH-000002"],
                "summary": ["Summary 1", "Summary 2"],
                "is_synthetic": [True, True],
            }
        )
        evaluation = {"mia_score": 0.85, "privacy_passed": True}
        return df, evaluation

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
        mock_commit.id = "synth123"
        client.commits_api.commit = MagicMock(return_value=mock_commit)
        resource.get_client = MagicMock(return_value=client)
        return resource

    def test_stores_to_lakefs(
        self,
        asset_context: AssetExecutionContext,
        pipeline_config: PipelineConfig,
        sample_synthetic_tuple: tuple[pl.DataFrame, dict[str, Any]],
        mock_lakefs_with_client: MagicMock,
    ) -> None:
        """Test synthetic_data_product stores data in LakeFS."""
        result = synthetic_data_product(
            asset_context,
            pipeline_config,
            sample_synthetic_tuple,
            mock_lakefs_with_client,
            synthetic_embeddings_snapshot={"path": "mock", "commit_id": "mock"},
        )

        mock_client = mock_lakefs_with_client.get_client.return_value
        mock_client.objects_api.upload_object.assert_called_once()
        assert result["num_records"] == 2
        assert "synthetic" in result["path"]

    def test_uses_trial_path(
        self,
        asset_context: AssetExecutionContext,
        sample_synthetic_tuple: tuple[pl.DataFrame, dict[str, Any]],
        mock_lakefs_with_client: MagicMock,
    ) -> None:
        """Test synthetic_data_product uses trial path when is_trial=True."""
        config = PipelineConfig(is_trial=True)

        synthetic_data_product(
            asset_context,
            config,
            sample_synthetic_tuple,
            mock_lakefs_with_client,
            synthetic_embeddings_snapshot={"path": "mock", "commit_id": "mock"},
        )

        mock_client = mock_lakefs_with_client.get_client.return_value
        call_kwargs = mock_client.objects_api.upload_object.call_args.kwargs
        assert "trial" in call_kwargs["path"]

    def test_adds_generated_at_timestamp(
        self,
        asset_context: AssetExecutionContext,
        pipeline_config: PipelineConfig,
        sample_synthetic_tuple: tuple[pl.DataFrame, dict[str, Any]],
        mock_lakefs_with_client: MagicMock,
    ) -> None:
        """Test synthetic_data_product adds generated_at timestamp."""
        mock_client = mock_lakefs_with_client.get_client.return_value

        # Capture uploaded content
        captured_content: list[bytes] = []

        def capture_upload(**kwargs: Any) -> None:
            captured_content.append(kwargs["content"])

        mock_client.objects_api.upload_object.side_effect = capture_upload

        synthetic_data_product(
            asset_context,
            pipeline_config,
            sample_synthetic_tuple,
            mock_lakefs_with_client,
            synthetic_embeddings_snapshot={"path": "mock", "commit_id": "mock"},
        )

        # Parse uploaded parquet
        uploaded_df = pl.read_parquet(io.BytesIO(captured_content[0]))
        assert "generated_at" in uploaded_df.columns


class TestSyntheticWeaviateIndex:
    """Tests for synthetic_weaviate_index asset."""

    @pytest.fixture
    def sample_embeddings_tuple(self) -> tuple[pl.DataFrame, list[list[float]]]:
        """Create sample embeddings tuple."""
        df = pl.DataFrame(
            {
                "reference": ["SYNTH-000001", "SYNTH-000002"],
                "date": ["2024-01-15", "2024-01-20"],
                "central_bank": ["FED", "ECB"],
                "speaker": ["Powell", "Lagarde"],
                "title": ["Outlook 1", "Outlook 2"],
                "summary": ["Summary 1", "Summary 2"],
                "monetary_stance": [3, 4],
                "trade_stance": [3, 5],
                "tariff_mention": [0, 1],
                "economic_outlook": [4, 3],
                "is_governor": [True, True],
                "is_synthetic": [True, True],
            }
        )
        embeddings = [[0.1] * 1024, [0.2] * 1024]
        return df, embeddings

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

    def test_uses_synthetic_collection(
        self,
        asset_context: AssetExecutionContext,
        pipeline_config: PipelineConfig,
        sample_embeddings_tuple: tuple[pl.DataFrame, list[list[float]]],
        mock_weaviate: MagicMock,
    ) -> None:
        """Test synthetic_weaviate_index uses SyntheticSpeeches collection."""
        result = synthetic_weaviate_index(
            asset_context,
            pipeline_config,
            sample_embeddings_tuple,
            mock_weaviate,
        )

        call_kwargs = mock_weaviate.ensure_collection.call_args.kwargs
        assert call_kwargs["name"] == "SyntheticSpeeches"
        assert result["collection"] == "SyntheticSpeeches"

    def test_uses_trial_collection(
        self,
        asset_context: AssetExecutionContext,
        sample_embeddings_tuple: tuple[pl.DataFrame, list[list[float]]],
        mock_weaviate: MagicMock,
    ) -> None:
        """Test synthetic_weaviate_index uses trial collection when is_trial=True."""
        config = PipelineConfig(is_trial=True)

        result = synthetic_weaviate_index(
            asset_context,
            config,
            sample_embeddings_tuple,
            mock_weaviate,
        )

        assert result["collection"] == "SyntheticSpeechesTrial"

    def test_inserts_all_objects(
        self,
        asset_context: AssetExecutionContext,
        pipeline_config: PipelineConfig,
        sample_embeddings_tuple: tuple[pl.DataFrame, list[list[float]]],
        mock_weaviate: MagicMock,
    ) -> None:
        """Test synthetic_weaviate_index inserts all objects."""
        result = synthetic_weaviate_index(
            asset_context,
            pipeline_config,
            sample_embeddings_tuple,
            mock_weaviate,
        )

        mock_weaviate.insert_objects.assert_called_once()
        call_kwargs = mock_weaviate.insert_objects.call_args.kwargs
        assert len(call_kwargs["objects"]) == 2
        assert len(call_kwargs["vectors"]) == 2
        assert result["object_count"] == 2

    def test_objects_have_is_synthetic_true(
        self,
        asset_context: AssetExecutionContext,
        pipeline_config: PipelineConfig,
        sample_embeddings_tuple: tuple[pl.DataFrame, list[list[float]]],
        mock_weaviate: MagicMock,
    ) -> None:
        """Test synthetic_weaviate_index sets is_synthetic=True on all objects."""
        synthetic_weaviate_index(
            asset_context,
            pipeline_config,
            sample_embeddings_tuple,
            mock_weaviate,
        )

        call_kwargs = mock_weaviate.insert_objects.call_args.kwargs
        for obj in call_kwargs["objects"]:
            assert obj["is_synthetic"] is True


class TestSyntheticSummariesSnapshot:
    """Tests for synthetic_summaries_snapshot asset."""

    @pytest.fixture
    def sample_synthetic_tuple(self) -> tuple[pl.DataFrame, dict[str, Any]]:
        """Create sample synthetic data tuple."""
        df = pl.DataFrame(
            {
                "reference": ["SYNTH-000001", "SYNTH-000002"],
                "summary": ["Summary 1", "Summary 2"],
            }
        )
        evaluation = {"mia_score": 0.85, "privacy_passed": True}
        return df, evaluation

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

    def test_stores_snapshot(
        self,
        asset_context: AssetExecutionContext,
        pipeline_config: PipelineConfig,
        sample_synthetic_tuple: tuple[pl.DataFrame, dict[str, Any]],
        mock_lakefs_with_client: MagicMock,
    ) -> None:
        """Test synthetic_summaries_snapshot stores snapshot in LakeFS."""
        # Mock synthetic_validation_report dependency (just needs to be a dict)
        mock_validation_report: dict[str, object] = {"path": "test", "commit_id": "abc123"}
        result = synthetic_summaries_snapshot(
            asset_context,
            pipeline_config,
            sample_synthetic_tuple,
            mock_lakefs_with_client,
            mock_validation_report,
        )

        mock_client = mock_lakefs_with_client.get_client.return_value
        mock_client.objects_api.upload_object.assert_called_once()
        assert result["num_records"] == 2
        assert "evaluation" in result


class TestSyntheticEmbeddingsSnapshot:
    """Tests for synthetic_embeddings_snapshot asset."""

    @pytest.fixture
    def sample_embeddings_tuple(self) -> tuple[pl.DataFrame, list[list[float]]]:
        """Create sample embeddings tuple."""
        df = pl.DataFrame(
            {
                "reference": ["SYNTH-000001", "SYNTH-000002"],
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

    def test_stores_embeddings_snapshot(
        self,
        asset_context: AssetExecutionContext,
        pipeline_config: PipelineConfig,
        sample_embeddings_tuple: tuple[pl.DataFrame, list[list[float]]],
        mock_lakefs_with_client: MagicMock,
    ) -> None:
        """Test synthetic_embeddings_snapshot stores snapshot in LakeFS."""
        # Mock synthetic_summaries_snapshot dependency (just needs to be a dict)
        mock_summaries_snapshot: dict[str, object] = {"path": "test", "commit_id": "abc123"}
        result = synthetic_embeddings_snapshot(
            asset_context,
            pipeline_config,
            sample_embeddings_tuple,
            mock_lakefs_with_client,
            mock_summaries_snapshot,
        )

        mock_client = mock_lakefs_with_client.get_client.return_value
        mock_client.objects_api.upload_object.assert_called_once()
        assert result["num_records"] == 2
        assert result["dimensions"] == 1024
        assert "size_mb" in result
