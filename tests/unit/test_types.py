"""Unit tests for brev_pipelines.types module.

Tests that TypedDict definitions have the expected structure and can be
used for type checking. Also tests Protocol types for Kubernetes clients.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from brev_pipelines.types import (
    DataProductMetadata,
    NIMCompletionResponse,
    NIMEmbeddingResponse,
    SafeSynthConfig,
    SafeSynthEvaluationResult,
    SafeSynthJobStatus,
    SnapshotMetadata,
    SpeechClassification,
    ValidationReportDict,
    ValidationTestResult,
    WeaviateIndexMetadata,
    WeaviateObject,
    WeaviatePropertyDef,
    WeaviateSearchResult,
)


class TestWeaviateTypes:
    """Tests for Weaviate-related TypedDicts."""

    def test_weaviate_property_def_minimal(self):
        """Test WeaviatePropertyDef with only required field."""
        prop: WeaviatePropertyDef = {"name": "title"}
        assert prop["name"] == "title"

    def test_weaviate_property_def_complete(self):
        """Test WeaviatePropertyDef with all fields."""
        prop: WeaviatePropertyDef = {
            "name": "monetary_stance",
            "type": "int",
            "description": "1=very_dovish to 5=very_hawkish",
        }
        assert prop["name"] == "monetary_stance"
        assert prop["type"] == "int"
        assert "dovish" in prop["description"]

    def test_weaviate_search_result_structure(self):
        """Test WeaviateSearchResult has expected structure."""
        result: WeaviateSearchResult = {
            "properties": {"reference": "BIS_2024_001", "title": "Test Speech"},
            "_distance": 0.15,
            "_certainty": 0.85,
        }
        assert result["_distance"] < 1.0
        assert result["_certainty"] > 0.5
        assert "reference" in result["properties"]

    def test_weaviate_object_structure(self):
        """Test WeaviateObject has all required fields."""
        obj: WeaviateObject = {
            "reference": "BIS_2024_001",
            "date": "2024-01-15",
            "central_bank": "FED",
            "speaker": "Jerome Powell",
            "title": "Monetary Policy Update",
            "text": "The Federal Reserve...",
            "monetary_stance": 3,
            "trade_stance": 3,
            "tariff_mention": False,
            "economic_outlook": 4,
            "is_governor": True,
        }
        assert obj["reference"].startswith("BIS")
        assert obj["monetary_stance"] == 3
        assert obj["is_governor"] is True


class TestSafeSynthTypes:
    """Tests for Safe Synthesizer TypedDicts."""

    def test_safe_synth_config_defaults(self):
        """Test SafeSynthConfig can be empty (all fields optional)."""
        config: SafeSynthConfig = {}
        assert len(config) == 0

    def test_safe_synth_config_complete(self):
        """Test SafeSynthConfig with all fields."""
        config: SafeSynthConfig = {
            "epsilon": 1.0,
            "delta": 1e-5,
            "piiReplacement": True,
            "temperature": 0.7,
            "runMiaEvaluation": True,
            "runAiaEvaluation": True,
        }
        assert config["epsilon"] == 1.0
        assert config["piiReplacement"] is True

    def test_safe_synth_job_status_completed(self):
        """Test SafeSynthJobStatus for successful job."""
        status: SafeSynthJobStatus = {
            "state": "completed",
            "succeeded": 1,
            "completion_time": "2024-01-15T12:00:00Z",
        }
        assert status["state"] == "completed"
        assert status["succeeded"] == 1

    def test_safe_synth_job_status_failed(self):
        """Test SafeSynthJobStatus for failed job."""
        status: SafeSynthJobStatus = {
            "state": "failed",
            "failed": 1,
            "error": "Out of memory",
        }
        assert status["state"] == "failed"
        assert "memory" in status["error"]

    def test_safe_synth_evaluation_result(self):
        """Test SafeSynthEvaluationResult structure."""
        result: SafeSynthEvaluationResult = {
            "mia_score": 0.92,
            "aia_score": 0.88,
            "privacy_passed": True,
            "input_records": 100,
            "output_records": 95,
            "job_id": "synth-job-123",
        }
        assert result["privacy_passed"] is True
        assert result["mia_score"] > 0.9


class TestAssetOutputTypes:
    """Tests for asset output TypedDicts."""

    def test_data_product_metadata(self):
        """Test DataProductMetadata structure."""
        metadata: DataProductMetadata = {
            "path": "lakefs://data/main/central-bank-speeches/speeches.parquet",
            "commit_id": "abc123def456",
            "num_records": 1500,
            "tariff_mentions": 45,
        }
        assert metadata["path"].startswith("lakefs://")
        assert metadata["num_records"] > 0

    def test_weaviate_index_metadata(self):
        """Test WeaviateIndexMetadata structure."""
        metadata: WeaviateIndexMetadata = {
            "collection": "CentralBankSpeeches",
            "object_count": 1500,
            "vector_dimensions": 1024,
        }
        assert metadata["collection"] == "CentralBankSpeeches"
        assert metadata["vector_dimensions"] == 1024

    def test_snapshot_metadata(self):
        """Test SnapshotMetadata structure."""
        metadata: SnapshotMetadata = {
            "path": "lakefs://data/main/intermediate/classifications.parquet",
            "commit_id": "abc123",
            "num_records": 100,
        }
        assert "intermediate" in metadata["path"]


class TestValidationTypes:
    """Tests for validation TypedDicts."""

    def test_validation_test_result_passed(self):
        """Test ValidationTestResult for passed test."""
        result: ValidationTestResult = {
            "name": "connection",
            "passed": True,
        }
        assert result["passed"] is True

    def test_validation_test_result_failed(self):
        """Test ValidationTestResult for failed test."""
        result: ValidationTestResult = {
            "name": "write_object",
            "passed": False,
            "error": "Permission denied",
        }
        assert result["passed"] is False
        assert "Permission" in result["error"]

    def test_validation_report_dict(self):
        """Test ValidationReportDict structure."""
        report: ValidationReportDict = {
            "component": "minio",
            "passed": True,
            "tests": [
                {"name": "connection", "passed": True},
                {"name": "list_buckets", "passed": True, "bucket_count": 3},
            ],
            "duration_ms": 125.5,
        }
        assert report["component"] == "minio"
        assert len(report["tests"]) == 2
        assert report["duration_ms"] > 0


class TestNIMTypes:
    """Tests for NIM API TypedDicts."""

    def test_nim_completion_response(self):
        """Test NIMCompletionResponse structure."""
        response: NIMCompletionResponse = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Generated text"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        }
        assert len(response["choices"]) == 1
        assert response["usage"]["total_tokens"] == 150

    def test_nim_embedding_response(self):
        """Test NIMEmbeddingResponse structure."""
        response: NIMEmbeddingResponse = {
            "data": [
                {"embedding": [0.1] * 1024, "index": 0},
                {"embedding": [0.2] * 1024, "index": 1},
            ],
            "model": "nvidia/llama-3_2-nemoretriever-300m-embed-v2",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 0,
                "total_tokens": 100,
            },
        }
        assert len(response["data"]) == 2
        assert len(response["data"][0]["embedding"]) == 1024


class TestClassificationTypes:
    """Tests for speech classification TypedDicts."""

    def test_speech_classification(self):
        """Test SpeechClassification structure."""
        classification: SpeechClassification = {
            "monetary_stance": 3,
            "trade_stance": 4,
            "tariff_mention": 1,
            "economic_outlook": 3,
        }
        assert 1 <= classification["monetary_stance"] <= 5
        assert classification["tariff_mention"] in (0, 1)


class TestK8sProtocols:
    """Tests for Kubernetes Protocol types.

    These tests verify that mocks can be used with the expected interface.
    The Protocol types are for static type checking, not runtime checks.
    """

    def test_k8s_batch_api_mock_usage(self):
        """Test mock can be used with K8sBatchV1Api interface."""
        mock_api = MagicMock()
        mock_job = MagicMock()
        mock_job.metadata.name = "test-job"
        mock_job.status.succeeded = 1
        mock_api.create_namespaced_job.return_value = mock_job
        mock_api.read_namespaced_job_status.return_value = mock_job

        # Verify the mock works with expected method calls
        result = mock_api.create_namespaced_job(namespace="default", body={})
        assert result.metadata.name == "test-job"

        status = mock_api.read_namespaced_job_status(name="test-job", namespace="default")
        assert status.status.succeeded == 1

    def test_k8s_core_api_mock_usage(self):
        """Test mock can be used with K8sCoreV1Api interface."""
        mock_api = MagicMock()
        mock_pod = MagicMock()
        mock_pod.metadata.name = "synth-job-abc"
        mock_pod_list = MagicMock()
        mock_pod_list.items = [mock_pod]
        mock_api.list_namespaced_pod.return_value = mock_pod_list
        mock_api.read_namespaced_pod_log.return_value = "Job completed"

        # Verify the mock works with expected method calls
        pods = mock_api.list_namespaced_pod(namespace="nvidia-ai", label_selector="job-name=test")
        assert len(pods.items) == 1
        assert pods.items[0].metadata.name == "synth-job-abc"

        logs = mock_api.read_namespaced_pod_log(name="synth-job-abc", namespace="nvidia-ai")
        assert "completed" in logs.lower()

    def test_k8s_apps_api_mock_usage(self):
        """Test mock can be used with K8sAppsV1Api interface."""
        mock_api = MagicMock()
        mock_deployment = MagicMock()
        mock_deployment.spec.replicas = 1
        mock_deployment.status.ready_replicas = 1
        mock_api.read_namespaced_deployment.return_value = mock_deployment
        mock_api.patch_namespaced_deployment_scale.return_value = None

        # Verify the mock works with expected method calls
        deployment = mock_api.read_namespaced_deployment(
            name="nemo-safe-synthesizer", namespace="nvidia-ai"
        )
        assert deployment.spec.replicas == 1
        assert deployment.status.ready_replicas == 1

        mock_api.patch_namespaced_deployment_scale(
            name="nemo-safe-synthesizer",
            namespace="nvidia-ai",
            body={"spec": {"replicas": 0}},
        )
        mock_api.patch_namespaced_deployment_scale.assert_called_once()
