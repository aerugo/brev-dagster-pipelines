"""Shared test fixtures for Dagster pipeline tests.

Provides mock implementations of all external services:
- MinIO (object storage)
- LakeFS (data versioning)
- Weaviate (vector database)
- NIM (LLM inference)
- Kubernetes (for Safe Synthesizer)

All fixtures follow INV-P010 (TDD) requirements for mocking external services.
"""

from __future__ import annotations

from collections.abc import Generator
from io import BytesIO
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
from dagster import build_asset_context

if TYPE_CHECKING:
    from dagster import AssetExecutionContext


# =============================================================================
# MinIO Fixtures
# =============================================================================


@pytest.fixture
def mock_minio_client() -> Generator[MagicMock, None, None]:
    """Mock MinIO client with standard operations.

    Provides a mock that simulates:
    - bucket_exists() returning True
    - put_object() succeeding
    - get_object() returning mock data
    - list_objects() returning empty iterator
    """
    with patch("minio.Minio") as mock_class:
        client = MagicMock()
        client.bucket_exists.return_value = True
        client.put_object.return_value = None
        client.make_bucket.return_value = None

        # Mock get_object to return BytesIO
        mock_response = MagicMock()
        mock_response.read.return_value = b"test data"
        mock_response.close.return_value = None
        client.get_object.return_value = mock_response

        # Mock list_objects
        client.list_objects.return_value = iter([])

        mock_class.return_value = client
        yield client


@pytest.fixture
def mock_minio_resource(mock_minio_client: MagicMock) -> MagicMock:
    """Mock MinIOResource with injected mock client."""
    resource = MagicMock()
    resource.endpoint = "localhost:9000"
    resource.access_key = "test-access-key"
    resource.secret_key = "test-secret-key"
    resource.secure = False
    resource.get_client.return_value = mock_minio_client
    return resource


# =============================================================================
# LakeFS Fixtures
# =============================================================================


@pytest.fixture
def mock_lakefs_client() -> Generator[MagicMock, None, None]:
    """Mock LakeFS client with standard operations.

    Provides a mock that simulates:
    - objects.upload_object() succeeding
    - objects.get_object() returning mock data
    - commits.commit() returning a commit object
    - branches operations
    """
    with patch("lakefs_sdk.client.LakeFSClient") as mock_class:
        client = MagicMock()

        # Mock objects API
        client.objects.upload_object.return_value = MagicMock(
            physical_address="s3://lakefs/data/test.parquet"
        )
        mock_obj_response = MagicMock()
        mock_obj_response.read.return_value = b"test parquet data"
        client.objects.get_object.return_value = mock_obj_response

        # Mock commits API
        mock_commit = MagicMock()
        mock_commit.id = "abc123def456"
        client.commits.commit.return_value = mock_commit

        # Mock branches API
        client.branches.list_branches.return_value = MagicMock(results=[MagicMock(id="main")])

        # Mock repositories API
        client.repositories.list_repositories.return_value = MagicMock(
            results=[MagicMock(id="central-bank-speeches")]
        )

        mock_class.return_value = client
        yield client


@pytest.fixture
def mock_lakefs_resource(mock_lakefs_client: MagicMock) -> MagicMock:
    """Mock LakeFSResource with injected mock client."""
    resource = MagicMock()
    resource.endpoint = "http://localhost:8000"
    resource.access_key = "test-access-key"
    resource.secret_key = "test-secret-key"
    resource.get_client.return_value = mock_lakefs_client
    resource.list_repositories.return_value = ["central-bank-speeches"]
    resource.health_check.return_value = True
    return resource


# =============================================================================
# Weaviate Fixtures
# =============================================================================


@pytest.fixture
def mock_weaviate_client() -> Generator[MagicMock, None, None]:
    """Mock Weaviate client with standard operations.

    Provides a mock that simulates:
    - collections.create() succeeding
    - collections.get() returning a collection
    - collection.data.insert_many() succeeding
    - collection.query.near_vector() returning results
    """
    with patch("weaviate.connect_to_local") as mock_connect:
        client = MagicMock()

        # Mock collection
        collection = MagicMock()

        # Mock insert_many response
        insert_response = MagicMock()
        insert_response.has_errors = False
        insert_response.uuids = {0: "uuid-1", 1: "uuid-2", 2: "uuid-3"}
        collection.data.insert_many.return_value = insert_response

        # Mock query response
        query_result = MagicMock()
        query_result.objects = [
            MagicMock(
                properties={"reference": "BIS_2024_001", "title": "Test Speech"},
                metadata=MagicMock(distance=0.1, certainty=0.9),
            )
        ]
        collection.query.near_vector.return_value = query_result

        # Mock aggregate
        aggregate_result = MagicMock()
        aggregate_result.total_count = 100
        collection.aggregate.over_all.return_value = aggregate_result

        client.collections.get.return_value = collection
        client.collections.create.return_value = collection
        client.collections.exists.return_value = True

        mock_connect.return_value = client
        yield client


@pytest.fixture
def mock_weaviate_resource(mock_weaviate_client: MagicMock) -> MagicMock:
    """Mock WeaviateResource with injected mock client."""
    resource = MagicMock()
    resource.host = "localhost"
    resource.http_port = 8080
    resource.grpc_port = 50051
    resource.get_client.return_value = mock_weaviate_client
    resource.ensure_collection.return_value = None
    resource.insert_objects.return_value = 3
    resource.get_object_count.return_value = 100
    return resource


# =============================================================================
# NIM LLM Fixtures
# =============================================================================


@pytest.fixture
def mock_nim_classification_response() -> MagicMock:
    """Mock NIM LLM response for speech classification.

    Returns a properly formatted classification response with:
    - monetary_stance
    - trade_stance
    - tariff_mention
    - economic_outlook
    """
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": """{
                        "monetary_stance": "neutral",
                        "trade_stance": "neutral",
                        "tariff_mention": 0,
                        "economic_outlook": "neutral"
                    }"""
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }
    response.raise_for_status = MagicMock()
    return response


@pytest.fixture
def mock_nim_summary_response() -> MagicMock:
    """Mock NIM LLM response for speech summarization."""
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "This speech discusses monetary policy and economic outlook."
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 500, "completion_tokens": 100, "total_tokens": 600},
    }
    response.raise_for_status = MagicMock()
    return response


@pytest.fixture
def mock_nim_resource(mock_nim_classification_response: MagicMock) -> MagicMock:
    """Mock NIMResource for LLM inference."""
    resource = MagicMock()
    resource.endpoint = "http://nvidia-nim.nvidia-nim.svc.cluster.local:8000"
    resource.model = "meta/llama3-8b-instruct"
    resource.timeout = 120

    # Mock generate method
    resource.generate.return_value = {
        "monetary_stance": "neutral",
        "trade_stance": "neutral",
        "tariff_mention": 0,
        "economic_outlook": "neutral",
    }

    resource.health_check.return_value = True
    return resource


# =============================================================================
# NIM Embedding Fixtures
# =============================================================================


@pytest.fixture
def mock_nim_embedding_response() -> MagicMock:
    """Mock NIM embedding response with 1024-dimensional vectors."""
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "data": [
            {"embedding": [0.1] * 1024, "index": 0},
            {"embedding": [0.2] * 1024, "index": 1},
            {"embedding": [0.3] * 1024, "index": 2},
        ],
        "model": "nvidia/llama-3_2-nemoretriever-300m-embed-v2",
        "usage": {"prompt_tokens": 100, "total_tokens": 100},
    }
    response.raise_for_status = MagicMock()
    return response


@pytest.fixture
def mock_nim_embedding_resource() -> MagicMock:
    """Mock NIMEmbeddingResource for vector embeddings."""
    resource = MagicMock()
    resource.endpoint = "http://nvidia-nim-embedding.nvidia-nim.svc.cluster.local:8000"
    resource.model = "nvidia/llama-3_2-nemoretriever-300m-embed-v2"
    resource.dimensions = 1024
    resource.timeout = 120

    # Mock embed_texts method
    def mock_embed_texts(texts: list[str]) -> list[list[float]]:
        return [[0.1 * (i + 1)] * 1024 for i in range(len(texts))]

    resource.embed_texts.side_effect = mock_embed_texts
    resource.embed_text.return_value = [0.1] * 1024
    resource.embed_query.return_value = [0.1] * 1024
    resource.health_check.return_value = True

    return resource


# =============================================================================
# Kubernetes Fixtures (for Safe Synthesizer)
# =============================================================================


@pytest.fixture
def mock_k8s_clients() -> Generator[dict[str, MagicMock], None, None]:
    """Mock Kubernetes clients for Safe Synthesizer.

    Provides mocks for:
    - BatchV1Api (job management)
    - CoreV1Api (pod operations)
    - AppsV1Api (deployment scaling)
    """
    with (
        patch("kubernetes.client.BatchV1Api") as batch_mock,
        patch("kubernetes.client.CoreV1Api") as core_mock,
        patch("kubernetes.client.AppsV1Api") as apps_mock,
        patch("kubernetes.config.load_incluster_config") as config_mock,
    ):
        batch_client = MagicMock()
        core_client = MagicMock()
        apps_client = MagicMock()

        # Mock job creation
        batch_client.create_namespaced_job.return_value = MagicMock(
            metadata=MagicMock(name="synth-job-123")
        )

        # Mock job status - completed successfully
        job_status = MagicMock()
        job_status.status.succeeded = 1
        job_status.status.failed = None
        job_status.status.completion_time = "2026-01-25T12:00:00Z"
        batch_client.read_namespaced_job_status.return_value = job_status

        # Mock job deletion
        batch_client.delete_namespaced_job.return_value = MagicMock()

        # Mock pod listing
        pod = MagicMock()
        pod.metadata.name = "synth-job-123-abc"
        core_client.list_namespaced_pod.return_value = MagicMock(items=[pod])
        core_client.read_namespaced_pod_log.return_value = "Job completed successfully"

        # Mock deployment operations
        deployment = MagicMock()
        deployment.spec.replicas = 0
        deployment.status.ready_replicas = 0
        apps_client.read_namespaced_deployment.return_value = deployment
        apps_client.patch_namespaced_deployment_scale.return_value = MagicMock()

        batch_mock.return_value = batch_client
        core_mock.return_value = core_client
        apps_mock.return_value = apps_client
        config_mock.return_value = None

        yield {
            "batch": batch_client,
            "core": core_client,
            "apps": apps_client,
        }


@pytest.fixture
def mock_safe_synth_resource(mock_k8s_clients: dict[str, MagicMock]) -> MagicMock:
    """Mock SafeSynthesizerResource for synthetic data generation."""
    resource = MagicMock()
    resource.namespace = "nvidia-ai"
    resource.service_endpoint = "http://safe-synthesizer:8000"
    resource.timeout = 3600

    # Mock synthesize method
    def mock_synthesize(
        input_data: list[dict],
        run_id: str,
        config: dict | None = None,
    ) -> tuple[list[dict], dict]:
        # Return synthetic data with same schema
        synthetic_data = [{**record, "is_synthetic": True} for record in input_data[:5]]
        evaluation = {
            "job_id": f"synth-{run_id}",
            "mia_score": 0.85,
            "aia_score": 0.90,
            "privacy_passed": True,
            "input_records": len(input_data),
            "output_records": len(synthetic_data),
        }
        return synthetic_data, evaluation

    resource.synthesize.side_effect = mock_synthesize
    return resource


# =============================================================================
# Sample Data Fixtures
# =============================================================================


@pytest.fixture
def sample_speeches_df() -> pl.DataFrame:
    """Sample speeches DataFrame for testing.

    Contains realistic data structure matching the CBS pipeline schema.
    """
    return pl.DataFrame(
        {
            "reference": ["BIS_2024_001", "BIS_2024_002", "ECB_2024_001"],
            "date": ["2024-01-15", "2024-01-20", "2024-02-01"],
            "central_bank": ["FED", "FED", "ECB"],
            "speaker": ["Jerome Powell", "Jerome Powell", "Christine Lagarde"],
            "title": ["Monetary Policy Outlook", "Economic Conditions", "Inflation Update"],
            "text": [
                "The Federal Reserve remains committed to price stability. " * 50,
                "Economic conditions continue to evolve. " * 50,
                "Inflation remains a key concern for policymakers. " * 50,
            ],
            "is_gov": [True, True, True],
        }
    )


@pytest.fixture
def sample_classified_speeches_df(sample_speeches_df: pl.DataFrame) -> pl.DataFrame:
    """Sample speeches DataFrame with classification columns added."""
    return sample_speeches_df.with_columns(
        [
            pl.Series("monetary_stance", [3, 4, 3]),
            pl.Series("trade_stance", [3, 2, 3]),
            pl.Series("tariff_mention", [0, 1, 0]),
            pl.Series("economic_outlook", [3, 4, 2]),
        ]
    )


@pytest.fixture
def sample_embeddings() -> list[list[float]]:
    """Sample embedding vectors for testing (1024 dimensions)."""
    return [
        [0.1] * 1024,
        [0.2] * 1024,
        [0.3] * 1024,
    ]


@pytest.fixture
def sample_speeches_with_embeddings(
    sample_classified_speeches_df: pl.DataFrame,
    sample_embeddings: list[list[float]],
) -> tuple[pl.DataFrame, list[list[float]]]:
    """Sample speeches with matching embeddings."""
    return sample_classified_speeches_df, sample_embeddings


# =============================================================================
# Dagster Context Fixtures
# =============================================================================


@pytest.fixture
def asset_context() -> AssetExecutionContext:
    """Build Dagster asset execution context for testing."""
    return build_asset_context()


@pytest.fixture
def asset_context_with_partition() -> AssetExecutionContext:
    """Build Dagster asset execution context with partition key."""
    return build_asset_context(partition_key="2024-01-15")


# =============================================================================
# File System Fixtures
# =============================================================================


@pytest.fixture
def temp_parquet_file(tmp_path: MagicMock, sample_speeches_df: pl.DataFrame) -> str:
    """Create a temporary parquet file for testing."""
    file_path = tmp_path / "test_speeches.parquet"
    sample_speeches_df.write_parquet(str(file_path))
    return str(file_path)


@pytest.fixture
def mock_parquet_bytes(sample_speeches_df: pl.DataFrame) -> bytes:
    """Generate parquet bytes from sample DataFrame."""
    buffer = BytesIO()
    sample_speeches_df.write_parquet(buffer)
    buffer.seek(0)
    return buffer.read()
