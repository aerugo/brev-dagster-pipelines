"""Dagster definitions for Brev Data Platform.

Environment variable handling uses Dagster's EnvVar pattern for consistency.
EnvVar.get_value() resolves at definition time; EnvVar() without get_value()
is used for secrets (MINIO keys, LakeFS keys) which Dagster resolves at runtime.
"""

import os

from dagster import Definitions, EnvVar

from brev_pipelines.assets.central_bank_speeches import central_bank_speeches_assets
from brev_pipelines.assets.demo import demo_assets
from brev_pipelines.assets.health import health_assets
from brev_pipelines.assets.synthetic_speeches import synthetic_speeches_assets
from brev_pipelines.assets.validation import validation_assets
from brev_pipelines.jobs import all_jobs
from brev_pipelines.resources.lakefs import LakeFSResource
from brev_pipelines.resources.minio import MinIOResource
from brev_pipelines.resources.nim import NIMResource
from brev_pipelines.resources.nim_embedding import NIMEmbeddingResource
from brev_pipelines.resources.safe_synth import SafeSynthesizerResource
from brev_pipelines.resources.weaviate import WeaviateResource


def _env(name: str, default: str) -> str:
    """Get environment variable value with default, using Dagster's EnvVar."""
    return EnvVar(name).get_value() or default


def _env_int(name: str, default: int) -> int:
    """Get environment variable as int with default, using Dagster's EnvVar."""
    value = EnvVar(name).get_value()
    return int(value) if value else default


def _env_bool(name: str, default: bool) -> bool:
    """Get environment variable as bool with default."""
    value = EnvVar(name).get_value()
    if value is None:
        return default
    return value.lower() in ("true", "1", "yes")


def _is_running_in_kubernetes() -> bool:
    """Detect if running inside a Kubernetes pod.

    Kubernetes automatically sets KUBERNETES_SERVICE_HOST in all pods.
    """
    return "KUBERNETES_SERVICE_HOST" in os.environ


# Auto-detect environment: use mock fallback only in local dev (not in k8s)
# Can be overridden with USE_MOCK_FALLBACK env var
_use_mock_fallback = _env_bool(
    "USE_MOCK_FALLBACK",
    default=not _is_running_in_kubernetes(),
)


defs = Definitions(
    assets=[
        *demo_assets,
        *health_assets,
        *validation_assets,
        *central_bank_speeches_assets,
        *synthetic_speeches_assets,
    ],
    jobs=all_jobs,
    resources={
        "minio": MinIOResource(
            endpoint=_env("MINIO_ENDPOINT", "minio.minio.svc.cluster.local:9000"),
            access_key=EnvVar("MINIO_ACCESS_KEY"),
            secret_key=EnvVar("MINIO_SECRET_KEY"),
            secure=False,
        ),
        "lakefs": LakeFSResource(
            endpoint=_env("LAKEFS_ENDPOINT", "lakefs.lakefs.svc.cluster.local:8000"),
            access_key=EnvVar("LAKEFS_ACCESS_KEY_ID"),
            secret_key=EnvVar("LAKEFS_SECRET_ACCESS_KEY"),
        ),
        "nim": NIMResource(
            endpoint=_env("NIM_ENDPOINT", "http://nim-llm.nvidia-ai.svc.cluster.local:8000"),
            use_mock_fallback=_use_mock_fallback,
        ),
        "nim_reasoning": NIMResource(
            endpoint=_env(
                "NIM_REASONING_ENDPOINT", "http://nim-reasoning.nvidia-ai.svc.cluster.local:8000"
            ),
            model="openai/gpt-oss-120b",
            timeout=600,  # Longer timeout for large reasoning model
            use_mock_fallback=_use_mock_fallback,
        ),
        "nim_embedding": NIMEmbeddingResource(
            endpoint=_env(
                "NIM_EMBEDDING_ENDPOINT",
                "http://nvidia-nim-embedding.nvidia-nim.svc.cluster.local:8000",
            ),
            use_mock_fallback=_use_mock_fallback,
        ),
        "safe_synth": SafeSynthesizerResource(
            namespace=_env("SAFE_SYNTH_NAMESPACE", "nvidia-ai"),
            service_endpoint=_env(
                "SAFE_SYNTH_ENDPOINT",
                "http://nemo-safe-synthesizer.nvidia-ai.svc.cluster.local:8000",
            ),
            nds_endpoint=_env(
                "NDS_ENDPOINT",
                "http://nemo-data-store.nvidia-ai.svc.cluster.local:3000",
            ),
            nds_token=_env("HF_TOKEN", ""),
            nds_repo=_env("NDS_REPO", "admin/central-bank-speeches"),
            priority_class=_env("SAFE_SYNTH_PRIORITY", "batch-high"),
            use_mock_fallback=_use_mock_fallback,
        ),
        "weaviate": WeaviateResource(
            host=_env("WEAVIATE_HOST", "weaviate.weaviate.svc.cluster.local"),
            port=_env_int("WEAVIATE_PORT", 80),
            grpc_host=_env("WEAVIATE_GRPC_HOST", "weaviate-grpc.weaviate.svc.cluster.local"),
            grpc_port=_env_int("WEAVIATE_GRPC_PORT", 50051),
        ),
    },
)
