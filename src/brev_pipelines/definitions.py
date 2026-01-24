"""Dagster definitions for Brev Data Platform."""

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
            endpoint=os.getenv("MINIO_ENDPOINT", "minio.minio.svc.cluster.local:9000"),
            access_key=EnvVar("MINIO_ACCESS_KEY"),
            secret_key=EnvVar("MINIO_SECRET_KEY"),
            secure=False,
        ),
        "lakefs": LakeFSResource(
            endpoint=os.getenv("LAKEFS_ENDPOINT", "lakefs.lakefs.svc.cluster.local:8000"),
            access_key=EnvVar("LAKEFS_ACCESS_KEY_ID"),
            secret_key=EnvVar("LAKEFS_SECRET_ACCESS_KEY"),
        ),
        "nim": NIMResource(
            endpoint=os.getenv(
                "NIM_ENDPOINT", "http://nim-llm.nvidia-ai.svc.cluster.local:8000"
            ),
        ),
        "nim_reasoning": NIMResource(
            endpoint=os.getenv(
                "NIM_REASONING_ENDPOINT", "http://nim-reasoning.nvidia-ai.svc.cluster.local:8000"
            ),
            model="openai/gpt-oss-120b",
            timeout=600,  # Longer timeout for large reasoning model
        ),
        "nim_embedding": NIMEmbeddingResource(
            endpoint=os.getenv(
                "NIM_EMBEDDING_ENDPOINT",
                "http://nvidia-nim-embedding.nvidia-nim.svc.cluster.local:8000",
            ),
        ),
        "safe_synth": SafeSynthesizerResource(
            namespace=os.getenv("SAFE_SYNTH_NAMESPACE", "nvidia-ai"),
            service_endpoint=os.getenv(
                "SAFE_SYNTH_ENDPOINT",
                "http://nemo-safe-synthesizer.nvidia-ai.svc.cluster.local:8000",
            ),
            nds_endpoint=os.getenv(
                "NDS_ENDPOINT",
                "http://nemo-data-store.nvidia-ai.svc.cluster.local:3000",
            ),
            nds_token=os.getenv("HF_TOKEN", ""),
            nds_repo=os.getenv("NDS_REPO", "admin/central-bank-speeches"),
            priority_class=os.getenv("SAFE_SYNTH_PRIORITY", "batch-high"),
        ),
        "weaviate": WeaviateResource(
            host=os.getenv("WEAVIATE_HOST", "weaviate.weaviate.svc.cluster.local"),
            port=int(os.getenv("WEAVIATE_PORT", "80")),
            grpc_host=os.getenv(
                "WEAVIATE_GRPC_HOST", "weaviate-grpc.weaviate.svc.cluster.local"
            ),
            grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT", "50051")),
        ),
    },
)
