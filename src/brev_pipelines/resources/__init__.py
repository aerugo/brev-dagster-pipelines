"""Dagster resources for Brev Data Platform."""

from brev_pipelines.resources.lakefs import (
    LakeFSConnectionError,
    LakeFSError,
    LakeFSNotFoundError,
    LakeFSResource,
)
from brev_pipelines.resources.minio import MinIOResource
from brev_pipelines.resources.nim import (
    NIMError,
    NIMRateLimitError,
    NIMResource,
    NIMServerError,
    NIMTimeoutError,
)
from brev_pipelines.resources.nim_embedding import NIMEmbeddingResource
from brev_pipelines.resources.safe_synth import SafeSynthesizerResource
from brev_pipelines.resources.safe_synth_retry import (
    SafeSynthError,
    SafeSynthJobFailedError,
    SafeSynthServerError,
    SafeSynthTimeoutError,
)
from brev_pipelines.resources.weaviate import (
    WeaviateCollectionError,
    WeaviateConnectionError,
    WeaviateError,
    WeaviateResource,
)

__all__ = [
    # LakeFS
    "LakeFSResource",
    "LakeFSError",
    "LakeFSConnectionError",
    "LakeFSNotFoundError",
    # MinIO
    "MinIOResource",
    # NIM
    "NIMError",
    "NIMRateLimitError",
    "NIMResource",
    "NIMServerError",
    "NIMTimeoutError",
    "NIMEmbeddingResource",
    # Safe Synthesizer
    "SafeSynthesizerResource",
    "SafeSynthError",
    "SafeSynthTimeoutError",
    "SafeSynthServerError",
    "SafeSynthJobFailedError",
    # Weaviate
    "WeaviateResource",
    "WeaviateError",
    "WeaviateConnectionError",
    "WeaviateCollectionError",
]
