"""Dagster definitions for Brev Data Platform."""

import os

from dagster import Definitions, EnvVar

from brev_pipelines.assets.demo import demo_assets
from brev_pipelines.assets.health import health_assets
from brev_pipelines.resources.lakefs import LakeFSResource
from brev_pipelines.resources.minio import MinIOResource
from brev_pipelines.resources.nim import NIMResource

defs = Definitions(
    assets=[
        *demo_assets,
        *health_assets,
    ],
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
                "NIM_ENDPOINT", "http://nvidia-nim-llm.nvidia-nim.svc.cluster.local:8000"
            ),
        ),
    },
)
