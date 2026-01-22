"""Health check assets for platform validation."""

import dagster as dg

from brev_pipelines.resources.lakefs import LakeFSResource
from brev_pipelines.resources.minio import MinIOResource
from brev_pipelines.resources.nim import NIMResource


@dg.asset(
    description="Platform health check",
    group_name="health",
)
def platform_health(
    context: dg.AssetExecutionContext,
    minio: MinIOResource,
    lakefs: LakeFSResource,
    nim: NIMResource,
) -> dict[str, str]:
    """Check health of all platform services."""
    health = {}

    # Check MinIO
    try:
        client = minio.get_client()
        client.list_buckets()
        health["minio"] = "healthy"
    except Exception as e:
        health["minio"] = f"error: {str(e)[:50]}"

    # Check LakeFS
    try:
        repos = lakefs.list_repositories()
        health["lakefs"] = f"healthy ({len(repos)} repos)"
    except Exception as e:
        health["lakefs"] = f"error: {str(e)[:50]}"

    # Check NIM
    health["nim"] = "healthy" if nim.health_check() else "unavailable"

    context.log.info(f"Platform health: {health}")
    return health


health_assets = [platform_health]
