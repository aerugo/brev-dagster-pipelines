"""LakeFS resource for Dagster.

Provides LakeFS data versioning capabilities for Dagster pipelines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dagster import ConfigurableResource
from pydantic import Field

if TYPE_CHECKING:
    from lakefs_sdk.client import LakeFSClient


class LakeFSResource(ConfigurableResource):  # type: ignore[type-arg]
    """LakeFS data versioning resource.

    Manages connections to LakeFS for data versioning operations.
    Supports both http:// and https:// endpoint configurations.

    Attributes:
        endpoint: LakeFS endpoint URL or host:port.
        access_key: LakeFS access key ID.
        secret_key: LakeFS secret access key.
    """

    endpoint: str = Field(description="LakeFS endpoint (host:port)")
    access_key: str = Field(description="Access key ID")
    secret_key: str = Field(description="Secret access key")

    def get_client(self) -> LakeFSClient:
        """Get LakeFS client instance.

        Returns:
            Configured LakeFSClient for API operations.
        """
        import lakefs_sdk
        from lakefs_sdk.client import LakeFSClient

        # Handle endpoint with or without protocol
        endpoint = self.endpoint
        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            host = endpoint
        else:
            host = f"http://{endpoint}"

        # lakefs_sdk doesn't export Configuration in __all__ but it's valid
        config = lakefs_sdk.Configuration(  # type: ignore[attr-defined]
            host=host,
            username=self.access_key,
            password=self.secret_key,
        )
        return LakeFSClient(config)  # type: ignore[no-untyped-call]

    def list_repositories(self) -> list[str]:
        """List all repositories."""
        client = self.get_client()
        repos = client.repositories_api.list_repositories()
        return [repo.id for repo in repos.results]

    def health_check(self) -> bool:
        """Check if LakeFS is accessible."""
        try:
            client = self.get_client()
            client.repositories_api.list_repositories()
            return True
        except Exception:
            return False
