"""LakeFS resource for Dagster."""

from dagster import ConfigurableResource
from pydantic import Field


class LakeFSResource(ConfigurableResource):
    """LakeFS data versioning resource."""

    endpoint: str = Field(description="LakeFS endpoint (host:port)")
    access_key: str = Field(description="Access key ID")
    secret_key: str = Field(description="Secret access key")

    def get_client(self):
        """Get LakeFS client instance."""
        import lakefs_sdk
        from lakefs_sdk.client import LakeFSClient

        config = lakefs_sdk.Configuration(
            host=f"http://{self.endpoint}",
            username=self.access_key,
            password=self.secret_key,
        )
        return LakeFSClient(config)

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
