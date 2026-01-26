"""LakeFS resource for Dagster.

Provides LakeFS data versioning capabilities for Dagster pipelines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dagster import ConfigurableResource
from pydantic import Field

if TYPE_CHECKING:
    from lakefs_sdk.client import LakeFSClient


# =============================================================================
# Exception Types
# =============================================================================


class LakeFSError(Exception):
    """Base exception for LakeFS errors."""

    pass


class LakeFSConnectionError(LakeFSError):
    """Raised when LakeFS connection fails."""

    pass


class LakeFSNotFoundError(LakeFSError):
    """Raised when requested object/branch/repo not found."""

    pass


# Connection-related error patterns
CONNECTION_ERROR_PATTERNS = (
    "connection",
    "timeout",
    "network",
    "refused",
    "unreachable",
    "timed out",
)

# Not found error patterns
NOT_FOUND_PATTERNS = ("not found", "404", "does not exist")


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

        Raises:
            LakeFSConnectionError: Cannot connect to LakeFS.
        """
        import lakefs_sdk
        from lakefs_sdk.client import LakeFSClient

        # Handle endpoint with or without protocol
        endpoint = self.endpoint
        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            host = endpoint
        else:
            host = f"http://{endpoint}"

        try:
            # lakefs_sdk doesn't export Configuration in __all__ but it's valid
            config = lakefs_sdk.Configuration(  # type: ignore[attr-defined]
                host=host,
                username=self.access_key,
                password=self.secret_key,
            )
            return LakeFSClient(config)  # type: ignore[no-untyped-call]
        except Exception as e:
            raise LakeFSConnectionError(
                f"Failed to connect to LakeFS at {self.endpoint}: {e}"
            ) from e

    def list_repositories(self) -> list[str]:
        """List all repositories.

        Returns:
            List of repository IDs.

        Raises:
            LakeFSConnectionError: Network or connection error.
            LakeFSError: Other LakeFS API errors.
        """
        try:
            client = self.get_client()
            repos = client.repositories_api.list_repositories()
            return [repo.id for repo in repos.results]
        except LakeFSConnectionError:
            raise
        except Exception as e:
            error_msg = str(e).lower()
            if any(pattern in error_msg for pattern in CONNECTION_ERROR_PATTERNS):
                raise LakeFSConnectionError(
                    f"Failed to list repositories: {e}"
                ) from e
            raise LakeFSError(f"Failed to list repositories: {e}") from e

    def get_object(
        self,
        repository: str,
        ref: str,
        path: str,
    ) -> bytes:
        """Get object from LakeFS.

        Args:
            repository: Repository name.
            ref: Branch or commit reference.
            path: Path to the object.

        Returns:
            Object content as bytes.

        Raises:
            LakeFSNotFoundError: Object not found.
            LakeFSConnectionError: Network error.
            LakeFSError: Other LakeFS errors.
        """
        try:
            client = self.get_client()
            return client.objects_api.get_object(
                repository=repository,
                ref=ref,
                path=path,
            )
        except LakeFSConnectionError:
            raise
        except Exception as e:
            error_msg = str(e).lower()
            if any(pattern in error_msg for pattern in NOT_FOUND_PATTERNS):
                raise LakeFSNotFoundError(
                    f"Object not found: {repository}/{ref}/{path}"
                ) from e
            if any(pattern in error_msg for pattern in CONNECTION_ERROR_PATTERNS):
                raise LakeFSConnectionError(
                    f"Failed to get object {path}: {e}"
                ) from e
            raise LakeFSError(f"Failed to get object {path}: {e}") from e

    def put_object(
        self,
        repository: str,
        branch: str,
        path: str,
        content: bytes,
    ) -> None:
        """Upload object to LakeFS.

        Args:
            repository: Repository name.
            branch: Branch name.
            path: Destination path.
            content: Content to upload.

        Raises:
            LakeFSConnectionError: Network error.
            LakeFSError: Other LakeFS errors.
        """
        try:
            client = self.get_client()
            client.objects_api.upload_object(
                repository=repository,
                branch=branch,
                path=path,
                content=content,
            )
        except LakeFSConnectionError:
            raise
        except Exception as e:
            error_msg = str(e).lower()
            if any(pattern in error_msg for pattern in CONNECTION_ERROR_PATTERNS):
                raise LakeFSConnectionError(
                    f"Failed to upload object {path}: {e}"
                ) from e
            raise LakeFSError(f"Failed to upload object {path}: {e}") from e

    def health_check(self) -> bool:
        """Check if LakeFS is accessible.

        Returns:
            True if LakeFS is accessible, False otherwise.
        """
        try:
            client = self.get_client()
            client.repositories_api.list_repositories()
            return True
        except Exception:
            return False


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "LakeFSResource",
    "LakeFSError",
    "LakeFSConnectionError",
    "LakeFSNotFoundError",
]
