"""MinIO resource for Dagster."""

from dagster import ConfigurableResource
from minio import Minio
from pydantic import Field


class MinIOResource(ConfigurableResource):
    """MinIO S3-compatible storage resource."""

    endpoint: str = Field(description="MinIO endpoint (host:port)")
    access_key: str = Field(description="Access key")
    secret_key: str = Field(description="Secret key")
    secure: bool = Field(default=False, description="Use HTTPS")

    def get_client(self) -> Minio:
        """Get MinIO client instance."""
        return Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure,
        )

    def ensure_bucket(self, bucket: str) -> None:
        """Ensure bucket exists, create if not."""
        client = self.get_client()
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
