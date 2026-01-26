"""Weaviate client resource for Dagster.

Provides methods for connecting to Weaviate, managing collections,
and performing vector operations. Uses Weaviate Python client v4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import weaviate
from dagster import ConfigurableResource
from pydantic import Field
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import MetadataQuery

from brev_pipelines.types import WeaviatePropertyDef, WeaviateSearchResult

if TYPE_CHECKING:
    from weaviate import WeaviateClient


# =============================================================================
# Exception Types
# =============================================================================


class WeaviateError(Exception):
    """Base exception for Weaviate errors."""

    pass


class WeaviateConnectionError(WeaviateError):
    """Raised when Weaviate connection fails."""

    pass


class WeaviateCollectionError(WeaviateError):
    """Raised when collection operation fails."""

    pass


# Connection-related error patterns
CONNECTION_ERROR_PATTERNS = (
    "connection",
    "timeout",
    "network",
    "refused",
    "unreachable",
    "timed out",
    "grpc",
)


class WeaviateResource(ConfigurableResource):
    """Weaviate vector database resource.

    Attributes:
        host: Weaviate HTTP service hostname.
        port: Weaviate HTTP port.
        grpc_host: Weaviate gRPC service hostname.
        grpc_port: Weaviate gRPC port for vector operations.
    """

    host: str = Field(
        default="weaviate.weaviate.svc.cluster.local",
        description="Weaviate HTTP host",
    )
    port: int = Field(default=80, ge=1, le=65535, description="Weaviate HTTP port")
    grpc_host: str = Field(
        default="weaviate-grpc.weaviate.svc.cluster.local",
        description="Weaviate gRPC host",
    )
    grpc_port: int = Field(default=50051, ge=1, le=65535, description="Weaviate gRPC port")

    def get_client(self) -> WeaviateClient:
        """Get a connected Weaviate client.

        Returns:
            Connected WeaviateClient instance.

        Raises:
            WeaviateConnectionError: Cannot connect to Weaviate.

        Note:
            Caller is responsible for closing the client when done.
        """
        try:
            client = weaviate.connect_to_custom(
                http_host=self.host,
                http_port=self.port,
                http_secure=False,
                grpc_host=self.grpc_host,
                grpc_port=self.grpc_port,
                grpc_secure=False,
            )
            return client
        except Exception as e:
            raise WeaviateConnectionError(
                f"Failed to connect to Weaviate at {self.host}:{self.port}: {e}"
            ) from e

    def ensure_collection(
        self,
        name: str,
        properties: list[WeaviatePropertyDef],
        vector_dimensions: int = 1024,  # noqa: ARG002
    ) -> None:
        """Ensure a collection exists with the given schema.

        Creates the collection if it doesn't exist. Does nothing if it already exists.

        Args:
            name: Collection name (PascalCase recommended).
            properties: List of WeaviatePropertyDef with 'name' and optional 'type', 'description'.
            vector_dimensions: Dimension of vectors to store (reserved for future use).

        Raises:
            WeaviateConnectionError: Cannot connect to Weaviate.
            WeaviateCollectionError: Failed to create collection.
        """
        try:
            client = self.get_client()
        except WeaviateConnectionError:
            raise

        try:
            if client.collections.exists(name):
                return

            # Build property definitions
            props: list[Property] = []
            for prop in properties:
                data_type = DataType.TEXT
                prop_type = prop.get("type", "text")
                if prop_type == "date":
                    data_type = DataType.DATE
                elif prop_type == "boolean":
                    data_type = DataType.BOOL
                elif prop_type == "int":
                    data_type = DataType.INT
                elif prop_type == "number":
                    data_type = DataType.NUMBER

                props.append(
                    Property(
                        name=prop["name"],
                        data_type=data_type,
                        description=prop.get("description", ""),
                    )
                )

            # Create collection with no vectorizer (we provide embeddings externally)
            client.collections.create(
                name=name,
                vectorizer_config=Configure.Vectorizer.none(),
                properties=props,
            )
        except WeaviateConnectionError:
            raise
        except Exception as e:
            raise WeaviateCollectionError(f"Failed to ensure collection {name}: {e}") from e
        finally:
            client.close()

    def insert_objects(
        self,
        collection_name: str,
        objects: list[dict[str, Any]],
        vectors: list[list[float]],
        batch_size: int = 100,  # noqa: ARG002
    ) -> int:
        """Insert objects with their vectors into a collection.

        Args:
            collection_name: Target collection name.
            objects: List of property dictionaries.
            vectors: Corresponding embedding vectors.
            batch_size: Objects per batch insert (reserved for future use).

        Returns:
            Number of objects inserted.

        Raises:
            ValueError: If objects and vectors have different lengths.
            WeaviateConnectionError: Cannot connect to Weaviate.
            WeaviateCollectionError: Failed to insert objects.
        """
        if len(objects) != len(vectors):
            msg = f"Objects ({len(objects)}) and vectors ({len(vectors)}) must have same length"
            raise ValueError(msg)

        try:
            client = self.get_client()
        except WeaviateConnectionError:
            raise

        try:
            collection = client.collections.get(collection_name)

            with collection.batch.dynamic() as batch:
                for obj, vector in zip(objects, vectors, strict=True):
                    batch.add_object(properties=obj, vector=vector)

            return len(objects)
        except WeaviateConnectionError:
            raise
        except Exception as e:
            raise WeaviateCollectionError(
                f"Failed to insert objects into {collection_name}: {e}"
            ) from e
        finally:
            client.close()

    def vector_search(
        self,
        collection_name: str,
        query_vector: list[float],
        limit: int = 10,
        return_properties: list[str] | None = None,  # noqa: ARG002
    ) -> list[WeaviateSearchResult]:
        """Perform vector similarity search.

        Args:
            collection_name: Collection to search.
            query_vector: Query embedding vector.
            limit: Maximum results to return.
            return_properties: Properties to include in results (reserved for future use).

        Returns:
            List of WeaviateSearchResult with properties, _distance, and _certainty.

        Raises:
            WeaviateConnectionError: Cannot connect to Weaviate.
            WeaviateCollectionError: Failed to perform search.
        """
        try:
            client = self.get_client()
        except WeaviateConnectionError:
            raise

        try:
            collection = client.collections.get(collection_name)

            results = collection.query.near_vector(
                near_vector=query_vector,
                limit=limit,
                return_metadata=MetadataQuery(distance=True, certainty=True),
            )

            output: list[WeaviateSearchResult] = []
            for obj in results.objects:
                result: WeaviateSearchResult = {
                    "properties": dict(obj.properties),  # type: ignore[arg-type]
                    "_distance": obj.metadata.distance or 0.0,
                    "_certainty": obj.metadata.certainty or 0.0,
                }
                output.append(result)

            return output
        except WeaviateConnectionError:
            raise
        except Exception as e:
            raise WeaviateCollectionError(
                f"Failed to search collection {collection_name}: {e}"
            ) from e
        finally:
            client.close()

    def get_object_count(self, collection_name: str) -> int:
        """Get the number of objects in a collection.

        Args:
            collection_name: Name of the collection.

        Returns:
            Number of objects in the collection.

        Raises:
            WeaviateConnectionError: Cannot connect to Weaviate.
            WeaviateCollectionError: Failed to get object count.
        """
        try:
            client = self.get_client()
        except WeaviateConnectionError:
            raise

        try:
            collection = client.collections.get(collection_name)
            response = collection.aggregate.over_all(total_count=True)
            return response.total_count or 0
        except WeaviateConnectionError:
            raise
        except Exception as e:
            raise WeaviateCollectionError(
                f"Failed to get object count for {collection_name}: {e}"
            ) from e
        finally:
            client.close()

    def health_check(self) -> bool:
        """Check if Weaviate is healthy.

        Returns:
            True if Weaviate is ready, False otherwise.
        """
        try:
            client = self.get_client()
            is_ready = client.is_ready()
            client.close()
            return is_ready
        except Exception:
            return False

    def delete_collection(self, name: str) -> bool:
        """Delete a collection if it exists.

        Args:
            name: Name of the collection to delete.

        Returns:
            True if collection was deleted, False if it didn't exist.

        Raises:
            WeaviateConnectionError: Cannot connect to Weaviate.
            WeaviateCollectionError: Failed to delete collection.
        """
        try:
            client = self.get_client()
        except WeaviateConnectionError:
            raise

        try:
            if client.collections.exists(name):
                client.collections.delete(name)
                return True
            return False
        except WeaviateConnectionError:
            raise
        except Exception as e:
            raise WeaviateCollectionError(f"Failed to delete collection {name}: {e}") from e
        finally:
            client.close()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "WeaviateResource",
    "WeaviateError",
    "WeaviateConnectionError",
    "WeaviateCollectionError",
]
