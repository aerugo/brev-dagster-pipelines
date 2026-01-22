"""Weaviate I/O Manager for vector storage.

Handles storing and retrieving objects with their embeddings in Weaviate.
Follows NEW INV-D004 (Weaviate Collections for Vector Data).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from dagster import ConfigurableIOManager, InputContext, OutputContext
from pydantic import Field

if TYPE_CHECKING:
    from brev_pipelines.resources.weaviate import WeaviateResource


class WeaviateIOManager(ConfigurableIOManager):
    """I/O Manager for storing text and embeddings in Weaviate.

    Expects output to be a tuple of (DataFrame, embeddings_list).
    Creates or updates the collection and inserts objects with vectors.

    Attributes:
        weaviate: Weaviate resource for vector operations.
        collection_prefix: Prefix for collection names.
    """

    weaviate: WeaviateResource = Field(description="Weaviate resource")
    collection_prefix: str = Field(
        default="",
        description="Prefix for collection names",
    )

    def handle_output(
        self,
        context: OutputContext,
        obj: tuple[pl.DataFrame, list[list[float]]],
    ) -> None:
        """Store DataFrame rows with embeddings to Weaviate.

        Args:
            context: Dagster output context.
            obj: Tuple of (DataFrame, list of embedding vectors).

        Raises:
            ValueError: If DataFrame and embeddings have different lengths.
        """
        if obj is None:
            context.log.warning("Received None object, skipping output")
            return

        df, embeddings = obj

        if len(df) != len(embeddings):
            msg = f"DataFrame has {len(df)} rows but got {len(embeddings)} embeddings"
            raise ValueError(msg)

        # Determine collection name from asset key
        asset_key = context.asset_key.path[-1] if context.asset_key else "default"
        collection_name = f"{self.collection_prefix}{asset_key}".replace("_", "")

        # Convert to PascalCase for Weaviate
        collection_name = "".join(word.title() for word in collection_name.split())

        # Define schema from DataFrame columns
        properties = []
        for col in df.columns:
            dtype = df[col].dtype
            prop_type = "text"
            if dtype == pl.Date:
                prop_type = "date"
            elif dtype == pl.Boolean:
                prop_type = "boolean"
            elif dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64):
                prop_type = "int"
            elif dtype in (pl.Float32, pl.Float64):
                prop_type = "number"

            properties.append({"name": col, "type": prop_type})

        # Ensure collection exists
        vector_dims = len(embeddings[0]) if embeddings else 1024
        self.weaviate.ensure_collection(
            name=collection_name,
            properties=properties,
            vector_dimensions=vector_dims,
        )

        # Convert DataFrame to list of dicts
        objects = df.to_dicts()

        # Insert objects with embeddings
        count = self.weaviate.insert_objects(
            collection_name=collection_name,
            objects=objects,
            vectors=embeddings,
        )

        context.log.info(f"Inserted {count} objects into Weaviate collection {collection_name}")

    def load_input(self, context: InputContext) -> int:
        """Return object count from Weaviate collection.

        Weaviate doesn't support efficient full retrieval, so we return
        the count instead. For actual data, use vector_search methods.

        Args:
            context: Dagster input context.

        Returns:
            Number of objects in the collection.
        """
        asset_key = context.asset_key.path[-1] if context.asset_key else "default"
        collection_name = f"{self.collection_prefix}{asset_key}".replace("_", "")
        collection_name = "".join(word.title() for word in collection_name.split())

        return self.weaviate.get_object_count(collection_name)
