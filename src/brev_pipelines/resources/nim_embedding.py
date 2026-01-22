"""NIM Embedding resource for Dagster.

Uses the locally deployed NIM embedding model (llama-3_2-nemoretriever-300m-embed-v2)
to generate text embeddings. The model exposes an OpenAI-compatible API endpoint.

Model: llama-3_2-nemoretriever-300m-embed-v2 (1024 dimensions)
"""

from __future__ import annotations

import time
from typing import Literal

import requests
from dagster import ConfigurableResource
from pydantic import Field


class NIMEmbeddingResource(ConfigurableResource):
    """NIM embedding resource for generating text embeddings via local NIM.

    Attributes:
        endpoint: NIM embedding service endpoint URL.
        model: Embedding model name.
        timeout: Request timeout in seconds.
        max_retries: Maximum retry attempts for failed requests.
    """

    endpoint: str = Field(
        default="http://nvidia-nim-embedding.nvidia-nim.svc.cluster.local:8000",
        description="NIM embedding service endpoint",
    )
    model: str = Field(
        default="nvidia/llama-3_2-nemoretriever-300m-embed-v2",
        description="Embedding model name",
    )
    timeout: int = Field(default=120, ge=1, le=600, description="Request timeout in seconds")
    max_retries: int = Field(default=3, ge=1, le=10, description="Maximum retry attempts")

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
        input_type: Literal["passage", "query"] = "passage",
    ) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.
            batch_size: Number of texts to embed per API call.
            input_type: Type of input - "passage" for documents, "query" for search queries.

        Returns:
            List of embedding vectors (1024 dimensions each).

        Raises:
            RuntimeError: If embedding generation fails after all retries.
        """
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = self._embed_batch(batch, input_type)
            all_embeddings.extend(embeddings)

        return all_embeddings

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text string to embed.

        Returns:
            Embedding vector (1024 dimensions).
        """
        return self._embed_batch([text], "passage")[0]

    def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a search query.

        Uses 'query' input_type optimized for retrieval.

        Args:
            query: Query text to embed.

        Returns:
            Embedding vector (1024 dimensions).
        """
        return self._embed_batch([query], "query")[0]

    def _embed_batch(
        self,
        texts: list[str],
        input_type: Literal["passage", "query"],
    ) -> list[list[float]]:
        """Embed a batch of texts with retry logic.

        Args:
            texts: List of texts to embed.
            input_type: Type of input for optimization.

        Returns:
            List of embedding vectors.

        Raises:
            RuntimeError: If all retry attempts fail.
        """
        # Truncate long texts (model has input limit)
        truncated_texts = [text[:8192] if len(text) > 8192 else text for text in texts]

        payload = {
            "model": self.model,
            "input": truncated_texts,
            "input_type": input_type,
            "encoding_format": "float",
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.endpoint}/v1/embeddings",
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()

                # Sort by index to maintain order
                sorted_data = sorted(data["data"], key=lambda x: x["index"])
                return [item["embedding"] for item in sorted_data]

            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)  # Exponential backoff
                continue

        raise RuntimeError(
            f"NIM embedding error after {self.max_retries} attempts: {last_error}"
        )

    def health_check(self) -> bool:
        """Check if the NIM embedding service is healthy.

        Returns:
            True if service is ready, False otherwise.
        """
        try:
            response = requests.get(
                f"{self.endpoint}/v1/health/ready",
                timeout=10,
            )
            return response.status_code == 200
        except Exception:
            return False

    @property
    def dimensions(self) -> int:
        """Return the embedding dimensions for the configured model.

        Returns:
            Number of dimensions (1024 for llama-3_2-nemoretriever-300m-embed-v2).
        """
        # llama-3_2-nemoretriever-300m-embed-v2 produces 1024-dimensional embeddings
        return 1024
