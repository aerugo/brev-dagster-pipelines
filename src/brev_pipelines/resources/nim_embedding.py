"""NIM Embedding resource for Dagster.

Uses the locally deployed NIM embedding model (nv-embedqa-e5-v5)
to generate text embeddings. The model exposes an OpenAI-compatible API endpoint.

Model: nvidia/nv-embedqa-e5-v5 (1024 dimensions)
"""

from __future__ import annotations

import time
from typing import Literal

import requests
from dagster import ConfigurableResource
from pydantic import Field


class NIMEmbeddingResource(ConfigurableResource):  # type: ignore[type-arg]
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
        default="nvidia/nv-embedqa-e5-v5",
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

    use_mock_fallback: bool = Field(
        default=True,
        description="Use mock embeddings if NIM service unavailable (for local dev)",
    )

    # Cache health check result
    _health_checked: bool = False
    _service_available: bool = False

    def _check_service_once(self) -> bool:
        """Check service availability once and cache the result."""
        if not self._health_checked:
            self._service_available = self.health_check()
            self._health_checked = True
        return self._service_available

    def _generate_mock_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate deterministic mock embeddings based on text hash."""
        import hashlib

        mock_embeddings = []
        for text in texts:
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            embedding = []
            for i in range(1024):
                idx = (i * 2) % 64
                val = int(text_hash[idx : idx + 2], 16) / 255.0 - 0.5
                embedding.append(val)
            mock_embeddings.append(embedding)
        return mock_embeddings

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
            RuntimeError: If all retry attempts fail and mock fallback is disabled.
        """
        # Check service availability first if mock fallback enabled
        if self.use_mock_fallback and not self._check_service_once():
            return self._generate_mock_embeddings(texts)

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
                # Try to extract detailed error message from response
                error_detail = str(e)
                if hasattr(e, "response") and e.response is not None:
                    try:
                        error_body = e.response.text[:500]
                        error_detail = f"{e} - Response: {error_body}"
                    except Exception:
                        pass
                last_error = Exception(error_detail)
                # If mock fallback enabled and connection failed, use mock immediately
                if self.use_mock_fallback:
                    self._service_available = False
                    return self._generate_mock_embeddings(texts)
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)  # Exponential backoff
                continue

        # If service unavailable and mock fallback enabled, return deterministic mock embeddings
        if self.use_mock_fallback:
            return self._generate_mock_embeddings(texts)

        raise RuntimeError(f"NIM embedding error after {self.max_retries} attempts: {last_error}")

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
            Number of dimensions (1024 for nv-embedqa-e5-v5).
        """
        # nv-embedqa-e5-v5 produces 1024-dimensional embeddings
        return 1024
