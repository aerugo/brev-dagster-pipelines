"""NVIDIA NIM LLM resource for Dagster."""

import requests
from dagster import ConfigurableResource
from pydantic import Field


# =============================================================================
# Exception Types
# =============================================================================


class NIMError(Exception):
    """Base exception for NIM errors."""

    pass


class NIMTimeoutError(NIMError):
    """Raised when NIM request times out."""

    pass


class NIMServerError(NIMError):
    """Raised when NIM returns 5xx error."""

    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        super().__init__(f"NIM server error {status_code}: {message}")


class NIMRateLimitError(NIMError):
    """Raised when NIM returns 429 rate limit error."""

    pass


# =============================================================================
# Resource
# =============================================================================


class NIMResource(ConfigurableResource):  # type: ignore[type-arg]
    """NVIDIA NIM LLM inference resource.

    Supports both small models (Llama 8B) and large models (GPT-OSS 120B).
    Timeout should be increased for larger models that take longer to generate.
    """

    endpoint: str = Field(description="NIM endpoint URL")
    model: str = Field(default="meta/llama-3.1-8b-instruct", description="Model name")
    timeout: int = Field(
        default=180, description="Request timeout in seconds (increase for large models)"
    )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        timeout_override: int | None = None,
    ) -> str:
        """Generate text using NIM LLM.

        Args:
            prompt: The input prompt for generation.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0.0-1.0).
            timeout_override: Optional timeout override for this specific request.

        Returns:
            Generated text.

        Raises:
            NIMTimeoutError: Request timed out.
            NIMServerError: Server returned 5xx error.
            NIMRateLimitError: Server returned 429 rate limit.
            NIMError: Other NIM-related errors.
        """
        effective_timeout = timeout_override or self.timeout

        try:
            response = requests.post(
                f"{self.endpoint}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=effective_timeout,
            )
        except requests.exceptions.Timeout as e:
            raise NIMTimeoutError(
                f"NIM request timed out after {effective_timeout}s: {e}"
            ) from e
        except requests.exceptions.ConnectionError as e:
            raise NIMError(f"NIM connection failed: {e}") from e
        except requests.exceptions.RequestException as e:
            raise NIMError(f"NIM request failed: {e}") from e

        # Handle HTTP errors
        if response.status_code == 429:
            raise NIMRateLimitError(f"NIM rate limited: {response.text}")

        if response.status_code >= 500:
            raise NIMServerError(response.status_code, response.text)

        if response.status_code != 200:
            raise NIMError(f"NIM error {response.status_code}: {response.text}")

        data = response.json()
        content: str = data["choices"][0]["message"]["content"]
        return content.strip()

    def generate_batch(
        self,
        prompts: list[str],
        max_tokens: int = 100,
        temperature: float = 0.7,
    ) -> list[str]:
        """Generate text for multiple prompts sequentially.

        For large models like GPT-OSS 120B, batch processing is done sequentially
        to avoid overwhelming the GPU. Consider using parallel workers for
        higher throughput.

        Args:
            prompts: List of input prompts.
            max_tokens: Maximum tokens per generation.
            temperature: Sampling temperature.

        Returns:
            List of generated texts (or error messages).
        """
        results = []
        for prompt in prompts:
            result = self.generate(prompt, max_tokens, temperature)
            results.append(result)
        return results

    def health_check(self) -> bool:
        """Check if NIM is healthy."""
        try:
            response = requests.get(f"{self.endpoint}/v1/health/ready", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
