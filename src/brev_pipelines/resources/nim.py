"""NVIDIA NIM LLM resource for Dagster."""

import requests
from dagster import ConfigurableResource
from pydantic import Field


class NIMResource(ConfigurableResource):
    """NVIDIA NIM LLM inference resource.

    Supports both small models (Llama 8B) and large models (GPT-OSS 120B).
    Timeout should be increased for larger models that take longer to generate.
    """

    endpoint: str = Field(description="NIM endpoint URL")
    model: str = Field(default="meta/llama-3.1-8b-instruct", description="Model name")
    timeout: int = Field(default=180, description="Request timeout in seconds (increase for large models)")

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
            Generated text, or error message if generation fails.
        """
        try:
            response = requests.post(
                f"{self.endpoint}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=timeout_override or self.timeout,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.Timeout:
            return f"LLM error: Request timed out after {timeout_override or self.timeout}s"
        except Exception as e:
            return f"LLM error: {e}"

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
