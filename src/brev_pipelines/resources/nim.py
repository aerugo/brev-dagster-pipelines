"""NVIDIA NIM LLM resource for Dagster."""

import requests
from dagster import ConfigurableResource
from pydantic import Field


class NIMResource(ConfigurableResource):
    """NVIDIA NIM LLM inference resource."""

    endpoint: str = Field(description="NIM endpoint URL")
    model: str = Field(default="meta/llama3-8b-instruct", description="Model name")
    timeout: int = Field(default=30, description="Request timeout in seconds")

    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> str:
        """Generate text using NIM LLM."""
        try:
            response = requests.post(
                f"{self.endpoint}/v1/completions",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["text"].strip()
        except Exception as e:
            return f"LLM error: {e}"

    def health_check(self) -> bool:
        """Check if NIM is healthy."""
        try:
            response = requests.get(f"{self.endpoint}/v1/health/ready", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
