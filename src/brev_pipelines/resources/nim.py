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
        """Initialize with status code and message."""
        self.status_code = status_code
        super().__init__(f"NIM server error {status_code}: {message}")


class NIMRateLimitError(NIMError):
    """Raised when NIM returns 429 rate limit error."""

    pass


# =============================================================================
# Resource
# =============================================================================


class NIMServiceUnavailableError(NIMError):
    """Raised when NIM service is unavailable and mock fallback should be used.

    This error signals the retry wrapper to skip retries and use fallback immediately.
    """

    pass


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
    use_mock_fallback: bool = Field(
        default=True,
        description="Skip retries and use fallback immediately if service unavailable (for local dev)",
    )

    # Cache health check result to avoid repeated calls
    _health_check_cache: bool | None = None
    _health_check_time: float = 0.0

    def _is_service_available(self) -> bool:
        """Check if service is available with caching.

        Caches the result for 30 seconds to avoid repeated health checks.
        """
        import time

        current_time = time.time()

        # Return cached result if still valid (30 second TTL)
        if self._health_check_cache is not None and current_time - self._health_check_time < 30:
            return self._health_check_cache

        # Perform health check
        self._health_check_cache = self.health_check()
        self._health_check_time = current_time
        return self._health_check_cache

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2000,  # High default to avoid truncation
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
            NIMServiceUnavailableError: Service unavailable and mock fallback enabled.
            NIMTimeoutError: Request timed out.
            NIMServerError: Server returned 5xx error.
            NIMRateLimitError: Server returned 429 rate limit.
            NIMError: Other NIM-related errors.
        """
        # Check service availability first if mock fallback enabled
        if self.use_mock_fallback and not self._is_service_available():
            raise NIMServiceUnavailableError(
                f"NIM service at {self.endpoint} is unavailable. "
                "Using fallback values (mock mode enabled)."
            )

        effective_timeout = timeout_override or self.timeout

        try:
            response = requests.post(
                f"{self.endpoint}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "include_reasoning": False,  # Suppress reasoning tokens in output
                },
                timeout=effective_timeout,
            )
        except requests.exceptions.Timeout as e:
            raise NIMTimeoutError(f"NIM request timed out after {effective_timeout}s: {e}") from e
        except requests.exceptions.ConnectionError as e:
            # If mock fallback enabled, convert connection error to service unavailable
            if self.use_mock_fallback:
                # Update cache to prevent further attempts
                self._health_check_cache = False
                self._health_check_time = __import__("time").time()
                raise NIMServiceUnavailableError(
                    f"NIM connection failed: {e}. Using fallback values."
                ) from e
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
        max_tokens: int = 2000,  # High default to avoid truncation
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

    # =========================================================================
    # GPT-OSS Structured Output Methods
    # =========================================================================

    def generate_json(
        self,
        prompt: str,
        system_prompt: str,
        schema_description: str,
        max_tokens: int = 2000,  # High default to avoid truncation
        temperature: float = 0.1,
    ) -> dict[str, object]:
        """Generate structured JSON output from GPT-OSS.

        Uses json_object mode (NOT json_schema) to avoid vLLM bug #23120.
        Includes reasoning suppression and Harmony token cleanup.

        Per INV-N010 and INV-N011:
        - Uses json_object response format (not json_schema)
        - Includes include_reasoning: false

        Args:
            prompt: User prompt with content to analyze.
            system_prompt: Base system instructions.
            schema_description: Description of expected JSON schema.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (lower = more deterministic).

        Returns:
            Parsed JSON dictionary.

        Raises:
            ValueError: If response cannot be parsed as JSON.
            requests.HTTPError: If NIM API returns an error.

        Example:
            >>> result = nim.generate_json(
            ...     prompt="Classify: The Fed raised rates",
            ...     system_prompt="You are a classifier",
            ...     schema_description="Return JSON with monetary_stance field",
            ... )
            >>> print(result)
            {"monetary_stance": "hawkish", ...}
        """
        from brev_pipelines.utils.harmony import extract_json_from_harmony

        # Build system prompt with reasoning level and schema
        full_system_prompt = f"""Reasoning: low
{system_prompt}

{schema_description}

Return ONLY valid JSON, no other text."""

        # Build request payload - use json_object, NOT json_schema
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": full_system_prompt},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
            "include_reasoning": False,
        }

        # Make API call
        response = requests.post(
            f"{self.endpoint}/v1/chat/completions",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()

        # Extract content from response
        result = response.json()
        content: str = result["choices"][0]["message"]["content"]

        # Parse JSON, handling any residual Harmony tokens
        return extract_json_from_harmony(content)

    def generate_classification(
        self,
        speech_text: str,
        max_tokens: int = 2000,  # High token budget to ensure complete JSON output
        temperature: float = 0.1,
    ) -> dict[str, int]:
        """Generate speech classification using GPT-OSS.

        Specialized method for central bank speech classification.
        Returns numeric values (1-5 scale) for each classification dimension.

        Args:
            speech_text: Text of the speech to classify.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            SpeechClassification TypedDict with numeric values:
            - monetary_stance: 1 (very_dovish) to 5 (very_hawkish)
            - trade_stance: 1 (very_protectionist) to 5 (very_globalist)
            - tariff_mention: 0 or 1
            - economic_outlook: 1 (very_negative) to 5 (very_positive)

        Raises:
            ValueError: If JSON cannot be extracted.
            ValidationError: If response doesn't match classification schema.

        Example:
            >>> result = nim.generate_classification("The Fed raised rates...")
            >>> result["monetary_stance"]
            4
        """
        from brev_pipelines.types import GPT_OSS_CLASSIFICATION_SCHEMA

        # Truncate speech to avoid context overflow
        truncated_text = speech_text[:8000]

        # Make the API call
        raw_json = self.generate_json(
            prompt=f"Analyze this central bank speech:\n\n{truncated_text}",
            system_prompt="You are a central bank speech classifier. Analyze the speech and classify it along multiple dimensions.",
            schema_description=GPT_OSS_CLASSIFICATION_SCHEMA,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # The raw_json is already extracted, now validate and convert
        from brev_pipelines.types import (
            GPT_OSS_MONETARY_SCALE,
            GPT_OSS_OUTLOOK_SCALE,
            GPT_OSS_TRADE_SCALE,
        )
        from brev_pipelines.utils.harmony import validate_and_convert_classification

        # Validate and convert
        validated = validate_and_convert_classification(raw_json)

        # Return numeric values
        return {
            "monetary_stance": GPT_OSS_MONETARY_SCALE[validated.monetary_stance],
            "trade_stance": GPT_OSS_TRADE_SCALE[validated.trade_stance],
            "tariff_mention": validated.tariff_mention,
            "economic_outlook": GPT_OSS_OUTLOOK_SCALE[validated.economic_outlook],
        }
