"""Integration tests for speech_classification asset with GPT-OSS (TDD - RED phase).

Tests verify the integration between speech_classification asset and
the new generate_classification() method with Harmony token handling.

All HTTP calls are mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import polars as pl
import pytest


class TestSpeechClassificationWithGenerateClassification:
    """Tests for speech_classification asset using generate_classification method."""

    @pytest.fixture
    def nim_resource(self):
        """Create NIM resource for testing."""
        from brev_pipelines.resources.nim import NIMResource

        return NIMResource(
            endpoint="http://test-nim:8000",
            model="openai/gpt-oss-120b",
            use_mock_fallback=False,
        )

    @pytest.fixture
    def mock_minio(self):
        """Create mock MinIO resource."""
        mock = MagicMock()
        mock.bucket_name = "test-bucket"
        # Mock checkpoint operations
        mock.client.list_objects.return_value = []
        return mock

    @pytest.fixture
    def mock_context(self):
        """Create mock Dagster context."""
        context = MagicMock()
        context.run_id = "test-run-123"
        context.log = MagicMock()
        return context

    @pytest.fixture
    def sample_speeches_df(self):
        """Create sample speeches DataFrame for testing."""
        return pl.DataFrame(
            {
                "reference": ["speech_001", "speech_002"],
                "title": ["Fed Rate Decision", "Trade Policy Update"],
                "speaker": ["Jerome Powell", "Janet Yellen"],
                "central_bank": ["Federal Reserve", "US Treasury"],
                "text": [
                    "The Federal Reserve raised interest rates by 25 basis points today...",
                    "We continue to support free trade agreements with our allies...",
                ],
            }
        )

    @pytest.fixture
    def mock_classification_response(self):
        """Mock NIM classification response with Harmony tokens."""
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": '<|channel|>final<|message|>{"monetary_stance": "hawkish", "trade_stance": "neutral", "tariff_mention": 0, "economic_outlook": "positive"}<|return|>',
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        }

    def test_classify_speech_uses_generate_classification(
        self, nim_resource, mock_classification_response
    ):
        """Asset should use generate_classification method."""
        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_classification_response
            mock_post.return_value.raise_for_status = MagicMock()

            result = nim_resource.generate_classification(
                speech_text="The Fed raised rates by 25 basis points..."
            )

            # Verify correct numeric values
            assert result["monetary_stance"] == 4  # hawkish
            assert result["trade_stance"] == 3  # neutral
            assert result["tariff_mention"] == 0
            assert result["economic_outlook"] == 4  # positive

    def test_classify_speech_handles_harmony_tokens_in_response(self, nim_resource):
        """Should correctly parse response with Harmony control tokens."""
        response_with_tokens = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": 'Analysis: Fed raised rates.<|channel|>final<|message|>{"monetary_stance": "very_hawkish", "trade_stance": "protectionist", "tariff_mention": 1, "economic_outlook": "negative"}<|return|>',
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 80, "total_tokens": 180},
        }

        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = response_with_tokens
            mock_post.return_value.raise_for_status = MagicMock()

            result = nim_resource.generate_classification(
                speech_text="The Fed aggressively raised rates..."
            )

            assert result["monetary_stance"] == 5  # very_hawkish
            assert result["tariff_mention"] == 1

    def test_classify_speech_sends_correct_request_format(
        self, nim_resource, mock_classification_response
    ):
        """Should send request with json_object format and include_reasoning: false."""
        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_classification_response
            mock_post.return_value.raise_for_status = MagicMock()

            nim_resource.generate_classification(speech_text="Test speech")

            # Verify request format
            call_args = mock_post.call_args
            request_body = call_args.kwargs["json"]

            # Must use json_object, NOT json_schema (vLLM bug #23120)
            assert request_body["response_format"]["type"] == "json_object"
            # Must include reasoning suppression
            assert request_body.get("include_reasoning") is False

    def test_classify_speech_includes_schema_in_system_prompt(
        self, nim_resource, mock_classification_response
    ):
        """System prompt should include classification schema description."""
        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_classification_response
            mock_post.return_value.raise_for_status = MagicMock()

            nim_resource.generate_classification(speech_text="Test speech")

            call_args = mock_post.call_args
            request_body = call_args.kwargs["json"]
            system_message = request_body["messages"][0]["content"]

            # Schema must include field names and allowed values
            assert "monetary_stance" in system_message
            assert "hawkish" in system_message
            assert "dovish" in system_message
            assert "trade_stance" in system_message

    def test_classify_speech_raises_on_invalid_stance_value(self, nim_resource):
        """Should raise ValidationError when response contains invalid stance."""
        invalid_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": '{"monetary_stance": "super_hawkish", "trade_stance": "neutral", "tariff_mention": 0, "economic_outlook": "positive"}',
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        }

        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = invalid_response
            mock_post.return_value.raise_for_status = MagicMock()

            from pydantic import ValidationError

            with pytest.raises(ValidationError):
                nim_resource.generate_classification(speech_text="Test speech")

    def test_classify_speech_truncates_long_text(self, nim_resource, mock_classification_response):
        """Should truncate speech text to avoid context overflow."""
        long_text = "A" * 20000  # Very long text

        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_classification_response
            mock_post.return_value.raise_for_status = MagicMock()

            nim_resource.generate_classification(speech_text=long_text)

            call_args = mock_post.call_args
            request_body = call_args.kwargs["json"]
            user_message = request_body["messages"][1]["content"]

            # Text should be truncated
            assert len(user_message) < 20000


class TestClassifyWithFallback:
    """Tests for classification with fallback handling."""

    @pytest.fixture
    def nim_resource(self):
        """Create NIM resource with mock fallback enabled."""
        from brev_pipelines.resources.nim import NIMResource

        return NIMResource(
            endpoint="http://test-nim:8000",
            model="openai/gpt-oss-120b",
            use_mock_fallback=True,
        )

    def test_http_500_error_raises_appropriately(self, nim_resource):
        """Should raise HTTPError on 500 server error."""
        import requests

        with patch("requests.post") as mock_post:
            mock_post.return_value.raise_for_status.side_effect = requests.exceptions.HTTPError(
                "500 Server Error"
            )

            with pytest.raises(requests.exceptions.HTTPError):
                nim_resource.generate_classification(speech_text="Test speech")

    def test_http_503_error_raises_appropriately(self, nim_resource):
        """Should raise HTTPError on 503 service unavailable."""
        import requests

        with patch("requests.post") as mock_post:
            mock_post.return_value.raise_for_status.side_effect = requests.exceptions.HTTPError(
                "503 Service Unavailable"
            )

            with pytest.raises(requests.exceptions.HTTPError):
                nim_resource.generate_classification(speech_text="Test speech")


class TestEndToEndClassificationPipeline:
    """End-to-end tests for classification pipeline integration."""

    def test_full_classification_flow_with_harmony_cleanup(self):
        """Test complete flow: raw response -> Harmony cleanup -> validation -> numeric output."""
        from brev_pipelines.utils.harmony import parse_classification_response

        # Simulate GPT-OSS response with all Harmony artifacts
        raw_response = """Let me analyze this speech for you.
<|channel|>final<|message|>{"monetary_stance": "hawkish", "trade_stance": "globalist", "tariff_mention": 0, "economic_outlook": "neutral"}<|return|>"""

        result = parse_classification_response(raw_response)

        # Verify clean numeric output
        assert result["monetary_stance"] == 4  # hawkish
        assert result["trade_stance"] == 4  # globalist
        assert result["tariff_mention"] == 0
        assert result["economic_outlook"] == 3  # neutral

    def test_classification_with_reasoning_prefix(self):
        """Test extraction when LLM includes reasoning before JSON."""
        from brev_pipelines.utils.harmony import parse_classification_response

        response_with_reasoning = """The speech discusses interest rate increases, indicating a hawkish stance on monetary policy. Trade policy is mentioned positively.

{"monetary_stance": "very_hawkish", "trade_stance": "very_globalist", "tariff_mention": 0, "economic_outlook": "very_positive"}"""

        result = parse_classification_response(response_with_reasoning)

        assert result["monetary_stance"] == 5
        assert result["trade_stance"] == 5
        assert result["economic_outlook"] == 5
