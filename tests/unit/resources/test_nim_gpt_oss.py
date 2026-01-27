"""Tests for NIM GPT-OSS JSON generation (TDD - RED phase).

Tests follow TDD - written BEFORE implementation.
All HTTP calls are mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestNIMResourceGenerateJson:
    """Tests for NIMResource.generate_json() method."""

    @pytest.fixture
    def nim_resource(self):
        """Create NIM resource for testing."""
        from brev_pipelines.resources.nim import NIMResource

        return NIMResource(
            endpoint="http://test-nim:8000",
            model="openai/gpt-oss-120b",
        )

    @pytest.fixture
    def mock_classification_response(self):
        """Mock NIM classification response with valid JSON."""
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": '{"monetary_stance": "hawkish", "trade_stance": "neutral", "tariff_mention": 0, "economic_outlook": "positive"}',
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        }

    @pytest.fixture
    def mock_response_with_harmony_tokens(self):
        """Mock NIM response with Harmony tokens in content."""
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": 'Analysis complete.<|channel|>final<|message|>{"monetary_stance": "hawkish", "trade_stance": "neutral", "tariff_mention": 0, "economic_outlook": "positive"}<|return|>',
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 80, "total_tokens": 180},
        }

    @pytest.fixture
    def mock_malformed_response(self):
        """Mock NIM response with malformed JSON."""
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": '{"monetary_stance": "hawkish", "trade_stance": "neutral"',  # Truncated
                    },
                    "finish_reason": "length",
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 500, "total_tokens": 600},
        }

    def test_generate_json_returns_dict(self, nim_resource, mock_classification_response):
        """generate_json should return parsed dictionary."""
        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_classification_response
            mock_post.return_value.raise_for_status = MagicMock()

            result = nim_resource.generate_json(
                prompt="Classify this speech",
                system_prompt="You are a classifier",
                schema_description="Return JSON with monetary_stance",
            )

            assert isinstance(result, dict)
            assert "monetary_stance" in result

    def test_generate_json_uses_json_object_format(
        self, nim_resource, mock_classification_response
    ):
        """Should use json_object response format, NOT json_schema."""
        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_classification_response
            mock_post.return_value.raise_for_status = MagicMock()

            nim_resource.generate_json(
                prompt="Test",
                system_prompt="Test",
                schema_description="Test",
            )

            # Verify the request used json_object, not json_schema
            call_args = mock_post.call_args
            request_body = call_args.kwargs["json"]
            assert request_body["response_format"]["type"] == "json_object"

    def test_generate_json_includes_reasoning_false(
        self, nim_resource, mock_classification_response
    ):
        """Should include include_reasoning: false in request."""
        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_classification_response
            mock_post.return_value.raise_for_status = MagicMock()

            nim_resource.generate_json(
                prompt="Test",
                system_prompt="Test",
                schema_description="Test",
            )

            call_args = mock_post.call_args
            request_body = call_args.kwargs["json"]
            assert request_body.get("include_reasoning") is False

    def test_generate_json_handles_harmony_tokens(
        self, nim_resource, mock_response_with_harmony_tokens
    ):
        """Should clean Harmony tokens from response."""
        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_response_with_harmony_tokens
            mock_post.return_value.raise_for_status = MagicMock()

            result = nim_resource.generate_json(
                prompt="Test",
                system_prompt="Test",
                schema_description="Test",
            )

            # Should have extracted clean JSON despite tokens
            assert "monetary_stance" in result
            assert "<|return|>" not in str(result)

    def test_generate_json_includes_schema_in_system_prompt(
        self, nim_resource, mock_classification_response
    ):
        """Schema description should be included in system prompt."""
        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_classification_response
            mock_post.return_value.raise_for_status = MagicMock()

            nim_resource.generate_json(
                prompt="Classify this",
                system_prompt="You are a classifier",
                schema_description="Return JSON with monetary_stance field",
            )

            call_args = mock_post.call_args
            request_body = call_args.kwargs["json"]
            system_message = request_body["messages"][0]["content"]
            assert "monetary_stance" in system_message

    def test_generate_json_uses_low_reasoning(self, nim_resource, mock_classification_response):
        """Should include 'Reasoning: low' in system prompt."""
        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_classification_response
            mock_post.return_value.raise_for_status = MagicMock()

            nim_resource.generate_json(
                prompt="Test",
                system_prompt="Test",
                schema_description="Test",
            )

            call_args = mock_post.call_args
            request_body = call_args.kwargs["json"]
            system_message = request_body["messages"][0]["content"]
            assert "Reasoning: low" in system_message

    def test_generate_json_raises_on_malformed_response(
        self, nim_resource, mock_malformed_response
    ):
        """Should raise ValueError on malformed JSON response."""
        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_malformed_response
            mock_post.return_value.raise_for_status = MagicMock()

            with pytest.raises(ValueError, match="Invalid JSON|No JSON"):
                nim_resource.generate_json(
                    prompt="Test",
                    system_prompt="Test",
                    schema_description="Test",
                )

    def test_generate_json_raises_on_http_error(self, nim_resource):
        """Should raise on HTTP error."""
        import requests

        with patch("requests.post") as mock_post:
            mock_post.return_value.raise_for_status.side_effect = requests.exceptions.HTTPError(
                "503 Service Unavailable"
            )

            with pytest.raises(requests.exceptions.HTTPError):
                nim_resource.generate_json(
                    prompt="Test",
                    system_prompt="Test",
                    schema_description="Test",
                )

    def test_generate_json_respects_max_tokens(self, nim_resource, mock_classification_response):
        """Should pass max_tokens parameter."""
        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_classification_response
            mock_post.return_value.raise_for_status = MagicMock()

            nim_resource.generate_json(
                prompt="Test",
                system_prompt="Test",
                schema_description="Test",
                max_tokens=250,
            )

            call_args = mock_post.call_args
            request_body = call_args.kwargs["json"]
            assert request_body["max_tokens"] == 250

    def test_generate_json_respects_temperature(self, nim_resource, mock_classification_response):
        """Should pass temperature parameter."""
        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_classification_response
            mock_post.return_value.raise_for_status = MagicMock()

            nim_resource.generate_json(
                prompt="Test",
                system_prompt="Test",
                schema_description="Test",
                temperature=0.1,
            )

            call_args = mock_post.call_args
            request_body = call_args.kwargs["json"]
            assert request_body["temperature"] == 0.1


class TestNIMResourceGenerateClassification:
    """Tests for NIMResource.generate_classification() method."""

    @pytest.fixture
    def nim_resource(self):
        """Create NIM resource for testing."""
        from brev_pipelines.resources.nim import NIMResource

        return NIMResource(
            endpoint="http://test-nim:8000",
            model="openai/gpt-oss-120b",
        )

    @pytest.fixture
    def mock_classification_response(self):
        """Mock NIM classification response."""
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": '{"monetary_stance": "hawkish", "trade_stance": "neutral", "tariff_mention": 0, "economic_outlook": "positive"}',
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        }

    def test_generate_classification_returns_numeric_dict(
        self, nim_resource, mock_classification_response
    ):
        """Should return SpeechClassification with numeric values."""
        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_classification_response
            mock_post.return_value.raise_for_status = MagicMock()

            result = nim_resource.generate_classification(
                speech_text="The Fed raised rates...",
            )

            assert isinstance(result, dict)
            assert result["monetary_stance"] == 4  # hawkish = 4
            assert result["trade_stance"] == 3  # neutral = 3
            assert result["tariff_mention"] == 0
            assert result["economic_outlook"] == 4  # positive = 4

    def test_generate_classification_uses_schema(self, nim_resource, mock_classification_response):
        """Should use GPT_OSS_CLASSIFICATION_SCHEMA in prompt."""
        with patch("requests.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json.return_value = mock_classification_response
            mock_post.return_value.raise_for_status = MagicMock()

            nim_resource.generate_classification(
                speech_text="Test speech",
            )

            call_args = mock_post.call_args
            request_body = call_args.kwargs["json"]
            system_message = request_body["messages"][0]["content"]
            # Should contain schema elements
            assert "monetary_stance" in system_message
            assert "hawkish" in system_message

    def test_generate_classification_raises_on_invalid_response(self, nim_resource):
        """Should raise ValidationError on invalid classification values."""
        from pydantic import ValidationError

        invalid_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": '{"monetary_stance": "invalid_value", "trade_stance": "neutral", "tariff_mention": 0, "economic_outlook": "positive"}',
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

            with pytest.raises(ValidationError):
                nim_resource.generate_classification(
                    speech_text="Test speech",
                )
