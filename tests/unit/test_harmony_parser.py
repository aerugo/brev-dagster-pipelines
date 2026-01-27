"""Tests for Harmony response format parsing (TDD - RED phase).

Tests follow TDD - written BEFORE implementation.
The Harmony format is used by GPT-OSS models.
"""

from __future__ import annotations


class TestHarmonyTokens:
    """Tests for Harmony token constants."""

    def test_harmony_tokens_defined(self):
        """Verify all Harmony tokens are defined."""
        from brev_pipelines.utils.harmony import HARMONY_TOKENS

        assert "<|start|>" in HARMONY_TOKENS
        assert "<|end|>" in HARMONY_TOKENS
        assert "<|return|>" in HARMONY_TOKENS
        assert "<|channel|>" in HARMONY_TOKENS
        assert "<|message|>" in HARMONY_TOKENS

    def test_harmony_token_pattern_defined(self):
        """Verify Harmony token regex pattern is defined."""
        import re

        from brev_pipelines.utils.harmony import HARMONY_TOKEN_PATTERN

        assert HARMONY_TOKEN_PATTERN is not None
        # Should match <|token|> format
        assert re.search(HARMONY_TOKEN_PATTERN, "<|return|>")
        assert re.search(HARMONY_TOKEN_PATTERN, "<|channel|>")
        assert not re.search(HARMONY_TOKEN_PATTERN, "regular text")


class TestRemoveHarmonyTokens:
    """Tests for remove_harmony_tokens function."""

    def test_removes_return_token(self):
        """Should remove <|return|> token."""
        from brev_pipelines.utils.harmony import remove_harmony_tokens

        content = "text<|return|>"
        result = remove_harmony_tokens(content)
        assert "<|return|>" not in result
        assert "text" in result

    def test_removes_channel_tokens(self):
        """Should remove <|channel|> tokens."""
        from brev_pipelines.utils.harmony import remove_harmony_tokens

        content = "<|channel|>final<|message|>text"
        result = remove_harmony_tokens(content)
        assert "<|channel|>" not in result
        assert "<|message|>" not in result
        assert "text" in result

    def test_removes_all_tokens(self):
        """Should remove all Harmony control tokens."""
        from brev_pipelines.utils.harmony import remove_harmony_tokens

        content = "<|start|>assistant<|channel|>final<|message|>text<|end|>"
        result = remove_harmony_tokens(content)
        assert "<|" not in result
        assert "|>" not in result
        assert "text" in result
        # Should preserve 'final' and 'assistant' as they're channel names
        assert "assistant" in result
        assert "final" in result

    def test_preserves_non_token_text(self):
        """Should preserve all non-token text."""
        from brev_pipelines.utils.harmony import remove_harmony_tokens

        content = "Hello <|return|> World"
        result = remove_harmony_tokens(content)
        assert "Hello" in result
        assert "World" in result

    def test_empty_string(self):
        """Should handle empty string."""
        from brev_pipelines.utils.harmony import remove_harmony_tokens

        result = remove_harmony_tokens("")
        assert result == ""


class TestExtractJsonFromHarmony:
    """Tests for extract_json_from_harmony function."""

    def test_clean_json_unchanged(self):
        """Clean JSON without Harmony tokens should be extracted."""
        from brev_pipelines.utils.harmony import extract_json_from_harmony

        content = '{"monetary_stance": "hawkish", "tariff_mention": 0}'
        result = extract_json_from_harmony(content)
        assert result == {"monetary_stance": "hawkish", "tariff_mention": 0}

    def test_removes_return_token(self):
        """Should remove <|return|> token from end."""
        from brev_pipelines.utils.harmony import extract_json_from_harmony

        content = '{"key": "value"}<|return|>'
        result = extract_json_from_harmony(content)
        assert result == {"key": "value"}

    def test_removes_channel_tokens(self):
        """Should remove <|channel|>final pattern."""
        from brev_pipelines.utils.harmony import extract_json_from_harmony

        content = '<|channel|>final<|message|>{"key": "value"}'
        result = extract_json_from_harmony(content)
        assert result == {"key": "value"}

    def test_removes_multiple_tokens(self):
        """Should remove all Harmony tokens."""
        from brev_pipelines.utils.harmony import extract_json_from_harmony

        content = '<|start|>assistant<|channel|>final<|message|>{"key": "value"}<|end|><|return|>'
        result = extract_json_from_harmony(content)
        assert result == {"key": "value"}

    def test_handles_reasoning_prefix(self):
        """Should extract JSON even with reasoning text before it."""
        from brev_pipelines.utils.harmony import extract_json_from_harmony

        content = 'The Fed raised rates so stance is hawkish.\n{"monetary_stance": "hawkish"}'
        result = extract_json_from_harmony(content)
        assert result == {"monetary_stance": "hawkish"}

    def test_handles_reasoning_with_tokens(self):
        """Should handle reasoning text plus Harmony tokens."""
        from brev_pipelines.utils.harmony import extract_json_from_harmony

        content = 'Analysis complete.<|channel|>final<|message|>{"key": "value"}<|return|>'
        result = extract_json_from_harmony(content)
        assert result == {"key": "value"}

    def test_raises_on_no_json(self):
        """Should raise ValueError when no JSON found."""
        import pytest

        from brev_pipelines.utils.harmony import extract_json_from_harmony

        with pytest.raises(ValueError, match="No JSON"):
            extract_json_from_harmony("Just plain text without any JSON")

    def test_raises_on_invalid_json(self):
        """Should raise ValueError on malformed JSON."""
        import pytest

        from brev_pipelines.utils.harmony import extract_json_from_harmony

        with pytest.raises(ValueError, match="Invalid JSON"):
            extract_json_from_harmony('{"key": "value"')  # Missing closing brace

    def test_handles_nested_json(self):
        """Should correctly parse nested JSON objects."""
        from brev_pipelines.utils.harmony import extract_json_from_harmony

        content = '{"outer": {"inner": "value"}, "list": [1, 2, 3]}'
        result = extract_json_from_harmony(content)
        assert result == {"outer": {"inner": "value"}, "list": [1, 2, 3]}

    def test_handles_whitespace(self):
        """Should handle JSON with extra whitespace."""
        from brev_pipelines.utils.harmony import extract_json_from_harmony

        content = '  \n  {"key": "value"}  \n  <|return|>'
        result = extract_json_from_harmony(content)
        assert result == {"key": "value"}

    def test_handles_unicode(self):
        """Should handle JSON with unicode characters."""
        from brev_pipelines.utils.harmony import extract_json_from_harmony

        content = '{"text": "Économie française"}'
        result = extract_json_from_harmony(content)
        assert result == {"text": "Économie française"}

    def test_handles_escaped_quotes(self):
        """Should handle JSON with escaped quotes."""
        from brev_pipelines.utils.harmony import extract_json_from_harmony

        content = '{"text": "He said \\"hello\\""}'
        result = extract_json_from_harmony(content)
        assert result == {"text": 'He said "hello"'}


class TestValidateAndConvert:
    """Tests for validate_and_convert_classification function."""

    def test_valid_classification(self):
        """Valid classification should return validated result."""
        from brev_pipelines.types import GPTOSSClassificationResult
        from brev_pipelines.utils.harmony import validate_and_convert_classification

        raw_json = {
            "monetary_stance": "hawkish",
            "trade_stance": "neutral",
            "tariff_mention": 0,
            "economic_outlook": "positive",
        }
        result = validate_and_convert_classification(raw_json)
        assert isinstance(result, GPTOSSClassificationResult)
        assert result.monetary_stance == "hawkish"

    def test_invalid_stance_raises(self):
        """Invalid stance value should raise ValidationError."""
        import pytest
        from pydantic import ValidationError

        from brev_pipelines.utils.harmony import validate_and_convert_classification

        raw_json = {
            "monetary_stance": "invalid_stance",
            "trade_stance": "neutral",
            "tariff_mention": 0,
            "economic_outlook": "positive",
        }
        with pytest.raises(ValidationError):
            validate_and_convert_classification(raw_json)

    def test_missing_field_raises(self):
        """Missing required field should raise ValidationError."""
        import pytest
        from pydantic import ValidationError

        from brev_pipelines.utils.harmony import validate_and_convert_classification

        raw_json = {
            "monetary_stance": "hawkish",
            # Missing other required fields
        }
        with pytest.raises(ValidationError):
            validate_and_convert_classification(raw_json)


class TestParseClassificationResponse:
    """Tests for parse_classification_response (end-to-end)."""

    def test_parses_clean_response(self):
        """Should parse clean JSON response."""
        from brev_pipelines.utils.harmony import parse_classification_response

        content = '{"monetary_stance": "hawkish", "trade_stance": "neutral", "tariff_mention": 0, "economic_outlook": "positive"}'
        result = parse_classification_response(content)

        assert isinstance(result, dict)
        assert result["monetary_stance"] == 4  # hawkish = 4
        assert result["trade_stance"] == 3  # neutral = 3
        assert result["tariff_mention"] == 0
        assert result["economic_outlook"] == 4  # positive = 4

    def test_parses_response_with_harmony_tokens(self):
        """Should parse response with Harmony tokens."""
        from brev_pipelines.utils.harmony import parse_classification_response

        content = '<|channel|>final<|message|>{"monetary_stance": "dovish", "trade_stance": "globalist", "tariff_mention": 1, "economic_outlook": "negative"}<|return|>'
        result = parse_classification_response(content)

        assert result["monetary_stance"] == 2  # dovish = 2
        assert result["trade_stance"] == 4  # globalist = 4
        assert result["tariff_mention"] == 1
        assert result["economic_outlook"] == 2  # negative = 2

    def test_parses_response_with_reasoning(self):
        """Should parse response with reasoning prefix."""
        from brev_pipelines.utils.harmony import parse_classification_response

        content = 'The speech mentions rate hikes.\n{"monetary_stance": "very_hawkish", "trade_stance": "very_protectionist", "tariff_mention": 1, "economic_outlook": "very_negative"}'
        result = parse_classification_response(content)

        assert result["monetary_stance"] == 5
        assert result["trade_stance"] == 1
        assert result["tariff_mention"] == 1
        assert result["economic_outlook"] == 1
