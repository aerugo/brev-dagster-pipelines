"""Tests for GPT-OSS type definitions (TDD - RED phase).

These tests verify the type definitions for GPT-OSS structured output.
Written BEFORE implementation per INV-P010.
"""

from __future__ import annotations

import pytest


class TestGPTOSSResponseFormat:
    """Tests for GPT-OSS response format TypedDict."""

    def test_nim_response_format_importable(self) -> None:
        """Verify NIMResponseFormat TypedDict is importable."""
        from brev_pipelines.types import NIMResponseFormat

        fmt: NIMResponseFormat = {"type": "json_object"}
        assert fmt["type"] == "json_object"

    def test_nim_response_format_with_schema(self) -> None:
        """Verify NIMResponseFormat can include json_schema."""
        from brev_pipelines.types import NIMResponseFormat

        fmt: NIMResponseFormat = {
            "type": "json_schema",
            "json_schema": {"name": "test", "schema": {}},
        }
        assert fmt["type"] == "json_schema"


class TestSpeechClassificationPydantic:
    """Tests for SpeechClassificationResult Pydantic model."""

    def test_valid_classification_result(self) -> None:
        """Valid classification result should pass validation."""
        from brev_pipelines.types import GPTOSSClassificationResult

        result = GPTOSSClassificationResult(
            monetary_stance="hawkish",
            trade_stance="neutral",
            tariff_mention=0,
            economic_outlook="positive",
        )
        assert result.monetary_stance == "hawkish"
        assert result.tariff_mention == 0

    def test_invalid_monetary_stance_raises(self) -> None:
        """Invalid monetary stance should raise ValidationError."""
        from pydantic import ValidationError

        from brev_pipelines.types import GPTOSSClassificationResult

        with pytest.raises(ValidationError):
            GPTOSSClassificationResult(
                monetary_stance="invalid_stance",  # Invalid
                trade_stance="neutral",
                tariff_mention=0,
                economic_outlook="positive",
            )

    def test_invalid_tariff_mention_raises(self) -> None:
        """tariff_mention must be 0 or 1."""
        from pydantic import ValidationError

        from brev_pipelines.types import GPTOSSClassificationResult

        with pytest.raises(ValidationError):
            GPTOSSClassificationResult(
                monetary_stance="hawkish",
                trade_stance="neutral",
                tariff_mention=2,  # Invalid - must be 0 or 1
                economic_outlook="positive",
            )

    def test_missing_field_raises(self) -> None:
        """Missing required field should raise ValidationError."""
        from pydantic import ValidationError

        from brev_pipelines.types import GPTOSSClassificationResult

        with pytest.raises(ValidationError):
            GPTOSSClassificationResult(
                monetary_stance="hawkish",
                # Missing trade_stance, tariff_mention, economic_outlook
            )

    def test_all_dovish_values(self) -> None:
        """Test all minimum-scale values."""
        from brev_pipelines.types import GPTOSSClassificationResult

        result = GPTOSSClassificationResult(
            monetary_stance="very_dovish",
            trade_stance="very_protectionist",
            tariff_mention=1,
            economic_outlook="very_negative",
        )
        assert result.monetary_stance == "very_dovish"

    def test_all_hawkish_values(self) -> None:
        """Test all maximum-scale values."""
        from brev_pipelines.types import GPTOSSClassificationResult

        result = GPTOSSClassificationResult(
            monetary_stance="very_hawkish",
            trade_stance="very_globalist",
            tariff_mention=1,
            economic_outlook="very_positive",
        )
        assert result.monetary_stance == "very_hawkish"

    def test_model_is_frozen(self) -> None:
        """Model should be immutable (frozen)."""
        from brev_pipelines.types import GPTOSSClassificationResult

        result = GPTOSSClassificationResult(
            monetary_stance="hawkish",
            trade_stance="neutral",
            tariff_mention=0,
            economic_outlook="positive",
        )
        with pytest.raises((TypeError, ValueError)):  # Pydantic frozen model
            result.monetary_stance = "dovish"  # type: ignore[misc]


class TestClassificationToNumeric:
    """Tests for converting classification to numeric values."""

    def test_classification_to_numeric_hawkish(self) -> None:
        """Hawkish should map to 4 on 1-5 scale."""
        from brev_pipelines.types import (
            GPTOSSClassificationResult,
            gpt_oss_classification_to_numeric,
        )

        result = GPTOSSClassificationResult(
            monetary_stance="hawkish",
            trade_stance="neutral",
            tariff_mention=0,
            economic_outlook="positive",
        )
        numeric = gpt_oss_classification_to_numeric(result)

        assert numeric["monetary_stance"] == 4
        assert numeric["trade_stance"] == 3
        assert numeric["tariff_mention"] == 0
        assert numeric["economic_outlook"] == 4

    def test_classification_to_numeric_extremes(self) -> None:
        """Test extreme values map to 1 and 5."""
        from brev_pipelines.types import (
            GPTOSSClassificationResult,
            gpt_oss_classification_to_numeric,
        )

        result = GPTOSSClassificationResult(
            monetary_stance="very_dovish",
            trade_stance="very_protectionist",
            tariff_mention=1,
            economic_outlook="very_negative",
        )
        numeric = gpt_oss_classification_to_numeric(result)

        assert numeric["monetary_stance"] == 1
        assert numeric["trade_stance"] == 1
        assert numeric["tariff_mention"] == 1
        assert numeric["economic_outlook"] == 1


class TestSchemaDescription:
    """Tests for classification schema description constant."""

    def test_classification_schema_description_exists(self) -> None:
        """Verify schema description constant is defined."""
        from brev_pipelines.types import GPT_OSS_CLASSIFICATION_SCHEMA

        assert "monetary_stance" in GPT_OSS_CLASSIFICATION_SCHEMA
        assert "trade_stance" in GPT_OSS_CLASSIFICATION_SCHEMA
        assert "tariff_mention" in GPT_OSS_CLASSIFICATION_SCHEMA
        assert "economic_outlook" in GPT_OSS_CLASSIFICATION_SCHEMA

    def test_schema_description_includes_allowed_values(self) -> None:
        """Schema description should list allowed enum values."""
        from brev_pipelines.types import GPT_OSS_CLASSIFICATION_SCHEMA

        assert "hawkish" in GPT_OSS_CLASSIFICATION_SCHEMA
        assert "dovish" in GPT_OSS_CLASSIFICATION_SCHEMA
        assert "neutral" in GPT_OSS_CLASSIFICATION_SCHEMA
