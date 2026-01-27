"""Harmony response format utilities for GPT-OSS.

GPT-OSS models output in the Harmony format which uses control tokens
to separate different output channels (analysis, final, commentary).

This module provides functions to:
1. Remove Harmony control tokens from responses
2. Extract JSON from Harmony-formatted content
3. Validate classification responses
4. Convert classification results to numeric values

References:
- https://cookbook.openai.com/articles/openai-harmony
- https://github.com/openai/harmony
- docs/reports/gpt-oss-reasoning-structured-output.md
"""

from __future__ import annotations

import json
import re

from brev_pipelines.types import (
    GPT_OSS_MONETARY_SCALE,
    GPT_OSS_OUTLOOK_SCALE,
    GPT_OSS_TRADE_SCALE,
    GPTOSSClassificationResult,
    SpeechClassification,
)

# =============================================================================
# Harmony Token Constants
# =============================================================================

# All known Harmony control tokens
HARMONY_TOKENS: frozenset[str] = frozenset(
    {
        "<|start|>",
        "<|end|>",
        "<|return|>",
        "<|call|>",
        "<|channel|>",
        "<|message|>",
        "<|constrain|>",
    }
)

# Regex pattern to match any Harmony token
# Matches: <|token_name|> where token_name is alphanumeric with underscores
HARMONY_TOKEN_PATTERN: str = r"<\|[a-zA-Z_]+\|>"


# =============================================================================
# Token Removal Functions
# =============================================================================


def remove_harmony_tokens(content: str) -> str:
    """Remove all Harmony control tokens from content.

    Removes tokens like <|return|>, <|channel|>, <|message|>, etc.
    Preserves all other text including channel names (analysis, final).

    Args:
        content: Raw content that may contain Harmony tokens.

    Returns:
        Content with all Harmony tokens removed.

    Example:
        >>> remove_harmony_tokens('<|return|>text<|end|>')
        'text'
        >>> remove_harmony_tokens('<|channel|>final<|message|>output')
        'finaloutput'
    """
    return re.sub(HARMONY_TOKEN_PATTERN, "", content)


# =============================================================================
# JSON Extraction Functions
# =============================================================================


def extract_json_from_harmony(content: str) -> dict[str, object]:
    r"""Extract and parse JSON from Harmony-formatted content.

    Handles:
    - Harmony control tokens (<|return|>, <|channel|>, etc.)
    - Reasoning text before the JSON
    - Whitespace and formatting

    Uses a robust approach:
    1. Remove all Harmony tokens
    2. Find the JSON object using bracket matching
    3. Parse and return the JSON

    Args:
        content: Raw content from NIM response.

    Returns:
        Parsed JSON as a dictionary.

    Raises:
        ValueError: If no valid JSON found in content.

    Example:
        >>> extract_json_from_harmony('{"key": "value"}<|return|>')
        {'key': 'value'}
        >>> extract_json_from_harmony('Analysis...\n{"key": "value"}')
        {'key': 'value'}
    """
    # Remove Harmony tokens first
    cleaned = remove_harmony_tokens(content)

    # Find the start of JSON object
    json_start = cleaned.find("{")
    if json_start == -1:
        raise ValueError(f"No JSON found in response: {content[:200]}...")

    # Find matching closing brace using bracket counting
    # This handles nested objects correctly
    brace_count = 0
    json_end = json_start
    in_string = False
    escape_next = False

    for i, char in enumerate(cleaned[json_start:], start=json_start):
        if escape_next:
            escape_next = False
            continue

        if char == "\\":
            escape_next = True
            continue

        if char == '"' and not escape_next:
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0:
                json_end = i + 1
                break

    if brace_count != 0:
        raise ValueError(f"Invalid JSON: unmatched braces in {content[:200]}...")

    json_str = cleaned[json_start:json_end]

    try:
        result: dict[str, object] = json.loads(json_str)
        return result
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e


# =============================================================================
# Validation Functions
# =============================================================================


def validate_and_convert_classification(
    raw_json: dict[str, object],
) -> GPTOSSClassificationResult:
    """Validate raw JSON against GPTOSSClassificationResult schema.

    Args:
        raw_json: Parsed JSON dictionary from LLM response.

    Returns:
        Validated GPTOSSClassificationResult.

    Raises:
        ValidationError: If JSON doesn't match expected schema.
    """
    return GPTOSSClassificationResult(**raw_json)


# =============================================================================
# End-to-End Parsing Functions
# =============================================================================


def parse_classification_response(content: str) -> SpeechClassification:
    """Parse and validate classification response from GPT-OSS.

    This is the main entry point for processing GPT-OSS classification
    responses. It handles:
    1. Harmony token cleanup
    2. JSON extraction
    3. Schema validation
    4. Conversion to numeric values

    Args:
        content: Raw response content from GPT-OSS.

    Returns:
        SpeechClassification TypedDict with numeric values.

    Raises:
        ValueError: If JSON cannot be extracted.
        ValidationError: If JSON doesn't match classification schema.

    Example:
        >>> content = '{"monetary_stance": "hawkish", ...}<|return|>'
        >>> result = parse_classification_response(content)
        >>> result["monetary_stance"]
        4
    """
    # Extract JSON from Harmony-formatted content
    raw_json = extract_json_from_harmony(content)

    # Validate against Pydantic model
    validated = validate_and_convert_classification(raw_json)

    # Convert to numeric values
    return SpeechClassification(
        monetary_stance=GPT_OSS_MONETARY_SCALE[validated.monetary_stance],
        trade_stance=GPT_OSS_TRADE_SCALE[validated.trade_stance],
        tariff_mention=validated.tariff_mention,
        economic_outlook=GPT_OSS_OUTLOOK_SCALE[validated.economic_outlook],
    )
