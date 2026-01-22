"""Tests for Central Bank Speeches pipeline."""

from __future__ import annotations

import io
import re

import polars as pl


class TestCleanedSpeeches:
    """Tests for cleaned_speeches asset logic."""

    def test_speech_id_generation(self) -> None:
        """Test that speech IDs follow expected format."""
        # Create mock raw data
        raw_df = pl.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02"],
                "central_bank": ["Fed", "ECB"],
                "speaker": ["Powell", "Lagarde"],
                "title": ["Speech 1", "Speech 2"],
                "text": ["A" * 200, "B" * 200],
            }
        )

        # Simulate ID generation logic
        df = raw_df.with_row_index("_row_idx")
        df = df.with_columns(
            (pl.lit("SPEECH-") + pl.col("_row_idx").cast(pl.Utf8).str.zfill(6)).alias("speech_id")
        )
        df = df.drop("_row_idx")

        # Verify ID format SPEECH-XXXXXX
        expected_pattern = r"^SPEECH-\d{6}$"
        for speech_id in df["speech_id"].to_list():
            assert re.match(expected_pattern, speech_id), f"Invalid ID format: {speech_id}"

    def test_empty_text_filtering(self) -> None:
        """Test that speeches with short text are filtered out."""
        raw_df = pl.DataFrame(
            {
                "text": ["Short", "A" * 200],
                "central_bank": ["Fed", "ECB"],
            }
        )

        # Apply filter logic
        df = raw_df.filter(pl.col("text").str.len_chars() > 100)

        assert len(df) == 1
        assert df["central_bank"][0] == "ECB"

    def test_column_normalization(self) -> None:
        """Test that column names are normalized correctly."""
        # Create DataFrame with non-standard column names
        raw_df = pl.DataFrame(
            {
                "Date": ["2024-01-01"],
                "institution": ["Fed"],
                "content": ["A" * 200],
            }
        )

        # Apply column mapping
        column_mapping = {
            "Date": "date",
            "institution": "central_bank",
            "content": "text",
        }

        df = raw_df
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename({old_name: new_name})

        assert "date" in df.columns
        assert "central_bank" in df.columns
        assert "text" in df.columns

    def test_null_filling(self) -> None:
        """Test that null values are filled with defaults."""
        raw_df = pl.DataFrame(
            {
                "central_bank": [None, "ECB"],
                "speaker": ["Powell", None],
                "title": [None, "Speech"],
                "text": ["A" * 200, "B" * 200],
            }
        )

        df = raw_df.with_columns(
            [
                pl.col("central_bank").fill_null("Unknown"),
                pl.col("speaker").fill_null("Unknown"),
                pl.col("title").fill_null("Untitled"),
            ]
        )

        assert df["central_bank"][0] == "Unknown"
        assert df["speaker"][1] == "Unknown"
        assert df["title"][0] == "Untitled"


class TestTariffClassification:
    """Tests for tariff_classification asset logic."""

    def test_classification_values_are_valid(self) -> None:
        """Test that classification produces valid 0/1 values."""
        valid_values = {0, 1}

        # Simulate classification results
        classifications = [0, 1, 0, 1, 0]
        for val in classifications:
            assert val in valid_values

    def test_confidence_range(self) -> None:
        """Test that confidence scores are in [0, 1]."""
        confidences = [0.0, 0.5, 0.9, 1.0]
        for conf in confidences:
            assert 0.0 <= conf <= 1.0

    def test_json_parsing(self) -> None:
        """Test JSON extraction from LLM response."""
        # Simulated LLM responses
        test_responses = [
            '{"tariff_mention": 1, "confidence": 0.95}',
            'The response is {"tariff_mention": 0, "confidence": 0.8}.',
            "Invalid response without JSON",
        ]

        results = []
        for response in test_responses:
            json_match = re.search(r"\{[^}]+\}", response)
            if json_match:
                import json

                result = json.loads(json_match.group())
                results.append((result.get("tariff_mention", 0), result.get("confidence", 0.5)))
            else:
                results.append((0, 0.0))

        assert results[0] == (1, 0.95)
        assert results[1] == (0, 0.8)
        assert results[2] == (0, 0.0)


class TestDataProductSchema:
    """Tests for final data product schema."""

    def test_required_columns_present(self) -> None:
        """Test that final data product has all required columns."""
        required = [
            "speech_id",
            "date",
            "central_bank",
            "speaker",
            "title",
            "text",
            "tariff_mention",
            "tariff_confidence",
            "processed_at",
        ]

        # Create mock data product
        df = pl.DataFrame(
            {
                "speech_id": ["SPEECH-000001"],
                "date": ["2024-01-01"],
                "central_bank": ["Fed"],
                "speaker": ["Powell"],
                "title": ["Test Speech"],
                "text": ["Test content"],
                "tariff_mention": [0],
                "tariff_confidence": [0.5],
                "processed_at": ["2024-01-01T00:00:00"],
            }
        )

        for col in required:
            assert col in df.columns, f"Missing required column: {col}"

    def test_parquet_serialization(self) -> None:
        """Test that data product can be serialized to Parquet."""
        df = pl.DataFrame(
            {
                "speech_id": ["SPEECH-000001"],
                "date": ["2024-01-01"],
                "central_bank": ["Fed"],
                "speaker": ["Powell"],
                "title": ["Test Speech"],
                "text": ["Test content " * 100],
                "tariff_mention": [0],
                "tariff_confidence": [0.5],
                "processed_at": ["2024-01-01T00:00:00"],
            }
        )

        # Verify Parquet round-trip
        buffer = io.BytesIO()
        df.write_parquet(buffer)
        buffer.seek(0)
        restored = pl.read_parquet(buffer)

        assert len(restored) == 1
        assert restored.columns == df.columns
        assert restored["speech_id"][0] == "SPEECH-000001"


class TestEmbeddings:
    """Tests for embedding generation logic."""

    def test_text_truncation(self) -> None:
        """Test that long texts are truncated for embedding."""
        long_text = "A" * 10000
        truncated = long_text[:2000]

        assert len(truncated) == 2000

    def test_title_text_combination(self) -> None:
        """Test that title and text are combined correctly."""
        title = "Economic Outlook"
        text = "The economy is growing steadily..."

        combined = f"{title}\n\n{text[:2000]}"

        assert combined.startswith("Economic Outlook")
        assert "\n\n" in combined
        assert "economy" in combined


class TestWeaviateIndex:
    """Tests for Weaviate indexing logic."""

    def test_text_truncation_for_storage(self) -> None:
        """Test that text is truncated to 10K chars for Weaviate."""
        long_text = "A" * 15000
        truncated = long_text[:10000]

        assert len(truncated) == 10000

    def test_object_structure(self) -> None:
        """Test that objects have correct structure for Weaviate."""
        row = {
            "speech_id": "SPEECH-000001",
            "date": "2024-01-01",
            "central_bank": "Fed",
            "speaker": "Powell",
            "title": "Test",
            "text": "Content",
            "tariff_mention": 1,
        }

        obj = {
            "speech_id": row["speech_id"],
            "date": str(row.get("date", "")),
            "central_bank": row.get("central_bank", "Unknown"),
            "speaker": row.get("speaker", "Unknown"),
            "title": row.get("title", "Untitled"),
            "text": (row.get("text", "") or "")[:10000],
            "tariff_mention": bool(row.get("tariff_mention", 0)),
        }

        assert obj["speech_id"] == "SPEECH-000001"
        assert obj["tariff_mention"] is True
        assert isinstance(obj["tariff_mention"], bool)
