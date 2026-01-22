"""Tests for validation assets."""

from brev_pipelines.assets.validation import ValidationResult


def test_validation_result_to_dict():
    """Test ValidationResult serialization."""
    result = ValidationResult(
        component="test",
        passed=True,
        tests=[{"name": "test1", "passed": True}],
        duration_ms=100.5,
    )

    d = result.to_dict()
    assert d["component"] == "test"
    assert d["passed"] is True
    assert len(d["tests"]) == 1
    assert d["tests"][0]["name"] == "test1"
    assert d["duration_ms"] == 100.5
    assert d["error"] is None


def test_validation_result_with_error():
    """Test ValidationResult with error."""
    result = ValidationResult(
        component="test",
        passed=False,
        error="Something went wrong",
    )

    d = result.to_dict()
    assert d["passed"] is False
    assert d["error"] == "Something went wrong"
