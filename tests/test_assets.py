"""Tests for demo assets."""

import pandas as pd


class MockContext:
    """Mock Dagster context for testing."""

    class MockLog:
        def info(self, msg):
            pass

        def debug(self, msg):
            pass

        def warning(self, msg):
            pass

    log = MockLog()


def test_raw_sample_data_shape():
    """Test raw sample data generates expected records."""
    from brev_pipelines.assets.demo import raw_sample_data

    df = raw_sample_data(MockContext())
    assert len(df) == 100
    assert "id" in df.columns
    assert "region" in df.columns
    assert "spend" in df.columns
    assert "active" in df.columns


def test_raw_sample_data_id_format():
    """Test raw sample data IDs have correct format."""
    from brev_pipelines.assets.demo import raw_sample_data

    df = raw_sample_data(MockContext())
    assert df["id"].iloc[0] == "CUST-0000"
    assert df["id"].iloc[99] == "CUST-0099"


def test_cleaned_data_adds_tier():
    """Test cleaned data adds tier column."""
    from brev_pipelines.assets.demo import cleaned_data

    raw_df = pd.DataFrame({
        "id": ["CUST-0001"],
        "name": ["Test"],
        "age": [30],
        "region": ["north"],
        "category": ["Premium"],
        "spend": [5000.0],
        "active": [True],
    })

    df = cleaned_data(MockContext(), raw_df)
    assert "tier" in df.columns
    assert df["region"].iloc[0] == "North"  # Title case


def test_cleaned_data_caps_spend():
    """Test cleaned data caps extreme spend values."""
    from brev_pipelines.assets.demo import cleaned_data

    raw_df = pd.DataFrame({
        "id": ["CUST-0001"],
        "name": ["Test"],
        "age": [30],
        "region": ["North"],
        "category": ["Premium"],
        "spend": [15000.0],  # Above cap
        "active": [True],
    })

    df = cleaned_data(MockContext(), raw_df)
    assert df["spend"].iloc[0] == 9000  # Capped


def test_cleaned_data_tier_assignment():
    """Test tier assignment based on spend."""
    from brev_pipelines.assets.demo import cleaned_data

    raw_df = pd.DataFrame({
        "id": ["CUST-0001", "CUST-0002", "CUST-0003"],
        "name": ["Low", "Med", "High"],
        "age": [30, 40, 50],
        "region": ["North", "South", "East"],
        "category": ["Basic", "Standard", "Premium"],
        "spend": [500.0, 3000.0, 7000.0],
        "active": [True, True, True],
    })

    df = cleaned_data(MockContext(), raw_df)
    assert df["tier"].iloc[0] == "Low Value"
    assert df["tier"].iloc[1] == "Medium Value"
    assert df["tier"].iloc[2] == "High Value"
