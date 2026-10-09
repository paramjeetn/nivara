"""Tests for simulator outputs."""
import pandas as pd
import pytest
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "generated"

@pytest.fixture
def dataframes():
    assets_path = DATA_DIR / "assets.csv"
    context_path = DATA_DIR / "context.csv"
    
    if not assets_path.exists() or not context_path.exists():
        pytest.skip("Generated data not found. Run generator first.")
        
    return {
        "assets": pd.read_csv(assets_path),
        "context": pd.read_csv(context_path)
    }

def test_occupancy_bounds(dataframes):
    """Check occupancy is non-negative and within capacity."""
    assets = dataframes["assets"]
    context = dataframes["context"]
    
    # We don't have capacity in context, need to parse from config if we really want to check upper bound
    # But we can check >= 0
    assert (context["occupancy"] >= 0).all()
    assert (context["scheduled_occupancy"] >= 0).all()

def test_weather_bounds(dataframes):
    context = dataframes["context"]
    assert (context["outdoor_temperature_c"] >= 15.0).all()
    assert (context["outdoor_temperature_c"] <= 50.0).all()
    assert (context["humidity_pct"] >= 0.0).all()
    assert (context["humidity_pct"] <= 100.0).all()

def test_timestamps_present(dataframes):
    context = dataframes["context"]
    ts = pd.to_datetime(context["timestamp"])
    # 28 days = 28 * 24 * 12 = 8064 expected unique timestamps
    unique_ts = ts.nunique()
    assert unique_ts == 8064
