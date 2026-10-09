"""Tests for energy balance and reconciliation."""
import pandas as pd
import pytest
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "generated"

@pytest.fixture
def dataframes():
    assets_path = DATA_DIR / "assets.csv"
    telemetry_path = DATA_DIR / "telemetry.csv"
    
    if not assets_path.exists() or not telemetry_path.exists():
        pytest.skip("Generated data not found. Run generator first.")
        
    return {
        "assets": pd.read_csv(assets_path),
        "telemetry": pd.read_csv(telemetry_path)
    }

def test_interval_energy_consistency(dataframes):
    """Check that interval_energy_kwh roughly equals power_kw / 12."""
    df = dataframes["telemetry"]
    
    # Exclude cases with missing power
    valid = df.dropna(subset=["power_kw", "interval_energy_kwh"])
    
    expected_energy = valid["power_kw"] / 12.0
    actual_energy = valid["interval_energy_kwh"]
    
    # Assert they are close (tolerance for rounding)
    diff = (expected_energy - actual_energy).abs()
    assert (diff < 0.001).all(), "Energy does not match power"

def test_energy_balance(dataframes):
    """Check that parent meters roughly equal the sum of their descendants + base load."""
    assets = dataframes["assets"]
    telemetry = dataframes["telemetry"]
    
    buildings = assets[assets["asset_type"] == "building"]["asset_id"].tolist()
    
    for building_id in buildings:
        # Find meter for this building
        meter = assets[(assets["parent_asset_id"] == building_id) & (assets["asset_type"] == "meter")]
        if meter.empty:
            continue
        meter_id = meter.iloc[0]["asset_id"]
        
        # Find all equipment under this building
        zones = assets[assets["parent_asset_id"] == building_id]["asset_id"].tolist()
        equipment_ids = assets[assets["parent_asset_id"].isin(zones)]["asset_id"].tolist()
        
        # Get meter telemetry
        meter_telemetry = telemetry[telemetry["asset_id"] == meter_id].set_index("timestamp")
        
        # Sum equipment telemetry
        eq_telemetry = telemetry[telemetry["asset_id"].isin(equipment_ids)]
        eq_sum = eq_telemetry.groupby("timestamp")["power_kw"].sum()
        
        # Compare
        # The difference should be the base load + some noise
        # Just check that meter is strictly greater or roughly equal to sum of equipment
        joined = meter_telemetry[["power_kw"]].join(eq_sum, rsuffix="_eq").dropna()
        
        assert (joined["power_kw"] >= joined["power_kw_eq"] * 0.95).all(), f"Meter power less than equipment sum for {building_id}"

def test_no_negative_values(dataframes):
    df = dataframes["telemetry"]
    assert (df["power_kw"].dropna() >= 0).all()
    assert (df["interval_energy_kwh"].dropna() >= 0).all()

def test_cumulative_energy_monotonic(dataframes):
    df = dataframes["telemetry"]
    for asset_id, group in df.groupby("asset_id"):
        diffs = group["cumulative_energy_kwh"].diff().dropna()
        # Due to noise, it might be slightly negative, but should be mostly >= 0
        assert (diffs >= -0.001).all(), f"Cumulative energy not monotonic for {asset_id}"
