"""Tests for anomaly detection module."""
import pandas as pd
import pytest
from pathlib import Path

from campus_energy.analytics.features import engineer_features
from campus_energy.analytics.anomalies import detect_anomalies

DATA_DIR = Path(__file__).parent.parent / "data" / "generated"

@pytest.fixture
def dataframes():
    paths = {
        "telemetry": DATA_DIR / "telemetry.csv",
        "context": DATA_DIR / "context.csv",
        "assets": DATA_DIR / "assets.csv",
        "gt": DATA_DIR / "anomaly_ground_truth.csv"
    }
    
    if not all(p.exists() for p in paths.values()):
        pytest.skip("Generated data not found.")
        
    dfs = {k: pd.read_csv(v) for k, v in paths.items()}
    
    # Merge telemetry with context
    tel = dfs["telemetry"]
    ctx = dfs["context"]
    assets = dfs["assets"]
    
    # Simple merge: add building_id and zone_id to telemetry
    tel = tel.merge(assets[["asset_id", "building_id", "zone_id"]], on="asset_id", how="left")
    
    # Merge context
    merged = tel.merge(ctx, on=["timestamp", "building_id", "zone_id"], how="left")
    
    # For meters without zones, we can just fill context from building level, but this is enough for test
    dfs["merged"] = merged
    
    return dfs

def test_detect_anomalies(dataframes):
    df = dataframes["merged"]
    assets = dataframes["assets"]
    gt = dataframes["gt"]
    
    # Feature engineering
    features = engineer_features(df, df) # second arg is dummy here since we merged
    
    # Detect
    alerts = detect_anomalies(features, assets)
    
    assert len(alerts) > 0, "No alerts detected"
    
    # Check that we detected the specific injected anomalies
    alert_types = [a.anomaly_type for a in alerts]
    assert "pump_no_flow" in alert_types, "Failed to detect pump_no_flow"
    assert "sensor_dropout" in alert_types, "Failed to detect sensor_dropout"
