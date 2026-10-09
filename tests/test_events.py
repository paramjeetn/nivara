"""Tests for event injection engine."""
import pandas as pd
import pytest
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "generated"

@pytest.fixture
def dataframes():
    paths = {
        "telemetry": DATA_DIR / "telemetry.csv",
        "events": DATA_DIR / "events.csv",
        "gt": DATA_DIR / "anomaly_ground_truth.csv"
    }
    
    if not all(p.exists() for p in paths.values()):
        pytest.skip("Generated data not found. Run generator first.")
        
    return {k: pd.read_csv(v) for k, v in paths.items()}

def test_events_exist(dataframes):
    events = dataframes["events"]
    gt = dataframes["gt"]
    
    assert len(events) > 0, "No events were generated"
    assert len(events) == len(gt), "Events and Ground Truth mismatch"

def test_sensor_dropout_effect(dataframes):
    tel = dataframes["telemetry"]
    events = dataframes["events"]
    
    # Find a sensor dropout event
    dropouts = events[events["event_type"] == "sensor_dropout"]
    if dropouts.empty:
        pytest.skip("No sensor dropout event injected")
        
    evt = dropouts.iloc[0]
    asset_id = evt["asset_id"]
    
    # Check telemetry during this period
    mask = (tel["asset_id"] == asset_id) & (tel["timestamp"] >= evt["start_time"]) & (tel["timestamp"] <= evt["end_time"])
    affected_readings = tel[mask]
    
    assert not affected_readings.empty, "No telemetry found during event window"
    # Dropout should result in missing power
    assert affected_readings["power_kw"].isna().all(), "Power should be missing during dropout"
    assert (affected_readings["quality_flag"] == "missing").all()
    assert (affected_readings["sensor_status"] == "offline").all()

def test_pump_no_flow_effect(dataframes):
    tel = dataframes["telemetry"]
    events = dataframes["events"]
    
    no_flow = events[events["event_type"] == "pump_no_flow"]
    if no_flow.empty:
        pytest.skip("No pump_no_flow event injected")
        
    evt = no_flow.iloc[0]
    asset_id = evt["asset_id"]
    
    mask = (tel["asset_id"] == asset_id) & (tel["timestamp"] >= evt["start_time"]) & (tel["timestamp"] <= evt["end_time"])
    affected_readings = tel[mask]
    
    assert not affected_readings.empty
    assert (affected_readings["flow_lpm"] == 0.0).all(), "Flow should be zero"
    # Power should still be reported
    assert affected_readings["power_kw"].notna().all(), "Power should still be present"

def test_duplicate_reading(dataframes):
    tel = dataframes["telemetry"]
    events = dataframes["events"]
    
    dups = events[events["event_type"] == "duplicate_reading"]
    if dups.empty:
        pytest.skip("No duplicate reading event injected")
        
    evt = dups.iloc[0]
    asset_id = evt["asset_id"]
    ts = evt["start_time"]
    
    # There should be exactly two rows with this timestamp and asset (one original, one duplicate)
    # The duplicate should have quality_flag = 'duplicated'
    readings = tel[(tel["asset_id"] == asset_id) & (tel["timestamp"] == ts)]
    assert len(readings) == 2, "Expected exactly one original and one duplicate row"
    
    flags = readings["quality_flag"].tolist()
    assert "valid" in flags
    assert "duplicated" in flags
