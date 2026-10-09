"""Anomaly detection module combining rules and statistical methods."""
import pandas as pd
import numpy as np
import uuid
from datetime import datetime
from sklearn.ensemble import IsolationForest

from campus_energy.schemas import Alert, Severity, AlertStatus
from campus_energy.analytics.baseline import ExpectedConsumptionModel

def detect_anomalies(features_df: pd.DataFrame, assets_df: pd.DataFrame) -> list[Alert]:
    """Run rules and statistical detectors over the features dataframe.
    
    In a real system, this would run incrementally. For the prototype,
    we run it batch-wise over the historical dataset.
    """
    alerts = []
    
    # 1. Train a simple baseline per asset on the first 7 days (assuming mostly normal)
    min_time = features_df["timestamp"].min()
    train_end = min_time + pd.Timedelta(days=7)
    
    # We will score the rest of the dataset
    test_mask = features_df["timestamp"] > train_end
    
    assets = features_df["asset_id"].unique()
    
    for asset_id in assets:
        asset_info = assets_df[assets_df["asset_id"] == asset_id].iloc[0]
        building_id = asset_info["building_id"]
        zone_id = asset_info["zone_id"] if pd.notna(asset_info["zone_id"]) else None
        
        asset_df = features_df[features_df["asset_id"] == asset_id].copy()
        
        # Train baseline model for expected consumption
        train_df = asset_df[asset_df["timestamp"] <= train_end]
        test_df = asset_df[asset_df["timestamp"] > train_end]
        
        if len(train_df) < 50 or len(test_df) == 0:
            continue
            
        model = ExpectedConsumptionModel()
        model.fit(train_df)
        
        # Predict on test
        asset_df.loc[test_df.index, "expected_power_kw"] = model.predict(test_df)
        asset_df.loc[test_df.index, "deviation"] = asset_df.loc[test_df.index, "power_kw"] - asset_df.loc[test_df.index, "expected_power_kw"]
        
        # Rule 1: High Deviation (Consumption >> Expected)
        # We flag if deviation > 30% of expected AND deviation > 0.5 kW
        dev_mask = (asset_df["timestamp"] > train_end) & (asset_df["deviation"] > 0.5) & (asset_df["deviation"] > 0.3 * asset_df["expected_power_kw"])
        
        # Find contiguous periods of deviation to form events
        asset_df["is_dev"] = dev_mask.astype(int)
        asset_df["dev_event"] = (asset_df["is_dev"].diff() == 1).cumsum()
        
        for evt_id, group in asset_df[asset_df["is_dev"] == 1].groupby("dev_event"):
            if len(group) >= 6:  # At least 30 mins sustained
                start = group["timestamp"].min()
                dev_mean = group["deviation"].mean()
                
                alerts.append(Alert(
                    alert_id=f"ALT-{uuid.uuid4().hex[:8].upper()}",
                    detected_at=group["timestamp"].max(),
                    event_start_estimate=start,
                    asset_id=asset_id,
                    building_id=building_id,
                    zone_id=zone_id,
                    anomaly_type="high_consumption",
                    severity=Severity.MEDIUM,
                    actual_value=round(group["power_kw"].mean(), 2),
                    expected_value=round(group["expected_power_kw"].mean(), 2),
                    deviation=round(dev_mean, 2),
                    evidence=f"Power exceeded expected by {round(dev_mean, 2)} kW for {len(group)*5} minutes.",
                    recommendation="Inspect equipment for efficiency degradation or unintended operation.",
                ))

        # Rule 2: Equipment ON while Occupancy is 0 (for HVAC/Lighting)
        if asset_info["asset_type"] in ["lighting", "hvac", "computer_lab"]:
            unocc_mask = (asset_df["occupancy"] == 0) & (asset_df["power_kw"] > asset_info["rated_power_kw"] * 0.2)
            asset_df["is_unocc"] = unocc_mask.astype(int)
            asset_df["unocc_event"] = (asset_df["is_unocc"].diff() == 1).cumsum()
            
            for evt_id, group in asset_df[asset_df["is_unocc"] == 1].groupby("unocc_event"):
                if len(group) >= 6: # 30 mins
                    start = group["timestamp"].min()
                    alerts.append(Alert(
                        alert_id=f"ALT-{uuid.uuid4().hex[:8].upper()}",
                        detected_at=group["timestamp"].max(),
                        event_start_estimate=start,
                        asset_id=asset_id,
                        building_id=building_id,
                        zone_id=zone_id,
                        anomaly_type="unoccupied_operation",
                        severity=Severity.MEDIUM if asset_info["asset_type"] == "lighting" else Severity.HIGH,
                        actual_value=round(group["power_kw"].mean(), 2),
                        expected_value=0.0,
                        evidence=f"Equipment consumed {round(group['power_kw'].mean(), 2)} kW while zone occupancy was 0.",
                        recommendation="Switch off equipment when zone is unoccupied.",
                    ))

        # Rule 3: Pump No Flow
        if asset_info["asset_type"] == "pump":
            noflow_mask = (asset_df["power_kw"] > asset_info["rated_power_kw"] * 0.5) & (asset_df["flow_lpm"] == 0)
            asset_df["is_noflow"] = noflow_mask.astype(int)
            asset_df["noflow_event"] = (asset_df["is_noflow"].diff() == 1).cumsum()
            
            for evt_id, group in asset_df[asset_df["is_noflow"] == 1].groupby("noflow_event"):
                if len(group) >= 3: # 15 mins
                    start = group["timestamp"].min()
                    alerts.append(Alert(
                        alert_id=f"ALT-{uuid.uuid4().hex[:8].upper()}",
                        detected_at=group["timestamp"].max(),
                        event_start_estimate=start,
                        asset_id=asset_id,
                        building_id=building_id,
                        zone_id=zone_id,
                        anomaly_type="pump_no_flow",
                        severity=Severity.HIGH,
                        actual_value=0.0,
                        expected_value=asset_info.get("flow_rated_lpm", 100.0),
                        evidence="Pump is consuming power but reporting 0 flow.",
                        recommendation="Inspect pump for blockage or dry running immediately.",
                    ))
                    
        # Rule 4: Sensor Dropout
        dropout_mask = (asset_df["quality_flag"] == "missing") | (asset_df["sensor_status"] == "offline")
        asset_df["is_drop"] = dropout_mask.astype(int)
        asset_df["drop_event"] = (asset_df["is_drop"].diff() == 1).cumsum()
        
        for evt_id, group in asset_df[asset_df["is_drop"] == 1].groupby("drop_event"):
            if len(group) >= 3:
                start = group["timestamp"].min()
                alerts.append(Alert(
                    alert_id=f"ALT-{uuid.uuid4().hex[:8].upper()}",
                    detected_at=group["timestamp"].max(),
                    event_start_estimate=start,
                    asset_id=asset_id,
                    building_id=building_id,
                    zone_id=zone_id,
                    anomaly_type="sensor_dropout",
                    severity=Severity.LOW,
                    evidence=f"Sensor offline or missing data for {len(group)*5} minutes.",
                    recommendation="Check sensor connectivity and meter communication.",
                ))
                
    return alerts
