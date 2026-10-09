"""Generates synthetic NLP complaints correlated with facility anomalies."""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import uuid

def generate_complaints(events_df: pd.DataFrame, assets_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    complaints = []
    
    # 1. Base random complaints (noise)
    start_time = pd.to_datetime("2025-09-01T08:00:00+05:30")
    end_time = pd.to_datetime("2025-09-29T18:00:00+05:30")
    
    random_complaints_text = [
        ("lighting", "Bulb is flickering in the hallway."),
        ("hvac", "It's slightly warm in the corner of the room."),
        ("power", "Socket is loose on desk 4."),
        ("equipment", "Projector fan is making a loud noise."),
    ]
    
    # Generate ~20 random background complaints
    for _ in range(20):
        c_cat, c_text = rng.choice(random_complaints_text)
        bld = rng.choice(assets_df[assets_df["asset_type"] == "building"]["asset_id"].values)
        random_ts = start_time + timedelta(minutes=int(rng.integers(0, int((end_time - start_time).total_seconds() / 60))))
        
        complaints.append({
            "complaint_id": f"CMP-{uuid.uuid4().hex[:8].upper()}",
            "timestamp": random_ts.isoformat(),
            "building_id": bld,
            "zone_id": None,
            "category": c_cat,
            "description": c_text,
            "status": "closed",
            "resolution_timestamp": (random_ts + timedelta(days=rng.uniform(1, 3))).isoformat()
        })
        
    # 2. Correlated complaints (tied to our injected anomalies)
    for _, evt in events_df.iterrows():
        evt_type = evt["event_type"]
        evt_start = pd.to_datetime(evt["start_time"])
        asset_id = evt["asset_id"]
        
        # Find the building/zone for this asset
        asset_info = assets_df[assets_df["asset_id"] == asset_id].iloc[0]
        bld_id = asset_info["building_id"]
        zone_id = asset_info["zone_id"] if pd.notna(asset_info["zone_id"]) else None
        
        # Complaint occurs a few hours after the event starts
        comp_time = evt_start + timedelta(hours=rng.uniform(1, 4))
        
        c_cat = "other"
        c_text = ""
        
        if evt_type == "lights_left_on":
            c_cat = "lighting"
            c_text = "Who left the lights on all night? Walking past the building and it's lit up like a Christmas tree."
        elif evt_type == "hvac_left_on":
            c_cat = "hvac"
            c_text = "The AC in the office is running full blast even though no one is here. The room is freezing!"
        elif evt_type == "pump_no_flow":
            c_cat = "equipment"
            c_text = "Loud grinding noise coming from the utility pump room. Also, water pressure seems slightly low."
        elif evt_type == "sensor_dropout":
            c_cat = "power"
            c_text = "The smart meter display on floor 2 is completely blank. Is the network down?"
            
        if c_text:
            complaints.append({
                "complaint_id": f"CMP-{uuid.uuid4().hex[:8].upper()}",
                "timestamp": comp_time.isoformat(),
                "building_id": bld_id,
                "zone_id": zone_id,
                "category": c_cat,
                "description": c_text,
                "status": "open",
                "resolution_timestamp": None
            })
            
    return pd.DataFrame(complaints).sort_values("timestamp")
