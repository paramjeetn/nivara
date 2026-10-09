"""Live replay engine.

Replays generated telemetry data through the ingestion API at accelerated speed.
"""
import asyncio
import httpx
import pandas as pd
import logging
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

async def run_replay(
    telemetry_path: str,
    api_url: str = "http://localhost:8000/api/v1/telemetry/batch",
    batch_size: int = 50,
    speed_factor: float = 360.0  # 1 hour = 10 seconds
):
    logger.info(f"Loading telemetry from {telemetry_path}")
    df = pd.read_csv(telemetry_path)
    
    # Sort by timestamp
    df["dt"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("dt").reset_index(drop=True)
    
    if df.empty:
        logger.warning("No telemetry found.")
        return
        
    start_sim_time = df["dt"].min()
    start_real_time = datetime.now()
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Group by exact timestamp to send batches together
        for ts, group in df.groupby("timestamp"):
            records = group.drop(columns=["dt"]).to_dict(orient="records")
            
            # Replace NaNs with None for JSON serialization
            for r in records:
                for k, v in r.items():
                    if pd.isna(v):
                        r[k] = None
            
            # Send batch
            try:
                resp = await client.post(api_url, json=records)
                if resp.status_code != 202:
                    logger.error(f"Failed to send batch: {resp.status_code} {resp.text}")
                else:
                    logger.info(f"Sent batch of {len(records)} readings for timestamp {ts}")
            except Exception as e:
                logger.error(f"Connection error: {e}")
                
            # Sleep to match replay speed
            # The next timestamp is usually 5 minutes (300s) later in sim time
            # In real time, 300s / 360 = 0.83 seconds
            sim_elapsed = (pd.to_datetime(ts) - start_sim_time).total_seconds()
            expected_real_elapsed = sim_elapsed / speed_factor
            
            actual_real_elapsed = (datetime.now() - start_real_time).total_seconds()
            sleep_time = expected_real_elapsed - actual_real_elapsed
            
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="data/generated/telemetry.csv")
    parser.add_argument("--url", type=str, default="http://localhost:8000/api/v1/telemetry/batch")
    parser.add_argument("--speed", type=float, default=360.0)
    args = parser.parse_args()
    
    asyncio.run(run_replay(args.file, args.url, speed_factor=args.speed))
