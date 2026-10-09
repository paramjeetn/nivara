"""FastAPI ingestion service for the campus platform."""
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from pydantic import ValidationError
from contextlib import asynccontextmanager

from campus_energy.schemas import TelemetryReading, Alert
from campus_energy.storage.repository import Repository

logger = logging.getLogger(__name__)
repo = Repository()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Ingestion API...")
    yield
    # Shutdown
    logger.info("Shutting down Ingestion API...")

app = FastAPI(title="Campus Energy Ingestion API", lifespan=lifespan)

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "ingestion"}

@app.post("/api/v1/telemetry", status_code=status.HTTP_202_ACCEPTED)
def ingest_telemetry(reading: Dict[str, Any], background_tasks: BackgroundTasks):
    """Ingest a single telemetry record."""
    try:
        # Validate against schema
        valid_reading = TelemetryReading(**reading)
    except ValidationError as e:
        # Convert errors to simple dicts to avoid JSON serialization issues with exception objects in ctx
        error_list = [{"loc": err.get("loc"), "msg": err.get("msg")} for err in e.errors()]
        raise HTTPException(status_code=422, detail=error_list)
        
    background_tasks.add_task(repo.insert_telemetry_batch, [valid_reading.model_dump(mode="json")])
    return {"status": "accepted", "reading_id": valid_reading.reading_id}

@app.post("/api/v1/telemetry/batch", status_code=status.HTTP_202_ACCEPTED)
def ingest_telemetry_batch(readings: List[Dict[str, Any]], background_tasks: BackgroundTasks):
    """Ingest a batch of telemetry records.
    
    Policy: Partial success. Valid records are accepted, invalid are dropped.
    Returns a report of accepted vs rejected counts.
    """
    valid_records = []
    errors = []
    
    for i, r in enumerate(readings):
        try:
            valid = TelemetryReading(**r)
            valid_records.append(valid.model_dump(mode="json"))
        except ValidationError as e:
            error_list = [{"loc": err.get("loc"), "msg": err.get("msg")} for err in e.errors()]
            errors.append({"index": i, "reading_id": r.get("reading_id", "unknown"), "errors": error_list})
            
    if valid_records:
        background_tasks.add_task(repo.insert_telemetry_batch, valid_records)
        
    return {
        "status": "batch_processed",
        "accepted": len(valid_records),
        "rejected": len(errors),
        "errors": errors[:10]  # Only return first 10 errors to avoid massive payloads
    }

@app.get("/api/v1/telemetry/{asset_id}")
def get_telemetry(asset_id: str, start_time: str, end_time: str):
    """Query telemetry by asset and time range."""
    records = repo.query_telemetry(asset_id, start_time, end_time)
    return {"asset_id": asset_id, "count": len(records), "data": records}

@app.get("/api/v1/alerts")
def get_alerts(status: str = None):
    """Query active or resolved alerts."""
    alerts = repo.get_alerts(status)
    return {"count": len(alerts), "data": alerts}
