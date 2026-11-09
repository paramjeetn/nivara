"""FastAPI ingestion service for the campus platform."""
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from pydantic import ValidationError
from contextlib import asynccontextmanager

from campus_energy.schemas import TelemetryReading, Alert
from campus_energy.storage.repository import Repository
from campus_energy.demo.inject import (
    FAULT_TYPES,
    clear_fault_registry,
    get_active_faults,
    inject_live_fault,
)
from campus_energy.demo.tour import get_default_tour

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


# --- Step 13: Live Fault Injection & Guided Demo Endpoints ---

from pathlib import Path
import pandas as pd
from pydantic import BaseModel, Field


class DemoInjectRequest(BaseModel):
    fault_type: str = Field(..., description="Canonical fault: lights_left_on, pump_no_flow, pipe_leak_overnight, sensor_dropout, pf_drop")
    asset_id: str = Field(..., description="Target asset ID")
    duration_hours: float = Field(default=4.0, ge=0.1, le=72.0)
    start_time: str | None = Field(default=None, description="Optional ISO start time")
    params: Dict[str, Any] = Field(default_factory=dict)


@app.post("/api/demo/inject", status_code=status.HTTP_200_OK)
def demo_inject(req: DemoInjectRequest):
    """Inject a live operational fault for jury demonstrations."""
    data_path = Path("data/generated/telemetry.csv")
    if data_path.exists():
        df = pd.read_csv(data_path)
    else:
        now = pd.Timestamp.now()
        df = pd.DataFrame([{
            "reading_id": f"INIT-{req.asset_id}-0",
            "timestamp": (now - pd.Timedelta(hours=req.duration_hours)).isoformat(),
            "asset_id": req.asset_id,
            "reading_type": "meter",
            "power_kw": 25.0,
            "interval_energy_kwh": 2.08,
            "cumulative_energy_kwh": 100.0,
            "voltage_v": 230.0,
            "current_a": 108.7,
            "power_factor": 0.95,
            "state": "ON",
            "flow_lpm": 50.0,
            "temperature_c": 24.0,
            "setpoint_c": 24.0,
            "sensor_status": "healthy",
            "quality_flag": "valid",
        }])

    try:
        modified_df, event = inject_live_fault(
            telemetry_df=df,
            fault_type=req.fault_type,
            asset_id=req.asset_id,
            start_time=req.start_time,
            duration_hours=req.duration_hours,
            **req.params,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    injected_rows = modified_df[modified_df["asset_id"] == req.asset_id].tail(int(req.duration_hours * 12) + 1)
    if not injected_rows.empty:
        clean_records = injected_rows.replace({float("nan"): None}).to_dict(orient="records")
        repo.insert_telemetry_batch(clean_records)

    tour = get_default_tour()
    tour.set_injected_event_id(event.event_id)

    return {
        "status": "injected",
        "event_id": event.event_id,
        "fault_type": req.fault_type,
        "asset_id": req.asset_id,
        "duration_hours": req.duration_hours,
        "start_time": event.start_time.isoformat(),
        "end_time": event.end_time.isoformat(),
        "severity": event.severity.value,
        "description": event.description,
        "event": event.model_dump(mode="json"),
    }


@app.post("/api/demo/reset", status_code=status.HTTP_200_OK)
def demo_reset():
    """Reset the demo simulation back to baseline."""
    clear_fault_registry()
    tour = get_default_tour()
    tour.reset_demo()
    return {
        "status": "reset",
        "message": "Demo simulation reset to clean baseline",
        "active_faults": 0,
        "tour_step": 1,
    }


@app.get("/api/demo/tour")
def get_tour_state():
    """Get current guided tour state and active faults count."""
    tour = get_default_tour()
    return {
        "current_step": tour.current_step,
        "step_info": tour.get_step_info(),
        "active_faults_count": len(get_active_faults()),
    }

