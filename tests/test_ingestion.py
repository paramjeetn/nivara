"""Tests for ingestion API and validation."""
from fastapi.testclient import TestClient
from campus_energy.ingestion.api import app
from campus_energy.storage.repository import Repository
import pytest

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "ingestion"}

def test_ingest_single_valid_telemetry():
    payload = {
        "reading_id": "test-123",
        "timestamp": "2025-09-01T00:00:00Z",
        "asset_id": "MTR-A",
        "reading_type": "meter",
        "power_kw": 50.0,
        "interval_energy_kwh": 4.16,
        "quality_flag": "valid"
    }
    response = client.post("/api/v1/telemetry", json=payload)
    assert response.status_code == 202
    assert response.json()["status"] == "accepted"

def test_ingest_invalid_telemetry_rejected():
    # Negative power should be rejected by Pydantic model
    payload = {
        "reading_id": "test-124",
        "timestamp": "2025-09-01T00:00:00Z",
        "asset_id": "MTR-A",
        "reading_type": "meter",
        "power_kw": -10.0,
        "interval_energy_kwh": 4.16,
    }
    response = client.post("/api/v1/telemetry", json=payload)
    assert response.status_code == 422

def test_batch_ingestion_partial_success():
    payload = [
        {
            "reading_id": "batch-1",
            "timestamp": "2025-09-01T00:05:00Z",
            "asset_id": "MTR-A",
            "reading_type": "meter",
            "power_kw": 50.0,
        },
        {
            "reading_id": "batch-2",
            "timestamp": "2025-09-01T00:05:00Z",
            "asset_id": "MTR-B",
            "reading_type": "meter",
            "power_factor": 1.5, # invalid power factor (>1)
        }
    ]
    response = client.post("/api/v1/telemetry/batch", json=payload)
    assert response.status_code == 202
    data = response.json()
    assert data["accepted"] == 1
    assert data["rejected"] == 1
    assert data["errors"][0]["reading_id"] == "batch-2"
