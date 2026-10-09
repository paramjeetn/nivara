# Campus Energy Intelligence Platform

This project simulates a university's electricity-monitoring infrastructure, identifies abnormal resource consumption, explains findings, and recommends operational actions.

## 1. Setup

**Requirements:**
- Python 3.11 or 3.12

**Installation:**
```bash
pip install -e ".[dev]"
```

## 2. Generating the Synthetic Data

Generate the full 28-day baseline with injected anomalies and faults:
```bash
generate-data
```
*(Or use `python src/campus_energy/simulator/generate.py`)*

This creates the CSVs in `data/generated/`:
- `assets.csv`
- `telemetry.csv`
- `context.csv`
- `events.csv`
- `anomaly_ground_truth.csv`

## 3. Running the Dashboard

Start the interactive Streamlit dashboard:
```bash
streamlit run src/campus_energy/dashboard/app.py
```

## 4. Running the Ingestion API

To run the live ingestion API (used for replay):
```bash
uvicorn campus_energy.ingestion.api:app --reload
```

## 5. Live Replay

With the API running, stream historical data live into the API:
```bash
python src/campus_energy/replay/player.py --speed 360
```
*(360x speed means 1 hour of simulation = 10 real seconds)*

## Testing

Run all validation tests (Gates A, B, C, D):
```bash
pytest tests/ -v
```

## Architecture

- **Simulator**: Generates baseline physical behavior (occupancy, weather, equipment).
- **Event Injector**: Injects anomalies like "Lights left ON", "Sensor dropouts", and "HVAC left running".
- **Analytics**: Uses physical rules + simple Ridge regression for expected consumption baseline.
- **Ingestion API**: FastAPI service that accepts telemetry and persists to SQLite.
- **Dashboard**: Streamlit app for real-time visualization, exploring alerts, and verifying sensor health.
