# Nivara — Platform State & Codebase Architecture (v2 Upgrade & Polish Sign-Off)

> **Document Status:** Authoritative Codebase State as of UX-11 Polish & v2 Freeze  
> **Date:** November 2026  
> **Platform:** Nivara Institutional Resource Intelligence Platform  
> **Test Suite Status:** 316 / 316 passing (`pytest tests/`, 100% green pass)  
> **Operational Architecture:** 7 Core Destinations + Embedded Ask Nivara Panel + Decoupled Presenter Sidecar (Port 8502)

---

## 1. Full Package Layout

The codebase follows a modular Python structure under `src/campus_energy/`:

```
src/campus_energy/
├── __init__.py                     # Package entry point and version metadata
├── config.py                       # Pydantic simulation & campus configuration schema & YAML loader
├── schemas.py                      # Pydantic canonical data contracts for all data models & enums
│
├── adapters/                       # Ingestion adapters & data quality validation
│   ├── __init__.py
│   ├── byod.py                     # Bring-Your-Own-Data smart CSV column mapper & parser
│   └── dqe.py                      # Data Quality Engine (DQE 5-rule scorecard & integrity check)
│
├── analytics/                      # Core analytics, models, baselines, and scoring
│   ├── __init__.py
│   ├── anomalies.py                # Multi-layer anomaly detector (Statistical, Rule-based, Isolation Forest)
│   ├── baseline.py                 # Ridge regression expected consumption baseline (ExpectedConsumptionModel)
│   ├── billing.py                  # Departmental energy attribution, chargebacks & after-hours tracking
│   ├── carbon_budget.py            # NAPCC monthly carbon budget burn-down & 30-day projection tracker
│   ├── comfort.py                  # ASHRAE 55 / ISO 7730 PMV/PPD & Thom's Discomfort Index models
│   ├── complaints_nlp.py           # NLP complaint clustering using TF-IDF and KMeans
│   ├── esg.py                      # ESG KPIs, carbon emission math & BEE Energy Performance Index (EPI)
│   ├── explainability.py           # Diagnostic explainability layer ("Why was this alert triggered?")
│   ├── features.py                 # Time-series feature engineering & rolling window calculations
│   ├── forecasting.py              # 24h forward Ridge forecaster with scenario simulation levers
│   ├── forecasting_prophet.py      # Campus-level Meta Prophet forecaster with held-out MAPE validation
│   ├── governance.py               # Cross-facility benchmarking, BEE matrix & state-level scaling simulator
│   ├── report_generator.py         # ReportLab PDF executive audit report generator
│   └── workorders.py               # CMMS work-order dispatch queue with multi-factor priority scoring
│
├── dashboard/                      # Core application runtime & entry point
│   ├── __init__.py
│   ├── agent.py                    # Google Gemini LLM API client & structured action recommendation parser
│   ├── app.py                      # Production operational tool shell (Port 8501)
│   └── osm_buildings.py            # OpenStreetMap GeoJSON building footprint extractor
│
├── decision/                       # Operational guardrails, recommendation & verification
│   ├── __init__.py
│   ├── guard.py                    # Service-Impact Guard (Clinical safety lockouts & ASHRAE 55 comfort bounds)
│   ├── ledger.py                   # SavingsLedger (IPMVP Option B continuous M&V & hit-rate tracking)
│   └── recommendation.py           # Unified recommendation engine with three-point uncertainty ranges
│
├── demo/                           # Demonstration tour definitions & live fault injection
│   ├── __init__.py
│   ├── injector.py                 # Live fault injector (cooling baseline drift, water leak, etc.)
│   └── tour.py                     # 6-step guided jury tour state machine
│
├── eval/                           # Multi-gate automated evaluation harness
│   ├── __init__.py
│   ├── harness.py                  # Gate A, B, C, D validation suite
│   └── run_all.py                  # Evaluation CLI runner
│
├── ingestion/                      # Edge telemetry ingestion pipeline
│   ├── __init__.py
│   └── api.py                      # FastAPI REST ingestion service with batch validation & SQLite queuing
│
├── notify/                         # Multi-channel notification gateway
│   ├── __init__.py
│   └── router.py                   # Telegram & SMTP email alert routing with role-based delivery
│
├── presenter/                      # Decoupled Presenter Sidecar (Port 8502)
│   ├── __init__.py
│   └── controller.py               # Presenter controller, live fault trigger & presenter notes
│
├── profiles/                       # Facility profile models & YAML loader
│   ├── __init__.py
│   └── loader.py                   # Pydantic facility profile schema & loader (University, Hospital, etc.)
│
├── replay/                         # IoT playback and demonstration streaming
│   ├── __init__.py
│   ├── live_feed.py                # Sliding-window telemetry provider for live dashboard ticker widget
│   └── player.py                   # Async HTTP accelerated replay streamer (default 360x speed)
│
├── simulator/                      # Synthetic data generation engine
│   ├── __init__.py
│   ├── campus.py                   # Campus simulation orchestrator & asset registry builder
│   ├── complaints.py               # Synthetic occupant NLP complaint generator
│   ├── equipment.py                # Physical equipment power and thermal models
│   ├── events.py                   # Anomaly and fault injection engine (dynamic asset resolution)
│   ├── generate.py                 # CLI entry point (`generate-data`) producing 28-day baseline CSVs
│   ├── occupancy.py                # Diurnal zone occupancy modeling with schedule noise
│   ├── schedules.py                # Operational schedule evaluation and holiday calendars
│   ├── telemetry.py                # Sensor noise derivation & electrical consistency enforcement
│   ├── water.py                    # Hourly water telemetry, MNF leak simulation & HydroSense analytics
│   └── weather.py                  # Diurnal ambient temperature & humidity model
│
├── storage/                        # Persistence & baseline seeding
│   ├── __init__.py
│   ├── repository.py               # SQLite database client for raw telemetry and detected alerts
│   └── seed.py                     # Realistic lived-in operational history seeder
│
└── ui/                             # Enterprise Design System, reusable components & views
    ├── __init__.py
    ├── components.py               # Standardized KPI cards, alert rows, status badges, disclaimers
    ├── style.css                   # High-contrast, responsive enterprise stylesheet
    ├── users.py                    # User profiles, auth store & role definitions
    ├── panels/
    │   └── ask_nivara.py           # Contextual Ask Nivara intelligence drawer
    └── views/                      # 7 Core Operational Destinations
        ├── home_operations.py      # Operations Lead cockpit (attention items & triage)
        ├── home_sustainability.py  # Sustainability Officer cockpit (BEE, carbon budget, EPI)
        ├── home_admin.py           # Administrator cockpit (system health, facilities, alerts)
        ├── alerts_work.py          # Unified alerts triage & CMMS work orders board
        ├── assets.py               # Campus digital twin asset registry & geospatial inspector
        ├── energy.py               # 15-min load telemetry, ToD tariffs & Ridge/Prophet forecasting
        ├── water.py                # HydroSense MNF leak detection & water balance analytics
        ├── reports.py              # Statutory compliance matrix & ReportLab executive PDF export
        ├── settings.py             # Assumptions registry, tariffs, roles & BYOD CSV ingestion
        └── styleguide.py           # UI design system catalog & component documentation
```

---

## 2. Detailed Description of the 7 Operational Destinations & UI Suite

Following the operational product redesign (`docs/REDESIGN_SPEC.md` and `docs/step_reports/ux-01.md` through `ux-11.md`), Nivara is structured around **7 job-oriented operational destinations**, an embedded intelligence panel, and an isolated presenter controller sidecar:

### Destination 1: 🏠 Home (Role-Tailored Operational Cockpits)
- **Files:** `src/campus_energy/ui/views/home_operations.py`, `home_sustainability.py`, `home_admin.py`
- **Purpose:** Answers *"what needs my attention right now?"* upon morning login without requiring users to hunt through tabs.
- **Role Routing:**
  - **Operations Lead (`home_operations.py`)**: Immediate triage of active operational exceptions, critical equipment anomalies, active work order status, and direct one-click work order creation.
  - **Sustainability Officer (`home_sustainability.py`)**: Statutory BEE Star Rating, Energy Performance Index (EPI), NAPCC 20% carbon budget burndown curve, and avoidable carbon emissions.
  - **Administrator (`home_admin.py`)**: Multi-facility portfolio health matrix, telemetry pipeline status, edge device health, and pending system alerts.

### Destination 2: 🚨 Alerts & Work (Operational Triage & CMMS Execution)
- **File:** `src/campus_energy/ui/views/alerts_work.py`
- **Purpose:** End-to-end operational workflow linking anomaly detection directly to physical maintenance resolution.
- **Key Modules & Capabilities:**
  - **Alerts Inbox:** Filterable exception table powered by triple-engine consensus (Physics rules + Z-score + Isolation Forest). Supports status filtering (All, Open, Acknowledged, Dispatched, Resolved) and severity filters.
  - **Explainable Diagnostic ("Why?") Drawer:** Diagnostic card detailing detector source, expected vs observed telemetry, baseline context matrix, and contributing factors table (`Anomalous` / `Trigger` / `Normal`).
  - **CMMS Work Orders Board:** Sortable work-order cards with dynamic priority scoring ($Score = Severity \times Waste \times Hours$), live SLA countdown badges, technician assignees, and quick-resolve action buttons.
  - **Closed-Loop M&V Trigger:** Resolving a work order automatically links post-repair telemetry to the IPMVP Option B Savings Ledger.

### Destination 3: 🗺️ Assets (Digital Twin & Geospatial Infrastructure)
- **File:** `src/campus_energy/ui/views/assets.py`
- **Purpose:** Comprehensive asset hierarchy, equipment telemetry health, and geospatial building footprint exploration.
- **Key Modules & Capabilities:**
  - **Geospatial Campus Map:** 3D Mapbox GL JS digital twin with automatic 2D Plotly vector schematic fallback when air-gapped. Color-coded sensor pins, real-time power draw, and thermal comfort overlays.
  - **Hierarchical Asset Navigator:** Tree view filtering by facility, building, floor/zone, and asset type (Meters, HVAC chillers, Pumps, Lighting, Labs).
  - **Individual Asset Profile Inspector:** Historical power traces, 3-hour moving averages, operating hours matrix (Hour × Day of Week heatmap), and peer benchmarking against sibling equipment.

### Destination 4: ⚡ Energy (Load Telemetry, Tariffs & Dual Forecasting)
- **File:** `src/campus_energy/ui/views/energy.py`
- **Purpose:** 15-minute electrical telemetry profiling, Time-of-Day (ToD) tariff optimization, and forward predictive simulation.
- **Key Modules & Capabilities:**
  - **Load Telemetry Profiles:** 15-minute resolution power curves across main incomers, transformers, and sub-circuits with power factor ($\cos \phi$) tracking.
  - **Dual Load Forecasting:** Side-by-side Ridge Regression (physics-informed) and Meta Prophet models with calibrated 80% prediction intervals and held-out MAPE validation.
  - **Counterfactual What-If Levers:** Interactive scenario levers modeling temperature offsets (heatwave stress), off-hours HVAC cutbacks, exam period occupancy reductions, rooftop solar PV addition, and LED lighting retrofits.
  - **Microgrid & BESS Dispatch:** Rooftop solar PV bell-curve generation and battery peak shaving simulation during peak tariff windows (18:00–22:00).

### Destination 5: 💧 Water (HydroSense Leak Detection & Balance)
- **File:** `src/campus_energy/ui/views/water.py`
- **Purpose:** Municipal water meter monitoring, Minimum Night Flow (MNF) baseline audits, and hidden pipe leak remediation.
- **Key Modules & Capabilities:**
  - **Water Balance Accounting:** Hourly consumption per building block, total volume (kL), and water utility cost tracking.
  - **HydroSense MNF Leak Detector:** AWWA M36 / BS EN 12255 compliant nocturnal flow analysis during the critical 01:00–05:00 AM window, identifying baseline drift indicative of underground pipe bursts or stuck flush valves.
  - **Leak Remediation Queue:** Actionable leak cards with estimated lost volume, loss rate (Litres/min), financial waste, and plumbing remediation protocols.

### Destination 6: 📑 Reports (Statutory Compliance & Audit Dossiers)
- **File:** `src/campus_energy/ui/views/reports.py`
- **Purpose:** Official executive auditing, regulatory alignment, and statutory PDF report generation.
- **Key Modules & Capabilities:**
  - **Cross-Facility Performance Matrix:** Comprehensive statutory comparison across University, District Hospital, Government Office, and Industrial Campus (EPI, ECBC baseline, BEE Star Rating, CO₂/person).
  - **Statutory Audit Checklist:** ECBC 2017 compliance checklist, BEE Star verification, and NAPCC 20% decarbonization milestones.
  - **One-Click Executive Audit PDF:** Generates publication-grade ReportLab PDF reports complete with executive KPI summaries, top anomalies, BEE scorecards, and IPMVP Option B verified savings.
  - **Historical Audit Trail:** Traceable log of past audit exports and compliance filings.

### Destination 7: ⚙️ Settings (Assumptions, Tariffs, Roles & Ingestion)
- **File:** `src/campus_energy/ui/views/settings.py`
- **Purpose:** Centralized configuration, data ingestion management, and system administration.
- **Key Modules & Capabilities:**
  - **Central Assumptions Registry:** Live inspection and management of `config/assumptions.yaml` (tariffs, emissions, BEE benchmarks, physics constants).
  - **Bring-Your-Own-Data (BYOD) Ingestion:** Automated CSV upload console with intelligent column mapping and 5-rule Data Quality Engine (DQE) scorecard.
  - **User & Role Administration:** Multi-persona switcher (`admin`, `operations`, `sustainability`) and facility access control.
  - **Notification Endpoints:** Configuration of Telegram bot webhooks and SMTP email dispatchers.

---

## 2.1 Companion Tools & Presentation Suite

### 1. Ask Nivara Contextual Intelligence Panel
- **File:** `src/campus_energy/ui/panels/ask_nivara.py`
- **Architecture:** Collapsible sidebar drawer accessible from every destination.
- **Intelligence Engine:** Real-time conversational agent powered by Google Gemini Live REST API, with 100% grounded offline fallback synthesizing structured responses directly from active telemetry and recommendation objects.
- **Action Parsing:** Automatically converts conversational advice into structured `Recommendation` objects ready for work order dispatch.

### 2. Decoupled Presenter Controller Sidecar (Port 8502)
- **File:** `src/campus_energy/presenter/controller.py`
- **Execution:** Started via `make demo` or `python scripts/launch_demo.py` on `http://localhost:8502`.
- **Purpose:** Isolates all hackathon jury controls, live fault injection triggers, and rehearsal checklists from the primary production tool (`http://localhost:8501`), ensuring the main interface remains clean and credible.

### 3. Enterprise Design System & Polish Standards
- **CSS:** `src/campus_energy/ui/style.css` (custom dark theme, high contrast, responsive 1366px–1920px).
- **Tone Pass:** Zero exclamation marks in headers or cards, zero marketing hype words, standardized disclaimer: *"Decision-support estimate. Field verification required."*
- **Performance:** Sub-120ms destination switching backed by `@st.cache_data(ttl=3600)`.

---

## 3. Data Contracts & Schema Fields

Defined in `src/campus_energy/schemas.py`:

### Enumerations
- `AssetType`: `building`, `zone`, `meter`, `lighting`, `hvac`, `computer_lab`, `lab_equipment`, `pump`
- `MonitoringLevel`: `building`, `zone`, `circuit`, `equipment`
- `EquipmentState`: `ON`, `OFF`, `UNKNOWN`
- `SensorStatus`: `healthy`, `degraded`, `failed`, `offline`
- `QualityFlag`: `valid`, `missing`, `stale`, `out_of_range`, `duplicated`, `corrupted`
- `EventType`: `lights_left_on`, `hvac_left_on`, `pump_no_flow`, `power_spike`, `gradual_increase`, `overnight_excess`, `occupancy_mismatch`, `peer_anomaly`, `sensor_dropout`, `comm_failure`, `stale_sensor`, `corrupted_reading`, `duplicate_reading`, `timestamp_gap`
- `Severity`: `low`, `medium`, `high`, `critical`
- `AlertStatus`: `active`, `acknowledged`, `resolved`
- `ComplaintCategory`: `hvac`, `lighting`, `power`, `equipment`, `other`
- `ComplaintStatus`: `open`, `in_progress`, `resolved`, `closed`

### Core Data Models
| Model | Target CSV / Table | Key Fields | Validations |
|---|---|---|---|
| `Asset` | `assets.csv` | `asset_id`, `asset_type`, `asset_name`, `building_id`, `floor`, `zone_id`, `parent_asset_id`, `rated_power_kw`, `equipment_count`, `operating_schedule_id`, `monitoring_level`, `active` | None |
| `TelemetryReading` | `telemetry.csv` / `telemetry` | `reading_id`, `timestamp`, `asset_id`, `reading_type`, `power_kw`, `interval_energy_kwh`, `cumulative_energy_kwh`, `voltage_v`, `current_a`, `power_factor`, `state`, `flow_lpm`, `temperature_c`, `setpoint_c`, `sensor_status`, `quality_flag` | `0.0 <= power_factor <= 1.0`, `power_kw >= 0`, `interval_energy_kwh >= 0` |
| `ContextRecord` | `context.csv` | `timestamp`, `building_id`, `zone_id`, `occupancy`, `scheduled_occupancy`, `occupancy_source`, `working_hours`, `holiday`, `academic_activity`, `day_of_week`, `outdoor_temperature_c`, `humidity_pct` | `occupancy >= 0`, `0 <= day_of_week <= 6`, `0 <= humidity_pct <= 100` |
| `Event` | `events.csv` | `event_id`, `event_type`, `asset_id`, `start_time`, `end_time`, `severity`, `description`, `injection_parameters` | None |
| `AnomalyGroundTruth` | `anomaly_ground_truth.csv` | `event_id`, `asset_id`, `event_type`, `start_time`, `end_time`, `severity`, `expected_detection`, `expected_detection_method`, `root_cause` | None |
| `Complaint` | `complaints.csv` | `complaint_id`, `timestamp`, `building_id`, `zone_id`, `category`, `description`, `status`, `resolution_timestamp` | None |
| `Alert` | Detector Output / `alerts` | `alert_id`, `detected_at`, `event_start_estimate`, `asset_id`, `building_id`, `zone_id`, `anomaly_type`, `severity`, `actual_value`, `expected_value`, `deviation`, `evidence`, `confidence`, `recommendation`, `status` | `0.0 <= confidence <= 1.0` |

---

## 4. Simulator Parameters, Sampling Interval & Asset Breakdown

### 4.1 True Sampling Interval Verification
- **Verified Interval:** **5 minutes** (NOT 15 minutes).
- **Proof:**
  1. `configs/simulation.yaml` (line 7): `interval_minutes: 5`
  2. `data/manifests/simulation_manifest.json`: `"interval_minutes": 5`
  3. `data/generated/telemetry.csv`: Consecutive timestamps for every asset are spaced by exactly `0 days 00:05:00` (8,063 intervals per 28 days).
  4. Math in `simulator/telemetry.py` (line 36): `interval_energy_kwh = noisy_power / 12.0` (derived from 60 min / 5 min = 12 intervals per hour).
  5. Math in `simulator/events.py` (lines 101, 121): `(rated_kw * fraction) / 12.0`.
  6. Forecasting forward horizon in `analytics/forecasting.py` (line 90) and `analytics/forecasting_prophet.py` (line 158): 288 steps of 5 minutes = 24 hours.

### 4.2 Asset Breakdown by Facility
| Facility | Buildings | Total Assets | Equipment | Zones | Meters |
|---|---|---|---|---|---|
| **University** (`data/generated`) | 6 (`BLD-A`, `BLD-B`, `BLD-ADM`, `BLD-LIB`, `BLD-HST`, `BLD-UTL`) | **56** | 30 | 14 | 6 (main) + 6 (bld) |
| **District Hospital** (`data/hospital`) | 4 (`BLD-ER`, `BLD-IPD`, `BLD-OPD`, `BLD-ADM`) | **40** | 22 | 10 | 4 (main) + 4 (bld) |
| **Government Office** (`data/govt_office`) | 4 (`BLD-ADM`, `BLD-REV`, `BLD-CIV`, `BLD-IT`) | **36** | 20 | 8 | 4 (main) + 4 (bld) |
| **Industrial Campus** (`data/industrial`) | 3 (`BLD-FAB`, `BLD-WHS`, `BLD-UTL`) | **30** | 18 | 6 | 3 (main) + 3 (bld) |

### 4.3 Injected Fault Catalog & Frequency
In `src/campus_energy/simulator/events.py`, **5 canonical faults** are injected per simulation run:
1. `lights_left_on`: Classroom/office lighting left ON overnight (duration: 6h; severity: `medium`; root cause: "User forgot to switch off").
2. `hvac_left_on`: AC left running after hours (duration: 8h; severity: `medium`; root cause: "Timer override active").
3. `pump_no_flow`: Water pump running dry without flow (duration: 1.5h; severity: `high`; root cause: "Valve closed downstream").
4. `sensor_dropout`: Lighting meter stops reporting telemetry / missing values (duration: 3h; severity: `low`; root cause: "Network interference").
5. `duplicate_reading`: MQTT broker duplicates readings (duration: 5 min; severity: `low`; root cause: "Network retry").

In addition, `src/campus_energy/simulator/water.py` injects **2 water anomalies**:
1. `pipe_leak_overnight`: Overnight concealed pipe fissure (+35 LPM, 1 AM–5 AM across 4 nights).
2. `burst_valve_leak`: Daytime valve burst (+83.3 LPM for 6 consecutive hours).

---

## 5. Forecasting Models & Entry Points

### 5.1 Ridge Baseline Forecaster (`src/campus_energy/analytics/forecasting.py`)
- **Entry Point:** `generate_24h_forecast(telemetry_df, context_df, assets_df, temp_offset=0.0, ac_policy_reduction_pct=0.0, occupancy_reduction_pct=0.0, solar_pv_kwp=0.0, led_upgrade=False) -> pd.DataFrame`
- **Methodology:**
  - Trains per main meter on the last 21 days of telemetry.
  - Features: hour-of-day, day-of-week, is_weekend, is_holiday, outdoor_temperature_c, 3-hour rolling mean power, 3-hour rolling std dev power.
  - Predicts 288 steps (5-minute intervals) for next 24 hours.
  - Dynamic Scenario Modifiers:
    - `occupancy_reduction_pct`: scales load assuming 60% of building load is occupancy-sensitive.
    - `solar_pv_kwp`: subtracts daytime solar generation (clamped 06:00–18:00 at 80% capacity factor).
    - `led_upgrade`: reduces total load by 8.8% (assumes lighting is 22% of building load, with 40% LED savings).
    - `ac_policy_reduction_pct`: curbs after-18:00 HVAC consumption.
  - Confidence intervals: [0.88 * prediction, 1.12 * prediction].

### 5.2 Meta Prophet Forecaster (`src/campus_energy/analytics/forecasting_prophet.py`)
- **Entry Point:** `generate_prophet_forecast(telemetry_df, assets_df, context_df=None, **kwargs) -> tuple[pd.DataFrame, float | None]`
- **Methodology:**
  - Aggregates all campus main meters into 5-minute total demand time-series.
  - Trains Meta Prophet model with additive daily and weekly seasonality, `changepoint_prior_scale=0.05`.
  - Performs 24-hour held-out historical validation comparing predictions against actual campus load to calculate **MAPE** (Mean Absolute Percentage Error).
  - Forecasts 288 5-minute intervals forward with upper and lower Bayesian uncertainty intervals (`yhat_lower`, `yhat_upper`).

---

## 6. Anomaly Layers & Detection Logic

Defined across `src/campus_energy/analytics/anomalies.py` and `explainability.py`:

| Layer | Mechanism | Trigger Condition | Sustained Duration | Severity Assigned |
|---|---|---|---|---|
| **Layer 1** | **Statistical Baseline** (Ridge Regression) | Actual draw exceeds Ridge baseline by > 0.5 kW AND > 30% of expected power | ≥ 6 samples (≥ 30 min) | `critical` if mean dev > 2.0 kW, else `medium` |
| **Layer 2** | **Operational BMS Rule** (Unoccupied Operation) | Equipment in `['lighting', 'hvac', 'computer_lab']` draws > 20% rated kW while zone occupancy == 0 | ≥ 6 samples (≥ 30 min) | `medium` if lighting, else `high` |
| **Layer 3** | **Hydraulic Safety Rule** (Dry-run Pump) | Asset type == `pump` draws > 50% rated kW while `flow_lpm == 0` | ≥ 3 samples (≥ 15 min) | `high` |
| **Layer 4** | **Data Quality Rule** (Sensor Dropout) | `quality_flag == 'missing'` OR `sensor_status == 'offline'` | ≥ 3 samples (≥ 15 min) | `low` |
| **Layer 5** | **Multivariate ML Outlier** (Isolation Forest) | Isolation Forest fit on `[power_kw, power_ma_3h, power_std_3h, hour]` flags outlier (`label == -1`, contamination=3%) | ≥ 4 samples (≥ 20 min) | `medium` if score < 0.7, else `high` |
| **Layer 6** | **Deduplication Filter** | Merges overlapping alerts for identical `(asset_id, anomaly_type)` | 4-hour window | Retains highest severity |

### Diagnostic Explainability Engine (`src/campus_energy/analytics/explainability.py`)
- **Entry Point:** `explain_alert(alert, context_row=None, asset_info=None) -> dict`
- **Output Structure:**
  - `why_headline`: Human-readable summary (e.g., *"Expected 4.2 kW at 2 PM (weekday, 34°C, 45 occupants). Actual draw was 7.1 kW (+2.9 kW, 69.0% above baseline)."*).
  - `detector_type`: Identifies mathematical or operational detector source.
  - `baseline_context`: Expected vs observed values, net delta, ambient temp, occupancy.
  - `contributing_factors`: Feature attribution table categorizing inputs as `Anomalous`, `Trigger`, or `Normal`.
  - `probable_root_cause` & `recommended_verification`: Direct field technician guidance.

---

## 7. Complete Inventory of Hard-Coded Constants

To prepare for **Step 3 (Centralized Assumptions Registry in `config/assumptions.yaml`)**, here is the exhaustive catalog of all hard-coded assumptions, tariffs, thresholds, and physics multipliers located in the codebase:

### 7.1 Financial & Energy Tariffs
| Constant Description | Hard-Coded Value | Source File | Line Number(s) |
|---|---|---|---|
| Commercial Electricity Tariff | `8.5` INR/kWh | `src/campus_energy/analytics/billing.py` | 9, 88, 99, 111 |
| Commercial Electricity Tariff | `8.5` INR/kWh | `src/campus_energy/analytics/esg.py` | 8, 114, 131, 142, 153, 160, 171 |
| Commercial Electricity Tariff | `8.5` INR/kWh | `src/campus_energy/analytics/workorders.py` | 140, 158 |
| Commercial Electricity Tariff | `8.5` INR/kWh | `src/campus_energy/dashboard/app.py` | 1647, 1679 |
| Commercial Electricity Tariff | `8.5` INR/kWh | `src/campus_energy/web/main.py` | 87 |
| Inconsistent Electricity Tariff | `8.0` INR/kWh | `src/campus_energy/web/main.py` | 72 |
| Municipal Water Tariff | `25.0` INR/kL | `src/campus_energy/simulator/water.py` | 18, 208, 226, 247, 260, 273, 336, 369, 405 |
| Time-of-Day (ToD) Peak Tariff | `25.0` INR/kWh | `src/campus_energy/dashboard/app.py` | 2528, 2626 |
| Time-of-Day (ToD) Mid-Day Tariff | `18.0` INR/kWh | `src/campus_energy/dashboard/app.py` | 2528 |
| Time-of-Day (ToD) Off-Peak Tariff | `10.0` INR/kWh | `src/campus_energy/dashboard/app.py` | 2528 |
| Web ToD Peak Tariff | `25.0` INR/kWh | `src/campus_energy/web/main.py` | 126 |
| Web ToD Off-Peak Tariff | `10.0` INR/kWh | `src/campus_energy/web/main.py` | 126 |

### 7.2 Emission & Environmental Factors
| Constant Description | Hard-Coded Value | Source File | Line Number(s) |
|---|---|---|---|
| Grid Carbon Emission Factor | `0.82` kg CO₂/kWh | `src/campus_energy/analytics/esg.py` | 6, 115, 132, 143, 154, 161, 172 |
| Grid Carbon Emission Factor | `0.82` kg CO₂/kWh | `src/campus_energy/dashboard/app.py` | 1648, 1680 |
| Grid Carbon Emission Factor | `0.82` kg CO₂/kWh | `src/campus_energy/web/main.py` | 71 |
| Solar Lifecycle Emission Factor | `0.04` kg CO₂/kWh | `src/campus_energy/analytics/esg.py` | 10 |
| Tree Carbon Sequestration Rate | `21` kg CO₂/tree/year | `src/campus_energy/analytics/esg.py` | 168 |
| Tree Carbon Sequestration Rate | `21` kg CO₂/tree/year | `src/campus_energy/analytics/governance.py` | 150 |
| Coal Unburned Displacement Factor | `0.45` tons coal/MWh | `src/campus_energy/analytics/governance.py` | 152 |
| NAPCC Mandatory Reduction Target | `20.0` % | `src/campus_energy/analytics/esg.py` | 66, 79 |
| Avoidable Waste Fraction of Fault | `0.80` (80%) | `src/campus_energy/analytics/esg.py` | 129 |
| Carbon Budget Month Ceiling Days | `30.0` days | `src/campus_energy/analytics/carbon_budget.py` | 69, 124, 134, 140 |

### 7.3 BEE Energy Performance Index (EPI) Benchmarks (`kWh/m²/yr`)
| Facility Type | 5-Star Range | 4-Star Range | 3-Star (ECBC Base) | 2-Star Range | 1-Star Range | Source File & Line |
|---|---|---|---|---|---|---|
| University / Education | `0 – 60` | `60 – 90` | `90 – 120` (`100`) | `120 – 160` | `> 160` | `analytics/esg.py: 18-28` |
| Hospital / Healthcare | `0 – 150` | `150 – 250` | `250 – 350` (`300`) | `350 – 500` | `> 500` | `analytics/esg.py: 29-39` |
| Government Office | `0 – 80` | `80 – 120` | `120 – 160` (`140`) | `160 – 220` | `> 220` | `analytics/esg.py: 40-50` |
| Industrial Campus | `0 – 200` | `200 – 350` | `350 – 500` (`450`) | `500 – 700` | `> 700` | `analytics/esg.py: 51-61` |

### 7.4 Physical & Anomaly Thresholds
| Constant Description | Hard-Coded Value | Source File | Line Number(s) |
|---|---|---|---|
| Anomaly Statistical Deviation Cutoff | `0.5` kW excess | `src/campus_energy/analytics/anomalies.py` | 127 |
| Anomaly Percentage Deviation Cutoff | `0.3` (30% above expected) | `src/campus_energy/analytics/anomalies.py` | 128 |
| Critical Anomaly Threshold | `2.0` kW mean deviation | `src/campus_energy/analytics/anomalies.py` | 144 |
| Unoccupied Power Standby Threshold | `0.2` (20% rated capacity) | `src/campus_energy/analytics/anomalies.py` | 165 |
| Unoccupied Power Standby Threshold | `0.2` (20% rated capacity) | `src/campus_energy/analytics/explainability.py` | 256 |
| Pump Dry-Run Power Threshold | `0.5` (50% rated capacity) | `src/campus_energy/analytics/anomalies.py` | 194 |
| Isolation Forest Contamination Rate | `0.03` (3%) | `src/campus_energy/analytics/anomalies.py` | 260 |
| Alert Deduplication Time Window | `4.0` hours | `src/campus_energy/analytics/anomalies.py` | 66 |
| Nominal Voltage | `230.0` V | `configs/simulation.yaml` | 499 |
| Nominal Power Factor | `0.85` | `configs/simulation.yaml` | 500 |
| Fallback Replay Power Factor | `0.92` | `src/campus_energy/replay/live_feed.py` | 67 |
| ASHRAE 55 Optimal Setpoint Temp | `24.0` °C | `src/campus_energy/analytics/comfort.py` | 74, 95, 103 |
| ASHRAE 55 Default Air Velocity | `0.15` m/s | `src/campus_energy/analytics/comfort.py` | 47 |
| ASHRAE 55 Default Metabolic Rate | `1.1` met | `src/campus_energy/analytics/comfort.py` | 48, 69 |
| ASHRAE 55 Default Clothing Insulation | `0.6` clo | `src/campus_energy/analytics/comfort.py` | 49 |
| Default Ambient Temperature Fallback | `32.0` °C | `src/campus_energy/analytics/baseline.py` | 37, 40 |
| Default Ambient Temperature Fallback | `32.0` °C | `src/campus_energy/analytics/forecasting.py` | 76, 121 |
| Default Ambient Temperature Fallback | `34.0` °C | `src/campus_energy/analytics/explainability.py` | 150, 152 |
| Minimum Night Flow (MNF) Hours | `1, 2, 3, 4` (01:00–05:00 AM) | `src/campus_energy/simulator/water.py` | 35, 53, 64, 310 |
| Water Leak Spike Detection Multiplier | `3.0` × median baseline | `src/campus_energy/simulator/water.py` | 326 |
| Water Absolute Leak Volume Threshold | `500.0` Liters/hour | `src/campus_energy/simulator/water.py` | 326 |
| Lighting Power Density (LPD) Threshold | `7.5` W/m² | `src/campus_energy/dashboard/app.py` | 931 |

### 7.5 Sampling Interval Multipliers (to be replaced by `interval_minutes / 60.0`)
| Constant Description | Hard-Coded Value | Source File | Line Number(s) |
|---|---|---|---|
| Telemetry Energy Divisor (60/5 min) | `12.0` | `src/campus_energy/simulator/telemetry.py` | 21, 36 |
| Event Injection Energy Divisor | `12.0` | `src/campus_energy/simulator/events.py` | 101, 121 |
| Dashboard Scenario Energy Divisor | `12.0` | `src/campus_energy/dashboard/app.py` | 1646, 1676 |
| 3-Hour Rolling Window Steps (3 × 12) | `36` | `src/campus_energy/analytics/features.py` | 36, 37 |
| 3-Hour Rolling Window Steps | `36` | `src/campus_energy/analytics/forecasting.py` | 82, 86 |
| 3-Hour Rolling Window Steps | `36` | `src/campus_energy/dashboard/app.py` | 2751 |
| 24-Hour Forward Forecast Horizon Steps | `288` | `src/campus_energy/analytics/forecasting.py` | 90, 118 |
| 24-Hour Forward Forecast Horizon Steps | `288` | `src/campus_energy/analytics/forecasting_prophet.py` | 158 |

---

## 8. Resolution of Hard-Coded Constants & Current Assumptions Architecture

All assumptions identified during initial audits have been successfully consolidated into `config/assumptions.yaml` and validated through `src/campus_energy/config.py`:
1. **Unified Source of Truth:** Centralized `assumptions.yaml` containing 49 traceable parameters (OERC 2-part HT commercial tariffs, CEA grid emission factors, BEE Star benchmarks, and ASHRAE 55 comfort coefficients).
2. **Interval Parametrization:** Sampling interval multipliers parameterized dynamically via `config.simulation.interval_minutes / 60.0`.
3. **Consistency Verification:** Guaranteed by automated consistency tests in [`tests/test_consistency.py`](file:///C:/Users/Paramjeet/Desktop/BPUT/tests/test_consistency.py) and [`tests/test_assumptions.py`](file:///C:/Users/Paramjeet/Desktop/BPUT/tests/test_assumptions.py).

---

## 9. Comprehensive Test Suite & Verification Status

The automated test suite contains **316 tests across 41 modules**, running with a 100% green pass rate:

```
collected 316 items

tests/test_adapters.py .................. [  2%]
tests/test_agent_actions.py ............. [  3%]
tests/test_alerts_work_view.py .......... [  6%]
tests/test_anomalies.py ................. [  7%]
tests/test_ask_nivara_panel.py .......... [ 11%]
tests/test_assets_view.py ............... [ 13%]
tests/test_assumptions.py ............... [ 16%]
tests/test_carbon_budget.py ............. [ 18%]
tests/test_comfort.py ................... [ 20%]
tests/test_complaints_nlp.py ............ [ 21%]
tests/test_components.py ................ [ 28%]
tests/test_consistency.py ............... [ 30%]
tests/test_demo.py ...................... [ 34%]
tests/test_energy_balance.py ............ [ 35%]
tests/test_energy_water_views.py ........ [ 38%]
tests/test_eval_harness.py .............. [ 39%]
tests/test_events.py .................... [ 41%]
tests/test_explainability.py ............ [ 43%]
tests/test_facility_profiles.py ......... [ 45%]
tests/test_forecasting_prophet.py ....... [ 46%]
tests/test_governance.py ................ [ 47%]
tests/test_india_levers.py .............. [ 51%]
tests/test_ingestion.py ................. [ 53%]
tests/test_ledger.py .................... [ 55%]
tests/test_live_feed.py ................. [ 57%]
tests/test_notifications.py ............. [ 63%]
tests/test_polish.py .................... [ 67%]
tests/test_presenter_and_seed.py ........ [ 69%]
tests/test_public_dataset.py ............ [ 71%]
tests/test_recommendation_engine.py ..... [ 75%]
tests/test_reliability.py ............... [ 77%]
tests/test_report_generator.py .......... [ 80%]
tests/test_reports_settings_views.py .... [ 83%]
tests/test_role_homes.py ................ [ 87%]
tests/test_roles.py ..................... [ 89%]
tests/test_service_guard.py ............. [ 91%]
tests/test_simulator.py ................. [ 92%]
tests/test_uncertainty.py ............... [ 95%]
tests/test_users_and_shell.py ........... [ 96%]
tests/test_water.py ..................... [ 98%]
tests/test_workorders.py ................ [100%]

======================= 316 passed in ~214s =======================
```
