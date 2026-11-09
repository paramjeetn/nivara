# Nivara — Platform State & Codebase Architecture (Step 2 Audit)

> **Document Status:** Comprehensive Codebase State as of Step 2 Audit  
> **Date:** November 2026  
> **Platform:** Nivara Campus Energy Intelligence Platform  
> **Test Suite Status:** 76 / 76 passing (`pytest tests/`)

---

## 1. Full Package Layout

The codebase follows a modular Python structure under `src/campus_energy/`:

```
src/campus_energy/
├── __init__.py                     # Package entry point and version metadata
├── config.py                       # Pydantic simulation & campus configuration schema & YAML loader
├── schemas.py                      # Pydantic canonical data contracts for all data models & enums
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
│   ├── forecasting.py              # 24h forward Ridge forecaster with 5 scenario simulation levers
│   ├── forecasting_prophet.py      # Campus-level Meta Prophet forecaster with held-out MAPE validation
│   ├── governance.py               # Cross-facility benchmarking, BEE matrix & state-level scaling simulator
│   ├── report_generator.py         # ReportLab PDF executive audit report generator
│   └── workorders.py               # CMMS work-order dispatch queue with multi-factor priority scoring
│
├── dashboard/                      # Interactive user interface
│   ├── __init__.py
│   ├── agent.py                    # Google Gemini LLM API client & structured action recommendation parser
│   ├── app.py                      # 10-page Streamlit multi-facility operational dashboard
│   └── osm_buildings.py            # OpenStreetMap GeoJSON building footprint extractor
│
├── ingestion/                      # Edge telemetry ingestion pipeline
│   ├── __init__.py
│   └── api.py                      # FastAPI REST ingestion service with batch validation & SQLite queuing
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
├── storage/                        # Persistence layer
│   ├── __init__.py
│   └── repository.py               # SQLite database client for raw telemetry and detected alerts
│
└── web/                            # Alternative lightweight web service
    ├── main.py                     # Secondary FastAPI web app with JSON endpoints
    └── templates/
        └── index.html              # HTML/JS static dashboard template
```

---

## 2. Detailed Description of Each Streamlit Dashboard Page

The dashboard (`src/campus_energy/dashboard/app.py`) features **10 dedicated operational pages** selectable via the sidebar navigation:

### Page 1: 📊 Executive Overview (ESG & Sustainability)
- **Purpose:** C-suite and facility director executive briefing on energy consumption, environmental footprint, compliance, and budget burn-down.
- **Displayed Metrics & Cards:**
  - 5 Top KPIs: Total Energy (kWh), Total Cost (₹), Carbon Emitted (t CO₂), Avoidable Waste (kWh, %), Potential Financial Savings (₹).
  - BEE Energy Performance Index (EPI) Card: Star Rating (1–5 stars), Annual Energy Intensity (`kWh/m²/yr`), Comparison vs ECBC Baseline (`%` deviation), and NAPCC 20% Reduction Target.
  - Interactive EPI Benchmark Bar Chart: Compares facility's current EPI directly against the ECBC Baseline and NAPCC Target.
  - Environmental Impact Callout: Translates total emissions into tree planting equivalents (~21 kg CO₂/tree/yr).
  - Institutional Carbon Budget & NAPCC Net-Zero Tracker: Progress bar tracking monthly carbon budget utilization, 4 metric cards (Monthly Target, Emitted to Date, Projected Month-End, Projected Overshoot / Remaining Budget), status alert banner, and a 30-day Plotly burn-down curve (actual vs pro-rated linear target vs 30-day projection vs budget ceiling).
  - Annualized projection row (`kWh/yr`, `₹/yr`, `t CO₂/yr`).
  - Daily energy consumption bar chart with 3-day moving average overlay.
  - Building energy distribution donut/pie chart.
  - Daily carbon footprint area trend chart (`kg CO₂`).
  - Report Exports: One-click CSV export and a professional ReportLab PDF download button.

### Page 2: 🏛️ Governance & Policy (Multi-Facility Benchmarking)
- **Purpose:** Multi-facility comparison, statutory compliance tracking under ECBC 2017, and state-wide scaling impact simulation.
- **Displayed Metrics & Cards:**
  - Cross-Facility Performance Matrix: Standardized table comparing University, District Hospital, Government Office, and Industrial Campus across 14 statutory fields (Floor Area, Occupancy, Energy, Cost, EPI, ECBC Baseline, BEE Rating, Status, Carbon, Carbon/Occupant, Avoidable Waste, Potential Savings).
  - Side-by-side grouped bar chart: Actual EPI vs ECBC Baseline across facilities.
  - Carbon Intensity Bar Chart: Annual CO₂ per peak occupant (`t CO₂/person`).
  - Carbon vs Floor Area Bubble Plot (bubble size proportional to annual MWh).
  - State-Level Scaling Impact Simulator: Interactive slider (10 to 500 facilities) projecting collective state-level outcomes: Total MWh avoided, Financial savings (₹ Crores/yr), Carbon mitigated (t CO₂/yr), Trees planted equivalent, and Metric tons of thermal coal unburned.
  - Policy alignment callout: Details compliance with NAPCC, NMEEE, and BEE PAT Scheme (ESCerts).
  - Mandatory ECBC Compliance Checklist: 5 statutory audit checkboxes with automatic compliance percentage scoring.

### Page 3: 🗺️ 3D Campus Map (Real-Time Energy & Thermal Heatmap)
- **Purpose:** Interactive geospatial 3D digital twin visualizing electrical load distribution and indoor comfort across physical campus blocks.
- **Displayed Metrics & Cards:**
  - View Mode Toggle: Switch between "Power Draw & Energy (kW)" and "Thermal Comfort (ASHRAE 55)".
  - Telemetry Clock Mode: Toggle between Live Real-Time (anchored to current clock) and Simulation Dataset Date.
  - Interactive Mapbox GL JS 3D Canvas: True OSM 3D extruded building geometry, custom building textures, color-coded sensor pins, glowing alert pulses on active equipment faults, and zone polygons.
  - Thermal Comfort Summary Metrics: Campus comfort categorization, Average PMV, Percentage of Dissatisfied Occupants (PPD %), Thom's Discomfort Index (DI), and comfort distribution breakdown.
  - Interactive Drawer / Inspector: Clicking any asset pin opens an inspector displaying real-time power draw, voltage, power factor, active alerts, and diagnostic recommendations.

### Page 4: 🤖 AI Forecasting & Scenario Simulator
- **Purpose:** 24-hour forward-looking predictive energy forecasting and interactive what-if policy exploration.
- **Displayed Metrics & Cards:**
  - Scenario Controls Sidebar:
    1. Temperature Offset slider (-5.0°C to +5.0°C) for heatwave modeling.
    2. HVAC Cutback slider (0% to 40% reduction after 18:00).
    3. Occupancy Reduction slider (0% to 80% for exam/break periods).
    4. Rooftop Solar PV addition slider (0 to 500 kWp).
    5. LED Lighting Retrofit toggle (models 40% reduction in lighting load).
  - 24-Hour Campus Load Forecast Chart: Shaded 88%–112% uncertainty confidence band, baseline Ridge prediction curve, and scenario-adjusted load curve.
  - Scenario Impact Summary: Grid Import Reduction (kWh), Cost Savings (₹), and Carbon Mitigated (kg CO₂).
  - Model Benchmark (Prophet vs. Ridge): Side-by-side held-out MAPE metric cards, dual-model forecast visualization with Prophet confidence interval vs. Ridge baseline.
  - Per-building forecasted peak load ranking table with heatmap background.

### Page 5: 🚨 Alerts & NLP (Anomaly Detection & Dispatch)
- **Purpose:** High-resolution anomaly triage, human-interpretable technical diagnostics, human complaint correlation, and CMMS work order generation.
- **Displayed Metrics & Cards:**
  - Severity Filter: Multi-select filtering critical, high, medium, and low alerts.
  - Active Alert Expanders:
    - Basic telemetry metadata: Asset ID, building, zone, actual vs expected kW, detection confidence bar.
    - AI Diagnostic ("Why?") Card: Identifies detection mechanism (Ridge baseline, BMS rule, hydraulic rule, or Isolation Forest) and structured plain-English diagnosis.
    - Baseline Context Matrix: Expected kW, Observed kW, Net deviation (kW and %), Ambient Temperature, and Zone Occupancy at detection time.
    - Contributing Factors Table: Status flag (`Anomalous` / `Trigger` / `Normal`) and technical detail for each variable.
    - Probable Root Cause & Physical Verification Protocol.
    - Correlated Human Complaints: Bubble cards matching occupant complaints from the same building and time window.
    - Action Dispatch Buttons: "Acknowledge" and "Dispatch Team".
  - Alert Analytics: Severity distribution pie chart, Anomaly types horizontal bar chart, and Complaint categories donut chart.
  - NLP Complaint Intelligence: TF-IDF + KMeans clustering of complaint texts, most recurring issue banner, theme count bar chart, and top cluster terms table.
  - Maintenance Work-Order & Fault Priority Queue:
    - 4 Summary KPIs: Active Work Orders count, Daily Energy Hemorrhage (kWh/day), Daily Waste Cost (₹/day), and Top Priority Asset.
    - Daily waste by assigned maintenance crew stacked bar chart.
    - Filterable work-order cards with dynamic priority score (0–100), SLA target countdown badge, estimated daily waste rate, recommended action, and "Quick Resolve" button.
    - Archive of resolved orders with "Reopen" functionality.

### Page 6: 💧 Water Intelligence (HydroSense)
- **Purpose:** Municipal water consumption tracking, Minimum Night Flow (MNF) baseline audits, and hidden pipe leak remediation.
- **Displayed Metrics & Cards:**
  - 5 Top KPIs: Total Water Consumed (kL), Water Utility Cost (₹ at ₹25/kL), Active Leaks count, Avoidable Water Loss (kL and ₹), and Night Leak LPM rate.
  - Daily Water Consumption Trend bar chart with 3-day moving average overlay.
  - Building Water Share donut chart showing volume and percentage per block.
  - Leak Alerts & Anomaly Remediation Queue: Visual alert cards highlighting severity, start time, duration, estimated lost volume, loss cost, and plumbing remediation recommendations.
  - Diurnal Flow Curve (AWWA M36 / BS EN 12255): 24-hour hourly flow profile with shaded red callout box highlighting the critical 01:00–05:00 AM Minimum Night Flow (MNF) window comparing actual flow against normal non-leak baseline.
  - MNF audit takeaway card highlighting potential billing cycle savings.

### Page 7: 🔋 Microgrid Dispatch (Solar PV + BESS)
- **Purpose:** Peak shaving and microgrid dispatch simulation combining rooftop solar generation and Battery Energy Storage Systems (BESS).
- **Displayed Metrics & Cards:**
  - Interactive Parameter Configuration: Solar array capacity (kW), Battery capacity (kWh), and Initial State of Charge (SOC %).
  - Real-time Microgrid Dispatch Strategy Chart: Displays campus gross load, solar PV bell-curve generation, optimized grid import curve, and battery charge/discharge bar chart across 24 hours.
  - Battery State of Charge (%) Chart: Hourly trajectory with 20% minimum SOC threshold and 90% maximum SOC threshold guidelines.
  - 4 Key Dispatch KPIs: Optimized Grid Import vs baseline (kWh), Solar Energy Utilized (% of load), Peak Load Shaved (kW and % reduction), and Daily Battery Degradation Cycles.
  - AI Peak Shaving Insight: Quantifies cost savings during the peak utility tariff window (16:00–21:00).

### Page 8: 💸 Department Billing & Chargebacks
- **Purpose:** Sub-meter energy allocation to academic and operational departments, eliminating flat floor-area splits.
- **Displayed Metrics & Cards:**
  - AI After-Hours Alert Banner: Identifies the department with the highest consumption between 19:00 and 06:00.
  - Department Ledger: 28-day breakdown of metered energy consumption (`total_kwh`) and billed cost (`cost_inr` at ₹8.50/kWh) with gradient highlighting.
  - One-click CSV billing export button.
  - Cost Distribution Donut Chart: Departmental share of total electricity expenditure.
  - Daily Stacked Energy Consumption Area Chart by department over time.
  - Building-level metered cost summary table.

### Page 9: 🔍 Asset Detail Explorer
- **Purpose:** Deep-dive operational profiling and diagnostic inspection of individual sub-meters and equipment assets.
- **Displayed Metrics & Cards:**
  - Asset Selector: Filter by asset type and asset dropdown.
  - Metadata Table: Asset ID, type, building, zone, rated power, equipment count, and monitoring level.
  - Power Consumption Trace: High-resolution time-series line chart with 3-hour rolling average curve and shaded event rectangles indicating anomalous injection periods.
  - Descriptive Summary KPIs: Mean power (kW), Peak power (kW), Minimum power (kW), and Total Energy (kWh).
  - Hour × Day of Week Heatmap: Matrix heatmap identifying peak operating hours across weekdays and weekends.
  - Peer Comparison Chart: Bar chart benchmarking the asset's average power against sibling equipment of the same type in the same building.

### Page 10: 💬 AI Facility Assistant (Gemini)
- **Purpose:** Natural language conversational assistant grounded in real-time facility telemetry, active alerts, and complaints.
- **Displayed Metrics & Cards:**
  - Gemini API Key secure password input.
  - Quick Prompt Buttons: "What are the most critical anomalies right now?", "Which department is wasting the most energy?", "How can we reduce peak demand tomorrow?", etc.
  - Interactive Chat History Stream: Real-time markdown responses from Gemini.
  - Pending Action Queue Table: Automatically extracts and parses structured action items from LLM responses into a CMMS-ready task queue (`action_type`, `target_asset`, `suggested_time`, `estimated_saving_kwh`, `status`, `detected_at`).
  - Action Queue management controls: "Clear Action Queue" and "Clear Chat".

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

## 8. Summary of Findings for Step 3 Implementation

1. **Assumptions Separation Required:** All business constants, regulatory benchmarks, tariffs, and physics parameters identified above must be migrated to `config/assumptions.yaml` with a single validated loading point (`campus_energy.config.load_assumptions`).
2. **Interval Parametrization:** Replace hardcoded `12.0`, `36`, and `288` literals with dynamic formulas based on `config.simulation.interval_minutes`.
3. **Consistency Fixes:**
   - Standardize `web/main.py` tariffs (`8.0` vs `8.5`).
   - Align fallback power factors (`0.85` vs `0.92`).
   - Standardize default ambient temperature fallbacks across modules (`32.0°C` vs `34.0°C`).
