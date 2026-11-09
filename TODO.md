# Nivara — Feature TODO Checklist

Pick one task at a time and work through it. Each item has an effort estimate and clear acceptance criteria.

---

## 🔴 Critical / Foundation

- [x] **TASK-01** · Facility-type switcher + per-facility data  
  Sidebar dropdown: University / District Hospital / Government Office / Industrial Campus.  
  Each facility has its own `configs/` YAML and generates data into a separate `data/<facility>/` folder.  
  Dashboard dynamically loads the right dataset when the user switches.  
  *Effort: 4 hrs · Status: **DONE***

- [x] **TASK-02** · BEE Energy Performance Index (EPI) scoring  
  Add a BEE star-rating card to Executive Overview using `kWh / floor_area_m²`.  
  Show: EPI score, star rating (1–5), benchmark comparison, compliance status.  
  Add `floor_area_m2` to campus config and `esg.py` calculation.  
  *Effort: 1.5 hrs · Status: **DONE***

---

## 🟡 High Impact — AI & Analytics

- [x] **TASK-03** · Prophet forecaster alongside Ridge  
  Add `analytics/forecasting_prophet.py`. Fit Prophet on campus-level total meter load.  
  Show comparison chart on the AI Forecasting page: Ridge vs. Prophet side-by-side.  
  Include MAPE metric for each model.  
  *Effort: 3 hrs · Status: **DONE***

- [x] **TASK-04** · NLP complaint clustering (TF-IDF + KMeans)  
  Add `analytics/complaints_nlp.py`. Cluster complaints into 5–8 recurring themes.  
  Add a word-cloud or bar chart of top complaint categories to Alerts & NLP page.  
  Show "most recurring issue" callout with recommended action.  
  *Effort: 3 hrs · Status: **DONE****

- [x] **TASK-05** · Structured action recommendation log  
  When the AI assistant recommends an action, parse it into a structured entry:  
  `{action_type, target_asset, suggested_time, estimated_saving_kwh, status: pending}`.  
  Show a "Pending Actions" table in the AI Assistant page.  
  *Effort: 2 hrs · Status: **DONE****

---

## 🟡 High Impact — Simulation & Scenarios

- [x] **TASK-06** · Additional scenario levers in AI Forecasting  
  Add 3 more scenario sliders:  
  1. Occupancy reduction % (holiday / exam period)  
  2. Solar PV addition (kWp) → show grid import reduction  
  3. Lighting upgrade (LED) → model 40% lighting load reduction  
  Apply them on top of existing temp_offset + AC policy sliders.  
  *Effort: 2 hrs · Status: **DONE****

- [x] **TASK-07** · Water resource module (stub → full)  
  Add `simulator/water.py` — daily synthetic water meter simulation per building.  
  Anomaly: consumption after midnight (leak proxy).  
  Add a "💧 Water Overview" dashboard tab with consumption trend, leak alerts.  
  *Effort: 4 hrs · Status: **DONE****

---

## 🟢 Win-Differentiating

- [x] **TASK-08** · Governance dashboard (multi-facility comparison)  
  New Streamlit page: "🏛️ Governance & Policy".  
  Show facility comparison table: EPI, CO₂/occupant, BEE rating, compliance score.  
  State-level impact projection: "If 100 similar facilities adopt Nivara...".  
  NAPCC CO₂ reduction target tracker.  
  ECBC compliance checklist per facility.  
  *Effort: 4 hrs · Status: **DONE****

- [x] **TASK-09** · Fault priority queue (maintenance work-order view)  
  Transform anomaly alerts into a sortable work-order queue.  
  Priority score = severity × energy_waste × hours_unresolved.  
  Show estimated savings if fixed (kWh/day), SLA countdown, assignee field.  
  *Effort: 2 hrs · Status: **DONE****

- [x] **TASK-10** · PDF report export  
  One-page facility summary PDF using `reportlab` or `fpdf2`.  
  Include: ESG KPIs, top anomalies, top recommendations, BEE score, charts.  
  Triggered from Executive Overview with a "Download Report" button.  
  *Effort: 2.5 hrs · Status: **DONE****

- [x] **TASK-11** · Anomaly explainability layer  
  For every alert, show a structured "Why?" card:  
  "Expected 4.2 kW at 2 PM (weekday, 34°C, 45 occupants). Actual: 7.1 kW. Deviation: +2.9 kW (69% above baseline)."  
  IsolationForest: show which features were anomalous.  
  *Effort: 2 hrs · Status: **DONE****

- [x] **TASK-12** · Thermal comfort heatmap (ASHRAE PMV proxy)  
  Use existing `outdoor_temperature_c` + `humidity_pct` to compute a simple comfort index per zone.  
  Overlay on the 3D campus map as a second heatmap layer toggle.  
  *Effort: 1.5 hrs · Status: **DONE****

- [x] **TASK-13** · Carbon budget tracker  
  Monthly CO₂ budget (linked to NAPCC/net-zero goal) with burn-down chart.  
  Alert when facility is on-track to overshoot target.  
  Add to Executive Overview as a progress bar + trend.  
  *Effort: 2 hrs · Status: **DONE****

- [x] **TASK-14** · Live telemetry ticker for demo  
  Wire the replay player to the dashboard for a live "sensor arriving" effect.  
  Show a small "Live Feed" widget in the sidebar with the last 5 incoming readings.  
  *Effort: 2 hrs · Status: **DONE****

---

## 📋 Notes
- Data for each facility lives in `data/university/`, `data/hospital/`, `data/govt_office/`, `data/industrial/`
- Configs live in `configs/university.yaml`, `configs/hospital.yaml`, `configs/govt_office.yaml`, `configs/industrial.yaml`
- Existing `data/generated/` and `configs/simulation.yaml` = University (default, already generated)
- Run `generate-data --config configs/<facility>.yaml --output-dir data/<facility>` to generate a facility's data
