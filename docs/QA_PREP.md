# 🎯 Nivara — Hackathon Jury Q&A Preparation & Defensibility Guide

> **Target Audience:** Hackathon Judging Panel, Technical Reviewers, Facility Directors & Energy Auditors  
> **Platform:** Nivara Campus Energy Intelligence Platform  
> **Engineering Philosophy:** Defensible Rigor, Transparent Boundaries, and Closed-Loop Verification

---

## Overview

Judges at competitive hackathons look beyond polished UI styling: they probe model design choices, data provenance, regulatory realism, safety failure modes, and deployment viability. 

This document arms the presentation team with authoritative, technically sound, and honest responses to the toughest anticipated questions from the hackathon jury.

---

### Question 1: "Isn't your system trained and evaluated only on synthetic data?"

#### 🎙️ The Winning 30-Second Answer:
> *"We employ a rigorous **dual-validation methodology**. We use a high-fidelity physics simulator because in real-world facilities, verified ground-truth timestamps for equipment faults are virtually non-existent without intentional sabotage. However, to prove real-world transferability, we also evaluated Nivara on the open-access **Building Data Genome 2 (BDG2)** dataset—achieving a **6.57% Weekday MAPE** on real college classroom meters without prior tuning."*

#### 🔍 Deep Technical Defense:
1. **The Ground Truth Paradox in Energy IoT:** In commercial buildings, anomalies (e.g., pump dry-runs, hidden distribution leaks, unseasonal chiller over-runs) are unlabeled, unmonitored, or hastily repaired without timestamp logging. Evaluating anomaly detection precision, recall, and detection delay requires known ground-truth events. Our synthetic generator injects 5 realistic physical faults across 8,063 timesteps while maintaining three-phase electrical consistency ($P = \sqrt{3} \cdot V \cdot I \cdot \cos \phi$).
2. **Step 15 External Transfer Proof:** We ran the exact same feature engineering, baseline models, and anomaly detectors on the public **Building Data Genome 2 (BDG2)** dataset (Miller et al., *Scientific Data* 2020, CC-BY 4.0 license, `UnivClass_Mary`).
3. **The Quantitative Proof:**
   - On synthetic data: 80.0% recall, 0 false alarms on physical rule ablations.
   - On real BDG2 classroom telemetry: **6.57% Weekday MAPE** (Ridge), **96.88% interval coverage**, and successfully flagged 3 unseasonal weekend HVAC overconsumption episodes.
   - Artifact: Verified in `reports/eval/realdata_summary.json` and reproducible via `make realdata`.

---

### Question 2: "Why did your baseline Seasonal-Naive forecaster achieve a better MAPE (8.27%) than Ridge (40.09%) and Prophet (40.47%)?"

#### 🎙️ The Winning 30-Second Answer:
> *"That is an honest, mathematically sound characteristic of academic facilities. Universities adhere to rigid diurnal calendar schedules where Monday 10:00 AM matches last Monday 10:00 AM almost perfectly. Furthermore, MAPE severely penalizes parametric regressions during low-load weekend intervals where tiny absolute errors blow up percentage denominators. However, Seasonal-Naive cannot simulate what-if scenarios or output prediction intervals—Ridge and Prophet can."*

#### 🔍 Deep Technical Defense:
1. **The Nature of Institutional Load Profiles:** Academic facilities follow strict class schedules. Seasonal-Naive ($y_t = y_{t - 7\text{ days}}$) acts as a persistent diurnal mirror. During unoccupied weekends, power drops to a flat baseload (~5–8 kW). Seasonal-Naive predicts this flat baseline with a tiny 0.98 kW MAE (**4.56% MAPE**).
2. **The MAPE Denominator Bias:** Mean Absolute Percentage Error is defined as $\frac{1}{n} \sum \frac{|y - \hat{y}|}{y}$. When a classroom's actual load drops to 2.5 kW on Sunday morning, a parametric model predicting 4.2 kW has an absolute error of only 1.7 kW, yet its MAPE is $\frac{1.7}{2.5} = 68\%$. This low-load distortion inflates the aggregate overall MAPE to 40.09%, even though Ridge's absolute MAE is a modest 9.35 kW across the entire campus.
3. **Why Parametric Models are Indispensable:**
   - **Counterfactual Scenario Simulation:** Seasonal-Naive cannot answer: *"What happens if ambient wet-bulb temperature rises by 3°C, or if we shift lecture timetables by 2 hours?"* Ridge and Prophet model exogenous temperature, humidity, and cyclic factors directly.
   - **Calibrated Prediction Intervals:** Seasonal-Naive produces a single point prediction. Ridge and Prophet produce **80% prediction intervals** with **81.4% and 82.7% empirical coverage**, giving facility managers risk-bounded operating envelopes.

---

### Question 3: "Why did you choose Ridge Regression and Meta Prophet instead of a deep neural network like LSTM, GRU, or Transformer (PatchTST)?"

#### 🎙️ The Winning 30-Second Answer:
> *"In real institutional facility operations, **interpretability, sample efficiency, and edge execution beat deep complexity**. Ridge and Prophet require zero GPU hardware, train in under 2 seconds on a CPU, provide instant closed-form prediction intervals, and run 100% offline on a low-power campus gateway. Deep learning would overfit a 28-day baseline and introduce opaque black-box predictions that facility directors reject."*

#### 🔍 Deep Technical Defense:
1. **Sample Efficiency:** Institutional telemetry frequently suffers from missing historical spans or recent sensor retrofits. A deep Transformer requires years of training history to generalize without catastrophic overfitting. Ridge with cyclic Fourier time encodings and Meta Prophet fit reliably on 14–28 days of data.
2. **Zero GPU Dependency:** Nivara is designed to run locally on an institutional campus server or standard BMS PC (even offline without internet or high-end CUDA accelerators). Ridge inference takes **< 5 milliseconds**; Prophet executes in **< 1.5 seconds**.
3. **Statutory Auditability & ASHRAE Guideline 14:** Energy performance contracting (IPMVP Option B / ASHRAE Guideline 14) requires that baseline regression coefficients be transparent, defensible, and inspectable by third-party auditors. A neural network's latent weights cannot be submitted to an electricity regulatory commission.
4. **Calibrated Intervals Out-of-the-Box:** Deep networks require complex Monte Carlo dropout or conformal prediction layers to estimate uncertainty. Ridge and Prophet provide well-calibrated prediction intervals that achieved **81.4% and 82.7% empirical coverage** directly.

---

### Question 4: "Can this system actually be deployed on a real campus tomorrow, or does it only work with your custom CSV headers?"

#### 🎙️ The Winning 30-Second Answer:
> *"It can be deployed tomorrow. In Step 14, we engineered a dedicated **Bring-Your-Own-Data (BYOD) Ingestion Engine** with regex-based smart heuristic mapping, automatic unit conversions, and a 5-rule Data Quality Engine. Whether your meters output Watts, kW, Joules, UTC, or Indian Standard Time, Nivara maps, normalizes, and scores the dataset automatically."*

#### 🔍 Deep Technical Defense:
1. **Smart Heuristic Mapping (`src/campus_energy/adapters/csv_upload.py`):**
   - Employs regex alias patterns for 10 physical dimensions (e.g., matching `reading_time`, `recorded_at`, `activepower_w`, `kw`, `frequency_hz`).
   - Handles custom column override mappings if a proprietary BMS uses non-standard abbreviations.
2. **Automatic Physical Conversions:**
   - Detects Watts and scales to kW ($\div 1000$).
   - Converts Joules to kWh ($\div 3.6 \times 10^6$) and Wh to kWh ($\div 1000$).
   - Normalizes naive or UTC timestamps to Indian Standard Time (`Asia/Kolkata`, `+05:30`).
3. **Data Quality Engine (DQE) Scorecard (`src/campus_energy/adapters/quality.py`):**
   - Evaluates incoming streams against 5 rules: missing intervals, duplicates, stuck/frozen sensors, physical outliers ($V > 300\text{V}$, $\text{PF} > 1.0$), and asset coverage.
   - Calculates a composite `quality_score` and dynamic confidence penalty (-0.40 to 0.0) that widens uncertainty bands rather than triggering false panic.
   - Fully tested and verified in [`tests/test_adapters.py`](file:///C:/Users/Paramjeet/Desktop/BPUT/tests/test_adapters.py).

---

### Question 5: "Why did you focus on deep closed-loop actionability rather than building 30 different dashboard pages?"

#### 🎙️ The Winning 30-Second Answer:
> *"Because **facility managers are suffering from dashboard fatigue**. The world does not need another passive monitoring screen that displays red line charts while waste continues unabated. Our north star is: `Problem -> Data -> AI -> Insight -> Action -> Verified Result`. In Nivara, every insight automatically generates a safe recommendation, routes a prioritized CMMS work order with SLA countdowns, and audits the savings under IPMVP Option B."*

#### 🔍 Deep Technical Defense:
1. **The Passive Dashboard Trap:** Most commercial energy software stops at visualization. An operator looks at a spike, doesn't know the financial cost or root cause, leaves it unaddressed, and the university loses lakhs of rupees.
2. **The 6-Stage Closed Loop:**
   - **Data:** 5-minute telemetry + microclimate + occupant complaints.
   - **AI:** Ridge & Prophet baselines + multi-layer anomaly detection + ASHRAE 55 PMV model.
   - **Insight:** Diagnostic "Why?" cards with factor attribution and standardized `{low, expected, high}` savings ranges.
   - **Guard:** Service-Impact Guard checking clinical lockouts and occupant thermal comfort bounds.
   - **Action:** CMMS work-order dispatch queue with live SLA timers and multi-channel Telegram/Email alerts.
   - **Verified Result:** IPMVP Option B continuous M&V ledger tracking empirical hit rate % and generating official ReportLab audit PDFs.
3. **Engineering Depth Beats Feature Sprawl:** 10 highly-focused, persona-tailored operational pages with sub-second cached transitions and 100% offline resilience deliver far greater operational impact than 30 disconnected mockups.

---

### Question 6: "How do you guarantee your AI won't turn off air conditioning or power to critical medical equipment in a hospital?"

#### 🎙️ The Winning 30-Second Answer:
> *"We built the **Service-Impact Guard** specifically to prevent algorithmic over-reach. Any load categorized as life-safety (ICU ventilators, surgical OTs), cold chain (vaccine storage), or critical IT (server rooms) is **hardcoded as non-curtailable in the Facility Profile**. Furthermore, for standard spaces, our ASHRAE 55 PMV engine blocks any thermostat setback that pushes thermal sensation past +0.5."*

#### 🔍 Deep Technical Defense:
1. **The Clinical Safety Principle:** Implemented in [src/campus_energy/decision/guard.py](file:///C:/Users/Paramjeet/Desktop/BPUT/src/campus_energy/decision/guard.py). Energy efficiency recommendations must never compromise life support, patient safety, vaccine potency, research integrity, or basic human comfort.
2. **Facility Profile Protection Schema (`src/campus_energy/profiles.py`):**
   - Each facility defines `ProtectedLoad` specifications with category tags: `life_safety`, `cold_chain`, `critical_it`, and `continuous_process`.
   - In healthcare facilities (e.g. `district_hospital`), ICU circuits, operating theatre chillers, and blood bank freezers have `is_curtailable: False`.
3. **Two-Stage Defense-in-Depth:**
   - **Stage 1 (Asset Lockout):** If an AI recommendation attempts to shed, curtail, shut down, or reschedule an asset matching a protected load, the Service Guard **immediately blocks it with status: `blocked`**.
   - **Stage 2 (ASHRAE 55 PMV Comfort Check):** For normal classrooms and offices, if an AI action proposes raising cooling setpoints (e.g. to shave ToD peak charges), the guard calculates the counterfactual Predicted Mean Vote (PMV). If $\text{PMV} > +0.5$ (discomfort boundary), the action is blocked.
4. **Proactive Safe Alternatives:** Rather than just blocking, Nivara automatically synthesizes a safe alternative (e.g., *"Safe alternative: Inspect chiller COP / clean condenser coils or dispatch rooftop solar BESS battery to shave peak demand without altering clinical indoor setpoints"*).
5. **Live Demo Verification:** Demonstrated in Tour Step 4 and verified passing in [`tests/test_service_guard.py`](file:///C:/Users/Paramjeet/Desktop/BPUT/tests/test_service_guard.py).

---

### Question 7: "What happens if the conference Wi-Fi cuts out or Google Gemini hits an API quota limit during the live demo?"

#### 🎙️ The Winning 30-Second Answer:
> *"Nothing breaks. Nivara is engineered with an uncompromising reliability standard: **Zero external cloud dependencies required**. If the Gemini API is unreachable or Wi-Fi is disconnected, the platform seamlessly engages its **Grounded Local Telemetry Fallback**, our 3D Mapbox switches to an interactive **2D Plotly Vector Schematic**, and all pages run inside isolated **Error Boundaries**."*

#### 🔍 Deep Technical Defense:
1. **Grounded Offline AI Assistant (`src/campus_energy/dashboard/agent.py`):**
   - If `api_key` is None, empty, or a network `ConnectionError` occurs, `chat_with_gemini` routes to `generate_offline_fallback`.
   - Synthesizes structured, defensible facility intelligence answers directly from active `Recommendation` objects, active alerts, and assumptions registry constants (`₹8.50/kWh`, `0.82 kgCO2e/kWh`).
   - Emits structured action tags parsed seamlessly into typed `Recommendation`s passed to the Service Guard—never throwing an unhandled exception or raw stack trace.
2. **2D Plotly Schematic Fallback (`src/campus_energy/dashboard/app.py`):**
   - If Mapbox tokens are absent or WebGL fails, the dashboard automatically renders an interactive 2D Plotly vector schematic plotting campus boundaries, building footprints, and color-coded telemetry pins with full hovercards.
3. **Page-Level Error Boundaries (`render_page_error_boundary`):**
   - Every page execution is wrapped in an isolated boundary. If an unexpected error occurs, it displays a styled diagnostic card with one-click recovery buttons (`🔄 Retry Page Execution` and `🏠 Reset to Executive Overview`) without blanking the dashboard or sidebar.
4. **Sub-Second Caching Tier (< 150ms):**
   - All compute functions use `@st.cache_data(ttl=3600)`. Cached operations resolve in under 150 milliseconds, eliminating multi-second lag during page transitions.
   - Verified passing across 8 reliability tests in [`tests/test_reliability.py`](file:///C:/Users/Paramjeet/Desktop/BPUT/tests/test_reliability.py).

---

### Question 8: "How do you avoid hallucinating speculative savings like 'This algorithm will save 80% of total campus energy'?"

#### 🎙️ The Winning 30-Second Answer:
> *"Nivara strictly prohibits single-point speculative claims. Every savings prediction is calculated as a **standardized three-point monotonic range `{low, expected, high}`** derived directly from physical wattages, official OERC utility tariffs, and empirical fault-waste fractions documented in our **Central Assumptions Registry**."*

#### 🔍 Deep Technical Defense:
1. **Monotonic Tripartite Ranges (`src/campus_energy/decision/recommendation.py`):**
   - Savings are never presented as single numbers. Every recommendation outputs a `SavingsBundle` with guaranteed ordering: $\text{low} \le \text{expected} \le \text{high}$ across kWh, INR, and kg CO2e.
   - Typically parameterized as a calibrated $\pm 20\%$ empirical spread around nominal engineering calculations.
2. **Zero Hardcoded Numbers (Central Assumptions Registry):**
   - Tariffs (₹8.50 base, ₹10.20 peak, ₹250/kVA demand), carbon factors (0.82 kgCO2e/kWh grid, 0.04 solar), and capital costs (₹1,200/kVAR APFC, ₹42,000/kWp solar) are defined in [config/assumptions.yaml](file:///C:/Users/Paramjeet/Desktop/BPUT/config/assumptions.yaml).
   - Verified by AST inspection in [`tests/test_assumptions.py`](file:///C:/Users/Paramjeet/Desktop/BPUT/tests/test_assumptions.py): zero hardcoded magic numbers exist in the analytics code.
3. **Audited Closed-Loop M&V (`src/campus_energy/decision/ledger.py`):**
   - Predicted savings are cross-checked against post-retrofit actual meter readings under IPMVP Option B.
   - We publish our empirical **Hit-Rate %**, explicitly showing when an intervention achieved or fell short of predicted savings.

---

### Question 9: "How does your system model complex Indian electrical billing, such as contract demand surcharges and power factor adjustments?"

#### 🎙️ The Winning 30-Second Answer:
> *"We implemented a dedicated **Indian Tariff Engine (`tariffs.py`)** encoding Odisha Electricity Regulatory Commission (OERC) and state commission HT Commercial rules. It models monthly energy charges across 3 ToD bands, enforces the 80% minimum billing demand floor on contract kVA, and applies exact statutory power factor penalties below 0.90 and rebates above 0.95."*

#### 🔍 Deep Technical Defense:
1. **Billing Demand Floor (kVA):**
   - Under Indian HT utility agreements, facilities pay for sanctioned demand regardless of usage. Nivara computes:
     $$\text{Billing Demand} = \max(\text{Recorded Peak kVA}, 0.80 \times \text{Contract Demand kVA})$$
   - If recorded peak is 300 kVA against a 450 kVA contract, the campus is billed for 360 kVA at ₹250/kVA/month ([src/campus_energy/tariffs.py#L96-L105](file:///C:/Users/Paramjeet/Desktop/BPUT/src/campus_energy/tariffs.py#L96-L105)).
2. **Three-Tier Time-of-Day (ToD) Pricing:**
   - **Peak (18:00–22:00 IST):** +20% surcharge (₹10.20/kWh).
   - **Off-Peak (00:00–06:00 IST):** -15% incentive rebate (₹7.225/kWh).
   - **Normal (06:00–18:00, 22:00–24:00 IST):** Base rate (₹8.50/kWh).
3. **Statutory Power Factor Penalty / Rebate Rules:**
   - If average $\text{PF} < 0.90$: Penalty surcharge of **1.0% of total energy charges per 0.01 drop** below 0.90.
   - If average $\text{PF} > 0.95$: Incentive rebate credit of **0.5% of total energy charges per 0.01 rise** above 0.95.
   - If $0.90 \le \text{PF} \le 0.95$: Zero adjustment (deadband).
   - Generates an APFC capacitor bank recommendation based on the national capex benchmark of ₹1,200/kVAR ([config/assumptions.yaml#L526-L533](file:///C:/Users/Paramjeet/Desktop/BPUT/config/assumptions.yaml#L526-L533)).

---

### Question 10: "How does Nivara support different stakeholders without creating clutter or duplicate dashboards?"

#### 🎙️ The Winning 30-Second Answer:
> *"Nivara features **Role-Based Persona Adaptation** in `roles.py`. Rather than duplicating pages, a single dropdown dynamically reconfigures metric card focus, diagnostic widgets, and the Gemini assistant's system prompt to suit the Administrator, the Operations Engineer, or the Sustainability Officer."*

#### 🔍 Deep Technical Defense:
1. **Administrator Persona:** Focuses on monthly billing demand, avoidable financial loss, carbon budget burn-down, and board-ready PDF audit exports. Gemini acts as an executive advisor focused on capital ROI and utility surcharges.
2. **Operations Persona:** Focuses on 15-minute load profiles, active anomaly alerts, root-cause "Why?" diagnostic cards, CMMS work-order SLAs, and the live fault injection studio. Gemini acts as Chief Facility Engineer diagnosing specific breaker kW ratings and equipment assets.
3. **Sustainability Officer Persona:** Focuses on cross-facility BEE Star Ratings, ECBC 2017 chapter-level statutory checklists, state-wide decarbonization scaling, and NAPCC 20% trajectory tracking. Gemini acts as a statutory compliance auditor.
4. **Tested & Verified:** The persona registry, role filtering, and dynamic prompt injection are verified in [`tests/test_roles.py`](file:///C:/Users/Paramjeet/Desktop/BPUT/tests/test_roles.py).

---

## Quick-Reference Numbers Cheat Sheet for Presenters

Keep these exact verified numbers at your fingertips during the jury presentation:

| Metric / Parameter | Value | Verification Reference |
| :--- | :--- | :--- |
| **Total Automated Tests** | **209 / 209 passing (100%)** | `pytest tests/ -v` |
| **Total Monitored Assets** | **56 Assets (5 Buildings)** | `data/generated/assets.csv` |
| **Telemetry Resolution** | **5-minute (8,064 timesteps, 28 days)** | `data/generated/telemetry.csv` |
| **Total Telemetry Records** | **290,305 rows** | `data/generated/telemetry.csv` |
| **Base Commercial Tariff** | **₹8.50 / kWh** | `config/assumptions.yaml` (`tariff.electricity_commercial`) |
| **Peak ToD Surcharge (18-22h)**| **+20% (₹10.20 / kWh)** | `src/campus_energy/tariffs.py` |
| **Demand Charge** | **₹250 / kVA / month** | `src/campus_energy/tariffs.py` |
| **Power Factor Penalty** | **1.0% per 0.01 drop below 0.90** | `src/campus_energy/tariffs.py` |
| **Power Factor Rebate** | **0.5% per 0.01 rise above 0.95** | `src/campus_energy/tariffs.py` |
| **APFC Capacitor Capex** | **₹1,200 / kVAR** | `config/assumptions.yaml` (`capital_cost.apfc_per_kvar`) |
| **BEE University ECBC Base** | **100.0 kWh/m²/yr** | `config/assumptions.yaml` (`bee.epi.university.ecbc_baseline`) |
| **Grid Carbon Emission Factor**| **0.82 kgCO2e / kWh** | `config/assumptions.yaml` (`emissions.grid_factor`) |
| **Solar Lifecycle Carbon** | **0.04 kgCO2e / kWh** | `config/assumptions.yaml` (`emissions.solar_lifecycle_factor`) |
| **NAPCC Reduction Target** | **20.0% from baseline** | `config/assumptions.yaml` (`policy.napcc_reduction_target_pct`) |
| **Anomaly Recall (Layered)** | **80.0% (4 of 5 ground-truth faults)**| `reports/eval/summary.json` |
| **Physical Rules Precision** | **100.0% (0 false alarms)** | `reports/eval/summary.json` |
| **Ridge Interval Coverage** | **81.4% (80% theoretical interval)**| `reports/eval/summary.json` |
| **Prophet Interval Coverage**| **82.7% (80% theoretical interval)**| `reports/eval/summary.json` |
| **Seasonal-Naive MAPE** | **8.27% (4.56% weekend)** | `reports/eval/summary.json` |
| **Ridge Weekday MAPE (Real BDG2)**| **6.57%** | `reports/eval/realdata_summary.json` |
| **Fault Reaction Latency** | **1.84 milliseconds** | `tests/test_demo.py` |
| **Page Cache Transition** | **< 150 milliseconds** | `tests/test_reliability.py` |

---
*QA Preparation Dossier for Nivara — Ready for Hackathon Jury Evaluation.*
