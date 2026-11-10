<USER_REQUEST>
# Nivara: Product Redesign Plan (Presentation Dashboard -> Real Operational Tool)

## 0. Purpose and rules for the coding agent

The analytics, decision layer, guard, ledger and evaluation work are done and stay. This plan changes **how the product is organized, how it looks, and how it behaves**, so that it feels like software a facility team opens every morning, not a slide deck that happens to be interactive.

### The test for every screen

Ask: "If a facility manager used this on a Tuesday with no presenter, would this element help them do their job?" If not, move it, shrink it, or delete it.

### Ground rules

- Do not delete analytics code. Relocate and reorganize UI only. All 209 tests must stay green.
- No new analytics in this plan. Reuse existing modules and `Recommendation` objects.
- Work step by step (section 9). One commit per step, `ux-NN: <summary>`, with a short report in `docs/step_reports/ux-NN.md` and before/after screenshots.
- Keep Streamlit (see section 8) but treat it as a front-end framework to be styled and structured properly, not as a notebook.

---

## 1. Diagnosis: why it currently reads as a presentation

Be explicit about these so they are not reintroduced:

- **Organized by features, not by jobs.** Pages are "AI Forecasting & What-If", "Microgrid Dispatch", "Water Intelligence", etc. Real users think "what needs my attention?", "what's my bill going to be?", "is that pump okay?".
- **Demo scaffolding inside the product.** A "Guided Jury Tour" toggle, "Jury Demo Moment" labels, a latency boast ("1.84 ms"), fault injection presented as a main-flow button, and a state-level impact scaler. A real user never needs these.
- **Proof and meta content in the main flow.** The evaluation scorecard, benchmark plots, BDG2 validation, and "evaluated on synthetic data" banners sit on a main page. That is evidence for judges, not an operational need.
- **Persona toggle.** Real users have a role from their login. A visible role switcher announces "this is a demo".
- **Infrastructure noise in the sidebar.** Live packet ticker, replay scrubber, system counters, and gateway status belong in an admin/data-sources area.
- **Everything shouts.** Emoji section headers, badges everywhere, three-point ranges and disclaimers on every number, 6-KPI ribbons, explanatory paragraphs on pages. Real tools are quiet, dense and scannable; they surface exceptions.
- **No notion of state over time.** Real tools have history: resolved items, activity logs, "last updated 3 min ago", open items of different ages. A freshly staged scenario looks staged.

---

## 2. Product principles

1. **Exception-first.** The home screen answers "what needs attention now?" Normal operation is quiet.
2. **Job-oriented navigation.** A small number of destinations named after what people do.
3. **Role from login, not from a toggle.** Each role lands on its own home. A "Switch user" item sits in the user menu.
4. **Progressive disclosure.** Show the answer; put the reasoning one click away (drawer, popover, detail page). Ranges, assumptions and disclaimers live behind an info affordance, not beside every number.
5. **Every number is actionable or explainable.** Clicking a KPI or chart opens the underlying list or asset.
6. **The system has memory.** Activity feed, audit trail, resolved items, history.
7. **Presenter tooling lives outside the product** (section 7).
8. **Quiet visual language.** One accent color, neutral surfaces, status colors reserved for status, no decorative emoji.

---

## 3. New information architecture

Replace 10 feature pages with **7 destinations** plus a top bar.

- **Home** (role-specific landing)
- **Alerts & Work** (alert inbox -> work orders, one workflow)
- **Assets** (list/explorer, with the map as a view mode)
- **Energy** (consumption, forecast, scenarios, tariff and billing)
- **Water**
- **Reports** (executive report, compliance, benchmarking, chargebacks)
- **Settings** (facility config, data sources, notification rules, assumptions, model details, simulator, users)

### Mapping from old pages

- Old 1 Executive Overview -> **Home** (administrator and sustainability variants) plus **Reports**
- Old 2 Governance & Policy -> **Reports** (benchmarking, ECBC checklist) and **Settings** (assumptions registry, model details/evaluation). The state-level impact scaler is removed from the product (kept only in the presenter kit, section 7).
- Old 3 3D Campus Map -> **Assets**, "Map" view mode
- Old 4 Forecasting & What-If -> **Energy**, "Forecast" and "Scenarios" tabs
- Old 5 Alerts, Complaints, Work Orders -> **Alerts & Work**. Complaints become a "Occupant complaints" tab.
- Old 6 Water -> **Water**
- Old 7 Microgrid -> **Energy**, "Solar & storage" tab
- Old 8 Billing & Chargebacks -> **Energy**, "Bills & chargeback" tab (and exports in **Reports**)
- Old 9 Asset Detail Explorer -> **Assets**, asset detail page
- Old 10 AI Assistant -> a **persistent side panel** ("Ask Nivara") available from any screen via the top bar, not a page

---

## 4. Global shell

### Top bar (persistent)

- Left: product name/logo, facility name (a switcher only if the user has access to multiple facilities)
- Center: global time range selector (Today, 7 days, 28 days, custom) that all screens respect
- Right: data freshness indicator ("Data current as of 14:05 IST", turns amber if stale), alert bell with open count, "Ask Nivara" button, user menu (name, role, Switch user, Sign out)

### Left navigation

- Only the seven destinations, with a small badge count on Alerts & Work.
- No counters, tickers, replay controls, or gateway status in the sidebar.

### Conventions

- Deep-linkable URLs via query params (`?page=assets&asset=EQ-UTL-P1-PM`).
- Detail views open in a right-hand drawer or a dedicated detail view with a back link, not inline expanders stacked on a long page.
- All timestamps in IST with relative time ("12 min ago") plus exact time on hover.
- Money in Indian grouping (₹1,27,500 / lakh, crore) with consistent units everywhere.
- Every table: sortable, filterable, with a CSV export where useful.

---

## 5. Screen specs

### 5.1 Login stub (role from user, not toggle)

- A simple sign-in screen backed by `config/users.yaml` (name, role, facility access). No real auth needed, but the flow should be real: sign in -> role home.
- Seed three demo users (administrator, operations lead, sustainability officer). "Switch user" in the user menu returns to sign-in.

### 5.2 Home: Operations lead

- **Needs attention** (the main element): a prioritized list combining open critical/high alerts, work orders nearing or past SLA, assets offline or with poor data quality. Each row: severity, asset, what happened in one line, impact (₹ per day or kW above expected), age, and the primary action button.
- **Today** strip (at most four numbers): Load now vs expected, open alerts, work orders due today, data health.
- **Load today vs forecast**: one chart with forecast band; anomaly markers clickable.
- **Recent activity**: last 10 events (alert raised, work order dispatched, resolved).
- Nothing else.

### 5.3 Home: Administrator

- **This month**: bill to date, projected month-end bill, versus last month and budget (if configured). Demand-charge risk as a single status line ("Peak demand 412 kVA vs contract 450 kVA").
- **Decisions waiting**: top recommendations needing approval, each with expected annual saving, payback, service-safe status, and Approve / Decline / Details.
- **Savings realized**: verified savings to date and the trend.
- **Spend by department** (compact), linking to Energy > Bills.

### 5.4 Home: Sustainability officer

- **Carbon this month vs target** (NAPCC trajectory) as one chart plus a status sentence.
- **BEE star and EPI** with gap to the next star, one compact card.
- **Verified impact** (kWh, ₹, kgCO2e) with the actions behind it.
- **Reports due / export** shortcut.

### 5.5 Alerts & Work

One workflow, two tabs plus complaints.

- **Alerts tab**: table (severity, asset, summary, detected, status, impact). Filters: status, severity, building, asset type. Clicking a row opens the **alert detail drawer**.
- **Alert detail drawer** (the most important screen in the product), top to bottom:
  - Headline: what is wrong, in one sentence, with expected vs actual.
  - Impact: cost and energy per day if left unresolved.
  - Recommended action with a **service-safe chip** (green "Safe to act", amber "Needs review", red "Not recommended" with the reason on click). Blocked actions appear inline here with the safe alternative offered, not in a separate showcase expander.
  - "Why did we flag this?" collapsed section (factor attribution, probable causes, field checks).
  - Actions: Create work order, Assign, Snooze, Mark as false alarm (this feedback is stored).
  - Timeline of everything that happened to this alert.
- **Work orders tab**: board or table with columns Proposed, Open, In progress, Done, Verified; SLA timers; assignee; filters. Resolve & Verify opens the verification result (estimated vs verified saving).
- **Occupant complaints tab**: cluster themes, recurring issues, linked assets/alerts.
- Notification activity is shown in the alert timeline ("Telegram sent to Ops Lead 14:07"), and rules are configured in Settings.

### 5.6 Assets

- **List view** (default): searchable table with type, building, zone, status, current kW, vs expected, protected-load marker, last seen.
- **Map view**: the existing digital twin, as a view toggle. Layers: Power, Comfort. Click a building or asset opens its detail page.
- **Asset detail page**: header with status and key facts; power trace with fault periods and anomalies; operating statistics; hour x weekday heatmap; peer comparison; alerts and work orders history for the asset; protected-load note where applicable.

### 5.7 Energy

Tabs:
- **Overview**: consumption by building/department over the selected range, load profile, peak demand versus contract, power factor, ToD cost split.
- **Forecast**: next 24 hours with band, expected peak and time, simple plain-language summary ("Peak expected at 15:30 around 380 kW"). Model details behind an info popover.
- **Scenarios**: the what-if levers as a planning tool, with results shown as an expected value and a range on hover or in a details line.
- **Solar & storage**: the microgrid view, framed as planning and monitoring.
- **Bills & chargeback**: bill breakdown (energy, demand, PF, ToD), department ledger, sub-meter reconciliation, CSV export.

### 5.8 Water

Consumption trends, minimum-night-flow leak watch, active leaks with estimated loss and work-order link. Same patterns as Alerts (rows open the same drawer).

### 5.9 Reports

- Executive report (PDF) with period selector and preview.
- Compliance (ECBC checklist) and BEE/EPI.
- Cross-facility benchmarking (only visible to roles with multi-facility access).
- Scheduled report settings (frequency, recipients) as a simple form.

### 5.10 Settings

- **Facility**: profile, protected loads, comfort bounds, tariff selection.
- **Data sources**: ingestion status, upload data (BYOD), data quality scorecard, source health. The live packet feed and replay controls live here.
- **Notifications**: rules by role and severity, channel status, test message, delivery log.
- **Assumptions**: the full registry with sources and confidence.
- **Model details**: forecast and anomaly evaluation, real-data benchmark, limitations (this is where the "proof layer" lives for anyone who wants it).
- **Simulator** (visible only when the data source is simulated): fault injection studio and reset. A small persistent label in the top bar reads "Simulated data source".
- **Users & roles**.

### 5.11 "Ask Nivara" panel

- Slide-over panel from the top bar, context-aware (knows which asset or alert is open).
- Suggested questions depend on role and on the current screen.
- Structured actions go through the same recommendation and guard flow, then show up in the Decisions or Alerts lists.

---

## 6. Visual design system

- **Layout:** fixed left nav, top bar, content max-width with a consistent 12-column grid, generous but not wasteful spacing, dense tables.
- **Color:** neutral grays and white (optional dark theme later), a single accent color, status colors only for status (critical, warning, ok, info). No rainbow charts; use one or two series colors plus gray.
- **Typography:** one sans-serif family, clear scale (page title, section title, body, caption), tabular numerals for numbers.
- **Components to standardize:** KPI card (value, label, small delta, optional sparkline), status chip, severity chip, service-safe chip, data table, drawer, empty state, loading skeleton, error state, info popover ("How is this calculated?" linking to the relevant assumptions), toast.
- **Icons:** a small consistent icon set (no emoji in headings or labels).
- **Charts:** direct labels, light gridlines, consistent time axes, hover tooltips with units, anomaly and fault markers in one style, forecast band in a light tint.
- **Copy:** short, plain, operational ("Pump P1 is drawing power with no flow"), no marketing phrases, no "AI-powered" labels, no exclamation marks.
- **Numbers:** show one value by default. Ranges, confidence, and the "decision-support estimate" note appear in the info popover and in exported reports, not beside every figure.
- **Statuses:** every list item has a status; every empty state says what it means and what to do.

---

## 7. Presenter kit (separate from the product)

The hackathon demo still needs a guided path and live fault injection, but these must not live inside the product UI.

- Create a **presenter controller** as a separate small Streamlit app (or a separate route opened in a second window) that: resets to a clean state, injects a chosen fault into the simulator through the same API the Settings > Simulator tool uses, advances the replay, resolves the fault, and shows the presenter script and timing.
- The product itself is driven normally on the main screen: the presenter clicks through Home -> Alerts & Work -> alert drawer -> work order -> verification, exactly as a user would.
- Move the state-level impact scaler, benchmark visuals, and pitch-only summaries into the presenter kit or slides.
- Remove the Guided Jury Tour toggle, "Jury Demo Moment" labels and latency claims from the product.
- Provide a `make demo` that starts both the product and the controller.

---

## 8. Implementation approach (Streamlit)

- Use `st.navigation` / `st.Page` for the seven destinations; hide default page chrome; configure `.streamlit/config.toml` theme (colors, font) and add one shared CSS file loaded once.
- Build a small `ui/` component library (`kpi_card`, `status_chip`, `data_table`, `drawer`/`st.dialog`, `empty_state`, `info_popover`) and use only those in pages. No page should hand-roll styling.
- Use `st.dataframe` with `column_config` for sortable tables; `st.dialog` or a side column for drawers; `st.fragment` for live-refreshing panels so the whole page doesn't rerun.
- Keep state in `st.session_state` plus query params for deep links.
- Cache heavy computations so navigation stays fast.
- Decision point for the team: if there is time and front-end skill, a React front end over the existing FastAPI would feel more "real", but it is a larger risk. The default here is Streamlit styled to a real-product standard.

---

## 9. Step-by-step implementation order

### Step UX-1. Audit and inventory
- List every current widget by page and tag it: **Keep in product**, **Move to Settings/Reports**, **Move to presenter kit**, **Delete**.
- Output: `docs/ux/inventory.md`.
- Acceptance: every widget has a tag and a destination.

### Step UX-2. Shell, navigation and login
- Implement top bar, seven-item nav, global time range, data freshness indicator, user menu, login stub with `config/users.yaml`, deep-link query params.
- Remove persona toggle, jury tour toggle and sidebar infrastructure widgets from the main shell.
- Acceptance: sign in as each role lands on a different Home; Switch user works; URLs deep-link.

### Step UX-3. Design system and component library
- Theme config, shared CSS, and the `ui/` components listed in section 6.
- Acceptance: a `/styleguide` dev page shows every component in all states; no page uses raw emoji headings.

### Step UX-4. Alerts & Work (highest priority screen)
- Alert table, alert detail drawer, work orders board/table, complaints tab, inline service-safe chip with blocked-action reason and safe alternative, timeline, false-alarm feedback.
- Acceptance: a full path from alert to work order to verified result can be done without leaving this destination; blocked recommendations show reason inline.

### Step UX-5. Role homes
- Build the three Home screens per sections 5.2 to 5.4, using existing KPIs and `Recommendation` data.
- Acceptance: each Home fits on one screen without scrolling at 1080p, and every element is clickable through to detail.

### Step UX-6. Assets
- List view, map view toggle, asset detail page with history.
- Acceptance: Home alerts link into the asset page; Map fallback still works.

### Step UX-7. Energy and Water
- Consolidate Forecast, Scenarios, Solar & storage, Bills & chargeback, and Water into the tabbed layouts, applying the "one value by default" number rule.
- Acceptance: no page shows a range or disclaimer inline; each has an info popover linking to assumptions.

### Step UX-8. Reports and Settings
- Move governance, ECBC, benchmarking, assumptions, model details, data sources, BYOD, notification rules, simulator, and users into their new homes.
- Acceptance: nothing from the old Governance page is visible in the main operational flow.

### Step UX-9. Ask Nivara panel
- Convert the assistant to the context-aware slide-over panel; keep the offline fallback and structured actions.
- Acceptance: opening it from an alert drawer pre-loads that alert's context.

### Step UX-10. Presenter kit and realistic seed state
- Build the presenter controller and `make demo`.
- Seed the demo database with a plausible operational history: closed and open alerts of varied ages, resolved work orders with verified savings, an activity log, a couple of snoozed items, one stale-data warning that has since cleared. The product should open looking lived-in.
- Acceptance: a fresh clone opens to a populated, believable Home; the presenter flow runs three times in a row from reset.

### Step UX-11. Polish and consistency pass
- Copy pass (plain language, no marketing phrases), empty/loading/error states on every screen, keyboard focus, responsive check at laptop and projector resolutions, performance check (navigation under 300 ms with caching).
- Acceptance: side-by-side before/after screenshots committed; checklist in `docs/ux/polish_checklist.md` all ticked.

---

## 10. Remove or relocate checklist

- Remove from product: Guided Jury Tour toggle, "Jury Demo Moment" labels, latency claims, persona toggle, state-level impact scaler, emoji section headers, inline "decision-support" disclaimers beside every number, sidebar system counters/packet ticker/replay scrubber/gateway status.
- Relocate to Settings: assumptions registry, model evaluation and real-data benchmark, data sources and quality, notification rules, simulator controls.
- Relocate to Reports: ECBC checklist, benchmarking, executive PDF.
- Relocate to presenter kit: fault-injection controller, scripted flow, pitch visuals.

## 11. One product-quality issue to fix in the same pass

The evaluation shows seasonal-naive forecasting at about 8% MAPE versus Ridge and Prophet at about 40%. A real product ships the model that works. Use seasonal-naive (same slot last week) as the production baseline, and have the ML models predict the **residual** on top of it using occupancy, weather and holiday features. Show a single "Forecast" in the product (with its band); keep the three-model comparison in Settings > Model details. Re-run the evaluation harness and report honestly if the hybrid does not beat the baseline, in which case ship the baseline.
</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-11-10T10:07:35+05:30.
</ADDITIONAL_METADATA>