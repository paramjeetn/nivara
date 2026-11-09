from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import pandas as pd
import numpy as np
from pathlib import Path
import math

from campus_energy import assumptions

app = FastAPI()
DATA_DIR = Path("data/generated")

def clean_float(val):
    if isinstance(val, float) and math.isnan(val):
        return 0.0
    return val

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    with open("src/campus_energy/web/templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/telemetry")
async def get_telemetry():
    try:
        tel = pd.read_csv(DATA_DIR / "telemetry.csv")
        latest_ts = tel["timestamp"].max()
        latest_data = tel[tel["timestamp"] == latest_ts]
        
        total_power = clean_float(latest_data["power_kw"].sum())
        avg_temp = clean_float(latest_data["temperature_c"].mean())
        
        bld_data = latest_data.groupby("asset_id").agg({
            "power_kw": "sum",
            "temperature_c": "mean"
        }).reset_index().fillna(0)
        
        return {
            "timestamp": latest_ts,
            "kpis": {
                "total_power_kw": round(total_power, 1),
                "avg_temp_c": round(avg_temp, 1),
                "active_buildings": len(bld_data)
            },
            "buildings": bld_data.to_dict(orient="records")
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/history")
async def get_history():
    try:
        tel = pd.read_csv(DATA_DIR / "telemetry.csv")
        tel['timestamp'] = pd.to_datetime(tel['timestamp'])
        hourly = tel.groupby(tel['timestamp'].dt.strftime('%Y-%m-%d %H:00'))["power_kw"].sum().reset_index().fillna(0)
        hourly = hourly.tail(24)
        return {
            "labels": hourly["timestamp"].tolist(),
            "values": [round(clean_float(v), 1) for v in hourly["power_kw"].tolist()]
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/esg")
async def get_esg():
    try:
        tel = pd.read_csv(DATA_DIR / "telemetry.csv")
        total_energy_kwh = clean_float(tel[tel["reading_type"] == "meter"]["interval_energy_kwh"].sum())
        wasted_energy = 14500.5
        grid_factor = float(assumptions.get_value("emissions.grid_factor"))
        tariff = float(assumptions.get_value("tariff.electricity_commercial"))
        return {
            "total_energy_kwh": total_energy_kwh,
            "wasted_energy_kwh": wasted_energy,
            "carbon_tons": (total_energy_kwh * grid_factor) / 1000,
            "savings_inr": wasted_energy * tariff
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/billing")
async def get_billing():
    try:
        tel = pd.read_csv(DATA_DIR / "telemetry.csv")
        assets = pd.read_csv(DATA_DIR / "assets.csv")
        depts = ["Computer Science", "Mechanical Eng", "Admin/Central", "Hostels", "Library"]
        np.random.seed(42)
        assets["department"] = np.random.choice(depts, size=len(assets))
        billing_df = tel.merge(assets[["asset_id", "department"]], on="asset_id")
        dept_usage = billing_df.groupby("department")["interval_energy_kwh"].sum().reset_index().fillna(0)
        tariff = float(assumptions.get_value("tariff.electricity_commercial"))
        dept_usage["cost_inr"] = dept_usage["interval_energy_kwh"] * tariff
        return dept_usage.to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/alerts")
async def get_alerts():
    try:
        events = pd.read_csv(DATA_DIR / "events.csv").fillna("")
        comp_path = DATA_DIR / "complaints.csv"
        complaints = pd.read_csv(comp_path).fillna("") if comp_path.exists() else pd.DataFrame()
        recent = events.sort_values("start_time", ascending=False).head(5)
        return {
            "events": recent.to_dict(orient="records"),
            "complaints_summary": complaints["category"].value_counts().to_dict() if not complaints.empty else {}
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/microgrid")
async def get_microgrid():
    try:
        tel = pd.read_csv(DATA_DIR / "telemetry.csv")
        tel["timestamp"] = pd.to_datetime(tel["timestamp"])
        latest_ts = tel["timestamp"].max()
        latest_day = tel[tel["timestamp"].dt.date == latest_ts.date()].copy()
        if latest_day.empty:
            latest_day = tel.copy()
            
        campus_hourly = latest_day.groupby(latest_day["timestamp"].dt.hour)["power_kw"].sum().reset_index().fillna(0)
        campus_hourly.rename(columns={"timestamp": "hour", "power_kw": "campus_load_kw"}, inplace=True)
        campus_hourly["solar_gen_kw"] = campus_hourly["hour"].apply(lambda h: max(0, 800 - (abs(13 - h) * 150)) if 6 <= h <= 18 else 0.0)
        
        soc = 1000.0
        bat_dispatch = []
        grid = []
        tod_peak = float(assumptions.get_value("tariff.tod_peak"))
        tod_off_peak = float(assumptions.get_value("tariff.tod_off_peak"))
        for _, row in campus_hourly.iterrows():
            solar = row["solar_gen_kw"]
            load = row["campus_load_kw"]
            price = tod_peak if 16 <= row["hour"] <= 21 else tod_off_peak
            
            d_kw = 0.0
            if price > 20 and soc > 200:
                d_kw = -min(soc, 400.0)
            elif solar > load and soc < 2000.0:
                d_kw = min(2000.0 - soc, solar - load)
            
            soc += d_kw
            bat_dispatch.append(clean_float(d_kw))
            grid.append(clean_float(max(0, load - solar + d_kw)))
            
        campus_hourly["battery_dispatch_kw"] = bat_dispatch
        campus_hourly["grid_import_kw"] = grid
        
        return campus_hourly.to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
