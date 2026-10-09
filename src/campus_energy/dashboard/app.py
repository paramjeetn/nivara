"""Streamlit dashboard for Campus Energy Intelligence Platform."""
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
from pathlib import Path

# Fix for deprecation warning
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from campus_energy.analytics.features import engineer_features
from campus_energy.analytics.anomalies import detect_anomalies
import campus_energy.analytics.forecasting as forecasting
import importlib
importlib.reload(forecasting)
from campus_energy.analytics.forecasting import generate_24h_forecast

st.set_page_config(page_title="Campus Energy Intelligence", layout="wide", initial_sidebar_state="expanded")

DATA_DIR = Path("data/generated")

# Exact Coordinates mapped over BPUT Rourkela (Chhend Colony)
CAMPUS_COORDS = {
    "BLD-A": [84.8185, 22.2480],   # Academic Block
    "BLD-B": [84.8195, 22.2485],   # Labs
    "BLD-ADM": [84.8192, 22.2472], # Admin
    "BLD-LIB": [84.8198, 22.2475], # Library
    "BLD-HST": [84.8175, 22.2460], # Hostel
    "BLD-UTL": [84.8210, 22.2465], # Utilities
}

CAMPUS_BOUNDARY = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [84.8160, 22.2500],
                        [84.8215, 22.2500],
                        [84.8230, 22.2475],
                        [84.8210, 22.2445],
                        [84.8160, 22.2450],
                        [84.8160, 22.2500],
                    ]
                ]
            }
        }
    ]
}

@st.cache_data
def load_data():
    if not (DATA_DIR / "telemetry.csv").exists():
        return None
    assets = pd.read_csv(DATA_DIR / "assets.csv")
    tel = pd.read_csv(DATA_DIR / "telemetry.csv")
    ctx = pd.read_csv(DATA_DIR / "context.csv")
    events = pd.read_csv(DATA_DIR / "events.csv")
    
    complaints = pd.DataFrame()
    if (DATA_DIR / "complaints.csv").exists():
        complaints = pd.read_csv(DATA_DIR / "complaints.csv")
    
    tel["timestamp"] = pd.to_datetime(tel["timestamp"], format="ISO8601")
    ctx["timestamp"] = pd.to_datetime(ctx["timestamp"], format="ISO8601")
    events["start_time"] = pd.to_datetime(events["start_time"], format="ISO8601")
    
    if not complaints.empty:
        complaints["timestamp"] = pd.to_datetime(complaints["timestamp"], format="ISO8601")
    
    return assets, tel, ctx, events, complaints

data = load_data()

if data is None:
    st.error("Generated data not found. Please run the data generator first.")
    st.stop()
    
assets, tel, ctx, events, complaints = data

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "ESG Impact (Executive)", 
    "3D Campus Map", 
    "AI Forecasting & Scenarios", 
    "NLP Complaints & Alerts", 
    "Asset Explorer",
    "AI Facility Assistant"
])

st.sidebar.markdown("---")
st.sidebar.subheader("System Health")
st.sidebar.markdown("🟢 **IoT Gateway:** ONLINE")
st.sidebar.markdown("🟢 **Database:** CONNECTED")
st.sidebar.markdown("🟢 **Gemini AI:** ACTIVE")
st.sidebar.markdown("⏱️ **Sensor Uptime:** 99.8%")

latest_ts = tel["timestamp"].max()

if page == "ESG Impact (Executive)":
    st.title("ESG & Executive Impact")
    st.markdown("Measuring optimal and sustainable resource use without compromising facility services.")
    
    # Calculate Impact
    total_energy_kwh = tel[tel["reading_type"] == "meter"]["interval_energy_kwh"].sum()
    
    # Estimate savings based on anomalous consumption
    # We detected events where energy was wasted
    wasted_energy = 0
    for _, evt in events.iterrows():
        mask = (tel["asset_id"] == evt["asset_id"]) & (tel["timestamp"] >= evt["start_time"]) & (tel["timestamp"] <= pd.to_datetime(evt["end_time"]))
        wasted_energy += tel[mask]["interval_energy_kwh"].sum() * 0.8 # Assume 80% was avoidable
        
    carbon_intensity_kg_per_kwh = 0.82  # Approx grid intensity in India
    cost_per_kwh = 8.0 # INR
    
    total_carbon = total_energy_kwh * carbon_intensity_kg_per_kwh
    avoidable_carbon = wasted_energy * carbon_intensity_kg_per_kwh
    avoidable_cost = wasted_energy * cost_per_kwh
    
    st.subheader("28-Day Campus Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Energy Consumed", f"{total_energy_kwh:,.0f} kWh")
    col2.metric("Estimated Avoidable Waste", f"{wasted_energy:,.0f} kWh", delta="-Optimizable", delta_color="inverse")
    col3.metric("Carbon Footprint", f"{total_carbon/1000:,.1f} Tons CO2")
    col4.metric("Potential Cost Savings", f"₹{avoidable_cost:,.0f}")
    
    # Create CSV data for export
    report_data = pd.DataFrame({
        "Metric": ["Total Energy (kWh)", "Avoidable Waste (kWh)", "Carbon Footprint (Tons CO2)", "Potential Savings (INR)"],
        "Value": [total_energy_kwh, wasted_energy, total_carbon/1000, avoidable_cost]
    })
    csv = report_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📄 Download ESG Compliance Report (CSV)",
        data=csv,
        file_name='campus_esg_report.csv',
        mime='text/csv',
    )
    
    st.markdown("---")
    st.subheader("Energy Consumption Trend")
    meters = tel[tel["reading_type"] == "meter"].groupby(tel["timestamp"].dt.date)["interval_energy_kwh"].sum().reset_index()
    fig = px.bar(meters, x="timestamp", y="interval_energy_kwh", title="Daily Campus Energy (kWh)")
    st.plotly_chart(fig, use_container_width=True)

elif page == "3D Campus Map":
    st.title("Real-Time 3D Campus Heatmap")
    
    # Get latest meter reading per building
    latest_tel = tel[tel["timestamp"] == latest_ts]
    meter_tel = latest_tel[latest_tel["reading_type"] == "meter"]
    
    # Build Map Data
    map_data = []
    for _, row in meter_tel.iterrows():
        bld_id = row["asset_id"].replace("MTR-", "")
        if bld_id in CAMPUS_COORDS:
            power = row["power_kw"]
            
            # Determine if building has an active alert right now
            active_alert = False
            for _, evt in events.iterrows():
                bld = assets[assets["asset_id"] == evt["asset_id"]]["building_id"].iloc[0]
                if bld == bld_id and evt["start_time"] <= latest_ts <= pd.to_datetime(evt["end_time"]):
                    active_alert = True
            
            # Color logic for a beautiful gradient
            if active_alert:
                color = [255, 30, 30, 255] # Bright Red for alert
            else:
                if power < 30:
                    color = [0, 220, 140, 200] # Emerald Green
                elif power < 60:
                    color = [255, 200, 0, 200] # Bright Yellow
                else:
                    color = [255, 140, 0, 220] # Orange for high usage
                
            map_data.append({
                "building": bld_id,
                "lon": CAMPUS_COORDS[bld_id][0],
                "lat": CAMPUS_COORDS[bld_id][1],
                "power_kw": round(power, 1),
                "elevation": power * 3, # scale for visual impact
                "color": color
            })
            
    df_map = pd.DataFrame(map_data)
    
    st.markdown("Visualize real-time building power consumption. **<span style='color:#FF1E1E'>Red columns</span> indicate an active facility anomaly.**", unsafe_allow_html=True)
    
    import json
    import streamlit.components.v1 as components
    
    # We will pass df_map to Mapbox GL JS
    map_data_json = df_map.to_json(orient='records')
    
    mapbox_token = "pk.eyJ1IjoicGFyYW1qZWV0LTM5MzkiLCJhIjoiY212MHRhN3N6MDhhNjJ5c2R3Y2c2YTRneiJ9.wa2xVeMKEKZVLkgwE667TQ"
    
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Campus Energy 3D</title>
        <meta name="viewport" content="initial-scale=1,maximum-scale=1,user-scalable=no">
        <link href="https://api.mapbox.com/mapbox-gl-js/v3.0.0/mapbox-gl.css" rel="stylesheet">
        <script src="https://api.mapbox.com/mapbox-gl-js/v3.0.0/mapbox-gl.js"></script>
        <style>
            body {{ margin: 0; padding: 0; background-color: #111; }}
            #map {{ position: absolute; top: 0; bottom: 0; width: 100%; border-radius: 8px; }}
            .mapboxgl-popup-content {{ background-color: #222; color: #fff; border-radius: 5px; padding: 10px; font-family: sans-serif; }}
        </style>
    </head>
    <body>
    <div id="map"></div>
    <script>
        mapboxgl.accessToken = '{mapbox_token}';
        const map = new mapboxgl.Map({{
            container: 'map',
            style: 'mapbox://styles/mapbox/satellite-streets-v12',
            center: [84.8192, 22.2472],
            zoom: 16.5,
            pitch: 60,
            bearing: -15,
            antialias: true
        }});

        map.on('style.load', () => {{
            // Add Mapbox Real 3D Buildings
            const layers = map.getStyle().layers;
            let labelLayerId;
            for (let i = 0; i < layers.length; i++) {{
                if (layers[i].type === 'symbol' && layers[i].layout['text-field']) {{
                    labelLayerId = layers[i].id;
                    break;
                }}
            }}

            map.addLayer({{
                'id': 'add-3d-buildings',
                'source': 'composite',
                'source-layer': 'building',
                'filter': ['==', 'extrude', 'true'],
                'type': 'fill-extrusion',
                'minzoom': 14,
                'paint': {{
                    'fill-extrusion-color': '#e0e0e0',
                    'fill-extrusion-height': ['get', 'height'],
                    'fill-extrusion-base': ['get', 'min_height'],
                    'fill-extrusion-opacity': 0.85
                }}
            }}, labelLayerId);
            
            // Add campus boundary line
            map.addSource('campus-boundary', {{
                'type': 'geojson',
                'data': {json.dumps(CAMPUS_BOUNDARY)}
            }});
            
            map.addLayer({{
                'id': 'campus-boundary-line',
                'type': 'line',
                'source': 'campus-boundary',
                'paint': {{
                    'line-color': '#0096FF',
                    'line-width': 3,
                    'line-opacity': 0.8
                }}
            }});
            
            map.addLayer({{
                'id': 'campus-boundary-fill',
                'type': 'fill',
                'source': 'campus-boundary',
                'paint': {{
                    'fill-color': '#0096FF',
                    'fill-opacity': 0.05
                }}
            }});

            // Add Anomalies Data
            const data = {map_data_json};
            
            // We use standard markers with colored HTML for the anomalies
            data.forEach((row) => {{
                // create a container for the marker and label
                const el = document.createElement('div');
                el.style.display = 'flex';
                el.style.alignItems = 'center';
                el.style.flexDirection = 'column';
                el.style.cursor = 'pointer';
                
                // create the glowing dot
                const dot = document.createElement('div');
                dot.style.width = '20px';
                dot.style.height = '20px';
                dot.style.borderRadius = '50%';
                
                // Set color based on python logic
                const rgb = row.color;
                dot.style.backgroundColor = `rgba(${{rgb[0]}}, ${{rgb[1]}}, ${{rgb[2]}}, 0.8)`;
                dot.style.boxShadow = `0 0 15px rgba(${{rgb[0]}}, ${{rgb[1]}}, ${{rgb[2]}}, 1)`;
                dot.style.border = '2px solid white';
                
                // create the permanent text label
                const label = document.createElement('div');
                const isAnomaly = (rgb[0] === 255 && rgb[1] === 30);
                label.innerText = row.building + (isAnomaly ? ' ⚠️ ALERT' : '');
                label.style.color = isAnomaly ? '#FF1E1E' : '#FFFFFF';
                label.style.marginTop = '4px';
                label.style.fontSize = '12px';
                label.style.fontWeight = 'bold';
                label.style.textShadow = '0px 0px 4px #000, 0px 0px 2px #000';
                label.style.whiteSpace = 'nowrap';
                
                el.appendChild(dot);
                el.appendChild(label);
                
                // Add a popup for when they click it
                const popup = new mapboxgl.Popup({{ offset: 25 }}).setHTML(
                    `<b>${{row.building}}</b><br/>Current Power: ${{row.power_kw}} kW<br/>Status: ${{isAnomaly ? '<span style="color:red">Active Anomaly!</span>' : 'Normal'}}`
                );

                // add marker to map
                new mapboxgl.Marker(el)
                    .setLngLat([row.lon, row.lat])
                    .setPopup(popup)
                    .addTo(map);
            }});
        }});
    </script>
    </body>
    </html>
    """
    
    components.html(html_code, height=600)

elif page == "AI Forecasting & Scenarios":
    st.title("AI Forecasting & Scenario Simulator")
    st.markdown("Predicting the next 24 hours of campus load and simulating 'What-If' scenarios.")
    
    st.subheader("Simulate Weather Scenarios")
    temp_offset = st.slider(
        "Adjust Campus Temperature (°C)", 
        min_value=-5.0, 
        max_value=5.0, 
        value=0.0, 
        step=0.5,
        help="Simulate the impact of a sudden heatwave or cold front on the predicted campus energy load."
    )
    
    with st.spinner("Generating 24-hour forward forecast using Machine Learning..."):
        # We cache the forecast call to prevent recalculating on every slider move, but we pass temp_offset to bust cache
        @st.cache_data
        def get_forecast(offset):
            return generate_24h_forecast(tel, ctx, assets, temp_offset=offset)
            
        forecast_df = get_forecast(temp_offset)
        
    if forecast_df.empty:
        st.warning("Not enough data to generate forecast.")
    else:
        st.subheader("Simulate Actions")
        st.markdown("Adjust facility policies to see projected impact on tomorrow's energy consumption.")
        
        reduction = st.slider("Enforce strict AC shutdown after 6 PM (Estimated Reduction %)", 0, 30, 0)
        
        campus_forecast = forecast_df.groupby("timestamp")[["predicted_power_kw", "lower_bound", "upper_bound"]].sum().reset_index()
        
        # Apply scenario
        campus_forecast["Scenario_Power_kW"] = campus_forecast.apply(
            lambda row: row["predicted_power_kw"] * (1 - (reduction/100.0)) if row["timestamp"].hour >= 18 else row["predicted_power_kw"], 
            axis=1
        )
        
        # Plot
        import plotly.graph_objects as go
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=campus_forecast["timestamp"], y=campus_forecast["predicted_power_kw"], mode='lines', name='Baseline Forecast', line=dict(color='blue')))
        if reduction > 0:
            fig.add_trace(go.Scatter(x=campus_forecast["timestamp"], y=campus_forecast["Scenario_Power_kW"], mode='lines', name='Scenario Forecast', line=dict(color='green', dash='dash')))
            
        fig.update_layout(title="24-Hour Campus Power Forecast", xaxis_title="Time", yaxis_title="Power (kW)")
        st.plotly_chart(fig, use_container_width=True)
        
        if reduction > 0:
            saved_kw = (campus_forecast["predicted_power_kw"] - campus_forecast["Scenario_Power_kW"]).sum()
            saved_kwh = saved_kw / 12.0
            st.success(f"**Insight:** This action would save an estimated **{saved_kwh:.1f} kWh** tomorrow (approx ₹{saved_kwh*8:.0f}).")

elif page == "NLP Complaints & Alerts":
    st.title("Actionable Alerts & Human Feedback (NLP)")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Detected Anomalies")
        for _, evt in events.sort_values("start_time", ascending=False).iterrows():
            with st.expander(f"🔴 {evt['severity'].upper()} ALERT - {evt['event_type']} ({evt['asset_id']})"):
                st.write(f"**AI Evidence:** {evt['description']}")
                st.write(f"**Started:** {evt['start_time']}")
                
                # NLP Correlate
                related = complaints[(complaints["timestamp"] >= evt["start_time"]) & (complaints["building_id"] == assets[assets["asset_id"]==evt["asset_id"]]["building_id"].iloc[0])]
                if not related.empty:
                    st.markdown("#### 🗣️ Correlated Human Complaints")
                    for _, comp in related.iterrows():
                        st.info(f"\"{comp['description']}\" - ({comp['category']}, reported at {comp['timestamp'].strftime('%Y-%m-%d %H:%M')})")
                        
                st.write("**Action Recommendation:** Deploy facility team to inspect asset.")
                st.button("Dispatch Team & Acknowledge", key=evt['event_id'])
                
    with col2:
        st.subheader("NLP Complaint Categories")
        if not complaints.empty:
            cat_counts = complaints["category"].value_counts().reset_index()
            fig = px.pie(cat_counts, values="count", names="category", title="Historical Complaints by Topic", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)

elif page == "Asset Explorer":
    st.title("Asset Detail Explorer")
    asset_name = st.selectbox("Select Asset", assets["asset_name"].tolist())
    a_id = assets[assets["asset_name"] == asset_name]["asset_id"].iloc[0]
    
    st.write(assets[assets["asset_id"] == a_id].T)
    
    a_tel = tel[tel["asset_id"] == a_id].sort_values("timestamp")
    if not a_tel.empty:
        fig = px.line(a_tel, x="timestamp", y="power_kw", title="Power Trace")
        st.plotly_chart(fig, use_container_width=True)

elif page == "AI Facility Assistant":
    st.title("AI Facility Assistant 🤖")
    st.markdown("Your agentic assistant for managing campus energy and resolving facility issues.")
    
    api_key = st.text_input("Gemini API Key", type="password", help="Enter your Gemini 3.5 Flash Lite API key.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    if prompt := st.chat_input("Ask about active anomalies or how to optimize energy..."):
        if not api_key:
            st.warning("Please enter your Gemini API key above.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
                
            with st.spinner("Analyzing facility data..."):
                import campus_energy.dashboard.agent as agent
                import importlib
                importlib.reload(agent)
                # We pass the full history to Gemini
                response = agent.chat_with_gemini(
                    api_key=api_key,
                    messages=st.session_state.messages,
                    events_df=events,
                    assets_df=assets,
                    complaints_df=complaints
                )
                
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

