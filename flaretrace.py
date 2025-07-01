# flaretrace_app.py
# Run with: streamlit run flaretrace_app.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from lightkurve import search_lightcurve
from datetime import datetime
from etsi.watchdog import DriftCheck, Monitor

# === CONFIG ===
TARGET = "Kepler-10"
MISSION = "Kepler"
WINDOW = 300
STEP = 60
DRIFT_THRESHOLD = 0.2
FEATURES = ["flux", "flux_err"]

st.set_page_config(layout="wide")
st.title("🪐 flaretrace | AI-Powered Stellar Drift & Contamination Monitor")

# === Light Curve Fetch ===
@st.cache_data(show_spinner=False)
def load_lightcurve(target, mission):
    lc = search_lightcurve(target, mission=mission).download().remove_outliers()
    df = pd.DataFrame({
        "time": lc.time.value,
        "flux": lc.flux.value.astype(float),
        "flux_err": lc.flux_err.value.astype(float)
    }).dropna()
    df["time_diff"] = df["time"].diff().fillna(0)
    df["datetime"] = pd.to_datetime(df["time"] + 2454833, unit="D", origin="julian")
    df.set_index("datetime", inplace=True)
    return df

# === DriftCheck Window-Based ===
@st.cache_data(show_spinner=False)
def compute_drift(df, window_size=WINDOW, step_size=STEP):
    logs = []
    check = DriftCheck(df.iloc[:window_size])
    for start in range(window_size, len(df) - window_size, step_size):
        current = df.iloc[start:start + window_size]
        result = check.run(current, features=FEATURES)
        record = {
            "start": df.index[start],
            "end": df.index[start + window_size - 1],
        }
        for feat in FEATURES:
            record[f"psi_{feat}"] = result[feat].score
            record[f"drift_{feat}"] = result[feat].is_drifted
        logs.append(record)
    return pd.DataFrame(logs)

# === Rolling PSI Monitoring ===
@st.cache_data(show_spinner=False)
def run_rolling_monitor(df, features=FEATURES):
    monitor = Monitor(reference_df=df.iloc[:WINDOW])
    monitor.enable_logging("logs/rolling_log.csv")
    results = monitor.watch_rolling(df, window=WINDOW, freq="D", features=features)

    return results

# === Placeholder for Spectral Correction ===
def simulate_spectral_correction(drift_df, spectrum_df):
    # This is placeholder logic: drop windows with high drift
    contaminated = drift_df[(drift_df["drift_flux"]) | (drift_df["drift_flux_err"])]
    st.info(f"⚠️ Spectral correction applied to {len(contaminated)} contaminated windows.")
    spectrum_df["corrected_flux"] = spectrum_df["flux"] * 0.98  # dummy correction
    return spectrum_df

# === UI Sidebar ===
with st.sidebar:
    target = st.text_input("🔭 Target Star", TARGET)
    mission = st.selectbox("📡 Mission", ["Kepler", "TESS"])
    do_fetch = st.button("🚀 Load Light Curve")

if do_fetch:
    df = load_lightcurve(target, mission)
    st.success(f"✅ Loaded {len(df)} records for {target} ({mission})")

    st.subheader("🔎 Light Curve (Flux over Time)")
    st.line_chart(df["flux"].rolling(10).mean())

    st.subheader("📊 Drift Detection (PSI)")
    drift_df = compute_drift(df)
    st.dataframe(drift_df.head(10))

    st.line_chart({
        "Flux PSI": drift_df["psi_flux"],
        "Flux Err PSI": drift_df["psi_flux_err"]
    })

    st.subheader("⏳ Rolling Drift Monitor (Daily)")
    results = run_rolling_monitor(df)
    st.success("📄 Drift logged to logs/rolling_log.csv")

    st.subheader("🧪 Feature Histogram (BLS Simulated)")
    feat = st.selectbox("Feature", FEATURES)
    result = DriftCheck(df.iloc[:WINDOW]).run(df.iloc[WINDOW:2*WINDOW], features=FEATURES)
    result[feat].plot()
    st.pyplot(plt.gcf())

    st.subheader("🧬 Spectral Contamination Correction")
    dummy_spec = pd.DataFrame({  # simulated spectral dummy data
        "wavelength": np.linspace(0.6, 1.2, 100),
        "flux": np.random.normal(1, 0.005, 100)
    })
    corrected = simulate_spectral_correction(drift_df, dummy_spec)
    st.line_chart({
        "Original": dummy_spec["flux"],
        "Corrected": corrected["corrected_flux"]
    })

    st.caption("flaretrace combines time-domain drift with atmospheric spectral correction logic.")
