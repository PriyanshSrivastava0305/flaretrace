# flaretrace_app.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

from core.data_fetcher import fetch_lightcurve
from core.drift_checker import run_rolling_monitor
from core.spectrum_corrector import match_drift_to_spectrum, apply_simple_correction
from core.retrieval_runner import run_platon_retrieval, plot_fit

st.set_page_config(layout="wide")
st.title("🚀 FlareTrace: Habitable Atmosphere Retrieval Pipeline")

# --- Sidebar config ---
st.sidebar.header("Configuration")
target = st.sidebar.text_input("Target Star", value="Kepler-10")
mission = st.sidebar.selectbox("Mission", ["Kepler", "TESS"], index=0)
window = st.sidebar.slider("Drift Rolling Window", min_value=50, max_value=1000, value=300)

# --- Load Lightcurve ---
st.subheader("📈 Lightcurve Data")
df = fetch_lightcurve(target, mission)
st.success(f"Fetched {len(df)} data points.")
st.line_chart(df.set_index("datetime")["flux"].rolling(3).mean())

# --- Drift Monitoring ---
st.subheader("🔍 Drift Detection (ETSI Watchdog)")
features = ["flux", "flux_err"]
drift_results = run_rolling_monitor(df, features=features, window=window)
st.success("✅ Drift log saved to `logs/rolling_log.csv`")

# Load and show drift log
log_df = pd.read_csv("logs/rolling_log.csv")
st.dataframe(log_df.tail(10))

# ✅ Add download button for drift log
csv_bytes = BytesIO()
log_df.to_csv(csv_bytes, index=False)
st.download_button(
    label="📥 Download Drift Log (CSV)",
    data=csv_bytes.getvalue(),
    file_name="drift_log.csv",
    mime="text/csv",
)

# --- Diagnostics ---
st.write("📊 Columns in the fetched DataFrame:", df.columns.tolist())
st.write("🪐 First few rows of the data:")
st.dataframe(df.head())

# --- Load Transmission Spectrum ---
st.subheader("🌈 Raw Transmission Spectrum")
spectrum_df = pd.read_csv("data/example_spectrum.csv")
st.dataframe(spectrum_df.head())
st.line_chart(spectrum_df.set_index("wavelength")["flux"].rolling(3).mean())

# --- Apply Correction ---
st.subheader("⚗️ Contamination Correction")
spectrum_df = match_drift_to_spectrum(log_df, spectrum_df)
corrected_df = apply_simple_correction(spectrum_df)
st.line_chart(corrected_df.set_index("wavelength")["corrected_flux"].rolling(3).mean())

# --- Run PLATON Retrieval ---
st.subheader("🔬 Atmospheric Retrieval (PLATON)")
model_wavelengths, model_depths, _ = run_platon_retrieval(
    corrected_df["wavelength"].values,
    corrected_df["corrected_flux"].values,
)

# ✅ Fix: call plot_fit and then display
fig = plot_fit(
    corrected_df["wavelength"].values,
    corrected_df["flux"].values,
    corrected_df["corrected_flux"].values,
    model_wavelengths,
    model_depths
)
st.pyplot(fig)
