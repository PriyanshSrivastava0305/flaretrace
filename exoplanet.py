import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightkurve import search_lightcurve
from lightkurve.periodogram import BoxLeastSquaresPeriodogram
from lightkurve.correctors import PLDCorrector
from astropy.timeseries import LombScargle
from sklearn.ensemble import IsolationForest
from etsi.watchdog import DriftCheck, Monitor
from datetime import datetime
import streamlit as st
import os

# === CONFIG ===
TARGET = "Kepler-10"
MISSION = 'Kepler'
WINDOW = 300
STEP = 60
PSI_THRESH = 0.2
DRIFT_LOG = "drift_log.csv"

# === UTILS ===
def fetch_lightcurve(target, mission='Kepler'):
    lc = search_lightcurve(target, mission=mission).download()
    if lc is None:
        raise ValueError(f"No light curve found for {target} with mission {mission}")
    lc = lc.remove_outliers()
    time = lc.time.value.astype(np.float64, copy=False)
    flux = lc.flux.value.astype(np.float64, copy=False)
    flux_err = lc.flux_err.value.astype(np.float64, copy=False)
    df = pd.DataFrame({
        'time': time,
        'flux': flux,
        'flux_err': flux_err
    }).dropna()
    df['time_diff'] = df['time'].diff().fillna(0)
    return df, lc

def run_bls(lc):
    bls = BoxLeastSquaresPeriodogram.from_lightcurve(lc)
    result = bls.power()
    return result.get_peak()[0], result

def run_lombscargle(lc):
    freq, power = LombScargle(lc.time, lc.flux).autopower()
    return 1 / freq[np.argmax(power)], freq, power

def remove_noise_and_trend(lc):
    cbv_corr = lc.correct(cotrend=True)
    return PLDCorrector.from_lightcurve(cbv_corr).correct()

def train_anomaly_model(df):
    iso = IsolationForest(contamination=0.01)
    df['anomaly'] = iso.fit_predict(df[['flux', 'flux_err']])
    return df

# === STREAMLIT UI ===
st.set_page_config(page_title="🔭 Advanced Exoplanet Lab", layout="wide")
st.title("🌌 Exoplanet & Stellar Lab — Advanced Drift & Pattern Analysis")

with st.sidebar:
    st.header("⚙️ Configuration")
    TARGET = st.text_input("Target Star", TARGET)
    MISSION = st.selectbox("Mission", ["Kepler", "TESS"], index=0)
    features = st.multiselect("Drift Detection Features", ["flux", "flux_err"], default=["flux", "flux_err"])
    step_size = st.slider("Simulation Step Size", 30, 300, STEP, 30)
    plot_zoom = st.slider("Zoom on Flux Plot (%)", 1, 100, 10)
    options = st.multiselect("Run Modules", [
        "Drift Detection (etsi-watchdog)",
        "Transit Detection (BLS)",
        "Rotation Period (LS)",
        "Noise Correction",
        "Anomaly Detection"
    ], default=[])

st.markdown("---")

try:
    df, lc = fetch_lightcurve(TARGET, MISSION)
    st.success(f"✅ Fetched {len(df)} records for {TARGET}")
    plot_df = df.copy()
    mid = len(plot_df) // 2
    zoom = int(len(plot_df) * plot_zoom / 100)
    st.line_chart(plot_df.iloc[mid - zoom: mid + zoom].set_index('time')['flux'], use_container_width=True)
except Exception as e:
    st.error(f"❌ Could not fetch data: {e}")
    st.stop()

col1, col2 = st.columns(2)

if "Drift Detection (etsi-watchdog)" in options:
    with col1:
        st.subheader("🚨 Drift Detection (etsi-watchdog)")
        ref = df.iloc[0:WINDOW].copy()
        drift_rows = []
        monitor = Monitor(reference_df=ref)
        results = monitor.watch_rolling(df=df, window=WINDOW, freq="T", features=features)
        for feat, result in results.items():
            st.markdown(f"#### Feature: `{feat}`")
            st.pyplot(result.plot(figsize=(6, 2)))
            st.code(result.summary(), language='text')

if "Transit Detection (BLS)" in options:
    with col2:
        st.subheader("🌑 Transit Detection (BLS)")
        peak, result = run_bls(lc)
        st.info(f"Best Period: {peak.period:.5f} d, Depth: {peak.depth:.6f}, Duration: {peak.duration:.4f} d")
        fig, ax = plt.subplots(figsize=(6, 3))
        result.plot(ax=ax)
        st.pyplot(fig)

if "Rotation Period (LS)" in options:
    with col1:
        st.subheader("🔄 Rotation Period (Lomb-Scargle)")
        period, freq, power = run_lombscargle(lc)
        st.info(f"Estimated Period: {period:.4f} days")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(1 / freq, power)
        ax.set_xlabel("Period (days)")
        ax.set_ylabel("Power")
        ax.set_title("Lomb-Scargle Periodogram")
        st.pyplot(fig)

if "Noise Correction" in options:
    with col2:
        st.subheader("🧹 Noise Correction")
        corrected = remove_noise_and_trend(lc)
        st.line_chart(corrected.flux.rename("Corrected Flux"), use_container_width=True)

if "Anomaly Detection" in options:
    with col1:
        st.subheader("🧠 Anomaly Detection")
        anomaly_df = train_anomaly_model(df.copy())
        st.dataframe(anomaly_df[anomaly_df['anomaly'] == -1], use_container_width=True)

st.markdown("---")
st.markdown("### 🌐 Open Science Tools")
st.markdown("You can publish cleaned datasets to Zenodo or GitHub, and create JOSS papers to share your drift-aware pipelines with the world.")
