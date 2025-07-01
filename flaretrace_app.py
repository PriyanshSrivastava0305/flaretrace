import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from core.spectrum_corrector import match_drift_to_spectrum, apply_simple_correction
from core.retrieval_runner import run_platon_retrieval, plot_fit
from core.data_fetcher import fetch_lightcurve

st.set_page_config(page_title="FlareTrace", layout="wide")
st.title("🚀 FlareTrace: Exoplanet Atmosphere Correction and Retrieval")

# Sidebar inputs
st.sidebar.header("Input Configuration")
target = st.sidebar.text_input("Exoplanet Target", "Kepler-10")
mission = st.sidebar.selectbox("Mission", ["Kepler", "TESS"])
psi_threshold = st.sidebar.slider("PSI Threshold", 0.0, 1.0, 0.2, 0.05)

# Load lightcurve
st.subheader("📈 Lightcurve Data")
with st.spinner("Fetching light curve..."):
    df = fetch_lightcurve(target, mission=mission)
st.success(f"Fetched {len(df)} data points.")

# Load rolling drift results
# drift_df = pd.read_csv("logs/rolling_log.csv", parse_dates=["timestamp_start", "timestamp_end"])
# drift_df = pd.read_csv("logs/rolling_log.csv")

# # Parse datetime columns if they exist
# for col in ["start", "end"]:
#     if col in drift_df.columns:
#         drift_df[col] = pd.to_datetime(drift_df[col])

# drift_df["drift_flux"] = drift_df["psi_flux"] > psi_threshold
# drift_df["drift_flux_err"] = drift_df["psi_err"] > psi_threshold

# Load drift results
drift_df = pd.read_csv("logs/rolling_log.csv")

# Set target feature from your drift logs
TARGET_FEATURE = "flux_err"

# Filter for target feature
flux_drift = drift_df[drift_df["feature"] == TARGET_FEATURE]

# Initialize columns with default False
drift_df["drift_flux"] = False
drift_df["drift_flux_err"] = False

# Mark drift presence
drift_df.loc[flux_drift.index, "drift_flux"] = flux_drift["drift"]
drift_df.loc[flux_drift.index, "drift_flux_err"] = flux_drift["score"] > flux_drift["threshold"]


# Load example spectrum
st.subheader("🌈 Spectrum Correction")
spectrum_df = pd.read_csv("data/example_spectrum.csv", parse_dates=["datetime"])
spectrum_df = match_drift_to_spectrum(drift_df, spectrum_df)
corrected_df = apply_simple_correction(spectrum_df)

# Plot drift-matched spectrum
st.markdown("### Spectrum Before and After Correction")
fig, ax = plt.subplots()
ax.plot(corrected_df["wavelength"], corrected_df["flux"], label="Original", alpha=0.5)
ax.plot(corrected_df["wavelength"], corrected_df["corrected_flux"], label="Corrected", alpha=0.9)
ax.set_xlabel("Wavelength (μm)")
ax.set_ylabel("Transit Depth")
ax.legend()
st.pyplot(fig)

# Run PLATON
st.subheader("🔍 Atmospheric Retrieval (PLATON)")
wavelength = corrected_df["wavelength"].values
flux = corrected_df["corrected_flux"].values

with st.spinner("Running PLATON retrieval..."):
    model_wavelengths, model_depths, _ = run_platon_retrieval(wavelength, flux)

# Plot model vs observed
plot_fit(wavelength, corrected_df["flux"].values, corrected_df["corrected_flux"].values,
         model_wavelengths, model_depths)
