import pandas as pd
import streamlit as st
from lightkurve import search_lightcurve

@st.cache_data(show_spinner="🔭 Fetching lightcurve from NASA archive...")
def fetch_lightcurve(target_name: str, mission="Kepler") -> pd.DataFrame:
    """
    Fetch, clean, and return lightcurve data as a pandas DataFrame.
    Handles the case when time column is the DataFrame index.
    """
    search_result = search_lightcurve(target_name, mission=mission)

    if not search_result:
        raise ValueError(f"No lightcurve found for target: {target_name}")

    lc = search_result.download_all().stitch().remove_nans().normalize()
    df = lc.to_pandas()

    # If time is the index, move it into a column named 'datetime'
    if df.index.name == "time":
        df.reset_index(inplace=True)
        df.rename(columns={"time": "datetime"}, inplace=True)

    # Rename other columns if needed
    rename_map = {}
    if "flux" in df.columns:
        rename_map["flux"] = "flux"
    if "flux_err" in df.columns:
        rename_map["flux_err"] = "flux_err"

    df.rename(columns=rename_map, inplace=True)

    if "datetime" not in df.columns:
        st.warning("⚠️ 'datetime' column not found. Attempting to use index as datetime.")
        df["datetime"] = pd.to_datetime(df.index.astype(str), errors="coerce")
    else:
        df["datetime"] = pd.to_datetime(df["datetime"].astype(str), errors="coerce")

    df.dropna(subset=["datetime"], inplace=True)

    return df
