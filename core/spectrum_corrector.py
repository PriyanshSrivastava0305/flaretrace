import pandas as pd

def match_drift_to_spectrum(drift_df: pd.DataFrame, spectrum_df: pd.DataFrame) -> pd.DataFrame:
    spectrum_df = spectrum_df.copy()
    spectrum_df["contaminated"] = False

    for _, row in drift_df.iterrows():
        if row.get("drift_flux", False) or row.get("drift_flux_err", False):
            mask = (spectrum_df["datetime"] >= row["start"]) & (spectrum_df["datetime"] <= row["end"])
            spectrum_df.loc[mask, "contaminated"] = True

    return spectrum_df

def apply_simple_correction(spectrum_df: pd.DataFrame, scale=0.98) -> pd.DataFrame:
    spectrum_df = spectrum_df.copy()
    spectrum_df["corrected_flux"] = spectrum_df["flux"]
    spectrum_df.loc[spectrum_df["contaminated"], "corrected_flux"] *= scale
    return spectrum_df
