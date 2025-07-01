import pandas as pd

def match_drift_to_spectrum(drift_df: pd.DataFrame, spectrum_df: pd.DataFrame) -> pd.DataFrame:
    # Convert timestamp to datetime if needed
    if 'timestamp' in drift_df.columns:
        drift_df['timestamp'] = pd.to_datetime(drift_df['timestamp'])

    spectrum_df["datetime"] = pd.to_datetime(spectrum_df["datetime"])

    # Match by closest drift timestamp
    spectrum_df["drift_score"] = spectrum_df["datetime"].apply(
        lambda t: drift_df.iloc[(drift_df["timestamp"] - t).abs().argsort().values[0]]["score"]
    )
    return spectrum_df

def apply_simple_correction(spectrum_df: pd.DataFrame) -> pd.DataFrame:
    # Simple contamination correction (e.g., subtracting drift score)
    spectrum_df["corrected_flux"] = spectrum_df["flux"] / (1 + spectrum_df["drift_score"])
    return spectrum_df
