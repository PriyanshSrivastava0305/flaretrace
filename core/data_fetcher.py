from lightkurve import search_lightcurve
import pandas as pd

def fetch_lightcurve(target: str, mission: str = "Kepler", quarter=None, cadence="long") -> pd.DataFrame:
    """
    Download and return a cleaned lightcurve for the given target from MAST.

    Args:
        target (str): Name of the target (e.g., 'Kepler-10', 'TOI-700').
        mission (str): 'Kepler' or 'TESS'.
        quarter (int): Optional; specify quarter or sector.
        cadence (str): 'short' or 'long'

    Returns:
        pd.DataFrame: Cleaned lightcurve with time, flux, flux_err
    """
    search = search_lightcurve(target, mission=mission, cadence=cadence)
    if quarter:
        search = search[search.table["quarter"] == quarter]
    
    try:
        lc = search.download().remove_nans().remove_outliers()
        df = pd.DataFrame({
            "time": lc.time.value,
            "flux": lc.flux.value,
            "flux_err": lc.flux_err.value,
        }).dropna()
        return df
    except Exception as e:
        print(f"Failed to download lightcurve: {e}")
        return pd.DataFrame()
