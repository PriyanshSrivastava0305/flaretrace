import pandas as pd
from lightkurve import search_lightcurve

def fetch_lightcurve(target_name: str, mission="Kepler") -> pd.DataFrame:
    """
    Fetch, clean, and return lightcurve data as a pandas DataFrame.
    """
    search_result = search_lightcurve(target_name, mission=mission)

    if not search_result:
        raise ValueError(f"No lightcurve found for target: {target_name}")

    lc = search_result.download_all().stitch().remove_nans().normalize()
    df = lc.to_pandas()

    print("📊 Original columns:", df.columns.tolist())

    # Attempt renaming
    rename_map = {}
    if "time" in df.columns:
        rename_map["time"] = "datetime"
    if "flux" in df.columns:
        rename_map["flux"] = "flux"
    if "flux_err" in df.columns:
        rename_map["flux_err"] = "flux_err"

    df.rename(columns=rename_map, inplace=True)

    # Convert to datetime only if possible
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"].astype(str), errors="coerce")
        df.dropna(subset=["datetime"], inplace=True)
    else:
        print("⚠️ 'datetime' column not found after renaming. Skipping time parsing.")

    return df

if __name__ == "__main__":
    print("🚀 Fetching lightcurve for Kepler-10...")
    try:
        df = fetch_lightcurve("Kepler-10", mission="Kepler")
        print("✅ Data fetched successfully.")
        print("📊 Final columns:", df.columns.tolist())
        print("🔍 Preview:")
        print(df.head())

        output_path = "data/kepler10_lightcurve.csv"
        df.to_csv(output_path, index=False)
        print(f"💾 Saved to '{output_path}' with {len(df)} rows.")
    except Exception as e:
        print(f"❌ Error: {e}")
