import pandas as pd
from etsi.watchdog import DriftCheck, Monitor

def run_drift_check(ref_df, live_df, features):
    check = DriftCheck(ref_df)
    results = check.run(live_df, features=features)
    return results

def run_rolling_monitor(df, features, window=300, freq="D"):
    monitor = Monitor(reference_df=df.iloc[:window])
    monitor.enable_logging("logs/rolling_log.csv")

    # If etsi-watchdog doesn’t log start/end, patch it here manually:
    results = monitor.watch_rolling(df, window=window, freq=freq, features=features)
    log = pd.read_csv("logs/rolling_log.csv")
    log["start"] = pd.to_datetime(df.index[:len(log)])
    log["end"] = log["start"] + pd.Timedelta(freq)
    log.to_csv("logs/rolling_log.csv", index=False)
    return results
