import pandas as pd
from etsi.watchdog import DriftCheck, Monitor

def run_drift_check(ref_df, live_df, features):
    check = DriftCheck(ref_df)
    results = check.run(live_df, features=features)
    return results

def run_rolling_monitor(df, features, window=300, freq="D"):
    monitor = Monitor(reference_df=df.iloc[:window])
    monitor.enable_logging("logs/rolling_log.csv")
    results = monitor.watch_rolling(df, window=window, freq=freq, features=features)
    return results
