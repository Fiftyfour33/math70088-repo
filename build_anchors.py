import os, json, argparse, math
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthBegin, QuarterBegin, DateOffset

def coerce_numeric(s):
    return pd.to_numeric(s.replace({'.': np.nan, 'NA': np.nan, 'NaN': np.nan, '': np.nan}), errors='coerce')

def load_series(cfg, root):
    path = os.path.join(root, cfg["file"])
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)

    # allow both custom and default column names
    dcol = cfg.get("date_col", "DATE")
    vcol = cfg.get("value_col", "VALUE")

    # fallback if user-specified names aren't present
    if dcol not in df.columns:
        if "DATE" in df.columns:
            dcol = "DATE"
        elif "observation_date" in df.columns:
            dcol = "observation_date"
        else:
            raise ValueError(f"{cfg['name']}: could not find a date column in {path}")

    if vcol not in df.columns:
        # try VALUE, then the first non-date column
        if "VALUE" in df.columns:
            vcol = "VALUE"
        else:
            candidates = [c for c in df.columns if c.lower() not in ("date","observation_date","realtime_start","realtime_end")]
            if not candidates:
                raise ValueError(f"{cfg['name']}: could not infer a value column in {path}")
            vcol = candidates[-1]  # pick the last (often the series-id column)

    out = pd.DataFrame({
        "date": pd.to_datetime(df[dcol], errors="coerce"),
        "value": coerce_numeric(df[vcol])
    }).dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # unit scaling
    scale = float(cfg.get("scale", 1.0))
    out["value"] = out["value"] * scale
    # Drop duplicates, keep last (latest vintage)
    out = out.drop_duplicates(subset=["date"], keep="last")
    return out

def stamp_effective_dates(df, frequency, policy, lag_months, lag_quarters):
    """
    Produce an effective date 'date_eff' for each observation according to:
      - frequency: "D" | "M" | "Q"
      - policy:
          * "as_is":    contemporaneous; stamp at start of the same period
          * "period_plus_months":   stamp at start of next month (+lag_months)
          * "period_plus_quarters": stamp at start of next quarter (+lag_quarters)
    Then 'lag_days' is applied elsewhere after this function.
    """
    s = df.copy()
    s["date"] = pd.to_datetime(s["date"], errors="coerce")
    if frequency == "D":
        # use as provided (daily)
        s["date_eff"] = s["date"]

    elif frequency == "M":
        # normalize to month start (same month)
        month_start = s["date"].dt.to_period("M").dt.start_time
        if policy == "as_is":
            s["date_eff"] = month_start
        elif policy == "period_plus_months":
            # first day of NEXT month (+ any extra months)
            shift = 1 + int(lag_months)
            s["date_eff"] = month_start + MonthBegin(shift)
        else:
            raise ValueError(f"Unknown stamp_policy for monthly: {policy}")

    elif frequency == "Q":
        # normalize to quarter start (same quarter)
        q_start = s["date"].dt.to_period("Q").dt.start_time
        if policy == "as_is":
            s["date_eff"] = q_start
        elif policy == "period_plus_quarters":
            shift = 1 + int(lag_quarters)
            s["date_eff"] = q_start + QuarterBegin(shift)
        else:
            raise ValueError(f"Unknown stamp_policy for quarterly: {policy}")

    else:
        # keep original date
        s["date_eff"] = s["date"]

    return s

def apply_day_lag(s, lag_days):
    if lag_days and int(lag_days) > 0:
        s["date_eff"] = s["date_eff"] + pd.to_timedelta(int(lag_days), unit="D")
    return s

def fuse_anchors(config_path, data_root, out_csv):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    series_cfgs = cfg["series"]
    cal_cfg = cfg.get("calendar", {})
    all_frames = []
    min_date, max_date = None, None

    for sc in series_cfgs:
        name = sc["name"]
        print(f"Processing {name} ...")
        df = load_series(sc, data_root)

        freq   = sc.get("frequency", "D")
        policy = sc.get("stamp_policy", "as_is")
        lag_m  = int(sc.get("lag_months", 0))
        lag_q  = int(sc.get("lag_quarters", 0))

        df = stamp_effective_dates(df, freq, policy, lag_m, lag_q)
        # Extra day lag for safety if desired
        df = apply_day_lag(df, sc.get("lag_days", 0))

        # keep only (date_eff, value)
        df = df[["date_eff", "value"]].rename(columns={"date_eff": "date", "value": name})
        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        if len(df):
            dmin, dmax = df["date"].min(), df["date"].max()
            min_date = dmin if min_date is None else min(min_date, dmin)
            max_date = dmax if max_date is None else max(max_date, dmax)

        all_frames.append(df)

    # build calendar
    start = pd.to_datetime(cal_cfg.get("start")) if cal_cfg.get("start") else min_date
    end   = pd.to_datetime(cal_cfg.get("end"))   if cal_cfg.get("end")   else max_date
    if start is None or end is None:
        raise ValueError("Cannot determine calendar start/end from data or config.")
    dates = pd.DataFrame({"date": pd.date_range(start, end, freq=cal_cfg.get("freq","D"))})

    # merge all and forward-fill daily
    anchors = dates.copy()
    for df in all_frames:
        anchors = anchors.merge(df, on="date", how="left")

    anchors = anchors.sort_values("date").reset_index(drop=True)
    anchors.iloc[:, 1:] = anchors.iloc[:, 1:].ffill()

    # clamp extreme values (avoid inf/nan)
    anchors.replace([np.inf, -np.inf], np.nan, inplace=True)
    anchors.iloc[:, 1:] = anchors.iloc[:, 1:].fillna(method="ffill")

    anchors.to_csv(out_csv, index=False)
    print(f"Saved to {out_csv} | rows={len(anchors)} | cols={anchors.shape[1]}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fuse macro CSVs into daily anchors.csv (leak-safe)")
    ap.add_argument("--config", default="anchors_config.json")
    ap.add_argument("--data-root", default=".")
    ap.add_argument("--out", default="anchors.csv")
    args = ap.parse_args()
    fuse_anchors(args.config, args.data_root, args.out)