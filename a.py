import argparse, os, math, warnings, json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple

# Modeling & metrics
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, brier_score_loss
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True, linewidth=160)
pd.set_option("display.width", 160)

def pr_auc(y_true, p):
    return float(average_precision_score(y_true, p))

def ece_score(y_true, p, n_bins=10):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    m = len(p)
    if m == 0:
        return 0.0

    # clip to [0,1]
    p = np.clip(p, 0.0, 1.0)

    # can't have more bins than points
    nb = int(max(1, min(n_bins, m)))

    # quantile edges
    edges = np.quantile(p, np.linspace(0.0, 1.0, nb + 1))

    edges[0], edges[-1] = 0.0, 1.0
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = min(1.0, edges[i - 1] + 1e-12)

    # assign bins
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, nb - 1)

    # weighted absolute calibration gap
    N = float(m)
    ece = 0.0
    for b in range(nb):
        mask = (idx == b)
        w = int(mask.sum())
        if w == 0:
            continue
        p_hat = float(p[mask].mean())
        p_obs = float(y[mask].mean())
        ece += (w / N) * abs(p_hat - p_obs)

    return float(ece)

def precision_at_rate(y_true, p, rate=0.02):
    n = len(p)
    k = max(1, int(rate * n))
    if k >= n:
        thr = p.min() - 1e-9
    else:
        thr = np.partition(p, -k)[-k]
    hits = (p >= thr).astype(int)
    tp = int((hits * y_true).sum())
    prec = tp / hits.sum() if hits.sum() else 0.0
    rec = tp / y_true.sum() if y_true.sum() else 0.0
    realized = hits.mean()
    return float(prec), float(rec), float(realized), float(thr)

def forward_path_min_return(close: np.ndarray, H: int) -> np.ndarray:
    c = close.astype(float)
    R = np.ones_like(c)
    R[1:] = c[1:] / c[:-1]
    out = np.full_like(c, np.nan, dtype=float)
    for t in range(len(c) - H):
        path = R[t+1:t+H+1]
        cum = np.cumprod(path)
        out[t] = cum.min() - 1.0
    return out

def forward_cum_return(close: np.ndarray, H: int) -> np.ndarray:
    c = close.astype(float)
    out = np.full_like(c, np.nan, dtype=float)
    for t in range(len(c) - H):
        out[t] = (c[t+H] / c[t]) - 1.0 if c[t] > 0 else np.nan
    return out

def rolling_vol(r: np.ndarray, win: int) -> np.ndarray:
    s = pd.Series(r).rolling(win, min_periods=max(5, win//2)).std(ddof=1).values
    return np.where(np.isfinite(s), s, np.nan)

def make_features(df: pd.DataFrame, price_col: str, vol_col: str|None, H: int) -> Tuple[pd.DataFrame, list]:
    EPS = 1e-12
    close = pd.to_numeric(df.get(price_col), errors="coerce").astype(float).values
    open_  = pd.to_numeric(df.get("open"), errors="coerce").astype(float).values if "open"  in df.columns else np.full(len(df), np.nan)
    high   = pd.to_numeric(df.get("high"), errors="coerce").astype(float).values if "high"  in df.columns else np.full(len(df), np.nan)
    low    = pd.to_numeric(df.get("low"),  errors="coerce").astype(float).values if "low"   in df.columns else np.full(len(df), np.nan)
    volume = pd.to_numeric(df.get("volume"), errors="coerce").astype(float).values if "volume" in df.columns else np.zeros(len(df))
    amount = pd.to_numeric(df.get("amount"), errors="coerce").astype(float).values if "amount" in df.columns else np.zeros(len(df))

    r = pd.Series(close).pct_change().fillna(0.0).values
    r_l1 = np.roll(r, 1); r_l1[0] = 0.0
    r_l2 = np.roll(r, 2); r_l2[:2] = 0.0
    r_l3 = np.roll(r, 3); r_l3[:3] = 0.0

    def rolling_vol(arr, win):
        return pd.Series(arr).rolling(win, min_periods=max(5, win//2)).std(ddof=1).values
    vol5  = rolling_vol(r, 5)
    vol10 = rolling_vol(r, 10)
    vol20 = rolling_vol(r, 20)

    mom5  = pd.Series(close).pct_change(5).values
    mom10 = pd.Series(close).pct_change(10).values
    ma20  = pd.Series(close).rolling(20, min_periods=10).mean().values
    sd20  = pd.Series(close).rolling(20, min_periods=10).std(ddof=1).values
    z20   = (close - ma20) / (sd20 + EPS)

    ln_hl = np.log((high + EPS) / (low + EPS))
    ln_co = np.log((close + EPS) / (open_ + EPS))
    # Parkinson rolling sigma: sqrt( (1/(4 ln2)) * mean(ln(high/low)^2) )
    par_var = ln_hl**2
    def roll_par_sigma(win):
        m = pd.Series(par_var).rolling(win, min_periods=max(5, win//2)).mean().values
        return np.sqrt(np.maximum(0.0, m / (4.0 * np.log(2.0))))
    par5  = roll_par_sigma(5)
    par10 = roll_par_sigma(10)
    par20 = roll_par_sigma(20)
    # Garman-Klass rolling sigma: mean(0.5 ln(H/L)^2 - (2 ln2 -1) ln(C/O)^2)
    gk_var_daily = 0.5 * ln_hl**2 - (2.0*np.log(2.0) - 1.0) * (ln_co**2)
    gk_var_daily = np.where(np.isfinite(gk_var_daily), gk_var_daily, 0.0)
    def roll_gk_sigma(win):
        m = pd.Series(gk_var_daily).rolling(win, min_periods=max(5, win//2)).mean().values
        return np.sqrt(np.maximum(0.0, m))
    gk5  = roll_gk_sigma(5)
    gk10 = roll_gk_sigma(10)
    gk20 = roll_gk_sigma(20)

    logv = np.log1p(np.maximum(0.0, volume))
    loga = np.log1p(np.maximum(0.0, amount))
    dv   = pd.Series(volume).pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    da   = pd.Series(amount).pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    # Amihud illiquidity proxy: |r| / (amount in millions)
    amihud = np.abs(r) / (np.maximum(amount, 0.0)/1e6 + EPS)
    # clip extreme tail for stability
    q99 = np.nanpercentile(amihud, 99)
    amihud = np.clip(amihud, 0.0, q99 if np.isfinite(q99) and q99>0 else amihud.max(initial=0.0))

    feats = np.column_stack([
        r_l1, r_l2, r_l3,                 # recent returns (lagged)
        vol5, vol10, vol20,               # c2c vol
        mom5, mom10, z20,                 # momentum & z-score
        par5, par10, par20,               # Parkinson sigmas
        gk5, gk10, gk20,                  # GK sigmas
        logv, loga, dv, da, amihud        # volume/amount features
    ])
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    # labels
    mminH = forward_path_min_return(close, H)
    retH  = forward_cum_return(close, H)

    out = pd.DataFrame({
        "date": df["__date__"].values,
        "close": close,
        "mminH": mminH,
        "retH":  retH
    })
    feat_cols = [f"f_{i}" for i in range(feats.shape[1])]
    Xdf = pd.DataFrame(feats, columns=feat_cols)
    out = pd.concat([out, Xdf], axis=1)

    # drop warmup and last H rows (no forward label)
    out = out.iloc[25:len(out)-H].reset_index(drop=True)
    return out, feat_cols

def build_labels(df_train: pd.DataFrame, df_test: pd.DataFrame,
                 loss_mode: str, theta_pct: float, alpha_q: float):
    if loss_mode == "drawdown":
        thr = -abs(theta_pct)
        y_tr = (df_train["mminH"].values <= thr).astype(int)
        y_te = (df_test["mminH"].values  <= thr).astype(int)
        label_info = {"mode": "drawdown", "theta_pct": theta_pct, "threshold": thr}
    elif loss_mode == "quantile":
        q = float(np.nanquantile(df_train["retH"].values, alpha_q))
        y_tr = (df_train["retH"].values <= q).astype(int)
        y_te = (df_test["retH"].values  <= q).astype(int)
        label_info = {"mode": "quantile", "alpha_q": alpha_q, "threshold": q}
    else:
        raise ValueError("loss_mode must be 'drawdown' or 'quantile'")
    return y_tr, y_te, label_info

def plot_series_with_events(df_all, split_idx, p_test, y_test, alert_mask, out_png):
    dates = df_all["date"].values
    close = df_all["close"].values
    n = len(df_all)
    test_idx = np.arange(split_idx, n)
    d_test = dates[split_idx:]
    c_test = close[split_idx:]

    fig = plt.figure(figsize=(11, 6.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.5, 1.2], hspace=0.15)

    # price panel
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(dates, close, lw=1.2)
    # shade first 80%
    ax1.axvspan(dates[0], dates[split_idx-1], color="#dddddd", alpha=0.5, label="train (80%)")
    # ground-truth events on test
    ev_idx = test_idx[y_test.astype(bool)]
    ax1.scatter(dates[ev_idx], close[ev_idx], color="crimson", s=18, label="loss events (gt)", zorder=5)
    # predicted alerts on test
    al_idx = test_idx[alert_mask.astype(bool)]
    ax1.scatter(dates[al_idx], close[al_idx], marker="x", color="royalblue", s=36, label="predicted alerts", zorder=6)
    ax1.set_title("Price with train shading (80%) and test markings (events vs predicted alerts)")
    ax1.set_ylabel("Close")
    ax1.legend(loc="upper left", ncol=3, frameon=False)

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.plot(d_test, p_test, lw=1.2)
    ax2.set_ylabel("Pred p(event)")
    ax2.set_xlabel("Date")

    for label in ax1.get_xticklabels():
        label.set_visible(False)

    plt.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Method A: single-asset few-day loss warning benchmark (standalone)")
    ap.add_argument("--csv", required=True, help="Path to single-asset CSV with date/close (OHLCV optional)")
    ap.add_argument("--date-col", default="date", help="Date column name (default: date)")
    ap.add_argument("--price-col", default="close", help="Price column name (default: close)")
    ap.add_argument("--volume-col", default="volume", help="Volume column name if exists (optional)")
    ap.add_argument("--horizon", type=int, default=3, help="Forward horizon H in days (default: 3)")
    ap.add_argument("--loss-mode", choices=["drawdown","quantile"], default="drawdown",
                    help="Labeling mode: drawdown (path-min <= -theta) or quantile (retH <= alpha-quantile)")
    ap.add_argument("--theta-pct", type=float, default=0.05, help="Theta for drawdown (e.g., 0.05=5%)")
    ap.add_argument("--alpha-q", type=float, default=0.05, help="Alpha for quantile threshold (e.g., 0.05)")
    ap.add_argument("--train-frac", type=float, default=0.8, help="Train fraction (chronological split, default 0.8)")
    ap.add_argument("--alert-rate", type=float, default=0.02, help="Fraction of days to alert on test (default 2%)")
    ap.add_argument("--outdir", default="outputs", help="Output directory")
    ap.add_argument("--seed", type=int, default=1337, help="Random seed")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # load CSV
    df = pd.read_csv(args.csv)
    # Normalize date column
    if args.date_col not in df.columns:
        raise ValueError(f"date column '{args.date_col}' not found in CSV.")
    df["__date__"] = pd.to_datetime(df[args.date_col], errors="coerce", infer_datetime_format=True)
    df = df.sort_values("__date__").reset_index(drop=True)
    if args.price_col not in df.columns:
        raise ValueError(f"price column '{args.price_col}' not found in CSV.")
    df = df.dropna(subset=["__date__", args.price_col]).reset_index(drop=True)

    # build features
    panel, feat_cols = make_features(df, args.price_col,
                                     args.volume_col if args.volume_col in df.columns else None,
                                     H=args.horizon)
    if len(panel) < 250:
        raise ValueError("Not enough usable rows after warmup/H truncation. Provide longer history.")

    # chronological split
    n = len(panel)
    split_idx = int(args.train_frac * n)
    train = panel.iloc[:split_idx].copy()
    test  = panel.iloc[split_idx:].copy()

    y_tr, y_te, label_info = build_labels(train, test, args.loss_mode, args.theta_pct, args.alpha_q)

    # prepare matrices
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train[feat_cols].values)
    X_te = scaler.transform(test[feat_cols].values)

    # classifier with isotonic calibration
    base = LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, n_jobs=1, random_state=args.seed
    )
    clf = CalibratedClassifierCV(base, method="isotonic", cv=5)
    clf.fit(X_tr, y_tr)
    p_te = clf.predict_proba(X_te)[:, 1]

    # metrics
    prauc  = pr_auc(y_te, p_te)
    brier  = brier_score_loss(y_te, p_te)
    ece10  = ece_score(y_te, p_te, n_bins=10)
    prec, rec, realized, thr_alert = precision_at_rate(y_te, p_te, rate=args.alert_rate)

    # alerts mask on test
    k = max(1, int(args.alert_rate * len(p_te)))
    thr = np.partition(p_te, -k)[-k] if k < len(p_te) else p_te.min() - 1e-9
    alert_mask = (p_te >= thr).astype(int)

    # combine panels for plotting
    panel_all = pd.concat([train, test], axis=0, ignore_index=True)
    plot_path = os.path.join(args.outdir, f"methodA_plot_{os.path.basename(args.csv).split('.')[0]}.png")
    plot_series_with_events(panel_all, split_idx=len(train),
                            p_test=p_te, y_test=y_te,
                            alert_mask=alert_mask,
                            out_png=plot_path)

    # save metrics & predictions
    metrics = dict(
        rows_test=int(len(y_te)),
        positives_test=int(y_te.sum()),
        pos_rate_test=float(y_te.mean()),
        pr_auc=float(prauc),
        brier=float(brier),
        ece10=float(ece10),
        precision_at_rate=float(prec),
        recall_at_rate=float(rec),
        realized_rate=float(realized),
        alert_threshold=float(thr_alert),
        label_info=label_info
    )
    with open(os.path.join(args.outdir, "methodA_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    out_pred = test[["date","close"]].copy()
    out_pred["y_true"] = y_te
    out_pred["p_hat"]  = p_te
    out_pred["alert"]  = alert_mask
    out_pred.to_csv(os.path.join(args.outdir, "methodA_predictions.csv"), index=False)

    # console summary
    print(f"\n=== Method A (single-asset) ===")
    print(f"File: {args.csv}")
    print(f"Loss: {label_info}")
    print(f"Test rows: {len(y_te)} | Positives: {y_te.sum()} ({y_te.mean():.2%})")
    print(f"PR-AUC: {prauc:.4f} | Brier: {brier:.4f} | ECE@10: {ece10:.3f}")
    print(f"P@{args.alert_rate:.1%}: {prec:.3f} | R@{args.alert_rate:.1%}: {rec:.3f} | realized: {realized:.3%}")
    print(f"Plot saved to {plot_path}")
    print(f"Metrics to {os.path.join(args.outdir, 'methodA_metrics.json')}")
    print(f"Predictions to {os.path.join(args.outdir, 'methodA_predictions.csv')}")

if __name__ == "__main__":
    main()