import argparse, os, glob, json, warnings, math, random
from typing import List, Tuple, Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, brier_score_loss

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
    p = np.clip(p, 0.0, 1.0)
    nb = int(max(1, min(n_bins, m)))
    edges = np.quantile(p, np.linspace(0.0, 1.0, nb + 1))
    edges[0], edges[-1] = 0.0, 1.0
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = min(1.0, edges[i - 1] + 1e-12)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, nb - 1)
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
    thr = np.partition(p, -k)[-k] if k < n else p.min() - 1e-9
    hits = (p >= thr).astype(int)
    tp = int((hits * y_true).sum())
    prec = tp / hits.sum() if hits.sum() else 0.0
    rec = tp / y_true.sum() if y_true.sum() else 0.0
    realized = hits.mean()
    return float(prec), float(rec), float(realized), float(thr)

def forward_path_min_return(close: np.ndarray, H: int) -> np.ndarray:
    """Min path return over next H days, based on closes only."""
    c = close.astype(float)
    R = np.ones_like(c)
    R[1:] = np.divide(c[1:], c[:-1], out=np.ones_like(c[1:]), where=c[:-1] > 0)
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

def make_symbol_features(df: pd.DataFrame, price_col: str, H: int) -> Tuple[pd.DataFrame, List[str]]:
    EPS = 1e-12

    close = pd.to_numeric(df.get(price_col), errors="coerce").astype(float).values
    open_  = pd.to_numeric(df.get("open"), errors="coerce").astype(float).values if "open"  in df.columns else np.full(len(df), np.nan)
    high   = pd.to_numeric(df.get("high"), errors="coerce").astype(float).values if "high"  in df.columns else np.full(len(df), np.nan)
    low    = pd.to_numeric(df.get("low"),  errors="coerce").astype(float).values if "low"   in df.columns else np.full(len(df), np.nan)
    volume = pd.to_numeric(df.get("volume"), errors="coerce").astype(float).values if "volume" in df.columns else np.zeros(len(df))
    amount = pd.to_numeric(df.get("amount"), errors="coerce").astype(float).values if "amount" in df.columns else np.zeros(len(df))

    # basic returns (lagged)
    r = pd.Series(close).pct_change().fillna(0.0).values
    r_l1 = np.roll(r, 1); r_l1[0] = 0.0
    r_l2 = np.roll(r, 2); r_l2[:2] = 0.0
    r_l3 = np.roll(r, 3); r_l3[:3] = 0.0

    # rolling close-to-close vol
    def rolling_vol(arr, win):
        return pd.Series(arr).rolling(win, min_periods=max(5, win//2)).std(ddof=1).values
    vol5  = rolling_vol(r, 5)
    vol10 = rolling_vol(r, 10)
    vol20 = rolling_vol(r, 20)

    # momentum & z-score
    mom5  = pd.Series(close).pct_change(5).values
    mom10 = pd.Series(close).pct_change(10).values
    ma20  = pd.Series(close).rolling(20, min_periods=10).mean().values
    sd20  = pd.Series(close).rolling(20, min_periods=10).std(ddof=1).values
    z20   = (close - ma20) / (sd20 + EPS)

    # range-based estimators
    ln_hl = np.log((high + EPS) / (low + EPS))
    ln_co = np.log((close + EPS) / (open_ + EPS))
    par_var = ln_hl**2
    def roll_par_sigma(win):
        m = pd.Series(par_var).rolling(win, min_periods=max(5, win//2)).mean().values
        return np.sqrt(np.maximum(0.0, m / (4.0 * np.log(2.0))))
    par5  = roll_par_sigma(5)
    par10 = roll_par_sigma(10)
    par20 = roll_par_sigma(20)
    gk_var_daily = 0.5 * ln_hl**2 - (2.0*np.log(2.0) - 1.0) * (ln_co**2)
    gk_var_daily = np.where(np.isfinite(gk_var_daily), gk_var_daily, 0.0)
    def roll_gk_sigma(win):
        m = pd.Series(gk_var_daily).rolling(win, min_periods=max(5, win//2)).mean().values
        return np.sqrt(np.maximum(0.0, m))
    gk5  = roll_gk_sigma(5)
    gk10 = roll_gk_sigma(10)
    gk20 = roll_gk_sigma(20)

    # volume/amount features
    logv = np.log1p(np.maximum(0.0, volume))
    loga = np.log1p(np.maximum(0.0, amount))
    dv   = pd.Series(volume).pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    da   = pd.Series(amount).pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    amihud = np.abs(r) / (np.maximum(amount, 0.0)/1e6 + EPS)
    q99 = np.nanpercentile(amihud, 99) if np.isfinite(amihud).any() else 0.0
    amihud = np.clip(amihud, 0.0, q99 if (q99 and np.isfinite(q99)) else np.nanmax(amihud) if amihud.size else 0.0)

    # assemble features
    feats = np.column_stack([
        r_l1, r_l2, r_l3,
        vol5, vol10, vol20,
        mom5, mom10, z20,
        par5, par10, par20,
        gk5, gk10, gk20,
        logv, loga, dv, da, amihud
    ])
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    feat_cols = [f"f_{i}" for i in range(feats.shape[1])]

    # labels
    mminH = forward_path_min_return(close, H)
    retH  = forward_cum_return(close, H)

    out = pd.DataFrame({
        "date": df["__date__"].values,
        "close": close,
        "mminH": mminH,
        "retH":  retH
    })
    Xdf = pd.DataFrame(feats, columns=feat_cols)
    out = pd.concat([out, Xdf], axis=1)
    out = out.iloc[25:len(out)-H].reset_index(drop=True)
    return out, feat_cols

def build_labels_train_test(df_train: pd.DataFrame, df_test: pd.DataFrame,
                            loss_mode: str, theta_pct: float, alpha_q: float) -> Tuple[np.ndarray, np.ndarray, dict]:
    if loss_mode == "drawdown":
        thr = -abs(theta_pct)
        y_tr = (df_train["mminH"].values <= thr).astype(int)
        y_te = (df_test["mminH"].values  <= thr).astype(int)
        info = {"mode":"drawdown","theta_pct":theta_pct,"threshold":thr}
    elif loss_mode == "quantile":
        q = float(np.nanquantile(df_train["retH"].values, alpha_q))
        y_tr = (df_train["retH"].values <= q).astype(int)
        y_te = (df_test["retH"].values  <= q).astype(int)
        info = {"mode":"quantile","alpha_q":alpha_q,"threshold":q}
    else:
        raise ValueError("loss_mode must be 'drawdown' or 'quantile'")
    return y_tr, y_te, info

def symbol_from_path(path: str) -> str:
    base = os.path.basename(path)
    return base.split(".")[0]

def load_many_csvs(folder: str, date_col: str, price_col: str, min_rows: int) -> Tuple[pd.DataFrame, List[str]]:
    paths = sorted(glob.glob(os.path.join(folder, "*.csv")))
    frames, symbols = [], []
    for p in paths:
        try:
            sym = symbol_from_path(p)
            df = pd.read_csv(p)
            if date_col not in df.columns or price_col not in df.columns:
                continue
            df["__date__"] = pd.to_datetime(df[date_col], errors="coerce", infer_datetime_format=True)
            df = df.dropna(subset=["__date__", price_col]).sort_values("__date__").reset_index(drop=True)
            if len(df) < min_rows:
                continue
            df["symbol"] = sym
            frames.append(df)
            symbols.append(sym)
        except Exception:
            continue
    return (pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()), symbols

def add_breadth_features(panel_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    if panel_df.empty:
        return panel_df, []

    EPS = 1e-12
    df = panel_df.sort_values(["symbol","date"]).copy()

    # per-symbol rolling MA20 & daily return
    ma20 = df.groupby("symbol", observed=False)["close"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    ret_t = df.groupby("symbol", observed=False)["close"].transform(lambda s: s.pct_change().fillna(0.0))

    above_ma20 = (df["close"] >= (ma20 + EPS)).astype(float)

    # cross-sectional stats by DATE
    grp = df.groupby("date", observed=False)
    breadth = pd.DataFrame({
        "date": grp.size().index,
        "breadth_above20": above_ma20.groupby(df["date"]).mean().values,
        "breadth_adv_frac": (ret_t > 0).groupby(df["date"]).mean().values,
        "breadth_cs_std":   ret_t.groupby(df["date"]).std(ddof=1).values,
        "breadth_med_ret":  ret_t.groupby(df["date"]).median().values
    })

    out = panel_df.merge(breadth, on="date", how="left")
    new_cols = ["breadth_above20","breadth_adv_frac","breadth_cs_std","breadth_med_ret"]
    for c in new_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    return out, new_cols

def symbol_holdout_splits(symbols: List[str], n_splits: int, test_symbols_per_fold: int, seed: int) -> Iterable[Tuple[List[str], List[str]]]:
    rng = random.Random(seed)
    syms = symbols[:]
    rng.shuffle(syms)
    for i in range(n_splits):
        start = (i * test_symbols_per_fold) % len(syms)
        test = syms[start:start+test_symbols_per_fold]
        if len(test) < test_symbols_per_fold:
            test = test + syms[:(test_symbols_per_fold - len(test))]
        train = [s for s in syms if s not in test]
        yield train, test

def plot_symbol_prediction(df_sym: pd.DataFrame, out_png: str, title_extra=""):
    dates = df_sym["date"].values
    close = df_sym["close"].values
    y_true = df_sym["y_true"].values
    p_hat  = df_sym["p_hat"].values
    alert  = df_sym["alert"].values.astype(bool)

    fig = plt.figure(figsize=(11, 6.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.5, 1.2], hspace=0.15)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(dates, close, lw=1.2)
    ev_idx = np.where(y_true.astype(bool))[0]
    ax1.scatter(dates[ev_idx], close[ev_idx], color="crimson", s=18, label="loss events (gt)", zorder=5)
    al_idx = np.where(alert)[0]
    ax1.scatter(dates[al_idx], close[al_idx], marker="x", color="royalblue", s=36, label="predicted alerts", zorder=6)
    ax1.set_title(f"{df_sym['symbol'].iloc[0]} - events vs predicted alerts {title_extra}")
    ax1.set_ylabel("Close")
    ax1.legend(loc="upper left", ncol=2, frameon=False)

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.plot(dates, p_hat, lw=1.2)
    ax2.set_ylabel("Pred p(event)")
    ax2.set_xlabel("Date")
    for label in ax1.get_xticklabels():
        label.set_visible(False)

    plt.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

def train_logistic_isotonic(X_tr, y_tr):
    base = LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, n_jobs=1, random_state=1337
    )
    clf = CalibratedClassifierCV(base, method="isotonic", cv=5)
    clf.fit(X_tr, y_tr)
    return clf

def train_xgb(X_tr, y_tr, X_va=None, y_va=None, use_gpu=False, seed=1337):
    try:
        import xgboost as xgb
    except Exception as e:
        raise RuntimeError("xgboost not installed; use --model logistic or `pip install xgboost`") from e

    params = dict(
        n_estimators=1500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        reg_lambda=2.0,
        objective="binary:logistic",
        tree_method="gpu_hist" if use_gpu else "hist",
        random_state=seed,
        n_jobs=0,
    )
    clf = xgb.XGBClassifier(**params)

    if X_va is not None and y_va is not None and len(X_va) > 100:
        try:
            clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False, early_stopping_rounds=100)
        except TypeError:
            # Older/newer API variants may not accept early_stopping_rounds
            try:
                clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            except TypeError:
                clf.fit(X_tr, y_tr)
    else:
        clf.fit(X_tr, y_tr)
    return clf

def run_one_fold(panel_full: pd.DataFrame, feat_cols: List[str],
                 train_syms: List[str], test_syms: List[str],
                 loss_mode: str, theta_pct: float, alpha_q: float,
                 model_name: str, use_gpu: bool, val_frac: float,
                 alert_rate: float, outdir_fold: str) -> dict:

    os.makedirs(outdir_fold, exist_ok=True)

    df_tr = panel_full[panel_full["symbol"].isin(train_syms)].copy()
    df_te = panel_full[panel_full["symbol"].isin(test_syms)].copy()

    # label thresholds from TRAIN only
    y_tr_all, y_te, label_info = build_labels_train_test(df_tr, df_te, loss_mode, theta_pct, alpha_q)
    df_tr["y"] = y_tr_all
    df_te["y"] = y_te

    # chronological val split inside TRAIN for XGB early stopping
    df_tr = df_tr.sort_values(["date"]).reset_index(drop=True)
    n_tr = len(df_tr)
    cut  = int((1.0 - val_frac) * n_tr)
    df_fit = df_tr.iloc[:cut]
    df_val = df_tr.iloc[cut:] if val_frac > 0 and cut < n_tr else None

    scaler = StandardScaler()
    X_fit = scaler.fit_transform(df_fit[feat_cols].values)
    y_fit = df_fit["y"].values.astype(int)
    X_va  = scaler.transform(df_val[feat_cols].values) if df_val is not None else None
    y_va  = df_val["y"].values.astype(int) if df_val is not None else None
    X_te  = scaler.transform(df_te[feat_cols].values)
    y_te  = df_te["y"].values.astype(int)

    # train
    if model_name == "xgb":
        clf = train_xgb(X_fit, y_fit, X_va, y_va, use_gpu=use_gpu)
        p_te = clf.predict_proba(X_te)[:, 1]
    else:
        clf = train_logistic_isotonic(np.vstack([X_fit, X_va]) if X_va is not None else X_fit,
                                      np.concatenate([y_fit, y_va]) if y_va is not None else y_fit)
        p_te = clf.predict_proba(X_te)[:, 1]

    # metrics (overall across test symbols)
    pr = pr_auc(y_te, p_te)
    br = brier_score_loss(y_te, p_te)
    ece = ece_score(y_te, p_te, n_bins=10)
    prec, rec, realized, thr = precision_at_rate(y_te, p_te, rate=alert_rate)

    # save per-row predictions & per-symbol plots
    df_out = df_te[["symbol","date","close"]].copy()
    df_out["y_true"] = y_te
    df_out["p_hat"]  = p_te

    # build alerts
    k = max(1, int(alert_rate * len(p_te)))
    thr_fold = np.partition(p_te, -k)[-k] if k < len(p_te) else p_te.min() - 1e-9
    df_out["alert"] = (p_te >= thr_fold).astype(int)

    csv_path = os.path.join(outdir_fold, "predictions.csv")
    df_out.to_csv(csv_path, index=False)

    # plot a handful of test symbols
    for sym in sorted(test_syms)[:6]:
        df_sym = df_out[df_out["symbol"] == sym].sort_values("date")
        if len(df_sym) == 0:
            continue
        out_png = os.path.join(outdir_fold, f"plot_{sym}.png")
        plot_symbol_prediction(df_sym, out_png)

    metrics = dict(
        ok=True,
        n_test=int(len(y_te)),
        pos_test=int(y_te.sum()),
        pos_rate=float(y_te.mean()),
        pr_auc=float(pr),
        brier=float(br),
        ece10=float(ece),
        precision=float(prec),
        recall=float(rec),
        realized=float(realized),
        thr=float(thr),
        label_info=label_info,
        model=model_name
    )
    with open(os.path.join(outdir_fold, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Fold: PR-AUC={pr:.4f} | Brier={br:.4f} | ECE10={ece:.3f} | "
          f"P@{alert_rate:.1%}={prec:.3f} | R@{alert_rate:.1%}={rec:.3f} | Rate={realized:.3%}")
    return metrics

def main():
    ap = argparse.ArgumentParser(description="Method B: Cross-asset DG with range-vol, volume/amount, breadth")
    ap.add_argument("--data-folder", required=True, help="Folder of CSVs (one per asset).")
    ap.add_argument("--date-col", default="date")
    ap.add_argument("--price-col", default="close")
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--loss-mode", choices=["drawdown","quantile"], default="drawdown")
    ap.add_argument("--theta-pct", type=float, default=0.05)
    ap.add_argument("--alpha-q", type=float, default=0.05)
    ap.add_argument("--min-rows", type=int, default=400, help="Minimum raw rows per symbol to include.")
    ap.add_argument("--model", choices=["logistic","xgb"], default="xgb")
    ap.add_argument("--gpu", action="store_true", help="Use XGBoost GPU if --model xgb")
    ap.add_argument("--val-frac", type=float, default=0.1, help="Chronological validation fraction inside TRAIN (for XGB early stopping).")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--test-symbols-per-fold", type=int, default=2)
    ap.add_argument("--alert-rate", type=float, default=0.02)
    ap.add_argument("--outdir", default="outputs_b_enhanced")
    ap.add_argument("--cache-parquet", default="features_panel_enhanced.parquet", help="Parquet cache (created if missing).")
    ap.add_argument("--rebuild-cache", action="store_true", help="Force rebuild of cached panel.")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load or build feature panel
    if (not args.rebuild_cache) and os.path.exists(args.cache_parquet):
        panel = pd.read_parquet(args.cache_parquet)
        needed = {"breadth_above20","breadth_adv_frac","breadth_cs_std","breadth_med_ret"}
        if not needed.issubset(set(panel.columns)):
            panel, breadth_cols = add_breadth_features(panel)
            panel.to_parquet(args.cache_parquet, index=False)
        feat_cols = [c for c in panel.columns if c.startswith("f_")] + ["breadth_above20","breadth_adv_frac","breadth_cs_std","breadth_med_ret"]
        symbols = sorted(panel["symbol"].unique().tolist())
        print(f"Loaded cached panel: {args.cache_parquet} | rows={len(panel)} | symbols={len(symbols)}")
    else:
        raw, symbols = load_many_csvs(args.data_folder, args.date_col, args.price_col, args.min_rows)
        if raw.empty:
            raise RuntimeError("No usable CSVs found.")
        feat_frames = []
        last_feats = None
        for sym, df_sym in raw.groupby("symbol", observed=False):
            df_sym = df_sym.sort_values("__date__").reset_index(drop=True)
            fdf, feats = make_symbol_features(df_sym, args.price_col, H=args.horizon)
            fdf["symbol"] = sym
            feat_frames.append(fdf)
            last_feats = feats
        panel = pd.concat(feat_frames, axis=0, ignore_index=True)
        # add breadth computed from closes across symbols
        panel, breadth_cols = add_breadth_features(panel)
        feat_cols = last_feats + breadth_cols
        panel.to_parquet(args.cache_parquet, index=False)
        symbols = sorted(panel["symbol"].unique().tolist())
        print(f"Built feature panel | rows={len(panel)} | symbols={len(symbols)} | saved to {args.cache_parquet}")

    if len(symbols) < args.test_symbols_per_fold + 1:
        raise ValueError("Not enough symbols for the requested test set size.")

    # CV over symbol holdouts
    fold_metrics = []
    for i, (train_syms, test_syms) in enumerate(symbol_holdout_splits(symbols, args.n_splits, args.test_symbols_per_fold, args.seed), 1):
        print(f"\n=== Fold {i}/{args.n_splits} - TRAIN={len(train_syms)} | TEST={test_syms} ===")
        outdir_fold = os.path.join(args.outdir, f"fold_{i}")
        m = run_one_fold(panel, feat_cols, train_syms, test_syms,
                         args.loss_mode, args.theta_pct, args.alpha_q,
                         args.model, args.gpu, args.val_frac,
                         args.alert_rate, outdir_fold)
        fold_metrics.append(m)

    # aggregate
    if fold_metrics:
        dfm = pd.DataFrame(fold_metrics)
        mean_row = dfm[["pr_auc","brier","ece10","precision","recall","realized"]].mean().to_dict()
        print("\n=== Mean across folds ===")
        print("PR-AUC={pr_auc:.4f} | Brier={brier:.4f} | ECE10={ece10:.3f} | "
              "P@rate={precision:.3f} | R@rate={recall:.3f} | Rate={realized:.3%}".format(**mean_row))
        dfm.to_csv(os.path.join(args.outdir, "metrics_symbol_holdout_folds.csv"), index=False)
        with open(os.path.join(args.outdir, "metrics_mean.json"), "w", encoding="utf-8") as f:
            json.dump(mean_row, f, indent=2)

if __name__ == "__main__":
    main()