import os, argparse, json, math, warnings
from typing import List, Tuple
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True, linewidth=160)
pd.set_option("display.width", 160)

from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

def pr_auc(y, p):
    if len(y) == 0: return 0.0
    return float(average_precision_score(y, p))

def ece_score(y_true, p, n_bins=10):
    y = np.asarray(y_true, float)
    p = np.clip(np.asarray(p, float), 0, 1)
    m = len(p)
    if m == 0: return 0.0
    nb = int(max(1, min(n_bins, m)))
    edges = np.quantile(p, np.linspace(0, 1, nb + 1))
    edges[0], edges[-1] = 0.0, 1.0
    for i in range(1, len(edges)):
        if edges[i] <= edges[i-1]:
            edges[i] = min(1.0, edges[i-1] + 1e-12)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, nb - 1)
    N = float(m); ece = 0.0
    for b in range(nb):
        msk = (idx == b); w = int(msk.sum())
        if not w: continue
        ece += (w / N) * abs(p[msk].mean() - y[msk].mean())
    return float(ece)

def precision_at_rate(y_true, p, rate=0.02):
    y_true = np.asarray(y_true, int)
    p = np.asarray(p, float)
    n = len(p)
    if n == 0:
        return 0.0, 0.0, 0.0, 1.1
    k = max(1, int(rate * n))
    thr = np.partition(p, -k)[-k] if k < n else p.min() - 1e-12
    hits = (p >= thr).astype(int)
    tp = int((hits * y_true).sum())
    prec = tp / max(1, hits.sum())
    rec  = tp / max(1, int(y_true.sum()))
    realized = hits.mean()
    return float(prec), float(rec), float(realized), float(thr)

def _pct_change(a, k=1):
    s = pd.Series(a, dtype=float)
    return s.pct_change(k).fillna(0.0).values

def _rolling_std(a, win):
    s = pd.Series(a, dtype=float)
    return s.rolling(win, min_periods=max(5, win // 2)).std(ddof=1).values

def _zscore(a, win=20):
    s = pd.Series(a, dtype=float)
    ma = s.rolling(win, min_periods=10).mean()
    sd = s.rolling(win, min_periods=10).std(ddof=1)
    return ((s - ma) / (sd + 1e-12)).values

def forward_mmin_close(close, H):
    c = close.astype(float); n = len(c)
    R = np.ones(n); R[1:] = c[1:] / np.maximum(c[:-1], 1e-12)
    out = np.full(n, np.nan, float)
    for t in range(n - H):
        out[t] = np.cumprod(R[t+1:t+1+H]).min() - 1.0
    return out

def forward_ret_close(close, H):
    c = close.astype(float); n = len(c)
    out = np.full(n, np.nan, float)
    for t in range(n - H):
        out[t] = (c[t+H] / c[t]) - 1.0 if c[t] > 0 else np.nan
    return out

def build_panel_from_csvs(folder: str, horizon: int) -> pd.DataFrame:
    rows = []
    for fn in os.listdir(folder):
        if not fn.lower().endswith(".csv"):
            continue
        sym = fn.split(".")[0]
        df = pd.read_csv(os.path.join(folder, fn))
        if "date" not in df.columns or "close" not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").dropna(subset=["date", "close"]).reset_index(drop=True)
        close = df["close"].astype(float).values

        ret1 = _pct_change(close, 1); ret2 = _pct_change(close, 2); ret3 = _pct_change(close, 3)
        vol5 = _rolling_std(_pct_change(close, 1), 5)
        vol10 = _rolling_std(_pct_change(close, 1), 10)
        vol20 = _rolling_std(_pct_change(close, 1), 20)
        mom5 = _pct_change(close, 5); mom10 = _pct_change(close, 10)
        z20 = _zscore(close, 20)

        mminH = forward_mmin_close(close, horizon)
        retH  = forward_ret_close(close, horizon)

        out = pd.DataFrame({
            "symbol": sym, "date": df["date"].values, "close": close,
            "f_ret1": ret1, "f_ret2": ret2, "f_ret3": ret3,
            "f_vol5": vol5, "f_vol10": vol10, "f_vol20": vol20,
            "f_mom5": mom5, "f_mom10": mom10, "f_z20": z20,
            "mminH": mminH, "retH": retH
        })
        # drop warmup and forward tail to avoid leakage
        out = out.iloc[25:len(out) - horizon]
        rows.append(out)

    if not rows:
        raise RuntimeError("No valid CSVs found with columns ['date','close'] in the data folder.")
    panel = pd.concat(rows, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    return panel

def make_labels_train_test(panel, train_symbols, loss_mode, theta_pct, alpha_q):
    df_tr = panel[panel["symbol"].isin(train_symbols)].copy()
    if loss_mode == "drawdown":
        thr = -abs(theta_pct)
        panel["y"] = (panel["mminH"] <= thr).astype(int)
        lbl_info = {"mode": "drawdown", "theta_pct": theta_pct, "threshold": thr}
    elif loss_mode == "quantile":
        q = float(np.nanquantile(df_tr["retH"].values, alpha_q))
        panel["y"] = (panel["retH"] <= q).astype(int)
        lbl_info = {"mode": "quantile", "alpha_q": alpha_q, "threshold": q}
    else:
        raise ValueError("loss_mode must be 'drawdown' or 'quantile'")
    return panel, lbl_info

def _ensure_list(x):
    if isinstance(x, list):
        return x
    return [c.strip() for c in str(x).split(",") if c.strip()]

def symbol_holdout_splits(symbols: List[str], n_splits: int, test_k: int, seed: int):
    rng = np.random.default_rng(seed)
    syms = np.array(sorted(symbols))
    rng.shuffle(syms)
    folds = []
    for i in range(n_splits):
        lo = (i * test_k) % len(syms)
        hi = min(lo + test_k, len(syms))
        test = syms[lo:hi]
        if len(test) < test_k:
            test = syms[-test_k:]
        train = np.array([s for s in syms if s not in test])
        folds.append((train.tolist(), test.tolist()))
    return folds

from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

def fit_predict_logistic(X_tr, y_tr, X_te):
    base = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000)
    clf = CalibratedClassifierCV(base, method="isotonic", cv=5)
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_te)[:, 1]

def fit_predict_xgb(X_tr, y_tr, X_va, y_va, X_te, gpu=False, scale_pos_weight=None, seed=1337):
    params = dict(
        n_estimators=2000, max_depth=5, subsample=0.8, colsample_bytree=0.8,
        learning_rate=0.03, objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", random_state=seed
    )
    if gpu:
        params.update(dict(tree_method="gpu_hist", predictor="gpu_predictor"))
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = float(scale_pos_weight)

    clf = xgb.XGBClassifier(**params)

    # early-stopping compatibility across xgboost versions
    try:
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False, early_stopping_rounds=50)
    except TypeError:
        try:
            cb = getattr(xgb.callback, "EarlyStopping", None)
            if cb is None:
                raise TypeError("callbacks not available")
            es = cb(rounds=50, save_best=True, maximize=False)
            clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[es])
        except Exception:
            clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    return clf.predict_proba(X_te)[:, 1], clf

def fit_anchor_regression_LPM(X, y, Z, lam=1e-3, gamma=2.0):
    X = np.asarray(X, float); y = np.asarray(y, float); Z = np.asarray(Z, float)
    p = X.shape[1]; m = Z.shape[1]
    XtX = X.T @ X; XtZ = X.T @ Z; ZtZ = Z.T @ Z
    XtY = X.T @ y; ZtY = Z.T @ y
    A11 = XtX + lam*np.eye(p)
    A22 = (1.0 + gamma) * ZtZ
    A = np.block([[A11, XtZ], [XtZ.T, A22]])
    b = np.concatenate([XtY, ZtY], axis=0)
    sol = np.linalg.solve(A + 1e-8*np.eye(A.shape[0]), b)
    beta = sol[:p]; alpha = sol[p:]
    return beta, alpha

def run_fold(panel, anchors_tbl, train_syms, test_syms, args):
    panel, lbl = make_labels_train_test(panel.copy(), train_syms, args.loss_mode, args.theta_pct, args.alpha_q)

    # merge anchors and robustly fill
    anc_list = _ensure_list(args.anchor_cols)
    df = panel.merge(anchors_tbl, on="date", how="left")

    present = [c for c in anc_list if c in df.columns]
    if len(present):
        # forward-fill then back-fill to avoid pre-anchor gaps
        df[present] = df[present].ffill().bfill()
        still_nan = [c for c in present if df[c].isna().all()]
        if still_nan:
            print(f"[WARN] Dropping all-NaN anchors after fill: {still_nan}")
            present = [c for c in present if c not in still_nan]
    else:
        present = []

    # optional short lags for anchors
    if args.anchor_lags > 0 and len(present):
        for c in present:
            for L in range(1, args.anchor_lags + 1):
                df[f"{c}_L{L}"] = df[c].shift(L)

    # masks
    tr_mask = df["symbol"].isin(train_syms)
    te_mask = df["symbol"].isin(test_syms)

    # feature columns
    base_cols = [c for c in df.columns if c.startswith("f_")]
    anc_cols = present[:]
    if args.anchor_lags > 0 and len(present):
        anc_cols += [f"{c}_L{L}" for c in present for L in range(1, args.anchor_lags + 1)]

    if args.mode in ("xgb_anchors", "logit_anchors", "anchor_reg"):
        feat_cols = base_cols + anc_cols
    else:
        feat_cols = base_cols

    # standardize features on TRAIN only
    X_all = df[feat_cols].to_numpy(dtype=float)
    X_all = np.nan_to_num(X_all, 0.0, 0.0, 0.0)
    scalerX = StandardScaler()
    X_all[tr_mask, :] = scalerX.fit_transform(X_all[tr_mask, :])
    X_all[~tr_mask, :] = scalerX.transform(X_all[~tr_mask, :])

    # chronological split inside TRAIN for early stopping/calibration
    df_train = df[tr_mask & (~te_mask)].sort_values("date")
    cut = int(0.85 * len(df_train))
    idx_tr = df_train.index[:cut]
    idx_va = df_train.index[cut:]

    X_tr = X_all[idx_tr]; y_tr = df.loc[idx_tr, "y"].to_numpy(int)
    X_va = X_all[idx_va]; y_va = df.loc[idx_va, "y"].to_numpy(int)
    X_te = X_all[df[te_mask].index]; y_te = df.loc[te_mask, "y"].to_numpy(int)

    # build anchor composite & bins
    Z_cols_for_bins = anc_cols if len(anc_cols) else []
    if len(Z_cols_for_bins) >= 1 and len(idx_tr) >= 50:
        Z_tr = df.loc[idx_tr, Z_cols_for_bins].to_numpy(float)
        col_means = np.nanmean(Z_tr, axis=0)
        inds = np.where(np.isnan(Z_tr))
        if inds[0].size:
            Z_tr[inds] = np.take(col_means, inds[1])

        Z_tr_mu, Z_tr_sd = Z_tr.mean(axis=0), Z_tr.std(axis=0) + 1e-12
        Z_tr_std = (Z_tr - Z_tr_mu) / Z_tr_sd
        comp_tr = Z_tr_std.sum(axis=1)
        valid_tr = np.isfinite(comp_tr)

        if valid_tr.sum() >= max(30, 5 * args.anchor_bins):
            qs = np.quantile(comp_tr[valid_tr], np.linspace(0, 1, args.anchor_bins + 1))
            for i in range(1, len(qs)):
                if qs[i] <= qs[i-1]: qs[i] = qs[i-1] + 1e-9

            Z_te = df.loc[te_mask, Z_cols_for_bins].to_numpy(float)
            inds_te = np.where(np.isnan(Z_te))
            if inds_te[0].size:
                Z_te[inds_te] = np.take(col_means, inds_te[1])
            Z_te_std = (Z_te - Z_tr_mu) / Z_tr_sd
            comp_te = Z_te_std.sum(axis=1)
            bins_te = np.clip(np.digitize(comp_te, qs[1:-1], right=True), 0, len(qs) - 2)
        else:
            bins_te = np.zeros(int(te_mask.sum()), dtype=int)
    else:
        bins_te = np.zeros(int(te_mask.sum()), dtype=int)

    if args.mode == "xgb_anchors":
        if not HAS_XGB:
            raise RuntimeError("xgboost not available in this environment.")
        pos = y_tr.sum(); neg = len(y_tr) - pos
        spw = (neg / max(1, pos)) if pos > 0 else 1.0
        p_te_raw, clf = fit_predict_xgb(X_tr, y_tr, X_va, y_va, X_te, gpu=args.gpu, scale_pos_weight=spw, seed=args.seed)

        # isotonic calibration on validation slice
        from sklearn.isotonic import IsotonicRegression
        p_va = clf.predict_proba(X_va)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_va, y_va)
        p_te = iso.transform(p_te_raw)

        model_name = "XGB (anchors-as-features, isotonic)"

    elif args.mode == "logit_anchors":
        p_te = fit_predict_logistic(np.vstack([X_tr, X_va]), np.hstack([y_tr, y_va]), X_te)
        model_name = "Logistic+Isotonic (anchors-as-features)"

    elif args.mode == "anchor_reg":
        # separate X vs Z for penalty
        Xcols = base_cols
        Zcols = anc_cols
        X_tr2 = df.loc[idx_tr, Xcols].to_numpy(float)
        X_va2 = df.loc[idx_va, Xcols].to_numpy(float)
        X_te2 = df.loc[te_mask, Xcols].to_numpy(float)

        Z_tr2 = df.loc[idx_tr, Zcols].to_numpy(float) if len(Zcols) else np.zeros((len(idx_tr), 0))
        Z_va2 = df.loc[idx_va, Zcols].to_numpy(float) if len(Zcols) else np.zeros((len(idx_va), 0))
        Z_te2 = df.loc[te_mask, Zcols].to_numpy(float) if len(Zcols) else np.zeros((int(te_mask.sum()), 0))

        sx = StandardScaler(); sz = StandardScaler() if Z_tr2.shape[1] > 0 else None
        X_tr2 = sx.fit_transform(np.nan_to_num(X_tr2, 0.0, 0.0, 0.0))
        X_va2 = sx.transform(np.nan_to_num(X_va2, 0.0, 0.0, 0.0))
        X_te2 = sx.transform(np.nan_to_num(X_te2, 0.0, 0.0, 0.0))
        if Z_tr2.shape[1] > 0:
            Z_tr2 = sz.fit_transform(np.nan_to_num(Z_tr2, 0.0, 0.0, 0.0))
            Z_va2 = sz.transform(np.nan_to_num(Z_va2, 0.0, 0.0, 0.0))
            Z_te2 = sz.transform(np.nan_to_num(Z_te2, 0.0, 0.0, 0.0))

        beta, alpha = fit_anchor_regression_LPM(X_tr2, y_tr, Z_tr2, lam=args.l2, gamma=args.gamma)
        s_va = X_va2 @ beta + (Z_va2 @ alpha if Z_va2.shape[1] else 0.0)
        s_te = X_te2 @ beta + (Z_te2 @ alpha if Z_te2.shape[1] else 0.0)

        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(s_va, y_va)
        p_te = iso.transform(s_te)

        model_name = f"Anchor Regression (gamma={args.gamma:.2f}, lambda={args.l2:.1e})"

    else:
        raise ValueError("mode must be xgb_anchors | logit_anchors | anchor_reg")

    pr = pr_auc(y_te, p_te)
    br = brier_score_loss(y_te, p_te)
    ece = ece_score(y_te, p_te, n_bins=10)
    P, R, realized, thr = precision_at_rate(y_te, p_te, rate=args.alert_rate)

    # per-anchor-bin robustness
    uniq = np.unique(bins_te)
    per_bin = []
    worst_prec = float("nan")
    for b in uniq:
        m = (bins_te == b)
        n_b = int(m.sum())
        if n_b < 20:
            per_bin.append(dict(bin=int(b), n=n_b, pr_auc=np.nan, prec=np.nan, rec=np.nan))
            continue
        pr_b = pr_auc(y_te[m], p_te[m])
        P_b, R_b, _, _ = precision_at_rate(y_te[m], p_te[m], rate=args.alert_rate)
        per_bin.append(dict(bin=int(b), n=n_b, pr_auc=float(pr_b), prec=float(P_b), rec=float(R_b)))
        worst_prec = P_b if math.isnan(worst_prec) else min(worst_prec, P_b)

    return {
        "ok": True,
        "model": model_name,
        "lbl": lbl,
        "pr_auc": pr, "brier": br, "ece10": ece,
        "prec_at_rate": P, "recall_at_rate": R, "rate": realized,
        "thr": thr,
        "per_anchor_bin": per_bin,
        "worst_bin_precision": worst_prec
    }

def main():
    ap = argparse.ArgumentParser(description="Method C: pooled DG + anchors")
    ap.add_argument("--data-folder", required=True, help="Folder with per-symbol CSVs (needs date,close columns)")
    ap.add_argument("--anchors-csv", required=True, help="Daily anchors.csv (from build_anchors.py)")
    ap.add_argument("--anchor-cols", required=True, help="Comma-separated anchor columns present in anchors.csv")
    ap.add_argument("--anchor-lag", type=int, default=0, help="Extra day lag to apply to anchors (0 if already lagged)")
    ap.add_argument("--anchor-lags", type=int, default=0, help="Create L1..Lk lag columns for anchors")
    ap.add_argument("--anchor-bins", type=int, default=3, help="Bins for robustness reporting (default 3)")

    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--loss-mode", choices=["drawdown","quantile"], default="drawdown")
    ap.add_argument("--theta-pct", type=float, default=0.05)
    ap.add_argument("--alpha-q", type=float, default=0.05)

    ap.add_argument("--mode", choices=["xgb_anchors","logit_anchors","anchor_reg"], default="xgb_anchors")
    ap.add_argument("--l2", type=float, default=1e-3, help="L2 for anchor regression")
    ap.add_argument("--gamma", type=float, default=2.0, help="Anchor penalty gamma (robustness strength)")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--test-symbols-per-fold", type=int, default=2)
    ap.add_argument("--alert-rate", type=float, default=0.05)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--cache-parquet", default="", help="Optional Parquet cache for panel features")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--outdir", default="outputs_c")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # normalize anchor cols
    args.anchor_cols = _ensure_list(args.anchor_cols)

    # build/load panel
    if args.cache_parquet and os.path.exists(args.cache_parquet) and not args.rebuild_cache:
        panel = pd.read_parquet(args.cache_parquet)
    else:
        panel = build_panel_from_csvs(args.data_folder, args.horizon)
        if args.cache_parquet:
            panel.to_parquet(args.cache_parquet, index=False)

    # load anchors, keep requested cols, optional extra lag, fill
    anchors_tbl_raw = pd.read_csv(args.anchors_csv)
    anchors_tbl_raw["date"] = pd.to_datetime(anchors_tbl_raw["date"], errors="coerce")

    keep_cols = ["date"] + [c for c in args.anchor_cols if c in anchors_tbl_raw.columns]
    missing = [c for c in args.anchor_cols if c not in anchors_tbl_raw.columns]
    if missing:
        print(f"[WARN] Missing anchors in file (skipping): {missing}")

    anchors_tbl = anchors_tbl_raw[keep_cols].sort_values("date").reset_index(drop=True)
    if len(keep_cols) > 1:
        anchors_tbl[keep_cols[1:]] = anchors_tbl[keep_cols[1:]].ffill().bfill()
        if args.anchor_lag > 0:
            for c in keep_cols[1:]:
                anchors_tbl[c] = anchors_tbl[c].shift(args.anchor_lag)

    # CV folds
    symbols = sorted(panel["symbol"].unique().tolist())
    folds = symbol_holdout_splits(symbols, n_splits=args.n_splits, test_k=args.test_symbols_per_fold, seed=args.seed)

    # run folds
    all_rows = []
    for i, (tr, te) in enumerate(folds, 1):
        print(f"\n=== Fold {i}/{args.n_splits} - TRAIN={len(tr)} | TEST={te} ===")
        out = run_fold(panel, anchors_tbl, tr, te, args)
        if out["ok"]:
            print(f"Fold: {out['model']} | label={out['lbl']['mode']} | "
                  f"PR-AUC={out['pr_auc']:.4f} | Brier={out['brier']:.4f} | ECE10={out['ece10']:.3f} | "
                  f"P@{args.alert_rate:.1%}={out['prec_at_rate']:.3f} | R@{args.alert_rate:.1%}={out['recall_at_rate']:.3f} | "
                  f"Rate={out['rate']:.3%} | worst-bin P={out['worst_bin_precision']:.3f}")
            row = dict(
                fold=i, model=out["model"], lbl_mode=out["lbl"]["mode"],
                pr_auc=out["pr_auc"], brier=out["brier"], ece10=out["ece10"],
                prec_at_rate=out["prec_at_rate"], recall_at_rate=out["recall_at_rate"],
                rate=out["rate"], worst_bin_precision=out["worst_bin_precision"]
            )
            all_rows.append(row)
            with open(os.path.join(args.outdir, f"fold{i}_per_anchor_bins.json"), "w", encoding="utf-8") as f:
                json.dump(out["per_anchor_bin"], f, indent=2)

    if all_rows:
        dfm = pd.DataFrame(all_rows)
        dfm.to_csv(os.path.join(args.outdir, "metrics_method_c.csv"), index=False)
        print("\n=== Mean across folds ===")
        print(dfm[["pr_auc","brier","ece10","prec_at_rate","recall_at_rate","rate","worst_bin_precision"]]
              .mean().to_string(float_format=lambda x: f"{x:.4f}"))
        with open(os.path.join(args.outdir, "metrics_method_c_mean.json"), "w", encoding="utf-8") as f:
            json.dump(dfm.mean(numeric_only=True).to_dict(), f, indent=2)

if __name__ == "__main__":
    main()