"""ML 信号提纯训练 v2: 因子矩阵验证(方向×标签×数据分层) + 置换重要性

用法: docker compose run --rm --entrypoint python freqtrade user_data/train_ml.py
"""
import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

BASE = [
    "chan_bsp_div", "chan_bsp_amp", "chan_stop_amp", "chan_bi_cnt",
    "chan_seg_dir", "chan_segseg_dir", "chan_ma_trend",
    "smc_sweep_bull_recent", "smc_sweep_bear_recent",
    "smc_fvg_bull", "smc_fvg_bear", "smc_killzone", "smc_btc_trend",
    "risk_dist", "hour", "is_major",
]
DERIVED = [
    "candle_body", "wick_up", "wick_dn", "close_pos",
    "ret1", "ret3", "ret6", "ret24",
    "atr20", "atr_ratio", "boll_bw", "pct_b", "pct_rank100",
    "rsi14", "macd_dif", "macd_hist", "macd_hist_slope",
    "close_ma5", "close_ma10", "close_ma20", "close_ma60",
    "ma5_slope", "ma10_slope", "ma20_slope", "ma60_slope",
    "zs_low_dist", "zs_high_dist", "rvol", "vol_trend",
    "btc_ret24", "btc_atr",
    # Alpha158 实验批次已回滚(AUC 0.655->0.570, 维度灾难)
]
FEATS = BASE + DERIVED
SPLIT = "2025-11-01"


def auc_of(model, X, y):
    return roc_auc_score(y, model.predict_proba(X)[:, 1])


def run(name, tr, te, label):
    Xtr, ytr = tr[FEATS], tr[label]
    Xte, yte = te[FEATS], te[label]
    if yte.nunique() < 2 or len(te) < 50:
        return None
    m = HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.05, max_depth=4,
        min_samples_leaf=25, l2_regularization=1.0, random_state=42,
    ).fit(Xtr, ytr)
    auc = auc_of(m, Xte, yte)
    base = yte.mean()
    proba = m.predict_proba(Xte)[:, 1]
    best = max(
        ((t, (yte[proba >= t].mean(), (proba >= t).mean()))
         for t in (0.5, 0.55, 0.6, 0.65) if (proba >= t).sum() >= 15),
        key=lambda x: x[1][0], default=None,
    )
    tip = f"best_thr={best[0]:.2f} wr={best[1][0]:.2f} keep={best[1][1]:.2f}" if best else ""
    print(f"{name:38s} AUC={auc:.3f} base={base:.2f} {tip}")
    return m, auc, Xte, yte


def main():
    df = pd.read_csv("/freqtrade/user_data/ml_signals.csv")
    df["date"] = pd.to_datetime(df["date"])
    df["chan_bsp_div"] = df["chan_bsp_div"].fillna(-1)
    df = df.fillna(0)
    for c in FEATS:            # 防御: 个别批次缺失的列补 0
        if c not in df.columns:
            df[c] = 0
    train, test = df[df["date"] < SPLIT], df[df["date"] >= SPLIT]
    print(f"total={len(df)} train={len(train)} test={len(test)} feats={len(FEATS)}")
    best_artifact = None
    for side in ("long", "short"):
        for sub_name, mask_tr, mask_te in (
            ("all", train.side == side, test.side == side),
            ("majors", (train.side == side) & (train.is_major == 1),
             (test.side == side) & (test.is_major == 1)),
        ):
            tr, te = train[mask_tr], test[mask_te]
            for label in ("label_fixed", "label_atr"):
                res = run(f"{side}/{sub_name}/{label}", tr, te, label)
                if res and (best_artifact is None or res[1] > best_artifact[0]):
                    best_artifact = (res[1], side, sub_name, label, *res)
    if best_artifact:
        auc, side, sub_name, label, m, _, Xte, yte = best_artifact
        print(f"\n>>> 最优: {side}/{sub_name}/{label} AUC={auc:.3f}")
        pi = permutation_importance(m, Xte, yte, n_repeats=5, random_state=42)
        top = sorted(zip(FEATS, pi.importances_mean), key=lambda x: -x[1])[:10]
        print("置换重要性Top10:")
        for k, v in top:
            print(f"  {k:24s} {v:+.4f}")
        joblib.dump(
            {"model": m, "feats": FEATS, "label": label, "side": side,
             "subset": sub_name, "split": SPLIT},
            f"/freqtrade/user_data/ml_best_{side}.joblib",
        )
        print(f"saved ml_best_{side}.joblib")


if __name__ == "__main__":
    main()
