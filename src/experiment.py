"""Model comparison on a time-based holdout (train Apr-25..Mar-26, validate Apr-26..Jun-26).

Writes reports/experiments.json and prints a table. Run: python -m src.experiment
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.features import CATEGORICAL, DATA_DIR, NUMERIC, build_features, clean_orders

ROOT = Path(__file__).resolve().parent.parent
SPLIT = "2026-04-01"


def make_lr():
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CATEGORICAL),
        ("num", Pipeline([("sc", StandardScaler())]), NUMERIC),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(C=0.3, max_iter=2000))])


def make_hgb(cats=CATEGORICAL, nums=NUMERIC):
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cats),
        ("num", "passthrough", nums),
    ])
    n_cat = len(cats)
    clf = HistGradientBoostingClassifier(
        learning_rate=0.04, max_iter=300, max_leaf_nodes=15, min_samples_leaf=60,
        l2_regularization=1.0, categorical_features=list(range(n_cat)), random_state=0,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def top_k_stats(y, p, frac):
    k = max(1, int(round(len(p) * frac)))
    idx = np.argsort(-p)[:k]
    return {"precision": float(y[idx].mean()), "recall": float(y[idx].sum() / max(1, y.sum())), "n": k}


def metrics(y, p):
    out = {
        "auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "accuracy_at_0.5": float(((p >= 0.5).astype(int) == y).mean()),
        "flagged_at_0.5": int((p >= 0.5).sum()),
    }
    for f in (0.05, 0.10, 0.20):
        out[f"top{int(f*100)}"] = top_k_stats(y, p, f)
    return out


def main():
    raw = pd.read_csv(DATA_DIR / "train.csv", dtype={"customer_id": str})
    df = clean_orders(raw)
    tr = df[df.order_placed_at < SPLIT]
    va = df[df.order_placed_at >= SPLIT]
    Xtr, Xva = build_features(tr), build_features(va)
    ytr, yva = tr.returned.values, va.returned.values
    results = {"n_train": len(tr), "n_valid": len(va), "valid_base_rate": float(yva.mean())}

    results["all_not_returned"] = {"accuracy": float((yva == 0).mean())}
    # simple rule baseline: prior returns OR cod
    rule = (va.customer_prior_returns.values > 0).astype(float) + (va.payment_mode.values == "cod") * 0.5
    results["rule_prior_returns_cod"] = metrics(yva, (rule + 1e-6 * np.random.RandomState(0).rand(len(rule))) / 1.6)

    lr = make_lr().fit(Xtr, ytr)
    results["logistic"] = metrics(yva, lr.predict_proba(Xva)[:, 1])
    hgb = make_hgb().fit(Xtr, ytr)
    p_hgb = hgb.predict_proba(Xva)[:, 1]
    results["hist_gbm"] = metrics(yva, p_hgb)
    results["blend"] = metrics(yva, 0.5 * p_hgb + 0.5 * lr.predict_proba(Xva)[:, 1])

    # The leaky model, to show what we threw away and why the "accuracy" looks great
    def leaky(X, d):
        X = X.copy()
        X["svc"] = d.last_service_event_type.fillna("NONE").values
        X["has_pickup"] = d.pickup_scheduled_at.notna().astype(int).values
        return X
    lk = make_hgb(CATEGORICAL + ["svc"], NUMERIC + ["has_pickup"]).fit(leaky(Xtr, tr), ytr)
    p_lk = lk.predict_proba(leaky(Xva, va))[:, 1]
    results["LEAKY_with_service_pickup"] = metrics(yva, p_lk)
    # ...and what that same model does on a dispatch-time snapshot (columns blank/NONE, like test)
    snap = va.copy(); snap["last_service_event_type"] = "NONE"; snap["pickup_scheduled_at"] = np.nan
    results["LEAKY_on_dispatch_snapshot"] = metrics(yva, lk.predict_proba(leaky(Xva, snap))[:, 1])

    # Segment AUC for the chosen model
    seg = {}
    for col in ["shield_member", "sales_channel", "family"]:
        for val, idx in Xva.groupby(col).groups.items():
            m = Xva.index.get_indexer(idx)
            if yva[m].sum() >= 10 and (1 - yva[m]).sum() >= 10:
                seg[f"{col}={val}"] = {"n": int(len(m)), "rate": float(yva[m].mean()), "auc": float(roc_auc_score(yva[m], p_hgb[m]))}
    results["segments_hist_gbm"] = seg

    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "experiments.json").write_text(json.dumps(results, indent=2))
    for k, v in results.items():
        if isinstance(v, dict) and "auc" in v:
            print(f"{k:32s} AUC {v['auc']:.3f}  PR-AUC {v['pr_auc']:.3f}  acc@0.5 {v['accuracy_at_0.5']:.3f}  "
                  f"top10 prec {v['top10']['precision']:.2f} rec {v['top10']['recall']:.2f}")
    print("all-not-returned accuracy", results["all_not_returned"])
    print(json.dumps(seg, indent=1))


if __name__ == "__main__":
    main()
