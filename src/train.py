"""Train the final model, evaluate it on the time holdout in rupees, score the test file.

Run: python -m src.train
Outputs:
  model/model.joblib          fitted pipeline + metadata used by the service
  predictions.csv             order_id, score  (score = calibrated return probability)
  reports/evaluation.json     holdout metrics, calibration, rupee tables
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.experiment import SPLIT, make_lr
from src.features import DATA_DIR, EXCLUDED, build_features, clean_orders
from src.policy import COSTS, decide, rupee_table

ROOT = Path(__file__).resolve().parent.parent


def bootstrap_auc(y, p, n=500, seed=0):
    rs = np.random.RandomState(seed)
    out = []
    for _ in range(n):
        i = rs.randint(0, len(y), len(y))
        if y[i].min() != y[i].max():
            out.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    raw = pd.read_csv(DATA_DIR / "train.csv", dtype={"customer_id": str})
    df = clean_orders(raw)
    tr, va = df[df.order_placed_at < SPLIT], df[df.order_placed_at >= SPLIT]

    # ---- holdout evaluation (model trained only on the past) ----
    m_hold = make_lr().fit(build_features(tr), tr.returned.values)
    Xva = build_features(va)
    p = m_hold.predict_proba(Xva)[:, 1]
    y = va.returned.values
    lo, hi = bootstrap_auc(y, p)

    deciles = pd.qcut(p, 10, labels=False, duplicates="drop")
    calib = (pd.DataFrame({"d": deciles, "p": p, "y": y}).groupby("d")
             .agg(n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean")).reset_index())

    shield = (Xva.shield_member.values == "Y")
    ev = {
        "holdout_period": f"{SPLIT} .. 2026-06-30",
        "n_holdout": int(len(y)), "base_rate": float(y.mean()),
        "auc": float(roc_auc_score(y, p)), "auc_95ci": [lo, hi],
        "pr_auc": float(average_precision_score(y, p)),
        "accuracy_if_flag_nobody": float((y == 0).mean()),
        "accuracy_at_0.5": float(((p >= 0.5) == y).mean()),
        "max_accuracy_any_threshold": float(max(((p >= t) == y).mean() for t in np.linspace(0, 1, 201))),
        "calibration_deciles": calib.to_dict(orient="records"),
        "auc_shield": float(roc_auc_score(y[shield], p[shield])),
        "auc_non_shield": float(roc_auc_score(y[~shield], p[~shield])),
        "rupees": rupee_table(y, p, Xva.shield_member.values, Xva.order_value_inr.values),
        "costs_used": COSTS,
    }

    # Error analysis: the confident misses on the holdout
    va2 = va.assign(score=p)
    ev["worst_false_negatives"] = (va2[va2.returned == 1].nsmallest(5, "score")
                                   [["order_id", "sku", "payment_mode", "customer_prior_returns", "score"]]
                                   .to_dict(orient="records"))
    ev["worst_false_positives"] = (va2[va2.returned == 0].nlargest(5, "score")
                                   [["order_id", "sku", "payment_mode", "customer_prior_returns", "score"]]
                                   .to_dict(orient="records"))

    # ---- final model on all history ----
    model = make_lr().fit(build_features(df), df.returned.values)
    test = pd.read_csv(DATA_DIR / "test_unlabelled.csv", dtype={"customer_id": str})
    sub = pd.read_csv(DATA_DIR / "sample_submission.csv")
    pt = model.predict_proba(build_features(test))[:, 1]
    pred = pd.DataFrame({"order_id": test.order_id, "score": np.round(pt, 6)})
    pred = sub[["order_id"]].merge(pred, on="order_id", how="left")
    assert pred.score.notna().all() and len(pred) == len(sub)
    pred.to_csv(ROOT / "predictions.csv", index=False)

    actions = [decide(s, sh)["action"] for s, sh in zip(pt, build_features(test).shield_member)]
    ev["test_scoring"] = {
        "n": int(len(pt)), "mean_score": float(pt.mean()),
        "expected_returns": float(pt.sum()),
        "actions": pd.Series(actions).value_counts().to_dict(),
    }

    (ROOT / "model").mkdir(exist_ok=True)
    Zall = model.named_steps["pre"].transform(build_features(df))
    ref = np.asarray(Zall.mean(axis=0)).ravel()
    joblib.dump({"pipeline": model, "ref": ref.tolist(), "base_rate": float(df.returned.mean()), "excluded": EXCLUDED,
                 "trained_on": f"{df.order_placed_at.min()} .. {df.order_placed_at.max()}",
                 "holdout_auc": ev["auc"]}, ROOT / "model" / "model.joblib")
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "evaluation.json").write_text(json.dumps(ev, indent=2, default=str))
    print(json.dumps({k: v for k, v in ev.items() if k not in ("calibration_deciles",)}, indent=1, default=str))
    print(calib.to_string(index=False))


if __name__ == "__main__":
    main()
