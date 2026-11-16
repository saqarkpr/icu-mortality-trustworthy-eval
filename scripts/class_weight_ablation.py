"""Diagnostic ablation, run after noticing Logistic Regression's OOF ECE
was much worse than Gradient-Boosted Trees' despite similar AUROC, and
suspecting `class_weight="balanced"` as the cause (rebalancing shifts
predicted probabilities away from the true base rate in exchange for a
better decision boundary -- it does not know about calibration at all).
Tests that suspicion directly instead of asserting it, for BOTH models:
GBT is trained with balanced `sample_weight` in the headline run, so the
same question applies to it. Uses the exact same builders as train.py so
the ablation cannot drift from the headline pipeline. Output feeds a
section of results/report.md.
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.data_utils import load_raw, get_features_and_label
from src.models import build_logreg, build_gbt, gbt_sample_weight
from src.evaluate_trust import all_metrics


def run(seed, n_splits):
    df = load_raw("data/PhysionetChallenge2012-set-a.csv.gz")
    X, y, _ = get_features_and_label(df)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    results = {"seed": seed, "n_splits": n_splits, "logreg": {}, "gbt": {}}
    for balanced in (True, False):
        key = "balanced" if balanced else "unweighted"
        oof_lr, oof_gbt = np.zeros(len(y)), np.zeros(len(y))
        for tr, val in skf.split(X, y):
            lr = build_logreg(seed, balanced=balanced)
            lr.fit(X.iloc[tr], y.iloc[tr])
            oof_lr[val] = lr.predict_proba(X.iloc[val])[:, 1]
            gbt = build_gbt(seed)
            gbt.fit(X.iloc[tr], y.iloc[tr], sample_weight=gbt_sample_weight(y.iloc[tr], balanced))
            oof_gbt[val] = gbt.predict_proba(X.iloc[val])[:, 1]
        for name, oof in (("logreg", oof_lr), ("gbt", oof_gbt)):
            m = all_metrics(oof, y.values)
            m["mean_predicted"] = float(oof.mean())
            results[name][key] = m
            print(f"{name:7s} {key:10s}: AUROC={m['auroc']:.3f} ECE={m['ece']:.3f} "
                  f"Brier={m['brier']:.3f} slope={m['calibration_slope']:.2f} "
                  f"intercept={m['calibration_intercept']:+.2f} mean_pred={m['mean_predicted']:.3f}")
    results["observed_rate"] = float(y.mean())
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-splits", type=int, default=5)
    args = ap.parse_args()
    res = run(args.seed, args.n_splits)
    os.makedirs("results", exist_ok=True)
    with open("results/class_weight_ablation.json", "w") as f:
        json.dump(res, f, indent=2)
    print("wrote results/class_weight_ablation.json")
