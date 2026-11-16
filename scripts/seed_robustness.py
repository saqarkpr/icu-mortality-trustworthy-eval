"""How much of the headline CV comparison is the particular seed-42 fold
split? Re-runs the exact train.py cross-validation for several seeds and
reports the spread of each model's pooled out-of-fold AUROC/ECE and of
the per-fold paired t-test p-value. Motivated by a concrete observation:
the same code and seed gave a GBT-vs-LogReg paired p-value of 0.109 under
one scikit-learn version and 0.190 under another -- a 5-fold t-test is a
low-power, fragile instrument, and it is better to show that than to
quote one p-value as if it were stable.
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.data_utils import load_raw, get_features_and_label
from src.models import build_logreg, build_gbt, gbt_sample_weight
from src.evaluate_trust import expected_calibration_error


def one_seed(X, y, seed, n_splits):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = {"logreg": np.zeros(len(y)), "gbt": np.zeros(len(y))}
    fold_auroc = {"logreg": [], "gbt": []}
    for tr, val in skf.split(X, y):
        lr = build_logreg(seed)
        lr.fit(X.iloc[tr], y.iloc[tr])
        oof["logreg"][val] = lr.predict_proba(X.iloc[val])[:, 1]
        gbt = build_gbt(seed)
        gbt.fit(X.iloc[tr], y.iloc[tr], sample_weight=gbt_sample_weight(y.iloc[tr]))
        oof["gbt"][val] = gbt.predict_proba(X.iloc[val])[:, 1]
        for m in oof:
            fold_auroc[m].append(roc_auc_score(y.iloc[val], oof[m][val]))
    p = float(stats.ttest_rel(fold_auroc["gbt"], fold_auroc["logreg"]).pvalue)
    row = {"seed": seed, "paired_fold_ttest_auroc_p": p}
    for m in oof:
        row[f"{m}_auroc"] = float(roc_auc_score(y, oof[m]))
        row[f"{m}_ece"] = float(expected_calibration_error(oof[m], y.values)[0])
    row["auroc_diff_gbt_minus_logreg"] = row["gbt_auroc"] - row["logreg_auroc"]
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 42])
    ap.add_argument("--n-splits", type=int, default=5)
    args = ap.parse_args()
    df = load_raw("data/PhysionetChallenge2012-set-a.csv.gz")
    X, y, _ = get_features_and_label(df)
    rows = []
    for s in args.seeds:
        r = one_seed(X, y, s, args.n_splits)
        rows.append(r)
        print(f"seed {s:3d}: LR AUROC {r['logreg_auroc']:.3f} ECE {r['logreg_ece']:.3f} | "
              f"GBT AUROC {r['gbt_auroc']:.3f} ECE {r['gbt_ece']:.3f} | "
              f"diff {r['auroc_diff_gbt_minus_logreg']:+.4f} | fold t-test p={r['paired_fold_ttest_auroc_p']:.3f}")
    keys = [k for k in rows[0] if k != "seed"]
    summary = {k: {"mean": float(np.mean([r[k] for r in rows])),
                   "std": float(np.std([r[k] for r in rows], ddof=1)),
                   "min": float(np.min([r[k] for r in rows])),
                   "max": float(np.max([r[k] for r in rows]))} for k in keys}
    os.makedirs("results", exist_ok=True)
    with open("results/seed_robustness.json", "w") as f:
        json.dump({"per_seed": rows, "summary": summary}, f, indent=2)
    print("wrote results/seed_robustness.json")
