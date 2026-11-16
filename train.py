"""5-fold stratified CV on Set A, producing out-of-fold predicted
probabilities for both models. Out-of-fold predictions are used (rather
than in-sample fit) so calibration is measured honestly -- a model is
never evaluated on rows it was trained on.
"""
import argparse
import json
import os
import numpy as np
import pandas as pd
import platform
import sklearn
from sklearn.model_selection import StratifiedKFold

from src.data_utils import load_raw, get_features_and_label
from src.models import build_logreg, build_gbt, gbt_sample_weight

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--data", default="data/PhysionetChallenge2012-set-a.csv.gz")
    ap.add_argument("--out", default="results/oof_predictions.json")
    args = ap.parse_args()

    df = load_raw(args.data)
    X, y, groups = get_features_and_label(df)

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    oof = {
        "logreg": np.zeros(len(y)),
        "gbt": np.zeros(len(y)),
    }
    fold_id = np.zeros(len(y), dtype=int)

    for k, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        lr = build_logreg(args.seed)
        lr.fit(X_tr, y_tr)
        oof["logreg"][val_idx] = lr.predict_proba(X_val)[:, 1]

        gbt = build_gbt(args.seed)
        sw = gbt_sample_weight(y_tr)
        gbt.fit(X_tr, y_tr, sample_weight=sw)
        oof["gbt"][val_idx] = gbt.predict_proba(X_val)[:, 1]

        fold_id[val_idx] = k
        print(f"fold {k}: train={len(train_idx)} val={len(val_idx)} "
              f"val_mortality_rate={y_val.mean():.3f}")

    out = {
        "seed": args.seed,
        "n_splits": args.n_splits,
        # GBT predictions are deterministic for a given scikit-learn version
        # but shift across versions; record what produced this file.
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__},
        "n_features": int(X.shape[1]),
        "recordid": df["recordid"].tolist(),
        "label": y.tolist(),
        "icu_type": groups.tolist(),
        "fold": fold_id.tolist(),
        "oof_logreg": oof["logreg"].tolist(),
        "oof_gbt": oof["gbt"].tolist(),
    }
    os.makedirs("results", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
