"""Diagnostic ablation, run after noticing Logistic Regression's OOF ECE
(0.189) was much worse than Gradient-Boosted Trees' (0.070) despite
similar AUROC, and suspecting `class_weight="balanced"` as the cause
(rebalancing shifts predicted probabilities away from the true base rate
in exchange for a better decision boundary -- it does not know about
calibration at all). Tests that suspicion directly instead of asserting
it. Output feeds a section of results/report.md.
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from src.data_utils import load_raw, get_features_and_label
from src.evaluate_trust import expected_calibration_error

SEED = 42

df = load_raw("data/PhysionetChallenge2012-set-a.csv.gz")
X, y, groups = get_features_and_label(df)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

results = {}
for label, kwargs in [("balanced", {"class_weight": "balanced"}), ("unweighted", {})]:
    oof = np.zeros(len(y))
    for tr, val in skf.split(X, y):
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=SEED, **kwargs)),
        ])
        pipe.fit(X.iloc[tr], y.iloc[tr])
        oof[val] = pipe.predict_proba(X.iloc[val])[:, 1]
    ece, _ = expected_calibration_error(oof, y.values, n_bins=10)
    auroc = roc_auc_score(y, oof)
    results[label] = {"ece": float(ece), "auroc": float(auroc)}
    print(f"LogisticRegression({label}): AUROC={auroc:.3f} ECE={ece:.3f}")

with open("results/class_weight_ablation.json", "w") as f:
    json.dump(results, f, indent=2)
