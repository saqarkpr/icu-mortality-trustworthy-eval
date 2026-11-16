"""Empirically verify the Survival/Length_of_stay leakage claim before
excluding those columns for good, rather than trusting the field
description alone. Run once; the printed result is quoted in
data/PROVENANCE.md.

History / correction: the first version of this script tested each column
with a *linear* single-feature logistic regression and got AUROC 0.259 for
`Survival`, which was then (wrongly) described as "confirming" leakage. A
linear test is the wrong instrument here: `Survival` is -1 for every
survivor, a small positive number for in-hospital deaths and a large
positive number for patients who died *after* discharge, so the relation
to the label is non-monotone and a linear logit cannot represent it. The
correct checks are (a) the challenge's own definition, which makes the
label a deterministic function of the two columns, and (b) any model that
can represent a non-monotone split, e.g. a shallow decision tree. Both are
run below; the linear test is kept only to show why it was misleading.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

from src.data_utils import load_raw, LABEL_COLUMN

df = load_raw("data/PhysionetChallenge2012-set-a.csv.gz")
y = df[LABEL_COLUMN].astype(int)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
out = {}

print("Survival by label (−1 = survived to follow-up, else days from admission to death):")
print(df.groupby(LABEL_COLUMN)["Survival"].describe())
print()
print("Length_of_stay by label:")
print(df.groupby(LABEL_COLUMN)["Length_of_stay"].describe())
print()

# (a) The deterministic rule implied by the challenge's own definitions:
#     died in hospital  <=>  died (Survival > 0) and died no later than
#     discharge (Survival <= Length_of_stay).
rule = ((df["Survival"] > 0) & (df["Survival"] <= df["Length_of_stay"])).astype(int)
agree = float((rule == y).mean())
out["rule_0<Survival<=LOS"] = {"agreement_with_label": agree,
                               "auroc": float(roc_auc_score(y, rule))}
print(f"Rule `0 < Survival <= Length_of_stay` reproduces the label on "
      f"{agree:.1%} of rows (AUROC {out['rule_0<Survival<=LOS']['auroc']:.3f})")

# (b) A model that CAN represent a non-monotone relation.
for cols in (["Survival"], ["Length_of_stay"], ["Survival", "Length_of_stay"]):
    X = df[cols].values
    tree = make_pipeline(SimpleImputer(strategy="median"),
                         DecisionTreeClassifier(max_depth=3, random_state=0))
    s = cross_val_score(tree, X, y, cv=cv, scoring="roc_auc")
    out[f"tree_depth3_{'+'.join(cols)}"] = {"auroc_mean": float(s.mean()), "auroc_std": float(s.std())}
    print(f"Depth-3 tree on {'+'.join(cols):24s}: AUROC {s.mean():.3f} +/- {s.std():.3f} (5-fold CV)")

# (c) The original linear test, kept to document why it misled.
for col in ("Survival", "Length_of_stay"):
    X = df[[col]].values
    clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                        LogisticRegression(class_weight="balanced"))
    s = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    out[f"linear_logreg_{col}"] = {"auroc_mean": float(s.mean()), "auroc_std": float(s.std())}
    print(f"Linear logistic on {col:18s}: AUROC {s.mean():.3f} +/- {s.std():.3f} "
          f"(5-fold CV)  <- non-monotone relation, misleading on its own")

os.makedirs("results", exist_ok=True)
with open("results/leakage_check.json", "w") as f:
    json.dump(out, f, indent=2)
print("wrote results/leakage_check.json")
