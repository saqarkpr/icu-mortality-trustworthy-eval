"""Empirically verify the Survival/Length_of_stay leakage claim before
excluding those columns for good, rather than trusting the field
description alone. Run once; the printed result is quoted in
data/PROVENANCE.md."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gzip
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

with gzip.open("data/PhysionetChallenge2012-set-a.csv.gz", "rt") as f:
    df = pd.read_csv(f)

y = df["In-hospital_death"].astype(int)

print("Survival value counts split by label:")
print(df.groupby("In-hospital_death")["Survival"].describe())
print()

# A trivial single-feature classifier on Survival alone
X = df[["Survival"]].values
clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                     LogisticRegression(class_weight="balanced"))
scores = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
print(f"AUROC using Survival ALONE as the only feature: "
      f"{scores.mean():.4f} +/- {scores.std():.4f}")

X2 = df[["Length_of_stay"]].values
clf2 = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                      LogisticRegression(class_weight="balanced"))
scores2 = cross_val_score(clf2, X2, y, cv=5, scoring="roc_auc")
print(f"AUROC using Length_of_stay ALONE as the only feature: "
      f"{scores2.mean():.4f} +/- {scores2.std():.4f}")
