"""Two real, fully-fit models -- deliberately classical/CPU-only, since
this is tabular per-stay data, not free text; an LLM has no honest role
here. The comparison is the same shape as the rest of the portfolio: a
simple, interpretable baseline vs. a stronger model, evaluated on
calibration as well as discrimination.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
import numpy as np


def build_logreg(seed):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
    ])


def build_gbt(seed):
    # HistGradientBoostingClassifier handles NaN natively -- no imputation,
    # so missingness patterns themselves remain informative to the model
    # (as they are for real clinicians: a value that was never drawn is
    # itself a signal).
    return HistGradientBoostingClassifier(random_state=seed, max_iter=200)


def gbt_sample_weight(y):
    return compute_sample_weight(class_weight="balanced", y=y)


def saps_baseline_score(df):
    """The clinical status-quo comparator: the raw SAPS-I severity score
    used directly as a risk score, no fitting at all. This mirrors what
    the original 2012 challenge asked participants to beat."""
    return df["SAPS-I"].astype(float).values
