"""Two real, fully-fit models -- deliberately classical/CPU-only, since
this is tabular per-stay data, not free text; an LLM has no honest role
here. The comparison is the same shape as the rest of the portfolio: a
simple, interpretable baseline vs. a stronger model, evaluated on
calibration as well as discrimination.

Both builders take a `balanced` flag so the class-weighting ablation
(scripts/class_weight_ablation.py) exercises exactly the same pipeline as
the headline run instead of a re-typed copy of it.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

# Non-default settings, listed in one place so the README can be honest
# about what "no tuning" means: these were chosen up front, not searched.
LOGREG_MAX_ITER = 2000
GBT_MAX_ITER = 200  # sklearn default is 100; no early stopping at n=3,200


def build_logreg(seed, balanced=True):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=LOGREG_MAX_ITER,
                                   class_weight="balanced" if balanced else None,
                                   random_state=seed)),
    ])


def build_gbt(seed):
    # HistGradientBoostingClassifier handles NaN natively -- no imputation,
    # so missingness patterns themselves remain informative to the model
    # (as they are for real clinicians: a value that was never drawn is
    # itself a signal).
    return HistGradientBoostingClassifier(random_state=seed, max_iter=GBT_MAX_ITER)


def gbt_sample_weight(y, balanced=True):
    if not balanced:
        return None
    return compute_sample_weight(class_weight="balanced", y=y)


def saps_baseline_score(df):
    """The clinical status-quo comparator: the raw SAPS-I severity score
    used directly as a risk score, no fitting at all. This mirrors what
    the original 2012 challenge asked participants to beat."""
    return df["SAPS-I"].astype(float).values
