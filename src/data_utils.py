"""Loading and honest feature/label separation for the PhysioNet/CinC 2012
Challenge ("Predicting Mortality of ICU Patients") aggregated dataset.

Data source: real PhysioNet Challenge 2012 records, one row per ICU stay,
each physiological variable aggregated (first/last/lowest/highest/median)
over the recorded stay. See data/PROVENANCE.md for exact provenance.

IMPORTANT (data leakage): the raw files also carry `Survival` and
`Length_of_stay`. `Survival` is defined by the challenge organizers as the
number of days between ICU admission and death for patients who died, and
-1 for patients who survived to discharge -- i.e. for non-survivors it is
almost a direct encoding of the label itself, and `Length_of_stay` is
entangled with the same event for the same reason (a stay's length is
partly determined by when/whether the patient died). Both are excluded
from the feature set below. This was verified empirically in
scripts/check_leakage.py before writing this module (see
data/PROVENANCE.md) rather than assumed from the field description alone.
"""
import gzip
import pandas as pd

LABEL_COLUMN = "In-hospital_death"
ID_COLUMN = "recordid"
LEAKAGE_COLUMNS = ["Survival", "Length_of_stay"]
# ICU admission type flags (one-hot minus the implicit 4th category, MICU,
# which is "all three flags 0" -- see data/PROVENANCE.md).
ICU_TYPE_COLUMNS = ["CCU", "CSRU", "SICU"]


def load_raw(path):
    with gzip.open(path, "rt") as f:
        df = pd.read_csv(f)
    return df


def icu_type(df):
    """Derive a single categorical ICU-admission-type column from the three
    binary flags (CCU/CSRU/SICU); all-zero means Medical ICU (MICU), the
    challenge's fourth category. The flags are mutually exclusive in the
    data (checked: no row has more than one set), which is asserted here
    rather than silently resolved by priority order."""
    flags = df[ICU_TYPE_COLUMNS].fillna(0).astype(int)
    assert (flags.sum(axis=1) <= 1).all(), "ICU type flags are not mutually exclusive"
    out = pd.Series("MICU", index=df.index, dtype=object)
    for col in ICU_TYPE_COLUMNS:
        out[flags[col] == 1] = col
    return out


def get_features_and_label(df):
    """Return (X, y, groups) with leakage columns and identifiers removed.
    `groups` is the derived ICU-type column, kept out of X separately so it
    can be used for the subgroup-calibration check without ever being fed
    to the models as a feature by accident."""
    y = df[LABEL_COLUMN].astype(int)
    groups = icu_type(df)
    drop_cols = [LABEL_COLUMN, ID_COLUMN] + LEAKAGE_COLUMNS
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return X, y, groups


def missingness_report(X):
    """Fraction of missing values per column, sorted descending."""
    return X.isna().mean().sort_values(ascending=False)
