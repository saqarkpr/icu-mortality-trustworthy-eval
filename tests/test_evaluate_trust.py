"""Unit checks for the metric code in src/evaluate_trust.py on synthetic
data with known answers. Run: python -m pytest -q"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import pytest

from src.evaluate_trust import (
    expected_calibration_error, calibration_intercept_slope, sensitivity_at_threshold,
    threshold_for_sensitivity, bootstrap_ci, paired_bootstrap_difference, subgroup_calibration,
)
from src.data_utils import icu_type


@pytest.fixture(scope="module")
def calibrated():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 50_000)
    y = (rng.uniform(0, 1, 50_000) < p).astype(int)
    return p, y


def test_ece_near_zero_when_calibrated(calibrated):
    p, y = calibrated
    assert expected_calibration_error(p, y)[0] < 0.01
    assert expected_calibration_error(p, y, strategy="quantile")[0] < 0.01


def test_ece_detects_systematic_shift(calibrated):
    p, y = calibrated
    ece, _ = expected_calibration_error(np.clip(p + 0.2, 0, 1), y)
    assert ece > 0.15


def test_ece_bins_partition_all_points(calibrated):
    p, y = calibrated
    for strategy in ("uniform", "quantile"):
        _, bins = expected_calibration_error(p, y, strategy=strategy)
        assert sum(b["count"] for b in bins) == len(p)


def test_calibration_intercept_slope_ideal(calibrated):
    p, y = calibrated
    cal = calibration_intercept_slope(p, y)
    assert abs(cal["intercept"]) < 0.05
    assert abs(cal["slope"] - 1) < 0.05
    assert abs(cal["observed_expected_ratio"] - 1) < 0.02


def test_calibration_slope_direction(calibrated):
    p, y = calibrated
    from scipy.special import expit, logit
    too_extreme = expit(2 * logit(np.clip(p, 1e-6, 1 - 1e-6)))
    too_timid = expit(0.5 * logit(np.clip(p, 1e-6, 1 - 1e-6)))
    assert calibration_intercept_slope(too_extreme, y)["slope"] < 0.6
    assert calibration_intercept_slope(too_timid, y)["slope"] > 1.8


def test_sensitivity_at_threshold_counts():
    p = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
    y = np.array([1, 0, 1, 0, 0])
    r = sensitivity_at_threshold(p, y, 0.5)
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (1, 1, 1, 2)
    assert r["sensitivity"] == 0.5 and r["ppv"] == 0.5 and r["flagged_rate"] == 0.4
    assert r["npv"] == pytest.approx(2 / 3)


def test_threshold_for_sensitivity_meets_target(calibrated):
    p, y = calibrated
    for target in (0.5, 0.8, 0.9):
        t = threshold_for_sensitivity(p, y, target)
        assert sensitivity_at_threshold(p, y, t)["sensitivity"] >= target


def test_bootstrap_ci_contains_point(calibrated):
    p, y = calibrated
    from sklearn.metrics import roc_auc_score
    ci = bootstrap_ci(lambda a, b: roc_auc_score(b, a), p[:3000], y[:3000], n_boot=100)
    assert ci["ci_low"] <= ci["point"] <= ci["ci_high"]


def test_paired_bootstrap_monotone_transform_is_zero(calibrated):
    p, y = calibrated
    from sklearn.metrics import roc_auc_score
    d = paired_bootstrap_difference(lambda a, b: roc_auc_score(b, a), p[:3000], np.sqrt(p[:3000]),
                                    y[:3000], n_boot=50)
    assert d["point"] == 0 and d["p_two_sided"] == 1.0


def test_subgroup_small_groups_flagged():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 200)
    y = (rng.uniform(0, 1, 200) < p).astype(int)
    g = np.array(["big"] * 190 + ["tiny"] * 10)
    out = subgroup_calibration(p, y, g)
    assert "ece" in out["big"] and "note" in out["tiny"]


def test_icu_type_derivation():
    df = pd.DataFrame({"CCU": [1, 0, 0, 0], "CSRU": [0, 1, 0, 0], "SICU": [0, 0, 1, 0]})
    assert icu_type(df).tolist() == ["CCU", "CSRU", "SICU", "MICU"]
    with pytest.raises(AssertionError):
        icu_type(pd.DataFrame({"CCU": [1], "CSRU": [1], "SICU": [0]}))
