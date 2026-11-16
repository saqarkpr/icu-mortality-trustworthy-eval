"""Trustworthy-evaluation metrics: calibration, subgroup calibration, a
clinically-framed operating-point summary, and uncertainty via bootstrap.
Same ECE/Brier/reliability-plot methodology as biomedical-rag-eval /
sci-lit-rag-eval, generalized from a QA-correctness label to a binary
clinical outcome label, plus the calibration quantities the clinical-ML
reporting literature (TRIPOD, Van Calster et al. 2019) asks for:
calibration-in-the-large and calibration slope.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

_EPS = 1e-6


# ---------------------------------------------------------------- calibration
def expected_calibration_error(probs, labels, n_bins=10, strategy="uniform"):
    """ECE = sum_b (n_b / n) * |observed_rate_b - mean_predicted_b|.

    strategy="uniform": equal-width bins on [0, 1] (the classic definition).
    strategy="quantile": equal-mass bins (each holds ~n/n_bins patients).
    The two can disagree substantially when predictions are concentrated in
    a narrow range, which is exactly the case for a 14%-prevalence problem
    where most patients are low-risk -- so both are reported.
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if strategy == "uniform":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        bin_edges = np.unique(np.quantile(probs, np.linspace(0.0, 1.0, n_bins + 1)))
        bin_edges[0], bin_edges[-1] = 0.0, 1.0
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    n_bins_eff = len(bin_edges) - 1
    ece = 0.0
    n = len(probs)
    bin_details = []
    for i in range(n_bins_eff):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins_eff - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        conf = probs[mask].mean()
        acc = labels[mask].mean()
        ece += (count / n) * abs(acc - conf)
        bin_details.append({"bin": [float(lo), float(hi)], "count": count,
                            "confidence": float(conf), "empirical_rate": float(acc)})
    return float(ece), bin_details


def calibration_intercept_slope(probs, labels):
    """Calibration-in-the-large and calibration slope (Cox 1958), the two
    numbers clinical prediction-model guidelines ask for alongside a plot.

    - intercept: fitted with the slope fixed at 1 (an offset model on the
      logit of the prediction). 0 = perfect on average; < 0 means the model
      systematically over-predicts risk; > 0 means it under-predicts.
    - slope: coefficient on logit(p) in an unpenalized logistic fit.
      1 = perfect; < 1 means predictions are too extreme (over-fitted /
      over-confident); > 1 means too timid.
    Also returns the observed:expected ratio for the same reason.
    """
    probs = np.clip(np.asarray(probs, dtype=float), _EPS, 1 - _EPS)
    labels = np.asarray(labels, dtype=float)
    lp = logit(probs)

    def nll_offset(a):
        z = a + lp
        return -np.sum(labels * np.log(expit(z) + _EPS) + (1 - labels) * np.log(1 - expit(z) + _EPS))
    intercept = float(minimize_scalar(nll_offset, bounds=(-10, 10), method="bounded").x)

    lr = LogisticRegression(C=np.inf, max_iter=5000)  # unpenalized
    lr.fit(lp.reshape(-1, 1), labels.astype(int))
    slope = float(lr.coef_[0][0])
    return {
        "intercept": intercept,
        "slope": slope,
        "observed_expected_ratio": float(labels.mean() / probs.mean()),
        "mean_predicted": float(probs.mean()),
        "observed_rate": float(labels.mean()),
    }


def reliability_plot(probs, labels, path, title, n_bins=10):
    """Reliability diagram (uniform bins) with a histogram of predicted
    risk underneath, so a reader can see where the patients actually are
    -- a dot in a bin with 12 patients should not be read like a dot in a
    bin with 2,000."""
    ece, bins = expected_calibration_error(probs, labels, n_bins=n_bins)
    centers = [(b["bin"][0] + b["bin"][1]) / 2 for b in bins]
    emp = [b["empirical_rate"] for b in bins]
    conf = [b["confidence"] for b in bins]
    counts = [b["count"] for b in bins]
    cal = calibration_intercept_slope(probs, labels)

    fig, (ax, axh) = plt.subplots(2, 1, figsize=(5, 6.5), sharex=True,
                                  gridspec_kw={"height_ratios": [4, 1.2]})
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    sizes = np.array(counts, dtype=float)
    sizes = 30 + 300 * (sizes / sizes.max())
    ax.scatter(conf, emp, s=sizes, alpha=0.75, edgecolor="black", zorder=3,
               label="Bin (size ∝ n patients)")
    ax.plot(conf, emp, alpha=0.4, zorder=2)
    ax.set_ylabel("Observed mortality rate (bin)")
    ax.set_title(f"{title}\nECE = {ece:.3f}   slope = {cal['slope']:.2f}   "
                 f"intercept = {cal['intercept']:+.2f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    axh.hist(np.asarray(probs), bins=np.linspace(0, 1, n_bins + 1), color="steelblue",
             edgecolor="black", alpha=0.8)
    axh.set_xlabel("Predicted risk")
    axh.set_ylabel("n")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return ece


# ------------------------------------------------------------ discrimination
def discrimination_metrics(probs, labels):
    return {
        "auroc": float(roc_auc_score(labels, probs)),
        "auprc": float(average_precision_score(labels, probs)),
    }


def all_metrics(probs, labels, n_bins=10):
    """The full scalar panel for one (model, dataset) pair."""
    ece, _ = expected_calibration_error(probs, labels, n_bins=n_bins)
    ece_q, _ = expected_calibration_error(probs, labels, n_bins=n_bins, strategy="quantile")
    cal = calibration_intercept_slope(probs, labels)
    return {
        **discrimination_metrics(probs, labels),
        "ece": ece,
        "ece_quantile": ece_q,
        "brier": float(brier_score_loss(labels, probs)),
        "calibration_intercept": cal["intercept"],
        "calibration_slope": cal["slope"],
        "observed_expected_ratio": cal["observed_expected_ratio"],
    }


# ---------------------------------------------------------------- subgroups
def subgroup_calibration(probs, labels, groups, min_group_size=30, n_bins=8):
    """ECE/Brier/AUROC/calibration slope broken out by a categorical
    subgroup (here: ICU admission type). Small groups are flagged rather
    than silently reported with an unstable estimate. Uses fewer bins
    (default 8) than the aggregate ECE because the groups are smaller."""
    df = pd.DataFrame({"prob": probs, "label": labels, "group": groups})
    out = {}
    for g, sub in df.groupby("group"):
        n = len(sub)
        entry = {"n": int(n), "mortality_rate": float(sub["label"].mean())}
        if n < min_group_size:
            entry["note"] = f"n<{min_group_size}, metrics omitted as unreliable"
        else:
            p, y = sub["prob"].values, sub["label"].values
            ece, _ = expected_calibration_error(p, y, n_bins=n_bins)
            entry["ece"] = float(ece)
            entry["brier"] = float(brier_score_loss(y, p))
            cal = calibration_intercept_slope(p, y)
            entry["calibration_slope"] = cal["slope"]
            entry["calibration_intercept"] = cal["intercept"]
            try:
                entry["auroc"] = float(roc_auc_score(y, p))
            except ValueError:
                entry["auroc"] = None
        out[g] = entry
    return out


# ---------------------------------------------------------- operating points
def sensitivity_at_threshold(probs, labels, threshold):
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    pred = (probs >= threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    flagged_rate = (tp + fp) / len(labels)
    return {
        "threshold": float(threshold), "sensitivity": sensitivity, "specificity": specificity,
        "ppv": ppv, "npv": npv, "flagged_rate": flagged_rate,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def threshold_for_sensitivity(probs, labels, target_sensitivity):
    """Smallest threshold (on `probs >= t`) whose sensitivity on this data
    is at least `target_sensitivity`. Whether this is a legitimate
    pre-specified threshold or a post-hoc one depends entirely on which
    data it is called on -- the caller is responsible for saying which."""
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    order = np.argsort(-probs, kind="stable")
    sorted_labels = labels[order]
    cum_tp = np.cumsum(sorted_labels)
    total_pos = sorted_labels.sum()
    idx = int(np.searchsorted(cum_tp, target_sensitivity * total_pos))
    return float(probs[order][min(idx, len(order) - 1)])


# ------------------------------------------------------------------ bootstrap
def bootstrap_ci(metric_fn, probs, labels, n_boot=1000, seed=0, alpha=0.05):
    """Percentile bootstrap CI of a scalar metric_fn(probs, labels),
    resampling patients with replacement (stratified on the label so every
    resample has the same number of deaths -- keeps AUPRC/ECE comparable
    across resamples at a 14% base rate)."""
    rng = np.random.default_rng(seed)
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    pos = np.flatnonzero(labels == 1)
    neg = np.flatnonzero(labels == 0)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        vals.append(metric_fn(probs[idx], labels[idx]))
    vals = np.asarray(vals, dtype=float)
    return {
        "point": float(metric_fn(probs, labels)),
        "ci_low": float(np.quantile(vals, alpha / 2)),
        "ci_high": float(np.quantile(vals, 1 - alpha / 2)),
        "n_boot": int(n_boot),
    }


def paired_bootstrap_difference(metric_fn, probs_a, probs_b, labels, n_boot=1000, seed=0,
                                alpha=0.05):
    """Paired bootstrap for metric(a) - metric(b) on the SAME patients
    (same resampled indices applied to both models, which is what makes it
    paired). Returns the percentile CI and a two-sided bootstrap p-value
    for the difference being zero."""
    rng = np.random.default_rng(seed)
    probs_a = np.asarray(probs_a)
    probs_b = np.asarray(probs_b)
    labels = np.asarray(labels)
    pos = np.flatnonzero(labels == 1)
    neg = np.flatnonzero(labels == 0)
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        diffs.append(metric_fn(probs_a[idx], labels[idx]) - metric_fn(probs_b[idx], labels[idx]))
    diffs = np.asarray(diffs, dtype=float)
    frac_le0 = float(np.mean(diffs <= 0))
    frac_ge0 = float(np.mean(diffs >= 0))
    return {
        "point": float(metric_fn(probs_a, labels) - metric_fn(probs_b, labels)),
        "ci_low": float(np.quantile(diffs, alpha / 2)),
        "ci_high": float(np.quantile(diffs, 1 - alpha / 2)),
        "p_two_sided": float(min(1.0, 2 * min(frac_le0, frac_ge0))),
        "n_boot": int(n_boot),
    }
