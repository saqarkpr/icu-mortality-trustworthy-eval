"""Trustworthy-evaluation metrics: calibration, subgroup calibration, and a
clinically-framed operating-point summary. Same ECE/Brier/reliability-plot
methodology as biomedical-rag-eval / sci-lit-rag-eval, generalized from a
QA-correctness label to a binary clinical outcome label.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


def expected_calibration_error(probs, labels, n_bins=10):
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    bin_details = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        count = mask.sum()
        if count == 0:
            continue
        conf = probs[mask].mean()
        acc = labels[mask].mean()
        ece += (count / n) * abs(acc - conf)
        bin_details.append({"bin": [float(lo), float(hi)], "count": int(count),
                             "confidence": float(conf), "empirical_rate": float(acc)})
    return float(ece), bin_details


def reliability_plot(probs, labels, path, title, n_bins=10):
    ece, bins = expected_calibration_error(probs, labels, n_bins=n_bins)
    centers = [(b["bin"][0] + b["bin"][1]) / 2 for b in bins]
    emp = [b["empirical_rate"] for b in bins]
    counts = [b["count"] for b in bins]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    sizes = np.array(counts, dtype=float)
    sizes = 30 + 300 * (sizes / sizes.max())
    ax.scatter(centers, emp, s=sizes, alpha=0.75, edgecolor="black", zorder=3)
    ax.plot(centers, emp, alpha=0.4, zorder=2)
    ax.set_xlabel("Mean predicted risk (bin)")
    ax.set_ylabel("Observed mortality rate (bin)")
    ax.set_title(f"{title}\nECE = {ece:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return ece


def discrimination_metrics(probs, labels):
    return {
        "auroc": float(roc_auc_score(labels, probs)),
        "auprc": float(average_precision_score(labels, probs)),
    }


def subgroup_calibration(probs, labels, groups, min_group_size=30):
    """ECE/Brier/AUROC broken out by a categorical subgroup (here: ICU
    admission type). Small groups are flagged rather than silently
    reported with an unstable estimate."""
    import pandas as pd
    df = pd.DataFrame({"prob": probs, "label": labels, "group": groups})
    out = {}
    for g, sub in df.groupby("group"):
        n = len(sub)
        entry = {"n": int(n), "mortality_rate": float(sub["label"].mean())}
        if n < min_group_size:
            entry["note"] = f"n<{min_group_size}, metrics omitted as unreliable"
        else:
            ece, _ = expected_calibration_error(sub["prob"].values, sub["label"].values, n_bins=8)
            entry["ece"] = float(ece)
            entry["brier"] = float(brier_score_loss(sub["label"], sub["prob"]))
            try:
                entry["auroc"] = float(roc_auc_score(sub["label"], sub["prob"]))
            except ValueError:
                entry["auroc"] = None
        out[g] = entry
    return out


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
    flagged_rate = (tp + fp) / len(labels)
    return {
        "threshold": threshold, "sensitivity": sensitivity, "specificity": specificity,
        "ppv": ppv, "flagged_rate": flagged_rate, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
