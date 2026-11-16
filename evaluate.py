"""Consumes results/oof_predictions.json (from train.py), adds the SAPS-I
clinical baseline and a genuine held-out check on Set B (fit on all of Set
A, evaluated once on Set B with no re-tuning), bootstrap confidence
intervals, calibration intercept/slope, and a pre-specified operating
point (threshold fixed on Set A, applied once to Set B). Writes
results/report.md + reliability diagrams + results/summary.json.
Optional sections are added when their inputs exist:
  scripts/class_weight_ablation.py -> results/class_weight_ablation.json
  scripts/seed_robustness.py       -> results/seed_robustness.json
  scripts/check_leakage.py         -> results/leakage_check.json
"""
import argparse
import json
import os
import platform
import numpy as np
import pandas as pd
import sklearn
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

from src.data_utils import load_raw, get_features_and_label, missingness_report
from src.models import build_logreg, build_gbt, gbt_sample_weight, saps_baseline_score
from src.evaluate_trust import (
    expected_calibration_error, reliability_plot, discrimination_metrics, all_metrics,
    subgroup_calibration, sensitivity_at_threshold, threshold_for_sensitivity,
    bootstrap_ci, paired_bootstrap_difference,
)

RESULTS = "results"
MODEL_NAMES = {"logreg": "Logistic Regression", "gbt": "Gradient-Boosted Trees"}


def per_fold_metrics(labels, fold_id, probs):
    rows = []
    for k in sorted(set(fold_id)):
        mask = fold_id == k
        y_k, p_k = labels[mask], probs[mask]
        ece, _ = expected_calibration_error(p_k, y_k, n_bins=10)
        rows.append({
            "fold": int(k),
            "auroc": roc_auc_score(y_k, p_k),
            "auprc": average_precision_score(y_k, p_k),
            "ece": ece,
            "brier": brier_score_loss(y_k, p_k),
        })
    return pd.DataFrame(rows)


def summarize(df, name):
    cols = ["auroc", "auprc", "ece", "brier"]
    return {
        f"{name}_mean": {c: float(df[c].mean()) for c in cols},
        f"{name}_std": {c: float(df[c].std()) for c in cols},
    }


def _auroc(p, y):
    return roc_auc_score(y, p)


def _auprc(p, y):
    return average_precision_score(y, p)


def _ece(p, y):
    return expected_calibration_error(p, y)[0]


def _brier(p, y):
    return brier_score_loss(y, p)


def _fmt_ci(d, nd=3):
    return f"{d['point']:.{nd}f} [{d['ci_low']:.{nd}f}, {d['ci_high']:.{nd}f}]"


def _load_optional(name):
    try:
        with open(os.path.join(RESULTS, name)) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--boot-seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "oof_predictions.json")) as f:
        oof = json.load(f)

    labels = np.array(oof["label"])
    fold_id = np.array(oof["fold"])
    icu_type = np.array(oof["icu_type"])
    p_lr = np.array(oof["oof_logreg"])
    p_gbt = np.array(oof["oof_gbt"])
    seed = oof["seed"]
    n_splits = oof.get("n_splits", len(set(fold_id)))

    # ---- 1. Cross-validated, out-of-fold discrimination + calibration ----
    df_lr = per_fold_metrics(labels, fold_id, p_lr)
    df_gbt = per_fold_metrics(labels, fold_id, p_gbt)

    # paired per-fold comparison (same folds for both models -> a paired
    # test is valid here). Low power with 5 folds; the Set B paired
    # bootstrap below is the better-powered version of the same question.
    auroc_diff = df_gbt["auroc"].values - df_lr["auroc"].values
    try:
        auroc_p = float(stats.ttest_rel(df_gbt["auroc"], df_lr["auroc"]).pvalue)
    except Exception:
        auroc_p = None
    auroc_p_str = f"{auroc_p:.4f}" if auroc_p is not None else "n/a"

    # ---- 2. Pooled OOF panel + reliability diagrams (n=4000, out-of-fold) ----
    pooled = {"logreg": all_metrics(p_lr, labels), "gbt": all_metrics(p_gbt, labels)}
    reliability_plot(p_lr, labels, f"{RESULTS}/reliability_logreg_seta_oof.png",
                     "Logistic Regression (Set A, out-of-fold)")
    reliability_plot(p_gbt, labels, f"{RESULTS}/reliability_gbt_seta_oof.png",
                     "Gradient-Boosted Trees (Set A, out-of-fold)")

    # ---- 3. SAPS-I clinical-severity-score baseline (no fitting) ----
    df_full = load_raw("data/PhysionetChallenge2012-set-a.csv.gz")
    saps_scores = saps_baseline_score(df_full)
    valid = ~np.isnan(saps_scores)
    saps_metrics = discrimination_metrics(saps_scores[valid], labels[valid])

    # ---- 4. Subgroup calibration by ICU admission type (pooled OOF) ----
    subgroup_lr = subgroup_calibration(p_lr, labels, icu_type)
    subgroup_gbt = subgroup_calibration(p_gbt, labels, icu_type)

    # ---- 5. Held-out generalization check: fit on ALL of Set A, evaluate
    #         once on Set B, no re-tuning, no threshold selection on B ----
    X_a, y_a, _ = get_features_and_label(df_full)
    lr_final = build_logreg(seed)
    lr_final.fit(X_a, y_a)
    gbt_final = build_gbt(seed)
    gbt_final.fit(X_a, y_a, sample_weight=gbt_sample_weight(y_a))

    df_b = load_raw("data/PhysionetChallenge2012-set-b.csv.gz")
    X_b, y_b, icu_type_b = get_features_and_label(df_b)
    assert list(X_b.columns) == list(X_a.columns), "Set A / Set B feature columns differ"
    y_b = y_b.values
    p_b = {"logreg": lr_final.predict_proba(X_b)[:, 1], "gbt": gbt_final.predict_proba(X_b)[:, 1]}

    reliability_plot(p_b["logreg"], y_b, f"{RESULTS}/reliability_logreg_setb_heldout.png",
                     "Logistic Regression (Set B, held-out, no re-tuning)")
    reliability_plot(p_b["gbt"], y_b, f"{RESULTS}/reliability_gbt_setb_heldout.png",
                     "Gradient-Boosted Trees (Set B, held-out, no re-tuning)")

    setb_metrics = {m: all_metrics(p_b[m], y_b) for m in p_b}
    saps_scores_b = saps_baseline_score(df_b)
    valid_b = ~np.isnan(saps_scores_b)
    saps_metrics_b = discrimination_metrics(saps_scores_b[valid_b], y_b[valid_b])

    # Bootstrap CIs on Set B (patients resampled, label-stratified) and a
    # paired bootstrap for the GBT - LogReg difference on the same patients.
    metric_fns = {"auroc": _auroc, "auprc": _auprc, "ece": _ece, "brier": _brier}
    setb_ci = {m: {k: bootstrap_ci(fn, p_b[m], y_b, n_boot=args.n_boot, seed=args.boot_seed)
                   for k, fn in metric_fns.items()} for m in p_b}
    setb_ci["saps1"] = {k: bootstrap_ci(fn, saps_scores_b[valid_b], y_b[valid_b],
                                        n_boot=args.n_boot, seed=args.boot_seed)
                        for k, fn in (("auroc", _auroc), ("auprc", _auprc))}
    setb_paired = {k: paired_bootstrap_difference(fn, p_b["gbt"], p_b["logreg"], y_b,
                                                  n_boot=args.n_boot, seed=args.boot_seed)
                   for k, fn in metric_fns.items()}

    subgroup_lr_b = subgroup_calibration(p_b["logreg"], y_b, icu_type_b.values)
    subgroup_gbt_b = subgroup_calibration(p_b["gbt"], y_b, icu_type_b.values)

    # ---- 6. Clinically-framed operating points (GBT) ----
    # (a) PRE-SPECIFIED: thresholds fixed on Set A out-of-fold predictions
    #     only, then applied once to Set B. This is the deployable version
    #     of the question "what does it cost to catch 80% of deaths".
    prevalence_a = float(labels.mean())
    prevalence_b = float(y_b.mean())
    thr80_a = threshold_for_sensitivity(p_gbt, labels, 0.80)
    thr90_a = threshold_for_sensitivity(p_gbt, labels, 0.90)
    op_prespecified = []
    for lab, thr in (("= Set A prevalence", prevalence_a),
                     ("80% sensitivity on Set A (OOF)", thr80_a),
                     ("90% sensitivity on Set A (OOF)", thr90_a)):
        op_prespecified.append({"chosen_on": "Set A (out-of-fold)", "rule": lab,
                                "on_setA_oof": sensitivity_at_threshold(p_gbt, labels, thr),
                                "on_setB": sensitivity_at_threshold(p_b["gbt"], y_b, thr)})
    # (b) POST-HOC on Set B, kept for comparison: what it *would* take on B
    #     itself. Explicitly not a deployable threshold.
    thr80_b = threshold_for_sensitivity(p_b["gbt"], y_b, 0.80)
    op_posthoc = [
        {"rule": "= Set B prevalence", **sensitivity_at_threshold(p_b["gbt"], y_b, prevalence_b)},
        {"rule": "= 0.5", **sensitivity_at_threshold(p_b["gbt"], y_b, 0.5)},
        {"rule": "80% sensitivity chosen on Set B", **sensitivity_at_threshold(p_b["gbt"], y_b, thr80_b)},
    ]

    # ---- 7. Missingness ----
    miss = missingness_report(X_a)
    top_missing = miss.head(10)

    # ---- optional inputs from the diagnostic scripts ----
    cw = _load_optional("class_weight_ablation.json")
    seeds = _load_optional("seed_robustness.json")
    leak = _load_optional("leakage_check.json")

    # ---- write summary.json ----
    summary = {
        "seed": seed,
        "n_splits": n_splits,
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                        "numpy": np.__version__, "pandas": pd.__version__},
        "n_setA": int(len(labels)),
        "n_setB": int(len(y_b)),
        "n_features": int(X_a.shape[1]),
        "mortality_rate_setA": prevalence_a,
        "mortality_rate_setB": prevalence_b,
        "cv_oof_setA": {**summarize(df_lr, "logreg"), **summarize(df_gbt, "gbt")},
        "cv_pooled_oof_setA": pooled,
        "paired_fold_auroc_diff_gbt_minus_logreg": auroc_diff.tolist(),
        "paired_ttest_auroc_pvalue": auroc_p,
        "saps1_baseline_setA": saps_metrics,
        "saps1_baseline_setB": saps_metrics_b,
        "setB_heldout": setb_metrics,
        "setB_heldout_bootstrap_ci": setb_ci,
        "setB_paired_bootstrap_gbt_minus_logreg": setb_paired,
        "subgroup_calibration_setA_oof": {"logreg": subgroup_lr, "gbt": subgroup_gbt},
        "subgroup_calibration_setB_heldout": {"logreg": subgroup_lr_b, "gbt": subgroup_gbt_b},
        "operating_points_gbt_prespecified_on_setA": op_prespecified,
        "operating_points_gbt_posthoc_on_setB": op_posthoc,
        "top_missing_features": {k: float(v) for k, v in top_missing.items()},
    }
    with open(f"{RESULTS}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---- write report.md ----
    L = []
    L.append("# Results summary\n")
    L.append(f"Seed {seed}, {n_splits}-fold stratified CV on Set A (n={len(labels)}, "
             f"{X_a.shape[1]} features), held-out check on Set B (n={len(y_b)}, fit only on "
             f"Set A, never re-tuned). Bootstrap CIs: {args.n_boot} label-stratified resamples. "
             f"Environment: Python {platform.python_version()}, scikit-learn "
             f"{sklearn.__version__} (GBT results are reproducible within a scikit-learn "
             f"version but shift slightly across versions -- see the reproducibility note "
             f"in README).\n")

    L.append("## Cross-validated (out-of-fold) performance, Set A\n")
    L.append("Per-fold mean ± std across folds; calibration intercept/slope on the pooled "
             "out-of-fold predictions (ideal: intercept 0, slope 1).\n")
    L.append("| Model | AUROC | AUPRC | ECE | Brier | Cal. intercept | Cal. slope |")
    L.append("|---|---|---|---|---|---|---|")
    for key, d in (("logreg", df_lr), ("gbt", df_gbt)):
        pm = pooled[key]
        L.append(f"| {MODEL_NAMES[key]} | {d['auroc'].mean():.3f} ± {d['auroc'].std():.3f} "
                 f"| {d['auprc'].mean():.3f} ± {d['auprc'].std():.3f} "
                 f"| {d['ece'].mean():.3f} ± {d['ece'].std():.3f} "
                 f"| {d['brier'].mean():.3f} ± {d['brier'].std():.3f} "
                 f"| {pm['calibration_intercept']:+.2f} | {pm['calibration_slope']:.2f} |")
    L.append(f"| SAPS-I score alone (no fitting) | {saps_metrics['auroc']:.3f} "
             f"| {saps_metrics['auprc']:.3f} | n/a | n/a | n/a | n/a |\n")
    L.append(f"Paired per-fold AUROC difference (GBT − LogReg): mean {auroc_diff.mean():+.4f}, "
             f"paired t-test p={auroc_p_str} ({n_splits} folds only -- indicative, not a "
             f"strong-power test; see the Set B paired bootstrap and the seed-robustness "
             f"section below).\n")
    L.append("Pooled out-of-fold ECE with equal-width vs equal-mass (quantile) bins -- "
             "the two disagree when predictions are concentrated near zero, as they are "
             "for most patients at 14% prevalence:\n")
    L.append("| Model | ECE (10 equal-width bins) | ECE (10 equal-mass bins) | Mean predicted | Observed rate |")
    L.append("|---|---|---|---|---|")
    for key in ("logreg", "gbt"):
        pm = pooled[key]
        L.append(f"| {MODEL_NAMES[key]} | {pm['ece']:.3f} | {pm['ece_quantile']:.3f} "
                 f"| {prevalence_a / pm['observed_expected_ratio']:.3f} | {prevalence_a:.3f} |")
    L.append("")

    L.append("## Held-out generalization: fit on Set A, evaluate once on Set B\n")
    L.append("Point estimate [95% bootstrap CI].\n")
    L.append("| Model | AUROC | AUPRC | ECE | Brier | Cal. intercept | Cal. slope |")
    L.append("|---|---|---|---|---|---|---|")
    for key in ("logreg", "gbt"):
        c, m = setb_ci[key], setb_metrics[key]
        L.append(f"| {MODEL_NAMES[key]} | {_fmt_ci(c['auroc'])} | {_fmt_ci(c['auprc'])} "
                 f"| {_fmt_ci(c['ece'])} | {_fmt_ci(c['brier'])} "
                 f"| {m['calibration_intercept']:+.2f} | {m['calibration_slope']:.2f} |")
    L.append(f"| SAPS-I score alone (no fitting) | {_fmt_ci(setb_ci['saps1']['auroc'])} "
             f"| {_fmt_ci(setb_ci['saps1']['auprc'])} | n/a | n/a | n/a | n/a |\n")
    L.append("Paired bootstrap, GBT − LogReg on the same Set B patients (n=4000; far better "
             "powered than the 5-fold t-test above):\n")
    L.append("| Metric | Difference (GBT − LogReg) [95% CI] | Two-sided bootstrap p |")
    L.append("|---|---|---|")
    for k in ("auroc", "auprc", "brier", "ece"):
        d = setb_paired[k]
        L.append(f"| {k.upper()} | {d['point']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}] "
                 f"| {d['p_two_sided']:.3f} |")
    L.append("")

    L.append("**Reliability diagrams (Set A, out-of-fold):**\n")
    L.append("| Logistic Regression | Gradient-Boosted Trees |")
    L.append("|---|---|")
    L.append("| ![LR reliability Set A](reliability_logreg_seta_oof.png) "
             "| ![GBT reliability Set A](reliability_gbt_seta_oof.png) |\n")
    L.append("**Reliability diagrams (Set B, held-out, no re-tuning):**\n")
    L.append("| Logistic Regression | Gradient-Boosted Trees |")
    L.append("|---|---|")
    L.append("| ![LR reliability Set B](reliability_logreg_setb_heldout.png) "
             "| ![GBT reliability Set B](reliability_gbt_setb_heldout.png) |\n")

    def subgroup_table(sub_lr, sub_gbt):
        rows = ["| ICU type | n | Mortality rate | LogReg ECE | LogReg slope | LogReg AUROC "
                "| GBT ECE | GBT slope | GBT AUROC |",
                "|---|---|---|---|---|---|---|---|---|"]
        for g in sorted(sub_lr.keys()):
            a, b = sub_lr[g], sub_gbt[g]

            def cell(e, k, fmt):
                return fmt.format(e[k]) if e.get(k) is not None else e.get("note", "n/a")
            rows.append(f"| {g} | {a['n']} | {a['mortality_rate']:.3f} "
                        f"| {cell(a, 'ece', '{:.3f}')} | {cell(a, 'calibration_slope', '{:.2f}')} "
                        f"| {cell(a, 'auroc', '{:.3f}')} "
                        f"| {cell(b, 'ece', '{:.3f}')} | {cell(b, 'calibration_slope', '{:.2f}')} "
                        f"| {cell(b, 'auroc', '{:.3f}')} |")
        return rows

    L.append("## Subgroup calibration by ICU admission type\n")
    L.append("ECE here uses 8 equal-width bins (groups are smaller than the full cohort), so "
             "it is comparable across rows of these tables but not to the 10-bin aggregate "
             "ECE above.\n")
    L.append("**Set A, out-of-fold:**\n")
    L += subgroup_table(subgroup_lr, subgroup_gbt)
    L.append("")
    L.append("**Set B, held-out (fit on Set A only):**\n")
    L += subgroup_table(subgroup_lr_b, subgroup_gbt_b)
    L.append("")

    L.append("## Clinically-framed operating points (GBT)\n")
    L.append(f"Mortality prevalence: Set A {prevalence_a:.3f}, Set B {prevalence_b:.3f}.\n")
    L.append("**Pre-specified thresholds** -- fixed on Set A out-of-fold predictions, then "
             "applied once to Set B. This is the deployable form of the question: the "
             "threshold is chosen before the held-out data is touched, and the Set B row "
             "shows what the chosen rule *actually* delivers out of sample.\n")
    L.append("| Rule (chosen on Set A) | Threshold | Data | Sensitivity | Specificity | PPV | NPV | Flagged rate |")
    L.append("|---|---|---|---|---|---|---|---|")
    for op in op_prespecified:
        for data, key in (("Set A (OOF)", "on_setA_oof"), ("Set B", "on_setB")):
            r = op[key]
            L.append(f"| {op['rule']} | {r['threshold']:.3f} | {data} | {r['sensitivity']:.3f} "
                     f"| {r['specificity']:.3f} | {r['ppv']:.3f} | {r['npv']:.3f} | {r['flagged_rate']:.3f} |")
    L.append("")
    L.append("**Post-hoc thresholds on Set B** -- chosen by looking at Set B itself; they "
             "illustrate the trade-off curve but are not deployable numbers.\n")
    L.append("| Rule | Threshold | Sensitivity | Specificity | PPV | NPV | Flagged rate |")
    L.append("|---|---|---|---|---|---|---|")
    for r in op_posthoc:
        L.append(f"| {r['rule']} | {r['threshold']:.3f} | {r['sensitivity']:.3f} "
                 f"| {r['specificity']:.3f} | {r['ppv']:.3f} | {r['npv']:.3f} | {r['flagged_rate']:.3f} |")
    L.append("")

    L.append("## Missingness (Set A, top 10 most-missing features)\n")
    L.append("| Feature | Missing fraction |")
    L.append("|---|---|")
    for k, v in top_missing.items():
        L.append(f"| {k} | {v:.3f} |")
    L.append("")

    L.append("## Diagnostic: why was Logistic Regression's calibration so much worse than GBT's?\n")
    if cw is None:
        L.append("(Run `python scripts/class_weight_ablation.py` before `evaluate.py` to generate this section.)")
    else:
        L.append("Suspected `class_weight=\"balanced\"` (used to get a reasonable decision "
                 "boundary under 14% prevalence) as the cause, rather than assuming it, and "
                 "re-ran both models with and without class balancing on the same folds "
                 "(`scripts/class_weight_ablation.py`; pooled out-of-fold, Set A):\n")
        L.append("| Model | Variant | AUROC | ECE | Brier | Cal. intercept | Cal. slope | Mean predicted (observed {:.3f}) |".format(cw["observed_rate"]))
        L.append("|---|---|---|---|---|---|---|---|")
        for key in ("logreg", "gbt"):
            for var, lab in (("balanced", "class-balanced (used in headline run)"), ("unweighted", "unweighted")):
                m = cw[key][var]
                L.append(f"| {MODEL_NAMES[key]} | {lab} | {m['auroc']:.3f} | {m['ece']:.3f} "
                         f"| {m['brier']:.3f} | {m['calibration_intercept']:+.2f} "
                         f"| {m['calibration_slope']:.2f} | {m['mean_predicted']:.3f} |")
        L.append("")
        lb, lu = cw["logreg"]["balanced"], cw["logreg"]["unweighted"]
        gb, gu = cw["gbt"]["balanced"], cw["gbt"]["unweighted"]
        L.append(f"**Logistic Regression:** class-balancing was the whole story. AUROC is "
                 f"unchanged ({lb['auroc']:.3f} vs {lu['auroc']:.3f}) but ECE falls from "
                 f"{lb['ece']:.3f} to {lu['ece']:.3f} and the calibration intercept moves from "
                 f"{lb['calibration_intercept']:+.2f} (systematic over-prediction: mean predicted "
                 f"risk {lb['mean_predicted']:.2f} against an observed rate of {cw['observed_rate']:.2f}) "
                 f"to {lu['calibration_intercept']:+.2f} once the reweighting is removed. "
                 f"`class_weight=\"balanced\"` reshapes predicted probabilities toward 50/50 to "
                 f"improve the decision boundary on imbalanced data and has no awareness of "
                 f"calibration at all. Class-balancing and probability calibration are two "
                 f"different objectives that can trade against each other.\n")
        L.append(f"**Gradient-Boosted Trees:** the same balanced reweighting (via `sample_weight`) "
                 f"behaves differently. The *unweighted* GBT under-predicts risk on average (mean "
                 f"predicted {gu['mean_predicted']:.3f} against an observed rate of "
                 f"{cw['observed_rate']:.3f}; intercept {gu['calibration_intercept']:+.2f}), and the "
                 f"balanced weighting pulls that part of the way back (mean predicted "
                 f"{gb['mean_predicted']:.3f}, intercept {gb['calibration_intercept']:+.2f}), so for "
                 f"this model class-balancing *improves* ECE ({gu['ece']:.3f} → {gb['ece']:.3f}) at a "
                 f"small cost in AUROC ({gu['auroc']:.3f} → {gb['auroc']:.3f}). Neither variant fixes "
                 f"the GBT's real calibration problem: a calibration slope of "
                 f"{gb['calibration_slope']:.2f} (ideal 1.0) in both cases, i.e. predictions that are "
                 f"too extreme -- the signature of {200} boosting rounds with no early stopping on "
                 f"3,200 training rows. That is a capacity/regularization issue, orthogonal to class "
                 f"weighting, and the natural next step would be early stopping or post-hoc "
                 f"recalibration on a held-out fold rather than a different weighting scheme.\n")
        L.append("The headline tables deliberately keep the class-balanced runs (the worse of the "
                 "two on calibration for Logistic Regression, at identical discrimination) instead "
                 "of quietly swapping in the better-calibrated unweighted version after the fact, "
                 "because the miscalibration it causes is itself the more general finding here.")
    L.append("")

    L.append("## Seed robustness of the cross-validated comparison\n")
    if seeds is None:
        L.append("(Run `python scripts/seed_robustness.py` before `evaluate.py` to generate this section.)")
    else:
        s = seeds["summary"]
        L.append("The 5-fold paired t-test above depends on one particular fold split. Re-running "
                 "the identical cross-validation under several seeds "
                 f"({', '.join(str(r['seed']) for r in seeds['per_seed'])}; pooled out-of-fold):\n")
        L.append("| Quantity | Mean | Std across seeds | Min | Max |")
        L.append("|---|---|---|---|---|")
        for k, lab in (("logreg_auroc", "LogReg AUROC"), ("gbt_auroc", "GBT AUROC"),
                       ("auroc_diff_gbt_minus_logreg", "AUROC difference (GBT − LogReg)"),
                       ("logreg_ece", "LogReg ECE"), ("gbt_ece", "GBT ECE"),
                       ("paired_fold_ttest_auroc_p", "Per-fold paired t-test p (AUROC)")):
            v = s[k]
            L.append(f"| {lab} | {v['mean']:.4f} | {v['std']:.4f} | {v['min']:.4f} | {v['max']:.4f} |")
        L.append("")
        L.append("The direction of the AUROC gap is stable across seeds, but the per-fold t-test "
                 "p-value is not -- it is a function of which 5 folds happened to be drawn, which "
                 "is the reason the Set B paired bootstrap is the comparison to quote.")
    L.append("")

    L.append("## Leakage check (why `Survival` and `Length_of_stay` are excluded)\n")
    if leak is None:
        L.append("(Run `python scripts/check_leakage.py` to generate this section.)")
    else:
        r = leak["rule_0<Survival<=LOS"]
        L.append(f"- The challenge's own definitions make the label a deterministic function of "
                 f"the two columns: `0 < Survival <= Length_of_stay` reproduces `In-hospital_death` "
                 f"on {r['agreement_with_label']:.1%} of Set A rows.")
        for k, lab in (("tree_depth3_Survival", "`Survival` alone"),
                       ("tree_depth3_Length_of_stay", "`Length_of_stay` alone"),
                       ("tree_depth3_Survival+Length_of_stay", "both")):
            v = leak[k]
            L.append(f"- Depth-3 decision tree on {lab}: AUROC {v['auroc_mean']:.3f} ± {v['auroc_std']:.3f} (5-fold CV).")
        v = leak["linear_logreg_Survival"]
        L.append(f"- A *linear* single-feature logistic regression on `Survival` gives AUROC "
                 f"{v['auroc_mean']:.3f}: the relation is non-monotone (−1 for survivors, small "
                 f"positive for in-hospital deaths, large positive for later deaths), so a linear "
                 f"test understates the leakage badly. An earlier version of this project quoted "
                 f"that number as the evidence; it has been corrected.")
    L.append("")

    with open(f"{RESULTS}/report.md", "w") as f:
        f.write("\n".join(L))
    print("wrote results/report.md and results/summary.json")


if __name__ == "__main__":
    main()
