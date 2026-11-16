# Data provenance

## Source

Real patient records from the **PhysioNet/Computing in Cardiology Challenge
2012** ("Predicting Mortality of ICU Patients"): 12,000 ICU stays split by
the challenge organizers into Set A (4,000, outcomes public), Set B (4,000,
outcomes public), and Set C (4,000, outcomes never publicly released, and
not used here). Each variable (heart rate, blood pressure, GCS, labs, etc.)
was recorded as an irregular time series over the ICU stay and is provided
here pre-aggregated per stay (first / last / lowest / highest / median),
along with demographics, ICU admission type, and standard clinical severity
scores (SAPS-I, SOFA).

**This project does not have access to the raw per-timestamp records.**
`physionet.org` itself is not reachable from this sandbox (outbound network
policy blocks it, confirmed via a direct connection test, not assumed).
The files used here (`PhysionetChallenge2012-set-a.csv.gz`,
`PhysionetChallenge2012-set-b.csv.gz`) are the same real challenge data,
retrieved instead from the public GitHub mirror maintained by Alistair
Johnson (a PhysioNet/MIMIC co-creator and the 2012 challenge's top-scoring
entrant): <https://github.com/alistairewj/challenge2012>, via `git clone`
(git access was not blocked, only the direct `physionet.org` HTTPS
connection was). That repository's README explicitly offers these files
"for convenience" and requests the citations reproduced below. This is
disclosed plainly rather than describing the data as "downloaded from
PhysioNet directly."

**Scope limitation, stated up front:** because the data here is already
aggregated to one row per stay, this project evaluates a per-stay tabular
risk model, not a sequence model over raw irregular time series. It should
not be described as time-series/sequence modeling or as foundation-model
work; see `README.md` Limitations for what it does and does not
demonstrate.

## Licensing and required citations

The underlying PhysioNet Challenge 2012 dataset is released under the
**Open Data Commons Attribution License v1.0**; the mirror repository's own
parsing code is Apache-2.0 (`PHYSIONET_DATA_LICENSE.txt` in this folder is
that repo's LICENSE file). Per the source repository's README, reuse should
cite:

```bibtex
@article{silva2012predicting,
  title={Predicting in-hospital mortality of icu patients: The physionet/computing in cardiology challenge 2012},
  author={Silva, Ikaro and Moody, George and Scott, Daniel J and Celi, Leo A and Mark, Roger G},
  journal={Computing in cardiology},
  volume={39}, pages={245}, year={2012}, publisher={NIH Public Access}
}
@inproceedings{johnson2012patient,
  title={Patient specific predictions in the intensive care unit using a Bayesian ensemble},
  author={Johnson, Alistair EW and Dunkley, Nic and Mayaud, Louis and Tsanas, Athanasios and Kramer, Andrew A and Clifford, Gari D},
  booktitle={Computing in Cardiology (CinC), 2012}, pages={249--252}, year={2012}, organization={IEEE}
}
```

## Excluded columns (checked empirically, not assumed)

`scripts/check_leakage.py` tests the two outcome-adjacent columns
directly rather than trusting the field description (full numbers in
`results/leakage_check.json` and the leakage section of
`results/report.md`):

- **`Survival`** (days from ICU admission to death for patients who died
  -- in hospital *or* after discharge -- and **-1** for patients alive at
  follow-up, by the challenge's own definition). Together with
  `Length_of_stay` it determines the label exactly: the rule
  `0 < Survival <= Length_of_stay` reproduces `In-hospital_death` on
  **100.0%** of Set A rows. A depth-3 decision tree on `Survival` alone
  reaches AUROC **0.988** (5-fold CV). **Excluded.**
- **`Length_of_stay`**: a depth-3 tree on this column alone gives AUROC
  **0.572** -- weak but real univariate signal, and more importantly it
  is one half of the deterministic rule above (for a patient who died,
  stay length is mechanically bounded by time of death). **Excluded.**

**Correction to an earlier version of this document.** The first version
of the leakage check used a *linear* single-feature logistic regression
and reported AUROC 0.259 for `Survival` and 0.495 for `Length_of_stay`,
describing the former as "confirming" leakage and the latter as "no
meaningful signal". Both readings were wrong, for the same reason: the
relation between `Survival` and the label is non-monotone (-1 for
survivors, small positive values for in-hospital deaths, large positive
values for patients who died later), which a linear logit cannot
represent, so that test badly understates the leakage. The conclusion
(exclude both) was right; the evidence quoted for it was not, and the
script now runs the deterministic-rule and tree checks and keeps the
linear numbers only to document why they misled.

`recordid` (an identifier) is dropped as a feature for the obvious reason.
`SAPS-I` and `SOFA` (clinical severity scores computed from the same early
ICU data, not from the outcome) are **kept** as features and also reported
separately as a fitting-free clinical baseline (`SAPS-I` alone) -- this
mirrors what the original 2012 challenge asked entrants to beat, and is a
legitimate prospective clinical score, not outcome leakage.

## Derived column

`icu_type` (CCU / CSRU / SICU / MICU) is derived from the three binary
`CCU`/`CSRU`/`SICU` flag columns; all-zero is treated as MICU (Medical
ICU), the challenge's fourth admission-type category, per the challenge
documentation.
