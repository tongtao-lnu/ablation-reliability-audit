# ablation-reliability-audit

Analysis artifacts for the manuscript

> **What in-distribution ablation actually measures: detection–delineation decomposition and
> construct mismatch in polyp segmentation benchmarks**

Tao Tong, Jiahui Li, Wanni Zu (corresponding author) — School of Physics, Liaoning University, Shenyang, China.

This repository is the public deposit referenced by the *Data availability* and *Code availability*
statements of the manuscript. It contains exactly the three items promised there:

| Directory | Contents |
|---|---|
| `analysis_plan/` | `ANALYSIS_PLAN.md` — the analysis plan fixed before the analyses were run: frozen evaluation choices, statistical plan, pre-specified hypotheses with their decision rules, capacity-control plan, sensitivity analyses and boundary conditions. |
| `per_image_metrics/` | Per-image and per-sample metric tables (CSV) that every reported contrast is computed from. |
| `derived_statistics/` | Machine-readable registries (JSON) holding the aggregate statistics, intervals, p-values and audit quantities quoted in the manuscript. |
| `code/` | The analysis code that turns the per-image tables into the derived statistics. |

`MANIFEST.sha256` lists the SHA-256 of every deposited file.

---

## 1. What the study evaluates

Three public polyp benchmarks are used: **Kvasir-SEG** (1,000 image–mask pairs), **CVC-ClinicDB**
(612) and **ETIS-LaribPolypDB** (196) — 1,808 pairs in total.

* **In-domain (source):** the union of Kvasir-SEG and CVC-ClinicDB, 1,612 pairs, pooled without
  regard to origin and re-split 70/15/15 under one fixed seed (1,128 train / 242 validation /
  242 test). All in-domain quantities are reported on the **242-image test split**
  (143 Kvasir-SEG + 99 CVC-ClinicDB).
* **Cross-domain (target):** ETIS-LaribPolypDB, **never used for training**, 196 images.
* **Transfer directions:** Kvasir→CVC (n = 612), CVC→Kvasir (n = 1,000), Kvasir→ETIS (n = 196),
  CVC→ETIS (n = 196). The two ETIS directions aggregate to the 16 cross-domain (model × direction)
  cells used in the decomposition.
* **Units:** 14 architecture-level units, 7 ablation configurations (one later-trained configuration
  appears descriptively only), 16 out-of-domain cells (8,016 samples) and 20 in-domain groups
  (4,840 samples).

## 2. Metric decomposition

For an image with reference mask G and prediction P, macro-average Dice factorises exactly as

```
S = D × Q,   D = 1[|P ∩ G| > 0]   (detection factor),
             Q = 2|P ∩ G| / (|P| + |G|)   (delineation factor)
```

with the convention Dice = 0 when P is empty, and the requirement G ≠ ∅. The identity holds for
macro-average (per-image mean) Dice and fails for micro-average (pooled-union) Dice. See
`analysis_plan/ANALYSIS_PLAN.md` §2 and §7 for the eight boundary conditions.

## 3. How the tables were produced

The per-image tables are the inputs; the code in `code/` computes every derived quantity:

1. Predictions are produced by the frozen models and reduced to per-image overlap counts
   (|P ∩ G|, |P|, |G|). Detection is read under a **frozen** criterion (|P ∩ G| > 0) and, for
   comparison, under a **substantive** criterion (at least one overlapping pixel).
2. `code/decompose.py` and `code/decompose_check.py` verify the identity numerically.
3. `code/p2_*_crosscheck.py`, `code/p31*_*.py`, `code/confound_control.py`,
   `code/alignment_test.py`, `code/variance_attribution*.py` (see the directory for the full set)
   produce the aggregate statistics, intervals and tests.
4. The JSON registries in `derived_statistics/` are the direct output of step 3 and are the files
   cited field-by-field in the manuscript.

### Paths

The scripts were written for the authors' local layout and contain absolute paths
(`D:/medical_segmentation/...`, `E:/paper2_ablation_reliability/...`). To re-run them, either
recreate that layout or set the path constants at the top of each script. The model definitions
and training code are **not** part of this deposit; the deposit covers the analysis code, as stated
in the manuscript.

### Environment

Python 3.9 with `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib` and `statsmodels`
(`statsmodels` supplies `multipletests` for the Holm correction). No GPU is required to re-derive
the statistics from the deposited tables.

## 4. Data

The three benchmark datasets are not redistributed here; they are publicly available:

* Kvasir-SEG — https://datasets.simula.no/kvasir-seg/
* CVC-ClinicDB — https://polyp.grand-challenge.org/CVCClinicDB/
* ETIS-LaribPolypDB — https://polyp.grand-challenge.org/ETISLarib/

Please cite the original dataset papers if you use them.

## 5. Citation

If you use these artifacts, please cite the manuscript above. A machine-readable `CITATION.cff`
will be added once the manuscript has a DOI.

## 6. Contact

Please open an issue in this repository, or contact the corresponding author, Wanni Zu
(School of Physics, Liaoning University).

## 7. License

No licence is granted for reuse at this stage. The files are deposited so that the reported
numbers can be inspected and reproduced; please contact the corresponding author before reusing
the code in another work.
