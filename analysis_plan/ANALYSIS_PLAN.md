# Analysis plan

*Fixed before the analyses were run.*

This document states the analysis plan for the manuscript

> **What in-distribution ablation actually measures: detection–delineation decomposition and
> construct mismatch in polyp segmentation benchmarks**

Tao Tong, Jiahui Li, Wanni Zu (corresponding author) — School of Physics, Liaoning University.

It is the "analysis plan" item promised by the *Data availability* and *Code availability*
statements. It records the design, the frozen evaluation choices, the statistical plan, the
pre-specified hypotheses and their decision rules, and the boundary conditions under which the
decomposition is claimed to hold. **Outcomes are reported in the manuscript; this document is
deliberately limited to what was fixed in advance.**

*Provenance.* The plan was frozen in the project's protocol document v4.2.1 (13 September 2026)
and is restated here in English together with the frozen parameter values recorded in the
deposited registries (`derived_statistics/confound.json::calibers.frozen_before_run`). No
decision recorded here was changed after an outcome was seen.

---

## 1. Design

Three public polyp benchmarks: Kvasir-SEG (1,000 image–mask pairs), CVC-ClinicDB (612) and
ETIS-LaribPolypDB (196); 1,808 pairs in total.

* **In-domain (source).** The union of Kvasir-SEG and CVC-ClinicDB, 1,612 pairs, pooled without
  regard to origin and re-split 70/15/15 under one fixed seed (1,128 train / 242 validation /
  242 test). Reported in-domain quantities use the **242-image test split** (143 Kvasir-SEG +
  99 CVC-ClinicDB).
* **Cross-domain (target).** ETIS-LaribPolypDB, **never used for training**, n = 196.
* **Transfer directions.** Kvasir→CVC (n = 612), CVC→Kvasir (n = 1,000), Kvasir→ETIS (n = 196),
  CVC→ETIS (n = 196). The two ETIS directions aggregate to **16 cross-domain (model × direction)
  cells** used in the decomposition.
* **Two families, never pooled.** (i) the **ablation grid**: seven configurations of the modules
  EGM, DPA and MSFA on a common backbone — all-off, the three single-module and the three
  two-module configurations; a paired comparison of seven configurations, *not* a complete
  three-factor design. (ii) the **architecture grid**: 13 public baselines plus the method under
  study, n = 14 architecture-level units. The two families use different protocols and are never
  merged in any regression or contrast.
* **Units of analysis:** 14 architecture-level units, 7 ablation configurations, 16 out-of-domain
  cells (8,016 samples) and 20 in-domain groups (4,840 samples).
* **One later-trained configuration** (all modules on) has no per-sample in-domain records; it is
  a descriptive anchor only and is excluded from paired analyses.

## 2. Primary quantities

For an image with reference mask `G` and prediction `P`, macro-average (per-image mean) Dice is

```
S = D × Q
D = 1[ |P ∩ G| > 0 ]                        detection factor
Q = 2 |P ∩ G| / ( |P| + |G| )               delineation factor
```

with the conventions that Dice = 0 when `P` is empty, and that `G ≠ ∅` is required. The
decomposition holds for macro-average Dice; for micro-average (pooled-union) Dice the two factors
do not separate.

Detection is read under two criteria, and each reading must be reported with its criterion:

* the **frozen** criterion `|P ∩ G| > 0`, which is required by the identity (Dice = 0 ⟺ no
  overlap);
* the **substantive** criterion `|P ∩ G| ≥ 1` pixel, which is not affected by the numerical
  smoothing constant used by the metric implementations.

## 3. Frozen evaluation choices

| Item | Frozen value |
|---|---|
| Pixel binarisation threshold τ | 0.5 (anchor); sensitivity over a 21-point grid τ ∈ {0.01, 0.05, 0.10, …, 0.95, 0.99} |
| Numerical smoothing constant | ID side ε = 10⁻⁸; OOD side ε = 10⁻⁶ |
| Detection criteria | frozen `|P ∩ G| > 0`; substantive `|P ∩ G| ≥ 1` px |
| Significance level α | 0.05 |
| Bootstrap resamples B | 10,000 |
| Bootstrap seed | 20260914 |
| Multiplicity | Holm, within each pre-specified family |
| Omnibus ordering test | Friedman |
| Capacity parameter bins | [30, 35), [35, 42), [42, 48) million parameters |
| Capacity-matching tolerance | 5 % |
| In-domain gate for capacity controls | 1.0 percentage point |

The capacity-matching branch carries its own frozen bootstrap setting (2,000 resamples, seed 42),
recorded in `derived_statistics/capacity_matched.json::frozen`; the main uncertainty intervals in
§4 use the 10,000-resample setting above.

Degradations in the stress test are applied to the **input image only**; model weights are frozen
and are never retrained, so any change is attributable to the input distribution rather than to
adaptation. The stress test uses the pixel threshold 0.5 and reads detection as at least one
overlapping pixel, and its statistical treatment is descriptive (the 14 architectures are not
independent, so any interval is descriptive only).

## 4. Statistical plan

All models are scored on the **same** images — the 242-image in-domain test set and the 196-image
cross-domain set — so every contrast is a **per-image paired** contrast and the per-image metric
tables are the input.

* **Paired test.** Two-sided **Wilcoxon signed-rank** test on per-image Dice differences.
* **Multiplicity.** **Holm** correction within each pre-specified family.
* **Uncertainty.** 10,000-resample **bootstrap percentile** interval with the fixed seed above;
  paired-contrast intervals resample images.
* **Omnibus ordering** within a family: **Friedman** test.
* **Pooled module effects** on cross-domain Dice are reported with their per-pair Wilcoxon and
  Holm p-values.
* The two model families are reported separately and **never pooled**.
* Where a contrast rests on very few units, the smallest attainable two-sided p is reported
  alongside the p-value, and a conclusion is licensed only if the design can in principle support
  it.

## 5. Pre-specified hypotheses and decision rules

The following thresholds were specified before evaluation. Any hypothesis that is refuted is
reported as refuted.

| # | Hypothesis | Refuted if |
|---|---|---|
| **H1** | In-domain detection is saturated: identically 1.000 under the frozen criterion, and 0.971–0.992 under the substantive criterion, across all architectures × all module combinations. | Any combination falls below 0.98 under the **substantive** criterion; in that case the affected combination is recorded as detection-available and the in-domain claim is weakened to "near-degenerate". |
| **H2** | Cross-domain detection accounts for at least 35 % of the variance of log Dice. | The share is at most 20 %; the framing then shifts to "delineation-dominated". |
| **H3** | In-domain delineation and cross-domain detection are not associated (architecture level, n = 14). Independence is rejected only when the association is **both** significant **and** \|ρ\| > 0.6. | Significant with \|ρ\| > 0.6. Because the smallest detectable \|ρ\| at n = 14 is 0.72 while the rejection threshold is 0.6, values in 0.538 < \|ρ\| ≤ 0.6 are reported as **undecidable** — neither supporting nor refuting. |
| **H4** | The flagship rank reversal reproduces across seeds. | The reversal disappears or flips sign. |
| **H5** | The cross-domain detection advantage of EGM is not explained by parameter count. | The advantage disappears after capacity matching. |

**Pre-specified one-number diagnostic.** The rank association between in-domain delineation and
cross-domain detection is reported as a screening quantity with an interpretation rule fixed in
advance: clearly positive → the ablation is usable; approximately zero → the ablation carries no
information about detection; clearly negative → the ablation is misleading in direction. The
diagnostic is reported as a mechanism-consistent direction only, and not as a result, whenever the
available unit count leaves the smallest attainable two-sided p above α.

## 6. Capacity-control plan

The central alternative explanation for a cross-domain detection advantage is **capacity** — a
module that simply adds parameters. Three devices were pre-specified before the outcome was seen:

1. **Stratification by module count.** Configurations are grouped by active-module count (0, 1, 2)
   so that within-stratum comparisons hold it constant (3 strata).
2. **Parameter regression.** Rank correlations between parameter count and each metric are computed
   separately within each protocol, and within the middle parameter bin the
   parameter-versus-detection ordering is tested for monotonicity.
3. **Capacity matching.** A plain encoder–decoder baseline is widened until its parameter count
   lands within the pre-specified **5 %** tolerance of the configuration under test, and a
   **5 percentage-point** cross-domain detection gap to that parameter-matched control is treated
   as evidence **against** a capacity explanation.

The decision protocol for the capacity branch was pre-registered and frozen before the matching
runs.

## 7. Sensitivity and robustness analyses

* **Threshold sensitivity.** A 21-point binarisation-threshold scan of the architecture ranking,
  reported as rank stability.
* **Multi-seed reproduction.** The key ablation cells are re-trained under three seeds; the
  reproduction of the rank reversal is the H4 test.
* **Gate-closing stress test.** 242 in-domain images × 14 architectures under 16 conditions: one
  clean plus five degradation families (blur, noise, JPEG, darkening, specular highlights) at three
  severities each. Weights are unchanged and only the input is perturbed.
* **Connected-component / fragmentation analysis** of the predicted masks, with an explicit
  comparison of the full-mask and visible-only readings.

## 8. Boundary conditions

| # | Boundary condition |
|---|---|
| **B1** | The decomposition holds only for macro-average (per-image mean) Dice; micro-average Dice does not separate. |
| **B2** | The detection factor depends on the binarisation threshold τ (the delineation factor depends on it only weakly), so τ must be reported and a threshold sensitivity scan given. |
| **B3** | `G ≠ ∅` is required; Dice is defined as 0 when `P` is empty, and that convention is stated explicitly. |
| **B4** | The sufficient condition for a rank reversal relies on the weaker assumption that delineation transfers across domains; that assumption must be declared and tested, and if it fails the result degrades to a one-directional statement. |
| **B5** | The delineation factor is not "pure boundary quality": it also absorbs over-segmentation, under-segmentation and false-positive localisation. |
| **B6** | Saturation of in-domain detection is criterion-dependent. Both metric implementations carry a smoothing constant (ε = 10⁻⁸ in-domain, 10⁻⁶ cross-domain) that makes Dice and recall strictly positive for any prediction, including an empty one. A detection rate of 1.000 under the frozen criterion is therefore a consequence of ε and not a measurement of overlap. **Both readings must always be given**: frozen criterion → no information; substantive criterion → the informative one. The frozen definition is retained because the identity requires it. The identity residual of order 10⁻¹⁰ is analytically attributable to the same ε, not to floating-point noise. |

## 9. Reporting rules

* Every reported number is traceable to a file and field in this deposit.
* Detection is never reported without its criterion.
* The two model families are never pooled.
* Non-significance is never reported as evidence of no effect; where the design cannot support a
  conclusion, that is stated.
* No absolute guarantees are made about future performance.
* Statistical reporting follows the journal's conventions for paired tests, effect sizes and
  multiplicity.

## 10. What is in this deposit

| Path | Contents |
|---|---|
| `analysis_plan/ANALYSIS_PLAN.md` | this document |
| `per_image_metrics/` | per-image and per-sample metric tables (CSV) |
| `derived_statistics/` | machine-readable aggregate statistics (JSON) quoted in the manuscript |
| `code/` | the analysis code that produces `derived_statistics/` from `per_image_metrics/` |
| `MANIFEST.sha256` | SHA-256 of every deposited file |
