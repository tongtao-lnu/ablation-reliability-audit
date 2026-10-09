# Detection--Delineation factorization of Dice — machine-generated report

- **Generated:** 2026-09-13T21:37:23
- **Script:** `02_code/analysis/decompose.py` (formal module; supersedes `decompose_check.py`)
- **OOD source:** `E:/paper2_ablation_reliability/03_results/raw/ood_per_sample/paper2_per_sample.csv` — 16 cells, 8016 samples
- **ID source:** `D:/medical_segmentation/experiments/{baseline,ablation}/*/sample_metrics.csv` — 20 groups, 4840 samples
- **Criterion (B6):** frozen `|P n G| > 0` (identity-bearing) and substantive `I >= 1 px` (metric > 1e-6)

## Verdict (P2-1 acceptance)

- max |residual| = **5.249e-10**  (OOD 5.249e-10, ID 0.000e+00)
- tolerance = 1e-06  ->  **PASS**
- max |direct residual - analytic residual| = 9.424e-17 (floating-point summation order)
- max relative error vs analytic residual = 1.158e-05 (at `kvasir_to_cvc/TransUNet`; the residual there is ~1e-12,
  so the *relative* figure is amplified by cancellation while the absolute gap stays at machine precision)
- Interpretation: the residual is not a numerical error of the implementation; it is
  exactly the Dice mass carried by undetected samples, which is non-zero only because
  the metric implementations add a smoothing constant eps (B6).

## A. Out-of-distribution cells (criterion: frozen, `recall_gt > 0`)

| cell | n | E[Dice] | Detect | Deline | Det*Del | residual | n_undet | Detect(sub) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cvc_to_etis/AttentionUNet | 196 | 0.3041 | 0.5663 | 0.5369 | 0.3041 | 2.973e-10 | 85 | 0.5663 |
| cvc_to_etis/EGAUNet | 196 | 0.5557 | 0.7908 | 0.7027 | 0.5557 | 1.042e-10 | 41 | 0.7908 |
| cvc_to_etis/TransUNet | 196 | 0.2080 | 0.5051 | 0.4118 | 0.2080 | 2.458e-10 | 97 | 0.5051 |
| cvc_to_etis/UNet | 196 | 0.3407 | 0.6888 | 0.4946 | 0.3407 | 2.378e-10 | 61 | 0.6888 |
| cvc_to_kvasir/AttentionUNet | 1000 | 0.3520 | 0.7670 | 0.4589 | 0.3520 | 4.137e-11 | 233 | 0.7670 |
| cvc_to_kvasir/EGAUNet | 1000 | 0.7704 | 0.9790 | 0.7869 | 0.7704 | 3.520e-12 | 21 | 0.9790 |
| cvc_to_kvasir/TransUNet | 1000 | 0.3870 | 0.8950 | 0.4324 | 0.3870 | 9.557e-12 | 105 | 0.8950 |
| cvc_to_kvasir/UNet | 1000 | 0.5149 | 0.9340 | 0.5513 | 0.5149 | 1.472e-11 | 66 | 0.9340 |
| kvasir_to_cvc/AttentionUNet | 612 | 0.4459 | 0.8480 | 0.5258 | 0.4459 | 3.862e-11 | 93 | 0.8480 |
| kvasir_to_cvc/EGAUNet | 612 | 0.7033 | 0.9379 | 0.7499 | 0.7033 | 1.383e-11 | 38 | 0.9379 |
| kvasir_to_cvc/TransUNet | 612 | 0.5037 | 0.9624 | 0.5233 | 0.5037 | 7.847e-12 | 23 | 0.9624 |
| kvasir_to_cvc/UNet | 612 | 0.5470 | 0.8954 | 0.6109 | 0.5470 | 2.982e-11 | 64 | 0.8954 |
| kvasir_to_etis/AttentionUNet | 196 | 0.3790 | 0.5765 | 0.6574 | 0.3790 | 5.249e-10 | 83 | 0.5765 |
| kvasir_to_etis/EGAUNet | 196 | 0.5525 | 0.7551 | 0.7317 | 0.5525 | 1.909e-10 | 48 | 0.7551 |
| kvasir_to_etis/TransUNet | 196 | 0.4499 | 0.7551 | 0.5958 | 0.4499 | 2.417e-10 | 48 | 0.7551 |
| kvasir_to_etis/UNet | 196 | 0.3862 | 0.6071 | 0.6361 | 0.3862 | 4.121e-10 | 77 | 0.6071 |

## B. In-distribution groups (criterion: frozen, `Recall > 0`)

Dice is reported as a fraction; the in-domain literature reports percentages (x100).

| group | metric | n | E[Dice] % | Detect | Deline % | Det*Del % | residual | Detect(sub) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline/AttentionUNet | Recall | 242 | 83.95 | 1.0000 | 83.95 | 83.95 | 0.000e+00 | 0.9876 |
| baseline/CaraNet | Recall | 242 | 83.16 | 1.0000 | 83.16 | 83.16 | 0.000e+00 | 0.9835 |
| baseline/M2SNet | Recall | 242 | 86.03 | 1.0000 | 86.03 | 86.03 | 0.000e+00 | 0.9752 |
| baseline/MultiResUNet | Dice (proxy) | 242 | 84.25 | 1.0000 | 84.25 | 84.25 | 0.000e+00 | 0.9793 |
| baseline/PSPNet | Recall | 242 | 86.46 | 1.0000 | 86.46 | 86.46 | 0.000e+00 | 0.9793 |
| baseline/PolypPVT | Recall | 242 | 86.38 | 1.0000 | 86.38 | 86.38 | 0.000e+00 | 0.9793 |
| baseline/PraNet | Dice (proxy) | 242 | 84.45 | 1.0000 | 84.45 | 84.45 | 0.000e+00 | 0.9711 |
| baseline/ResUNet | Recall | 242 | 79.19 | 1.0000 | 79.19 | 79.19 | 0.000e+00 | 0.9835 |
| baseline/SANet | Recall | 242 | 84.88 | 1.0000 | 84.88 | 84.88 | 0.000e+00 | 0.9793 |
| baseline/SegNet | Recall | 242 | 85.06 | 1.0000 | 85.06 | 85.06 | 0.000e+00 | 0.9835 |
| baseline/TransUNet | Recall | 242 | 83.91 | 1.0000 | 83.91 | 83.91 | 0.000e+00 | 0.9917 |
| baseline/UACANet | Recall | 242 | 83.33 | 1.0000 | 83.33 | 83.33 | 0.000e+00 | 0.9711 |
| baseline/UNet | Recall | 242 | 82.38 | 1.0000 | 82.38 | 82.38 | 0.000e+00 | 0.9793 |
| ablation/baseline | Recall | 242 | 81.97 | 1.0000 | 81.97 | 81.97 | 0.000e+00 | 0.9752 |
| ablation/egm_only | Recall | 242 | 86.67 | 1.0000 | 86.67 | 86.67 | 0.000e+00 | 0.9876 |
| ablation/dpa_only | Recall | 242 | 86.65 | 1.0000 | 86.65 | 86.65 | 0.000e+00 | 0.9876 |
| ablation/msfa_only | Recall | 242 | 87.46 | 1.0000 | 87.46 | 87.46 | 0.000e+00 | 0.9835 |
| ablation/egm_dpa | Recall | 242 | 86.96 | 1.0000 | 86.96 | 86.96 | 0.000e+00 | 0.9917 |
| ablation/egm_msfa | Recall | 242 | 88.32 | 1.0000 | 88.32 | 88.32 | 0.000e+00 | 0.9917 |
| ablation/dpa_msfa | Recall | 242 | 87.12 | 1.0000 | 87.12 | 87.12 | 0.000e+00 | 0.9876 |

## C. Summary and the criterion-dependence reading (B6)

- OOD: Detection in [0.505, 0.979], mean 0.779; Delineation in [0.412, 0.787]
- ID: Detection = **1.000** for all 20 groups under the frozen criterion (range [1.000, 1.000])
- ID: Detection under the substantive criterion in [0.9711, 0.9917]; **0/20 groups saturated** (spread 0.0207)
- OOD: Detection under the same substantive criterion in [0.5051, 0.9790] (spread 0.4739)
- ID Delineation in [79.19, 88.32] %
- Same-criterion contrast of the detection spread (OOD / ID) = **22.9x**

Reading: the in-domain detection factor is pinned at 1.000 by the frozen criterion, which is
constructed from the metric smoothing constant rather than from measured overlap. Reporting
only that value would leave the factor without discriminative power; the substantive reading
shows it is not saturated. Both readings must always be reported together (B6).

## D. Validation of the Dice-as-detection proxy

`(Dice > 0) == (Recall > 0)` on all 18 groups that export both columns; max disagreement = 0 samples.  This justifies using `Dice` as the detection
metric for the two baselines (MultiResUNet, PraNet) that do not export `Recall`.

## E. Reproduce

```bash
D:/miniconda/aniconda/envs/medical_seg/python.exe E:/paper2_ablation_reliability/02_code/analysis/decompose.py
```
