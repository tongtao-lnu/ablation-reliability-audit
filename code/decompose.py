# -*- coding: utf-8 -*-
"""Detection--Delineation factorization of the macro-average Dice score.

Formalises the prototype ``decompose_check.py`` into a reusable module and
recomputes the factorization on **all existing predictions**:

* out-of-distribution (OOD): 16 cross-domain cells
  (4 transfer directions x 4 architectures, 8016 samples);
* in-distribution (ID): 20 groups
  (13 trained baselines + 7 ablation configurations, 242 samples each).

The factorization identity is

    E[Dice]     = Detection x Delineation
    Detection   = P(|P n G| > 0)                (frozen criterion, B6)
    Delineation = E[Dice | |P n G| > 0]

It is *exact* whenever Dice = 0 for every undetected sample.  The residual

    residual = E[Dice] - Detection x Delineation = sum_{undetected} Dice / n

is therefore analytically attributable rather than a fitting error.  On this
benchmark it is non-zero only because the metric implementations add a
smoothing constant eps (B6): Dice is ~eps > 0 even for an empty prediction.
Acceptance criterion for P2-1:

    max |residual| < 1e-6        (previous prototype run: 5.2e-10)

Boundary conditions (see ``00_docs/v4.2深化_形式化命题与边界条件_2026-09-13.md``, section 3):

    B1  the identity holds for the **macro**-average (per-sample mean) Dice only;
    B2  the detection factor depends on the binarization threshold tau (not
        re-scanned here; threshold sensitivity belongs to P2-2);
    B3  G != {} is required; the convention P = {} -> Dice = 0 is used explicitly;
    B4  cross-domain ranking reversal relies on a weak transferability
        assumption (not exercised by this module; relevant to Proposition 2);
    B5  the delineation factor is **not** pure boundary quality: it also absorbs
        over/under-segmentation and false-positive localization error;
    B6  the detection rate is a **function of the criterion**.  The frozen
        criterion |P n G| > 0 is eps-constructed and saturates at 1.000
        in-domain, so the substantive criterion I >= 1 px (== metric > 1e-6)
        must be reported alongside it.

Read-only with respect to the source predictions.  Every artefact is written to E:.

Usage::

    python decompose.py                 # run self-check, factorize, write outputs
    python decompose.py --selfcheck-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Frozen paths and constants
# --------------------------------------------------------------------------- #
OOD_CSV = (
    "E:/paper2_ablation_reliability/03_results/raw/ood_per_sample/paper2_per_sample.csv"
)
ID_ROOT = "D:/medical_segmentation/experiments"
OUT_CSV = "E:/paper2_ablation_reliability/03_results/stats/decomposition.csv"
OUT_JSON = "E:/paper2_ablation_reliability/03_results/stats/decomposition.json"
OUT_MD = "E:/paper2_ablation_reliability/03_results/stats/decomposition_report.md"

# Metric smoothing constants (B6).  ID-side implementations use 1e-8, OOD-side 1e-6.
EPS_ID = 1e-8
EPS_OOD = 1e-6

# Criterion thresholds.
FROZEN_THRESH = 0.0      # detected  <=>  metric > 0        (frozen, identity-bearing)
SUBSTANTIVE_THRESH = 1e-6  # detected  <=>  metric > 1e-6   (substantive, I >= 1 px)

# P2-1 acceptance tolerance on the identity residual.
RESIDUAL_TOL = 1e-6

# In-domain source inventory (folder names on D:, relative to ID_ROOT).
ID_BASELINE_ARCHS = [
    "AttentionUNet",
    "CaraNet",
    "M2SNet",
    "MultiResUNet",
    "PSPNet",
    "PolypPVT",
    "PraNet",
    "ResUNet",
    "SANet",
    "SegNet",
    "TransUNet",
    "UACANet",
    "UNet",
]
ID_ABLATION_CFGS = [
    "baseline",
    "egm_only",
    "dpa_only",
    "msfa_only",
    "egm_dpa",
    "egm_msfa",
    "dpa_msfa",
]

# Expected sample counts (hard facts; used as a sanity check, not as an assumption).
EXPECTED_OOD_N = 8016
EXPECTED_ID_N = 20 * 242


# --------------------------------------------------------------------------- #
# Core factorization
# --------------------------------------------------------------------------- #
def factorize(dice, metric, thr: float = FROZEN_THRESH) -> dict:
    """Factorize macro-average Dice given a per-sample detection metric.

    Parameters
    ----------
    dice : array_like
        Per-sample Dice in [0, 1].  ``Dice = 0`` must hold for an undetected
        sample (B3 convention: the empty prediction contributes 0, not 0/0).
    metric : array_like
        Per-sample detection metric: ``recall_gt`` on the OOD side (no eps, so a
        true 0 exists) or ``Recall`` / ``Dice`` on the ID side.
    thr : float
        Detection threshold.  ``thr = 0`` is the frozen criterion ``|P n G| > 0``;
        ``thr = 1e-6`` is the substantive criterion ``I >= 1 px`` (B6).

    Returns
    -------
    dict
        ``n``, ``mean_dice``, ``detection``, ``delineation``, ``product``,
        ``residual``, ``predicted_residual``, ``n_undetected``,
        ``rel_err_vs_predicted``.  ``delineation`` is NaN when nothing is
        detected (detection = 0), which cannot occur on this benchmark.
    """
    dice = np.asarray(dice, dtype=float).ravel()
    metric = np.asarray(metric, dtype=float).ravel()
    if dice.size != metric.size:
        raise ValueError(f"dice/metric length mismatch: {dice.size} vs {metric.size}")
    if dice.size == 0:
        raise ValueError("empty input")

    n = dice.size
    det_mask = metric > thr
    n_undet = int((~det_mask).sum())

    detection = float(det_mask.mean())
    if det_mask.any():
        delineation = float(dice[det_mask].mean())
    else:
        delineation = float("nan")
    product = detection * delineation
    mean_dice = float(dice.mean())
    residual = mean_dice - product

    # Analytic residual: the mass carried by undetected samples.  This is an
    # identity, not an approximation, so rel_err must be ~ 1e-16.
    predicted = float(dice[~det_mask].sum() / n)
    rel_err = abs(residual - predicted) / max(abs(residual), 1e-300)

    return dict(
        n=n,
        mean_dice=mean_dice,
        detection=detection,
        delineation=delineation,
        product=product,
        residual=residual,
        predicted_residual=predicted,
        n_undetected=n_undet,
        rel_err_vs_predicted=rel_err,
    )


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_ood(path: str = OOD_CSV) -> pd.DataFrame:
    """Load the cross-domain per-sample table (16 cells, 8016 rows)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"OOD per-sample table not found: {path}")
    d = pd.read_csv(path)
    required = {"dir", "arch", "dice", "recall_gt"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"OOD table missing columns: {sorted(missing)}")
    if len(d) != EXPECTED_OOD_N:
        raise ValueError(f"OOD table has {len(d)} rows, expected {EXPECTED_OOD_N}")
    return d


def load_id(folder: str) -> pd.DataFrame:
    """Load one in-domain ``sample_metrics.csv`` (242 rows).

    ``folder`` is relative to ``ID_ROOT``, e.g. ``baseline/UNet``.
    """
    path = os.path.join(ID_ROOT, folder, "sample_metrics.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"ID sample metrics not found: {path}")
    d = pd.read_csv(path)
    if "Dice" not in d.columns:
        raise ValueError(f"{path} has no 'Dice' column: {list(d.columns)}")
    return d


def id_metric_column(df: pd.DataFrame) -> tuple:
    """Pick the detection metric column for an in-domain table.

    Returns ``(column, source_label)``.  ``Recall`` (= |P n G| / |G|) is the
    direct realization of the frozen criterion; two baselines (MultiResUNet,
    PraNet) do not export it, so ``Dice`` is used as a proxy.  The proxy is
    validated empirically in :func:`validate_proxy` on the 18 groups that have
    both columns.
    """
    if "Recall" in df.columns:
        return "Recall", "Recall"
    return "Dice", "Dice (proxy)"


# --------------------------------------------------------------------------- #
# Self-check
# --------------------------------------------------------------------------- #
def selfcheck(verbose: bool = True) -> None:
    """Unit checks on synthetic inputs with known exact answers."""
    # (a) Fully detected, perfect delineation -> residual exactly 0.
    r = factorize([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    assert r["detection"] == 1.0 and r["delineation"] == 1.0
    assert r["residual"] == 0.0, r

    # (b) Half undetected with Dice = 0 on the undetected half -> residual 0 (B3).
    r = factorize([1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0])
    assert r["detection"] == 0.5 and r["delineation"] == 1.0, r
    assert r["product"] == 0.5 and r["residual"] == 0.0, r
    assert r["n_undetected"] == 2, r

    # (c) eps contamination: undetected samples carry Dice ~ eps -> residual
    #     equals their mean mass (this is the OOD regime).  Compared with a
    #     tolerance because summing 1.0 + eps loses low-order bits.
    eps = 1e-6
    r = factorize([1.0, 1.0, eps, eps], [1.0, 1.0, 0.0, 0.0])
    assert np.isclose(r["residual"], 2 * eps / 4, rtol=1e-8, atol=1e-15), r
    assert np.isclose(r["residual"], r["predicted_residual"], rtol=1e-8, atol=1e-15), r

    # (d) Any input: residual == sum_{undetected} Dice / n  exactly (up to
    #     floating-point summation order).
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(1, 500))
        dice = rng.random(n) * (rng.random(n) > 0.3)
        metric = rng.random(n) * (rng.random(n) > 0.3)
        r = factorize(dice, metric)
        assert np.isclose(r["residual"], r["predicted_residual"], rtol=1e-9, atol=1e-15), r

    # (e) Conventions.
    r = factorize([1.0, 0.0], [0.0, 0.0])
    assert r["detection"] == 0.0 and np.isnan(r["delineation"]), r  # nothing detected
    r = factorize([0.5], [1e-6], thr=SUBSTANTIVE_THRESH)           # strict inequality
    assert r["detection"] == 0.0, r
    r = factorize([0.5], [1e-6 * 1.0000001], thr=SUBSTANTIVE_THRESH)
    assert r["detection"] == 1.0, r
    r = factorize([0.5], [EPS_OOD])        # eps-only overlap still counts as frozen-detected
    assert r["detection"] == 1.0, r

    if verbose:
        print("[selfcheck] PASS  (a-e: exact residuals, analytic identity, conventions)")


# --------------------------------------------------------------------------- #
# Verification of the Dice-as-detection proxy
# --------------------------------------------------------------------------- #
def validate_proxy() -> list:
    """Check ``(Dice > 0) == (Recall > 0)`` on every group that has both.

    Justifies using ``Dice`` as the detection metric for the two baselines that
    do not export ``Recall``.
    """
    out = []
    for folder in ([f"baseline/{a}" for a in ID_BASELINE_ARCHS] + [
        f"ablation/{c}" for c in ID_ABLATION_CFGS
    ]):
        df = load_id(folder)
        if "Recall" not in df.columns:
            continue
        dis = int(((df["Dice"] > 0) != (df["Recall"] > 0)).sum())
        out.append(dict(group=folder, n=len(df), n_disagree=dis))
    return out


# --------------------------------------------------------------------------- #
# Convergence harness
# --------------------------------------------------------------------------- #
@dataclass
class Row:
    scope: str
    group_id: str
    n: int
    det_source: str
    mean_dice: float
    detection: float
    delineation: float
    product: float
    residual: float
    predicted_residual: float
    rel_err_vs_predicted: float
    n_undetected: int
    detection_sub: float
    delineation_sub: float
    product_sub: float
    residual_sub: float
    n_undetected_sub: int
    notes: str = ""


def _row(scope, gid, src, dice, metric, notes) -> Row:
    """Factorize one group under both criteria (frozen + substantive)."""
    fz = factorize(dice, metric, FROZEN_THRESH)
    fs = factorize(dice, metric, SUBSTANTIVE_THRESH)
    return Row(
        scope=scope,
        group_id=gid,
        n=fz["n"],
        det_source=src,
        mean_dice=fz["mean_dice"],
        detection=fz["detection"],
        delineation=fz["delineation"],
        product=fz["product"],
        residual=fz["residual"],
        predicted_residual=fz["predicted_residual"],
        rel_err_vs_predicted=fz["rel_err_vs_predicted"],
        n_undetected=fz["n_undetected"],
        detection_sub=fs["detection"],
        delineation_sub=fs["delineation"],
        product_sub=fs["product"],
        residual_sub=fs["residual"],
        n_undetected_sub=fs["n_undetected"],
        notes=notes,
    )


def compute() -> pd.DataFrame:
    """Factorize all OOD cells and all ID groups; return a tidy table."""
    rows: list[Row] = []

    # --- OOD: 4 directions x 4 architectures -------------------------------- #
    od = load_ood()
    for (dr, ar), g in od.groupby(["dir", "arch"], sort=True):
        rows.append(
            _row(
                "OOD", f"{dr}/{ar}", "recall_gt", g["dice"], g["recall_gt"],
                "recall_gt carries no eps: 0 is exact",
            )
        )

    # --- ID: 13 baselines + 7 ablation configurations ----------------------- #
    for a in ID_BASELINE_ARCHS:
        df = load_id(f"baseline/{a}")
        col, src = id_metric_column(df)
        rows.append(_row("ID", f"baseline/{a}", src, df["Dice"], df[col], ""))
    for c in ID_ABLATION_CFGS:
        df = load_id(f"ablation/{c}")
        col, src = id_metric_column(df)
        rows.append(_row("ID", f"ablation/{c}", src, df["Dice"], df[col], ""))

    df = pd.DataFrame([asdict(r) for r in rows])

    # --- Sanity checks ------------------------------------------------------ #
    n_ood = int(df.loc[df.scope == "OOD", "n"].sum())
    n_id = int(df.loc[df.scope == "ID", "n"].sum())
    if n_ood != EXPECTED_OOD_N or n_id != EXPECTED_ID_N:
        raise AssertionError(f"sample coverage mismatch: OOD={n_ood}, ID={n_id}")
    if len(df[df.scope == "OOD"]) != 16 or len(df[df.scope == "ID"]) != 20:
        raise AssertionError("expected 16 OOD cells and 20 ID groups")
    return df


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _summary(df: pd.DataFrame) -> dict:
    ood = df[df.scope == "OOD"]
    idf = df[df.scope == "ID"]
    id_sub = idf.detection_sub
    ood_sub = ood.detection_sub
    id_spread = float(id_sub.max() - id_sub.min())
    ood_spread = float(ood_sub.max() - ood_sub.min())
    gap = (df.residual - df.predicted_residual).abs()
    return dict(
        n_ood_cells=int(len(ood)),
        n_id_groups=int(len(idf)),
        ood_samples=int(ood.n.sum()),
        id_samples=int(idf.n.sum()),
        max_abs_residual=float(df.residual.abs().max()),
        max_abs_residual_ood=float(ood.residual.abs().max()),
        max_abs_residual_id=float(idf.residual.abs().max()),
        max_abs_gap_vs_analytic=float(gap.max()),
        tol=RESIDUAL_TOL,
        tol_pass=bool(df.residual.abs().max() < RESIDUAL_TOL),
        max_rel_err=float(df.rel_err_vs_predicted.max()),
        max_rel_err_cell=str(df.loc[df.rel_err_vs_predicted.idxmax(), "group_id"]),
        ood_detection_min=float(ood.detection.min()),
        ood_detection_max=float(ood.detection.max()),
        ood_detection_mean=float(ood.detection.mean()),
        ood_delineation_min=float(ood.delineation.min()),
        ood_delineation_max=float(ood.delineation.max()),
        id_detection_frozen_min=float(idf.detection.min()),
        id_detection_frozen_max=float(idf.detection.max()),
        id_detection_sub_min=float(id_sub.min()),
        id_detection_sub_max=float(id_sub.max()),
        id_detection_sub_spread=id_spread,
        ood_detection_sub_min=float(ood_sub.min()),
        ood_detection_sub_max=float(ood_sub.max()),
        ood_detection_sub_spread=ood_spread,
        id_saturated_sub=int((id_sub >= 1.0 - 1e-12).sum()),
        id_delineation_min=float(idf.delineation.min()),
        id_delineation_max=float(idf.delineation.max()),
        # Same-criterion contrast (B6): spread of the substantive detection rate.
        # This is the "22.9x" reading in the frozen protocol.
        contrast_same_criterion=(ood_spread / id_spread) if id_spread else float("inf"),
    )


def write_outputs(df: pd.DataFrame, s: dict, proxy: list, ood: pd.DataFrame) -> None:
    df.to_csv(OUT_CSV, index=False)
    payload = dict(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        script="02_code/analysis/decompose.py",
        criterion=dict(
            frozen="|P n G| > 0  (metric > 0)",
            substantive="I >= 1 px  (metric > 1e-6)",
            eps_id=EPS_ID,
            eps_ood=EPS_OOD,
        ),
        sources=dict(
            ood=OOD_CSV,
            id_root=ID_ROOT,
            id_baselines=ID_BASELINE_ARCHS,
            id_ablations=ID_ABLATION_CFGS,
        ),
        summary=s,
        proxy_validation=proxy,
        rows=df.to_dict("records"),
    )
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    write_markdown(df, s, proxy, ood)


def _fmt_table(rows: list, header: list, aligns: list) -> list:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(aligns) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return out


def write_markdown(df: pd.DataFrame, s: dict, proxy: list, ood: pd.DataFrame) -> None:
    o = df[df.scope == "OOD"]
    i = df[df.scope == "ID"]
    L: list = []
    L.append("# Detection--Delineation factorization of Dice — machine-generated report")
    L.append("")
    L.append(f"- **Generated:** {datetime.now().isoformat(timespec='seconds')}")
    L.append("- **Script:** `02_code/analysis/decompose.py` (formal module; supersedes `decompose_check.py`)")
    L.append(f"- **OOD source:** `{OOD_CSV}` — 16 cells, {s['ood_samples']} samples")
    L.append(f"- **ID source:** `{ID_ROOT}/{{baseline,ablation}}/*/sample_metrics.csv` — 20 groups, {s['id_samples']} samples")
    L.append("- **Criterion (B6):** frozen `|P n G| > 0` (identity-bearing) and substantive `I >= 1 px` (metric > 1e-6)")
    L.append("")
    L.append("## Verdict (P2-1 acceptance)")
    L.append("")
    L.append(f"- max |residual| = **{s['max_abs_residual']:.3e}**  (OOD {s['max_abs_residual_ood']:.3e}, ID {s['max_abs_residual_id']:.3e})")
    L.append(f"- tolerance = {s['tol']:.0e}  ->  **{'PASS' if s['tol_pass'] else 'FAIL'}**")
    L.append(f"- max |direct residual - analytic residual| = {s['max_abs_gap_vs_analytic']:.3e} (floating-point summation order)")
    L.append(f"- max relative error vs analytic residual = {s['max_rel_err']:.3e} (at `{s['max_rel_err_cell']}`; the residual there is ~1e-12,")
    L.append("  so the *relative* figure is amplified by cancellation while the absolute gap stays at machine precision)")
    L.append("- Interpretation: the residual is not a numerical error of the implementation; it is")
    L.append("  exactly the Dice mass carried by undetected samples, which is non-zero only because")
    L.append("  the metric implementations add a smoothing constant eps (B6).")
    L.append("")
    L.append("## A. Out-of-distribution cells (criterion: frozen, `recall_gt > 0`)")
    L.append("")
    rows = []
    for _, r in o.iterrows():
        rows.append([
            r["group_id"], r["n"], f"{r['mean_dice']:.4f}", f"{r['detection']:.4f}",
            f"{r['delineation']:.4f}", f"{r['product']:.4f}", f"{r['residual']:.3e}",
            r["n_undetected"], f"{r['detection_sub']:.4f}",
        ])
    L += _fmt_table(
        rows,
        ["cell", "n", "E[Dice]", "Detect", "Deline", "Det*Del", "residual", "n_undet", "Detect(sub)"],
        ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
    )
    L.append("")
    L.append("## B. In-distribution groups (criterion: frozen, `Recall > 0`)")
    L.append("")
    L.append("Dice is reported as a fraction; the in-domain literature reports percentages (x100).")
    L.append("")
    rows = []
    for _, r in i.iterrows():
        rows.append([
            r["group_id"], f"{r['det_source']}", r["n"], f"{r['mean_dice']*100:.2f}",
            f"{r['detection']:.4f}", f"{r['delineation']*100:.2f}", f"{r['product']*100:.2f}",
            f"{r['residual']:.3e}", f"{r['detection_sub']:.4f}",
        ])
    L += _fmt_table(
        rows,
        ["group", "metric", "n", "E[Dice] %", "Detect", "Deline %", "Det*Del %", "residual", "Detect(sub)"],
        ["---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
    )
    L.append("")
    L.append("## C. Summary and the criterion-dependence reading (B6)")
    L.append("")
    L.append(f"- OOD: Detection in [{s['ood_detection_min']:.3f}, {s['ood_detection_max']:.3f}], mean {s['ood_detection_mean']:.3f}; "
             f"Delineation in [{s['ood_delineation_min']:.3f}, {s['ood_delineation_max']:.3f}]")
    L.append(f"- ID: Detection = **1.000** for all {s['n_id_groups']} groups under the frozen criterion "
             f"(range [{s['id_detection_frozen_min']:.3f}, {s['id_detection_frozen_max']:.3f}])")
    L.append(f"- ID: Detection under the substantive criterion in "
             f"[{s['id_detection_sub_min']:.4f}, {s['id_detection_sub_max']:.4f}]; "
             f"**{s['id_saturated_sub']}/{s['n_id_groups']} groups saturated** "
             f"(spread {s['id_detection_sub_spread']:.4f})")
    L.append(f"- OOD: Detection under the same substantive criterion in "
             f"[{s['ood_detection_sub_min']:.4f}, {s['ood_detection_sub_max']:.4f}] "
             f"(spread {s['ood_detection_sub_spread']:.4f})")
    L.append(f"- ID Delineation in [{s['id_delineation_min']*100:.2f}, {s['id_delineation_max']*100:.2f}] %")
    L.append(f"- Same-criterion contrast of the detection spread (OOD / ID) = "
             f"**{s['contrast_same_criterion']:.1f}x**")
    L.append("")
    L.append("Reading: the in-domain detection factor is pinned at 1.000 by the frozen criterion, which is")
    L.append("constructed from the metric smoothing constant rather than from measured overlap. Reporting")
    L.append("only that value would leave the factor without discriminative power; the substantive reading")
    L.append("shows it is not saturated. Both readings must always be reported together (B6).")
    L.append("")
    L.append("## D. Validation of the Dice-as-detection proxy")
    L.append("")
    if proxy:
        mx = max(p["n_disagree"] for p in proxy)
        L.append(f"`(Dice > 0) == (Recall > 0)` on all {len(proxy)} groups that export both columns; "
                 f"max disagreement = {mx} samples.  This justifies using `Dice` as the detection")
        L.append("metric for the two baselines (MultiResUNet, PraNet) that do not export `Recall`.")
    L.append("")
    L.append("## E. Reproduce")
    L.append("")
    L.append("```bash")
    L.append("D:/miniconda/aniconda/envs/medical_seg/python.exe E:/paper2_ablation_reliability/02_code/analysis/decompose.py")
    L.append("```")
    L.append("")
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selfcheck-only", action="store_true",
                    help="run the unit self-check and exit")
    ap.add_argument("--no-write", action="store_true",
                    help="compute and print, but do not write artefacts")
    args = ap.parse_args(argv)

    selfcheck()
    if args.selfcheck_only:
        return 0

    df = compute()
    proxy = validate_proxy()
    s = _summary(df)

    pd.set_option("display.width", 200)
    print()
    print("=" * 100)
    print("A. OOD cells  (frozen criterion: recall_gt > 0)")
    print("=" * 100)
    o = df[df.scope == "OOD"][
        ["group_id", "n", "mean_dice", "detection", "delineation", "product",
         "residual", "n_undetected"]
    ].copy()
    for c in ["mean_dice", "detection", "delineation", "product"]:
        o[c] = o[c].map(lambda v: f"{v:.4f}")
    print(o.to_string(index=False, float_format=lambda v: f"{v:.3e}"))

    print()
    print("=" * 100)
    print("B. ID groups  (frozen criterion: Recall > 0)")
    print("=" * 100)
    i = df[df.scope == "ID"][
        ["group_id", "det_source", "n", "mean_dice", "detection", "delineation",
         "product", "residual", "detection_sub"]
    ].copy()
    for c in ["mean_dice", "detection", "delineation", "product", "detection_sub"]:
        i[c] = i[c].map(lambda v: f"{v:.6f}")
    print(i.to_string(index=False))

    print()
    print("=" * 100)
    print("C. Verdict")
    print("=" * 100)
    print(f"  OOD cells = {s['n_ood_cells']}   ID groups = {s['n_id_groups']}")
    print(f"  samples   = {s['ood_samples']} (OOD) + {s['id_samples']} (ID)")
    print(f"  max |residual| = {s['max_abs_residual']:.3e}   (tolerance {s['tol']:.0e})")
    print(f"  max |direct - analytic| = {s['max_abs_gap_vs_analytic']:.3e}   "
          f"max rel err = {s['max_rel_err']:.3e}")
    print(f"  ID detection: frozen = 1.000 (all {s['n_id_groups']}) | "
          f"substantive [{s['id_detection_sub_min']:.4f}, {s['id_detection_sub_max']:.4f}], "
          f"{s['id_saturated_sub']}/{s['n_id_groups']} saturated")
    print(f"  contrast (same criterion, OOD/ID detection spread) = {s['contrast_same_criterion']:.1f}x")
    print(f"  proxy validation: {(max(p['n_disagree'] for p in proxy) if proxy else 'n/a')} disagreements")
    print()
    print(f"  P2-1 ACCEPTANCE (max |residual| < {s['tol']:.0e}): "
          f"{'PASS' if s['tol_pass'] else 'FAIL'}")

    if not s["tol_pass"]:
        print("\n  !! residual exceeds tolerance — implementation problem, do not proceed.", file=sys.stderr)
        return 1

    if not args.no_write:
        write_outputs(df, s, proxy, load_ood())
        print()
        print(f"  wrote: {OUT_CSV}")
        print(f"  wrote: {OUT_JSON}")
        print(f"  wrote: {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
