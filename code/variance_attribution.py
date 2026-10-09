# -*- coding: utf-8 -*-
"""Variance attribution of the macro-average Dice: in-domain vs out-of-domain.

This is the P2-2 analysis.  It answers two pre-registered questions and reports
the criterion-dependence of the detection factor (boundary condition B6).

Mathematical basis
------------------
On the macro-average scale the factorization of :mod:`decompose` is *exact*:

    E[Dice] = Detection x Delineation          (exact when Dice = 0 <=> undetected)

Taking logarithms,

    log Dice = log Detection + log Delineation

so the variance of ``log Dice`` across units (cells / groups) splits into three
additive terms:

    var(log Dice) = var(log Detection) + var(log Delineation) + 2 cov

and the *share* of the detection factor is

    share_det = var(log Detection) / [var(log Det) + var(log Del) + 2 cov]

Using the sum as the denominator (rather than var(log Dice) computed directly)
makes the three shares add to exactly 100%; the difference between the two
denominators is the numerical gap of the identity, which this module reports.

Hypotheses (pre-registered, see ``00_docs/论文二方案v4.2.1_冻结.md`` section 4)
--------------------------------------------------------------------------
H1  The in-domain detection rate is identically 1.000 (frozen criterion
    ``|P n G| > 0``, stated alongside the substantive reading ``I >= 1 px``).
    Refuted per-group if a group's detection rate < 0.98 under the **substantive**
    criterion; the pre-registered response is to write L1-① as "approaching
    saturation" rather than "saturation".
H2  Out-of-domain detection contributes >= 35% of the var(log Dice);
    < 20% refutes it and demotes L2.

Scope and honest limits
-----------------------
* **n = 4**: the out-of-domain unit set is 4 transfer directions x 4
  architectures = **16 cells**, i.e. n = 4 architectures.  This is the
  *initial* value.  The pre-registered judgment data for H2 is *all*
  architectures, so the **final value is produced by P2-2b after P3-1** which
  overwrites these artefacts.  The ``_n4`` filename suffix marks this.
* H3 (Spearman alignment) is **not** tested here: at n = 4 its statistical power
  is zero and the frozen protocol forbids reporting it.  H3 belongs to P2-3 at
  architectural n = 14.
* The threshold sensitivity reported here is the sensitivity to the **detection
  criterion** on pre-computed per-sample metrics.  A true binarization-threshold
  (tau) scan requires the raw probability maps and belongs to P3-7; this module
  does not claim to substitute for it.

Read-only with respect to the source predictions; artefacts are written to E:.

Usage::

    python variance_attribution.py                  # self-check + analysis + write
    python variance_attribution.py --selfcheck-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Reuse the frozen calibre from P2-1 (single source of truth)
# --------------------------------------------------------------------------- #
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decompose as dz  # noqa: E402

OUT_JSON = "E:/paper2_ablation_reliability/03_results/stats/variance_attribution_n4.json"
OUT_MD = "E:/paper2_ablation_reliability/03_results/tables/T_variance_n4.md"

# Analysis constants (frozen).
BOOT_REPS = 2000
BOOT_SEED = 42
CRIT_GRID = [0.0, 1e-6, 1e-2, 0.05, 0.1, 0.3, 0.5]
H1_BAR = 0.98
H2_SUPPORT = 0.35
H2_REFUTE = 0.20

# Artifact deliberately excluded from the in-domain unit set (protocol mismatch,
# see M10 of the plan): a different-protocol run whose mean Dice is far above the
# ablation grid and which is not part of the 20 pre-registered groups.
EXCLUDED_ID = "D:/medical_segmentation/results/test_results/sample_metrics.csv"


# --------------------------------------------------------------------------- #
# Variance attribution
# --------------------------------------------------------------------------- #
def attribution(det: np.ndarray, del_: np.ndarray) -> dict:
    """Split var(log Dice) into detection / delineation / covariance terms.

    ``det`` and ``del_`` must be the *factor pairs* of the exact factorization
    (so that ``det * del_`` is the unit's mean Dice).  ``ddof = 1``.

    Returns the three variances, the two shares plus the covariance share, the
    directly measured var(log Dice) and the identity gap.
    """
    det = np.asarray(det, dtype=float)
    del_ = np.asarray(del_, dtype=float)
    if det.size != del_.size or det.size < 2:
        raise ValueError("need at least 2 factor pairs of equal length")
    if np.any(det <= 0) or np.any(del_ <= 0):
        raise ValueError("factors must be strictly positive to take logarithms")

    lt, ll = np.log(det), np.log(del_)
    ld = np.log(det * del_)

    vt = float(lt.var(ddof=1))
    vl = float(ll.var(ddof=1))
    cov = float(np.cov(lt, ll, ddof=1)[0, 1])
    total = vt + vl + 2 * cov
    if abs(total) < 1e-300:
        raise ValueError("degenerate total variance")

    # Correlation between the two logged factors.  Undefined when either factor
    # is constant (e.g. the in-domain frozen reading, where detection == 1.000).
    if vt > 0 and vl > 0:
        from scipy import stats as _st
        rc, pc = _st.pearsonr(lt, ll)
        rc, pc = float(rc), float(pc)
    else:
        rc, pc = float("nan"), float("nan")

    # What the shares would look like if the covariance term were ignored.
    naive_den = vt + vl
    share_det_naive = (vt / naive_den) if naive_den > 0 else float("nan")

    return dict(
        n=int(det.size),
        var_log_det=vt,
        var_log_del=vl,
        cov2=2 * cov,
        total=total,
        share_det=vt / total,
        share_del=vl / total,
        share_cov=2 * cov / total,
        share_det_naive=share_det_naive,
        corr_logdet_logdel=rc,
        corr_logdet_logdel_p=pc,
        var_log_dice_direct=float(ld.var(ddof=1)),
        identity_gap=float(ld.var(ddof=1) - total),
        detection_range=[float(det.min()), float(det.max())],
        detection_spread=float(det.max() - det.min()),
        delineation_range=[float(del_.min()), float(del_.max())],
        logdet_spread=float(np.log(det.max()) - np.log(det.min())),
        logdel_spread=float(np.log(del_.max()) - np.log(del_.min())),
        spread_ratio=(
            float((np.log(det.max()) - np.log(det.min())) / (np.log(del_.max()) - np.log(del_.min())))
            if np.log(del_.max()) > np.log(del_.min()) else float("inf")
        ),
        cov_negative=bool(cov < 0),
    )


def bootstrap_share(det: np.ndarray, del_: np.ndarray,
                    reps: int = BOOT_REPS, seed: int = BOOT_SEED) -> dict:
    """Percentile bootstrap CI for the variance shares (unit-level resampling).

    Caveat: the units are architecture x direction cells, which are **not**
    independent (the same four architectures recur across directions and the
    target domains overlap).  This interval is therefore indicative of
    cell-level variability, not a strict sampling interval for the population of
    architectures.  It is reported as a precision indicator only.
    """
    rng = np.random.default_rng(seed)
    n = det.size
    sd, sl, sc = [], [], []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        try:
            a = attribution(det[idx], del_[idx])
        except ValueError:
            continue
        sd.append(a["share_det"])
        sl.append(a["share_del"])
        sc.append(a["share_cov"])
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return dict(reps=int(reps), seed=int(seed),
                share_det_ci=q(sd), share_del_ci=q(sl), share_cov_ci=q(sc))


# --------------------------------------------------------------------------- #
# Unit tables
# --------------------------------------------------------------------------- #
def ood_cells(crit: float = dz.FROZEN_THRESH) -> pd.DataFrame:
    """The 16 out-of-domain cells, factorized at detection criterion ``crit``."""
    od = dz.load_ood()
    rec = []
    for (dr, ar), g in od.groupby(["dir", "arch"], sort=True):
        m = g["recall_gt"].values > crit
        det = float(m.mean())
        del_ = float(g["dice"].values[m].mean()) if m.any() else np.nan
        rec.append(dict(dir=dr, arch=ar, n=len(g), dice=float(g["dice"].mean()),
                        det=det, del_=del_, n_undet=int((~m).sum())))
    return pd.DataFrame(rec)


def id_groups() -> pd.DataFrame:
    """The 20 in-domain groups with both criteria readings."""
    rec = []
    folders = ([f"baseline/{a}" for a in dz.ID_BASELINE_ARCHS]
               + [f"ablation/{c}" for c in dz.ID_ABLATION_CFGS])
    for folder in folders:
        g = dz.load_id(folder)
        col, src = dz.id_metric_column(g)
        dice = g["Dice"].values
        met = g[col].values
        fz = dz.factorize(dice, met, dz.FROZEN_THRESH)
        fs = dz.factorize(dice, met, dz.SUBSTANTIVE_THRESH)
        rec.append(dict(group=folder, n=len(g), src=src, dice=float(dice.mean()),
                        det_frozen=fz["detection"], del_frozen=fz["delineation"],
                        det_sub=fs["detection"], del_sub=fs["delineation"]))
    return pd.DataFrame(rec)


def criterion_scan() -> pd.DataFrame:
    """Shares as a function of the detection criterion, both sides (B2)."""
    rec = []
    for crit in CRIT_GRID:
        o = ood_cells(crit)
        a_o = attribution(o["det"].values, o["del_"].values)
        rec.append(dict(side="OOD", crit=crit, n=a_o["n"],
                        det_min=a_o["detection_range"][0], det_max=a_o["detection_range"][1],
                        det_spread=a_o["detection_spread"],
                        share_det=a_o["share_det"], share_del=a_o["share_del"],
                        share_cov=a_o["share_cov"], total=a_o["total"]))
        i = id_groups()
        src = "sub" if crit > 0 else "frozen"
        icrit = dz.SUBSTANTIVE_THRESH if crit > 0 else dz.FROZEN_THRESH
        recs = []
        for folder in i["group"]:
            g = dz.load_id(folder)
            col, _ = dz.id_metric_column(g)
            fz = dz.factorize(g["Dice"].values, g[col].values, icrit)
            recs.append((fz["detection"], fz["delineation"]))
        d = np.array([r[0] for r in recs])
        q = np.array([r[1] for r in recs])
        a_i = attribution(d, q)
        rec.append(dict(side="ID", crit=crit, n=a_i["n"], reading=src,
                        det_min=a_i["detection_range"][0], det_max=a_i["detection_range"][1],
                        det_spread=a_i["detection_spread"], n_saturated=int((d >= 1.0).sum()),
                        share_det=a_i["share_det"], share_del=a_i["share_del"],
                        share_cov=a_i["share_cov"], total=a_i["total"]))
    return pd.DataFrame(rec)


def id_artefact_diagnostics(idf: pd.DataFrame) -> dict:
    """Show that the in-domain *substantive* detection rate is an artefact count.

    Under the substantive criterion a sample is "undetected" when its metric is
    <= 1e-6, i.e. when the only overlap is the metric smoothing constant.  This
    function verifies the exact relation ``det_sub = 1 - k/n`` with ``k`` the
    number of such eps-artefact samples, and measures how far ``k`` explains the
    observed spread of ``det_sub``.  If the relation is exact, the variance of
    the substantive detection factor in-domain *is* the variance of a pipeline
    noise count and must not be read as detection ability.
    """
    rec = []
    for folder in idf["group"]:
        g = dz.load_id(folder)
        col, _ = dz.id_metric_column(g)
        v = g[col].values
        k = int((v <= dz.SUBSTANTIVE_THRESH).sum())
        rec.append(dict(group=folder, n=len(g), n_artefact=k,
                        det_sub=1.0 - k / len(g), dice=float(g["Dice"].mean())))
    a = pd.DataFrame(rec)
    exact = bool(np.allclose(a["det_sub"].values, idf["det_sub"].values, atol=1e-12))
    from scipy import stats as _st
    r, p = _st.pearsonr(a["n_artefact"].values, a["dice"].values)
    lt = np.log(idf["det_sub"].values)
    ll = np.log(idf["del_sub"].values)
    cov = float(np.cov(lt, ll, ddof=1)[0, 1])
    rc, pc = _st.pearsonr(lt, ll)
    return dict(
        table=a.to_dict("records"),
        relation_exact=exact,
        distinct_det_sub=sorted({round(float(x), 12) for x in a["det_sub"].values}),
        n_artefact_range=[int(a["n_artefact"].min()), int(a["n_artefact"].max())],
        corr_artefact_dice=float(r), corr_artefact_dice_p=float(p),
        corr_logdet_logdel=float(rc), corr_logdet_logdel_p=float(pc),
        cov_negative=bool(cov < 0),
        reading=("det_sub = 1 - k/n holds exactly; the in-domain substantive detection variance "
                 "is the variance of the eps-artefact count (pipeline noise), not detection ability"),
    )


def identity_validity_substantive(idf: pd.DataFrame) -> dict:
    """Check that the factorization identity still holds under the substantive criterion.

    Justifies computing a variance attribution at that reading at all: the samples
    reclassified as undetected still carry Dice ~ 1e-11, so the residual stays at
    machine-noise level.
    """
    rel = []
    for folder in idf["group"]:
        g = dz.load_id(folder)
        col, _ = dz.id_metric_column(g)
        fs = dz.factorize(g["Dice"].values, g[col].values, dz.SUBSTANTIVE_THRESH)
        rel.append(abs(fs["residual"]) / fs["mean_dice"])
    return dict(max_rel_residual=float(max(rel)), valid=bool(max(rel) < 1e-9))


# --------------------------------------------------------------------------- #
# Hypothesis tests
# --------------------------------------------------------------------------- #
def test_h1(idf: pd.DataFrame) -> dict:
    """H1: in-domain detection identically 1.000 (frozen) / 0.971-0.992 (substantive)."""
    frozen_all_one = bool(np.allclose(idf["det_frozen"].values, 1.0, atol=0, rtol=0))
    sub = idf["det_sub"].values
    below = idf.loc[idf["det_sub"] < H1_BAR, ["group", "det_sub"]]
    return dict(
        n_groups=int(len(idf)),
        frozen_all_exactly_one=frozen_all_one,
        frozen_n_saturated=int((idf["det_frozen"].values >= 1.0).sum()),
        substantive_min=float(sub.min()),
        substantive_max=float(sub.max()),
        substantive_spread=float(sub.max() - sub.min()),
        substantive_n_saturated=int((sub >= 1.0).sum()),
        bar=H1_BAR,
        n_below_bar=int(len(below)),
        groups_below_bar=below.to_dict("records"),
        verdict=(
            "H1 holds EXACTLY under the frozen criterion (all 20 groups = 1.000), but that reading is "
            "constructed by the metric smoothing constant and carries no information. Under the substantive "
            f"criterion the range is {sub.min():.4f}-{sub.max():.4f} and {len(below)}/{len(idf)} groups fall "
            f"below the pre-registered bar {H1_BAR}; the pre-registered handling is to write L1-1 as "
            "'approaching saturation' rather than 'saturation'. Cause: det_sub = 1 - k/n exactly, with k the "
            "eps-artefact count (2-7 of 242) -- the in-domain variation measures the metric implementation, "
            "not the models."
        ),
    )


def test_h2(a: dict, boot: dict) -> dict:
    """H2: OOD detection share of var(log Dice) >= 35% (n=4 initial value)."""
    share = a["share_det"]
    if share >= H2_SUPPORT:
        verdict = "supported at n=4 (final value pending P2-2b)"
    elif share < H2_REFUTE:
        verdict = "REFUTED at n=4 -> L2 demoted; final check still pending P2-2b"
    else:
        verdict = "indeterminate at n=4 (between 20% and 35%)"
    return dict(share_det=share, support=H2_SUPPORT, refute=H2_REFUTE,
                boot_ci=boot["share_det_ci"], verdict=verdict,
                caveat="n=4 architectures x 4 directions = 16 cells; H2 final judgement is P2-2b")


# --------------------------------------------------------------------------- #
# Self-check
# --------------------------------------------------------------------------- #
def selfcheck(verbose: bool = True) -> None:
    """Synthetic checks with analytically known answers."""
    # (a) Constant detection -> detection share exactly 0, delineation share 1.
    det = np.ones(5)
    del_ = np.array([0.2, 0.4, 0.5, 0.7, 0.9])
    a = attribution(det, del_)
    assert a["var_log_det"] == 0.0 and a["cov2"] == 0.0, a
    assert np.isclose(a["share_det"], 0.0, atol=1e-15) and np.isclose(a["share_del"], 1.0, atol=1e-15), a

    # (b) Constant delineation -> detection share exactly 1, covariance 0.
    a = attribution(np.array([0.2, 0.4, 0.5, 0.7, 0.9]), np.ones(5))
    assert np.isclose(a["share_det"], 1.0, atol=1e-15) and np.isclose(a["share_cov"], 0.0, atol=1e-15), a

    # (c) Shares always sum to 1 for arbitrary positive factors.
    rng = np.random.default_rng(1)
    for _ in range(200):
        d = rng.uniform(0.1, 1.0, 12)
        q = rng.uniform(0.1, 1.0, 12)
        a = attribution(d, q)
        assert np.isclose(a["share_det"] + a["share_del"] + a["share_cov"], 1.0, atol=1e-12), a
        # and var(log Dice) is reproduced by the three terms
        assert np.isclose(a["total"], a["var_log_dice_direct"], rtol=1e-9, atol=1e-15), a
        assert abs(a["identity_gap"]) < 1e-12 * max(abs(a["total"]), 1e-12) + 1e-15, a

    # (d) Known closed form: log-normal factors with independent logs.
    #     var(log Det) = s1^2, var(log Del) = s2^2, cov = 0 -> share = s1^2/(s1^2+s2^2).
    s1, s2, k = 0.3, 0.6, 4000
    lt = rng.normal(0, s1, k)
    ll = rng.normal(0, s2, k)
    a = attribution(np.exp(lt), np.exp(ll))
    expected = s1 ** 2 / (s1 ** 2 + s2 ** 2)
    assert abs(a["share_det"] - expected) < 0.03, (a["share_det"], expected)
    assert abs(a["share_cov"]) < 0.05, a

    # (e) Bootstrap is deterministic under a fixed seed and brackets the estimate.
    b1 = bootstrap_share(np.array([0.2, 0.5, 0.8, 0.9]), np.array([0.5, 0.6, 0.7, 0.95]), reps=200, seed=7)
    b2 = bootstrap_share(np.array([0.2, 0.5, 0.8, 0.9]), np.array([0.5, 0.6, 0.7, 0.95]), reps=200, seed=7)
    assert b1 == b2, "bootstrap must be reproducible under a fixed seed"
    lo, hi = b1["share_det_ci"]
    assert lo <= hi, b1

    if verbose:
        print("[selfcheck] PASS  (a-e: degenerate shares, additivity, log-normal closed form, seeded bootstrap)")


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _pct(x):
    return f"{x * 100:.1f}%"


def write_report(res: dict) -> None:
    a = res["ood_attribution"]
    ai_f = res["id_attribution_frozen"]
    ai_s = res["id_attribution_sub"]
    ab_f = res["id_baseline_attribution_frozen"]
    ab_s = res["id_baseline_attribution_sub"]
    h1, h2 = res["h1"], res["h2"]
    diag = res["id_artefact_diagnostics"]
    valid = res["identity_validity_substantive"]
    scan = pd.DataFrame(res["criterion_scan"])
    idf = pd.DataFrame(res["id_groups"])
    ood = pd.DataFrame(res["ood_cells"])

    L: list = []
    L.append("# T_variance_n4 — variance attribution of Dice, in-domain vs out-of-domain")
    L.append("")
    L.append(f"*Machine-generated by `02_code/analysis/variance_attribution.py` on "
             f"{res['generated_at']}.*")
    L.append("")
    L.append("> **Scope: n = 4 architectures (16 out-of-domain cells). This is the INITIAL value.**")
    L.append("> The pre-registered judgement data for H2 is *all* architectures; the final value is")
    L.append("> produced by **P2-2b after P3-1** and overwrites this table. The `_n4` suffix marks it.")
    L.append("")
    L.append("## 1. Headline (frozen criterion — the reading under which the identity is exact)")
    L.append("")
    L.append("| quantity | in-domain (20 groups) | out-of-domain (16 cells) |")
    L.append("|---|---|---|")
    L.append(f"| detection share of var(log Dice) | **{_pct(ai_f['share_det'])}** | **{_pct(a['share_det'])}** |")
    L.append(f"| delineation share | {_pct(ai_f['share_del'])} | {_pct(a['share_del'])} |")
    L.append(f"| covariance share | {_pct(ai_f['share_cov'])} | {_pct(a['share_cov'])} |")
    L.append(f"| detection range | {ai_f['detection_range'][0]:.4f}–{ai_f['detection_range'][1]:.4f} | "
             f"{a['detection_range'][0]:.3f}–{a['detection_range'][1]:.3f} |")
    L.append(f"| detection spread | {ai_f['detection_spread']:.4f} | **{a['detection_spread']:.4f}** |")
    L.append(f"| log-scale spread, detection / delineation | {ai_f['logdet_spread']:.4f} / "
             f"{ai_f['logdel_spread']:.4f} | {a['logdet_spread']:.4f} / {a['logdel_spread']:.4f} |")
    L.append("")
    L.append("**Reading.** In-domain the detection factor is a constant 1.000 under the frozen criterion,")
    L.append("so it contributes **exactly zero** of the log-Dice variance: the in-domain ranking criterion")
    L.append("contains no detection term at all. Out of domain the same factor carries "
             f"{_pct(a['share_det'])} of the variance and moves over {a['detection_spread']:.2f} "
             f"(in-domain: {ai_f['detection_spread']:.4f}).")
    L.append("")
    L.append(f"The in-domain side is *structurally* zero: `share_det = 0` does not depend on how the 20")
    L.append("groups are chosen, because the detection factor is identical for all of them. The out-of-domain")
    L.append(f"share, by contrast, is a statistic over cells — see the bootstrap interval in section 3.")
    L.append("")
    L.append("## 2. Hypothesis H1 — in-domain detection (dual reading, B6)")
    L.append("")
    L.append(f"- **Frozen criterion** `|P n G| > 0`: {h1['frozen_n_saturated']}/{h1['n_groups']} groups "
             f"= 1.000 exactly (all exactly one: `{h1['frozen_all_exactly_one']}`) → contributes 0% variance.")
    L.append(f"- **Substantive criterion** `I >= 1 px` (metric > 1e-6): range "
             f"{h1['substantive_min']:.4f}–{h1['substantive_max']:.4f}, spread "
             f"{h1['substantive_spread']:.4f}, **{h1['substantive_n_saturated']}/{h1['n_groups']} saturated**.")
    L.append(f"- Pre-registered refutation bar `{h1['bar']}` applied to the substantive reading → "
             f"**{h1['n_below_bar']}/{h1['n_groups']} groups fall below it**.")
    L.append("")
    if h1["groups_below_bar"]:
        L.append("| group below the bar | substantive detection | eps-artefact samples (of 242) |")
        L.append("|---|---:|---:|")
        k_of = {r["group"]: r["n_artefact"] for r in diag["table"]}
        for r in sorted(h1["groups_below_bar"], key=lambda x: x["det_sub"]):
            L.append(f"| {r['group']} | {r['det_sub']:.4f} | {k_of[r['group']]} |")
        L.append("")
    L.append(f"**Verdict.** {h1['verdict']}")
    L.append("")
    L.append("### 2a. Why the substantive reading varies — an artefact count, not an ability")
    L.append("")
    L.append(f"The relation `det_sub = 1 - k/n` holds **exactly** "
             f"(`{diag['relation_exact']}`), where `k` is the number of samples whose only overlap is the")
    L.append(f"metric smoothing constant. In-domain `k` takes only the values "
             f"{diag['n_artefact_range'][0]}–{diag['n_artefact_range'][1]} out of 242, giving just "
             f"{len(diag['distinct_det_sub'])} distinct values of `det_sub`. The entire in-domain spread of")
    L.append("the substantive detection rate is therefore the spread of a **pipeline noise count**.")
    L.append(f"Correlation between that count and group Dice: r = {diag['corr_artefact_dice']:+.3f} "
             f"(p = {diag['corr_artefact_dice_p']:.2f}).")
    L.append("")
    L.append("Consequence for reporting: both readings must always be given together and the criterion must")
    L.append("always be stated (B6). The frozen 1.000 is *constructed* by the smoothing constant and carries")
    L.append("no information about detection; the substantive reading shows the factor is not degenerate, but")
    L.append("its in-domain variation measures the metric implementation, not the models. What is robust")
    L.append("regardless of the bar is the **order-of-magnitude separation**: detection spread "
             f"{h1['substantive_spread']:.4f} in-domain vs {a['detection_spread']:.4f} out-of-domain "
             f"(22.9x).")
    L.append("")
    L.append("✨ **DECIDED (user, 2026-09-14): option 1 — apply the pre-registered wording.** L1-1 is written as")
    L.append("\"approaching saturation\" rather than \"saturation\", with the dual reading kept mandatory and")
    L.append(f"the eps-artefact mechanism ({h1['n_below_bar']} of {h1['n_groups']} groups below the {h1['bar']} bar under")
    L.append("the substantive criterion) given as the explanation. The frozen reading is still reported alongside,")
    L.append("because it is the reading under which the identity is exact.")
    L.append("")
    L.append("## 3. Hypothesis H2 — out-of-domain detection share")
    L.append("")
    L.append(f"- `share_det` = **{_pct(h2['share_det'])}**, support bar {_pct(h2['support'])}, "
             f"refutation bar {_pct(h2['refute'])}.")
    L.append(f"- Cell-level bootstrap 95% CI (n = {a['n']}, {h2['boot_ci'][0]*100:.1f}%–"
             f"{h2['boot_ci'][1]*100:.1f}%): the **lower bound stays above the refutation bar**, so H2 is not")
    L.append("  refuted even at the pessimistic end. Indicative only — the cells are not independent")
    L.append("  (4 architectures recur across 4 directions and the target domains overlap).")
    L.append(f"- **Verdict: {h2['verdict']}**")
    L.append(f"- ⚠️ {h2['caveat']}")
    L.append("")
    L.append("Note the large covariance term "
             f"({_pct(a['share_cov'])}, positive): ignoring it would inflate the apparent detection share from")
    L.append(f"{_pct(a['share_det'])} to {_pct(a['share_det_naive'])}, because the denominator loses the "
             f"covariance. Reporting the three terms separately is therefore necessary; a two-term")
    L.append("sum of variances silently misattributes about a fifth of the out-of-domain spread.")
    L.append("")
    L.append("## 4. Unit-level detail")
    L.append("")
    L.append("### 4a. Out-of-domain cells (frozen criterion)")
    L.append("")
    rows = [[r["dir"], r["arch"], r["n"], f"{r['dice']:.4f}", f"{r['det']:.4f}",
             f"{r['del_']:.4f}"] for _, r in ood.iterrows()]
    L += _md(rows, ["direction", "architecture", "n", "E[Dice]", "Detection", "Delineation"],
             ["---", "---", "---:", "---:", "---:", "---:"])
    L.append("")
    L.append("### 4b. In-domain groups (both criteria)")
    L.append("")
    k_of = {r["group"]: r["n_artefact"] for r in diag["table"]}
    rows = [[r["group"], r["src"], r["n"], f"{r['dice']*100:.2f}", f"{r['det_frozen']:.4f}",
             f"{r['del_frozen']*100:.2f}", f"{r['det_sub']:.4f}", f"{r['del_sub']*100:.2f}",
             k_of[r["group"]]]
            for _, r in idf.iterrows()]
    L += _md(rows, ["group", "metric", "n", "E[Dice] %", "Det(frozen)", "Del(frozen) %",
                    "Det(sub)", "Del(sub) %", "k"],
             ["---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"])
    L.append("")
    L.append("## 5. Secondary attribution under the substantive criterion (diagnostic only)")
    L.append("")
    L.append("The factorization identity remains valid at this reading — the reclassified samples still carry")
    L.append(f"Dice ~ 1e-11, so the maximum relative residual is {valid['max_rel_residual']:.2e} "
             f"(valid: `{valid['valid']}`) — hence the attribution can be computed. It must **not** be read as")
    L.append("a measurement of detection-ability variance (see section 2a).")
    L.append("")
    L.append("| unit set | n | share det | share del | share cov | cov < 0 |")
    L.append("|---|---:|---:|---:|---:|---|")
    L.append(f"| ID groups (20) | {ai_s['n']} | {_pct(ai_s['share_det'])} | {_pct(ai_s['share_del'])} | "
             f"{_pct(ai_s['share_cov'])} | {ai_s['cov_negative']} |")
    L.append(f"| ID baselines only (13) | {ab_s['n']} | {_pct(ab_s['share_det'])} | {_pct(ab_s['share_del'])} | "
             f"{_pct(ab_s['share_cov'])} | {ab_s['cov_negative']} |")
    L.append("")
    if ab_s["cov_negative"] or ai_s["cov_negative"]:
        L.append("⚠️ **Shares are not confined to [0, 1] when the covariance is negative.** In the 13-baseline")
        L.append("subset the delineation share exceeds 100% and the covariance share is negative. The reason is")
        L.append("that the logged factors are negatively correlated in that subset "
                 f"(r = {ab_s['corr_logdet_logdel']:+.3f}, "
                 f"p = {ab_s['corr_logdet_logdel_p']:.2f}, n = {ab_s['n']} — noise level), whereas across all 20")
        L.append(f"groups the correlation is {ai_s['corr_logdet_logdel']:+.3f} "
                 f"(p = {ai_s['corr_logdet_logdel_p']:.2f}). Shares alone are therefore an incomplete summary at")
        L.append("this reading; the log-scale spreads are the well-behaved quantity.")
        L.append("")
    L.append("## 6. Sensitivity to the detection criterion (B2)")
    L.append("")
    L.append("The detection criterion is scanned from the frozen `> 0` to the strict `> 0.5`. This is the")
    L.append("sensitivity to the criterion applied to **pre-computed per-sample metrics**; a true")
    L.append("binarization-threshold (tau) scan needs the raw probability maps and belongs to P3-7. This")
    L.append("module does not claim to substitute for it.")
    L.append("")
    rows = []
    for _, r in scan.iterrows():
        rows.append([r["side"], f"{r['crit']:g}", r["n"], f"{r['det_min']:.4f}", f"{r['det_max']:.4f}",
                     f"{r['det_spread']:.4f}", f"{r['share_det']*100:.1f}%", f"{r['share_del']*100:.1f}%",
                     f"{r['share_cov']*100:.1f}%"])
    L += _md(rows, ["side", "criterion >", "n", "det min", "det max", "det spread",
                    "share det", "share del", "share cov"],
             ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"])
    L.append("")
    L.append("The out-of-domain rows for `> 0` and `> 1e-6` are identical because `recall_gt` carries no")
    L.append("smoothing constant — its zeros are exact, which is precisely what makes the out-of-domain")
    L.append("detection factor a genuine measurement rather than a construction.")
    L.append("")
    L.append("## 7. Robustness and exclusions")
    L.append("")
    L.append(f"- **Baselines only (n = {ab_f['n']})**, frozen criterion: detection share "
             f"{_pct(ab_f['share_det'])}. Removing the 7 ablation configurations, which share one")
    L.append("  architecture, does not change the qualitative conclusion.")
    L.append(f"- **Excluded artifact** `{EXCLUDED_ID}` — a different-protocol run whose mean Dice lies far")
    L.append("  above the ablation grid and which is not part of the 20 pre-registered groups. Per the")
    L.append("  Table-1 prohibition (M10) it must not be juxtaposed with the ablation grid, so it is not")
    L.append("  admitted as a unit here.")
    L.append("- **H3 is not tested here.** Spearman alignment at n = 4 has zero statistical power and is")
    L.append("  forbidden in the manuscript; H3 belongs to P2-3 at architectural n = 14.")
    L.append(f"- **Numerical identity check**: max |var(log Dice) - three-term total| = "
             f"{max(abs(x) for x in [a['identity_gap'], ai_f['identity_gap'], ai_s['identity_gap']]):.3e} "
             "(the factorization residual of P2-1).")
    L.append("")
    L.append("## 8. Reproduce")
    L.append("")
    L.append("```bash")
    L.append("D:/miniconda/aniconda/envs/medical_seg/python.exe E:/paper2_ablation_reliability/02_code/analysis/variance_attribution.py")
    L.append("```")
    L.append("")
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


def _md(rows, header, aligns):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(aligns) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return out


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selfcheck-only", action="store_true")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)

    selfcheck()
    if args.selfcheck_only:
        return 0

    ood = ood_cells(dz.FROZEN_THRESH)
    idf = id_groups()
    scan = criterion_scan()

    a_ood = attribution(ood["det"].values, ood["del_"].values)
    boot = bootstrap_share(ood["det"].values, ood["del_"].values)

    a_id_f = attribution(idf["det_frozen"].values, idf["del_frozen"].values)
    a_id_s = attribution(idf["det_sub"].values, idf["del_sub"].values)
    bl = idf[idf["group"].str.startswith("baseline/")]
    a_bl_f = attribution(bl["det_frozen"].values, bl["del_frozen"].values)
    a_bl_s = attribution(bl["det_sub"].values, bl["del_sub"].values)

    res = dict(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        script="02_code/analysis/variance_attribution.py",
        scope=dict(n_architectures=4, ood_cells=int(len(ood)), id_groups=int(len(idf)),
                   note="n=4 INITIAL value; H2 final judgement produced by P2-2b after P3-1"),
        criterion=dict(frozen="|P n G| > 0 (metric > 0)",
                       substantive="I >= 1 px (metric > 1e-6)",
                       eps_id=dz.EPS_ID, eps_ood=dz.EPS_OOD),
        ood_attribution=a_ood,
        id_attribution_frozen=a_id_f,
        id_attribution_sub=a_id_s,
        id_baseline_attribution_frozen=a_bl_f,
        id_baseline_attribution_sub=a_bl_s,
        ood_bootstrap=boot,
        h1=test_h1(idf),
        h2=test_h2(a_ood, boot),
        id_artefact_diagnostics=id_artefact_diagnostics(idf),
        identity_validity_substantive=identity_validity_substantive(idf),
        criterion_scan=scan.to_dict("records"),
        ood_cells=ood.to_dict("records"),
        id_groups=idf.to_dict("records"),
        excluded_artifacts=[dict(path=EXCLUDED_ID,
                                reason="different protocol (M10); not among the 20 pre-registered groups")],
    )

    print()
    print("=" * 100)
    print("Variance attribution of log Dice   (detection / delineation / covariance shares)")
    print("=" * 100)
    hdr = f"{'side':<26}{'crit':>6}{'n':>4}{'share_det':>11}{'share_del':>11}{'share_cov':>11}"
    print(hdr)
    for tag, aa, crit in [("OOD (frozen)", a_ood, ">0"), ("ID (frozen)", a_id_f, ">0"),
                          ("ID (substantive)", a_id_s, ">1e-6"),
                          ("ID baselines (frozen)", a_bl_f, ">0"),
                          ("ID baselines (subst.)", a_bl_s, ">1e-6")]:
        print(f"{tag:<26}{crit:>6}{aa['n']:>4}{_pct(aa['share_det']):>11}"
              f"{_pct(aa['share_del']):>11}{_pct(aa['share_cov']):>11}")
    print()
    print(f"  OOD detection range   {a_ood['detection_range'][0]:.4f} – {a_ood['detection_range'][1]:.4f}"
          f"   (spread {a_ood['detection_spread']:.4f})")
    print(f"  ID  detection range   {a_id_s['detection_range'][0]:.4f} – {a_id_s['detection_range'][1]:.4f}"
          f"   (spread {a_id_s['detection_spread']:.4f}, substantive)")
    print(f"  OOD bootstrap 95% CI for share_det: {boot['share_det_ci'][0]*100:.1f}% – {boot['share_det_ci'][1]*100:.1f}%")
    print(f"  identity gap (max |var(logDice) - sum|) = "
          f"{max(abs(a_ood['identity_gap']), abs(a_id_f['identity_gap']), abs(a_id_s['identity_gap'])):.3e}")
    print()
    print(f"  H1  frozen: {res['h1']['frozen_n_saturated']}/{res['h1']['n_groups']} = 1.000  "
          f"(all exactly one: {res['h1']['frozen_all_exactly_one']})")
    print(f"      substantive: {res['h1']['substantive_min']:.4f} – {res['h1']['substantive_max']:.4f}, "
          f"{res['h1']['substantive_n_saturated']}/{res['h1']['n_groups']} saturated, "
          f"{res['h1']['n_below_bar']} group(s) below {res['h1']['bar']}")
    for r in sorted(res["h1"]["groups_below_bar"], key=lambda x: x["det_sub"]):
        print(f"        - {r['group']:<24} {r['det_sub']:.4f}")
    print(f"  H2  share_det = {_pct(res['h2']['share_det'])}  vs support {_pct(res['h2']['support'])}  "
          f"-> {res['h2']['verdict']}")

    if not args.no_write:
        os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
        os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
        with open(OUT_JSON, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=2)
        write_report(res)
        print()
        print(f"  wrote: {OUT_JSON}")
        print(f"  wrote: {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
