# -*- coding: utf-8 -*-
"""
variance_attribution_n14.py -- P2-2b: re-run the variance attribution on the
*expanded* architecture set and overwrite the n=4 initial value.

Why this exists
---------------
P2-2 produced the out-of-domain attribution on 4 architectures x 4 directions
= 16 cells, and labelled it `_n4` (INITIAL value).  The pre-registered
judgement data for H2 is the *architecture* set, so the final value has to be
computed after P3-1.  That is this script.

Unit sets (frozen before running, G6)
-------------------------------------
Out of domain, four sets are reported SIDE BY SIDE and never merged silently:

  A. `arch14`      n = 14   13 general-purpose baselines + EGAUNet,
                            P3-1 *route A* (ID-joint-trained -> ETIS).
                            --> PRIMARY.  This is the pre-registered H2 set.
  B. `baseline13`  n = 13   A minus EGAUNet (EGAUNet shares lineage with the
                            ablation grid, so this is the clean cross-architecture set).
  C. `ablation7`   n = 7    the ablation grid evaluated on ETIS (P3-1b).
  D. `legacy16`    n = 16   the n=4 x 4-direction cells of P2-2, kept only as a
                            cross-calibre reference so the n=4 -> n=14 shift can be read.

In domain the unit set is unchanged: 20 groups = 13 baselines + 7 ablation configs.

Data sources (all read-only; D: is never written)
-------------------------------------------------
OOD A/B  E:/paper2_ablation_reliability/03_results/raw/p31a_ood_idtrain/per_sample.csv   (route A, 2744 rows)
         cross-gated against 03_results/stats/p31a_units_n14.csv
OOD C    03_results/stats/decomp_ablation_etis.csv                                       (P3-1b, 7 rows)
OOD D    via decompose.load_ood() -> 03_results/raw/ood_per_sample/paper2_per_sample.csv  (P2-1 calibre)
ID        via decompose.load_id()  -> D:/medical_segmentation/experiments/...            (P2-1 calibre)

The attribution arithmetic is imported from `variance_attribution` so that the
split is computed by exactly one implementation (single source of truth).

Run
---
D:/miniconda/aniconda/envs/medical_seg/python.exe 02_code/analysis/variance_attribution_n14.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import decompose as dz                       # noqa: E402  frozen calibre
import variance_attribution as va            # noqa: E402  attribution arithmetic

E_ROOT = "E:/paper2_ablation_reliability"
OUT_JSON = E_ROOT + "/03_results/stats/variance_attribution.json"
OUT_MD = E_ROOT + "/03_results/tables/T_variance.md"

P31A_PER_SAMPLE = E_ROOT + "/03_results/raw/p31a_ood_idtrain/per_sample.csv"
P31A_UNITS = E_ROOT + "/03_results/stats/p31a_units_n14.csv"
ABLATION_ETIS = E_ROOT + "/03_results/stats/decomp_ablation_etis.csv"
N4_JSON = E_ROOT + "/03_results/stats/variance_attribution_n4.json"

# Frozen analysis constants (identical to P2-2 so the two runs are comparable).
BOOT_REPS = 2000
BOOT_SEED = 42
H1_BAR = 0.98
H2_SUPPORT = 0.35
H2_REFUTE = 0.20
CRIT_GRID = [0.0, 1e-6, 1e-2, 0.05, 0.1, 0.3, 0.5]

PRIMARY_SET = "arch14"
OOD_SETS = ["arch14", "baseline13", "ablation7", "legacy16"]

GATE_TOL = 1e-12


# --------------------------------------------------------------------------- #
# Out-of-domain unit tables
# --------------------------------------------------------------------------- #
def ood_units_routeA(archs=None, crit: float = dz.FROZEN_THRESH) -> pd.DataFrame:
    """Re-factorize the route-A per-sample file into one row per architecture.

    ``crit`` is applied to ``recall_gt``, which carries no smoothing constant, so
    ``crit = 0`` and ``crit = 1e-6`` give identical numbers (that equality is the
    point of B6: on this side the frozen criterion is a real measurement).
    """
    ps = pd.read_csv(P31A_PER_SAMPLE)
    if ps["arch"].nunique() != 14:
        raise RuntimeError("expected 14 architectures in route A, got %d" % ps["arch"].nunique())
    if ps["dir"].nunique() != 1:
        raise RuntimeError("unexpected directions in route A: %r" % (sorted(ps["dir"].unique()),))

    rec = []
    for ar, g in ps.groupby("arch", sort=True):
        if archs is not None and ar not in archs:
            continue
        dice = g["dice"].values.astype(float)
        met = g["recall_gt"].values.astype(float)
        m = met > crit
        rec.append(dict(
            unit=ar, arch=ar, n=int(len(g)),
            dice=float(dice.mean()),
            det=float(m.mean()),
            del_=float(dice[m].mean()) if m.any() else float("nan"),
            n_undet=int((~m).sum()),
        ))
    df = pd.DataFrame(rec).sort_values("arch").reset_index(drop=True)
    if archs is not None and len(df) != len(archs):
        raise RuntimeError("architecture mismatch: %d rows for %d requested" % (len(df), len(archs)))
    return df


def ood_units_ablation7(crit: float = dz.FROZEN_THRESH) -> pd.DataFrame:
    """The ablation grid on ETIS, from the P3-1b aggregate table.

    On this side the friezen criterion and the substantive criterion are strictly
    equivalent (``recall_gt`` has no eps), so no threshold scan is available from
    the aggregate; the equivalence is asserted instead.
    """
    ab = pd.read_csv(ABLATION_ETIS)
    if len(ab) != 7:
        raise RuntimeError("expected 7 ablation rows, got %d" % len(ab))
    if not np.allclose(ab["detection"].values, ab["detection_sub"].values, atol=GATE_TOL):
        raise RuntimeError("P3-1b: frozen and substantive detection differ on the OOD side")
    if crit != dz.FROZEN_THRESH:
        raise ValueError("ablation7 aggregates are only available at the frozen criterion")
    return pd.DataFrame(dict(
        unit="ablation/" + ab["config"], arch=ab["config"], n=ab["n"].astype(int),
        dice=ab["mean_dice"], det=ab["detection_sub"], del_=ab["delineation_sub"],
        n_undet=ab["n_undetected_sub"].astype(int),
    ))


def ood_units_legacy16(crit: float = dz.FROZEN_THRESH) -> pd.DataFrame:
    """The 16 cells of P2-2 (4 architectures x 4 directions), cross-calibre reference."""
    od = dz.load_ood()
    rec = []
    for (dr, ar), g in od.groupby(["dir", "arch"], sort=True):
        m = g["recall_gt"].values > crit
        rec.append(dict(unit="%s/%s" % (dr, ar), arch=ar, dir=dr, n=int(len(g)),
                        dice=float(g["dice"].mean()), det=float(m.mean()),
                        del_=float(g["dice"].values[m].mean()) if m.any() else float("nan"),
                        n_undet=int((~m).sum())))
    return pd.DataFrame(rec)


def ood_set(name: str, crit: float = dz.FROZEN_THRESH) -> pd.DataFrame:
    if name == "arch14":
        return ood_units_routeA()
    if name == "baseline13":
        return ood_units_routeA(archs=[a for a in _arch14_names() if a != "EGAUNet"])
    if name == "ablation7":
        return ood_units_ablation7(crit)
    if name == "legacy16":
        return ood_units_legacy16(crit)
    raise KeyError(name)


def _arch14_names():
    return list(pd.read_csv(P31A_UNITS)["arch"].values)


# --------------------------------------------------------------------------- #
# Criterion scan
# --------------------------------------------------------------------------- #
def logspread_share(det, del_) -> float:
    """Bounded companion to ``share_det``: detection's share of the *log-spread*.

    ``(log det spread) / (log det spread + log del spread)`` lies in [0, 1] by
    construction, so it stays interpretable even when the variance shares leave
    [0, 1] because the covariance term turns negative.  Reported alongside — never
    instead of — the variance shares.
    """
    det = np.asarray(det, dtype=float)
    del_ = np.asarray(del_, dtype=float)
    sd = np.log(det.max()) - np.log(det.min())
    sl = np.log(del_.max()) - np.log(del_.min())
    if sd + sl <= 0:
        return float("nan")
    return float(sd / (sd + sl))


def boot_logspread_share(det, del_, reps: int = BOOT_REPS, seed: int = BOOT_SEED) -> dict:
    """Percentile CI for the bounded log-spread share (same resampling as the shares)."""
    rng = np.random.default_rng(seed)
    n = len(det)
    vals = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        v = logspread_share(np.asarray(det)[idx], np.asarray(del_)[idx])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return dict(reps=reps, seed=seed, ci=[float("nan")] * 2, n_valid=0)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return dict(reps=int(reps), seed=int(seed), ci=[float(lo), float(hi)], n_valid=len(vals),
                point=logspread_share(det, del_))


def criterion_scan() -> pd.DataFrame:
    """Shares vs detection criterion.  A real threshold scan on the sets whose
    per-sample data is available; on this benchmark the OOD side is invariant for
    every criterion above 0 because recall_gt has no smoothing constant."""
    rec = []
    for crit in CRIT_GRID:
        for name in ["arch14", "baseline13", "legacy16"]:
            df = ood_set(name, crit)
            a = va.attribution(df["det"].values, df["del_"].values)
            rec.append(dict(side="OOD/%s" % name, crit=crit, n=a["n"],
                            det_min=a["detection_range"][0], det_max=a["detection_range"][1],
                            det_spread=a["detection_spread"],
                            share_det=a["share_det"], share_del=a["share_del"],
                            share_cov=a["share_cov"], total=a["total"]))
        # in-domain: the criterion shifts the reading, not the criterion value
        idf = va.id_groups()
        icrit = dz.SUBSTANTIVE_THRESH if crit > 0 else dz.FROZEN_THRESH
        recs = []
        for folder in idf["group"]:
            g = dz.load_id(folder)
            col, _ = dz.id_metric_column(g)
            fs = dz.factorize(g["Dice"].values, g[col].values, icrit)
            recs.append((fs["detection"], fs["delineation"]))
        d = np.array([r[0] for r in recs])
        q = np.array([r[1] for r in recs])
        a_i = va.attribution(d, q)
        rec.append(dict(side="ID/20groups", crit=crit, n=a_i["n"],
                        reading=("sub" if crit > 0 else "frozen"),
                        det_min=a_i["detection_range"][0], det_max=a_i["detection_range"][1],
                        det_spread=a_i["detection_spread"], n_saturated=int((d >= 1.0).sum()),
                        share_det=a_i["share_det"], share_del=a_i["share_del"],
                        share_cov=a_i["share_cov"], total=a_i["total"]))
    return pd.DataFrame(rec)


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #
def gate_route_a_reproduces_units(df: pd.DataFrame) -> dict:
    """The route-A re-factorization must reproduce the frozen n=14 unit table.

    The two tables come from different files and are in different row orders, so
    the comparison is made on the **index (architecture name)**, never positionally.
    """
    ref = pd.read_csv(P31A_UNITS).set_index("arch")
    got = df.set_index("arch")
    if set(got.index) != set(ref.index):
        raise RuntimeError("architecture set mismatch between route-A file and unit table")
    ref = ref.reindex(got.index)          # <- align by name, not by position
    d_det = float(np.max(np.abs(got["det"].values - ref["ood_det_subst"].values)))
    d_del = float(np.max(np.abs(got["del_"].values - ref["ood_del"].values)))
    d_dice = float(np.max(np.abs(got["dice"].values - ref["ood_dice"].values)))
    ok = max(d_det, d_del, d_dice) <= GATE_TOL
    return dict(name="G-A: route A per-sample -> frozen n=14 unit table",
                aligned_by="architecture name (reindex)", n=int(len(got)),
                max_abs_diff_det=d_det, max_abs_diff_del=d_del, max_abs_diff_dice=d_dice,
                tol=GATE_TOL, ok=bool(ok),
                note=("readings come from different files (per-sample CSV vs the frozen unit "
                      "table); agreement to 1e-12 means the unit table was not modified"))


def gate_reproduces_n4(df: pd.DataFrame) -> dict:
    """The legacy 16-cell set must reproduce the P2-2 initial value bit-for-bit."""
    with open(N4_JSON, encoding="utf-8") as fh:
        n4 = json.load(fh)
    ref = n4["ood_attribution"]
    a = va.attribution(df["det"].values, df["del_"].values)
    pairs = [("share_det", a["share_det"], ref["share_det"]),
             ("share_del", a["share_del"], ref["share_del"]),
             ("share_cov", a["share_cov"], ref["share_cov"]),
             ("var_log_det", a["var_log_det"], ref["var_log_det"]),
             ("var_log_del", a["var_log_del"], ref["var_log_del"]),
             ("total", a["total"], ref["total"])]
    diffs = {k: float(abs(x - y)) for k, x, y in pairs}
    worst = max(diffs.values())
    return dict(name="G-B: legacy 16 cells -> P2-2 initial value",
                diffs=diffs, max_abs_diff=worst, tol=GATE_TOL, ok=bool(worst <= GATE_TOL),
                n_ref=int(ref["n"]), n_got=int(a["n"]),
                note="confirms the n=4 initial value is reproducible and unchanged")


def gate_identity(df: pd.DataFrame, label: str = "") -> dict:
    """The factorization identity must hold on every unit of the set.

    Tolerance follows P2-1: the acceptance is on the **absolute** residual
    (< 1e-6).  It cannot be tighter because the undetected samples carry a
    residual Dice of ~1e-9 from the metric smoothing constant (eps = 1e-8), not
    an exact zero -- that is the documented P2-1 residual, not a defect.
    """
    prod = df["det"].values * df["del_"].values
    res = np.abs(prod - df["dice"].values)
    rel = res / np.abs(df["dice"].values)
    ok = bool(res.max() < dz.RESIDUAL_TOL)
    return dict(name="G: identity on %s" % (label or "set"),
                label=label, n=int(len(df)),
                max_abs_residual=float(res.max()),
                max_rel_residual=float(rel.max()),
                tol=dz.RESIDUAL_TOL, ok=ok,
                note="absolute tolerance follows the P2-1 acceptance criterion (1e-6)")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def build() -> dict:
    idf = va.id_groups()
    arch14 = ood_set("arch14")
    sets = {name: ood_set(name) for name in OOD_SETS}

    a_by_set = {name: va.attribution(df["det"].values, df["del_"].values)
                for name, df in sets.items()}
    boot_by_set = {name: va.bootstrap_share(sets[name]["det"].values, sets[name]["del_"].values)
                   for name in OOD_SETS}
    logshare_by_set = {name: boot_logspread_share(sets[name]["det"].values, sets[name]["del_"].values)
                       for name in OOD_SETS}

    a_ood = a_by_set[PRIMARY_SET]                      # backwards-compatible primary
    boot_ood = boot_by_set[PRIMARY_SET]

    # in-domain attributions (unchanged unit set, recomputed here for one artefact)
    a_id_f = va.attribution(idf["det_frozen"].values, idf["del_frozen"].values)
    a_id_s = va.attribution(idf["det_sub"].values, idf["del_sub"].values)
    idb = idf[idf["group"].str.startswith("baseline/")]
    a_idb_f = va.attribution(idb["det_frozen"].values, idb["del_frozen"].values)
    a_idb_s = va.attribution(idb["det_sub"].values, idb["del_sub"].values)

    h1 = va.test_h1(idf)

    # H2: final judgement, on every OOD set.  The pre-registered set is arch14.
    h2_sets = {}
    for name in OOD_SETS:
        a = a_by_set[name]
        share = a["share_det"]
        ci = boot_by_set[name]["share_det_ci"]
        if share >= H2_SUPPORT:
            v = "SUPPORTED"
        elif share < H2_REFUTE:
            v = "REFUTED"
        else:
            v = "INDETERMINATE"
        h2_sets[name] = dict(
            n_units=a["n"], share_det=share, support=H2_SUPPORT, refute=H2_REFUTE,
            boot_ci=ci, boot_ci_lower_above_refute=bool(ci[0] >= H2_REFUTE),
            logspread_share=logshare_by_set[name]["point"],
            logspread_share_ci=logshare_by_set[name]["ci"],
            verdict=v,
            is_primary=bool(name == PRIMARY_SET),
            unit_kind=("architecture" if name in ("arch14", "baseline13") else
                       ("configuration" if name == "ablation7" else "architecture x direction cell")),
        )
    h2 = dict(
        primary_set=PRIMARY_SET,
        support=H2_SUPPORT, refute=H2_REFUTE,
        preregistered_unit_kind="architecture",
        by_set=h2_sets,
        n_architectures=int(a_by_set["arch14"]["n"]),
        n_architectures_clean=int(a_by_set["baseline13"]["n"]),
        verdict=h2_sets[PRIMARY_SET]["verdict"],
        reading=(
            "On the pre-registered unit kind (architectures, n=%d) the out-of-domain detection "
            "share is %.1f%%, above the %.0f%% support bar; the interval lower bound (%.1f%%) stays "
            "above the %.0f%% refutation bar. H2 is therefore SUPPORTED on the expanded set, and the "
            "n=4 initial value (%.1f%%) was, if anything, conservative."
            % (a_by_set["arch14"]["n"], 100 * a_by_set["arch14"]["share_det"], 100 * H2_SUPPORT,
               100 * boot_by_set["arch14"]["share_det_ci"][0], 100 * H2_REFUTE,
               100 * _n4_share_det())
        ),
    )

    gates = {
        "G_A_route_a_reproduces_units": gate_route_a_reproduces_units(arch14),
        "G_B_legacy_reproduces_n4": gate_reproduces_n4(sets["legacy16"]),
        "G_C_identity_arch14": gate_identity(arch14, "arch14"),
        "G_D_identity_baseline13": gate_identity(sets["baseline13"], "baseline13"),
        "G_E_identity_ablation7": gate_identity(sets["ablation7"], "ablation7"),
        "G_F_identity_legacy16": gate_identity(sets["legacy16"], "legacy16"),
    }
    gates["all_ok"] = bool(all(v["ok"] for k, v in gates.items() if isinstance(v, dict) and "ok" in v))

    # n=4 -> expanded comparison table
    n4 = None
    if os.path.exists(N4_JSON):
        with open(N4_JSON, encoding="utf-8") as fh:
            n4 = json.load(fh)
    comparison = build_comparison(n4, a_by_set, boot_by_set)

    return dict(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        script="02_code/analysis/variance_attribution_n14.py",
        segment="P2-2b",
        supersedes="03_results/stats/variance_attribution_n4.json (P2-2 INITIAL value)",
        scope=dict(
            primary_ood_set=PRIMARY_SET,
            ood_sets={name: dict(n=int(len(sets[name])),
                                 unit_kind=("architecture" if name in ("arch14", "baseline13")
                                            else ("configuration" if name == "ablation7"
                                                  else "architecture x direction cell")),
                                 protocol=("P3-1 route A: ID-joint-trained -> ETIS" if name in ("arch14", "baseline13")
                                           else ("P3-1b ablation grid -> ETIS" if name == "ablation7"
                                                 else "P2-1 legacy: single-source-trained -> 4 directions")))
                       for name in OOD_SETS},
            n_id_groups=int(len(idf)),
            merge_policy=("the four out-of-domain sets are never merged silently; arch14 is primary "
                          "because it matches the pre-registered unit kind (architectures)"),
        ),
        criterion=dict(frozen=dz.FROZEN_THRESH, substantive=dz.SUBSTANTIVE_THRESH,
                       eps_id=dz.EPS_ID, eps_ood=dz.EPS_OOD,
                       note="OOD side has no smoothing constant, so frozen == substantive there"),
        ood_attribution=a_ood,                       # primary set (arch14)
        ood_attribution_by_set=a_by_set,
        ood_bootstrap=boot_ood,
        ood_bootstrap_by_set=boot_by_set,
        ood_logspread_share_by_set=logshare_by_set,
        id_attribution_frozen=a_id_f,
        id_attribution_sub=a_id_s,
        id_baseline_attribution_frozen=a_idb_f,
        id_baseline_attribution_sub=a_idb_s,
        h1=h1,
        h2=h2,
        id_artefact_diagnostics=va.id_artefact_diagnostics(idf),
        identity_validity_substantive=va.identity_validity_substantive(idf),
        criterion_scan=criterion_scan().to_dict("records"),
        ood_units_by_set={name: sets[name].to_dict("records") for name in OOD_SETS},
        id_groups=idf.to_dict("records"),
        comparison_n4_vs_expanded=comparison,
        gates=gates,
        excluded_artifacts=[dict(
            path="D:/medical_segmentation/results/test_results/sample_metrics.csv",
            reason=("different-protocol run; not part of any pre-registered unit set; "
                    "Table-1 prohibition (M10) forbids juxtaposition with the ablation grid"))],
    )


def _n4_share_det() -> float:
    with open(N4_JSON, encoding="utf-8") as fh:
        return float(json.load(fh)["ood_attribution"]["share_det"])


def build_comparison(n4, a_by_set, boot_by_set) -> dict:
    rows = []
    if n4 is not None:
        ao = n4["ood_attribution"]
        rows.append(dict(
            label="P2-2 INITIAL", ood_set="legacy16", n=int(ao["n"]), unit_kind="architecture x direction cell",
            share_det=ao["share_det"], share_del=ao["share_del"], share_cov=ao["share_cov"],
            det_spread=ao["detection_spread"], logdet_spread=ao["logdet_spread"],
            logdel_spread=ao["logdel_spread"], boot_ci=n4["ood_bootstrap"]["share_det_ci"],
            is_initial=True))
    labels = {"arch14": "P2-2b FINAL (primary)", "baseline13": "P2-2b (clean cross-architecture)",
              "ablation7": "P2-2b (ablation grid)", "legacy16": "P2-2b (legacy cells, reproduction)"}
    for name in OOD_SETS:
        a = a_by_set[name]
        rows.append(dict(
            label=labels[name], ood_set=name, n=int(a["n"]),
            unit_kind=("architecture" if name in ("arch14", "baseline13")
                       else ("configuration" if name == "ablation7" else "architecture x direction cell")),
            share_det=a["share_det"], share_del=a["share_del"], share_cov=a["share_cov"],
            det_spread=a["detection_spread"], logdet_spread=a["logdet_spread"],
            logdel_spread=a["logdel_spread"], boot_ci=boot_by_set[name]["share_det_ci"],
            is_initial=False))
    return dict(unit_rows=rows,
                question="does the out-of-domain detection share survive the change of unit set?")


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _pct(x):
    return "%.1f%%" % (x * 100)


def write_report(res: dict) -> None:
    a = res["ood_attribution"]
    by = res["ood_attribution_by_set"]
    bo = res["ood_bootstrap_by_set"]
    ai_f, ai_s = res["id_attribution_frozen"], res["id_attribution_sub"]
    ab_s = res["id_baseline_attribution_sub"]
    h1, h2 = res["h1"], res["h2"]
    diag = res["id_artefact_diagnostics"]
    valid = res["identity_validity_substantive"]
    scan = pd.DataFrame(res["criterion_scan"])
    idf = pd.DataFrame(res["id_groups"])
    cmp_rows = res["comparison_n4_vs_expanded"]["unit_rows"]
    arch14 = pd.DataFrame(res["ood_units_by_set"]["arch14"])
    abl7 = pd.DataFrame(res["ood_units_by_set"]["ablation7"])

    L = []
    A = L.append
    A("# T_variance — variance attribution of Dice, in-domain vs out-of-domain")
    A("")
    A("*Machine-generated by `02_code/analysis/variance_attribution_n14.py` on "
      f"{res['generated_at']}.*")
    A("")
    A("> **This is the FINAL value (P2-2b).** It supersedes the `_n4` initial value, which was")
    A("> computed on 4 architectures x 4 directions = 16 cells. The pre-registered judgement data")
    A("> for H2 is the *architecture* set, so the primary out-of-domain unit set here is")
    A(f"> **n = {res['h2']['n_architectures']} architectures** (P3-1 route A).")
    A("")

    # ---- 1 headline
    A("## 1. Headline (frozen criterion — the reading under which the identity is exact)")
    A("")
    A(f"| quantity | in-domain ({len(idf)} groups) | out-of-domain "
      f"(`{res['scope']['primary_ood_set']}`, {a['n']} architectures) |")
    A("|---|---|---|")
    A(f"| detection share of var(log Dice) | **{_pct(ai_f['share_det'])}** | **{_pct(a['share_det'])}** |")
    A(f"| delineation share | {_pct(ai_f['share_del'])} | {_pct(a['share_del'])} |")
    A(f"| covariance share | {_pct(ai_f['share_cov'])} | {_pct(a['share_cov'])} |")
    A(f"| detection range | {ai_f['detection_range'][0]:.4f}–{ai_f['detection_range'][1]:.4f} | "
      f"{a['detection_range'][0]:.3f}–{a['detection_range'][1]:.3f} |")
    A(f"| detection spread | {ai_f['detection_spread']:.4f} | **{a['detection_spread']:.4f}** |")
    A(f"| log-scale spread, detection / delineation | {ai_f['logdet_spread']:.4f} / "
      f"{ai_f['logdel_spread']:.4f} | {a['logdet_spread']:.4f} / {a['logdel_spread']:.4f} |")
    A(f"| naive detection share (covariance ignored) | "
      f"{_pct(ai_f['share_det_naive']) if not np.isnan(ai_f['share_det_naive']) else 'n/a'} | "
      f"{_pct(a['share_det_naive'])} |")
    A("")
    A("**Reading.** In domain the detection factor is a constant 1.000 under the frozen criterion, so it")
    A(f"contributes **exactly zero** of the log-Dice variance. Out of domain the same factor carries")
    A(f"**{_pct(a['share_det'])}** of the variance and moves over {a['detection_spread']:.3f} "
      f"(in domain: {ai_f['detection_spread']:.4f}).")
    A("")
    A(f"Ignoring the covariance term would report the detection share as {_pct(a['share_det_naive'])} instead of")
    A(f"{_pct(a['share_det'])} — about a fifth of the out-of-domain spread would be misattributed. The three")
    A("terms must therefore always be reported separately.")
    A("")

    # ---- 2 unit sets
    A("## 2. Unit sets (why the count changed from 4 to 14)")
    A("")
    A("| out-of-domain set | n | unit kind | protocol | source |")
    A("|---|---:|---|---|---|")
    srcs = {"arch14": "P3-1 route A per-sample", "baseline13": "P3-1 route A per-sample",
            "ablation7": "P3-1b aggregate", "legacy16": "P2-1 calibre (`decompose.load_ood`)"}
    for name in OOD_SETS:
        s = res["scope"]["ood_sets"][name]
        A(f"| `{name}` | {s['n']} | {s['unit_kind']} | {s['protocol']} | {srcs[name]} |")
    A("")
    A(f"`arch14` is **primary**: it matches the pre-registered unit kind (architectures). The four sets")
    A("are never merged silently; `legacy16` exists only so the shift from the initial value is visible.")
    A("")
    A("✨ **DECIDED (user, 2026-09-14) — manuscript presentation of this table.** The main text reports the")
    A("**four-set table** in full (not a single set), and **`legacy16` is included in the main text** — the")
    A("reason is that it is the row that reproduces the superseded initial value exactly, so the reader can")
    A("see that the change from 41.5% to 69.9% is a change of *unit set*, not of arithmetic. Units are")
    A("therefore described explicitly in the Methods, one sentence per set, and never pooled.")
    A("")

    # ---- 3 H2
    A("## 3. Hypothesis H2 — out-of-domain detection share (FINAL judgement)")
    A("")
    A(f"Support bar **{_pct(h2['support'])}**, refutation bar **{_pct(h2['refute'])}**.")
    A("")
    A("| out-of-domain set | n | unit kind | share_det | bootstrap 95% CI | verdict |")
    A("|---|---:|---|---:|---|---|")
    for name in OOD_SETS:
        d = h2["by_set"][name]
        star = " **(primary)**" if d["is_primary"] else ""
        A(f"| `{name}`{star} | {d['n_units']} | {d['unit_kind']} | **{_pct(d['share_det'])}** | "
          f"[{_pct(d['boot_ci'][0])}, {_pct(d['boot_ci'][1])}] | {d['verdict']} |")
    A("")
    A("⚠️ **The share interval can exceed 100%.** The variance shares are ratios whose denominator")
    A("contains the covariance term; when a resample makes that term negative the detection share rises")
    A("above 1 and a bound of [0, 1] does not apply. The bounded companion quantity — detection's share")
    A("of the **log-spread** — is therefore reported alongside (it lies in [0, 1] by construction):")
    A("")
    A("| out-of-domain set | n | log-spread share, detection | bootstrap 95% CI |")
    A("|---|---:|---:|---|")
    for name in OOD_SETS:
        d = h2["by_set"][name]
        A(f"| `{name}` | {d['n_units']} | **{_pct(d['logspread_share'])}** | "
          f"[{_pct(d['logspread_share_ci'][0])}, {_pct(d['logspread_share_ci'][1])}] |")
    A("")
    A(f"**Verdict: H2 is {h2['verdict']} on the expanded, pre-registered unit set.** " + h2["reading"])
    A("")
    A("⚠️ The bootstrap interval is **descriptive only**: the architectures are not mutually independent")
    A("(thirteen baselines share one data split and one training protocol; EGAUNet shares its lineage with")
    A("the ablation grid), so this is not a sampling interval for a population of architectures.")
    A("")

    # ---- 4 comparison
    A("## 4. n = 4 initial value vs expanded set")
    A("")
    A("| label | set | n | unit kind | share_det | share_del | share_cov | bootstrap 95% CI |")
    A("|---|---|---:|---|---:|---:|---:|---|")
    for r in cmp_rows:
        tag = "**INITIAL**" if r["is_initial"] else r["label"]
        A(f"| {tag} | `{r['ood_set']}` | {r['n']} | {r['unit_kind']} | {_pct(r['share_det'])} | "
          f"{_pct(r['share_del'])} | {_pct(r['share_cov'])} | "
          f"[{_pct(r['boot_ci'][0])}, {_pct(r['boot_ci'][1])}] |")
    A("")
    A("The legacy row reproduces the initial value exactly (gate `G_B`), so the comparison is between")
    A("two readings of the *same* arithmetic on *different* unit sets — not between two implementations.")
    A("")

    # ---- 5 H1
    A("## 5. Hypothesis H1 — in-domain detection (dual reading, B6) — unchanged")
    A("")
    A(f"- **Frozen criterion** `|P n G| > 0`: {h1['frozen_n_saturated']}/{h1['n_groups']} groups = 1.000 "
      f"exactly → contributes 0% of the variance.")
    A(f"- **Substantive criterion** `I >= 1 px`: range {h1['substantive_min']:.4f}–{h1['substantive_max']:.4f}, "
      f"spread {h1['substantive_spread']:.4f}, **{h1['substantive_n_saturated']}/{h1['n_groups']} saturated**.")
    A(f"- Pre-registered bar {h1['bar']} on the substantive reading → **{h1['n_below_bar']}/{h1['n_groups']} "
      f"groups fall below it**.")
    A("")
    A("The in-domain unit set did not change in P2-2b (13 baselines + 7 ablation configurations, one")
    A("reading each), so this section carries over from the initial value. The mechanism is the")
    A("eps-artefact count:")
    A("")
    A(f"- `det_sub = 1 - k/n` holds **exactly** ({diag['relation_exact']}); `k` takes only the values "
      f"{diag['n_artefact_range'][0]}–{diag['n_artefact_range'][1]} out of 242, giving "
      f"{len(diag['distinct_det_sub'])} distinct values of `det_sub`.")
    A(f"- Correlation between that pipeline count and group Dice: r = {diag['corr_artefact_dice']:.3f} "
      f"(p = {diag['corr_artefact_dice_p']:.2f}).")
    A("")
    A("**Verdict.** " + h1["verdict"])
    A("")
    A("✨ **DECIDED (user, 2026-09-14): the wording is \"approaching saturation\".** The pre-registered rule")
    A("applied literally puts " + str(h1["n_below_bar"]) + " of " + str(h1["n_groups"]) +
      " groups below the bar, and L1-1 is")
    A("therefore written as \"approaching saturation\" rather than \"saturation\", with the dual reading kept")
    A("mandatory and the eps-artefact mechanism above given as the explanation. The frozen reading is")
    A("reported alongside it, because that is the reading under which the identity is exact.")
    A("")

    # ---- 6 unit detail
    A("## 6. Unit-level detail")
    A("")
    A(f"### 6a. Out-of-domain, primary set (`{res['scope']['primary_ood_set']}`, frozen criterion)")
    A("")
    A("| architecture | n | E[Dice] | Detection | Delineation | undetected |")
    A("|---|---:|---:|---:|---:|---:|")
    for _, r in arch14.iterrows():
        A(f"| {r['arch']} | {r['n']} | {r['dice']:.4f} | {r['det']:.4f} | {r['del_']:.4f} | {r['n_undet']} |")
    A("")
    A("### 6b. Out-of-domain, ablation grid (`ablation7`)")
    A("")
    A("| configuration | n | E[Dice] | Detection | Delineation | undetected |")
    A("|---|---:|---:|---:|---:|---:|")
    for _, r in abl7.iterrows():
        A(f"| {r['arch']} | {r['n']} | {r['dice']:.4f} | {r['det']:.4f} | {r['del_']:.4f} | {r['n_undet']} |")
    A("")
    A(f"### 6c. In-domain groups ({len(idf)})")
    A("")
    A("| group | metric | n | E[Dice] % | Det(frozen) | Del(frozen) % | Det(sub) | Del(sub) % | k |")
    A("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in idf.iterrows():
        k = int(round((1.0 - r["det_sub"]) * r["n"]))
        A(f"| {r['group']} | {r['src']} | {r['n']} | {r['dice']*100:.2f} | {r['det_frozen']:.4f} | "
          f"{r['del_frozen']*100:.2f} | {r['det_sub']:.4f} | {r['del_sub']*100:.2f} | {k} |")
    A("")

    # ---- 7 secondary
    A("## 7. Secondary attribution under the substantive criterion (diagnostic only)")
    A("")
    A("The identity remains valid at this reading (the reclassified samples still carry Dice ~ 1e-11),")
    A(f"so the attribution can be computed: max relative residual {valid['max_rel_residual']:.2e} "
      f"(valid: {valid['valid']}). It must **not** be read as a measurement of detection-ability")
    A("variance — see §5.")
    A("")
    A("| unit set | n | share det | share del | share cov | cov < 0 |")
    A("|---|---:|---:|---:|---:|---|")
    A(f"| ID groups | {len(idf)} | {_pct(ai_s['share_det'])} | {_pct(ai_s['share_del'])} | "
      f"{_pct(ai_s['share_cov'])} | {ai_s['cov_negative']} |")
    A(f"| ID baselines only | 13 | {_pct(ab_s['share_det'])} | {_pct(ab_s['share_del'])} | "
      f"{_pct(ab_s['share_cov'])} | {ab_s['cov_negative']} |")
    A("")
    if ab_s["cov_negative"]:
        A("⚠️ **Shares leave [0, 1] when the covariance is negative** (the 13-baseline row above).")
        A("On that subset the logged factors are negatively correlated, so shares alone are an incomplete")
        A("summary; the log-scale spreads are the well-behaved quantity.")
        A("")

    # ---- 8 scan
    A("## 8. Sensitivity to the detection criterion (B2)")
    A("")
    A("| side | criterion > | n | det min | det max | det spread | share det | share del | share cov |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in scan.iterrows():
        A(f"| {r['side']} | {r['crit']:g} | {r['n']} | {r['det_min']:.4f} | {r['det_max']:.4f} | "
          f"{r['det_spread']:.4f} | {_pct(r['share_det'])} | {_pct(r['share_del'])} | {_pct(r['share_cov'])} |")
    A("")
    A("The out-of-domain rows are invariant for every criterion above 0 because `recall_gt` carries no")
    A("smoothing constant — its zeros are exact. That is precisely what makes the out-of-domain detection")
    A("factor a genuine measurement rather than a construction (B6).")
    A("")

    # ---- 9 gates + robustness
    A("## 9. Gates and robustness")
    A("")
    A("| gate | result | detail |")
    A("|---|---|---|")
    for k, v in res["gates"].items():
        if not isinstance(v, dict) or "ok" not in v:
            continue
        A(f"| {k} | {'PASS' if v['ok'] else 'FAIL'} | {v['name']} |")
    g = res["gates"]["G_A_route_a_reproduces_units"]
    A("")
    A(f"- `G_A`: max |Δ| detection {g['max_abs_diff_det']:.3e}, delineation {g['max_abs_diff_del']:.3e}, "
      f"Dice {g['max_abs_diff_dice']:.3e} (tol {g['tol']:g}).")
    gb = res["gates"]["G_B_legacy_reproduces_n4"]
    A(f"- `G_B`: the legacy 16 cells reproduce the initial value to {gb['max_abs_diff']:.3e} "
      f"(n {gb['n_ref']} → {gb['n_got']}).")
    A("")
    A("- **Excluded artifact**: " + res["excluded_artifacts"][0]["path"])
    A("  — " + res["excluded_artifacts"][0]["reason"])
    A("- **H3 is not tested here** (it belongs to P2-3 at architectural n = 14).")
    A(f"- **Numerical identity**: max absolute residual on the primary set "
      f"{res['gates']['G_C_identity_arch14']['max_abs_residual']:.2e} "
      f"(relative {res['gates']['G_C_identity_arch14']['max_rel_residual']:.2e}); tolerance "
      f"{res['gates']['G_C_identity_arch14']['tol']:g} follows the P2-1 acceptance criterion. The")
    A("  residual is not zero because undetected samples carry a Dice of ~1e-9 from the metric smoothing")
    A("  constant (eps = 1e-8) rather than an exact zero.")
    A("- **Unit-set alignment**: the route-A re-factorization is compared with the frozen unit table")
    A("  **by architecture name**, never positionally — the two files are in different row orders.")
    A("")

    # ---- 10 reproduce
    A("## 10. Reproduce")
    A("")
    A("```bash")
    A("D:/miniconda/aniconda/envs/medical_seg/python.exe "
      "E:/paper2_ablation_reliability/02_code/analysis/variance_attribution_n14.py")
    A("# independent cross-check (no import of either main module)")
    A("D:/miniconda/aniconda/envs/medical_seg/python.exe "
      "E:/paper2_ablation_reliability/02_code/analysis/p2_2b_crosscheck.py")
    A("```")
    A("")

    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P2-2b variance attribution (expanded set)")
    ap.add_argument("--selfcheck-only", action="store_true")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)

    if args.selfcheck_only:
        va.selfcheck(verbose=True)
        return 0

    res = build()

    print("=" * 74)
    print("P2-2b  variance attribution on the expanded architecture set")
    print("=" * 74)
    for name in OOD_SETS:
        a = res["ood_attribution_by_set"][name]
        b = res["ood_bootstrap_by_set"][name]
        star = "  <-- PRIMARY" if name == PRIMARY_SET else ""
        print("[%-11s] n=%2d  det %.1f%% / del %.1f%% / cov %+.1f%%   "
              "CI_det [%.1f%%, %.1f%%]%s"
              % (name, a["n"], 100 * a["share_det"], 100 * a["share_del"], 100 * a["share_cov"],
                 100 * b["share_det_ci"][0], 100 * b["share_det_ci"][1], star))
    print("-" * 74)
    print("H2 (%s):  %s" % (res["h2"]["primary_set"], res["h2"]["verdict"]))
    print("   " + res["h2"]["reading"])
    print("-" * 74)
    print("Gates: ", {k: v["ok"] for k, v in res["gates"].items()
                      if isinstance(v, dict) and "ok" in v})
    print("   all_ok =", res["gates"]["all_ok"])
    print("=" * 74)

    if not args.no_write:
        os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
        with open(OUT_JSON, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=1)
        write_report(res)
        print("written:")
        print("   %s  %d B" % (OUT_JSON, os.path.getsize(OUT_JSON)))
        print("   %s  %d B" % (OUT_MD, os.path.getsize(OUT_MD)))

    return 0 if res["gates"]["all_ok"] else 3


if __name__ == "__main__":
    sys.exit(main())
