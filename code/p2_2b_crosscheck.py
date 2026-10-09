# -*- coding: utf-8 -*-
"""
p2_2b_crosscheck.py -- G4 independent re-computation for P2-2b.

Independence contract
---------------------
This script **imports neither** `variance_attribution.py`, **nor**
`variance_attribution_n14.py`, **nor** `decompose.py`.  It:

* re-reads the raw per-sample files (route-A CSV, the 16-cell calibre CSV, and the
  20 in-domain `sample_metrics.csv` files straight off D:),
* re-factorizes detection / delineation from scratch,
* re-does the variance split with an explicit two-pass formula
  (``sum((x-mx)(y-my))/(n-1)``) instead of ``ndarray.var`` / ``np.cov``,
* opens only files that are read-only.

Every compared quantity is listed with both readings and the absolute difference.
Exit code 0 iff every item matches.

Run
---
D:/miniconda/aniconda/envs/medical_seg/python.exe 02_code/analysis/p2_2b_crosscheck.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

E_ROOT = "E:/paper2_ablation_reliability"
ID_ROOT = "D:/medical_segmentation/experiments"

JSON_MAIN = E_ROOT + "/03_results/stats/variance_attribution.json"
JSON_N4 = E_ROOT + "/03_results/stats/variance_attribution_n4.json"
P31A_PER_SAMPLE = E_ROOT + "/03_results/raw/p31a_ood_idtrain/per_sample.csv"
P31A_UNITS = E_ROOT + "/03_results/stats/p31a_units_n14.csv"
ABLATION_ETIS = E_ROOT + "/03_results/stats/decomp_ablation_etis.csv"
LEGACY_OOD = E_ROOT + "/03_results/raw/ood_per_sample/paper2_per_sample.csv"

FROZEN = 0.0
SUBST = 1e-6

ID_BASELINE_ARCHS = ["AttentionUNet", "CaraNet", "M2SNet", "MultiResUNet", "PSPNet",
                     "PolypPVT", "PraNet", "ResUNet", "SANet", "SegNet", "TransUNet",
                     "UACANet", "UNet"]
ID_ABLATION_CFGS = ["baseline", "egm_only", "dpa_only", "msfa_only", "egm_dpa",
                    "egm_msfa", "dpa_msfa"]

TOL = 1e-12
RESULTS = []


def check(label, got, ref, tol=TOL):
    d = float(abs(got - ref))
    ok = bool(d <= tol)
    RESULTS.append(dict(label=label, got=float(got), ref=float(ref), abs_diff=d, tol=tol, ok=ok))
    return ok


def two_pass_split(det, del_):
    """Three-term variance split, written out longhand (independent arithmetic path)."""
    lt = np.log(np.asarray(det, dtype=float))
    ll = np.log(np.asarray(del_, dtype=float))
    n = lt.size
    m1 = lt.sum() / n
    m2 = ll.sum() / n
    vt = ((lt - m1) ** 2).sum() / (n - 1)
    vl = ((ll - m2) ** 2).sum() / (n - 1)
    cov = ((lt - m1) * (ll - m2)).sum() / (n - 1)
    total = vt + vl + 2 * cov
    ld = lt + ll
    m3 = ld.sum() / n
    vd = ((ld - m3) ** 2).sum() / (n - 1)
    sd = float(lt.max() - lt.min())
    sl = float(ll.max() - ll.min())
    return dict(n=int(n), var_log_det=float(vt), var_log_del=float(vl), cov2=float(2 * cov),
                total=float(total), share_det=float(vt / total), share_del=float(vl / total),
                share_cov=float(2 * cov / total),
                detection_spread=float(np.asarray(det).max() - np.asarray(det).min()),
                logdet_spread=sd, logdel_spread=sl,
                logspread_share=float(sd / (sd + sl)) if (sd + sl) > 0 else float("nan"),
                var_log_dice_direct=float(vd), identity_gap=float(vd - total))


def factorize_units(dice, metric, thr):
    dice = np.asarray(dice, dtype=float)
    met = np.asarray(metric, dtype=float)
    m = met > thr
    return dict(det=float(m.mean()),
                del_=float(dice[m].mean()) if m.any() else float("nan"),
                dice=float(dice.mean()), n_undet=int((~m).sum()))


# --------------------------------------------------------------------------- #
# Unit tables re-derived from raw files
# --------------------------------------------------------------------------- #
def ood_route_a(crit=FROZEN):
    ps = pd.read_csv(P31A_PER_SAMPLE)
    out = {}
    for ar, g in ps.groupby("arch", sort=True):
        out[ar] = factorize_units(g["dice"].values, g["recall_gt"].values, crit)
    return out


def ood_legacy(crit=FROZEN):
    od = pd.read_csv(LEGACY_OOD)
    out = {}
    for (dr, ar), g in od.groupby(["dir", "arch"], sort=True):
        out["%s/%s" % (dr, ar)] = factorize_units(g["dice"].values, g["recall_gt"].values, crit)
    return out


def ood_ablation():
    ab = pd.read_csv(ABLATION_ETIS)
    return {r["config"]: dict(det=float(r["detection_sub"]), del_=float(r["delineation_sub"]),
                              dice=float(r["mean_dice"]), n_undet=int(r["n_undetected_sub"]))
            for _, r in ab.iterrows()}


def id_groups_raw():
    """Re-derive all 20 in-domain groups straight from the D: per-sample CSVs."""
    out = {}
    for folder in (["baseline/" + a for a in ID_BASELINE_ARCHS]
                   + ["ablation/" + c for c in ID_ABLATION_CFGS]):
        p = os.path.join(ID_ROOT, folder, "sample_metrics.csv")
        d = pd.read_csv(p)
        col = "Recall" if "Recall" in d.columns else "Dice"
        dice = d["Dice"].values
        met = d[col].values
        fz = factorize_units(dice, met, FROZEN)
        fs = factorize_units(dice, met, SUBST)
        k = int((met <= SUBST).sum())
        out[folder] = dict(det_frozen=fz["det"], del_frozen=fz["del_"], dice=fz["dice"],
                           det_sub=fs["det"], del_sub=fs["del_"], k=k, n=int(len(d)))
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    with open(JSON_MAIN, encoding="utf-8") as fh:
        J = json.load(fh)
    with open(JSON_N4, encoding="utf-8") as fh:
        J4 = json.load(fh)

    print("=" * 78)
    print("P2-2b  G4 independent cross-check  (no import of any main module)")
    print("=" * 78)

    # ---- 1. structure
    print("\n-- structure --")
    check("json: primary set is arch14", 1.0 if J["scope"]["primary_ood_set"] == "arch14" else 0.0, 1.0)
    check("json: n architectures", float(J["h2"]["n_architectures"]), 14.0, tol=0.0)

    # ---- 2. OOD unit values, re-derived from raw files
    print("\n-- OOD unit values (route A, re-derived) --")
    ra = ood_route_a()
    ref_units = pd.read_csv(P31A_UNITS).set_index("arch")
    worst = 0.0
    for ar, v in ra.items():
        r = ref_units.loc[ar]
        worst = max(worst, abs(v["det"] - r["ood_det_subst"]),
                    abs(v["del_"] - r["ood_del"]), abs(v["dice"] - r["ood_dice"]))
    check("route A: max |unit value - frozen unit table|", worst, 0.0, tol=TOL)

    # ---- 3. attribution on the primary set
    print("\n-- attribution: arch14 (primary) --")
    names = sorted(ra)
    det = np.array([ra[a]["det"] for a in names])
    dls = np.array([ra[a]["del_"] for a in names])
    sp = two_pass_split(det, dls)
    A = J["ood_attribution_by_set"]["arch14"]
    for k in ["var_log_det", "var_log_del", "cov2", "total", "share_det", "share_del",
              "share_cov", "detection_spread", "logdet_spread", "logdel_spread"]:
        check("arch14 %s" % k, sp[k], A[k])
    check("arch14 n", float(sp["n"]), float(A["n"]), tol=0.0)
    check("arch14 log-spread share", sp["logspread_share"],
          J["ood_logspread_share_by_set"]["arch14"]["point"])
    check("arch14 identity gap (|var(logDice) - 3-term|)", abs(sp["identity_gap"]), 0.0, tol=1e-12)

    # ---- 4. baseline13
    print("\n-- attribution: baseline13 --")
    n13 = [a for a in names if a != "EGAUNet"]
    sp13 = two_pass_split(np.array([ra[a]["det"] for a in n13]),
                          np.array([ra[a]["del_"] for a in n13]))
    B = J["ood_attribution_by_set"]["baseline13"]
    for k in ["share_det", "share_del", "share_cov", "var_log_det", "total"]:
        check("baseline13 %s" % k, sp13[k], B[k])
    check("baseline13 n", float(sp13["n"]), float(B["n"]), tol=0.0)

    # ---- 5. ablation7
    print("\n-- attribution: ablation7 --")
    ab = ood_ablation()
    an = sorted(ab)
    sp7 = two_pass_split(np.array([ab[c]["det"] for c in an]),
                         np.array([ab[c]["del_"] for c in an]))
    C = J["ood_attribution_by_set"]["ablation7"]
    for k in ["share_det", "share_del", "share_cov", "var_log_det", "total"]:
        check("ablation7 %s" % k, sp7[k], C[k])
    check("ablation7 n", float(sp7["n"]), float(C["n"]), tol=0.0)
    # the ablation aggregates must reproduce the json's unit rows too
    ju = {r["arch"]: r for r in J["ood_units_by_set"]["ablation7"]}
    w7 = 0.0
    for c in an:
        w7 = max(w7, abs(ab[c]["det"] - ju[c]["det"]), abs(ab[c]["del_"] - ju[c]["del_"]),
                 abs(ab[c]["dice"] - ju[c]["dice"]))
    check("ablation7: unit values", w7, 0.0)

    # ---- 6. legacy16 reproduces the n=4 initial value
    print("\n-- legacy 16 cells -> n=4 initial value --")
    lg = ood_legacy()
    ln = sorted(lg)
    spL = two_pass_split(np.array([lg[u]["det"] for u in ln]),
                         np.array([lg[u]["del_"] for u in ln]))
    A4 = J4["ood_attribution"]
    for k in ["share_det", "share_del", "share_cov", "var_log_det", "total"]:
        check("legacy16 %s vs n4 json" % k, spL[k], A4[k])
    check("legacy16 n vs n4 json", float(spL["n"]), float(A4["n"]), tol=0.0)
    D = J["ood_attribution_by_set"]["legacy16"]
    for k in ["share_det", "share_del", "share_cov", "var_log_det", "total"]:
        check("legacy16 %s vs P2-2b json" % k, spL[k], D[k])

    # ---- 7. in-domain groups, re-derived from D:
    print("\n-- in-domain 20 groups (re-derived from D:) --")
    idg = id_groups_raw()
    ju = {r["group"]: r for r in J["id_groups"]}
    wf = ws = wd = 0.0
    for g, v in idg.items():
        r = ju[g]
        wf = max(wf, abs(v["det_frozen"] - r["det_frozen"]), abs(v["del_frozen"] - r["del_frozen"]))
        ws = max(ws, abs(v["det_sub"] - r["det_sub"]), abs(v["del_sub"] - r["del_sub"]))
        wd = max(wd, abs(v["dice"] - r["dice"]))
    check("ID: max |frozen reading diff|", wf, 0.0)
    check("ID: max |substantive reading diff|", ws, 0.0)
    check("ID: max |mean Dice diff|", wd, 0.0)
    check("ID: n groups", float(len(idg)), 20.0, tol=0.0)

    # ---- 8. in-domain attributions
    print("\n-- attribution: in-domain --")
    gs = sorted(idg)
    spIF = two_pass_split(np.array([idg[g]["det_frozen"] for g in gs]),
                          np.array([idg[g]["del_frozen"] for g in gs]))
    spIS = two_pass_split(np.array([idg[g]["det_sub"] for g in gs]),
                          np.array([idg[g]["del_sub"] for g in gs]))
    spIB = two_pass_split(np.array([idg[g]["det_sub"] for g in gs if g.startswith("baseline/")]),
                          np.array([idg[g]["del_sub"] for g in gs if g.startswith("baseline/")]))
    FI = J["id_attribution_frozen"]
    SI = J["id_attribution_sub"]
    BI = J["id_baseline_attribution_sub"]
    for k in ["share_det", "share_del", "share_cov", "var_log_det", "total"]:
        check("ID frozen %s" % k, spIF[k], FI[k])
    for k in ["share_det", "share_del", "share_cov", "var_log_det", "total"]:
        check("ID substantive %s" % k, spIS[k], SI[k])
    for k in ["share_det", "share_del", "share_cov", "var_log_det", "total"]:
        check("ID baseline-only substantive %s" % k, spIB[k], BI[k])
    check("ID frozen detection is exactly 1.000",
          1.0 if np.all([idg[g]["det_frozen"] == 1.0 for g in gs]) else 0.0, 1.0)

    # ---- 9. artefact relation det_sub = 1 - k/n, exactly
    print("\n-- eps-artefact relation --")
    worst_rel = 0.0
    for g in gs:
        worst_rel = max(worst_rel, abs(idg[g]["det_sub"] - (1.0 - idg[g]["k"] / idg[g]["n"])))
    check("ID: max |det_sub - (1 - k/n)|", worst_rel, 0.0, tol=1e-15)
    ks = sorted({idg[g]["k"] for g in gs})
    check("ID: min k", float(min(ks)), float(J["id_artefact_diagnostics"]["n_artefact_range"][0]), tol=0.0)
    check("ID: max k", float(max(ks)), float(J["id_artefact_diagnostics"]["n_artefact_range"][1]), tol=0.0)
    check("ID: distinct det_sub count", float(len({round(idg[g]["det_sub"], 12) for g in gs})),
          float(len(J["id_artefact_diagnostics"]["distinct_det_sub"])), tol=0.0)

    # ---- 10. H2 verdict, recomputed
    print("\n-- H2 verdict --")
    check("H2 share_det vs support bar (>=35%)",
          1.0 if A["share_det"] >= 0.35 else 0.0,
          1.0 if J["h2"]["by_set"]["arch14"]["verdict"] == "SUPPORTED" else 0.0, tol=0.0)
    ci = J["ood_bootstrap_by_set"]["arch14"]["share_det_ci"]
    check("H2 CI lower bound above refutation bar (20%)",
          1.0 if ci[0] >= 0.20 else 0.0, 1.0, tol=0.0)

    # ---- 11. report
    n_ok = sum(1 for r in RESULTS if r["ok"])
    n_bad = len(RESULTS) - n_ok
    print("\n" + "-" * 78)
    print("ITEMS COMPARED: %d    MATCH: %d    DIFF: %d" % (len(RESULTS), n_ok, n_bad))
    if n_bad:
        print("\nDIFFERENCES:")
        for r in RESULTS:
            if not r["ok"]:
                print("  %-52s got %.17g  ref %.17g  |d| %.3e  tol %.1e"
                      % (r["label"], r["got"], r["ref"], r["abs_diff"], r["tol"]))
    print("-" * 78)
    print("OVERALL: %s" % ("ALL MATCH" if n_bad == 0 else "MISMATCH"))
    print("=" * 78)

    out = dict(script="02_code/analysis/p2_2b_crosscheck.py",
               independent_of=["variance_attribution.py", "variance_attribution_n14.py",
                               "decompose.py"],
               inputs=[P31A_PER_SAMPLE, P31A_UNITS, ABLATION_ETIS, LEGACY_OOD,
                       JSON_MAIN, JSON_N4, "%s/<folder>/sample_metrics.csv" % ID_ROOT],
               n_items=len(RESULTS), n_match=n_ok, n_diff=n_bad,
               status=("ALL_MATCH" if n_bad == 0 else "MISMATCH"),
               items=RESULTS)
    outp = E_ROOT + "/03_results/audit/p2_2b_crosscheck.json"
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with open(outp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print("written: %s  %d B" % (outp, os.path.getsize(outp)))
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
