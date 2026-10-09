# -*- coding: utf-8 -*-
"""Independent cross-check of the P2-2 variance attribution.

Written from scratch: it does **not** import ``variance_attribution.py`` or
``decompose.py``.  It recomputes the out-of-domain cell factors, the three
variance terms and the shares with plain pandas/numpy, then compares them against
the delivered JSON.  Exits non-zero on any mismatch, so it can gate the segment.
"""
import json
import sys

import numpy as np
import pandas as pd

OOD = "E:/paper2_ablation_reliability/03_results/raw/ood_per_sample/paper2_per_sample.csv"
ID_ROOT = "D:/medical_segmentation/experiments"
JSON = "E:/paper2_ablation_reliability/03_results/stats/variance_attribution_n4.json"

BASELINES = ["AttentionUNet", "CaraNet", "M2SNet", "MultiResUNet", "PSPNet", "PolypPVT",
             "PraNet", "ResUNet", "SANet", "SegNet", "TransUNet", "UACANet", "UNet"]
ABLATIONS = ["baseline", "egm_only", "dpa_only", "msfa_only", "egm_dpa", "egm_msfa", "dpa_msfa"]


def shares(det, dl):
    lt, ll = np.log(det), np.log(dl)
    vt, vl = lt.var(ddof=1), ll.var(ddof=1)
    c2 = 2 * np.cov(lt, ll, ddof=1)[0, 1]
    tot = vt + vl + c2
    return vt / tot, vl / tot, c2 / tot, tot


# ---- out of domain: 16 cells -------------------------------------------------
d = pd.read_csv(OOD)
det, dl = [], []
for (dr, ar), g in d.groupby(["dir", "arch"]):
    m = g["recall_gt"].values > 0
    det.append(m.mean())
    dl.append(g["dice"].values[m].mean())
s_det, s_del, s_cov, tot = shares(np.array(det), np.array(dl))

# ---- in domain: 20 groups, frozen criterion only ------------------------------
idet, idl = [], []
for folder in [f"baseline/{a}" for a in BASELINES] + [f"ablation/{c}" for c in ABLATIONS]:
    g = pd.read_csv(f"{ID_ROOT}/{folder}/sample_metrics.csv")
    col = "Recall" if "Recall" in g.columns else "Dice"
    m = g[col].values > 0
    idet.append(m.mean())
    idl.append(g["Dice"].values[m].mean())
i_det, i_del, i_cov, i_tot = shares(np.array(idet), np.array(idl))

# ---- compare against the delivered artefacts ---------------------------------
r = json.load(open(JSON, encoding="utf-8"))
a, b = r["ood_attribution"], r["id_attribution_frozen"]

checks = [
    ("OOD n cells", r["ood_attribution"]["n"], len(det)),
    ("OOD share_det", a["share_det"], s_det),
    ("OOD share_del", a["share_del"], s_del),
    ("OOD share_cov", a["share_cov"], s_cov),
    ("OOD total", a["total"], tot),
    ("ID n groups", b["n"], len(idet)),
    ("ID share_det", b["share_det"], i_det),
    ("ID share_del", b["share_del"], i_del),
    ("ID share_cov", b["share_cov"], i_cov),
]
ok = True
print(f"{'check':<18}{'delivered':>22}{'independent':>22}   {'ok':>3}")
for name, got, exp in checks:
    good = abs(float(got) - float(exp)) < 1e-12
    ok &= good
    print(f"{name:<18}{float(got):>22.15f}{float(exp):>22.15f}   {'Y' if good else 'N':>3}")

print()
print(f"  independently: OOD share_det = {s_det*100:.2f}%  (delivered {a['share_det']*100:.2f}%)")
print(f"  independently: ID  share_det = {i_det*100:.2f}%   (delivered {b['share_det']*100:.2f}%)")
print(f"  frozen headline 41.5% reproduced: {abs(s_det - 0.415) < 0.005}")
print(f"  shares sum to 1 (OOD): {abs(s_det + s_del + s_cov - 1) < 1e-12}")
print(f"  shares sum to 1 (ID) : {abs(i_det + i_del + i_cov - 1) < 1e-12}")
print(f"  H2 support (>=0.35)  : {'PASS' if s_det >= 0.35 else 'FAIL'}")
print()
print("RESULT:", "MATCH" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
