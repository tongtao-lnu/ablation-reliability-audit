# -*- coding: utf-8 -*-
"""Independent recomputation of the P2-1 numbers.

Deliberately written from scratch (no import of ``decompose.py``) so that the
delivered artefacts are cross-checked by an independent code path.
"""
import glob
import os
import sys

import pandas as pd

OOD = "E:/paper2_ablation_reliability/03_results/raw/ood_per_sample/paper2_per_sample.csv"
CSV = "E:/paper2_ablation_reliability/03_results/stats/decomposition.csv"

d = pd.read_csv(OOD)
res = []
for (dr, ar), g in d.groupby(["dir", "arch"]):
    m = g["recall_gt"].values > 0
    resid = g["dice"].mean() - (m.mean() * g.loc[m, "dice"].mean() if m.any() else 0.0)
    res.append((f"{dr}/{ar}", resid))
mx_ood = max(abs(v) for _, v in res)

id_res = []
for f in sorted(
    glob.glob("D:/medical_segmentation/experiments/baseline/*/sample_metrics.csv")
    + glob.glob("D:/medical_segmentation/experiments/ablation/*/sample_metrics.csv")
):
    g = pd.read_csv(f)
    col = "Recall" if "Recall" in g.columns else "Dice"
    m = g[col].values > 0
    resid = g["Dice"].mean() - (m.mean() * g.loc[m, "Dice"].mean() if m.any() else 0.0)
    id_res.append((os.path.basename(os.path.dirname(f)), resid))
mx_id = max(abs(v) for _, v in id_res)

deliv = pd.read_csv(CSV)
mx_deliv = deliv["residual"].abs().max()

print(f"independent      : OOD cells {len(res)}, ID groups {len(id_res)}")
print(f"independent      : max|resid| OOD = {mx_ood:.6e}  ID = {mx_id:.6e}  overall = {max(mx_ood, mx_id):.6e}")
print(f"decomposition.csv: rows {len(deliv)}, max|resid| = {mx_deliv:.6e}")
print(f"CSV columns      : {list(deliv.columns)}")
ok = abs(max(mx_ood, mx_id) - mx_deliv) < 1e-15
print(f"agreement        : {'MATCH' if ok else 'MISMATCH'}")
print(f"tolerance 1e-6   : {'PASS' if max(mx_ood, mx_id) < 1e-6 else 'FAIL'}")
sys.exit(0 if ok and max(mx_ood, mx_id) < 1e-6 else 1)
