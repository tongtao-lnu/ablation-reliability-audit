#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
P3-6 派生量：**ID−OOD 均值差**在两种口径下的变化（gap compression / sign flip）

为什么单独算：段内排名一致性（Spearman ρ 逐单元）保持，并不等于"口径无偏"。
真正决定论文结论的是**跨集合的均值差**（域间差距）。本脚本从
`03_results/stats/caliber_swap.json` 与 `fragmentation.json` 直接读取，
计算 12 个条件下：

    gap_F = E_F[ID] − E_F[OOD]        （与 F 口径的域间差距，恒定）
    gap_V = E_V[ID] − E_V[OOD]        （V 口径下的域间差距）
    compression = 1 − gap_V / gap_F   （压缩比；>0 压缩，<0 放大）
    sign_flip   = sign(gap_F) != sign(gap_V)

跑法
----
D:/miniconda/aniconda/envs/medical_seg/python.exe 02_code/analysis/p3_6_gap_derivation.py
"""

from __future__ import annotations

import json
import os
import sys

E_ROOT = "E:/paper2_ablation_reliability"
IN_SWAP = E_ROOT + "/03_results/stats/caliber_swap.json"
OUT_JSON = E_ROOT + "/03_results/stats/p3_6_gap.json"
OUT_MD = E_ROOT + "/03_results/tables/T_p3_6_gap.md"


def main():
    swap = json.load(open(IN_SWAP, "r", encoding="utf-8"))
    rows = []
    for key, c in swap["per_condition"].items():
        d = {r["unit_set"]: r for r in c["by_unit_set"]}
        gap_f = d["ID"]["mean_F"] - d["OOD"]["mean_F"]
        gap_v = d["ID"]["mean_V"] - d["OOD"]["mean_V"]
        comp = (1.0 - gap_v / gap_f) if gap_f != 0 else float("nan")
        rows.append(dict(condition=key, area=c["area"], placement=c["placement"],
                         E_F_ID=d["ID"]["mean_F"], E_F_OOD=d["OOD"]["mean_F"],
                         E_V_ID=d["ID"]["mean_V"], E_V_OOD=d["OOD"]["mean_V"],
                         gap_F=gap_f, gap_V=gap_v, compression=comp,
                         sign_flip=bool((gap_f > 0) != (gap_v > 0))))
    rows.sort(key=lambda r: (r["placement"] != "target", r["area"]))

    target = [r for r in rows if r["placement"] == "target"]
    rand = [r for r in rows if r["placement"] == "random"]
    out = dict(script="02_code/analysis/p3_6_gap_derivation.py",
               source=os.path.basename(IN_SWAP),
               note="ID-OOD mean gap under F vs V caliber; derived from the by_unit_set block",
               readouts=dict(
                   n_sign_flips=sum(r["sign_flip"] for r in rows),
                   max_compression=max(r["compression"] for r in rows),
                   min_compression=min(r["compression"] for r in rows),
                   mean_compression_target=sum(r["compression"] for r in target) / len(target),
                   mean_compression_random=sum(r["compression"] for r in rand) / len(rand)),
               rows=rows)

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)

    L = ["# T_p3_6_gap — the ID-OOD mean gap under the two calibers", "",
         "*Derived from `caliber_swap.json` by `p3_6_gap_derivation.py`.*", "",
         "| condition | placement | E_F(ID) | E_F(OOD) | gap_F | E_V(ID) | E_V(OOD) | gap_V |"
         " compression | sign flip |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|:--:|"]
    for r in rows:
        L.append("| %s | %s | %.4f | %.4f | %+.4f | %.4f | %.4f | %+.4f | %+.1f%% | %s |"
                 % (r["condition"], r["placement"], r["E_F_ID"], r["E_F_OOD"], r["gap_F"],
                    r["E_V_ID"], r["E_V_OOD"], r["gap_V"], r["compression"] * 100,
                    "**yes**" if r["sign_flip"] else "no"))
    L.append("")
    L.append("- sign flips: **%d / %d** conditions." % (out["readouts"]["n_sign_flips"], len(rows)))
    L.append("- mean compression: target-placed **%+.1f%%**, randomly-placed **%+.1f%%**."
             % (out["readouts"]["mean_compression_target"] * 100,
                out["readouts"]["mean_compression_random"] * 100))
    L.append("- range of compression: **%+.1f%% to %+.1f%%**."
             % (out["readouts"]["min_compression"] * 100,
                out["readouts"]["max_compression"] * 100))
    L.append("")
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))

    print("\n".join(L))
    print("wrote %s (%d B)" % (OUT_JSON, os.path.getsize(OUT_JSON)))
    print("wrote %s (%d B)" % (OUT_MD, os.path.getsize(OUT_MD)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
