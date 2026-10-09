# -*- coding: utf-8 -*-
"""
论文二 统计主干 —— 把全部"跨域失败解剖"结论一次算清，并给出可发表的统计量

解决的问题(v2 方案的严谨性缺口):
    之前所有数字都是点估计(如"崩溃型占 90.6%±2.5")，没有
      - 比例的置信区间(Wilson)
      - 配对的显著性检验(Wilcoxon signed-rank)
      - 多架构的联合检验(Friedman + Kendall's W)
      - 救回净例数的不确定性(bootstrap CI)
      - 多重比较校正(Holm)
      - 效应量(共同语言效应量 / 中位差 bootstrap CI)

输入(全部已存在，零训练成本):
    results_cross/{cvc_to_kvasir,kvasir_to_cvc,kvasir_to_etis,cvc_to_etis}/{arch}/predictions.npy

输出:
    verify/out/paper2_per_sample.csv   逐样本主表(可直接做图/复查)
    verify/out/paper2_stats.json       全部统计量
    verify/out/paper2_tables.md        可直接贴进论文的表格
"""
import os
import json
import numpy as np
from scipy import ndimage
from scipy import stats

ROOT = "D:/medical_segmentation"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
os.chdir(ROOT)

ARCHS = ["UNet", "AttentionUNet", "TransUNet", "EGAUNet"]
DIRS = [
    ("cvc_to_kvasir", "CVC->Kvasir"),
    ("kvasir_to_cvc", "Kvasir->CVC"),
    ("kvasir_to_etis", "Kvasir->ETIS"),
    ("cvc_to_etis", "CVC->ETIS"),
]

FAIL_THRESHOLDS = [0.3, 0.5, 0.7]   # 失败定义
COLLAPSE_INNER_IOU = 0.3            # 崩溃型判据: 质心未命中 或 内IoU < 0.3


# ---------------------------------------------------------------- 统计工具
def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def boot_ci(x, stat=np.mean, n_boot=5000, seed=42):
    """百分比自助法置信区间。"""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    vals = stat(x[idx], axis=1)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def holm(pvals):
    """Holm-Bonferroni 校正，返回校正后 p 值(保持原顺序)。"""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    prev = 0.0
    for rank, i in enumerate(order):
        p = pvals[i] * (m - rank)
        p = max(p, prev)
        adj[i] = min(p, 1.0)
        prev = adj[i]
    return adj


# ---------------------------------------------------------------- 逐样本指标
def analyze_sample(mask, pred, gt_cache=None):
    """双轴分解: 检测(崩溃) vs 边界。gt_cache 复用同一张 GT 的距离场。"""
    g = mask > 0.5
    p = np.asarray(pred).astype(bool)
    gs, ps = int(g.sum()), int(p.sum())
    out = {"gt_area": gs, "pred_area": ps}

    if ps > 0:
        cy, cx = ndimage.center_of_mass(p)
        cy = min(max(int(round(cy)), 0), g.shape[0] - 1)
        cx = min(max(int(round(cx)), 0), g.shape[1] - 1)
        out["centroid_hit"] = int(g[cy, cx])
    else:
        out["centroid_hit"] = 0

    out["recall_gt"] = float(np.logical_and(g, p).sum() / gs) if gs else 1.0
    out["area_ratio"] = float(ps / gs) if gs else 1.0

    if gt_cache is None:
        if gs >= 5:
            dist = ndimage.distance_transform_edt(g) + ndimage.distance_transform_edt(~g)
            b5, b10 = dist <= 5, dist <= 10
        else:
            b5, b10 = g.copy(), g.copy()
    else:
        g, b5, b10 = gt_cache

    gb, pb = g & b5, p & b5
    den = int(gb.sum()) + int(pb.sum())
    out["band_dice"] = float(2 * np.logical_and(gb, pb).sum() / den) if den else 1.0

    inner, pin = g & ~b10, p & ~b10
    u = int(np.logical_or(inner, pin).sum())
    out["inner_iou"] = float(np.logical_and(inner, pin).sum() / u) if u else 1.0

    lab, ncomp = ndimage.label(p)
    out["ncomp"] = int(ncomp)
    if ncomp > 0 and ps:
        sizes = ndimage.sum(p, lab, range(1, ncomp + 1))
        out["largest_frac"] = float(sizes.max() / ps)
    else:
        out["largest_frac"] = 1.0
    return out


# ---------------------------------------------------------------- 主流程
all_rows = []          # 逐样本长表
meta = {}              # (dir, arch) -> 汇总

for d, lab in DIRS:
    base = os.path.join("results_cross", d)
    if not os.path.isdir(base):
        print(f"[跳过] 缺少 {base}")
        continue
    avail = [a for a in ARCHS if os.path.exists(os.path.join(base, a, "predictions.npy"))]
    if not avail:
        print(f"[跳过] {d} 无预测文件")
        continue

    loaded = {}
    for a in avail:
        arr = np.load(os.path.join(base, a, "predictions.npy"), allow_pickle=True)
        loaded[a] = arr
    n = len(loaded[avail[0]])

    # 校验样本顺序对齐(配对检验的前提)
    names0 = [x["name"] for x in loaded[avail[0]]]
    for a in avail[1:]:
        assert [x["name"] for x in loaded[a]] == names0, f"{d}/{a} 样本顺序不对齐!"

    # GT 距离场只算一次，4 个架构复用
    gt_cache_list = []
    for i in range(n):
        g = loaded[avail[0]][i]["mask"] > 0.5
        gs = int(g.sum())
        if gs >= 5:
            dist = ndimage.distance_transform_edt(g) + ndimage.distance_transform_edt(~g)
            b5, b10 = dist <= 5, dist <= 10
        else:
            b5, b10 = g.copy(), g.copy()
        gt_cache_list.append((g, b5, b10))

    for a in avail:
        recs = []
        for i in range(n):
            x = loaded[a][i]
            r = analyze_sample(x["mask"], x["pred"], gt_cache=gt_cache_list[i])
            r["name"] = x["name"]
            r["dir"] = d
            r["arch"] = a
            r["dice"] = float(x["dice"])
            r["iou"] = float(x["iou"])
            r["hd95"] = float(x["hd95"])
            recs.append(r)
        all_rows.extend(recs)

        dl = np.array([r["dice"] for r in recs])
        meta[(d, a)] = {"dir": d, "arch": a, "n": n, "dice": dl}
        print(f"  完成 {d:<14} {a:<14} n={n}  Dice={dl.mean()*100:.2f}", flush=True)
    del loaded, gt_cache_list


def col(rows, k):
    return np.array([r[k] for r in rows], dtype=float)


# ---------------------------------------------------------------- 汇总统计
stats_out = {"dataset": {}, "tests": {}, "rescue": {}, "threshold_sweep": {}}

for d, lab in DIRS:
    stats_out["dataset"][d] = {}
    for a in ARCHS:
        rows = [r for r in all_rows if r["dir"] == d and r["arch"] == a]
        if not rows:
            continue
        dl = col(rows, "dice")
        fail = dl < 0.5
        n = len(dl)
        hit = col(rows, "centroid_hit")
        inner = col(rows, "inner_iou")
        collapse = fail & ((hit == 0) | (inner < COLLAPSE_INNER_IOU))
        k, nf = int(collapse.sum()), int(fail.sum())
        lo, hi = wilson(k, nf)
        dlo, dhi = boot_ci(dl * 100)
        stats_out["dataset"][d][a] = {
            "n": n,
            "dice_mean": float(dl.mean() * 100), "dice_std": float(dl.std() * 100),
            "dice_median": float(np.median(dl) * 100),
            "dice_ci95": [dlo, dhi],
            "failure_n_dice_lt_50": nf,
            "failure_rate_dice_lt_50": float(fail.mean()),
            "collapse_n": k,
            "collapse_fraction_of_failures": float(k / nf) if nf else float("nan"),
            "collapse_fraction_ci95": [lo, hi],
            "centroid_hit_rate": float(hit.mean()),
            "gt_coverage_mean": float(col(rows, "recall_gt").mean()),
            "area_ratio_median": float(np.median(col(rows, "area_ratio"))),
            "band_dice_mean": float(col(rows, "band_dice").mean()),
            "inner_iou_mean": float(inner.mean()),
            "ncomp_median": float(np.median(col(rows, "ncomp"))),
            "largest_frac_median": float(np.median(col(rows, "largest_frac"))),
            "hd95_mean": float(np.nanmean(col(rows, "hd95"))),
        }

# ---- 配对检验: EGA-UNet vs 每个 baseline
for d, lab in DIRS:
    if (d, "EGAUNet") not in meta:
        continue
    stats_out["tests"][d] = {"pairwise_vs_EGAUNet": {}, "friedman": {}}
    ega = meta[(d, "EGAUNet")]["dice"]
    others = [a for a in ARCHS if a != "EGAUNet" and (d, a) in meta]
    pvals, entries = [], []
    for a in others:
        o = meta[(d, a)]["dice"]
        diff = (ega - o) * 100
        try:
            w = stats.wilcoxon(ega, o)
            p = float(w.pvalue)
        except Exception:
            p = float("nan")
        pvals.append(p if not np.isnan(p) else 1.0)
        entries.append({
            "vs": a,
            "median_delta_dice_pts": float(np.median(diff)),
            "delta_ci95": boot_ci(diff, stat=np.median),
            "p_wilcoxon_raw": p,
            "common_language_p_EGA_better": float(np.mean(ega > o)),
        })
    adj = holm(pvals)
    for e, pa in zip(entries, adj):
        e["p_wilcoxon_holm"] = float(pa)
        stats_out["tests"][d]["pairwise_vs_EGAUNet"][e["vs"]] = e

    # Friedman 联合检验(4 架构 x n 样本)
    if len(others) + 1 >= 3:
        mat = [meta[(d, a)]["dice"] for a in [*others, "EGAUNet"]]
        try:
            fr = stats.friedmanchisquare(*mat)
            chi2 = float(fr.statistic)
            k_arch = len(mat)
            nn = len(mat[0])
            stats_out["tests"][d]["friedman"] = {
                "chi2": chi2, "p": float(fr.pvalue),
                "k_architectures": k_arch, "n_samples": nn,
                "kendalls_w": chi2 / (nn * (k_arch - 1)),
            }
        except Exception as ex:
            stats_out["tests"][d]["friedman"] = {"error": str(ex)}

# ---- 救回分析(单向优势) + bootstrap CI
for d, lab in DIRS:
    if (d, "UNet") not in meta or (d, "EGAUNet") not in meta:
        continue
    a_base, a_new = "UNet", "EGAUNet"
    db = meta[(d, a_base)]["dice"]
    dn = meta[(d, a_new)]["dice"]
    n = len(db)
    ab = (db < 0.2) & (dn > 0.7)   # 基线崩溃 -> 新模型成功
    ba = (dn < 0.2) & (db > 0.7)   # 新模型崩溃 -> 基线成功
    net = int(ab.sum()) - int(ba.sum())
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n, size=(5000, n))
    nets = ab[idx].sum(axis=1) - ba[idx].sum(axis=1)
    stats_out["rescue"][d] = {
        "comparison": f"{a_base} -> {a_new}",
        "n": n,
        "rescued": int(ab.sum()), "rescued_pct": float(ab.sum() / n * 100),
        "rescued_ci95_pct": list(np.array(wilson(int(ab.sum()), n)) * 100),
        "reversed": int(ba.sum()), "reversed_pct": float(ba.sum() / n * 100),
        "net": net,
        "net_ci95": [float(np.percentile(nets, 2.5)), float(np.percentile(nets, 97.5))],
        "net_ci95_excludes_zero": bool(np.percentile(nets, 2.5) > 0 or np.percentile(nets, 97.5) < 0),
    }
    # 被救回样本是否更小
    ga_all = col([r for r in all_rows if r["dir"] == d and r["arch"] == a_new], "gt_area")
    if ab.sum() > 0:
        stats_out["rescue"][d]["rescued_median_gt_area"] = float(np.median(ga_all[ab]))
        stats_out["rescue"][d]["all_median_gt_area"] = float(np.median(ga_all))
        stats_out["rescue"][d]["area_ratio_rescued_over_all"] = float(
            np.median(ga_all[ab]) / np.median(ga_all)) if np.median(ga_all) else float("nan")
        try:
            mw = stats.mannwhitneyu(ga_all[ab], ga_all[~ab], alternative="two-sided")
            stats_out["rescue"][d]["p_area_smaller_mannwhitney"] = float(mw.pvalue)
        except Exception:
            pass

# ---- 阈值扫描(崩溃占比 vs 失败严重程度)
for d, lab in DIRS:
    stats_out["threshold_sweep"][d] = {}
    for a in ARCHS:
        rows = [r for r in all_rows if r["dir"] == d and r["arch"] == a]
        if not rows:
            continue
        dl = col(rows, "dice")
        hit = col(rows, "centroid_hit")
        inner = col(rows, "inner_iou")
        sweep = {}
        for t in FAIL_THRESHOLDS:
            fail = dl < t
            if fail.sum() == 0:
                sweep[str(t)] = None
                continue
            collapse = fail & ((hit == 0) | (inner < COLLAPSE_INNER_IOU))
            lo, hi = wilson(int(collapse.sum()), int(fail.sum()))
            sweep[str(t)] = {
                "n_failures": int(fail.sum()),
                "collapse_fraction": float(collapse.sum() / fail.sum()),
                "ci95": [lo, hi],
            }
        stats_out["threshold_sweep"][d][a] = sweep

# ---------------------------------------------------------------- 落盘
with open(os.path.join(OUT, "paper2_stats.json"), "w", encoding="utf-8") as f:
    json.dump(stats_out, f, indent=2, ensure_ascii=False)

# 逐样本主表 CSV
keys = ["dir", "arch", "name", "dice", "iou", "hd95", "gt_area", "pred_area",
        "centroid_hit", "recall_gt", "area_ratio", "band_dice", "inner_iou",
        "ncomp", "largest_frac"]
with open(os.path.join(OUT, "paper2_per_sample.csv"), "w", encoding="utf-8") as f:
    f.write(",".join(keys) + "\n")
    for r in all_rows:
        f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")

# 可读表格
lines = []
for d, lab in DIRS:
    if d not in stats_out["dataset"]:
        continue
    lines.append(f"\n### {lab}\n")
    lines.append("| 架构 | Dice(%) | 失败率 | 崩溃占失败 | 崩溃比例95%CI | GT覆盖 | 面积比 | 带Dice | 内IoU | 碎片数 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for a in ARCHS:
        s = stats_out["dataset"][d].get(a)
        if not s:
            continue
        ci = s["collapse_fraction_ci95"]
        lines.append("| {} | {:.2f} | {:.1f}% | {:.1f}% | [{:.1f}, {:.1f}] | {:.3f} | {:.2f} | {:.3f} | {:.3f} | {:.0f} |".format(
            a, s["dice_mean"], s["failure_rate_dice_lt_50"] * 100,
            s["collapse_fraction_of_failures"] * 100, ci[0] * 100, ci[1] * 100,
            s["gt_coverage_mean"], s["area_ratio_median"], s["band_dice_mean"],
            s["inner_iou_mean"], s["ncomp_median"]))
with open(os.path.join(OUT, "paper2_tables.md"), "w", encoding="utf-8") as f:
    f.write("# 论文二 统计表(自动生成)\n" + "\n".join(lines) + "\n")

print("\n输出:")
print(" ", os.path.join(OUT, "paper2_stats.json"))
print(" ", os.path.join(OUT, "paper2_per_sample.csv"))
print(" ", os.path.join(OUT, "paper2_tables.md"))
