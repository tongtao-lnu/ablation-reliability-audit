# -*- coding: utf-8 -*-
"""
论文二 分布内对照侧 —— 模块消融的"分布内打平"证据

论文二的论证骨架:
    分布内评估**无法区分** EGM / DPA / MSFA 三个模块(本脚本给出统计证据)
    -> 但在未见域 ETIS 上它们分道扬镳(ablation_zeroshot_etis.py + paper2_stats.py)
    => "分布内指标不能预测模块的域稳健性"

本脚本还要回答审稿人一定会问的问题:
    "EGA-UNet 的优势会不会只是因为它更强?"
    -> 在**分布内 Dice 统计不可区分**的条件下比较模块,即可排除这个混淆。

输入: experiments/ablation/{7 configs}/sample_metrics.csv   (242 例, 同一测试集, 同文件名可配对)
      experiments/baseline/{13 models}/sample_metrics.csv   (参照)
输出: verify/out/paper2_indist_stats.json, verify/out/paper2_indist_table.md
"""
import os
import csv
import json
import numpy as np
from scipy import stats

ROOT = "D:/medical_segmentation"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

ABLATIONS = ["baseline", "egm_only", "dpa_only", "msfa_only",
             "egm_dpa", "egm_msfa", "dpa_msfa"]
MODULE_OF = {
    "baseline": (0, 0, 0), "egm_only": (1, 0, 0), "dpa_only": (0, 1, 0),
    "msfa_only": (0, 0, 1), "egm_dpa": (1, 1, 0), "egm_msfa": (1, 0, 1),
    "dpa_msfa": (0, 1, 1),
}


def load_csv(path):
    """返回 {filename: dice} 与有序数组。"""
    d = {}
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            d[row["filename"]] = float(row["Dice"])
    return d


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    dd = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / dd), min(1.0, (c + h) / dd))


def boot_ci(x, stat=np.mean, n_boot=5000, seed=42):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    v = stat(x[idx], axis=1)
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))


def holm(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, float)
    prev = 0.0
    for rank, i in enumerate(order):
        p = min(max(pvals[i] * (m - rank), prev), 1.0)
        adj[i] = p
        prev = p
    return adj


data = {}
for c in ABLATIONS:
    p = os.path.join(ROOT, "experiments", "ablation", c, "sample_metrics.csv")
    if os.path.exists(p):
        data[c] = load_csv(p)

names = sorted(set.intersection(*[set(d.keys()) for d in data.values()]))
print(f"对齐样本数: {len(names)}  (配置 {len(data)} 个)")

# ---------------- 分布内描述统计 ----------------
indist = {}
for c, d in data.items():
    x = np.array([d[k] for k in names]) * 100
    e, p2, m = MODULE_OF[c]
    indist[c] = {
        "modules": {"EGM": e, "DPA": p2, "MSFA": m},
        "n": len(x),
        "dice_mean": float(x.mean()), "dice_std": float(x.std()),
        "dice_median": float(np.median(x)),
        "dice_ci95": boot_ci(x),
        "failure_rate_dice_lt_50": float((x < 50).mean()),
        "failure_n": int((x < 50).sum()),
        "hd95_median": None,
    }

# ---------------- 关键检验: 三个单模块是否分布内不可区分 ----------------
singles = ["egm_only", "dpa_only", "msfa_only"]
pair_tests = {}
if all(s in data for s in singles):
    pvals = []
    pairs = [(singles[i], singles[j]) for i in range(len(singles)) for j in range(i + 1, len(singles))]
    for a, b in pairs:
        xa = np.array([data[a][k] for k in names])
        xb = np.array([data[b][k] for k in names])
        try:
            p = float(stats.wilcoxon(xa, xb).pvalue)
        except Exception:
            p = float("nan")
        pvals.append(p)
        pair_tests[f"{a}_vs_{b}"] = {
            "median_delta_pts": float(np.median(xa - xb) * 100),
            "mean_delta_pts": float((xa.mean() - xb.mean()) * 100),
            "p_wilcoxon": p,
            "p_holm": None,
        }
    adj = holm([p if not np.isnan(p) else 1.0 for p in pvals])
    for (k, v), pa in zip(pair_tests.items(), adj):
        v["p_holm"] = float(pa)

    # Friedman 联合检验(7 配置)
    mat = [np.array([data[c][k] for k in names]) for c in ABLATIONS if c in data]
    fr = stats.friedmanchisquare(*mat)
    friedman = {
        "chi2": float(fr.statistic), "p": float(fr.pvalue),
        "k_configs": len(mat), "n": len(names),
        "kendalls_w": float(fr.statistic) / (len(names) * (len(mat) - 1)),
    }

    # 三单模块 Friedman(最关键的比较)
    mat3 = [np.array([data[s][k] for k in names]) for s in singles]
    fr3 = stats.friedmanchisquare(*mat3)
    friedman_singles = {
        "chi2": float(fr3.statistic), "p": float(fr3.pvalue),
        "k_configs": 3, "n": len(names),
        "kendalls_w": float(fr3.statistic) / (len(names) * 2),
        "dice_means": {s: float(np.mean([data[s][k] for k in names]) * 100) for s in singles},
        "dice_spread_pts": float(
            max(np.mean([data[s][k] for k in names]) for s in singles)
            - min(np.mean([data[s][k] for k in names]) for s in singles)) * 100,
    }
else:
    friedman = friedman_singles = {}

# ---------------- 模块主效应(分布内) ----------------
main_effects = {}
for mod, flag_idx in [("EGM", 0), ("DPA", 1), ("MSFA", 2)]:
    on = [np.array([data[c][k] for k in names]) * 100 for c in data if MODULE_OF[c][flag_idx]]
    off = [np.array([data[c][k] for k in names]) * 100 for c in data if not MODULE_OF[c][flag_idx]]
    if on and off:
        main_effects[mod] = {
            "n_on": len(on), "n_off": len(off),
            "mean_on": float(np.mean([o.mean() for o in on])),
            "mean_off": float(np.mean([o.mean() for o in off])),
            "effect_pts": float(np.mean([o.mean() for o in on]) - np.mean([o.mean() for o in off])),
        }

result = {
    "n_aligned_samples": len(names),
    "in_distribution": indist,
    "pairwise_single_modules": pair_tests,
    "friedman_all_configs": friedman,
    "friedman_single_modules": friedman_singles,
    "main_effects_in_dist": main_effects,
    "note": "全部来自 experiments/ablation/*/sample_metrics.csv，242 例分布内测试集，逐样本可配对",
}

with open(os.path.join(OUT, "paper2_indist_stats.json"), "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

# ---------------- 可读表格 ----------------
L = ["# 分布内模块消融(对照侧)", "",
     f"对齐样本数: {len(names)}", "",
     "| 配置 | EGM | DPA | MSFA | Dice(%) | 95%CI | 失败率(Dice<50) |",
     "|---|---|---|---|---|---|---|"]
for c in ABLATIONS:
    if c not in indist:
        continue
    s = indist[c]
    m = s["modules"]
    L.append("| {} | {} | {} | {} | {:.2f} | [{:.2f}, {:.2f}] | {:.1f}% ({}/{}) |".format(
        c, m["EGM"], m["DPA"], m["MSFA"], s["dice_mean"],
        s["dice_ci95"][0], s["dice_ci95"][1],
        s["failure_rate_dice_lt_50"] * 100, s["failure_n"], s["n"]))

if friedman_singles:
    fs = friedman_singles
    L += ["", "## 关键检验: 三个单模块在分布内是否可区分", "",
          "| 单模块 | 分布内 Dice(%) |", "|---|---|"]
    for s, v in fs["dice_means"].items():
        L.append(f"| {s} | {v:.2f} |")
    L += ["",
          f"三配置 Friedman: chi2={fs['chi2']:.3f}, p={fs['p']:.4g}, Kendall's W={fs['kendalls_w']:.4f}",
          f"平均 Dice 极差: {fs['dice_spread_pts']:.2f} 个百分点", ""]
    L += ["| 单模块两两比较 | 中位差(点) | 均值差(点) | Wilcoxon p | Holm p |", "|---|---|---|---|---|"]
    for k, v in pair_tests.items():
        L.append("| {} | {:+.2f} | {:+.2f} | {:.4g} | {:.4g} |".format(
            k, v["median_delta_pts"], v["mean_delta_pts"],
            v["p_wilcoxon"], v["p_holm"] if v["p_holm"] is not None else float("nan")))

with open(os.path.join(OUT, "paper2_indist_table.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")

# ---------------- 控制台 ----------------
print("\n" + "=" * 78)
print("  分布内模块消融 (242 例, 同一测试集)")
print("=" * 78)
print(f'  {"config":<13}{"EGM":>4}{"DPA":>4}{"MSFA":>5}{"Dice":>9}{"95%CI":>18}{"fail<50":>10}')
print("  " + "-" * 72)
for c in ABLATIONS:
    if c not in indist:
        continue
    s = indist[c]
    m = s["modules"]
    print(f'  {c:<13}{m["EGM"]:>4}{m["DPA"]:>4}{m["MSFA"]:>5}{s["dice_mean"]:>9.2f}'
          f'   [{s["dice_ci95"][0]:>6.2f},{s["dice_ci95"][1]:>6.2f}]'
          f'{s["failure_rate_dice_lt_50"]*100:>9.1f}%')
print("=" * 78)

if friedman_singles:
    fs = friedman_singles
    print("\n  【关键】三个单模块在分布内是否可区分?")
    for s, v in fs["dice_means"].items():
        print(f"    {s:<12} Dice = {v:.2f}%")
    print(f"    Friedman chi2={fs['chi2']:.3f}  p={fs['p']:.4g}  Kendall's W={fs['kendalls_w']:.4f}")
    print(f"    平均 Dice 极差 = {fs['dice_spread_pts']:.2f} 个百分点  <-- 分布内几乎打平")
    print("\n    两两配对 Wilcoxon (Holm 校正):")
    for k, v in pair_tests.items():
        sig = "显著" if (v["p_holm"] is not None and v["p_holm"] < 0.05) else "不显著"
        print(f"      {k:<24} 中位差 {v['median_delta_pts']:+.2f} 点  "
              f"p={v['p_wilcoxon']:.4g}  Holm={v['p_holm']:.4g}  -> {sig}")

print("\n  【模块主效应(分布内)】")
for mod, v in main_effects.items():
    print(f"    {mod:<5} 效应 = {v['effect_pts']:+6.2f} 点  (n_on={v['n_on']}, n_off={v['n_off']})")

print("\n输出:")
print(" ", os.path.join(OUT, "paper2_indist_stats.json"))
print(" ", os.path.join(OUT, "paper2_indist_table.md"))
