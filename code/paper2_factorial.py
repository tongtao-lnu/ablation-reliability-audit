# -*- coding: utf-8 -*-
"""
论文二 析因分析 —— 2^3 模块消融 x 域偏移

设计: 7 个配置 = EGM/DPA/MSFA 三个二值因子的部分析因(缺全开格)
     全部在同一份分布内数据(Kvasir+CVC)上、同一训练协议下训练。

关键方法学: 用**逐样本配对对比**估计模块主效应，而不是比较边际均值。
     对 EGM 而言有三对"仅 EGM 不同"的配置:
        (baseline, egm_only)  (dpa_only, egm_dpa)  (msfa_only, egm_msfa)
     DPA / MSFA 同理各三对。每对给 196 例 ETIS 的逐样本 ΔDice -> Wilcoxon 配对检验。

核心论证:
     模块在**分布内**配对效应很小且不显著(paper2_indist.py)，
     但在**未见域**配对效应很大且高度显著 => 分布内指标不能预测模块的域稳健性。

输入:
    results_ablation_etis/{config}/predictions.npy     (ETIS, 196)
    experiments/ablation/{config}/sample_metrics.csv   (分布内, 242)
输出:
    verify/out/paper2_factorial.json / paper2_factorial_table.md
"""
import os
import csv
import json
import numpy as np
from scipy import stats

ROOT = "D:/medical_segmentation"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

CONFIGS = ["baseline", "egm_only", "dpa_only", "msfa_only",
           "egm_dpa", "egm_msfa", "dpa_msfa"]
MOD = {
    "baseline": (0, 0, 0), "egm_only": (1, 0, 0), "dpa_only": (0, 1, 0),
    "msfa_only": (0, 0, 1), "egm_dpa": (1, 1, 0), "egm_msfa": (1, 0, 1),
    "dpa_msfa": (0, 1, 1),
}
IDX = {"EGM": 0, "DPA": 1, "MSFA": 2}

# 每对: 仅该模块不同, 其余因子固定
PAIRS = {
    "EGM": [("baseline", "egm_only"), ("dpa_only", "egm_dpa"), ("msfa_only", "egm_msfa")],
    "DPA": [("baseline", "dpa_only"), ("egm_only", "egm_dpa"), ("msfa_only", "dpa_msfa")],
    "MSFA": [("baseline", "msfa_only"), ("egm_only", "egm_msfa"), ("dpa_only", "dpa_msfa")],
}


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


def boot_ci(x, stat=np.mean, n_boot=5000, seed=7):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    v = stat(x[idx], axis=1)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


# ---------------------------------------------------------------- 载入
ood, ood_meta = {}, {}
for c in CONFIGS:
    p = os.path.join(ROOT, "results_ablation_etis", c, "predictions.npy")
    if not os.path.exists(p):
        print(f"[缺] {p}")
        continue
    arr = np.load(p, allow_pickle=True)
    ood[c] = arr
    ood_meta[c] = {
        "dice": np.array([x["dice"] for x in arr]) * 100,
        "name": [x["name"] for x in arr],
    }

names0 = ood_meta[CONFIGS[0]]["name"]
for c in ood:
    assert ood_meta[c]["name"] == names0, f"{c} 样本顺序不对齐"

ind, ind_meta = {}, {}
for c in CONFIGS:
    p = os.path.join(ROOT, "experiments", "ablation", c, "sample_metrics.csv")
    if not os.path.exists(p):
        continue
    d = {}
    with open(p, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            d[row["filename"]] = float(row["Dice"]) * 100
    ind_meta[c] = d

inames = sorted(set.intersection(*[set(d.keys()) for d in ind_meta.values()]))

# ---------------------------------------------------------------- 描述
desc = {}
for c in CONFIGS:
    if c not in ood:
        continue
    x = ood_meta[c]["dice"]
    o = np.array([ind_meta[c][k] for k in inames]) if c in ind_meta else None
    desc[c] = {
        "modules": dict(zip(["EGM", "DPA", "MSFA"], MOD[c])),
        "indist_dice": float(o.mean()) if o is not None else None,
        "ood_dice": float(x.mean()),
        "ood_dice_std": float(x.std()),
        "ood_dice_ci95": boot_ci(x),
        "retention_pct": float(x.mean() / o.mean() * 100) if o is not None else None,
        "ood_failure_rate_lt50": float((x < 50).mean()),
        "ood_collapse_rate_lt10": float((x < 10).mean()),
    }

ind_means = {c: desc[c]["indist_dice"] for c in desc if desc[c]["indist_dice"] is not None}
ood_means = {c: desc[c]["ood_dice"] for c in desc}

# 分布内 spread vs 域外 spread
ind_spread = max(ind_means.values()) - min(ind_means.values())
ood_spread = max(ood_means.values()) - min(ood_means.values())

# 分布内 Dice 能否预测域外 Dice?  (跨 7 配置)
cs = [c for c in CONFIGS if c in ind_means and c in ood_means]
sp = stats.spearmanr([ind_means[c] for c in cs], [ood_means[c] for c in cs])
pe = stats.pearsonr([ind_means[c] for c in cs], [ood_means[c] for c in cs])

# ---------------------------------------------------------------- 配对主效应
effects = {}
for mod, pairs in PAIRS.items():
    recs = []
    for off_c, on_c in pairs:
        # 域外(ETIS)
        o = ood_meta[on_c]["dice"] - ood_meta[off_c]["dice"]
        try:
            p_ood = float(stats.wilcoxon(ood_meta[on_c]["dice"], ood_meta[off_c]["dice"]).pvalue)
        except Exception:
            p_ood = float("nan")
        # 分布内
        io = np.array([ind_meta[on_c][k] - ind_meta[off_c][k] for k in inames])
        try:
            p_ind = float(stats.wilcoxon([ind_meta[on_c][k] for k in inames],
                                         [ind_meta[off_c][k] for k in inames]).pvalue)
        except Exception:
            p_ind = float("nan")
        recs.append({
            "pair": f"{off_c} -> {on_c}",
            "ood_mean_delta_pts": float(o.mean()),
            "ood_median_delta_pts": float(np.median(o)),
            "ood_delta_ci95": boot_ci(o),
            "ood_p_wilcoxon": p_ood,
            "ood_win_rate": float((o > 0).mean()),
            "ind_mean_delta_pts": float(io.mean()),
            "ind_median_delta_pts": float(np.median(io)),
            "ind_p_wilcoxon": p_ind,
        })
    # 合并三对的所有逐样本差(池化)
    pool_ood = np.concatenate([ood_meta[on]["dice"] - ood_meta[off]["dice"] for off, on in pairs])
    pool_ind = np.concatenate([[ind_meta[on][k] - ind_meta[off][k] for k in inames]
                               for off, on in pairs])
    effects[mod] = {
        "pairs": recs,
        "pooled_ood_mean_delta_pts": float(pool_ood.mean()),
        "pooled_ood_delta_ci95": boot_ci(pool_ood),
        "pooled_ood_win_rate": float((pool_ood > 0).mean()),
        "pooled_ind_mean_delta_pts": float(pool_ind.mean()),
        "pooled_ind_win_rate": float((pool_ind > 0).mean()),
        "amplification_ratio": float(pool_ood.mean() / pool_ind.mean()) if pool_ind.mean() != 0 else None,
        "n_ood_samples": int(len(pool_ood)),
        "n_ind_samples": int(len(pool_ind)),
    }
    # 三对各自的 p 做 Holm 校正
    adj = holm([r["ood_p_wilcoxon"] if not np.isnan(r["ood_p_wilcoxon"]) else 1.0 for r in recs])
    for r, a in zip(recs, adj):
        r["ood_p_holm"] = float(a)

# 边际主效应(简单平均)
marg = {}
for mod, p2 in [("EGM", 0), ("DPA", 1), ("MSFA", 2)]:
    on = [ood_means[c] for c in cs if MOD[c][p2]]
    off = [ood_means[c] for c in cs if not MOD[c][p2]]
    oni = [ind_means[c] for c in cs if MOD[c][p2]]
    offi = [ind_means[c] for c in cs if not MOD[c][p2]]
    marg[mod] = {
        "ood_mean_on": float(np.mean(on)), "ood_mean_off": float(np.mean(off)),
        "ood_effect_pts": float(np.mean(on) - np.mean(off)),
        "ind_mean_on": float(np.mean(oni)), "ind_mean_off": float(np.mean(offi)),
        "ind_effect_pts": float(np.mean(oni) - np.mean(offi)),
    }

result = {
    "design": "2^3 部分析因 (EGM/DPA/MSFA), 7 格, ETIS n=196, 分布内 n=242",
    "per_config": desc,
    "spread": {"indist_pts": ind_spread, "ood_pts": ood_spread,
               "amplification": ood_spread / ind_spread if ind_spread else None},
    "indist_predicts_ood": {
        "spearman_rho": float(np.asarray(sp).ravel()[0]), "spearman_p": float(np.asarray(sp).ravel()[1]),
        "pearson_r": float(pe[0]), "pearson_p": float(pe[1]),
    },
    "matched_pair_effects": effects,
    "marginal_main_effects": marg,
}
with open(os.path.join(OUT, "paper2_factorial.json"), "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------- 表格
L = ["# 模块消融 x 域偏移 析因分析", "",
     "## 表 1  7 配置: 分布内 vs 未见域 ETIS", "",
     "| 配置 | EGM | DPA | MSFA | 分布内 Dice(%) | ETIS Dice(%) | 保持率 | ETIS 失败率 | ETIS 崩溃率 |",
     "|---|---|---|---|---|---|---|---|---|"]
for c in CONFIGS:
    if c not in desc:
        continue
    s = desc[c]
    m = s["modules"]
    L.append("| {} | {} | {} | {} | {:.2f} | {:.2f} | {:.1f}% | {:.1f}% | {:.1f}% |".format(
        c, m["EGM"], m["DPA"], m["MSFA"],
        s["indist_dice"] if s["indist_dice"] is not None else float("nan"),
        s["ood_dice"], s["retention_pct"],
        s["ood_failure_rate_lt50"] * 100, s["ood_collapse_rate_lt10"] * 100))
L += ["",
      f"分布内 7 配置 Dice 极差: **{ind_spread:.2f}** 点 vs 未见域极差: **{ood_spread:.2f}** 点 "
      f"(放大 {ood_spread/ind_spread:.1f}x)" if ind_spread else "",
      f"分布内 Dice vs ETIS Dice 的 Spearman rho = {sp[0]:.3f} (p = {sp[1]:.3g})", ""]

L += ["## 表 2  逐样本配对主效应(仅单一模块不同)", "",
      "| 模块 | 配对 | 分布内 ΔDice(点) | 分布内 p | ETIS ΔDice(点) | ETIS 95%CI | ETIS p | ETIS p(Holm) | ETIS 胜率 |",
      "|---|---|---|---|---|---|---|---|---|"]
for mod, e in effects.items():
    for r in e["pairs"]:
        L.append("| {} | {} | {:+.2f} | {:.4g} | {:+.2f} | [{:+.1f}, {:+.1f}] | {:.3g} | {:.3g} | {:.1f}% |".format(
            mod, r["pair"], r["ind_mean_delta_pts"], r["ind_p_wilcoxon"],
            r["ood_mean_delta_pts"], r["ood_delta_ci95"][0], r["ood_delta_ci95"][1],
            r["ood_p_wilcoxon"], r["ood_p_holm"], r["ood_win_rate"] * 100))
L += ["", "## 表 3  池化配对主效应", "",
      "| 模块 | 分布内 ΔDice(点) | 分布内胜率 | ETIS ΔDice(点) | ETIS 95%CI | ETIS 胜率 | 放大倍数 |",
      "|---|---|---|---|---|---|---|"]
for mod, e in effects.items():
    L.append("| {} | {:+.2f} | {:.1f}% | {:+.2f} | [{:+.1f}, {:+.1f}] | {:.1f}% | {:.2f}x |".format(
        mod, e["pooled_ind_mean_delta_pts"], e["pooled_ind_win_rate"] * 100,
        e["pooled_ood_mean_delta_pts"], e["pooled_ood_delta_ci95"][0],
        e["pooled_ood_delta_ci95"][1], e["pooled_ood_win_rate"] * 100,
        e["amplification_ratio"] or float("nan")))

with open(os.path.join(OUT, "paper2_factorial_table.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(x for x in L if x is not None) + "\n")

# ---------------------------------------------------------------- 控制台
print("\n" + "=" * 92)
print("  模块消融 x 域偏移 | 分布内(242) vs ETIS 未见域(196)")
print("=" * 92)
print(f'  {"config":<12}{"EGM":>4}{"DPA":>4}{"MSFA":>5}{"分布内Dice":>12}{"ETIS Dice":>11}'
      f'{"保持率":>9}{"ETIS崩溃<10":>12}')
print("  " + "-" * 86)
for c in CONFIGS:
    if c not in desc:
        continue
    s = desc[c]
    m = s["modules"]
    print(f'  {c:<12}{m["EGM"]:>4}{m["DPA"]:>4}{m["MSFA"]:>5}'
          f'{s["indist_dice"]:>12.2f}{s["ood_dice"]:>11.2f}{s["retention_pct"]:>8.1f}%'
          f'{s["ood_collapse_rate_lt10"]*100:>11.1f}%')
print("=" * 92)
print(f"  分布内极差 {ind_spread:.2f} 点  ->  未见域极差 {ood_spread:.2f} 点   "
      f"(放大 {ood_spread/ind_spread:.1f}x)")
print(f"  分布内 Dice vs ETIS Dice:  Spearman rho={sp[0]:.3f} (p={sp[1]:.3g}) | "
      f"Pearson r={pe[0]:.3f} (p={pe[1]:.3g})")

print("\n  逐样本配对主效应(仅单一模块不同):")
print(f'  {"模块":<6}{"配对":<30}{"分布内Δ":>10}{"p":>10}{"ETIS Δ":>10}{"ETIS p":>12}{"胜率":>8}')
print("  " + "-" * 86)
for mod, e in effects.items():
    for r in e["pairs"]:
        print(f'  {mod:<6}{r["pair"]:<30}{r["ind_mean_delta_pts"]:>+10.2f}'
              f'{r["ind_p_wilcoxon"]:>10.3g}{r["ood_mean_delta_pts"]:>+10.2f}'
              f'{r["ood_p_wilcoxon"]:>12.3g}{r["ood_win_rate"]*100:>7.1f}%')
print("\n  池化:")
for mod, e in effects.items():
    print(f'    {mod:<5} 分布内 {e["pooled_ind_mean_delta_pts"]:+6.2f} 点 (胜率 '
          f'{e["pooled_ind_win_rate"]*100:.1f}%)  ->  ETIS {e["pooled_ood_mean_delta_pts"]:+6.2f} 点 '
          f'(胜率 {e["pooled_ood_win_rate"]*100:.1f}%)  放大 '
          f'{(e["amplification_ratio"] or float("nan")):.2f}x')

print("\n输出:")
print(" ", os.path.join(OUT, "paper2_factorial.json"))
print(" ", os.path.join(OUT, "paper2_factorial_table.md"))
