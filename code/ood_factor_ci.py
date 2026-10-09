# -*- coding: utf-8 -*-
"""
F1 面板 b 的不确定性：分布内 / 跨域 两套 (D, Q) 的 95% CI
===========================================================
G3 门禁：图中任何"因子移动"的论断都必须带区间，否则只是点估计。

定义（与 decomposition.csv / decomp_ablation_etis_full8.csv 完全一致）：
    D  = 检出比例 = #{IoU > 0} / n          （比例量 → 用二项 Wilson 区间）
    Q  = mean(Dice) / D                    （只在检出样本上有定义）
    mean Dice = D x Q                      （恒等式，残差 < 1e-9）

⚠️ 未检出判据必须用 **重叠像素数 |P ∩ G| == 0**（frozen criterion，B6），
   不能用 `Dice > 0` 或 `IoU > 0` 判：本管线的 Dice/IoU 带平滑常数 eps，
   零重叠样本会得到 ~1e-11 的正值，用浮点比较会把 72 例未检出全部误判成检出。
   实测：|P ∩ G| == 0 逐配置复现冻结表 n_undetected（baseline 72 / msfa_only 60 /
   egm_dpa 31 / egm_msfa 29），已内置 assert 守住这条。

两套区间来源不同，故意如此：
  跨域 n=196：对**同一批图像**自助重采样 B=20000（seed 冻结），
              逐次重算 D / Q / mean Dice —— 配对口径，与 hero_gap_ci.py 同源。
  分布内 n=242：检出失败恰好 0 例。**不声称 D 恰为 1.000**，
              用"三倍法则"给出下界 1 - 3/n ≈ 0.988，这是对 0 计数的诚实区间。
              （ID 的 Recall 最小值为 3.7e-13 > 0 —— 这正是 ε 伪影，
                与 decompose.py 记录的 "detection saturates at 1.000" 一致。）

输出 03_results/stats/ood_factor_ci.json
"""
import os
import json
import numpy as np
import pandas as pd

ABL = r"D:/medical_segmentation/results_ablation_etis"
IDE = r"D:/medical_segmentation/experiments/ablation"
OUT = r"E:/paper2_ablation_reliability/03_results/stats/ood_factor_ci.json"
FULL8 = r"E:/paper2_ablation_reliability/03_results/stats/decomp_ablation_etis_full8.csv"

CFGS = ["baseline", "egm_only", "dpa_only", "msfa_only", "egm_dpa", "egm_msfa", "dpa_msfa"]
B, SEED = 20000, 20260916
RNG = np.random.default_rng(SEED)


def ood_units(cfg):
    """按文件名对齐的逐例 (Dice, 重叠像素数)（跨域 196 例）。"""
    arr = np.load(os.path.join(ABL, cfg, "predictions.npy"), allow_pickle=True)
    out = {}
    for x in arr:
        ov = int((x["pred"].astype(bool) & x["mask"].astype(bool)).sum())
        out[str(x["name"]).replace(".npy", "")] = (float(x["dice"]), ov)
    return out


def id_units(cfg):
    """分布内 242 例逐例 (Dice, Recall)。det_source = Recall（与 decomposition.csv 一致）。"""
    df = pd.read_csv(os.path.join(IDE, cfg, "sample_metrics.csv"))
    return {str(r["filename"]): (float(r["Dice"]), float(r["Recall"]))
            for _, r in df.iterrows()}


def factors(dice, hit):
    """冻结判据：未检出 = 零重叠（hit == 0）。hit 为重叠像素数或 Recall。"""
    dice = np.asarray(dice, dtype=float)
    hit = np.asarray(hit, dtype=float)
    n = dice.size
    k = int((hit <= 0).sum())
    D = 1.0 - k / n
    md = float(dice.mean())
    Q = md / D if D > 0 else float("nan")
    return n, k, D, md, Q


def boot(dice, hit):
    """同一批样本的自助重采样（配对口径）：检出与否用 hit，水平用 Dice。"""
    dice = np.asarray(dice, dtype=float)
    hit = np.asarray(hit, dtype=float)
    n = dice.size
    idx = RNG.integers(0, n, size=(B, n))
    sd, sh = dice[idx], hit[idx]
    Db = (sh > 0).mean(axis=1)
    mdb = sd.mean(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        Qb = np.where(Db > 0, mdb / Db, np.nan)
    return Db, Qb, mdb


def ci95(x):
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def main():
    full8 = pd.read_csv(FULL8).set_index("config")
    out = {"seed": SEED, "n_boot": B,
           "undetected_criterion": "overlap |P n G| == 0  (frozen B6); ID uses Recall > 0",
           "cross_domain": {}, "in_domain": {}}

    names = None
    for c in CFGS:
        d = ood_units(c)
        if names is None:
            names = set(d)
        assert set(d) == names, f"{c} 文件名集合与其他配置不一致"
        keys = sorted(names)
        dice = np.array([d[k][0] for k in keys])
        hit = np.array([d[k][1] for k in keys], dtype=float)
        n, k, D, md, Q = factors(dice, hit)
        # 与冻结表的交叉核对（双路复核）
        assert int(full8.loc[c, "n_undetected"]) == k, \
            f"{c} 未检出数不一致: 我 {k} vs 冻结 {int(full8.loc[c,'n_undetected'])}"
        assert abs(float(full8.loc[c, "detection"]) - D) < 1e-9, f"{c} D 不一致"
        assert abs(float(full8.loc[c, "mean_dice"]) - md) < 1e-9, f"{c} mean_dice 不一致"
        Db, Qb, mdb = boot(dice, hit)
        out["cross_domain"][c] = {
            "n": n, "n_undetected": k,
            "D": D, "D_ci95": ci95(Db), "D_ci_method": "paired bootstrap over cases",
            "Q": Q, "Q_ci95": ci95(Qb),
            "mean_dice": md, "mean_dice_ci95": ci95(mdb),
            "identity_residual": float(abs(md - D * Q)),
        }
        print(f"[ood] {c:>10}  n={n} k={k:>2}  D={D:.4f} [{ci95(Db)[0]:.3f},{ci95(Db)[1]:.3f}]  "
              f"Q={Q:.4f} [{ci95(Qb)[0]:.3f},{ci95(Qb)[1]:.3f}]  MATCH")

    for c in CFGS:
        d = id_units(c)
        keys = sorted(d)
        dice = np.array([d[k][0] for k in keys])
        hit = np.array([d[k][1] for k in keys], dtype=float)
        n, k, D, md, Q = factors(dice, hit)
        assert k == 0, f"{c} 分布内检出失败应为 0 例，实得 {k}"
        Db, Qb, mdb = boot(dice, hit)
        D_low = 1.0 - 3.0 / n                     # 三倍法则：0/n 事件的 95% 上界
        out["in_domain"][c] = {
            "n": n, "n_undetected": k,
            "D": 1.0, "D_ci95": [D_low, 1.0],
            "D_ci_method": "rule of three (0 events observed)",
            "min_recall": float(hit.min()),
            "Q": Q, "Q_ci95": ci95(Qb),
            "mean_dice": md, "mean_dice_ci95": ci95(mdb),
            "identity_residual": float(abs(md - 1.0 * Q)),
        }
        print(f"[id ] {c:>10}  n={n} k={k:>2}  D=1.000 [{D_low:.3f},1.000]  "
              f"Q={Q:.4f} [{ci95(Qb)[0]:.3f},{ci95(Qb)[1]:.3f}]")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
