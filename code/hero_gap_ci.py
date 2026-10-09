# -*- coding: utf-8 -*-
"""
旗舰对的 Dice 差值 · 配对自助置信区间（只读，不写任何源数据）
==============================================================
F1 面板 (d) 的"名次反转"目前只有点估计。审稿人必问：in-domain 那 0.005 的差
是真差还是噪声？这里对同一批图像做配对自助（resample images），给出：
  * in-domain  (n = 242, Kvasir+CVC 测试集)
  * cross-domain (n = 196, ETIS)
两域上 msfa_only - egm_dpa 的 Dice 差值及其 95% 自助 CI。

判读规则（冻结）：
  若某域 CI 跨 0 ⇒ 该域的排名差无法与噪声区分；
  若两域 CI 一跨 0、一不跨 0 ⇒ "排名反转"由跨域检出驱动，而非域内噪声。

依赖：numpy, pandas。随机种子固定，可复现。
"""
import json
import os
import numpy as np
import pandas as pd

ID_ROOT = r"D:/medical_segmentation/experiments"
OOD_ABL = r"D:/medical_segmentation/results_ablation_etis"
OUT = r"E:/paper2_ablation_reliability/03_results/stats/hero_gap_ci.json"

HERO_A, HERO_B = "msfa_only", "egm_dpa"
N_BOOT = 20000
SEED = 20260916


def ood_dice(cfg):
    arr = np.load(os.path.join(OOD_ABL, cfg, "predictions.npy"), allow_pickle=True)
    return {str(d["name"]).replace(".npy", ""): float(d["dice"]) for d in arr}


def id_dice(cfg):
    p = os.path.join(ID_ROOT, "ablation", cfg, "sample_metrics.csv")
    d = pd.read_csv(p)
    col = "Dice" if "Dice" in d.columns else "dice"
    key = "filename" if "filename" in d.columns else "name"
    return d.set_index(key)[col].astype(float).to_dict(), float(d[col].mean())


def paired_boot(a, b, keys, n_boot=N_BOOT, seed=SEED):
    """配对自助：对同一批图像重采样，返回差值均值与 95% 分位 CI。"""
    rng = np.random.default_rng(seed)
    da = np.array([a[k] for k in keys]); db = np.array([b[k] for k in keys])
    diff = da - db
    n = diff.size
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return dict(n=int(n), mean_diff=float(diff.mean()), ci95=[float(lo), float(hi)],
                crosses_zero=bool(lo <= 0.0 <= hi),
                se=float(means.std(ddof=1)),
                p_two_sided_frac=float(2 * min((means <= 0).mean(), (means >= 0).mean())))


def main():
    a_o, b_o = ood_dice(HERO_A), ood_dice(HERO_B)
    k_o = sorted(set(a_o) & set(b_o))
    a_i, ma = id_dice(HERO_A)
    b_i, mb = id_dice(HERO_B)
    k_i = sorted(set(a_i) & set(b_i))

    res = {
        "hero_a": HERO_A, "hero_b": HERO_B,
        "metric": "mean Dice difference (A - B), paired bootstrap over images",
        "n_boot": N_BOOT, "seed": SEED,
        "cross_domain": paired_boot(a_o, b_o, k_o),
        "in_domain": paired_boot(a_i, b_i, k_i),
        "id_mean_dice_check": {HERO_A: ma, HERO_B: mb},
    }
    res["verdict"] = (
        "in-domain gap indistinguishable from 0, cross-domain gap well separated"
        if res["in_domain"]["crosses_zero"] and not res["cross_domain"]["crosses_zero"]
        else "CHECK: pattern not as expected -> user must adjudicate")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    for k in ("in_domain", "cross_domain"):
        r = res[k]
        print(f"{k:>13}: n={r['n']:>3}  mean(A-B)={r['mean_diff']:+.6f}  "
              f"CI95=[{r['ci95'][0]:+.6f}, {r['ci95'][1]:+.6f}]  "
              f"cross0={r['crosses_zero']}  SE={r['se']:.6f}")
    print("\nverdict:", res["verdict"])
    print("[saved]", OUT)


if __name__ == "__main__":
    main()
