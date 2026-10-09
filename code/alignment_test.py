# -*- coding: utf-8 -*-
"""
文件名: alignment_test.py
功能: 【论文二 P2-3】对齐性检验 —— Spearman(ID 勾画, OOD 检出)，架构级 n=14

────────────────────────────────────────────────────────────────────────────
H3（冻结原文, 00_docs/论文二方案v4.2.1_冻结.md §4）:
    主张   : Spearman(ID 勾画, OOD 检出) **不显著**
    口径   : 架构级 —— 全部可用架构（n=14 = 13 baseline + EGAUNet）
    推翻判据: **显著且 |ρ| > 0.6** → L2 的"错位"表述必须撤回, 改为"分布内低估而非无视"
    灰带   : 显著门槛 n=14 为 |ρ|=0.538, 推翻门槛 |ρ|>0.6
             ⇒ **0.538 < |ρ| ≤ 0.6 判"不可判定"**, 既不推翻也不支持, 如实写
    必报   : 效应量 CI 与 **MDE（n=14 → |ρ| ≥ 0.72）**
────────────────────────────────────────────────────────────────────────────

⚠️ 本段必须一并声明的三条口径事实（否则会被审稿人指为"拿恒等式当证据"）
 1. **ID 侧冻结判据下 `id_del` ≡ `id_dice`**（精确相等, 本脚本断言）。
    原因: 冻结判据 |P∩G|>0 下 ID 检出恒为 1.000（B6, ε 构造）⇒ Dice = Det×Delin = Delin。
    故主检验在数值上等价于 Spearman(ID Dice, OOD 检出)。这不是选择, 而是冻结口径的后果,
    **必须显式写明**; 相应地 Spearman(id_del, id_dice) = **+1.000000（代数恒等）**,
    **禁作"检验有功效"的证据**（计划 P2-3 验收点明文要求）。
 2. **OOD 侧两条判据严格等价**（P3-1b 实测: OOD 预测用 no-eps 的 |P∩G|/|G|,
    0 就是精确 0）⇒ `ood_det_frozen` ≡ `ood_det_subst`, 本脚本断言并报"判据无关"。
 3. `ood_det` vs `ood_dice` 的 Spearman **很高但非 1**, 仍**不得**当功效证据:
    Dice = Det×Delin 是构造性恒等式, 该相关反映的是"勾画变异小于检出变异",
    不是"检验能发现关联"。

⚠️ n=14 且架构**彼此不独立**（13 个 baseline 共享同一批数据与同一套协议,
   EGAUNet 与消融网格同源）⇒ bootstrap CI **偏乐观**, 仅作描述性区间, 不得当推断区间。
   G3 诚实性要求: 报 CI 时必须同时报"单元非独立"。

⚠️ 灰带口径的实质含义（本段实测 |ρ| 远低于灰带下限, 故灰带不适用, 但 MDE 限定仍适用）:
   MDE@80% = 0.72 ⇒ 本设计对 **|ρ| < 0.72** 的关联**没有检验力**。
   因此结论措辞必须是 **"we cannot exclude a moderate association"**,
   **绝不写 "no association"**（propositions.md §6.3 固定措辞）。

用法:
    PY=D:/miniconda/aniconda/envs/medical_seg/python.exe
    $PY 02_code/analysis/alignment_test.py
产物:
    03_results/stats/alignment.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ 路径 / 常量
E_ROOT = "E:/paper2_ablation_reliability"
UNITS_CSV = f"{E_ROOT}/03_results/stats/p31a_units_n14.csv"
MDE_JSON = f"{E_ROOT}/03_results/stats/proposition_check.json"      # 只读: MDE 预注册来源
OUT_JSON = f"{E_ROOT}/03_results/stats/alignment.json"

X_COL = "id_del"            # 自变量: 分布内勾画因子（P2-1 冻结判据口径, n=14 齐全）
Y_COL = "ood_det_subst"     # 因变量: 分布外实质检出

# ---- 跑前冻结的判据（G6） ----
ALPHA = 0.05
RHO_REJECT = 0.60           # 推翻门槛 |ρ| > 0.6
RHO_SIG_GREY_LO = 0.538     # 预注册显著门槛（灰带下限）
MDE_PRE = 0.72              # 预注册 MDE@80%（K=14）
MDE_TOL = 0.03              # 本次 MC 复核与预注册值的容许差（0.70 vs 0.72 属同一区间）

# ---- MC / bootstrap 设定（固定种子, 可复现） ----
PERM_N = 1_000_000
PERM_SEED = 20260914
PERM_CHUNK = 100_000
BOOT_B = 10_000
BOOT_SEED = 20260914
MDE_REPS = 8_000
MDE_SEED = 42               # 与 proposition_check.py 同种子（同口径复核）
MDE_GRID = np.arange(0.20, 1.00, 0.02)

TOL_IDENT = 1e-15           # 恒等式/精确相等的容差（P3-1bc 教训: 浮点派生量禁用 == 0）


# ------------------------------------------------------------------ 基础统计
def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman ρ（平均秩, 与 scipy.stats.spearmanr 同口径）。"""
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = math.sqrt(float((rx * rx).sum()) * float((ry * ry).sum()))
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def _rankdata(a: np.ndarray) -> np.ndarray:
    """平均秩（并列取均值）, 与 scipy.stats.rankdata(method='average') 一致。"""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sa = a[order]
    i = 0
    n = len(a)
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def t_approx_p(r_abs: float, n: int) -> float:
    """Spearman 的 t 近似双侧 p（df = n-2）。n > 8 时使用（见 propositions.md §6.3）。"""
    if n <= 2:
        return float("nan")
    r_abs = min(abs(r_abs), 1.0 - 1e-15)
    t = r_abs * math.sqrt((n - 2) / max(1e-15, 1.0 - r_abs ** 2))
    from scipy import stats as _s
    return float(2 * _s.t.sf(abs(t), n - 2))


def t_approx_rho_crit(n: int, alpha: float = ALPHA) -> float:
    """t 近似下 α 双侧的临界 |ρ|（显著性门槛）。"""
    from scipy import stats as _s
    tc = _s.t.ppf(1 - alpha / 2, n - 2)
    return float(tc / math.sqrt(n - 2 + tc ** 2))


def perm_null(n: int, reps: int = PERM_N, seed: int = PERM_SEED,
              chunk: int = PERM_CHUNK) -> dict:
    """H0（独立）下的 Spearman 零分布 —— 随机排列 MC（n! 过大无法全枚举）。

    H0 下 y 的秩是 {0..n-1} 的均匀随机排列 ⇒ ρ = 1 - 6·Σd²/(n(n²-1))。
    返回 |ρ| 的分位数与临界值（真双侧: P(|ρ| ≥ c) = α）。
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(n, dtype=np.int64)
    denom = n * (n * n - 1) / 6.0
    abs_rho = None
    done = 0
    while done < reps:
        m = min(chunk, reps - done)
        ry = rng.random((m, n)).argsort(axis=1)          # 每行一个随机排列
        s = ((ry - idx[None, :]) ** 2).sum(axis=1)       # Σd²
        r = 1.0 - s / denom
        ar = np.abs(r)
        abs_rho = ar if abs_rho is None else np.concatenate([abs_rho, ar])
        done += m
    abs_rho.sort()
    # 临界值 = |ρ| 的 (1-α) 分位点：P(perm |ρ| > q) = α（离散分布的 MC 近似）
    q = float(np.quantile(abs_rho, 1.0 - ALPHA))
    return dict(reps=int(reps), seed=int(seed),
                abs_rho_sorted=abs_rho,        # 仅供内部算 p 用, 写 JSON 前 pop 掉
                quantile_95=q,
                crit_exactish=q,
                mean_abs=float(abs_rho.mean()))


def perm_p(r_obs: float, abs_null_sorted: np.ndarray) -> dict:
    """MC 置换 p 值（无偏形式 (b+1)/(N+1)）＋ 标准误。"""
    n = len(abs_null_sorted)
    b = int(np.searchsorted(abs_null_sorted, abs(r_obs), side="left"))
    n_ge = n - b
    p = (n_ge + 1) / (n + 1)
    se = math.sqrt(max(p * (1 - p), 1e-18) / (n + 1))
    # 离散性: 零分布下 |ρ| 的最小正取值 → 给"最小可达 p"参考
    return dict(p_mc=float(p), n_null=int(n), n_ge=int(n_ge), se=float(se))


def bootstrap_ci_rho(x: np.ndarray, y: np.ndarray, b: int = BOOT_B,
                     seed: int = BOOT_SEED) -> dict:
    """配对 bootstrap（重采样**单元**）95% CI。⚠️ 单元非独立 ⇒ 区间偏乐观。"""
    rng = np.random.default_rng(seed)
    n = len(x)
    out = np.empty(b, dtype=float)
    for i in range(b):
        idx = rng.integers(0, n, n)
        out[i] = spearman_rho(x[idx], y[idx])
    out = out[~np.isnan(out)]
    return dict(b=int(b), seed=int(seed), n_valid=int(len(out)),
                lo=float(np.quantile(out, 0.025)),
                hi=float(np.quantile(out, 0.975)),
                median=float(np.median(out)),
                frac_negative=float((out < 0).mean()))


def fisher_ci(rho: float, n: int, alpha: float = ALPHA) -> dict:
    """Fisher-z 解析 CI（作 bootstrap 的独立对照口径）。"""
    from scipy import stats as _s
    z = math.atanh(min(max(rho, -0.999999), 0.999999))
    se = 1.0 / math.sqrt(n - 3)
    zc = _s.norm.ppf(1 - alpha / 2)
    return dict(z=float(z), se=float(se),
                lo=float(math.tanh(z - zc * se)),
                hi=float(math.tanh(z + zc * se)))


def mc_power(k: int, rho_s: float, reps: int = MDE_REPS, seed: int = MDE_SEED,
             alpha: float = ALPHA) -> float:
    """给定域数 k 与总体 Spearman ρ 的 MC 功效（与 proposition_check.py 同口径）。"""
    rng = np.random.default_rng(seed)
    rp = 2.0 * math.sin(math.pi * rho_s / 6.0)       # 二元正态下 Spearman → Pearson
    z = rng.multivariate_normal([0, 0], [[1, rp], [rp, 1]], size=(reps, k))
    ra = _rank_within(z[:, :, 0])
    rb = _rank_within(z[:, :, 1])
    ra = ra - ra.mean(1, keepdims=True)
    rb = rb - rb.mean(1, keepdims=True)
    r = np.clip((ra * rb).sum(1) / np.sqrt((ra * ra).sum(1) * (rb * rb).sum(1)), -1, 1)
    t = r * np.sqrt((k - 2) / np.maximum(1e-15, 1 - r * r))
    from scipy import stats as _s
    p = 2 * _s.t.sf(np.abs(t), k - 2)
    return float((p < alpha).mean())


def _rank_within(a: np.ndarray) -> np.ndarray:
    """沿 axis=1 对每个 rep 内部求秩（a: (reps, k)）。"""
    order = np.argsort(a, axis=1, kind="stable")
    ranks = np.empty_like(a, dtype=float)
    ranks[np.arange(a.shape[0])[:, None], order] = np.arange(a.shape[1])[None, :]
    return ranks


def mde_recheck(k: int) -> dict:
    """本次独立复核 MDE@80%（步长 0.02）。一次遍历同时填功效网格与 MDE。"""
    grid = {}
    mde = None
    for r in MDE_GRID:
        p = round(mc_power(k, float(r)), 4)
        grid[f"{float(r):.2f}"] = p
        if mde is None and p >= 0.80:
            mde = round(float(r), 2)
    return dict(k=int(k), reps=MDE_REPS, seed=MDE_SEED, step=0.02,
                mde80=mde, power_grid=grid)


def reversal_lists(x: np.ndarray, y: np.ndarray, names) -> dict:
    """成对排名反转清单（ID 勾画序 vs OOD 检出序）。"""
    n = len(x)
    rx = _rankdata(x)
    ry = _rankdata(y)
    conc, disc, ties = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = rx[i] - rx[j], ry[i] - ry[j]
            if dx == 0 or dy == 0:
                ties.append((names[i], names[j]))
            elif dx * dy > 0:
                conc.append((names[i], names[j]))
            else:
                disc.append((names[i], names[j]))
    total = n * (n - 1) // 2
    tau_a = (len(conc) - len(disc)) / total if total else float("nan")
    return dict(n_pairs=total, n_concordant=len(conc), n_discordant=len(disc),
                n_tied=len(ties), kendall_tau_a=float(tau_a),
                discordant_pairs=disc,
                examples=disc[:8])


# ------------------------------------------------------------------ 主流程
def main(argv=None):
    ap = argparse.ArgumentParser(description="P2-3 alignment test (H3), n=14 architecture-level")
    ap.add_argument("--units", default=UNITS_CSV)
    ap.add_argument("--out", default=OUT_JSON)
    args = ap.parse_args(argv)

    df = pd.read_csv(args.units)
    n = len(df)
    names = df["arch"].tolist()
    x = df[X_COL].to_numpy(dtype=float)
    y = df[Y_COL].to_numpy(dtype=float)

    # ---- 口径断言（关键: 不满足则拒绝出结论） ----
    assert n == 14, f"口径必须为架构级 n=14, 实得 {n}"
    assert n == len(set(names)), "架构名重复"
    ident_id = bool(np.allclose(df["id_del"], df["id_dice"], rtol=0, atol=TOL_IDENT))
    ident_ood = bool((df["ood_det_frozen"] == df["ood_det_subst"]).all())
    assert ident_id, "ID 侧冻结判据下 id_del 应精确等于 id_dice（B6 后果）"
    assert ident_ood, "OOD 侧两判据应严格等价（P3-1b）"

    # ---- 主检验 ----
    rho = spearman_rho(x, y)
    p_t = t_approx_p(abs(rho), n)
    rho_crit_t = t_approx_rho_crit(n)

    null = perm_null(n)
    abs_null = null.pop("abs_rho_sorted")
    p_perm = perm_p(rho, abs_null)
    rho_crit_perm = null["crit_exactish"]

    boot = bootstrap_ci_rho(x, y)
    fisher = fisher_ci(rho, n)

    mde = mde_recheck(n)
    mde_ref = None
    if os.path.exists(MDE_JSON):
        try:
            mde_ref = json.load(open(MDE_JSON, encoding="utf-8")).get(
                "minimal_design", {}).get("budget_B_domains", {}).get("table", {}).get(
                str(n), {}).get("mde80")
        except Exception:
            mde_ref = None

    # ---- 判定（跑前冻结的判据, 逐字执行） ----
    significant = bool(min(p_t, p_perm["p_mc"]) < ALPHA)
    absr = abs(rho)
    if significant and absr > RHO_REJECT:
        verdict = "REJECT_H3"          # 推翻: 显著且 |ρ|>0.6
    elif (RHO_SIG_GREY_LO < absr <= RHO_REJECT) and significant:
        verdict = "GREY_UNDECIDABLE"   # 灰带: 不推翻也不支持
    elif absr > RHO_SIG_GREY_LO:
        verdict = "GREY_UNDECIDABLE_NONSIG"
    else:
        verdict = "NOT_REJECTED"

    # ---- 附带量（含恒等式标注） ----
    aux = {}
    for a, b in [("id_del", "id_dice"), ("ood_det_subst", "ood_dice"),
                 ("ood_det_subst", "ood_del"), ("id_det_subst", "ood_det_subst"),
                 ("id_del", "ood_del"), ("id_dice", "ood_dice")]:
        aux[f"{a}__{b}"] = dict(
            rho=float(spearman_rho(df[a].to_numpy(float), df[b].to_numpy(float))),
            p_t=float(t_approx_p(abs(spearman_rho(df[a].to_numpy(float),
                                                 df[b].to_numpy(float))), n)))
    identity_note = {
        "id_del__id_dice": ("ALGEBRAIC IDENTITY — 冻结判据下 ID 检出恒为 1.000 (B6), "
                            "故 Dice ≡ Delin。**禁作功效证据**。"),
        "ood_det_subst__ood_dice": ("CONSTRUCTIVE — Dice = Det×Delin 是恒等式; 该高相关反映"
                                    "勾画变异小于检出变异, **禁作功效证据**。"),
    }

    payload = dict(
        segment="P2-3", mode="alignment_test_H3",
        generated_at=datetime.now().isoformat(timespec="seconds"),
        script="02_code/analysis/alignment_test.py",
        hypothesis=dict(
            id="H3",
            statement="Spearman(ID 勾画, OOD 检出) 不显著",
            caliber="架构级; 全部可用架构 n=14 (13 baseline + EGAUNet)",
            reject_criterion=f"显著且 |ρ| > {RHO_REJECT}",
            grey_band=f"{RHO_SIG_GREY_LO} < |ρ| ≤ {RHO_REJECT} → 不可判定",
            verdict=verdict,
        ),
        data=dict(
            units_csv=args.units, n=int(n),
            x_col=X_COL, y_col=Y_COL,
            units=[dict(arch=str(names[i]), x=float(x[i]), y=float(y[i]))
                   for i in range(n)],
            caliber_assertions=dict(
                n_is_14=bool(n == 14),
                id_del_equals_id_dice=ident_id,
                ood_frozen_equals_ood_subst=ident_ood,
                id_det_frozen_is_one=bool((df["id_det_frozen"] == 1.0).all()),
            ),
        ),
        primary=dict(
            rho=float(rho), abs_rho=float(absr), n=int(n),
            p_t_approx=float(p_t),
            p_perm_mc=p_perm,
            rho_crit_alpha05_t_approx=float(rho_crit_t),
            rho_crit_alpha05_perm_mc=float(rho_crit_perm),
            rho_crit_preregistered=float(RHO_SIG_GREY_LO),
            significant_at_alpha05=significant,
        ),
        effect_size_ci=dict(
            bootstrap=boot, fisher_z=fisher,
            caveat=("架构彼此不独立（13 baseline 共享数据与协议, EGAUNet 与消融网格同源）"
                    "⇒ bootstrap 区间**偏乐观**, 仅描述性, 不得作推断区间。"),
        ),
        power=dict(
            mde_preregistered=float(MDE_PRE),
            mde_rechecked=mde.get("mde80"),
            mde_from_proposition_check_json=mde_ref,
            consistent_with_prereg=bool(mde.get("mde80") is not None
                                        and abs(mde["mde80"] - MDE_PRE) <= MDE_TOL),
            recheck=mde,
            mandated_wording=("we cannot exclude a moderate association"
                              "（**绝不写 no association**）"),
        ),
        rank_reversal=reversal_lists(x, y, names),
        auxiliary_spearman=aux,
        identity_flags=identity_note,
        notes=[
            "主检验在数值上等价于 Spearman(ID Dice, OOD 检出)：冻结判据下 ID 检出恒 1.000 ⇒ Dice ≡ Delin。",
            "OOD 侧冻结/实质两判据严格等价（no-eps），故本结论对判据选择不敏感。",
            "|ρ| 远低于灰带下限 ⇒ 灰带不适用；但 |ρ| < MDE(0.72) ⇒ 结论为弱结论，措辞受 MDE 限定。",
        ],
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    # ---------------- 控制台速览 ----------------
    print("=" * 92)
    print("P2-3  对齐性检验 H3  ——  Spearman(ID 勾画, OOD 检出),  架构级 n=14")
    print("=" * 92)
    print(f"  {'arch':14s} {'id_del':>12s} {'ood_det':>10s}")
    for i in range(n):
        print(f"  {names[i]:14s} {x[i]:12.6f} {y[i]:10.6f}")
    print("-" * 92)
    print(f"  ρ (点估计)              = {rho:+.6f}")
    print(f"  p (t 近似, df=12)       = {p_t:.6f}")
    print(f"  p (置换 MC, N={PERM_N}) = {p_perm['p_mc']:.6f}  (se {p_perm['se']:.2e})")
    print(f"  显著门槛 |ρ| (t 近似)   = {rho_crit_t:.4f}   [预注册 {RHO_SIG_GREY_LO}]")
    print(f"  显著门槛 |ρ| (置换 MC)  = {rho_crit_perm:.4f}")
    print(f"  推翻门槛 |ρ|            = {RHO_REJECT}")
    print(f"  bootstrap 95% CI        = [{boot['lo']:+.4f}, {boot['hi']:+.4f}]  "
          f"(neg {boot['frac_negative']*100:.1f}%)")
    print(f"  Fisher-z 95% CI         = [{fisher['lo']:+.4f}, {fisher['hi']:+.4f}]")
    print(f"  MDE@80% (预注册/复核)   = {MDE_PRE} / {mde.get('mde80')}   "
          f"[proposition_check.json: {mde_ref}]")
    print(f"  >>> 判定 = {verdict}")
    print("-" * 92)
    print("  附带量（含恒等式标注）:")
    for k, v in aux.items():
        tag = "  <== 恒等, 禁作功效证据" if k in identity_note else ""
        print(f"    Spearman({k:26s}) = {v['rho']:+.6f}  p_t={v['p_t']:.6f}{tag}")
    rr = payload["rank_reversal"]
    print(f"  排名: 一致 {rr['n_concordant']} / 反转 {rr['n_discordant']} / 并列 {rr['n_tied']}"
          f"  (共 {rr['n_pairs']} 对, Kendall τa = {rr['kendall_tau_a']:+.4f})")
    print(f"\n已写: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
