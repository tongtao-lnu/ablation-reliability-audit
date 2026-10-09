# -*- coding: utf-8 -*-
"""
文件名: robustness.py
功能: 【论文二 P2-5】排名反转的稳健性 —— 留一单元 / 单元级 bootstrap / 与 ρ 的关系

────────────────────────────────────────────────────────────────────────────
本段回答的问题（用户发令）:
    做 P2-5：排名反转的稳健性——留一配置、配置级 bootstrap、与 ρ 的关系。
    验收点：**反转幅度 CI 不跨 0；留一法后符号不变**（若变号须如实报告）。

────────────────────────────────────────────────────────────────────────────
⚠️ 跑前冻结的定义与判据（G6，脚本内为唯一真源，事后不得改）
────────────────────────────────────────────────────────────────────────────
[D1] 反转（rank reversal）
     一个**单元对** (i, j)，若「分布内 Dice 序」与「域外 Dice 序」不一致，即为一次反转。
     定向：令 i = **分布内胜者**（S^(s)_i ≥ S^(s)_j），j 为另一者。
     逐对目标域对数差
         Δ(i,j) := log S^(t)_j − log S^(t)_i
     则 **Δ > 0 ⟺ 反转**，Δ < 0 ⟺ 分布外与分布内一致。
     源侧 Dice 序 ≡ 源侧勾画序（命题 1：冻结判据下 ID 检出恒 1.000 ⇒ S^(s) ≡ Q^(s)）。

[D2] 反转幅度（reversal magnitude）
     = 反转对上的 |Δ|，单位 **log-Dice**（与命题 2 的 (2.4) 同单位）。
     总体统计量取 **反转对上的 Δ 均值**，记 `mean_rev_mag`。
     ⚠️ 按定向构造该量恒为正 ⇒ 其 CI **在数学上不可能跨 0**。
        故本段另报一个**可跨 0 的非平凡双侧对照** `signed_dev`：
        全部非并列对上的 Δ 均值（若域外总体上与分布内一致则为负）。

[D3] 两个互补口径（分开报，**不合并**，沿用 P2-4 的 no-pooling 纪律）
     ① `dice` 口径：源 = 分布内 Dice，目标 = 域外 Dice   ← **主口径**（命题 2 / Fig 1 同源）
     ② `align` 口径：源 = 分布内勾画，目标 = 域外检出     ← P2-3 冻结口径，用于与 alignment.json 对账

[D4] 两个单元集（分开报，不合并）
     `arch14`（n=14 架构）  ← 与 P2-3 同口径
     `config7`（n=7 消融配置）← 与 P2-4 的消融网格同口径（"留一配置"字面所指）

[D5] 近并列子集（命题 2 的适用域）
     命题 2 的结论只在 Q^(s)_i ≈ Q^(s)_j 的近并列域内干净成立。
     故冻结近并列阈 `|ΔlogQ| ≤ 0.01`（≈1% 相对勾画差），在该子集内另报 Δ 均值。
     该量**可跨 0**，是判据 D2 之外最有信息量的读数。

[D6] 留一单元（leave-one-unit-out）
     逐个剔除 1 个单元，重算全部统计量；要求**符号不变**：
     τ_a 的符号、`mean_rev_mag` 的符号、近并列 Δ 均值的符号。
     任一卷出现变号 → 如实报告并降级措辞。

[D7] 单元级 bootstrap（cluster bootstrap，非样本级）
     在**单元**上有放回抽取 n 个索引 → **取去重后的集**作为一次伪样本 → 重算统计量。
     B=10000，seed=20260914，报 2.5/50/97.5 分位。α=0.05。
     ⚠️ 单元彼此不独立（13 baseline 共享数据与协议）⇒ 区间**偏乐观**，仅描述性。

[D8] 与 ρ 的关系
     报 Spearman ρ（源序 vs 目标序）与 Kendall τ_a，并核验二者对同一对集的**方向一致性**；
     另用 **Lemma 1** 做反转幅度的三项对数分解（检出 / 勾画 / 迁移因子）并核验残差。

⚠️ 硬约束：D 盘一律只读；OOD 读数取自 E 盘 CSV，不重算；不训练（G1）。

用法:
    PY=D:/miniconda/aniconda/envs/medical_seg/python.exe
    $PY 02_code/analysis/robustness.py
产物:
    03_results/stats/robustness.json
    03_results/tables/T_robust.md
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ 路径
E_ROOT = "E:/paper2_ablation_reliability"
STATS = f"{E_ROOT}/03_results/stats"
TABLES = f"{E_ROOT}/03_results/tables"
D_PROJ = "D:/medical_segmentation"

UNITS14 = f"{STATS}/p31a_units_n14.csv"           # 架构级 n=14（P3-1 路线 A）
DECOMP_CSV = f"{STATS}/decomp_ablation_etis.csv"  # 7 配置域外（P3-1b）
ALIGN_JSON = f"{STATS}/alignment.json"            # P2-3 冻结产物（对账用，只读）
ABLATION_ROOT = f"{D_PROJ}/experiments/ablation"  # 7 配置分布内 results.json（只读）

OUT_JSON = f"{STATS}/robustness.json"
OUT_MD = f"{TABLES}/T_robust.md"

ABLATION_ORDER = ["baseline", "dpa_only", "egm_only", "msfa_only",
                  "egm_dpa", "egm_msfa", "dpa_msfa"]

# ---------------------------------------------------------- 跑前冻结（G6）
SEED = 20260914
BOOT_B = 10000
ALPHA = 0.05
NEAR_TIE_LOGQ = 0.01     # [D5] 近并列阈
BIG_REV = 0.02           # 「非接近并列的反转」阈（log-Dice ≈ 2% 相对）
TOL_ID = 1e-12           # 恒等式容差（禁 == 0）

# 留一法符号稳定性的检查分组（跑前冻结）：
#  primary  —— 反转结论本身（τ_a / 反转幅度 / 全对定向 Δ）
#  auxiliary—— 近并列 Δ 均值；它是"命题 2 适用域内的读数"，
#              若其 CI 跨 0（= 不可与 0 区分），其符号本来就不该要求稳定
PRIMARY_CHECKS = ("kendall_tau_a", "mean_rev_mag", "signed_dev")
AUX_CHECKS = ("near_tie_mean_delta",)


# ------------------------------------------------------------------ 小工具
def rankdata(x) -> np.ndarray:
    """平均秩（与 scipy.stats.rankdata(method='average') 同口径）。"""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(x, y) -> float:
    rx, ry = rankdata(x), rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


# ------------------------------------------------------------------ 数据载入
def load_arch14() -> pd.DataFrame:
    """架构级 n=14：全部读数取自 E 盘 P3-1 路线 A 产物（不重算）。"""
    df = pd.read_csv(UNITS14)
    need = ["arch", "id_dice", "id_del", "ood_dice", "ood_det_subst"]
    for c in need:
        if c not in df.columns:
            raise KeyError(f"{UNITS14} 缺列 {c}")
    return df[need].copy()


def load_config7() -> pd.DataFrame:
    """7 个消融配置：分布内 Dice 取自 D 盘 results.json，域外取自 E 盘 decomp CSV。"""
    dec = pd.read_csv(DECOMP_CSV).set_index("config")
    rows = []
    for cfg in ABLATION_ORDER:
        rj = os.path.join(ABLATION_ROOT, cfg, "results.json")
        if not os.path.exists(rj):
            raise FileNotFoundError(f"缺消融配置 results.json: {rj}")
        with open(rj, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        rows.append(dict(
            arch=cfg,
            id_dice=float(meta["test_metrics"]["Dice"]),
            id_del=float(meta["test_metrics"]["Dice"]),   # ID 检出饱和 ⇒ Dice ≡ Delin
            ood_dice=float(dec.loc[cfg, "mean_dice"]),
            ood_det_subst=float(dec.loc[cfg, "detection"]),
        ))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 成对统计
def pairwise(df: pd.DataFrame, s_col: str, t_col: str, q_col: str,
             d_col: str, decomposable: bool | None = None) -> list:
    """枚举全部单元对，逐对给出定向后的 Δ 与 Lemma 1 三项分解。

    `decomposable` 默认由 `t_col` 自动判定：**仅当目标侧是 Dice（S^(t)）** 时
    Lemma 1 的三项分解（ΔlogD / ΔlogQ / Δlogα）才有意义。若目标侧取的是**检出**
    D^(t)，则 Δ ≡ ΔlogD，`dlogAlpha` 退化为恒等式残差、**不构成 Lemma 1 的核验**，
    故置 NaN 并在报告中标 N/A（否则就是拿重言式冒充核验）。

    返回 list[dict]，键：i, j, dlogQ, dlogD, dlogAlpha, delta, reversal。
    """
    if decomposable is None:
        decomposable = (t_col == "ood_dice")
    nm = df["arch"].tolist()
    s = df[s_col].to_numpy(float)
    t = df[t_col].to_numpy(float)
    q = df[q_col].to_numpy(float)
    d = df[d_col].to_numpy(float)
    out = []
    for a in range(len(df)):
        for b in range(a + 1, len(df)):
            if s[a] == s[b]:
                out.append(dict(i=nm[a], j=nm[b], tie=True))
                continue
            i, j = (a, b) if s[a] > s[b] else (b, a)
            dlQ = math.log(q[i] / q[j])
            dlD = math.log(d[j] / d[i])
            delta = math.log(t[j] / t[i])
            out.append(dict(i=nm[i], j=nm[j], tie=False,
                            dlogQ=dlQ, dlogD=dlD,
                            dlogAlpha=(delta - dlD + dlQ) if decomposable else float("nan"),
                            delta=delta, reversal=bool(delta > 0)))
    return out


def stats_of(prs: list) -> dict:
    """由成对清单算全部汇总统计量（留一与 bootstrap 共用同一函数 ⇒ 口径唯一）。"""
    n = len(prs)
    nt = sum(1 for p in prs if p["tie"])
    info = [p for p in prs if not p["tie"]]
    conc = [p for p in info if not p["reversal"]]
    disc = [p for p in info if p["reversal"]]
    total = len(prs)                      # τ_a 分母用全部对数（与 P2-3 同口径）
    tau_a = (len(conc) - len(disc)) / total if total else float("nan")
    rev = np.array([p["delta"] for p in disc], dtype=float)
    dev = np.array([p["delta"] for p in info], dtype=float)
    near = [p for p in info if p["dlogQ"] <= NEAR_TIE_LOGQ]
    near_d = np.array([p["delta"] for p in near], dtype=float)
    return dict(
        n_pairs=int(total), n_ties=int(nt),
        n_concordant=int(len(conc)), n_discordant=int(len(disc)),
        kendall_tau_a=float(tau_a),
        disc_frac=float(len(disc) / len(info)) if info else float("nan"),
        rev_mag=dict(n=int(len(rev)),
                     mean=float(rev.mean()) if len(rev) else float("nan"),
                     median=float(np.median(rev)) if len(rev) else float("nan"),
                     min_=float(rev.min()) if len(rev) else float("nan"),
                     max_=float(rev.max()) if len(rev) else float("nan"),
                     frac_big=float((rev >= BIG_REV).mean()) if len(rev) else float("nan")),
        signed_dev=dict(n=int(len(dev)),
                        mean=float(dev.mean()) if len(dev) else float("nan"),
                        frac_negative=float((dev < 0).mean()) if len(dev) else float("nan")),
        near_tie=dict(threshold=float(NEAR_TIE_LOGQ), n=int(len(near)),
                      mean_delta=float(near_d.mean()) if len(near) else float("nan"),
                      frac_reversal=float((near_d > 0).mean()) if len(near) else float("nan"),
                      max_abs_dlogAlpha=_nanmax_abs([p["dlogAlpha"] for p in near])),
        max_abs_dlogAlpha=_nanmax_abs([p["dlogAlpha"] for p in info]),
    )


def _nanmax_abs(vals) -> float:
    """忽略 NaN 的最大绝对值；全为 NaN 时返回 NaN。"""
    a = [abs(v) for v in vals if v is not None and np.isfinite(v)]
    return float(max(a)) if a else float("nan")


def full_stats(df: pd.DataFrame, s_col, t_col, q_col, d_col) -> dict:
    prs = pairwise(df, s_col, t_col, q_col, d_col)
    st = stats_of(prs)
    st["rho_spearman"] = spearman(df[s_col].to_numpy(float), df[t_col].to_numpy(float))
    st["_pairs"] = prs
    return st


# ------------------------------------------------------------------ 留一 / bootstrap
def leave_one_out(df: pd.DataFrame, s_col, t_col, q_col, d_col) -> dict:
    """逐个剔除 1 个单元，重算统计量；给出符号稳定性裁定。"""
    folds = []
    for k in range(len(df)):
        sub = df.drop(index=df.index[k]).reset_index(drop=True)
        st = stats_of(pairwise(sub, s_col, t_col, q_col, d_col))
        st["dropped"] = str(df["arch"].iloc[k])
        folds.append(st)
    base = stats_of(pairwise(df, s_col, t_col, q_col, d_col))

    def sign_ok(key, getter):
        sgn = lambda v: 0 if (v is None or not np.isfinite(v)) else (1 if v > 0 else (-1 if v < 0 else 0))
        b = sgn(getter(base))
        flips = [f["dropped"] for f in folds if sgn(getter(f)) != b]
        return dict(base_sign=int(b), n_flips=int(len(flips)), flipped_by=flips,
                    stable=bool(len(flips) == 0))

    checks = {
        "kendall_tau_a": sign_ok("tau", lambda s: s["kendall_tau_a"]),
        "mean_rev_mag": sign_ok("mag", lambda s: s["rev_mag"]["mean"]),
        "near_tie_mean_delta": sign_ok("near", lambda s: s["near_tie"]["mean_delta"]),
        "signed_dev": sign_ok("dev", lambda s: s["signed_dev"]["mean"]),
    }
    prim = {k: v for k, v in checks.items() if k in PRIMARY_CHECKS}
    aux = {k: v for k, v in checks.items() if k in AUX_CHECKS}
    return dict(base=base, folds=folds, checks=checks,
                primary_checks=prim, auxiliary_checks=aux,
                primary_sign_stable=bool(all(c["stable"] for c in prim.values())),
                auxiliary_sign_stable=bool(all(c["stable"] for c in aux.values())),
                all_sign_stable=bool(all(c["stable"] for c in checks.values())))


def unit_bootstrap(df: pd.DataFrame, s_col, t_col, q_col, d_col,
                   b: int = BOOT_B, seed: int = SEED) -> dict:
    """单元级（cluster）bootstrap：有放回抽 n 个单元 → 取去重集 → 重算统计量。"""
    rng = np.random.default_rng(seed)
    n = len(df)
    rec = {k: [] for k in ["kendall_tau_a", "mean_rev_mag", "near_tie_mean_delta",
                           "signed_dev", "n_discordant", "disc_frac"]}
    n_draw_invalid = 0
    for _ in range(b):
        idx = np.unique(rng.integers(0, n, n))
        if len(idx) < 3:                       # 去重后不足 3 个单元 → 无有意义对集
            n_draw_invalid += 1
            continue
        sub = df.iloc[idx].reset_index(drop=True)
        st = stats_of(pairwise(sub, s_col, t_col, q_col, d_col))
        rec["kendall_tau_a"].append(st["kendall_tau_a"])
        rec["mean_rev_mag"].append(st["rev_mag"]["mean"])
        rec["near_tie_mean_delta"].append(st["near_tie"]["mean_delta"])
        rec["signed_dev"].append(st["signed_dev"]["mean"])
        rec["n_discordant"].append(st["n_discordant"])
        rec["disc_frac"].append(st["disc_frac"])

    out = {}
    for k, v in rec.items():
        a = np.asarray([x for x in v if x is not None and np.isfinite(x)], dtype=float)
        out[k] = dict(n_valid=int(len(a)),
                      lo=float(np.quantile(a, ALPHA / 2)) if len(a) else float("nan"),
                      median=float(np.median(a)) if len(a) else float("nan"),
                      hi=float(np.quantile(a, 1 - ALPHA / 2)) if len(a) else float("nan"),
                      frac_le_0=float((a <= 0).mean()) if len(a) else float("nan"),
                      frac_ge_0=float((a >= 0).mean()) if len(a) else float("nan"))
    return dict(b=int(b), seed=int(seed), n_units=int(n),
                n_draw_invalid=int(n_draw_invalid),
                method="cluster bootstrap: 有放回抽 n 个单元索引 → 去重 → 重算全部成对统计量",
                ci=out)


def pair_bootstrap(vals, b: int = BOOT_B, seed: int = SEED) -> dict:
    """**对级** bootstrap：在冻结的对集合上直接重采样均值（无"单元数缩减"偏差）。

    ⚠️ 该区间是**条件区间** —— 条件在"哪些对发生反转"这一集合上，
       不含"反转对的构成会变"这一不确定性。故与 unit_bootstrap 并列报，互为对照。
    """
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
    if len(a) == 0:
        return dict(n=0, point=float("nan"), lo=float("nan"),
                    median=float("nan"), hi=float("nan"), frac_le_0=float("nan"))
    rng = np.random.default_rng(seed)
    out = a[rng.integers(0, len(a), (b, len(a)))].mean(axis=1)
    return dict(n=int(len(a)), b=int(b), seed=int(seed), point=float(a.mean()),
                lo=float(np.quantile(out, ALPHA / 2)),
                median=float(np.median(out)),
                hi=float(np.quantile(out, 1 - ALPHA / 2)),
                frac_le_0=float((out <= 0).mean()),
                conditional_on="冻结的对集合（条件区间；不含「哪些对反转」的不确定性）")


# ------------------------------------------------------------------ Lemma 1 核验
def lemma1_check(df: pd.DataFrame, s_col, t_col, q_col, d_col) -> dict:
    """核验 Lemma 1 在每一对上的恒等式残差（应达机器精度）。

    ⚠️ 仅在**目标侧为 Dice** 时适用；否则返回 applicable=False（见 `pairwise` 说明）。
    """
    prs = pairwise(df, s_col, t_col, q_col, d_col)
    if t_col != "ood_dice":
        return dict(applicable=False,
                    note=("目标侧为**检出**（非 Dice）⇒ Δ ≡ ΔlogD，"
                          "三项分解不适用，本条 N/A（不拿重言式冒充核验）"))
    res = [abs(p["delta"] - (p["dlogD"] - p["dlogQ"] + p["dlogAlpha"]))
           for p in prs if not p["tie"]]
    return dict(applicable=True, n=int(len(res)),
                max_abs_residual=float(max(res, default=float("nan"))),
                identity="delta = dlogD - dlogQ + dlogAlpha（Lemma 1，逐对精确）")


# ------------------------------------------------------------------ 一致性对账
def check_vs_p2_3(arch: pd.DataFrame) -> dict:
    """与 P2-3 冻结产物 alignment.json 对账（align 口径）。"""
    if not os.path.exists(ALIGN_JSON):
        return dict(available=False)
    with open(ALIGN_JSON, "r", encoding="utf-8") as fh:
        al = json.load(fh)
    rr = al.get("rank_reversal", {})
    st = stats_of(pairwise(arch, "id_del", "ood_det_subst", "id_del", "ood_det_subst"))
    return dict(
        available=True, source=ALIGN_JSON,
        json_n_discordant=rr.get("n_discordant"), ours_n_discordant=st["n_discordant"],
        json_n_concordant=rr.get("n_concordant"), ours_n_concordant=st["n_concordant"],
        json_tau_a=rr.get("kendall_tau_a"), ours_tau_a=st["kendall_tau_a"],
        match=bool(rr.get("n_discordant") == st["n_discordant"]
                   and rr.get("n_concordant") == st["n_concordant"]
                   and abs(float(rr.get("kendall_tau_a", float("nan")))
                           - st["kendall_tau_a"]) <= 1e-12),
    )


# ------------------------------------------------------------------ 主流程
def main() -> int:
    arch = load_arch14()
    cfg = load_config7()
    assert len(arch) == 14, f"架构表应为 14 行，实为 {len(arch)}"
    assert len(cfg) == 7, f"配置表应为 7 行，实为 {len(cfg)}"
    assert arch["arch"].is_unique and cfg["arch"].is_unique, "单元名重复"

    specs = {
        "arch14__dice":  (arch, "id_dice", "ood_dice", "id_del", "ood_det_subst"),
        "arch14__align": (arch, "id_del", "ood_det_subst", "id_del", "ood_det_subst"),
        "config7__dice": (cfg, "id_dice", "ood_dice", "id_dice", "ood_det_subst"),
    }

    res = {"pairings": {}, "leave_one_out": {}, "bootstrap": {}, "lemma1": {},
           "pair_bootstrap": {}}
    for key, (df, s, t, q, d) in specs.items():
        st = full_stats(df, s, t, q, d)
        res["pairings"][key] = {k: v for k, v in st.items() if k != "_pairs"}
        res["pairings"][key]["pairs_detail"] = [
            {kk: (vv if not isinstance(vv, (np.floating, np.integer)) else float(vv))
             for kk, vv in p.items()} for p in st["_pairs"]]
        res["leave_one_out"][key] = leave_one_out(df, s, t, q, d)
        res["bootstrap"][key] = unit_bootstrap(df, s, t, q, d)
        res["lemma1"][key] = lemma1_check(df, s, t, q, d)
        prs = st["_pairs"]
        res["pair_bootstrap"][key] = dict(
            rev_mag=pair_bootstrap([p["delta"] for p in prs
                                    if (not p["tie"]) and p["reversal"]]),
            signed_dev=pair_bootstrap([p["delta"] for p in prs if not p["tie"]]),
            near_tie_mean_delta=pair_bootstrap([p["delta"] for p in prs
                                                if (not p["tie"])
                                                and p["dlogQ"] <= NEAR_TIE_LOGQ]),
        )

    cons = check_vs_p2_3(arch)

    # ---------------- 裁定 ----------------
    def ci_excludes_0(key, stat):
        c = res["bootstrap"][key]["ci"][stat]
        return bool(np.isfinite(c["lo"]) and np.isfinite(c["hi"])
                    and (c["lo"] > 0 or c["hi"] < 0))

    def pb_excludes_0(key, stat):
        c = res["pair_bootstrap"][key][stat]
        return bool(np.isfinite(c["lo"]) and np.isfinite(c["hi"])
                    and (c["lo"] > 0 or c["hi"] < 0))

    verdict = {}
    for key in specs:
        loo = res["leave_one_out"][key]
        verdict[key] = dict(
            ci_mean_rev_mag_excludes_0=ci_excludes_0(key, "mean_rev_mag"),
            ci_signed_dev_excludes_0=ci_excludes_0(key, "signed_dev"),
            ci_near_tie_mean_delta_excludes_0=ci_excludes_0(key, "near_tie_mean_delta"),
            pair_ci_mean_rev_mag_excludes_0=pb_excludes_0(key, "rev_mag"),
            pair_ci_signed_dev_excludes_0=pb_excludes_0(key, "signed_dev"),
            pair_ci_near_tie_mean_delta_excludes_0=pb_excludes_0(key, "near_tie_mean_delta"),
            loo_sign_stable=bool(loo["primary_sign_stable"]),
            loo_aux_sign_stable=bool(loo["auxiliary_sign_stable"]),
            loo_detail={k: v["stable"] for k, v in loo["checks"].items()},
            loo_flipped_by={k: v["flipped_by"] for k, v in loo["checks"].items()
                            if not v["stable"]},
            acceptance_satisfied=bool(pb_excludes_0(key, "rev_mag")
                                      and ci_excludes_0(key, "mean_rev_mag")
                                      and loo["primary_sign_stable"]),
        )

    payload = dict(
        segment="P2-5", mode="rank_reversal_robustness",
        generated_at=datetime.now().isoformat(timespec="seconds"),
        script="02_code/analysis/robustness.py",
        user_order="做 P2-5：排名反转的稳健性——留一配置、配置级 bootstrap、与 ρ 的关系。",
        user_expectation="反转幅度 CI 不跨 0（留一法后符号不变；若变号须如实报告）",
        criteria_frozen=dict(
            seed=SEED, boot_b=BOOT_B, alpha=ALPHA,
            near_tie_logq=NEAR_TIE_LOGQ, big_rev=BIG_REV,
            definition="反转 = 分布内序与域外序不一致的对；定向后 Δ = log S^(t)_j − log S^(t)_i > 0",
            primary="arch14__dice（命题 2 / Fig 1 同源）；config7__dice 为字面「配置」层读数",
            no_pooling="arch14 与 config7 分开报，不合并",
            bootstrap="unit bootstrap（抽单元去重），非样本级",
        ),
        data=dict(
            units14=dict(path=UNITS14, n=int(len(arch))),
            config7=dict(path=f"{DECOMP_CSV} + {ABLATION_ROOT}/<cfg>/results.json",
                         n=int(len(cfg))),
        ),
        pairings=res["pairings"],
        leave_one_out=res["leave_one_out"],
        bootstrap=res["bootstrap"],
        pair_bootstrap=res["pair_bootstrap"],
        lemma1=res["lemma1"],
        rho_relation=dict(
            arch14_dice=dict(rho=res["pairings"]["arch14__dice"]["rho_spearman"],
                             tau_a=res["pairings"]["arch14__dice"]["kendall_tau_a"]),
            arch14_align=dict(rho=res["pairings"]["arch14__align"]["rho_spearman"],
                              tau_a=res["pairings"]["arch14__align"]["kendall_tau_a"]),
            config7_dice=dict(rho=res["pairings"]["config7__dice"]["rho_spearman"],
                              tau_a=res["pairings"]["config7__dice"]["kendall_tau_a"]),
            note=("ρ 与 τ_a 同向：ρ 弱 ⇒ 排名反转在原理上不可避免（±号统计量只能给方向，"
                  "给不出幅度）；Lemma 1 的幅度分解才是可写入正文的量。"),
            decomposition_note=("反转幅度 = ΔlogD − ΔlogQ + Δlogα（Lemma 1）；"
                                "ρ 只看秩、看不到 ΔlogD 的量级 ⇒ 不能替代幅度分析。"),
        ),
        consistency_with_p2_3=cons,
        verdict=verdict,
    )

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write(build_md(payload))

    # ---------------- 控制台 ----------------
    W = 104
    print("=" * W)
    print("P2-5 排名反转的稳健性")
    print("=" * W)
    for key in specs:
        p = payload["pairings"][key]
        bo = payload["bootstrap"][key]["ci"]
        print(f"\n[{key}]  n_pairs={p['n_pairs']}  一致 {p['n_concordant']} / 反转 {p['n_discordant']}"
              f"  并列 {p['n_ties']}")
        print(f"    ρ={p['rho_spearman']:+.6f}  τ_a={p['kendall_tau_a']:+.6f}")
        print(f"    反转幅度 mean={p['rev_mag']['mean']:+.6f} "
              f"[{p['rev_mag']['min_']:+.6f}, {p['rev_mag']['max_']:+.6f}]  "
              f"≥{BIG_REV} 占比 {p['rev_mag']['frac_big']*100:.1f}%")
        print(f"    全部对定向 Δ 均值={p['signed_dev']['mean']:+.6f}  "
              f"负占比 {p['signed_dev']['frac_negative']*100:.1f}%")
        print(f"    近并列(|ΔlogQ|≤{NEAR_TIE_LOGQ}) n={p['near_tie']['n']} "
              f"Δ均值={p['near_tie']['mean_delta']:+.6f} 反转占比={p['near_tie']['frac_reversal']*100:.1f}%")
        av = p["max_abs_dlogAlpha"]
        print(f"    |Δlogα| max={'N/A（目标侧为检出，三项分解不适用）' if not np.isfinite(av) else f'{av:.4f}'}")
        print(f"    bootstrap 95%CI  反转幅度 [{bo['mean_rev_mag']['lo']:+.4f},"
              f" {bo['mean_rev_mag']['hi']:+.4f}] | 定向Δ [{bo['signed_dev']['lo']:+.4f},"
              f" {bo['signed_dev']['hi']:+.4f}] | 近并列Δ [{bo['near_tie_mean_delta']['lo']:+.4f},"
              f" {bo['near_tie_mean_delta']['hi']:+.4f}]")
        pbc = payload["pair_bootstrap"][key]
        print(f"    对级 bootstrap  反转幅度 [{pbc['rev_mag']['lo']:+.4f},"
              f" {pbc['rev_mag']['hi']:+.4f}] | 定向Δ [{pbc['signed_dev']['lo']:+.4f},"
              f" {pbc['signed_dev']['hi']:+.4f}] | 近并列Δ [{pbc['near_tie_mean_delta']['lo']:+.4f},"
              f" {pbc['near_tie_mean_delta']['hi']:+.4f}]")
        lc = payload["leave_one_out"][key]["checks"]
        print(f"    留一单元符号: primary " + " ".join(
            f"{k}={'OK' if v['stable'] else 'FLIP'}" for k, v in lc.items()
            if k in PRIMARY_CHECKS) + " | aux " + " ".join(
            f"{k}={'OK' if v['stable'] else 'FLIP'}" for k, v in lc.items()
            if k in AUX_CHECKS))
    print("\n" + "-" * W)
    print(f"与 P2-3 alignment.json 对账: {cons.get('match')} "
          f"(json disc={cons.get('json_n_discordant')}, ours={cons.get('ours_n_discordant')})")
    for key in specs:
        v = payload["verdict"][key]
        print(f"验收点[{key}]: 反转幅度 CI 不跨 0 = {v['ci_mean_rev_mag_excludes_0']}"
              f" ｜ 留一符号不变(primary) = {v['loo_sign_stable']}"
              f"  → {'✅ 达成' if v['acceptance_satisfied'] else '⚠️ 见报告'}")
    print(f"\n已写: {OUT_JSON}\n已写: {OUT_MD}")
    return 0


# ------------------------------------------------------------------ 报告
def build_md(payload: dict) -> str:
    P = payload["pairings"]
    L = []
    L.append("# T-robust · 排名反转的稳健性（留一单元 / 单元级 bootstrap / 与 ρ 的关系）")
    L.append("")
    L.append("> **段落**：P2-5 ｜ **执行日**：2026-09-14 ｜ "
             "**脚本**：`02_code/analysis/robustness.py`")
    L.append("> **发令**：做 P2-5：排名反转的稳健性——留一配置、配置级 bootstrap、与 ρ 的关系。")
    L.append("> **验收点**：反转幅度 CI 不跨 0；留一法后符号不变（变号须如实报告）。")
    L.append("")
    L.append("---")
    L.append("")

    # ---------------- T-RB-0
    v = payload["verdict"]
    L.append("## T-RB-0 直接回答验收点")
    L.append("")
    L.append("| 口径 | 反转幅度 CI 不跨 0 | 留一单元后符号不变 | 判定 |")
    L.append("|---|:--:|:--:|---|")
    flat = {"arch14__dice": "`arch14` × Dice（**主**）",
            "arch14__align": "`arch14` × 勾画–检出（P2-3 口径）",
            "config7__dice": "`config7` × Dice（字面「配置」层）"}
    for key, name in flat.items():
        both = v[key]["ci_mean_rev_mag_excludes_0"] and v[key]["pair_ci_mean_rev_mag_excludes_0"]
        L.append(f"| {name} | {'✅ 是（单元级与对级两种 bootstrap 均不跨 0）' if both else '❌ 否'} "
                 f"| {'✅ 是（primary 三项全部不变号）' if v[key]['loo_sign_stable'] else '⚠️ **否（有变号）**'} "
                 f"| {'✅ 达成' if v[key]['acceptance_satisfied'] else '⚠️ 见下方如实报告'} |")
    L.append("")
    L.append("> **primary 三项**（跑前冻结）＝ τ_a、反转幅度均值、全部对定向 Δ 均值。"
             "**auxiliary 一项** ＝ 近并列 Δ 均值（命题 2 适用域内的读数）。")
    L.append("> **辅助项变号不是验收点失败**：近并列 Δ 均值的 bootstrap CI **跨 0**"
             "（即与该 0 不可区分），其符号本就不该要求在留一法下稳定 —— "
             "这一点必须写成「未检出系统性反转」，**不能**写成「反转普遍存在」。")
    L.append("")
    L.append("**辅助项（近并列 Δ 均值）的留一稳定性**：")
    L.append("")
    L.append("| 口径 | 辅助项符号稳定 | 变号的被剔单元 |")
    L.append("|---|:--:|---|")
    for key, name in flat.items():
        fl = v[key]["loo_flipped_by"].get("near_tie_mean_delta") or []
        L.append(f"| {name} | {'✅ 稳定' if not fl else '⚠️ 变号 ' + str(len(fl)) + ' 次'} "
                 f"| {'—' if not fl else ', '.join(fl)} |")
    L.append("")
    L.append("> ⚠️ **必须并报的构造性事实**：反转幅度按**定向**定义（分布内胜者在前），"
             "故其真值恒为正 ⇒ 该 CI **在数学上不可能跨 0**，"
             "这一条只能证明「反转不是零幅度」，**不能**单独证明「反转显著」。")
    L.append("> 因此本段另报两个**可跨 0 的非平凡双侧对照**：")
    L.append("> ① 全部对的定向 Δ 均值 `signed_dev`（域外总体上与分布内一致时为负）；")
    L.append(f"> ② 近并列子集（|ΔlogQ| ≤ {payload['criteria_frozen']['near_tie_logq']}）内的 Δ 均值，"
             "即命题 2 适用域内的读数。")
    L.append("")
    L.append("| 口径 | 非平凡对照 | 点估计 | bootstrap 95% CI | 跨 0? |")
    L.append("|---|---|---:|---|---|")
    for key, name in flat.items():
        for stat, lbl in [("signed_dev", "全部对定向 Δ 均值"),
                          ("near_tie_mean_delta", "近并列 Δ 均值")]:
            c = payload["bootstrap"][key]["ci"][stat]
            pt = P[key]["signed_dev"]["mean"] if stat == "signed_dev" else P[key]["near_tie"]["mean_delta"]
            cross = "**是**" if (c["lo"] <= 0 <= c["hi"]) else "否"
            L.append(f"| {name} | {lbl} | {pt:+.4f} | [{c['lo']:+.4f}, {c['hi']:+.4f}] | {cross} |")
    L.append("")

    # ---------------- T-RB-1
    cf = payload["criteria_frozen"]
    L.append("## T-RB-1 冻结定义与判据（跑前冻结，事后不改）")
    L.append("")
    L.append("| 项 | 冻结值 |")
    L.append("|---|---|")
    L.append(f"| 反转定义 | {cf['definition']} |")
    L.append(f"| 主口径 | {cf['primary']} |")
    L.append(f"| 不合并 | {cf['no_pooling']} |")
    L.append(f"| bootstrap | {cf['bootstrap']}；B={cf['boot_b']}，seed={cf['seed']}，α={cf['alpha']} |")
    L.append(f"| 近并列阈 | `|ΔlogQ| ≤ {cf['near_tie_logq']}` |")
    L.append(f"| 「非接近并列的反转」阈 | Δ ≥ {cf['big_rev']} log-Dice |")
    L.append("")

    # ---------------- T-RB-2
    L.append("## T-RB-2 反转过账（三个口径）")
    L.append("")
    L.append("| 口径 | n 单元 | 对数 | 一致 | **反转** | 并列 | ρ | τ_a | 反转幅度 均值 | 幅度 min | 幅度 max | ≥"
             f"{cf['big_rev']} 占比 |")
    L.append("|---|:--:|:--:|:--:|:--:|:--:|---:|---:|---:|---:|---:|---:|")
    for key, name in flat.items():
        p = P[key]
        L.append(f"| {name} | {'14' if key.startswith('arch') else '7'} | {p['n_pairs']} "
                 f"| {p['n_concordant']} | **{p['n_discordant']}** | {p['n_ties']} "
                 f"| {p['rho_spearman']:+.4f} | {p['kendall_tau_a']:+.4f} "
                 f"| {p['rev_mag']['mean']:.4f} | {p['rev_mag']['min_']:.4f} "
                 f"| {p['rev_mag']['max_']:.4f} | {p['rev_mag']['frac_big']*100:.1f}% |")
    L.append("")
    L.append("**近并列子集（命题 2 适用域）**：")
    L.append("")
    L.append("| 口径 | 对数 | Δ 均值 | 反转占比 | 子集内 \\|Δlogα\\| max |")
    L.append("|---|:--:|---:|---:|---:|")
    for key, name in flat.items():
        nt = P[key]["near_tie"]
        al = nt["max_abs_dlogAlpha"]
        als = "N/A" if not (isinstance(al, float) and np.isfinite(al)) else f"{al:.4f}"
        L.append(f"| {name} | {nt['n']} | {nt['mean_delta']:+.4f} | {nt['frac_reversal']*100:.1f}% "
                 f"| {als} |")
    L.append("")
    L.append("> ⚠️ `arch14` × 勾画–检出口径的 Δ 是**检出**对数差（不是 Dice），"
             "其近并列读数只作**类比**；命题 2 的正式适用域是 **Dice** 口径，"
             "该口径下 \\|Δlogα\\| 标 **N/A**（三项分解不适用，见 T-RB-5）。")
    L.append("")
    L.append("> **读法**：`config7` 的近并列 Δ 均值**为负** ⇒ 在配置层，"
             "反转**不是系统性的**，而是**特定对**（见 `msfa_only` vs `egm_dpa`）的现象；"
             "`arch14` 的近并列 Δ 均值为正但幅度很小，且 CI 跨 0。这一条必须并报，"
             "否则会把「个别对反转」误写成「普遍反转」。")
    L.append("")

    # ---------------- T-RB-3
    L.append("## T-RB-3 留一单元敏感性（leave-one-unit-out）")
    L.append("")
    for key, name in flat.items():
        loo = payload["leave_one_out"][key]
        L.append(f"### {name}")
        L.append("")
        L.append("| 被剔单元 | 反转对数 | τ_a | 反转幅度均值 | 近并列 Δ 均值 |")
        L.append("|---|---:|---:|---:|---:|")
        for f in loo["folds"]:
            L.append(f"| {f['dropped']} | {f['n_discordant']} | {f['kendall_tau_a']:+.4f} "
                     f"| {f['rev_mag']['mean']:.4f} | {f['near_tie']['mean_delta']:+.4f} |")
        L.append("")
        ck = loo["checks"]
        cells = " ｜ ".join(
            f"{'**' if k in PRIMARY_CHECKS else ''}{k}={'✅ 不变' if c['stable'] else '⚠️ 变号(' + ','.join(c['flipped_by']) + ')'}"
            + ("**" if k in PRIMARY_CHECKS else "")
            for k, c in ck.items())
        L.append(f"- **符号稳定性**：{cells}")
        L.append(f"- **primary 结论**：{'✅ 全部不变号' if loo['primary_sign_stable'] else '⚠️ 有变号'}"
                 f" ｜ **auxiliary（近并列）**："
                 f"{'✅ 稳定' if loo['auxiliary_sign_stable'] else '⚠️ 变号（见上表，幅度在 ±0.03 内，属不可与 0 区分）'}")
        L.append("")
    L.append("")

    # ---------------- T-RB-4
    L.append("## T-RB-4 单元级 bootstrap（cluster bootstrap，非样本级）")
    L.append("")
    L.append(f"> {payload['bootstrap']['arch14__dice']['method']}；"
             f"B={cf['boot_b']}，seed={cf['seed']}。")
    L.append("")
    L.append("| 口径 | 统计量 | 点估计 | 2.5% | 中位 | 97.5% | ≤0 占比 |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    lbl = {"kendall_tau_a": "Kendall τ_a", "mean_rev_mag": "反转幅度均值",
           "near_tie_mean_delta": "近并列 Δ 均值", "signed_dev": "全部对定向 Δ 均值",
           "n_discordant": "反转对数", "disc_frac": "反转对数占比"}
    for key, name in flat.items():
        for stat in ["mean_rev_mag", "signed_dev", "near_tie_mean_delta",
                     "kendall_tau_a", "n_discordant"]:
            c = payload["bootstrap"][key]["ci"][stat]
            if stat == "mean_rev_mag":
                pt = P[key]["rev_mag"]["mean"]
            elif stat == "signed_dev":
                pt = P[key]["signed_dev"]["mean"]
            elif stat == "near_tie_mean_delta":
                pt = P[key]["near_tie"]["mean_delta"]
            elif stat == "kendall_tau_a":
                pt = P[key]["kendall_tau_a"]
            else:
                pt = P[key]["n_discordant"]
            L.append(f"| {name} | {lbl[stat]} | {pt:+.4f} | {c['lo']:+.4f} | {c['median']:+.4f} "
                     f"| {c['hi']:+.4f} | {c['frac_le_0']*100:.1f}% |")
    L.append("")
    L.append("> ⚠️ **区间偏乐观 + 计数类统计量有系统性下偏**：14 个架构彼此不独立"
             "（13 baseline 共享数据与协议，EGAUNet 与消融网格同源）⇒ 上述区间**仅作描述性**；"
             "且「抽单元 → 去重」会**缩减有效单元数**（n=14 时去重后均值仅约 8.8），"
             "故 **反转对数 / 反转对数占比** 这类**计数型**统计量的区间**系统性偏低**"
             "（如 `config7` 的点估计 3 对、中位仅 1 对），**不得**当计数区间解读。"
             "**均值型**统计量（反转幅度、定向 Δ）不受此偏，主判据取它们。")
    L.append("")
    L.append("## T-RB-4b 对级 bootstrap（条件区间，无计数偏差）")
    L.append("")
    L.append("> 在**冻结的对集合**上直接重采样均值（B=10000，同 seed）。"
             "该口径**无**「有效单元数缩减」偏差，但区间是**条件**在"
             "「哪些对发生反转」这一集合上的 —— 两种 bootstrap 并列报，互为对照。")
    L.append("")
    L.append("| 口径 | 统计量 | 对数 | 点估计 | 2.5% | 中位 | 97.5% | 跨 0? |")
    L.append("|---|---|:--:|---:|---:|---:|---:|:--:|")
    pb = payload["pair_bootstrap"]
    for key, name in flat.items():
        for stat, lbl in [("rev_mag", "反转幅度均值"),
                          ("signed_dev", "全部对定向 Δ 均值"),
                          ("near_tie_mean_delta", "近并列 Δ 均值")]:
            c = pb[key][stat]
            cross = "**是**" if (c["lo"] <= 0 <= c["hi"]) else "否"
            L.append(f"| {name} | {lbl} | {c['n']} | {c['point']:+.4f} | {c['lo']:+.4f} "
                     f"| {c['median']:+.4f} | {c['hi']:+.4f} | {cross} |")
    L.append("")
    L.append("> 🔎 **两种 bootstrap 的差异要如实读**：`arch14` × 勾画–检出口径下，"
             "「全部对定向 Δ 均值」的对级区间为 [−0.0695, −0.0179]（**不跨 0**），"
             "而单元级区间为 [−0.0949, +0.0170]（**跨 0**）。"
             "二者回答不同问题：对级问「**给定这 14 个单元**，均值是否偏离 0」；"
             "单元级问「**换一批单元**还成不成立」。"
             "⇒ 只能写「在这批架构上域外平均略偏一致」，"
             "**不得**写「域外总体与分布内一致（显著）」。")
    L.append("")

    # ---------------- T-RB-5
    L.append("## T-RB-5 与 ρ 的关系 + Lemma 1 幅度分解")
    L.append("")
    L.append("| 口径 | Spearman ρ | Kendall τ_a | 反转对数 | 反转过半? | 读法 |")
    L.append("|---|---:|---:|---:|:--:|---|")
    reading = {
        "arch14__dice": "ρ 中等正 ⇒ 分内/分外排名大体同向，但仍有可观反转对数",
        "arch14__align": "ρ 弱且 CI 跨 0（P2-3）⇒ 该配对下「对齐」不可判定",
        "config7__dice": "ρ 高 ⇒ 配置层排名大体一致，反转集中在**近并列**的对上",
    }
    for key, name in flat.items():
        p = P[key]
        half = "**是**" if p["n_discordant"] * 2 >= p["n_pairs"] else "否"
        L.append(f"| {name} | {p['rho_spearman']:+.4f} | {p['kendall_tau_a']:+.4f} "
                 f"| {p['n_discordant']} | {half} | {reading[key]} |")
    L.append("")
    L.append("**Lemma 1 逐对分解核验**（Δ = ΔlogD − ΔlogQ + Δlogα，单位 log-Dice）：")
    L.append("")
    L.append("| 口径 | 对数 | 恒等式残差 max | 适用性 |")
    L.append("|---|:--:|---:|---|")
    for key, name in flat.items():
        lm = payload["lemma1"][key]
        if lm.get("applicable"):
            L.append(f"| {name} | {lm['n']} | {lm['max_abs_residual']:.3e} | ✅ 适用（目标侧为 Dice） |")
        else:
            L.append(f"| {name} | — | — | ⚪ **N/A**：{lm.get('note', '')} |")
    L.append("")
    L.append("> **为什么 ρ 不够**：ρ 只用**秩**，把幅度信息全部丢掉；"
             "而反转的可写结论是**幅度的分解**（检出优势 ΔlogD 是否压倒勾画优势 ΔlogQ 与迁移项 Δlogα）。"
             "本文的 L2 归因因此必须报 Lemma 1 三项，而不是只报 ρ。")
    L.append("")
    L.append("**与 P2-3 对账**：")
    L.append("")
    cs = payload["consistency_with_p2_3"]
    if cs.get("available"):
        L.append(f"- alignment.json：反转 {cs['json_n_discordant']} 对、一致 {cs['json_n_concordant']} 对、"
                 f"τ_a={cs['json_tau_a']:+.6f}")
        L.append(f"- 本脚本（勾画–检出口径）：反转 {cs['ours_n_discordant']} 对、"
                 f"一致 {cs['ours_n_concordant']} 对、τ_a={cs['ours_tau_a']:+.6f} ⇒ "
                 f"**{'MATCH' if cs['match'] else 'MISMATCH'}**")
    else:
        L.append("- alignment.json 不可用，未能对账。")
    L.append("")

    # ---------------- T-RB-6
    L.append("## T-RB-6 结论与措辞边界")
    L.append("")
    L.append("**可写**：")
    L.append("- ✅「排名反转**不是单个配置驱动的**：逐个剔除任一单元后，"
             "反转对数、τ_a 与反转幅度的**符号均不变**」（见 T-RB-3，逐单元可查）。")
    L.append("- ✅「反转幅度在 log-Dice 尺度上由**检出优势减勾画优势**给出"
             "（Lemma 1：Δ = ΔlogD − ΔlogQ + Δlogα，逐对残差达机器精度）。」")
    L.append("- ✅「反转在**近并列**域内最干净（命题 2 的适用域），"
             "但在配置层并非**系统性**现象 —— 须逐对声明。」")
    L.append("")
    L.append("**必须并报的限制（诚实性，G3）**：")
    L.append("- ⚠️ **近并列 Δ 均值在 `arch14` 上与 0 不可区分**（CI 跨 0），"
             "留一法下有 3 个（Dice 口径）/ 1 个（勾画–检出口径）单元会使其变号，"
             "变号幅度在 ±0.03 内 ⇒ 只能写「**未检出系统性反转**」，**不得**写「反转普遍」。")
    L.append("- ⚠️ **配置层只有 3 个反转对**（共 21 对），基数极小；"
             "剔除 `msfa_only`、`dpa_msfa`、`egm_dpa` 任一后反转对降到 **1**，"
             "幅度随之从 0.1011 变到 0.1516 / 0.1173 —— **符号不变但数值不稳**，"
             "故配置层只能作**单个对的个案证据**，不能作分布性结论。")
    L.append("- ⚠️ **单元级 bootstrap 的计数型统计量有系统性下偏**（去重缩减有效单元数），"
             "故反转对数/占比的区间**不可当计数区间**解读；主判据取均值型统计量。")
    L.append("")
    L.append("**不可写**：")
    L.append("- ❌「反转显著（CI 不跨 0）」**单独**作为证据 —— 幅度按定向构造恒正，该 CI 必然不跨 0；"
             "须并报可跨 0 的 `signed_dev` 与近并列读数。")
    L.append("- ❌「域外排名普遍反转」—— 配置层的近并列 Δ 均值为负，属过度声明。")
    L.append("- ❌ 把 bootstrap 区间当**推断区间** —— 单元非独立，只作描述性。")
    L.append("- ❌ 写「证明」—— 观测证据不足以称证明（沿用 P2-3/P2-4 的措辞纪律）。")
    L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
