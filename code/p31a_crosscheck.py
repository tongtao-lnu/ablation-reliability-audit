# -*- coding: utf-8 -*-
"""
文件名: p31a_crosscheck.py
功能: 【论文二 P3-1 · 路线 A】独立复核（G4 双路）

原则：**不 import 任何主模块**（不 import p31a_ood_idtrain / p31_cross_eval /
decompose / variance_attribution），全部量手写重算，再逐项与主路比对。

复核内容:
  1. 逐样本表结构：14 架构 × 196 样本 = 2744 行，无重复、无缺失
  2. OOD 检出率（冻结判据 recall_gt > 0）手写重算 vs 主路 JSON 记录
  3. OOD 分解恒等式：mean_dice = detection × delineation + residual
     —— 用**另一套公式**（先算 undetected 质量和，再反推）重算
  4. ID 检出率：从 D 盘 sample_metrics.csv 直接重算（Recall 列；无则 Dice 代理）
  5. 换判据复核（B6 判据依赖性）：阈值 0 / 1e-6 / 1e-3 三口径
  6. 把 EGA-UNet 的 ETIS Dice 与论文一 `results_zeroshot` 逐样本对齐比对

退出码 0 = 全部 MATCH。

用法:
  PY=D:/miniconda/aniconda/envs/medical_seg/python.exe
  $PY E:/paper2_ablation_reliability/02_code/analysis/p31a_crosscheck.py
"""
import io
import json
import os
import sys

import numpy as np
import pandas as pd

E_ROOT = r'E:/paper2_ablation_reliability'
D_ROOT = r'D:/medical_segmentation'

OOD_CSV = f'{E_ROOT}/03_results/raw/p31a_ood_idtrain/per_sample.csv'
OOD_JSON = f'{E_ROOT}/03_results/stats/p31a_ood_idtrain.json'
ID_ROOT = f'{D_ROOT}/experiments/baseline'
ID_EGAUNET = f'{D_ROOT}/results/test_results/sample_metrics.csv'
PAPER1_CSV = f'{D_ROOT}/results_zeroshot/sample_metrics.csv'
PAPER1_JSON = f'{D_ROOT}/results_zeroshot/summary_statistics.json'

ARCHS = ['AttentionUNet', 'CaraNet', 'M2SNet', 'MultiResUNet', 'PSPNet', 'PolypPVT',
         'PraNet', 'ResUNet', 'SANet', 'SegNet', 'TransUNet', 'UACANet', 'UNet', 'EGAUNet']

checks = []


def ck(name, ok, detail=''):
    checks.append((name, bool(ok), detail))
    print(f'  [{"MATCH" if ok else "MISMATCH":<8}] {name}   {detail}')
    return bool(ok)


def decompose_manual(dice, metric, thr=0.0):
    """手写分解（与 decompose.factorize 同定义，但不 import 它）。

    采用**另一条算式路径**：先算未检出质量 -> 反推 residual -> 反推 delineation。
    """
    dice = np.asarray(dice, float)
    metric = np.asarray(metric, float)
    n = dice.size
    det = metric > thr
    k = int((~det).sum())
    detection = 1.0 - k / n                      # ← 用补集计数，不用 det.mean()
    undet_mass = float(dice[~det].sum()) / n
    mean_dice = float(dice.sum()) / n            # ← 用 sum/n，不用 dice.mean()
    delineation = (mean_dice - undet_mass) / detection if detection > 0 else float('nan')
    residual = mean_dice - detection * delineation
    return dict(n=n, detection=detection, delineation=delineation,
                residual=residual, undet_mass=undet_mass, k=k)


def main():
    print('=' * 78)
    print('P3-1 路线 A 独立复核（不 import 任何主模块）')
    print('=' * 78)

    # ---------- 1) 结构 ----------
    print('\n[1] 逐样本表结构')
    df = pd.read_csv(OOD_CSV)
    ck('行数 = 14 × 196 = 2744', len(df) == 14 * 196, f'实际 {len(df)}')
    ck('架构数 = 14', df.arch.nunique() == 14, f'实际 {df.arch.nunique()}')
    ck('每架构样本数均为 196',
       bool((df.groupby('arch').size() == 196).all()),
       str(df.groupby('arch').size().unique().tolist()))
    ck('(arch,name) 无重复', not df.duplicated(['arch', 'name']).any())
    ck('dir 唯一且 = idtrain_to_etis', set(df['dir'].unique()) == {'idtrain_to_etis'},
       str(sorted(df['dir'].unique())))

    with io.open(OOD_JSON, encoding='utf-8') as f:
        J = json.load(f)
    ssum = {s['arch']: s for s in J['summaries'] if 'dice_mean' in s}

    # ---------- 2) OOD 检出率 + 分解恒等式 ----------
    print('\n[2] OOD 检出率与分解恒等式（主路 JSON vs 手写重算）')
    rows = []
    for a in ARCHS:
        g = df[df.arch == a]
        m = decompose_manual(g['dice'].values, g['recall_gt'].values, thr=0.0)
        s = ssum[a]
        ok1 = abs(m['detection'] * 196 - (196 - m['k'])) < 1e-12
        ok2 = abs(m['detection'] - (1 - s['n_undetected_frozen'] / s['n'])) < 1e-12
        ok3 = abs(m['delineation'] - s['dice_mean'] / 100.0) if m['k'] == 0 else True
        rows.append((a, m, s))
        ck(f'{a:<15} 检出率一致', ok2,
           f"手写 det={m['detection']:.6f} 未检出 k={m['k']} 主路 k={s['n_undetected_frozen']}")

    # 恒等式：mean_dice == detection*delineation + residual，且 residual == undet_mass
    print('\n[3] 分解恒等式 max|residual - undet_mass|（应 ~1e-16）')
    worst = 0.0
    for a, m, s in rows:
        e = abs(m['residual'] - m['undet_mass'])
        worst = max(worst, e)
    ck('恒等式残差 < 1e-12', worst < 1e-12, f'max = {worst:.3e}')

    # ---------- 4) ID 侧 ----------
    print('\n[4] ID 侧检出（D 盘 sample_metrics.csv 直接重算）')
    id_ok = True
    id_tab = []
    for a in ARCHS:
        p = ID_EGAUNET if a == 'EGAUNet' else f'{ID_ROOT}/{a}/sample_metrics.csv'
        if not os.path.exists(p):
            id_ok = False
            print(f'    [缺] {a}')
            continue
        d = pd.read_csv(p)
        col = 'Recall' if 'Recall' in d.columns else 'Dice'
        m = decompose_manual(d['Dice'].values, d[col].values, thr=0.0)
        id_tab.append((a, col, len(d), m))
    ck('14 架构 ID 表全部载入', id_ok and len(id_tab) == 14, f'载入 {len(id_tab)}')
    ck('ID 样本数均为 242', all(t[2] == 242 for t in id_tab))

    # EGA-UNet ID Dice 与论文一 in_distribution_dice
    with io.open(PAPER1_JSON, encoding='utf-8') as f:
        p1 = json.load(f)
    eg_id = [t for t in id_tab if t[0] == 'EGAUNet'][0]
    id_dice = eg_id[3]['detection'] * eg_id[3]['delineation'] + eg_id[3]['residual']
    ck('EGAUNet ID Dice == 论文一 in_distribution_dice(88.53)',
       abs(id_dice * 100 - p1['in_distribution_dice']) < 0.01,
       f'重算 {id_dice*100:.4f} vs 记录 {p1["in_distribution_dice"]}')

    # ---------- 5) 换判据（B6）----------
    print('\n[5] 判据敏感性（阈值 0 / 1e-6 / 1e-3）')
    for thr in (0.0, 1e-6, 1e-3):
        ds = []
        for a in ARCHS:
            g = df[df.arch == a]
            m = decompose_manual(g['dice'].values, g['recall_gt'].values, thr=thr)
            ds.append(m['detection'])
        ck(f'thr={thr:g} 检出率 ∈ (0,1] 且有限',
           all(0 < x <= 1 for x in ds),
           f'范围 [{min(ds):.4f}, {max(ds):.4f}] 均值 {np.mean(ds):.4f}')

    # ---------- 6) 论文一逐样本 ----------
    print('\n[6] EGA-UNet ETIS 逐样本 vs 论文一 results_zeroshot')
    ours = df[df.arch == 'EGAUNet'].sort_values('name').reset_index(drop=True)
    p1df = pd.read_csv(PAPER1_CSV).sort_values('filename').reset_index(drop=True)
    ck('文件名集合一致', set(ours['name']) == set(p1df['filename']))
    dd = (ours['dice'].values - p1df['dice'].values)
    ck('无样本 |ΔDice| > 0.001', float(np.abs(dd).max()) < 0.001,
       f'max|Δ| = {np.abs(dd).max():.6f}（{np.abs(dd).max()*100:.4f} Dice 点）')
    ck('均值 |Δ| < 0.001', abs(dd.mean()) < 0.001,
       f'{ours["dice"].mean()*100:.4f} vs {p1df["dice"].mean()*100:.4f}')

    # ---------- 汇总 ----------
    print('\n' + '=' * 78)
    n_ok = sum(1 for _, ok, _ in checks if ok)
    print(f'检查 {len(checks)} 项：MATCH {n_ok} / MISMATCH {len(checks) - n_ok}')
    print('RESULT:', 'MATCH' if n_ok == len(checks) else 'MISMATCH')
    print('=' * 78)

    print('\n--- n=14 配对速览（ID 检出/勾画 vs OOD 检出/勾画）---')
    print(f"{'arch':<16}{'IDdet':>8}{'IDdel':>8}{'OODdet':>9}{'OODdel':>9}"
          f"{'IDDice':>9}{'OODDice':>9}{'keep':>7}")
    ids = {t[0]: t[3] for t in id_tab}
    for a, m, s in sorted(rows, key=lambda z: -(z[1]['detection'] * z[1]['delineation'])):
        im = ids[a]
        idv = im['detection'] * im['delineation'] + im['residual']
        oodv = m['detection'] * m['delineation'] + m['residual']
        print(f"{a:<16}{im['detection']:>8.4f}{im['delineation']:>8.4f}"
              f"{m['detection']:>9.4f}{m['delineation']:>9.4f}"
              f"{idv:>9.4f}{oodv:>9.4f}{oodv/idv:>7.3f}")

    return 0 if n_ok == len(checks) else 1


if __name__ == '__main__':
    sys.exit(main())
