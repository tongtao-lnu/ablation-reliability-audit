# -*- coding: utf-8 -*-
"""
文件名: p3_1bc_crosscheck.py
功能: 【G4 双路复核】P3-1b / P3-1c 的独立重算（**不 import 任何主模块**）

原则: 复核路径与主路径**不得共享代码**。本脚本只用 numpy/pandas 从**原始产物**重算:
    - P3-1c 用 **落盘的预测掩膜**（indist_pred/<arch>/masks/*.npy）+ **D 盘 GT** 重算混淆计数与度量，
      再与 indist_pred/<arch>/sample_metrics.csv 逐样本比对；
      并额外与 P3-7 的独立前向产物（03_results/raw/stress_test/per_sample.csv 的 clean 档）比对。
    - P3-1b 直接用 predictions.npy 内的 mask/pred 重算 inter/|G|/|P|，手算分解与残差恒等式，
      再与 decomp_ablation_etis.csv 比对。

输出: 03_results/audit/p3_1bc_crosscheck.json

用法:
    PY=D:/miniconda/aniconda/envs/medical_seg/python.exe
    $PY E:/paper2_ablation_reliability/02_code/analysis/p3_1bc_crosscheck.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

E = 'E:/paper2_ablation_reliability'
GT_DIR = 'D:/medical_segmentation/processed_data/test/masks'
ETIS_ROOT = 'D:/medical_segmentation/results_ablation_etis'
EPS_ID = 1e-8
EPS_OOD = 1e-6
TOL_F = 1e-12      # 浮点派生量容差（1 ulp ~ 1e-16，取 1e-12 留 4 个数量级余量）
TOL_ID = 1e-15     # 恒等式残差容差
ARCHS = ['MultiResUNet', 'PraNet']
CFGS = ['baseline', 'egm_only', 'dpa_only', 'msfa_only', 'egm_dpa', 'egm_msfa', 'dpa_msfa']
OUT = f'{E}/03_results/audit/p3_1bc_crosscheck.json'


# --------------------------------------------------------------------------- #
def crosscheck_p3_1c():
    """用落盘掩膜独立重算，与 CSV 及 P3-7 旁路产物比对。"""
    res = []
    p37 = pd.read_csv(f'{E}/03_results/raw/stress_test/per_sample.csv')
    for arch in ARCHS:
        d = f'{E}/03_results/raw/indist_pred/{arch}'
        csv = pd.read_csv(f'{d}/sample_metrics.csv')
        rows = []
        for r in csv.itertuples(index=False):
            pred = np.load(os.path.join(d, 'masks', r.name)).astype(bool)
            gt = (np.load(os.path.join(GT_DIR, r.name)).astype(np.float32) > 0.5)
            I = int(np.logical_and(pred, gt).sum()); G = int(gt.sum()); P = int(pred.sum())
            rows.append(dict(name=r.name, inter=I, gs=G, ps=P,
                             dice=(2.0 * I + EPS_ID) / (P + G + EPS_ID),
                             iou=(I + EPS_ID) / (P + G - I + EPS_ID),
                             recall=(I + EPS_ID) / (G + EPS_ID),
                             recall_exact=(I / G) if G else float('nan'),
                             det_subst=int(I >= 1)))
        chk = pd.DataFrame(rows)

        # (a) 与 CSV 逐样本比对
        a_dice = float(np.max(np.abs(chk['dice'].values - csv['Dice'].values)))
        a_rec = float(np.max(np.abs(chk['recall'].values - csv['Recall'].values)))
        a_int = int(np.max(np.abs(chk['inter'].values - csv['inter'].values)))
        a_gs = int(np.max(np.abs(chk['gs'].values - csv['gt_area'].values)))
        a_ps = int(np.max(np.abs(chk['ps'].values - csv['pred_area'].values)))
        a_det = int((chk['det_subst'].values != csv['det_subst'].values).sum())

        # (b) 与 P3-7 独立前向（clean 档）比对
        c = (p37[(p37.arch == arch) & (p37.degradation == 'clean')]
             .set_index('name').sort_index())
        m = chk.set_index('name').sort_index()
        common = c.index.intersection(m.index)
        b_int = int(np.max(np.abs(c.loc[common, 'inter'].values - m.loc[common, 'inter'].values)))
        b_dice = float(np.max(np.abs(c.loc[common, 'dice'].values - m.loc[common, 'dice'].values)))
        b_rec = float(np.max(np.abs(c.loc[common, 'recall_metric'].values
                                   - m.loc[common, 'recall'].values)))

        res.append(dict(arch=arch, n=len(chk),
                        vs_csv=dict(max_abs_dice=a_dice, max_abs_recall=a_rec,
                                    max_abs_inter=a_int, max_abs_gt_area=a_gs,
                                    max_abs_pred_area=a_ps, n_det_subst_mismatch=a_det,
                                    # 整数计数: 严格 0；浮点派生量: 容差 1e-12
                                    match=bool(a_int == 0 and a_gs == 0 and a_ps == 0
                                               and a_det == 0
                                               and a_dice <= TOL_F and a_rec <= TOL_F)),
                        vs_p37_clean=dict(n_common=len(common), max_abs_inter=b_int,
                                          max_abs_dice=b_dice, max_abs_recall=b_rec,
                                          match=bool(b_int == 0
                                                     and b_dice <= TOL_F
                                                     and b_rec <= TOL_F))))
    return res


# --------------------------------------------------------------------------- #
def crosscheck_p3_1b():
    """独立重算 7 配置分解；手算残差恒等式；与 CSV 比对。"""
    ref = pd.read_csv(f'{E}/03_results/stats/decomp_ablation_etis.csv').set_index('config')
    res = []
    for cfg in CFGS:
        arr = np.load(os.path.join(ETIS_ROOT, cfg, 'predictions.npy'), allow_pickle=True)
        dice = np.array([float(x['dice']) for x in arr], dtype=float)
        inter = np.array([int(np.logical_and(np.asarray(x['pred']).astype(bool),
                                             np.asarray(x['mask']) > 0.5).sum()) for x in arr],
                         dtype=float)
        gs = np.array([int((np.asarray(x['mask']) > 0.5).sum()) for x in arr], dtype=float)
        recall = inter / gs

        for thr, tag in ((0.0, 'frozen'), (1e-6, 'sub')):
            det_m = recall > thr
            n_und = int((~det_m).sum())
            detection = float(det_m.mean())
            delin = float(dice[det_m].mean()) if det_m.any() else float('nan')
            product = detection * delin
            mean_dice = float(dice.mean())
            residual = mean_dice - product
            predicted = float(dice[~det_m].sum() / len(dice))
            r = ref.loc[cfg]
            col_d = 'detection' if tag == 'frozen' else 'detection_sub'
            col_e = 'delineation' if tag == 'frozen' else 'delineation_sub'
            col_n = 'n_undetected' if tag == 'frozen' else 'n_undetected_sub'
            res.append(dict(
                config=cfg, criterion=tag,
                max_abs_detection=abs(detection - float(r[col_d])),
                max_abs_delineation=abs(delin - float(r[col_e])) if delin == delin else 0.0,
                n_undetected_mismatch=(n_und != int(r[col_n])),
                identity_gap=abs(residual - predicted),
                residual=residual, predicted_residual=predicted,
                match=bool(abs(detection - float(r[col_d])) <= TOL_F
                           and n_und == int(r[col_n])
                           and abs(residual - predicted) <= TOL_ID)))
    return res


# --------------------------------------------------------------------------- #
def main():
    out = dict(segment='G4-crosscheck', scope=['P3-1b', 'P3-1c'],
               note='独立重算，不 import decompose / p3_1b / p3_1c / p3_7 主模块',
               p3_1c=crosscheck_p3_1c(), p3_1b=crosscheck_p3_1b())
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    print('=' * 100)
    print('G4 双路复核')
    print('=' * 100)
    print('\n[P3-1c] 用落盘掩膜独立重算')
    for r in out['p3_1c']:
        v = r['vs_csv']; w = r['vs_p37_clean']
        print(f"  {r['arch']:14s} n={r['n']}  vs CSV: dice Δ={v['max_abs_dice']:.2e} "
              f"recall Δ={v['max_abs_recall']:.2e} inter Δ={v['max_abs_inter']} "
              f"gt Δ={v['max_abs_gt_area']} pred Δ={v['max_abs_pred_area']} "
              f"det 不一致={v['n_det_subst_mismatch']}  -> {'MATCH' if v['match'] else 'DIFF'}")
        print(f"  {'':14s}        vs P3-7 clean(独立前向, n={w['n_common']}): "
              f"inter Δ={w['max_abs_inter']} dice Δ={w['max_abs_dice']:.2e} "
              f"recall Δ={w['max_abs_recall']:.2e}  -> {'MATCH' if w['match'] else 'DIFF'}")
    print('\n[P3-1b] 独立重算分解与残差恒等式')
    for r in out['p3_1b']:
        print(f"  {r['config']:10s} {r['criterion']:7s} det Δ={r['max_abs_detection']:.2e} "
              f"delin Δ={r['max_abs_delineation']:.2e} "
              f"残差恒等式 gap={r['identity_gap']:.2e}  -> {'MATCH' if r['match'] else 'DIFF'}")

    ok_c = all(r['vs_csv']['match'] and r['vs_p37_clean']['match'] for r in out['p3_1c'])
    ok_b = all(r['match'] for r in out['p3_1b'])
    print(f"\n  总体: P3-1c {'MATCH' if ok_c else 'DIFF'} | P3-1b {'MATCH' if ok_b else 'DIFF'}")
    print(f"  已写: {OUT}")
    return 0 if (ok_c and ok_b) else 1


if __name__ == '__main__':
    raise SystemExit(main())
