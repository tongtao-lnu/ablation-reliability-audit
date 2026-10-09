# -*- coding: utf-8 -*-
"""
文件名: p3_1c_indist_pred.py
功能: 【论文二 P3-1c】为 MultiResUNet 与 PraNet 生成**分布内逐样本预测掩膜**（并算出 Recall 列）

背景 (v4.2.2):
    P3-1c 由 ⛔ 硬前置降为 ⭐ 增值项（原理由"不补 2 架构则 n 上不了 13"已被证伪）。
    真实定位：**首次为分布内生成逐样本预测掩膜**（`indist_pred/<架构>/`）。
    用处：① P3-6/P3-7 的 ID 侧预测管线 ② 独立复核 ε 与实现一致性。
    事实：这 2 个架构的既有 `sample_metrics.csv` 只有 `Dice,IoU,HD95,sample_idx`，**无 Recall**，
          且目录内**无预测掩膜**（见 00_docs/资产清单与路径映射.md）。

口径 (与既有 ID 产物逐字对齐):
    - 数据: D:/medical_segmentation/processed_data/test (242 张, 352x352)
    - 张量: img (H,W,3) f32 -> transpose(2,0,1); mask (H,W) -> 二值 > 0.5
    - 前向: `with autocast():`（既有 ID 产物均在此设置下产生）
    - 度量 (utils/metrics.py, eps=1e-8, threshold=0.5):
        Dice      = (2I + eps) / (P + G + eps)
        IoU       = (I + eps) / (P + G - I + eps)
        Precision = (I + eps) / (P + eps)
        Recall    = (I + eps) / (G + eps)      <- 本段新增列
      另记 no-eps 精确重合 recall_exact = I / G（供实质检出判据 B6 使用）

两条复现闸门 (G6, 跑前冻结):
    G-1 (主, 逐样本管线一致性): 新算逐样本 Dice 必须复现既有
        experiments/baseline/<arch>/sample_metrics.csv 的 Dice -> 期望 max|Δ| = 0
    G-2 (计划原文, 架构级一致性): 新算均值 vs results.json test_metrics.Dice -> |Δ| <= 1e-6
        ⚠️ 预检发现: 既有 sample_metrics.csv 与 results.json **本就存在 1e-6~4e-5 级差异**
        （13/13 架构皆有, 11/13 超 1e-6），故 G-2 对 PraNet 将报 CONFLICT。
        该差异**非本段引入**；本段如实并报，并输出 13 架构对照表作为证据，交用户裁定。

输出 (落盘 E):
    03_results/raw/indist_pred/<arch>/masks/<name>.npy      (bool 预测掩膜, 与输入同名)
    03_results/raw/indist_pred/<arch>/masks_png/<stem>.png  (0/255 预览, 供人眼查看)
    03_results/raw/indist_pred/<arch>/sample_metrics.csv    (242 行, 含 Recall)
    03_results/raw/indist_pred/<arch>/gates.json
    03_results/stats/indist_pred_summary.json

用法:
    PY=D:/miniconda/aniconda/envs/medical_seg/python.exe
    # 冒烟
    $PY .../p3_1c_indist_pred.py --limit 8 --out-root 03_results/raw/indist_pred_smoke
    # 全量
    $PY .../p3_1c_indist_pred.py --out-root 03_results/raw/indist_pred \
        --stats 03_results/stats/indist_pred_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# import 副作用: chdir 到 D:/medical_segmentation; E 盘一律走 X.eabs
import p31_cross_eval as X
from decompose import EPS_ID, FROZEN_THRESH, SUBSTANTIVE_THRESH
import p3_7_stress_test as S            # 单一真源: load_id_test / PIXEL_THRESH / ID_TEST_DIR

# --------------------------------------------------------------------------- #
# 冻结常数
# --------------------------------------------------------------------------- #
ARCHS = ['MultiResUNet', 'PraNet']          # 本段目标（既有 csv 无 Recall、目录无掩膜）
BASELINE_DIR = 'experiments/baseline'       # 相对 PROJECT_ROOT(D:)
CKPT_NAME = 'best_model.pth'
PIXEL_THRESH = 0.5

OUT_DEFAULT = '03_results/raw/indist_pred'
STATS_DEFAULT = '03_results/stats/indist_pred_summary.json'

CSV_COLS = ['sample_idx', 'name', 'Dice', 'IoU', 'Recall', 'Precision', 'HD95',
            'gt_area', 'pred_area', 'inter', 'recall_exact',
            'det_frozen', 'det_subst', 'weight_format']


# --------------------------------------------------------------------------- #
# 误差度量（逐字对齐 utils/metrics.py::SegmentationMetrics.update）
# --------------------------------------------------------------------------- #
def metrics_from_counts(inter: int, gs: int, ps: int) -> dict:
    """给定混淆计数，返回与 utils/metrics.py 完全一致的度量（eps=1e-8）。"""
    I, G, P = float(inter), float(gs), float(ps)
    dice = (2.0 * I + EPS_ID) / (P + G + EPS_ID)
    iou = (I + EPS_ID) / (P + G - I + EPS_ID)
    precision = (I + EPS_ID) / (P + EPS_ID)
    recall = (I + EPS_ID) / (G + EPS_ID)          # <- 本段新增（ID 侧度量口径）
    recall_exact = (I / G) if G > 0 else float('nan')   # no-eps, 供实质判据
    return dict(Dice=float(dice), IoU=float(iou), Precision=float(precision),
                Recall=float(recall), recall_exact=float(recall_exact),
                det_frozen=int(recall > FROZEN_THRESH),      # 恒 1（eps 构造）
                det_subst=int(inter >= 1))                   # 实质: I >= 1 px


# --------------------------------------------------------------------------- #
# 单架构推理
# --------------------------------------------------------------------------- #
def run_arch(arch, device, data, out_arch, save_masks=True):
    ckpt_dir = os.path.join(BASELINE_DIR, arch)
    wpath = os.path.join(ckpt_dir, CKPT_NAME)
    if not os.path.exists(wpath):
        raise FileNotFoundError(f'{arch}: 权重缺失 {wpath}')

    model = X.build_model(arch)
    fmt, miss, unexp = X.load_weights(model, wpath, device)
    if len(miss) or len(unexp):
        # 与 P3-1/P3-7 同铁律: 键不匹配会静默全零预测, 必须硬拦
        raise RuntimeError(f'{arch}: 权重键不匹配 (missing={len(miss)}, '
                           f'unexpected={len(unexp)}) -> 拒绝运行。 path={wpath} fmt={fmt}')
    model = model.to(device).eval()

    if save_masks:
        os.makedirs(os.path.join(out_arch, 'masks'), exist_ok=True)
        os.makedirs(os.path.join(out_arch, 'masks_png'), exist_ok=True)

    rows = []
    with torch.no_grad():
        for idx, (name, img, gt) in enumerate(tqdm(data, desc=arch, ncols=88, leave=False)):
            xt = torch.from_numpy(img.transpose(2, 0, 1).copy()).float().unsqueeze(0).to(device)
            with autocast():
                out = model(xt)
                if isinstance(out, tuple):
                    out = out[0]
            prob = torch.sigmoid(out.float())[0, 0].cpu().numpy()
            pred = (prob > PIXEL_THRESH)

            inter = int(np.logical_and(pred, gt).sum())
            gs = int(gt.sum())
            ps = int(pred.sum())
            m = metrics_from_counts(inter, gs, ps)

            if save_masks:
                np.save(os.path.join(out_arch, 'masks', name), pred.astype(bool))
                stem = os.path.splitext(name)[0]
                try:
                    import cv2
                    cv2.imwrite(os.path.join(out_arch, 'masks_png', f'{stem}.png'),
                                (pred.astype(np.uint8) * 255))
                except Exception:
                    pass

            rows.append(dict(sample_idx=idx, name=name, HD95=float('nan'),
                             gt_area=gs, pred_area=ps, inter=inter,
                             weight_format=fmt, **m))

    del model
    torch.cuda.empty_cache()
    df = pd.DataFrame(rows)[CSV_COLS]
    return df, fmt


# --------------------------------------------------------------------------- #
# 闸门
# --------------------------------------------------------------------------- #
def gate_1_sample_csv(arch, df, tol=1e-9):
    """G-1: 逐样本 Dice 复现既有 experiments/baseline/<arch>/sample_metrics.csv。

    ⚠️ 既有 CSV 的索引列名**不统一**: 多数用 `sample_idx`, 而 **PolypPVT 用 `sample_id`**
       （实测), 且 MultiResUNet/PraNet 无 `filename` 列。→ 先做列名归一化, 再比对;
       若归一化后仍缺索引列, 返回 NO_IDX_COL 而不是抛异常（避免整段因单个架构的
       列名差异而中断, 已完成的架构产物会白跑）。
    """
    ref_path = os.path.join(BASELINE_DIR, arch, 'sample_metrics.csv')
    if not os.path.exists(ref_path):
        return dict(status='NO_REF', ref_path=ref_path)
    ref = pd.read_csv(ref_path)
    if 'sample_idx' not in ref.columns and 'sample_id' in ref.columns:
        ref = ref.rename(columns={'sample_id': 'sample_idx'})
        idx_col = 'sample_id(renamed)'
    else:
        idx_col = 'sample_idx'
    if 'sample_idx' not in ref.columns or 'Dice' not in ref.columns:
        return dict(status='NO_IDX_COL', ref_path=ref_path,
                    cols=[str(c) for c in ref.columns])
    ref = ref.sort_values('sample_idx').reset_index(drop=True)
    mine = df.sort_values('sample_idx').reset_index(drop=True)
    if len(ref) != len(mine):
        return dict(status='LEN_MISMATCH', n_ref=len(ref), n_mine=len(mine),
                    idx_col=idx_col)
    d = np.abs(ref['Dice'].values - mine['Dice'].values)
    maxd = float(np.nanmax(d))
    return dict(status=('MATCH' if maxd <= tol else 'DIFF'),
                ref_path=ref_path, n=len(ref), idx_col=idx_col,
                max_abs_diff=maxd, n_gt_tol=int((d > tol).sum()),
                tol=tol)


def gate_2_results_json(arch, df, tol=1e-6):
    """G-2: 新算均值 vs results.json test_metrics.Dice。"""
    rj_path = os.path.join(BASELINE_DIR, arch, 'results.json')
    if not os.path.exists(rj_path):
        return dict(status='NO_REF', ref_path=rj_path)
    rj = json.load(open(rj_path, encoding='utf-8'))
    ref = float(rj.get('test_metrics', {}).get('Dice', float('nan')))
    mine = float(df['Dice'].mean())
    diff = abs(mine - ref)
    return dict(status=('PASS' if diff <= tol else 'CONFLICT'),
                ref_path=rj_path, ours=mine, ref=ref, abs_diff=diff, tol=tol)


def context_table_13():
    """13 架构对照: 既有 sample_metrics.csv 均值 vs results.json -> 证明 G-2 冲突是既有的。"""
    # ⚠️ BASELINE_DIR 是 **D 盘相对**路径（脚本已 chdir 到 PROJECT_ROOT），不可走 eabs
    base_abs = os.path.join(X.PROJECT_ROOT, BASELINE_DIR)
    out = []
    for a in sorted(os.listdir(base_abs)):
        d = os.path.join(base_abs, a)
        if not os.path.isdir(d):
            continue
        smp = os.path.join(d, 'sample_metrics.csv')
        rjp = os.path.join(d, 'results.json')
        if not (os.path.exists(smp) and os.path.exists(rjp)):
            continue
        sm = pd.read_csv(smp)
        rj = json.load(open(rjp, encoding='utf-8'))
        m_rj = float(rj.get('test_metrics', {}).get('Dice', float('nan')))
        m_sm = float(sm['Dice'].mean())
        out.append(dict(arch=a, n=len(sm),
                        mean_sample_metrics=m_sm, mean_results_json=m_rj,
                        abs_diff=abs(m_sm - m_rj),
                        has_recall=bool('Recall' in sm.columns),
                        gt_1e6=bool(abs(m_sm - m_rj) > 1e-6)))
    return out


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description='P3-1c: ID per-sample prediction masks (MultiResUNet, PraNet)')
    ap.add_argument('--archs', nargs='+', default=ARCHS)
    ap.add_argument('--limit', type=int, default=None, help='只用前 N 张（冒烟）')
    ap.add_argument('--out-root', default=OUT_DEFAULT)
    ap.add_argument('--stats', default=STATS_DEFAULT)
    ap.add_argument('--force', action='store_true', help='忽略 partial 缓存重跑')
    ap.add_argument('--no-masks', action='store_true', help='只算指标, 不写掩膜')
    args = ap.parse_args(argv)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}  cuda={torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'gpu={torch.cuda.get_device_name(0)}')

    data = S.load_id_test(args.limit)
    print(f'archs={len(args.archs)}  images={len(data)}')

    out_root = X.eabs(args.out_root)
    stats_path = X.eabs(args.stats)

    payload = dict(
        segment='P3-1c', mode='in_distribution_per_sample_prediction',
        generated_at=datetime.now().isoformat(timespec='seconds'),
        script='02_code/analysis/p3_1c_indist_pred.py',
        protocol=dict(
            data=S.ID_TEST_DIR, n_images=len(data), archs=list(args.archs),
            pixel_threshold=PIXEL_THRESH, eps_id=EPS_ID,
            ckpt=os.path.join(BASELINE_DIR, '<arch>', CKPT_NAME),
            metric_source='utils/metrics.py::SegmentationMetrics (eps=1e-8, thr=0.5)',
            forward='autocast() — 与既有 ID 产物一致',
            new_column='Recall = (I + eps) / (G + eps)',
            criterion_frozen=f'recall > {FROZEN_THRESH}',
            criterion_substantive=f'I >= 1 px (== recall_exact > {SUBSTANTIVE_THRESH})',
            reuse=['p31_cross_eval.build_model', 'p31_cross_eval.load_weights',
                   'p3_7_stress_test.load_id_test'],
            gates=dict(G1='逐样本 Dice 复现既有 sample_metrics.csv (tol 1e-9)',
                       G2='均值复现 results.json test Dice (tol 1e-6) — 预检已知既有产物不一致'),
        ),
        archs=[],
        context_13arch=context_table_13(),
    )

    for arch in args.archs:
        out_arch = os.path.join(out_root, arch)
        csv_path = os.path.join(out_arch, 'sample_metrics.csv')
        cached = os.path.exists(csv_path) and not args.force

        if cached:
            print(f'\n>>> {arch}  [缓存] {csv_path}')
            df = pd.read_csv(csv_path)
            fmt = str(df['weight_format'].iloc[0]) if 'weight_format' in df else 'cached'
        else:
            print(f'\n>>> {arch}   <- {BASELINE_DIR}/{arch}/{CKPT_NAME}')
            df, fmt = run_arch(arch, device, data, out_arch, save_masks=not args.no_masks)
            os.makedirs(out_arch, exist_ok=True)
            df.to_csv(csv_path, index=False)
            print(f'    已写: {csv_path}  ({len(df)} 行)')
            if not args.no_masks:
                print(f'    掩膜: {out_arch}/masks/*.npy  ({len(df)} 个)')

        g1 = gate_1_sample_csv(arch, df)
        g2 = gate_2_results_json(arch, df)
        rec = dict(arch=arch, ckpt_dir=os.path.join(BASELINE_DIR, arch),
                   weight_format=fmt, n=len(df),
                   out_masks=(None if args.no_masks else os.path.join(out_arch, 'masks')),
                   out_csv=csv_path,
                   mean_dice=float(df['Dice'].mean()),
                   mean_recall=float(df['Recall'].mean()),
                   mean_precision=float(df['Precision'].mean()),
                   gt_area_sum=int(df['gt_area'].sum()),
                   pred_area_sum=int(df['pred_area'].sum()),
                   inter_sum=int(df['inter'].sum()),
                   n_undetected_sub=int((df['det_subst'] == 0).sum()),
                   gate_1_sample_csv=g1, gate_2_results_json=g2)
        payload['archs'].append(rec)

        os.makedirs(out_arch, exist_ok=True)
        with open(os.path.join(out_arch, 'gates.json'), 'w', encoding='utf-8') as fh:
            json.dump(dict(arch=arch, gate_1_sample_csv=g1, gate_2_results_json=g2),
                      fh, ensure_ascii=False, indent=2)

        print(f"    mean Dice = {rec['mean_dice']:.12f}   Recall = {rec['mean_recall']:.12f}   "
              f"未检出 = {rec['n_undetected_sub']}/{rec['n']}")
        print(f"    [G-1] {g1.get('status')}  max|Δ| vs 既有 sample_metrics.csv = "
              f"{g1.get('max_abs_diff', float('nan')):.3e}")
        print(f"    [G-2] {g2.get('status')}  |Δ| vs results.json = {g2.get('abs_diff', float('nan')):.3e}")

    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f'\n已写: {stats_path}')

    # ---------------- 控制台速览 ----------------
    print('\n' + '=' * 96)
    print('13 架构对照：既有 sample_metrics.csv 均值 vs results.json（证明 G-2 冲突是既有的）')
    ct = payload['context_13arch']
    print(f"  {'arch':14s} {'mean(sm)':>14s} {'mean(rj)':>14s} {'|Δ|':>11s}  {'>1e-6':>6s}  hasRecall")
    for r in ct:
        print(f"  {r['arch']:14s} {r['mean_sample_metrics']:14.10f} {r['mean_results_json']:14.10f} "
              f"{r['abs_diff']:11.3e}  {str(r['gt_1e6']):>6s}  {r['has_recall']}")
    n_gt = sum(1 for r in ct if r['gt_1e6'])
    print(f"  -> {n_gt}/{len(ct)} 架构 |Δ| > 1e-6；最大 {max(r['abs_diff'] for r in ct):.3e}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
