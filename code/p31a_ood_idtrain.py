# -*- coding: utf-8 -*-
"""
文件名: p31a_ood_idtrain.py
功能: 【论文二 P3-1 · 路线 A】用「分布内联合训练」权重零样本打到 ETIS，补齐 14 架构 OOD 读数

口径（路线 A，2026-09-13 用户拍板）:
    ID 联合训练权重  (experiments/baseline/<arch>  ｜  experiments/ablation/baseline)
      → ETIS (data_zeroshot/etis, 196 张, 第三域, 训练从未见过)
    14 架构 = 13 baseline + EGAUNet(=ablation/baseline, 完整模型)

    与既有 16 格（源域单独训练 → 另一域）**不同口径，不可混算**；
    本口径下 ID 侧读数与 OOD 侧读数**来自同一个模型**，域效应与训练过程差异解耦。

设计（单一真源，不重写协议）:
    - 模型工厂 / 权重载入  复用 p31_cross_eval.build_model / load_weights
    - 数据加载 复用 cross_dataset.dataset.CrossDataset
    - 指标     复用 cross_dataset.train_cross.calculate_metrics / calculate_hd95
    - 必须 `with autocast():` —— 既有产物与 ID 侧读数均在该设置下产生（P3-1 实测：漏掉会破坏逐位一致）
    - 架构清单 复用 decompose.ID_BASELINE_ARCHS（避免两处硬编码漂移）

输出列与 `03_results/raw/ood_per_sample/paper2_per_sample.csv` 对齐
    (dir / arch / name / dice / iou / hd95 / gt_area / pred_area / recall_gt / ...)
    其中 dir 固定为 'idtrain_to_etis'，recall_gt = |P n G| / |G|（与 ID 侧 Recall 同义）

双路复核（G4）:
    EGAUNet 的 ETIS Dice 应与论文一既有产物
    `D:/medical_segmentation/results_zeroshot/summary_statistics.json` 的
    `dice_mean = 64.2839` 一致（同权重 = ablation/baseline, 同数据 = ETIS 196 张）。
    本脚本会把它作为独立对照路打印。

已知限制:
    `calculate_hd95` 内 `np.random.choice` 无种子 → HD95 **逐样本不可复现**（P3-1 已证），
    只可用均值。本脚本照常输出 HD95，但复核只看均值。

用法:
    PY=D:/miniconda/aniconda/envs/medical_seg/python.exe
    # 冒烟（单架构，与论文一对照）
    $PY E:/paper2_ablation_reliability/02_code/analysis/p31a_ood_idtrain.py \
        --archs EGAUNet --out-root 03_results/raw/p31a_ood_idtrain
    # 全量 14 架构
    $PY E:/paper2_ablation_reliability/02_code/analysis/p31a_ood_idtrain.py \
        --out-root 03_results/raw/p31a_ood_idtrain \
        --out 03_results/stats/p31a_ood_idtrain.json
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
E_ROOT = r'E:/paper2_ablation_reliability'

# 复用统一模型工厂 + 权重载入（单一真源；import 副作用仅 chdir 到 D 盘工程根）
import p31_cross_eval as X

from cross_dataset.dataset import CrossDataset
from cross_dataset.train_cross import calculate_metrics, calculate_hd95
from torch.cuda.amp import autocast

# 架构清单复用 decompose 的冻结表（13 baseline），再补 EGAUNet
from decompose import ID_BASELINE_ARCHS

TARGET_DIR = 'data_zeroshot/etis'          # 第三域，196 张
DIRECTION_LABEL = 'idtrain_to_etis'
BASELINE_DIR = 'experiments/baseline'      # 13 个 baseline 的联合训练权重
# ⚠️ EGA-UNet 的联合训练权重**不**在 experiments/ 下（那是论文一主训练产线）：
#    真身 = outputs/checkpoints/ega_unet/best_model.pth  （296 键，实测 0 missing / 0 unexpected）
#    ⚠️ 勿用 experiments/ablation/baseline —— 那是 "Baseline U-Net"（31.04 M，Dice 81.97），
#       键结构与 models.ega_unet.EGAUNet 不符（250/116 不匹配），会静默得到全零预测。
EGAUNET_DIR = 'outputs/checkpoints/ega_unet'
EGAUNET_ID_CSV = 'results/test_results/sample_metrics.csv'   # EGA-UNet 分布内逐样本(242)

# 论文一既有对照值（独立第二路；EGA-UNet 在 ETIS 的 Dice）
PAPER1_REF = os.path.join(
    X.PROJECT_ROOT, 'results_zeroshot', 'summary_statistics.json'
)


def build_jobs(archs=None):
    """返回 [(arch, ckpt_dir), ...]。13 baseline + EGAUNet(outputs/checkpoints/ega_unet)。"""
    jobs = [(a, f'{BASELINE_DIR}/{a}') for a in ID_BASELINE_ARCHS]
    jobs.append(('EGAUNet', EGAUNET_DIR))
    if archs:
        want = set(archs)
        jobs = [j for j in jobs if j[0] in want]
    return jobs


def eval_arch(arch, ckpt_dir, device, ds, want_hd95=True):
    """对单个架构在 ETIS 上跑逐样本推理；返回 (summary, rows)。"""
    wpath = os.path.join(ckpt_dir, 'best_model.pth')
    if not os.path.exists(wpath):
        return None, None

    model = X.build_model(arch)
    fmt, miss, unexp = X.load_weights(model, wpath, device)
    model = model.to(device).eval()

    rows = []
    with torch.no_grad():
        for i in tqdm(range(len(ds)), desc=f'{arch}', ncols=88, leave=False):
            img, mask, name = ds[i]
            img_t = img.unsqueeze(0).to(device)
            mask_t = mask.unsqueeze(0).to(device)
            with autocast():                      # ⚠️ 必须：与既有产物同前向精度
                out = model(img_t)
                if isinstance(out, tuple):
                    out = out[0]
            out = out.float()

            d, iou = calculate_metrics(out, mask_t)
            hd = calculate_hd95(out, mask_t) if want_hd95 else float('nan')

            # 逐样本几何量（与 paper2_per_sample.csv 同义，recall_gt = |P∩G|/|G|）
            p = (torch.sigmoid(out) > 0.5).cpu().numpy().squeeze().astype(bool)
            g = (mask_t.cpu().numpy().squeeze() > 0.5)
            gs, ps = int(g.sum()), int(p.sum())
            rec = float(np.logical_and(g, p).sum() / gs) if gs else 1.0

            rows.append(dict(dir=DIRECTION_LABEL, arch=arch, name=name,
                             dice=float(d), iou=float(iou),
                             hd95=float(hd) if want_hd95 else float('nan'),
                             gt_area=gs, pred_area=ps, recall_gt=rec,
                             weight_format=fmt))

    del model
    torch.cuda.empty_cache()

    dice = np.array([r['dice'] for r in rows], dtype=float)
    iou = np.array([r['iou'] for r in rows], dtype=float)
    hd = np.array([r['hd95'] for r in rows], dtype=float)
    rec = np.array([r['recall_gt'] for r in rows], dtype=float)
    summary = dict(
        arch=arch, ckpt_dir=ckpt_dir, n=len(rows),
        weight_format=fmt, n_missing_keys=len(miss), n_unexpected_keys=len(unexp),
        dice_mean=float(dice.mean() * 100), dice_std=float(dice.std() * 100),
        iou_mean=float(iou.mean() * 100),
        hd95_mean=float(np.nanmean(hd)) if want_hd95 else None,
        recall_gt_mean=float(rec.mean()),
        n_undetected_frozen=int((rec <= 0.0).sum()),
        n_undetected_substantive=int((rec <= 1e-6).sum()),
    )
    return summary, rows


def paper1_reference():
    """论文一 EGAUNet 在 ETIS 的既有读数（第二路）。"""
    if not os.path.exists(PAPER1_REF):
        return None
    with open(PAPER1_REF, encoding='utf-8') as f:
        d = json.load(f)
    return dict(dice_mean=float(d['dice_mean']), n=int(d['n_samples']),
                in_distribution_dice=d.get('in_distribution_dice'),
                dice_retention=d.get('dice_retention'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--archs', nargs='+', default=None, help='子集，默认全部 14 个')
    ap.add_argument('--out-root', default='03_results/raw/p31a_ood_idtrain')
    ap.add_argument('--out', default=None, help='汇总 JSON（相对路径按 E 盘解析）')
    ap.add_argument('--no-hd95', action='store_true')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}  cuda={torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'gpu={torch.cuda.get_device_name(0)}')
    print(f'target={TARGET_DIR}  direction={DIRECTION_LABEL}')

    jobs = build_jobs(args.archs)
    print(f'jobs={len(jobs)}: ' + ', '.join(a for a, _ in jobs))
    assert len(jobs) == (len(args.archs) if args.archs else 14), (
        f'架构数应为 14（或指定子集），实为 {len(jobs)}')

    ds = CrossDataset(TARGET_DIR, augment=False, strong_augment=False)

    summaries, all_rows = [], []
    for arch, ckpt_dir in jobs:
        print(f'\n>>> {arch}   <- {ckpt_dir}/best_model.pth')
        s, rows = eval_arch(arch, ckpt_dir, device, ds, want_hd95=not args.no_hd95)
        if s is None:
            print('    [跳过] 权重缺失')
            summaries.append(dict(arch=arch, ckpt_dir=ckpt_dir, status='NO_WEIGHT'))
            continue
        summaries.append(s)
        all_rows.extend(rows)
        print(f"    n={s['n']}  Dice={s['dice_mean']:.4f}  "
              f"IoU={s['iou_mean']:.4f}  HD95={s['hd95_mean']:.2f}  "
              f"recall_gt={s['recall_gt_mean']:.4f}  "
              f"未检出(冻结)={s['n_undetected_frozen']}/{s['n']}  "
              f"keys(miss/unexp)={s['n_missing_keys']}/{s['n_unexpected_keys']}")

    # ---------------- 落盘 ----------------
    out_root = X.eabs(args.out_root)
    os.makedirs(out_root, exist_ok=True)
    df = pd.DataFrame(all_rows)
    csv_p = os.path.join(out_root, 'per_sample.csv')
    df.to_csv(csv_p, index=False)
    print(f'\n已写: {csv_p}  ({len(df)} 行)')

    ref = paper1_reference()
    cross = None
    if ref:
        eg = next((s for s in summaries if s.get('arch') == 'EGAUNet'), None)
        if eg:
            cross = dict(
                paper1_dice_mean=ref['dice_mean'], paper1_n=ref['n'],
                ours_dice_mean=eg['dice_mean'], ours_n=eg['n'],
                abs_diff=abs(ref['dice_mean'] - eg['dice_mean']),
                n_match=bool(ref['n'] == eg['n']),
            )
            print(f"\n[G4 第二路] 论文一 results_zeroshot Dice={ref['dice_mean']:.4f} "
                  f"(n={ref['n']})  vs  本次 {eg['dice_mean']:.4f} (n={eg['n']})  "
                  f"|Δ|={cross['abs_diff']:.4f}")

    payload = dict(
        segment='P3-1', route='A', mode='ood_etis_from_id_train',
        protocol=dict(
            target=TARGET_DIR, n_samples=len(ds),
            weights='ID 联合训练 (experiments/baseline/<arch> | experiments/ablation/baseline)',
            reuse=['p31_cross_eval.build_model', 'CrossDataset',
                   'calculate_metrics', 'calculate_hd95', 'autocast'],
            source_list='decompose.ID_BASELINE_ARCHS + EGAUNet',
        ),
        n_archs=len(summaries),
        n_samples_total=len(df),
        summaries=summaries,
        paper1_crosscheck=cross,
    )
    if args.out:
        outp = X.eabs(args.out)
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        with open(outp, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f'已写: {outp}')

    # 控制台速览
    print('\n' + '=' * 78)
    print(f"{'arch':<16}{'Dice':>9}{'IoU':>9}{'HD95':>9}{'recall_gt':>11}")
    print('-' * 78)
    for s in sorted(summaries, key=lambda z: -z.get('dice_mean', -1)):
        if s.get('status') == 'NO_WEIGHT':
            print(f"{s['arch']:<16}{'-- 权重缺失':>12}")
            continue
        print(f"{s['arch']:<16}{s['dice_mean']:>9.4f}{s['iou_mean']:>9.4f}"
              f"{s['hd95_mean']:>9.2f}{s['recall_gt_mean']:>11.4f}")


if __name__ == '__main__':
    main()
