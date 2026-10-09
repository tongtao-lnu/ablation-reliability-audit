"""
文件名: p31_cross_eval.py
功能: 【论文二 P3-1】跨域零样本推理 · 统一评估器

设计原则（不改协议）:
    - 数据加载复用 cross_dataset.dataset.CrossDataset（不做任何增强/归一化差异）
    - 指标计算复用 cross_dataset.train_cross.calculate_metrics / calculate_hd95
    - 模型构造复用 train_single.py 的 BASELINE_MODELS 参数表 + cross get_model 的 EGAUNet
    → 与 results_cross/ 里既有 predictions.npy / detailed_metrics.npz 完全同一协议

两种模式:
    --gate   复现闸门：用「既有跨域权重」重跑已跑过的格子，与既有产物逐位比对
             （不改协议，用来证明本脚本与产出既有结果的脚本等价）
    --run    正式推理：跑新格子并落到 --out-root

用法:
    PY=D:/miniconda/aniconda/envs/medical_seg/python.exe
    $PY p31_cross_eval.py --gate --dirs kvasir_to_etis cvc_to_etis \
        --archs UNet AttentionUNet TransUNet --out 03_results/raw/p31_gate.json
    $PY p31_cross_eval.py --run --dirs ... --archs ... --out-root 03_results/raw/cross_all
"""
import os
import sys
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = r'D:/medical_segmentation'
E_ROOT = r'E:/paper2_ablation_reliability'
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)


def eabs(p):
    """把相对路径解析到 E 盘工作区（脚本已 chdir 到 D 盘工程根，相对路径会跑偏）"""
    if p is None:
        return None
    return p if os.path.isabs(p) else os.path.join(E_ROOT, p)

from cross_dataset.dataset import CrossDataset
from cross_dataset.train_cross import calculate_metrics, calculate_hd95
from torch.cuda.amp import autocast  # ⚠️ 必用：既有产物均在 autocast() 下前向

# ----------------------------------------------------------------------------
# 1) 目标域 -> 数据目录（与既有产物完全一致）
# ----------------------------------------------------------------------------
TARGET_DIR = {
    'etis':   'data_zeroshot/etis',
    'cvc':    'data_cross/cvc_full',
    'kvasir': 'data_cross/kvasir_full',
}

# 方向 -> (源域, 目标域)
DIRECTION = {
    'kvasir_to_cvc':    ('kvasir', 'cvc'),
    'cvc_to_kvasir':    ('cvc',    'kvasir'),
    'kvasir_to_etis':   ('kvasir', 'etis'),
    'cvc_to_etis':      ('cvc',    'etis'),
}

# 源域 -> 存放「在该源域上训练之权重」的既有跨域目录（与 zeroshot_baselines_etis.py 一致）
SOURCE_WEIGHT_DIR = {
    'kvasir': 'results_cross/kvasir_to_cvc',
    'cvc':    'results_cross/cvc_to_kvasir',
}

# ----------------------------------------------------------------------------
# 2) 模型工厂（参数表逐字抄自 train_single.py::BASELINE_MODELS + train_cross.py::get_model）
# ----------------------------------------------------------------------------
def build_model(arch):
    if arch == 'EGAUNet':
        from models.ega_unet import EGAUNet
        return EGAUNet(in_channels=3, num_classes=1, base_filters=64)
    if arch == 'UNet':
        from models.unet import UNet
        return UNet(in_channels=3, num_classes=1, base_filters=64)
    if arch == 'AttentionUNet':
        from models.baselines.attention_unet import AttentionUNet
        return AttentionUNet(in_channels=3, num_classes=1, base_filters=64)
    if arch == 'TransUNet':
        from models.baselines.transunet import TransUNet
        return TransUNet(in_channels=3, num_classes=1, base_filters=64,
                         embed_dim=256, num_heads=8, num_layers=4)
    if arch == 'SANet':
        from models.baselines.sanet import SANet
        return SANet(in_channels=3, num_classes=1, base_filters=64)
    if arch == 'SegNet':
        from models.baselines.segnet import SegNet
        return SegNet(in_channels=3, num_classes=1, base_filters=64)
    if arch == 'ResUNet':
        from models.baselines.resunet import ResUNet
        return ResUNet(in_channels=3, num_classes=1, base_filters=64)
    if arch == 'MultiResUNet':
        from models.baselines.multiresunet import MultiResUNet
        return MultiResUNet(in_channels=3, num_classes=1, base_filters=32)
    if arch == 'PSPNet':
        from models.baselines.pspnet import PSPNet
        return PSPNet(in_channels=3, num_classes=1, base_filters=64)
    if arch == 'PolypPVT':
        from models.baselines.polyp_pvt import PolypPVT
        return PolypPVT(in_channels=3, num_classes=1, embed_dim=64)
    if arch == 'PraNet':
        from models.baselines.pranet import PraNet
        return PraNet(in_channels=3, num_classes=1, channel=32)
    if arch == 'M2SNet':
        from models.baselines.m2snet import M2SNet
        return M2SNet(in_channels=3, num_classes=1, base_filters=64)
    if arch == 'CaraNet':
        from models.baselines.caranet import CaraNet
        return CaraNet(in_channels=3, num_classes=1, base_filters=64)
    if arch == 'UACANet':
        from models.baselines.uacanet import UACANet
        return UACANet(in_channels=3, num_classes=1, base_filters=64)
    raise ValueError(f'未注册架构: {arch}')


BASELINE_DIR = 'experiments/baseline'


def load_weights(model, path, device):
    """兼容两种权重格式：raw state_dict（跨域训练）与 checkpoint dict（train_single）"""
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and 'model_state_dict' in obj:
        sd = obj['model_state_dict']
        fmt = 'checkpoint_dict'
    else:
        sd = obj
        fmt = 'raw_state_dict'
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return fmt, list(missing), list(unexpected)


# ----------------------------------------------------------------------------
# 3) 评估
# ----------------------------------------------------------------------------
def evaluate_cell(arch, direction, device, want_hd95=True, save_preds=False):
    src, tgt = DIRECTION[direction]
    wpath = os.path.join(SOURCE_WEIGHT_DIR[src], arch, 'best_model.pth')
    if not os.path.exists(wpath):
        return None

    ds = CrossDataset(TARGET_DIR[tgt], augment=False, strong_augment=False)

    model = build_model(arch)
    fmt, miss, unexp = load_weights(model, wpath, device)
    model = model.to(device).eval()

    dices, ious, hd95s, names, preds = [], [], [], [], []
    with torch.no_grad():
        for i in tqdm(range(len(ds)), desc=f'{direction}/{arch}', ncols=88, leave=False):
            img, mask, name = ds[i]
            img_t = img.unsqueeze(0).to(device)
            mask_t = mask.unsqueeze(0).to(device)
            with autocast():
                out = model(img_t)
                if isinstance(out, tuple):
                    out = out[0]
            d, iou = calculate_metrics(out.float(), mask_t)
            hd = calculate_hd95(out.float(), mask_t) if want_hd95 else float('nan')
            dices.append(d); ious.append(iou); hd95s.append(hd); names.append(name)
            if save_preds:
                pred = (torch.sigmoid(out.float()) > 0.5).cpu().numpy().squeeze()
                preds.append({'name': name,
                              'image': img.numpy().transpose(1, 2, 0),
                              'mask': mask.numpy().squeeze(),
                              'pred': pred, 'dice': d, 'iou': iou, 'hd95': hd})

    del model
    torch.cuda.empty_cache()
    return {'arch': arch, 'direction': direction, 'source': src, 'target': tgt,
            'n': len(dices), 'weight_format': fmt,
            'missing_keys': len(miss), 'unexpected_keys': len(unexp),
            'dice': float(np.mean(dices) * 100), 'dice_std': float(np.std(dices) * 100),
            'iou': float(np.mean(ious) * 100),
            'hd95': float(np.mean(hd95s)) if want_hd95 else None,
            'dice_list': [d * 100 for d in dices],
            'iou_list': [i * 100 for i in ious],
            'hd95_list': hd95s, 'names': names, '_preds': preds}


# ----------------------------------------------------------------------------
# 4) 复现闸门：与既有产物比对
# ----------------------------------------------------------------------------
def gate_compare(cell):
    existing = os.path.join('results_cross', cell['direction'], cell['arch'])
    npz_p = os.path.join(existing, 'detailed_metrics.npz')
    rj_p = os.path.join(existing, 'results.json')
    if not os.path.exists(npz_p):
        return {'status': 'NO_EXISTING_ARTIFACT'}

    z = np.load(npz_p)
    old_dice = np.asarray(z['dice'], dtype=np.float64)
    new_dice = np.asarray(cell['dice_list'], dtype=np.float64)
    out = {'status': 'COMPARED', 'n_old': int(old_dice.size), 'n_new': int(new_dice.size),
           'mean_old': float(old_dice.mean()), 'mean_new': float(new_dice.mean())}

    if old_dice.size == new_dice.size:
        d = np.abs(old_dice - new_dice)
        out['dice_max_absdiff'] = float(d.max())
        out['dice_bitwise_equal'] = bool(np.array_equal(old_dice, new_dice))
        out['dice_n_exact'] = int((d == 0).sum())
    else:
        out['mean_absdiff'] = abs(out['mean_old'] - out['mean_new'])

    # IoU / HD95
    for key, npzkey in (('iou', 'iou'), ('hd95', 'hd95')):
        if npzkey in z:
            oldv = np.asarray(z[npzkey], dtype=np.float64)
            newv = np.asarray(cell[f'{key}_list'], dtype=np.float64)
            if oldv.size == newv.size:
                dd = np.abs(oldv - newv)
                out[f'{key}_max_absdiff'] = float(dd.max())
                out[f'{key}_bitwise_equal'] = bool(np.array_equal(oldv, newv))
                out[f'{key}_n_exact'] = int((dd == 0).sum())

    if os.path.exists(rj_p):
        with open(rj_p) as f:
            rj = json.load(f)
        out['json_dice'] = rj.get('dice')
        out['json_dice_absdiff'] = (abs(rj['dice'] - cell['dice']) if 'dice' in rj else None)
        out['json_hd95'] = rj.get('hd95')
    return out


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dirs', nargs='+', default=['kvasir_to_etis', 'cvc_to_etis'])
    ap.add_argument('--archs', nargs='+', default=['UNet', 'AttentionUNet', 'TransUNet'])
    ap.add_argument('--mode', choices=['gate', 'run'], default='gate')
    ap.add_argument('--out', default=None)
    ap.add_argument('--out-root', default=None)
    ap.add_argument('--no-hd95', action='store_true')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}  cuda={torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'gpu={torch.cuda.get_device_name(0)}')

    results = []
    for d in args.dirs:
        for a in args.archs:
            print(f'\n>>> {d} / {a}')
            cell = evaluate_cell(a, d, device, want_hd95=not args.no_hd95)
            if cell is None:
                print('    [跳过] 权重缺失')
                results.append({'arch': a, 'direction': d, 'status': 'NO_WEIGHT'})
                continue
            if args.mode == 'gate':
                g = gate_compare(cell)
                cell_out = {k: v for k, v in cell.items() if not k.startswith('_')}
                cell_out['gate'] = g
                print(f"    new dice={cell['dice']:.6f}  | gate={g}")
                results.append(cell_out)
            else:
                out_root = args.out_root or '03_results/raw/cross_all'
                os.makedirs(out_root, exist_ok=True)

    if args.out:
        outp = eabs(args.out)
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        with open(outp, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'\n已写: {outp}')


if __name__ == '__main__':
    main()
