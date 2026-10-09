# -*- coding: utf-8 -*-
"""
p3_4_train_capacity.py -- P3-4：训练**容量匹配的 plain UNet**（单个，按 --bf 指定）

用途
----
为 P2-4 三件套之三（`T-CF-3c` 分支 A/B 判据）训练容量匹配对照组：
把 plain U-Net 加宽到 ≈ 某个模块配置的参数量，看该模块配置的域外检出优势
**是否消失** —— 这是分支 A/B 的分水岭。

匹配表（T-CF-3a，已实测复现，偏差 ±0.001%）
------------------------------------------
| 模块配置 | 参数量(M) | 匹配 plain UNet | base_filters | 实参数量(M) |
|---|---:|---|:--:|---:|
| baseline  | 31.038 | plainUNet_bf64 | 64 | 31.038 |（= 既有 ablation/baseline，可复用，不必重训）
| dpa_only  | 32.514 | plainUNet_bf66 | 66 | 33.007 |
| egm_only  | 38.022 | plainUNet_bf71 | 71 | 38.197 |
| egm_dpa   | 39.498 | plainUNet_bf72 | 72 | 39.280 |
| msfa_only | 39.955 | plainUNet_bf73 | 73 | 40.378 |
| dpa_msfa  | 41.431 | plainUNet_bf74 | 74 | 41.492 |
| egm_msfa  | 46.940 | plainUNet_bf79 | 79 | 47.287 |

**骨架不变、只变宽度**：用 `AblationUNet(use_egm/dpa/msfa 全 False, base_filters=bf)`，
而非 `models/unet.py::UNet` —— 后者换成另一个实现（换头/换结构）会**引入架构混淆**，
不是纯粹的容量对照。

D 盘零改写（铁律）
-----------------
* `models/ablation_models.py` **源码不改**。本脚本运行时：
  1) 往 `ABLATION_CONFIGS` **注入**新配置（含额外键 `base_filters`）；
  2) 给 **`train_single.get_ablation_model`** 打补丁，使其从配置里取 `base_filters`。
     ⚠️ 必须补丁 `train_single` 的模块级名字：`train_single.py` L18 以
     `from models.ablation_models import get_ablation_model` 绑定，改 `ablation_models`
     里的同名函数**不会**影响已绑定的引用。
* 训练产物落**新目录** `experiments/capacity_matched/ablation/<name>/`（仅新增）。

协议冻结（G6：与消融 7 格逐项一致）
----------------------------------
    --type ablation --epochs 100 --batch_size 4 --lr 1e-4 --seed 42
    val_interval 5 / early_stop 30 / AdamW(wd 1e-5) / CosineAnnealingLR(T_max=100, eta_min=1e-6)
    GradScaler + autocast(fp16) / BCEDiceLoss（plain UNet 无 EGM ⇒ 与 baseline 同族）

跑法
----
    PY=D:/miniconda/aniconda/envs/medical_seg/python.exe
    $PY E:/paper2_ablation_reliability/02_code/tools/longrun.py \\
        --name p3_4_train_bf72 -- $PY <本文件> --bf 72

烟测（务必先做，1 epoch 即可）：
    ... <本文件> --bf 72 --epochs 1 --output_dir ./experiments/_p34_smoke
"""

from __future__ import annotations

import argparse
import os
import sys

D_ROOT = "D:/medical_segmentation"

# name -> base_filters（冻结，对应 T-CF-3a）
BF_TARGETS = {
    64: ("plainUNet_bf64", 31.0376),
    66: ("plainUNet_bf66", 33.0073),
    71: ("plainUNet_bf71", 38.1967),
    72: ("plainUNet_bf72", 39.2800),
    73: ("plainUNet_bf73", 40.3784),
    74: ("plainUNet_bf74", 41.4920),
    79: ("plainUNet_bf79", 47.2873),
}
PARAMS_TOL_M = 1e-3

PLAIN_CFG = {"use_egm": False, "use_dpa": False, "use_msfa": False,
             "description": "Capacity-matched plain UNet"}


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--bf", type=int, required=True, choices=sorted(BF_TARGETS))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--output_dir", type=str, default="./experiments/capacity_matched")
    a = ap.parse_args()

    name, expect_m = BF_TARGETS[a.bf]
    if a.bf == 64:
        print("[P3-4] 提示：bf64 与既有 ablation/baseline 等价（31.0376 M），"
              "通常可直接复用既有权重，无需重训。")

    sys.path.insert(0, D_ROOT)
    os.chdir(D_ROOT)

    import models.ablation_models as am

    cfg = dict(PLAIN_CFG)
    cfg["base_filters"] = a.bf          # 额外键：供补丁后的工厂读取
    cfg["description"] = f"Capacity-matched plain UNet (base_filters={a.bf})"
    am.ABLATION_CONFIGS[name] = cfg

    # 注入后自检参数量（G6 冻结值）
    _net = am.AblationUNet(in_channels=3, num_classes=1, base_filters=a.bf,
                           use_egm=False, use_dpa=False, use_msfa=False)
    _p = sum(x.numel() for x in _net.parameters()) / 1e6
    del _net
    assert abs(_p - expect_m) < PARAMS_TOL_M, (
        f"bf={a.bf} 参数量 {_p:.4f} M ≠ 冻结值 {expect_m} M"
    )

    print("=" * 78)
    print(f"[P3-4] 注入 {name}（未改 D 盘源码）| base_filters={a.bf} | 参数量 {_p:.4f} M")
    print("=" * 78)

    sys.argv = [
        "train_single.py",
        "--model", name,
        "--type", "ablation",
        "--epochs", str(a.epochs),
        "--batch_size", str(a.batch_size),
        "--output_dir", a.output_dir,
    ]
    print("[P3-4] 生效 argv:", " ".join(sys.argv[1:]))
    print("=" * 78)

    # ⚠️ 必须在 import train_single **之后** 补丁其模块级名字
    import train_single

    _orig = train_single.get_ablation_model

    def _patched(config_name, in_channels=3, num_classes=1, base_filters=64):
        c = am.ABLATION_CONFIGS[config_name]
        return am.AblationUNet(
            in_channels=in_channels,
            num_classes=num_classes,
            base_filters=c.get("base_filters", base_filters),
            use_egm=c["use_egm"], use_dpa=c["use_dpa"], use_msfa=c["use_msfa"],
        ), c

    train_single.get_ablation_model = _patched
    assert train_single.get_ablation_model is not _orig
    train_single.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
