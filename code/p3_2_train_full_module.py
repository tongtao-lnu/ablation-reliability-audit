# -*- coding: utf-8 -*-
"""
p3_2_train_full_module.py -- P3-2：补 2^3 第 8 格（egm_dpa_msfa，三模块全开）

作用
----
用**与既有 7 格逐项相同的训练协议**训练 2^3 析因设计的第 8 格：
    egm_dpa_msfa = U-Net + EGM(开) + DPA(开) + MSFA(开)

为什么需要它
------------
`experiments/ablation/` 只有 7 个子目录（baseline / egm_only / dpa_only / msfa_only /
egm_dpa / egm_msfa / dpa_msfa），2^3 析因缺"三模块全开"这一格。后果是
`experiments/paper_tables/Table1_Ablation_Study.md` 第 8 行
`EGA-UNet (Full) | 48.40 M | Dice 96.38%` **没有同协议产物可复现**
（该行已冻结为引用禁令 M10；而 Table1 前 7 行的参数量 7/7 与我们 7 格逐项吻合，
反证前 7 行同协议、第 8 行异常）。

本段补上第 8 格后，即可给出**同协议**读数，并用它替换/说明 Table1 旧 Full 行。

设计（G7 落盘 / **D 盘只读**）
---------------------------
* **不修改 D 盘源码**：`D:/medical_segmentation/models/ablation_models.py` 保持原样
  （已留备份 `ablation_models.py.bak_p32_20260914`）。第 8 格配置在本脚本内
  **运行时注入** `ABLATION_CONFIGS` —— `train_single.py` 第 18 行以
  `from models.ablation_models import get_ablation_model, ABLATION_CONFIGS` 导入，
  绑定的是**同一个 dict 对象**，故注入对其可见。
* 训练输出仍落 D 盘既定位置 `experiments/ablation/egm_dpa_msfa/`
  （与 7 格同构；**仅新增目录**，不改动任何既有产物）。

协议冻结（G6：与 7 格逐项一致，跑前写死，事后不改）
-------------------------------------------------
    --type ablation --epochs 100 --batch_size 4 --lr 1e-4 --seed 42
    val_interval 5 / early_stop 30                    （train_single 默认）
    AdamW(lr=1e-4, weight_decay=1e-5)
    CosineAnnealingLR(T_max=100, eta_min=1e-6)
    GradScaler + autocast(fp16)                       （train_epoch 内）
    CombinedLoss                                      （因 use_egm=True，
                                                       与 egm_only/egm_dpa/egm_msfa 同族）
    每 val_interval 在 val 上选优 -> best_model.pth；末尾用 best 在 test 上评测

⏱ 预算实测校准：`dpa_msfa` 的 `training_time_min` = 138.76 min（100 epoch）。
   第 8 格参数量更大（48.42 M vs 41.43 M），预计 **~2–2.5 h**（计划给的是 0.5–2 h，
   以实测为准）。**train_single 无断点续跑**：中断即需重跑 ⇒ 跑前须接电源 + 永不睡眠。

跑法
----
    PY=D:/miniconda/aniconda/envs/medical_seg/python.exe
    $PY E:/paper2_ablation_reliability/02_code/tools/longrun.py \\
        --name p3_2_train_egm_dpa_msfa \\
        -- $PY E:/paper2_ablation_reliability/02_code/analysis/p3_2_train_full_module.py

额外参数原样追加（后缀覆盖前缀，argparse 取最后一个），便于**烟测**：
    ... p3_2_train_full_module.py --epochs 1 --output_dir ./experiments/_p32_smoke
"""

from __future__ import annotations

import os
import sys

# --------------------------------------------------------------------------- #
# 冻结常量（G6）
# --------------------------------------------------------------------------- #
D_ROOT = "D:/medical_segmentation"
CONFIG_NAME = "egm_dpa_msfa"
EXPECTED_PARAMS_M = 48.4154      # base_filters=64 / in_ch=3 / num_classes=1 的实测值
PARAMS_TOL_M = 0.001

FULL_CFG = {
    "use_egm": True,
    "use_dpa": True,
    "use_msfa": True,
    "description": "U-Net + EGM + DPA + MSFA (2^3 full)",
}

# --------------------------------------------------------------------------- #
# 运行时注入第 8 格（不改 D 盘文件）
# --------------------------------------------------------------------------- #
sys.path.insert(0, D_ROOT)
os.chdir(D_ROOT)

import models.ablation_models as am

_ref = am.ABLATION_CONFIGS["dpa_msfa"]
assert set(FULL_CFG) == set(_ref), "第 8 格字段与既有 7 格不一致（G6 冻结失败）"
assert FULL_CFG["use_egm"] and FULL_CFG["use_dpa"] and FULL_CFG["use_msfa"], "三模块须全开"

am.ABLATION_CONFIGS[CONFIG_NAME] = FULL_CFG
assert CONFIG_NAME in am.ABLATION_CONFIGS

# 注入后自检：能构模 + 参数量符合冻结值
_net, _cfg = am.get_ablation_model(CONFIG_NAME, in_channels=3, num_classes=1, base_filters=64)
_p = sum(x.numel() for x in _net.parameters()) / 1e6
del _net
assert abs(_p - EXPECTED_PARAMS_M) < PARAMS_TOL_M, (
    f"参数量 {_p:.4f} M 与冻结值 {EXPECTED_PARAMS_M} M 不符"
)

print("=" * 78)
print("[P3-2] 第 8 格运行时注入成功（未改动 D 盘源码）")
print(f"[P3-2] configs = {list(am.ABLATION_CONFIGS.keys())}")
print(f"[P3-2] {CONFIG_NAME}: EGM/DPA/MSFA = 1/1/1 | 参数量 {_p:.4f} M")
print(f"[P3-2] 参照：Table1 旧 Full 行声称 48.40 M / Dice 96.38%")
print("=" * 78)

# --------------------------------------------------------------------------- #
# 冻结协议并交给 train_single
# --------------------------------------------------------------------------- #
EXTRA = sys.argv[1:]
sys.argv = [
    "train_single.py",
    "--model", CONFIG_NAME,
    "--type", "ablation",
    "--epochs", "100",
    "--batch_size", "4",
] + EXTRA
print("[P3-2] 生效 argv:", " ".join(sys.argv[1:]))
print("=" * 78)

from train_single import main  # noqa: E402

main()
