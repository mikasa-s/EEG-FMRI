# CMCL-EEG

本项目用于：

- 联合 EEG-fMRI 预训练
- EEG-only LOSO 微调与测试
- 离线 LOSO 评估与可视化

当前只保留主流程：

1. 数据预处理
2. 离线预训练
3. LOSO 微调
4. LOSO 离线评估与可视化

旧版 README 已备份到：

- `README.backup_before_rewrite.md`

## 1. 数据集

- `ds002336`：XP1，参与联合预训练，也用于 LOSO 微调
- `ds002338`：XP2，参与联合预训练，也用于 LOSO 微调
- `ds002739`：PDC，参与联合预训练；如需下游，可按 LOSO 配置运行
- `ds009999`：SEED，不参与联合预训练 cache 构建，只用于 EEG-only LOSO 微调

## 2. 路径约定

默认情况下，`CMCL-EEG` 与原始数据集目录位于同一级目录。README 中的示例统一使用相对路径，例如：

- `..\ds002336`
- `..\ds002338`
- `..\ds002739`
- `..\SEED`

不使用 `E:\...` 这类盘符绝对路径。

目录约定如下：

- 各数据集 cache
  - `cache/ds002336`
  - `cache/ds002338`
  - `cache/ds002739`
  - `cache/ds009999`
- 联合预训练 cache
  - `cache/joint_contrastive`
- 预训练权重
  - `pretrained_weights/pretrain_full/<objective>/checkpoints/best.pth`
  - `pretrained_weights/pretrain_strict/<dataset>/<subject>/<objective>/checkpoints/best.pth`
- 微调输出
  - `outputs/<dataset>/finetune`
- LOSO 离线评估输出
  - `outputs/<dataset>/offline_eval`

`<objective>` 可选：

- `contrastive`
- `infonce`
- `barlow_twins`

## 3. 预处理

### 3.1 各数据集 cache

Windows：

```powershell
.\scripts\ds00233x\prepare_ds00233x.ps1 -DatasetName ds002336 -DsRoot ..\ds002336 -OutputRoot cache\ds002336
.\scripts\ds00233x\prepare_ds00233x.ps1 -DatasetName ds002338 -DsRoot ..\ds002338 -OutputRoot cache\ds002338
.\scripts\ds002739\prepare_ds002739.ps1 -DsRoot ..\ds002739 -OutputRoot cache\ds002739
.\scripts\ds009999\prepare_ds009999.ps1 -DsRoot ..\SEED -OutputRoot cache\ds009999
```

Linux：

```bash
./scripts_linux/ds00233x/prepare_ds00233x.sh --dataset-name ds002336 --ds-root ../ds002336 --output-root cache/ds002336
./scripts_linux/ds00233x/prepare_ds00233x.sh --dataset-name ds002338 --ds-root ../ds002338 --output-root cache/ds002338
./scripts_linux/ds002739/prepare_ds002739.sh --ds-root ../ds002739 --output-root cache/ds002739
./scripts_linux/ds009999/prepare_ds009999.sh --ds-root ../SEED --output-root cache/ds009999
```

### 3.2 联合预训练 cache

联合预训练 cache 只需要为 `ds002336`、`ds002338`、`ds002739` 构建。

Windows：

```powershell
.\scripts\prepare_joint_contrastive.ps1
```

Linux：

```bash
./scripts_linux/prepare_joint_contrastive.sh
```

输出：

- `cache/joint_contrastive/manifest_all.csv`

当前 joint pretrain 的 EEG 通道策略：

- 每个数据集保留自己的全部通道
- 跨数据集公共通道排在前面
- 各数据集剩余通道接在后面
- 预训练时同一 batch 只采同一数据集，不做 padding

如果你之前的 `cache/joint_contrastive` 是旧版“纯交集通道”结构，建议整库重建。

## 4. 离线预训练

### 4.1 预训练模式

支持两种模式：

- `full`
  - 直接使用 `cache/joint_contrastive/manifest_all.csv` 的全部样本
  - 不排除任何数据集或被试
  - 不需要 `--target-dataset`
- `strict`
  - 仍然直接使用 `cache/joint_contrastive`
  - 只在启动预训练时，从 `manifest_all.csv` 中排除“目标数据集中的目标测试被试”
  - 不修改原始 cache
  - 只传 `--target-dataset` 时，会自动遍历该数据集全部被试并逐个训练

### 4.2 主方法 `contrastive`

Windows：

```powershell
python run_pretrain.py --config configs\train_joint_contrastive.yaml --pretrain-mode full
python run_pretrain.py --config configs\train_joint_contrastive.yaml --pretrain-mode strict --target-dataset ds002336
python run_pretrain.py --config configs\train_joint_contrastive.yaml --pretrain-mode strict --target-dataset ds002338
```

Linux：

```bash
python run_pretrain.py --config configs/train_joint_contrastive.yaml --pretrain-mode full
python run_pretrain.py --config configs/train_joint_contrastive.yaml --pretrain-mode strict --target-dataset ds002336
python run_pretrain.py --config configs/train_joint_contrastive.yaml --pretrain-mode strict --target-dataset ds002338
```

## 5. LOSO 微调

README 只保留 LOSO 微调，不再介绍单 fold 微调。

### 5.1 预训练权重自动定位

如果 `finetune.contrastive_checkpoint_path` 为空，程序会根据以下配置自动定位预训练权重：

- `finetune.pretrain_mode`
- `finetune.pretrain_objective`
- `finetune.pretrain_output_root`

如果找不到对应预训练权重，并且配置里开启了：

```yaml
finetune:
  allow_missing_pretrain_checkpoint: true
```

则会退回随机初始化。

### 5.2 LOSO 命令

Windows：

```powershell
python run_finetune.py --config configs\finetune_ds002336.yaml --loso --root-dir cache\ds002336 --output-dir outputs\ds002336\finetune
python run_finetune.py --config configs\finetune_ds002338.yaml --loso --root-dir cache\ds002338 --output-dir outputs\ds002338\finetune
python run_finetune.py --config configs\finetune_ds002739.yaml --loso --root-dir cache\ds002739 --output-dir outputs\ds002739\finetune
python run_finetune.py --config configs\finetune_ds009999.yaml --loso --root-dir cache\ds009999 --output-dir outputs\ds009999\finetune
```

Linux：

```bash
python run_finetune.py --config configs/finetune_ds002336.yaml --loso --root-dir cache/ds002336 --output-dir outputs/ds002336/finetune
python run_finetune.py --config configs/finetune_ds002338.yaml --loso --root-dir cache/ds002338 --output-dir outputs/ds002338/finetune
python run_finetune.py --config configs/finetune_ds002739.yaml --loso --root-dir cache/ds002739 --output-dir outputs/ds002739/finetune
python run_finetune.py --config configs/finetune_ds009999.yaml --loso --root-dir cache/ds009999 --output-dir outputs/ds009999/finetune
```

## 6. LOSO 离线评估与可视化

### 6.1 预训练表征可视化

Windows：

```powershell
python run_visualize.py contrastive --config configs\train_joint_contrastive.yaml --checkpoint pretrained_weights\pretrain_full\contrastive\checkpoints\best.pth --output-dir outputs\visualizations\contrastive --batch-size 128 --max-samples 1000 --tsne-max-points 200 --heatmap-max-points 128
```

Linux：

```bash
python run_visualize.py contrastive --config configs/train_joint_contrastive.yaml --checkpoint pretrained_weights/pretrain_full/contrastive/checkpoints/best.pth --output-dir outputs/visualizations/contrastive --batch-size 128 --max-samples 1000 --tsne-max-points 200 --heatmap-max-points 128
```

### 6.2 LOSO 离线评估与混淆矩阵

Windows：

```powershell
python run_visualize.py offline-loso --dataset-name ds002336 --config configs\finetune_ds002336.yaml --checkpoints-root save\2336-671\run_output\finetune --output-dir outputs\ds002336\offline_eval --allow-missing-pretrain-checkpoint
python run_visualize.py offline-loso --dataset-name ds002338 --config configs\finetune_ds002338.yaml --checkpoints-root save\2338\run_output\finetune --output-dir outputs\ds002338\offline_eval --allow-missing-pretrain-checkpoint
python run_visualize.py offline-loso --dataset-name ds009999 --config configs\finetune_ds009999.yaml --checkpoints-root save\9999\run_output\finetune --output-dir outputs\ds009999\offline_eval --allow-missing-pretrain-checkpoint
```

Linux：

```bash
python run_visualize.py offline-loso --dataset-name ds002336 --config configs/finetune_ds002336.yaml --checkpoints-root save/2336-671/run_output/finetune --output-dir outputs/ds002336/offline_eval --allow-missing-pretrain-checkpoint
python run_visualize.py offline-loso --dataset-name ds002338 --config configs/finetune_ds002338.yaml --checkpoints-root save/2338/run_output/finetune --output-dir outputs/ds002338/offline_eval --allow-missing-pretrain-checkpoint
python run_visualize.py offline-loso --dataset-name ds009999 --config configs/finetune_ds009999.yaml --checkpoints-root save/9999/run_output/finetune --output-dir outputs/ds009999/offline_eval --allow-missing-pretrain-checkpoint
```
