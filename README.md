# EEG-fMRI-Contrastive

一个面向 EEG-fMRI 联合学习的训练仓库，当前支持：

- 联合对比预训练
- EEG-only 微调分类
- EEG shared/private 表征拆分
- 离线 EEG band-power 目标回归
- Windows PowerShell 和 Linux Bash 两套脚本
- Optuna 自动搜索

当前主流程已经调整为：

1. 先执行原始数据预处理。
2. 预处理阶段把 EEG 重采样到 `200 Hz`。
3. 在预处理后的 EEG 段上离线计算 `band_power.npy`。
4. 预训练阶段直接读取 band-power 目标，不在训练时在线计算。
5. 微调阶段默认只使用 EEG，分类输入模式支持 `shared`、`private`、`concat`，默认 `concat`。

## 1. 项目概览

### 1.1 预训练目标

当前预训练模型保留原有 EEG encoder 和 fMRI encoder 主体，只在编码器外增加轻量 head：

- EEG shared head
- EEG private head
- fMRI shared head
- EEG private 到 5 维 band-power 的预测 head

预训练损失由三部分组成：

- `InfoNCE(eeg_shared, fmri_shared)`
- `MSE(band_power_pred, band_power_target)`
- `separation_loss(eeg_shared, eeg_private)`

对应实现位置：

- `mmcontrast/models/shared_private.py`
- `mmcontrast/models/multimodal_model.py`
- `mmcontrast/losses.py`

### 1.2 微调目标

微调阶段默认走 EEG-only 分类，支持三种 EEG 特征模式：

- `shared`
- `private`
- `concat`

默认配置是：

```yaml
finetune:
  fusion: eeg_only
  classifier_mode: concat
```

对应实现位置：

- `mmcontrast/models/classifier.py`
- `configs/finetune_ds002336.yaml`
- `configs/finetune_ds002338.yaml`
- `configs/finetune_ds002739.yaml`

## 2. 支持的数据集

当前仓库内置支持：

- `ds002336`
- `ds002338`
- `ds002739`

约定的数据目录通常是仓库同级或上级，例如：

```text
OpenNeuro/
  EEG-fMRI-Contrastive/
  ds002336/
  ds002338/
  ds002739/
```

脚本会自动尝试这些位置：

- `../ds002336`
- `../ds002338`
- `../ds002739`
- `data/ds002336`
- `data/ds002338`
- `data/ds002739`
- `../data/ds002336`
- `../data/ds002338`
- `../data/ds002739`

如果你的目录不在这些位置，请显式传 `--ds-root` 或 `--ds002336-root` / `--ds002338-root` / `--ds002739-root`。

## 3. 目录结构

```text
EEG-fMRI-Contrastive/
  configs/
    train_joint_contrastive.yaml
    finetune_ds002336.yaml
    finetune_ds002338.yaml
    finetune_ds002739.yaml
    optuna_ds002336.yaml
    optuna_ds002336_linux.yaml
    optuna_ds002338.yaml
    optuna_ds002338_linux.yaml
    optuna_ds002739.yaml
    optuna_ds002739_linux.yaml
  preprocess/
    prepare_joint_contrastive.py
    prepare_ds00233x.py
    prepare_ds002739.py
    compute_eeg_band_power_targets.py
    run_spm_preproc_ds00233x.m
  mmcontrast/
    datasets/
    models/
    losses.py
    contrastive_runner.py
    contrastive_trainer.py
    finetune_runner.py
    finetune_trainer.py
  scripts/
    prepare_joint_contrastive.ps1
    run_pretrain_and_finetune.ps1
    run_optuna_pretrain_and_finetune.ps1
    ds00233x/
      prepare_ds00233x.ps1
      prepare_ds00233x_spm.ps1
    ds002739/
      prepare_ds002739.ps1
  scripts_linux/
    prepare_joint_contrastive.sh
    run_pretrain_and_finetune.sh
    run_optuna_pretrain_and_finetune.sh
    ds00233x/
      prepare_ds00233x.sh
      prepare_ds00233x_spm.sh
    ds002739/
      prepare_ds002739.sh
  run_train.py
  run_finetune.py
  run_optuna_search.py
  requirements.txt
  README.md
```

## 4. 环境准备

### 4.1 Windows

```powershell
conda activate mamba
cd D:\OpenNeuro\EEG-fMRI-Contrastive
python -m pip install -r requirements.txt
```

如果 PowerShell 阻止脚本执行，可以临时使用：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\prepare_joint_contrastive.ps1
```

### 4.2 Linux

```bash
conda activate mamba
cd /path/to/EEG-fMRI-Contrastive
python -m pip install -r requirements.txt
```

如果脚本没有执行权限：

```bash
chmod +x scripts_linux/*.sh scripts_linux/ds00233x/*.sh scripts_linux/ds002739/*.sh
```

## 5. 数据预处理总流程

推荐按下面顺序执行：

1. 联合预训练缓存预处理。
2. 生成离线 band-power 目标。
3. 单数据集微调缓存预处理。
4. 运行联合预训练。
5. 运行单数据集 LOSO 微调。

说明：

- `scripts/prepare_joint_contrastive.ps1` 和 `scripts_linux/prepare_joint_contrastive.sh` 已经自动串上了 band-power 计算。
- 如果你只单独重算 band-power，可以直接运行 `preprocess/compute_eeg_band_power_targets.py`。
- 微调缓存预处理不会自动重算 band-power，因为 band-power 只在预训练阶段使用。

## 6. EEG band-power 目标说明

### 6.1 计算时机

band-power 必须在原始 EEG 预处理完成之后再计算，而不是直接对原始 EEG 文件计算。

当前仓库的原始 EEG 预处理会把每个 EEG 段重采样到：

- `200 Hz`

因此 band-power 计算也必须基于 `200 Hz` 的结果。

### 6.2 时间长度

当前默认 EEG 段长度为：

- `8` 秒

所以 band-power 脚本默认期望每段 EEG 的长度是：

- `8 x 200 = 1600`

如果不是这个长度，脚本会报错，提示先完成原始预处理。

### 6.3 频带定义

当前固定为 5 维：

- `delta`: `0.5-4`
- `theta`: `4-8`
- `alpha`: `8-13`
- `beta`: `13-30`
- `gamma`: `30-40`

注意：

- 原始 EEG 预处理带通范围是 `0.5-40 Hz`
- 所以 `gamma` 必须是 `30-40 Hz`
- 不能再写成高于 `40 Hz` 的频段

### 6.4 保存格式

band-power 目标保存为：

- shape `[N, 5]`

对于 subject pack 模式，脚本行为是：

- 读取 `eeg.npy`
- 计算 `band_power.npy`
- 只新增或覆盖 `band_power.npy`
- 最多顺手更新 `metadata.json`
- 不会删除整个 subject 目录
- 不会重写 `eeg.npy`、`fmri.npy`、`sample_id.npy` 等其他数组

对应脚本：

- `preprocess/compute_eeg_band_power_targets.py`

## 7. 联合预训练缓存预处理

联合预训练缓存用于跨数据集对比预训练，默认输出到：

- `cache/joint_contrastive`

产物通常包括：

- `manifest_all.csv`
- `eeg_channels_target.csv`
- `subjects/<dataset>_<subject>/...`
- `subjects/<dataset>_<subject>/band_power.npy`

### 7.1 Windows

```powershell
.\scripts\prepare_joint_contrastive.ps1 `
  -Ds002336Root ..\ds002336 `
  -Ds002338Root ..\ds002338 `
  -Ds002739Root ..\ds002739 `
  -OutputRoot cache\joint_contrastive `
  -Datasets ds002336,ds002338,ds002739 `
  -EegWindowSec 8.0 `
  -NumWorkers 2
```

说明：

- 脚本会先调用 `preprocess/prepare_joint_contrastive.py`
- 然后自动调用 `preprocess/compute_eeg_band_power_targets.py`
- band-power 计算默认使用 `--sample-rate-hz 200`

### 7.2 Linux

```bash
./scripts_linux/prepare_joint_contrastive.sh \
  --ds002336-root ../ds002336 \
  --ds002338-root ../ds002338 \
  --ds002739-root ../ds002739 \
  --output-root cache/joint_contrastive \
  --datasets ds002336,ds002338,ds002739 \
  --eeg-window-sec 8.0 \
  --num-workers 2
```

### 7.3 单独重算 band-power

如果 joint cache 已经存在，只想补或重算 band-power：

#### Windows

```powershell
python .\preprocess\compute_eeg_band_power_targets.py `
  --manifest-csv cache\joint_contrastive\manifest_all.csv `
  --root-dir cache\joint_contrastive `
  --sample-rate-hz 200 `
  --window-sec 8 `
  --overwrite
```

#### Linux

```bash
python preprocess/compute_eeg_band_power_targets.py \
  --manifest-csv cache/joint_contrastive/manifest_all.csv \
  --root-dir cache/joint_contrastive \
  --sample-rate-hz 200 \
  --window-sec 8 \
  --overwrite
```

## 8. 单数据集微调缓存预处理

单数据集微调缓存默认用于 LOSO 划分和分类训练，通常输出到：

- `cache/ds002336`
- `cache/ds002338`
- `cache/ds002739`

### 8.1 ds002336 / ds002338

脚本：

- Windows: `scripts/ds00233x/prepare_ds00233x.ps1`
- Linux: `scripts_linux/ds00233x/prepare_ds00233x.sh`

默认行为：

- EEG 按 block 切分
- `block_window_sec=8.0`
- `block_overlap_sec=2.0`
- EEG patch 形状通常为 `[C, 8, 200]`
- 默认生成 `loso_subjectwise`
- 默认 `eeg_only`
- 可通过 `--target-channel-manifest` 对齐公共通道顺序

#### Windows

```powershell
.\scripts\ds00233x\prepare_ds00233x.ps1 `
  -DatasetName ds002336 `
  -DsRoot ..\ds002336 `
  -OutputRoot cache\ds002336 `
  -SplitMode loso `
  -TrainingReady `
  -EegOnly `
  -TargetChannelManifest cache\joint_contrastive\eeg_channels_target.csv
```

```powershell
.\scripts\ds00233x\prepare_ds00233x.ps1 `
  -DatasetName ds002338 `
  -DsRoot ..\ds002338 `
  -OutputRoot cache\ds002338 `
  -SplitMode loso `
  -TrainingReady `
  -EegOnly `
  -TargetChannelManifest cache\joint_contrastive\eeg_channels_target.csv
```

#### Linux

```bash
./scripts_linux/ds00233x/prepare_ds00233x.sh \
  --dataset-name ds002336 \
  --ds-root ../ds002336 \
  --output-root cache/ds002336 \
  --split-mode loso \
  --training-ready \
  --eeg-only \
  --target-channel-manifest cache/joint_contrastive/eeg_channels_target.csv
```

```bash
./scripts_linux/ds00233x/prepare_ds00233x.sh \
  --dataset-name ds002338 \
  --ds-root ../ds002338 \
  --output-root cache/ds002338 \
  --split-mode loso \
  --training-ready \
  --eeg-only \
  --target-channel-manifest cache/joint_contrastive/eeg_channels_target.csv
```

### 8.2 ds002336 / ds002338 的 SPM 版本

如果你已经准备好 MATLAB + SPM，可以先做 SPM fMRI 预处理，再走 Python 打包：

#### Windows

```powershell
.\scripts\ds00233x\prepare_ds00233x_spm.ps1 `
  -DatasetName ds002338 `
  -DsRoot ..\ds002338 `
  -OutputRoot cache\ds002338 `
  -SplitMode loso `
  -TrainingReady `
  -EegOnly `
  -TargetChannelManifest cache\joint_contrastive\eeg_channels_target.csv
```

#### Linux

```bash
./scripts_linux/ds00233x/prepare_ds00233x_spm.sh \
  --dataset-name ds002338 \
  --ds-root ../ds002338 \
  --output-root cache/ds002338 \
  --split-mode loso \
  --training-ready \
  --eeg-only \
  --target-channel-manifest cache/joint_contrastive/eeg_channels_target.csv
```

### 8.3 ds002739

脚本：

- Windows: `scripts/ds002739/prepare_ds002739.ps1`
- Linux: `scripts_linux/ds002739/prepare_ds002739.sh`

默认行为：

- EEG 目标采样率 `200 Hz`
- EEG 默认窗口 `8.0` 秒
- fMRI 默认窗口 `6.0` 秒
- 默认生成 `loso_subjectwise`
- 默认 `eeg_only`
- 可通过 `--target-channel-manifest` 对齐公共通道顺序

#### Windows

```powershell
.\scripts\ds002739\prepare_ds002739.ps1 `
  -DsRoot ..\ds002739 `
  -OutputRoot cache\ds002739 `
  -SplitMode loso `
  -TrainingReady `
  -EegOnly `
  -TargetChannelManifest cache\joint_contrastive\eeg_channels_target.csv
```

#### Linux

```bash
./scripts_linux/ds002739/prepare_ds002739.sh \
  --ds-root ../ds002739 \
  --output-root cache/ds002739 \
  --split-mode loso \
  --training-ready \
  --eeg-only \
  --target-channel-manifest cache/joint_contrastive/eeg_channels_target.csv
```

## 9. 联合预训练

主入口：

- `run_train.py`

默认配置：

- `configs/train_joint_contrastive.yaml`

关键配置项：

```yaml
train:
  temperature: 0.07
  band_power_loss_weight: 1.0
  separation_loss_weight: 0.1
  head_dropout: 0.0

eeg_model:
  shared_dim: 256
  private_dim: 256
  band_power_dim: 5

fmri_model:
  shared_dim: 256
```

### 9.1 Windows

```powershell
python .\run_train.py `
  --config configs\train_joint_contrastive.yaml `
  --manifest cache\joint_contrastive\manifest_all.csv `
  --root-dir cache\joint_contrastive `
  --output-dir outputs\joint_contrastive\contrastive
```

### 9.2 Linux

```bash
python run_train.py \
  --config configs/train_joint_contrastive.yaml \
  --manifest cache/joint_contrastive/manifest_all.csv \
  --root-dir cache/joint_contrastive \
  --output-dir outputs/joint_contrastive/contrastive
```

### 9.3 常用覆盖参数

#### Windows

```powershell
python .\run_train.py `
  --config configs\train_joint_contrastive.yaml `
  --set train.epochs=30 `
  --set train.batch_size=16 `
  --set train.eval_batch_size=16 `
  --set train.lr=3e-4 `
  --set train.band_power_loss_weight=1.0 `
  --set train.separation_loss_weight=0.1
```

#### Linux

```bash
python run_train.py \
  --config configs/train_joint_contrastive.yaml \
  --set train.epochs=30 \
  --set train.batch_size=16 \
  --set train.eval_batch_size=16 \
  --set train.lr=3e-4 \
  --set train.band_power_loss_weight=1.0 \
  --set train.separation_loss_weight=0.1
```

## 10. 单数据集微调

主入口：

- `run_finetune.py`

微调配置文件：

- `configs/finetune_ds002336.yaml`
- `configs/finetune_ds002338.yaml`
- `configs/finetune_ds002739.yaml`

当前默认是 EEG-only 微调：

```yaml
finetune:
  fusion: eeg_only
  classifier_mode: concat
```

`classifier_mode` 可选：

- `shared`
- `private`
- `concat`

### 10.1 ds002336

#### Windows

```powershell
python .\run_finetune.py `
  --config configs\finetune_ds002336.yaml `
  --classifier-mode concat
```

#### Linux

```bash
python run_finetune.py \
  --config configs/finetune_ds002336.yaml \
  --classifier-mode concat
```

### 10.2 ds002338

#### Windows

```powershell
python .\run_finetune.py `
  --config configs\finetune_ds002338.yaml `
  --classifier-mode concat
```

#### Linux

```bash
python run_finetune.py \
  --config configs/finetune_ds002338.yaml \
  --classifier-mode concat
```

### 10.3 ds002739

#### Windows

```powershell
python .\run_finetune.py `
  --config configs\finetune_ds002739.yaml `
  --classifier-mode concat
```

#### Linux

```bash
python run_finetune.py \
  --config configs/finetune_ds002739.yaml \
  --classifier-mode concat
```

### 10.4 指定 checkpoint

#### Windows

```powershell
python .\run_finetune.py `
  --config configs\finetune_ds002338.yaml `
  --contrastive-checkpoint outputs\joint_contrastive\contrastive\checkpoints\best.pth `
  --classifier-mode concat
```

#### Linux

```bash
python run_finetune.py \
  --config configs/finetune_ds002338.yaml \
  --contrastive-checkpoint outputs/joint_contrastive/contrastive/checkpoints/best.pth \
  --classifier-mode concat
```

## 11. 一键跑完整流程

完整流程脚本会按顺序完成：

1. 联合缓存预处理
2. band-power 计算
3. 联合预训练
4. 目标数据集缓存预处理
5. LOSO 微调
6. 汇总 `loso_finetune_summary.csv`

### 11.1 Windows

```powershell
.\scripts\run_pretrain_and_finetune.ps1 `
  -PretrainDatasets ds002336,ds002338,ds002739 `
  -TargetDataset ds002338
```

常用附加参数：

- `-SkipPretrain`
- `-SkipFinetune`
- `-TestOnly`
- `-GpuCount 1`
- `-GpuIds 0`
- `-PretrainEpochs 30`
- `-FinetuneEpochs 30`
- `-PretrainBatchSize 16`
- `-FinetuneBatchSize 16`

### 11.2 Linux

```bash
./scripts_linux/run_pretrain_and_finetune.sh \
  --pretrain-datasets ds002336,ds002338,ds002739 \
  --target-dataset ds002338
```

常用附加参数：

- `--skip-pretrain`
- `--skip-finetune`
- `--test-only`
- `--gpu-count 1`
- `--gpu-ids 0`
- `--pretrain-epochs 30`
- `--finetune-epochs 30`
- `--pretrain-batch-size 16`
- `--finetune-batch-size 16`

## 12. Optuna 自动搜索

主入口：

- `run_optuna_search.py`

当前仓库同时提供：

- Windows 配置：`configs/optuna_*.yaml`
- Linux 配置：`configs/optuna_*_linux.yaml`

### 12.1 Windows

```powershell
python .\run_optuna_search.py --study-config configs\optuna_ds002338.yaml --mode full
```

```powershell
python .\run_optuna_search.py --study-config configs\optuna_ds002338.yaml --mode finetune_only
```

```powershell
python .\run_optuna_search.py --study-config configs\optuna_ds002338.yaml --mode pretrain_only
```

### 12.2 Linux

```bash
python run_optuna_search.py --study-config configs/optuna_ds002338_linux.yaml --mode full
```

```bash
python run_optuna_search.py --study-config configs/optuna_ds002338_linux.yaml --mode finetune_only
```

```bash
python run_optuna_search.py --study-config configs/optuna_ds002338_linux.yaml --mode pretrain_only
```

说明：

- `full` 会跑预训练和微调
- `finetune_only` 会跳过预训练
- `pretrain_only` 会跳过微调

如果你不想让 Optuna 联合预训练用到全部数据集，需要改对应 `optuna_*.yaml` 里的：

- Windows: `study.static_args -> -PretrainDatasets`
- Linux: `study.static_args -> --pretrain-datasets`

例如：

```yaml
study:
  static_args:
    - --target-dataset
    - ds002338
    - --pretrain-datasets
    - ds002338
```

## 13. Baseline 说明

当前 EEG baseline 统一在：

- `mmcontrast/baselines/eeg_baseline.py`

支持：

- `svm`
- `labram`
- `cbramod`
- `eeg_deformer`
- `eegnet`
- `conformer`
- `tsception`

说明：

- foundation baseline 主要是 `labram` 和 `cbramod`
- 传统 baseline 自带分类头
- `labram` 已改成按输入动态适配，不再写死成固定通道数
- 初始化日志会打印当前原始 EEG 通道数和公共通道重叠信息

## 14. 主要输出文件

### 14.1 预处理输出

- `cache/joint_contrastive/manifest_all.csv`
- `cache/joint_contrastive/eeg_channels_target.csv`
- `cache/joint_contrastive/subjects/<dataset>_<subject>/band_power.npy`
- `cache/<dataset>/loso_subjectwise/fold_*/manifest_train.csv`
- `cache/<dataset>/loso_subjectwise/fold_*/manifest_val.csv`
- `cache/<dataset>/loso_subjectwise/fold_*/manifest_test.csv`

### 14.2 训练输出

- `outputs/joint_contrastive/contrastive/checkpoints/best.pth`
- `outputs/joint_contrastive/contrastive/final_metrics.json`
- `outputs/<dataset>/finetune/fold_*/test_metrics.json`
- `outputs/<dataset>/finetune/loso_finetune_summary.csv`
- `pretrained_weights/contrastive_best.pth`
