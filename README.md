# EEG-fMRI-Contrastive

一个自包含融合工程，用于把 eeg-CBraMod 和 NeuroSTORM 作为双塔编码器，进行 EEG-fMRI 跨模态对比学习与下游分类。

当前工程不再依赖外部 NeuroSTORM 仓库路径；NeuroSTORM 的必需骨干代码已经内嵌到本仓库中。旧的 Brain-JEPA 源码目录仍保留在仓库里，但默认训练入口、主配置和 fMRI 适配层都已经切换到 NeuroSTORM volume 路径。

## 1. 当前默认能力

- 内置 EEG 基础模型 CBraMod。
- 内置 fMRI 基础模型 NeuroSTORM。
- 支持 EEG + fMRI 双塔对称 InfoNCE 对比学习。
- 支持分类微调，融合方式可选 concat、eeg_only、fmri_only。
- 支持 manifest 驱动的数据接入。
- 支持对 fMRI 体积做在线缩放接口：
  - spatial: none / pad_or_crop / interpolate
  - temporal: none / pad_or_crop / interpolate

## 2. 当前默认输入约定

EEG 仍然使用 patch 后的三维数组：

- EEG: [C, S, P]

fMRI 默认切换为 NeuroSTORM 的体积输入：

- fMRI raw sample: [H, W, D, T] 或 [C, H, W, D, T]
- 训练时实际喂给模型的张量: [B, C, H, W, D, T]

默认 NeuroSTORM 配置要求目标体积为 96 x 96 x 96 x 20，但你可以通过配置把不符合要求的数据在线 pad、crop 或 interpolate 到目标尺寸。

注意：缩放只能解决尺寸不一致，不能修复 partial-brain coverage 这种采集层面的信息缺失。

## 3. 目录重点

```text
EEG-fMRI-Contrastive/
  run_train.py
  run_finetune.py
  requirements.txt
  configs/
    train_contrastive_neurostorm_volume.yaml
    finetune_classifier_neurostorm_volume.yaml
  mmcontrast/
    datasets/
      paired_manifest_dataset.py
      fmri_volume_ops.py
    backbones/
      eeg_cbramod/
      fmri_neurostorm/
    models/
      eeg_adapter.py
      fmri_adapter.py
      multimodal_model.py
      classifier.py
```

## 4. 环境安装

建议先进入仓库根目录，再安装依赖：

```bash
cd EEG-fMRI-Contrastive
pip install -r requirements.txt
```

NeuroSTORM 路径至少需要：

- monai
- einops

mamba-ssm 已加入 requirements，但在 Windows 上不一定总能直接安装成功。当前仓库内置了一个轻量 fallback mixer，缺少 mamba-ssm 时仍可跑通结构；如果你要尽量贴近原版 NeuroSTORM，仍建议安装 mamba-ssm。

## 5. 快速开始

下面所有命令都默认在仓库根目录执行。

推荐启动方式：

```powershell
cd EEG-fMRI-Contrastive
```

默认路径约定：

- 仓库目录是 `EEG-fMRI-Contrastive/`
- `ds002336/` 和 `ds002739/` 默认与仓库目录同级
- 缓存、日志、训练输出默认都写在仓库内部的 `cache/`、`outputs/` 等相对目录下
- PowerShell 脚本直接调用当前终端里的 `python`

### 5.1 处理 ds002739

处理全部被试，按被试并行，`-NumWorkers` 控制并行进程数：

```powershell
.\scripts\ds002739\prepare_ds002739.ps1 -NumWorkers 8
```

如果你想单独指定输出目录，避免覆盖当前缓存：

```powershell
.\scripts\ds002739\prepare_ds002739.ps1 -NumWorkers 8 -OutputRoot cache\ds002739_parallel
```

如果只想处理部分被试：

```powershell
.\scripts\ds002739\prepare_ds002739.ps1 -Subjects sub-09 sub-13 -NumWorkers 2 -SplitMode none
```

### 5.2 处理 ds002336

如果你直接使用原始 fMRI 做 Python 预处理：

```powershell
.\scripts\ds002336\prepare_ds002336.ps1
```

如果你使用 SPM12 预处理后的 fMRI，这是当前更推荐的路径。这个命令会先跑全部人的 SPM12 预处理，再自动生成模型输入缓存：

```powershell
.\scripts\ds002336\prepare_ds002336_spm.ps1 -ParallelJobs 4
```

说明：

- `-ParallelJobs` 控制同时启动多少个 MATLAB 进程。
- 这个脚本要求本机已经能直接调用 `matlab`，并且 SPM12 已可用。
- 当前 Python 侧会优先读取 `derivatives/spm12_preproc` 下的最终 NIfTI，不会再回退去读原始 fMRI，除非你显式改脚本参数。

### 5.3 一次性顺序处理两个数据集

如果你只是想顺序执行仓库默认的数据准备脚本：

```powershell
.\scripts\run_prepare_all.ps1
```

这个脚本会先处理 ds002739，再处理 ds002336。它本身不额外传并行参数；如果你需要并行控制，优先直接调用各自的数据集脚本。

## 6. 数据接入

Manifest CSV 必需列：

- eeg_path
- fmri_path

可选列：

- sample_id
- label
- eeg_shape
- fmri_shape

示例：

```csv
sample_id,eeg_path,fmri_path,label
s0001,eeg/sample_0001.npy,fmri/sample_0001.npy,0
s0002,eeg/sample_0002.npy,fmri/sample_0002.npy,1
```

## 7. 关键配置

当前默认训练配置：

- configs/train_contrastive_neurostorm_volume.yaml
- configs/finetune_classifier_neurostorm_volume.yaml

核心字段说明：

### data

- train_manifest_csv / val_manifest_csv / test_manifest_csv: manifest 路径
- root_dir: 相对路径根目录
- fmri_input_type: 默认 volume
- fmri_target_shape: NeuroSTORM 目标尺寸，默认 [96, 96, 96, 20]
- fmri_spatial_strategy: none / pad_or_crop / interpolate
- fmri_temporal_strategy: none / pad_or_crop / interpolate
- fmri_normalize_nonzero_only: 是否只对非零脑区做 z-score

### fmri_model

- backbone: neurostorm
- img_size: 必须和 data.fmri_target_shape 保持一致
- patch_size: 默认 [6, 6, 6, 1]
- depths: 默认 [2, 2, 6, 2]
- num_heads: 默认 [3, 6, 12, 24]
- embed_dim: 默认 24

### 形状校验规则

- EEG 样本必须是 [C, S, P]
- NeuroSTORM 目标体积必须是四维 [H, W, D, T]
- fmri_model.img_size 必须与 data.fmri_target_shape 一致
- img_size 必须能被 patch_size 整除
- 如果 manifest 中的原始体积尺寸与 img_size 不同，必须显式打开 spatial 或 temporal 缩放策略，否则启动前就会报错

## 8. 训练与微调命令

### 8.1 ds002739 对比学习

直接用仓库默认配置启动：

```powershell
.\scripts\ds002739\run_ds002739_contrastive.ps1
```

默认配置文件是：

- `configs/train_ds002739.yaml`

如果你想改配置文件：

```powershell
.\scripts\ds002739\run_ds002739_contrastive.ps1 -Config configs\train_ds002739.yaml
```

### 8.2 ds002739 分类微调

如果你已经有对比学习得到的 checkpoint，直接传进去：

```powershell
.\scripts\ds002739\run_ds002739_finetune.ps1 -ContrastiveCheckpoint .\outputs\ds002739\contrastive\checkpoints\best.pth
```

如果不传 `-ContrastiveCheckpoint`，就会按配置或随机初始化进入微调。

默认配置文件是：

- `configs/finetune_ds002739.yaml`

### 8.3 ds002336 对比学习

```powershell
.\scripts\ds002336\run_ds002336_contrastive.ps1
```

默认配置文件是：

- `configs/train_ds002336.yaml`

### 8.4 ds002336 分类微调

```powershell
.\scripts\ds002336\run_ds002336_finetune.ps1 -ContrastiveCheckpoint .\outputs\ds002336\contrastive\checkpoints\best.pth
```

默认配置文件是：

- `configs/finetune_ds002336.yaml`

### 8.5 直接用 Python 入口

如果你不想走 PowerShell 包装脚本，也可以直接调 Python：

```bash
python run_train.py --config configs/train_ds002739.yaml
```

```bash
python run_finetune.py --config configs/finetune_ds002739.yaml --contrastive-checkpoint outputs/ds002739/contrastive/checkpoints/best.pth
```

```bash
python run_train.py --config configs/train_ds002336.yaml
```

```bash
python run_finetune.py --config configs/finetune_ds002336.yaml --contrastive-checkpoint outputs/ds002336/contrastive/checkpoints/best.pth
```

## 9. 当前实现边界

- 当前默认路径已经切到 NeuroSTORM volume 输入。
- 旧的 Brain-JEPA 相关配置文件仍在仓库里，但不再是默认入口。
- 如果你后续删除外部 NeuroSTORM-main 文件夹，不会影响这个仓库当前的 NeuroSTORM 主路径。