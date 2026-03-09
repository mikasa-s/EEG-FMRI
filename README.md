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

建议先安装依赖：

```bash
pip install -r requirements.txt
```

NeuroSTORM 路径至少需要：

- monai
- einops

mamba-ssm 已加入 requirements，但在 Windows 上不一定总能直接安装成功。当前仓库内置了一个轻量 fallback mixer，缺少 mamba-ssm 时仍可跑通结构；如果你要尽量贴近原版 NeuroSTORM，仍建议安装 mamba-ssm。

## 5. 数据接入

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

## 6. 关键配置

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

## 7. 训练入口

默认对比学习：

```bash
python run_train.py
```

默认分类微调：

```bash
python run_finetune.py
```

也可以覆盖常用字段：

```bash
python run_train.py --config configs/train_contrastive_neurostorm_volume.yaml --train-manifest outputs/volume_dataset/manifest_train.csv --val-manifest outputs/volume_dataset/manifest_val.csv --root-dir outputs/volume_dataset --output-dir outputs/contrastive_volume_run2 --set data.fmri_spatial_strategy='interpolate' --set data.fmri_temporal_strategy='pad_or_crop'
```

```bash
python run_finetune.py --config configs/finetune_classifier_neurostorm_volume.yaml --train-manifest outputs/volume_dataset/manifest_train.csv --val-manifest outputs/volume_dataset/manifest_val.csv --test-manifest outputs/volume_dataset/manifest_test.csv --root-dir outputs/volume_dataset --contrastive-checkpoint outputs/contrastive_neurostorm_volume/checkpoints/best.pth
```

## 8. 当前实现边界

- 当前默认路径已经切到 NeuroSTORM volume 输入。
- 旧的 Brain-JEPA 相关配置文件仍在仓库里，但不再是默认入口。
- 如果你后续删除外部 NeuroSTORM-main 文件夹，不会影响这个仓库当前的 NeuroSTORM 主路径。