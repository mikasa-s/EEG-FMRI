# EEG-fMRI-Contrastive

当前仓库只保留一条正式工作流：

1. 选择一个或多个数据集做 EEG-fMRI 联合对比学习预训练。
2. 预训练阶段先做时间对齐和通道统一，再把全部预训练样本合并成一个训练集。
3. 选择单个目标数据集做分类微调。
4. 微调阶段只使用目标数据集自己的 trial 或 block 切窗逻辑，并按 LOSO 划分 train、val、test。

旧的“按单数据集单独跑 contrastive，再逐折做 contrastive 验证”的流程已经删除。当前不再支持旧的 per-dataset contrastive 运行脚本，也不再保留旧的单数据集 contrastive 配置文件。

## 当前目录

```text
EEG-fMRI-Contrastive/
  README.md
  requirements.txt
  run_train.py
  run_finetune.py
  run_optuna_search.py
  configs/
    train_joint_contrastive.yaml
    finetune_ds002336.yaml
    finetune_ds002338.yaml
    finetune_ds002739.yaml
    optuna_ds002336.yaml
    optuna_ds002338.yaml
    optuna_ds002739.yaml
  preprocess/
    prepare_ds00233x.py
    prepare_ds002739.py
    prepare_joint_contrastive.py
    run_spm_preproc_ds00233x.m
    preprocess_common.py
  mmcontrast/
    contrastive_runner.py
    contrastive_trainer.py
    finetune_runner.py
    finetune_trainer.py
    datasets/
    models/
    losses.py
    metrics.py
  scripts/
    run_pretrain_and_finetune.ps1
    run_optuna_pretrain_and_finetune.ps1
    prepare_joint_contrastive.ps1
    ds00233x/
      prepare_ds00233x.ps1
      prepare_ds00233x_spm.ps1
    ds002739/
      prepare_ds002739.ps1
  scripts_linux/
    run_pretrain_and_finetune.sh
    run_optuna_pretrain_and_finetune.sh
    prepare_joint_contrastive.sh
    ds00233x/
      prepare_ds00233x.sh
      prepare_ds00233x_spm.sh
    ds002739/
      prepare_ds002739.sh
```

## Linux 脚本说明

- `scripts_linux/` 下提供与 `scripts/` 同名、同用途的 Bash 版本入口脚本。
- 原有 PowerShell 脚本保留不变，Windows 用户继续使用 `scripts/`。
- Linux/macOS 用户建议使用 `scripts_linux/`。

首次使用建议先授予执行权限：

```bash
chmod +x scripts_linux/*.sh scripts_linux/ds00233x/*.sh scripts_linux/ds002739/*.sh
```

## 核心规则

### 1. 联合预训练和微调已经解耦

联合预训练使用 [preprocess/prepare_joint_contrastive.py](preprocess/prepare_joint_contrastive.py)。

- 以单个 fMRI TR 为锚点。
- 对每个 TR 提取该时刻之前固定长度的连续 EEG 窗口，默认 8 s。
- 步长固定为 1 TR。
- 允许跨 trial 提取 EEG，只要 run 内时间轴连续。
- 不再按 EEG event index 和 fMRI event index 直接一一配对。
- 预训练阶段不再划分 train、val、test，全部样本直接写入一个 manifest_all.csv，全部作为训练集。

单数据集微调使用 [preprocess/prepare_ds00233x.py](preprocess/prepare_ds00233x.py) 或 [preprocess/prepare_ds002739.py](preprocess/prepare_ds002739.py)。

- 仍然是目标数据集自己的分类切窗逻辑。
- EEG 不允许跨 trial。
- 标签来自当前 trial 或 block 的分类定义。
- 微调阶段才做 LOSO 划分。
- 微调读取阶段会再次按有序目标通道子集映射 EEG（`data.eeg_channel_subset: auto`），最终送入模型的是通道子集后的张量。

### 2. 可选数据集联合预训练

如果只选择 ds002336：

- 只处理 ds002336。
- 生成 ds002336 的联合预训练样本。
- 直接把这些样本全部送入 contrastive 训练。

如果同时选择 ds002336 和 ds002739：

- 分别处理两个数据集。
- 在预处理阶段完成时间对齐和通道统一。
- 把两个数据集的预训练样本拼接成一个 joint manifest。
- 再统一送入 contrastive 训练。

现在也支持 ds002338：

- 可单独把 ds002338 预处理并追加到已有 joint cache。
- 可与 ds002336、ds002739 任意组合做联合预训练。
- ds002338 走 ds002336 同模板流程，且 fMRI 仍要求先经 MATLAB SPM 预处理。

### 3. 联合 cache 增量更新

- `scripts/prepare_joint_contrastive.ps1` 新增默认增量模式：cache 已有数据集时，默认跳过该数据集重算。
- 仅当你显式传 `-ForceRefreshDatasets` 时才会重算指定数据集。
- `prepare_joint_contrastive.py` 支持 ds002336 与 ds002338 数据集级并行（`--num-workers`）。
- 新增数据集接入时，会只处理新增数据集，并更新：
  - 通道名称规范化
  - 公共通道交集
  - 通道映射关系
- 如果目标通道交集缩小，脚本会对已缓存 subject pack 的 EEG 通道做增量重映射，不会强制全量重导所有数据集。

### 4. subject 命名统一

- 所有数据集导出的标准 subject 统一为 sub01、sub02 这种格式。
- 多数据集场景下再加 subject_uid，格式为 <dataset>_<subject>，例如 ds002336_sub01、ds002739_sub01。
- original_subject 只保留在映射表中做追踪。
- 每次预处理都会导出 subject_mapping.csv。

### 5. EEG 通道统一

- 跨数据集联合预训练前，会先规范化各数据集 EEG 通道名称。
- 公共通道交集按名称求，不按索引求。
- 所有数据集在联合预训练阶段映射到相同通道集合和相同顺序。
- 预处理会导出：
  - eeg_channels_dataset.csv
  - eeg_channels_target.csv
  - eeg_channel_mapping.csv
- 目标数据集微调如果要复用 joint pretrain backbone，应传入 joint 导出的 eeg_channels_target.csv，保证输入通道顺序一致。

## 环境准备

```powershell
conda activate mamba
cd D:\OpenNeuro\EEG-fMRI-Contrastive
pip install -r requirements.txt
```

Linux/macOS：

```bash
conda activate mamba
cd /path/to/OpenNeuro/EEG-fMRI-Contrastive
pip install -r requirements.txt
```

如果出现 `ModuleNotFoundError: No module named 'nibabel'`，先确认当前环境是 `mamba`，再执行：

```powershell
python -m pip install -r requirements.txt
```

## 数据预处理入口（推荐先执行）

联合数据集预处理（只构建联合缓存，不启动对比学习和微调）：

```powershell
.\scripts\prepare_joint_contrastive.ps1 -Datasets ds002338 -OutputRoot cache\joint_contrastive
```

```bash
./scripts_linux/prepare_joint_contrastive.sh --datasets ds002338 --output-root cache/joint_contrastive
```

单数据集微调预处理：

```powershell
.\scripts\ds00233x\prepare_ds00233x.ps1 -DatasetName ds002338 -DsRoot ..\ds002338 -OutputRoot cache\ds002338
```

```bash
./scripts_linux/ds00233x/prepare_ds00233x.sh --dataset-name ds002338 --ds-root ../ds002338 --output-root cache/ds002338
```

现在文档顺序为：先预处理，再训练入口。

## 唯一主入口

正式入口只有一个：

```powershell
.\scripts\run_pretrain_and_finetune.ps1
```

```bash
./scripts_linux/run_pretrain_and_finetune.sh
```

这个脚本会做两件事：

1. 根据 PretrainDatasets 生成联合预训练缓存并训练 contrastive backbone。
2. 根据 TargetDataset 生成单数据集微调缓存并执行 LOSO 微调。

### 常用参数

- PretrainDatasets：预训练使用哪些数据集，可选 ds002336、ds002338、ds002739。
- TargetDataset：微调目标数据集，只能选一个。
- JointTrainConfig：联合预训练配置，默认 [configs/train_joint_contrastive.yaml](configs/train_joint_contrastive.yaml)。
- Ds002336FinetuneConfig：ds002336 微调配置。
- Ds002739FinetuneConfig：ds002739 微调配置。
- JointCacheRoot：联合预训练缓存目录，默认 cache/joint_contrastive。
- JointOutputRoot：联合预训练输出目录，默认 outputs/joint_contrastive。
- PretrainedWeightsDir：联合预训练 best checkpoint 统一存放目录，默认 pretrained_weights。
- JointEegWindowSec：联合预训练的 EEG 连续上下文窗口长度，默认 8 秒。
- PretrainEpochs：覆盖 contrastive epoch。
- FinetuneEpochs：覆盖 finetune epoch。
- PretrainBatchSize：仅覆盖预训练 batch size。
- FinetuneBatchSize：仅覆盖微调训练 batch size。
- BatchSize：兼容旧参数，会同时覆盖 PretrainBatchSize 和 FinetuneBatchSize。
- EvalBatchSize：覆盖微调评估 batch size。
- NumWorkers：覆盖 DataLoader worker 数。
- NumWorkers：也会传递给联合预处理脚本，用于 ds002336 / ds002338 并行处理。
- SkipPretrain：跳过联合预训练，只做微调。
- SkipFinetune：跳过微调，只做联合预训练。
- TestOnly：只加载已有微调 checkpoint 做测试。
- ForceCpu：强制 CPU。

脚本会在微调前打印预训练 best checkpoint 来源，默认路径为 `pretrained_weights/contrastive_best.pth`（由 `JointOutputRoot/contrastive/checkpoints/best.pth` 同步而来）。如果路径不存在，会显式提示当前微调将不加载对比学习权重。

计时日志已统一为 fold/stage 级：

- 预训练阶段在 `contrastive/final_metrics.json` 记录 `fold_elapsed_seconds`。
- 微调阶段在每个 fold 结束打印 `fold_elapsed=...s`，并在每折 `final_metrics.json` 记录 `fold_elapsed_seconds`。

### 示例 1：只用 ds002336 预训练，并在 ds002336 上微调

```powershell
.\scripts\run_pretrain_and_finetune.ps1 -PretrainDatasets ds002336 -TargetDataset ds002336
```

### 示例 2：用 ds002336 和 ds002739 联合预训练，再在 ds002739 上微调

```powershell
.\scripts\run_pretrain_and_finetune.ps1 -PretrainDatasets ds002336,ds002739 -TargetDataset ds002739
```

### 示例 2b：先追加 ds002338 到 joint cache，再在 ds002338 上微调

```powershell
.\scripts\run_pretrain_and_finetune.ps1 -PretrainDatasets ds002338 -TargetDataset ds002338
```

### 示例 3：跳过预训练，只复用已有 joint checkpoint 做 ds002336 微调

```powershell
.\scripts\run_pretrain_and_finetune.ps1 -SkipPretrain -TargetDataset ds002336
```

## 低层入口

如果你只想单独运行某一步，也可以直接调用低层脚本。

### 联合预训练数据预处理

```powershell
.\scripts\prepare_joint_contrastive.ps1
```

```bash
./scripts_linux/prepare_joint_contrastive.sh
```

输出目录默认是 cache/joint_contrastive，关键文件包括：

- manifest_all.csv
- run_summary.csv
- subject_mapping.csv
- eeg_channels_dataset.csv
- eeg_channels_target.csv
- eeg_channel_mapping.csv

常用增量参数：

- `-SkipExistingDatasets $true`：默认启用，已存在数据集跳过重算。
- `-ForceRefreshDatasets ds002336`：仅重算指定数据集。

### 单数据集微调预处理

ds002336 / ds002338（通用 233x 入口）：

```powershell
.\scripts\ds00233x\prepare_ds00233x.ps1 -DatasetName ds002336 -DsRoot ..\ds002336 -OutputRoot cache\ds002336
.\scripts\ds00233x\prepare_ds00233x.ps1 -DatasetName ds002338 -DsRoot ..\ds002338 -OutputRoot cache\ds002338
```

```bash
./scripts_linux/ds00233x/prepare_ds00233x.sh --dataset-name ds002336 --ds-root ../ds002336 --output-root cache/ds002336
./scripts_linux/ds00233x/prepare_ds00233x.sh --dataset-name ds002338 --ds-root ../ds002338 --output-root cache/ds002338
```

ds002336 / ds002338（先跑 MATLAB SPM，再跑 Python 预处理）：

```powershell
.\scripts\ds00233x\prepare_ds00233x_spm.ps1 -DatasetName ds002336 -DsRoot ..\ds002336 -OutputRoot cache\ds002336 -ParallelWorkers 4
.\scripts\ds00233x\prepare_ds00233x_spm.ps1 -DatasetName ds002338 -DsRoot ..\ds002338 -OutputRoot cache\ds002338 -ParallelWorkers 4
```

```bash
./scripts_linux/ds00233x/prepare_ds00233x_spm.sh --dataset-name ds002336 --ds-root ../ds002336 --output-root cache/ds002336 --parallel-workers 4
./scripts_linux/ds00233x/prepare_ds00233x_spm.sh --dataset-name ds002338 --ds-root ../ds002338 --output-root cache/ds002338 --parallel-workers 4
```

`prepare_ds00233x_spm.ps1` 的 `-ParallelWorkers` 可直接控制 MATLAB 并行核心数，会调用 [preprocess/run_spm_preproc_ds00233x.m](preprocess/run_spm_preproc_ds00233x.m)。

ds002739：

```powershell
.\scripts\ds002739\prepare_ds002739.ps1
```

```bash
./scripts_linux/ds002739/prepare_ds002739.sh
```

如果目标数据集微调要复用联合预训练的 backbone，建议把 joint 导出的 eeg_channels_target.csv 传给对应 prepare 脚本的 TargetChannelManifest。

## 训练入口

- [run_train.py](run_train.py) 只用于联合对比学习预训练，只接受一个 manifest。
- [run_finetune.py](run_finetune.py) 只用于单目标数据集微调，仍然接受 train、val、test 三个 manifest。

## 当前配置约定

### 联合预训练

配置文件： [configs/train_joint_contrastive.yaml](configs/train_joint_contrastive.yaml)

- data.manifest_csv 指向 joint manifest_all.csv。
- 预训练阶段不再使用 val_manifest_csv 或 test_manifest_csv。
- contrastive 训练不再执行验证逻辑，checkpoint 仅按 train loss 选择。
- EEG/fMRI 编码器是否加载已有权重由 `eeg_model.checkpoint_path` 与 `fmri_model.checkpoint_path` 控制。
  - 为空字符串：随机初始化。
  - 非空：在构建模型时加载对应 checkpoint（路径相对项目根目录解析）。

代码入口（联合预训练读取权重的位置）：

- [mmcontrast/contrastive_trainer.py](mmcontrast/contrastive_trainer.py)：`build_model()` 会把 `eeg_model.checkpoint_path`/`fmri_model.checkpoint_path` 传入编码器。
- [mmcontrast/models/eeg_adapter.py](mmcontrast/models/eeg_adapter.py)：`EEGCBraModAdapter` 在 `checkpoint_path` 非空时加载权重。
- [mmcontrast/models/fmri_adapter.py](mmcontrast/models/fmri_adapter.py)：`FMRINeuroSTORMAdapter` 在 `checkpoint_path` 非空时加载权重。

实操示例（让联合预训练先加载 EEG+fMRI 预训练权重）：

```powershell
python .\run_train.py --config configs\train_joint_contrastive.yaml `
  --set eeg_model.checkpoint_path='pretrained_weights/CBraMod_pretrained_weights.pth' `
  --set fmri_model.checkpoint_path='pretrained_weights/pt_neurostorm_mae_ratio0.5.ckpt'
```

### 单数据集微调

配置文件：

- [configs/finetune_ds002336.yaml](configs/finetune_ds002336.yaml)
- [configs/finetune_ds002338.yaml](configs/finetune_ds002338.yaml)
- [configs/finetune_ds002739.yaml](configs/finetune_ds002739.yaml)

这些配置仍然保留 LOSO 微调所需的 train、val、test manifest 结构。

### EEG Baseline 权重策略

`finetune.eeg_baseline` 现已支持以下策略：

- `category=traditional`：始终随机初始化，不加载预训练权重。
- `category=foundation`：通过 `load_pretrained_weights` 控制是否加载权重。
  - `true`：按模型类型加载（例如 cbramod 从 `finetune.contrastive_checkpoint_path` 提取 EEG encoder）。
  - `false`：强制随机初始化。

新增 LaBraM baseline 选项：

- `model_name: labram`
- `labram_model_name: labram_base_patch200_200`（可改 large/huge）
- `labram_checkpoint_path: pretrained_weights/labram-base.pth`

注意：当前 LaBraM baseline 需要输入 EEG 通道数为 62。

### 新增 Baseline 的改动清单

如果你要增加一个新的 baseline（把 LaBraM 当作 baseline 就属于这类场景），建议按下面的分层改：

1. 模型定义放哪里
- 通用骨干定义：放在 [mmcontrast/backbones](mmcontrast/backbones) 下，建议按模态建子目录（例如 `eeg_xxx`）。
- 训练侧适配器（统一输入输出、加载 checkpoint）：放在 [mmcontrast/models](mmcontrast/models)（例如 [mmcontrast/models/eeg_labram_adapter.py](mmcontrast/models/eeg_labram_adapter.py)）。
- baseline 路由与策略（traditional/foundation、是否加载预训练）：放在 [mmcontrast/baselines/eeg_baseline.py](mmcontrast/baselines/eeg_baseline.py)。

2. 配置校验必须同步
- 在 [mmcontrast/config.py](mmcontrast/config.py) 的 baseline 白名单里注册 `model_name`，否则配置会在训练前被拦截。
- 如果 baseline 需要专用 checkpoint（如 LaBraM），在这里补充路径存在性校验。

3. 配置文件与入口参数
- 在 [configs/finetune_ds002336.yaml](configs/finetune_ds002336.yaml)、[configs/finetune_ds002338.yaml](configs/finetune_ds002338.yaml)、[configs/finetune_ds002739.yaml](configs/finetune_ds002739.yaml) 增加/同步 baseline 字段默认值。
- 如果需要命令行覆盖项，补充 [run_finetune.py](run_finetune.py) 的参数映射。

4. 最低验证
- 先跑 `python -m py_compile` 做语法校验。
- 再用单 fold 小 epoch（或 `optuna --dry-run`）验证配置链路确实命中你的 baseline。

### 微调阶段如何控制是否读取预训练权重

代码入口（微调读取权重的位置）：

- [mmcontrast/finetune_trainer.py](mmcontrast/finetune_trainer.py)：初始化时解析并写入 `finetune.contrastive_checkpoint_path`。
- [mmcontrast/models/classifier.py](mmcontrast/models/classifier.py)：
  - `finetune.contrastive_checkpoint_path` 非空时，加载整套对比学习 backbone（EEG+fMRI）参数。
  - 若为空但 `eeg_model.checkpoint_path`/`fmri_model.checkpoint_path` 有值，则分别走模态级 checkpoint。
- [mmcontrast/baselines/eeg_baseline.py](mmcontrast/baselines/eeg_baseline.py)：`finetune.eeg_baseline.load_pretrained_weights` 控制 baseline 是否加载预训练权重（含 LaBraM）。

控制方式：

- 默认主流程 [scripts/run_pretrain_and_finetune.ps1](scripts/run_pretrain_and_finetune.ps1) 会把联合预训练 best checkpoint 同步到 `pretrained_weights/contrastive_best.pth`，并在微调时自动传给 `--contrastive-checkpoint`。
- 禁用“加载对比学习权重”：把 `finetune.contrastive_checkpoint_path` 置空（或不要传 `--contrastive-checkpoint`）。
- 使用 EEG baseline 且禁用其预训练加载：设置 `finetune.eeg_baseline.load_pretrained_weights: false`。
- 使用 LaBraM baseline：设置 `model_name: labram`，并将 `labram_checkpoint_path` 指向 `pretrained_weights/labram-base.pth`。

## EEG Baseline 模型

微调阶段支持使用 7 个 EEG 基线模型，分为两类：

### 基础模型（Foundation Models）

基础模型需要额外的分类头，支持加载预训练权重：

1. **LaBraM** - 大规模 EEG 基础模型
2. **CBraMod** - 对比学习预训练的 EEG 骨干

### 传统模型（Traditional Models）

传统模型是端到端的分类模型，内置分类头，不需要额外分类头：

3. **SVM** - 支持向量机（传统机器学习基线）
4. **EEG-Deformer** - Deformer 架构（CNN + Transformer + DIP 融合）
5. **EEGNet** - 经典 EEG 深度学习方法
6. **Conformer** - CNN + Transformer 混合架构
7. **TSception** - 多尺度时空卷积网络

所有模型定义均严格遵循官方实现，不从外部模型文件导入。

### 配置方法

在微调配置文件中设置 `finetune.eeg_baseline`：

```yaml
finetune:
  eeg_baseline:
    enabled: true
    model_name: eegnet  # 7 个选项：svm, labram, cbramod, eeg_deformer, eegnet, conformer, tsception
    num_classes: 2
    num_channels: 62
    num_timepoints: 200
    load_pretrained_weights: false  # 基础模型设为 true 可加载预训练权重
    checkpoint_path: ""  # 基础模型预训练权重路径
```

### 模型类别与融合策略

- **传统模型**（svm, eeg_deformer, eegnet, conformer, tsception）：
  - 内置分类头，直接输出 logits
  - 仅支持 `finetune.fusion: eeg_only`
  - 不需要额外分类头

- **基础模型**（labram, cbramod）：
  - 输出特征向量，需要额外分类头
  - 支持所有融合策略（eeg_only, fmri_only, concat）
  - 可通过 `load_pretrained_weights: true` 加载预训练权重

### 使用示例

使用 EEGNet 作为基线（传统模型）：

```powershell
python .\run_finetune.py --config configs\finetune_ds002336.yaml `
  --set finetune.eeg_baseline.enabled=true `
  --set finetune.eeg_baseline.model_name=eegnet `
  --set finetune.fusion=eeg_only
```

使用 LaBraM 作为基线（基础模型，加载预训练权重）：

```powershell
python .\run_finetune.py --config configs\finetune_ds002336.yaml `
  --set finetune.eeg_baseline.enabled=true `
  --set finetune.eeg_baseline.model_name=labram `
  --set finetune.eeg_baseline.load_pretrained_weights=true `
  --set finetune.eeg_baseline.checkpoint_path=pretrained_weights/labram-base.pth `
  --set finetune.fusion=eeg_only
```

### 添加新基线模型

如需添加新的基线模型，需要修改以下文件：

1. **模型定义**：[mmcontrast/baselines/eeg_baseline.py](mmcontrast/baselines/eeg_baseline.py)
   - 添加模型类（严格遵循官方实现）
   - 在 `VALID_MODEL_NAMES` 中注册
   - 在 `MODEL_CATEGORIES` 中指定类别

2. **配置校验**：[mmcontrast/config.py](mmcontrast/config.py)
   - 在 baseline 白名单中注册 `model_name`

3. **配置文件**：更新所有 `finetune_*.yaml` 的默认值

4. **验证**：
   ```powershell
   python -m py_compile mmcontrast/baselines/eeg_baseline.py
   python -c "from mmcontrast.baselines import EEGBaselineModel; m = EEGBaselineModel(model_name='new_model', num_classes=2, num_channels=62, num_timepoints=200)"
   ```

## Optuna

Optuna 现在仍然保留，但已经切换到当前的新主流程，不再依赖旧的单数据集 contrastive 运行脚本。

Optuna 现在仍然保留，但已经切换到当前的新主流程，不再依赖旧的单数据集 contrastive 运行脚本。

你可以先用 dry-run 快速确认 study 配置是否真实被执行链路消费：

```powershell
python .\run_optuna_search.py --study-config configs\optuna_ds002336.yaml --mode full --dry-run
python .\run_optuna_search.py --study-config configs\optuna_ds002739.yaml --mode full --dry-run
```

### 入口

- [run_optuna_search.py](run_optuna_search.py)：通用 Optuna 搜索入口。
- [scripts/run_optuna_pretrain_and_finetune.ps1](scripts/run_optuna_pretrain_and_finetune.ps1)：把单个 trial 的输出目录映射到当前唯一主流程的包装脚本。

### Study 配置

- [configs/optuna_ds002336.yaml](configs/optuna_ds002336.yaml)
- [configs/optuna_ds002338.yaml](configs/optuna_ds002338.yaml)
- [configs/optuna_ds002739.yaml](configs/optuna_ds002739.yaml)

Linux 请使用对应副本（避免 `powershell` 命令缺失）：

- [configs/optuna_ds002336_linux.yaml](configs/optuna_ds002336_linux.yaml)
- [configs/optuna_ds002338_linux.yaml](configs/optuna_ds002338_linux.yaml)
- [configs/optuna_ds002739_linux.yaml](configs/optuna_ds002739_linux.yaml)

这三个配置已经适配当前工作流：

- full：联合预训练 + 单目标数据集微调。
- finetune_only：跳过预训练，只做目标数据集微调。
- pretrain_only：跳过微调，只做联合预训练。

`pretrain_only` 下，真正参与搜索与执行的是：

- `runtime_configs.train_base`（默认 `configs/train_joint_contrastive.yaml`）
- mode 的 `parameter_groups` 里属于 `pretrain` 的参数
- `study.static_args` 里的 `-PretrainDatasets`

因此，不同 `optuna_ds*.yaml` 在 `pretrain_only` 模式下是否“结果一样”，取决于这三部分是否完全一致。
如果三者一致，则搜索空间和目标一致，结果分布应接近；如果有任一项不同（例如 `-PretrainDatasets` 或参数范围不同），结果就会不同。

并且现在支持 `parameter_groups`，用于按阶段组织搜索参数：

- `pretrain`：只作用于 `train_joint_contrastive.yaml`。
- `finetune`：只作用于 `finetune_*.yaml`。

`modes.<mode>.parameter_groups` 会决定该模式下启用哪些参数组。
如果某个 mode 通过 `-SkipPretrain` 或 `-SkipFinetune` 关闭了某个阶段，`run_optuna_search.py` 会自动只加载激活阶段的 runtime config，未激活阶段的参数更新会被自动忽略，不会再出现“LOSO 微调模式却混入对比学习参数”的问题。

### 当前保留的搜索参数

参数选择已经按你之前那套恢复进来了：

- train_epochs
- pretrain_batch_size
- finetune_batch_size
- pretrain_lr
- finetune_lr
- weight_decay
- min_lr
- hidden_dim
- grad_clip
- early_stop_patience

其中 ds002336 和 ds002739 各自的候选值也保留了之前的区别，例如 pretrain/finetune 的 batch size、train_epochs 和 early_stop_patience 的搜索范围仍然沿用原来的两套配置。

### parameter_names / parameter_groups 含义

- `parameter_names`：直接列出当前 mode 启用的参数名。
- `parameter_groups`：先在顶层定义参数组（如 `pretrain`、`finetune`），再在 mode 里引用分组。

推荐优先使用 `parameter_groups`，结构更清晰，也更适合 full / finetune_only / pretrain_only 三种模式复用。

### 使用示例

搜索 ds002336 的完整流程：

```powershell
python .\run_optuna_search.py --study-config configs\optuna_ds002336.yaml --mode full
```

只搜索 ds002739 的微调阶段：

```powershell
python .\run_optuna_search.py --study-config configs\optuna_ds002739.yaml --mode finetune_only
```

只搜索联合预训练阶段：

```powershell
python .\run_optuna_search.py --study-config configs\optuna_ds002336.yaml --mode pretrain_only
```

你也可以对任意目标 study 配置跑 `pretrain_only`，它会统一走联合对比学习评估指标：

```powershell
python .\run_optuna_search.py --study-config configs\optuna_ds002739.yaml --mode pretrain_only
python .\run_optuna_search.py --study-config configs\optuna_ds002338.yaml --mode pretrain_only
```

Linux 示例：

```bash
python ./run_optuna_search.py --study-config configs/optuna_ds002338_linux.yaml --mode full
python ./run_optuna_search.py --study-config configs/optuna_ds002338_linux.yaml --mode pretrain_only
```

### Optuna 每个 trial 使用多 GPU（trial 串行）

当前 `run_optuna_search.py` 采用串行 trial（不会并发启动多个 trial）。

- 也就是说：一次只跑一个 trial。
- 当你设置 `--gpu-count > 1` 时，是“当前 trial 内部”用 DDP 多卡训练。
- 适用于 full / finetune_only / pretrain_only 三种 mode。

Windows 示例（每个 trial 使用 2 张卡）：

```powershell
python .\run_optuna_search.py --study-config configs\optuna_ds002336.yaml --mode full --gpu-count 2
```

Windows 指定物理卡（例如仅用 0,1）：

```powershell
python .\run_optuna_search.py --study-config configs\optuna_ds002336.yaml --mode finetune_only --gpu-count 2 --gpu-ids 0,1
```

Linux 示例：

```bash
python ./run_optuna_search.py --study-config configs/optuna_ds002338_linux.yaml --mode full --gpu-count 2 --gpu-ids 0,1
```

实现方式：

- Optuna wrapper 会把 `gpu-count/gpu-ids` 透传到主流程脚本。
- 主流程在每个 trial 内使用 `python -m torch.distributed.run --nproc_per_node=<gpu_count>` 启动 `run_train.py` / `run_finetune.py`。
- 当 `--gpu-count=1` 或启用 `--force-cpu` 时，自动回退为单进程运行。

### 多 GPU 联合预训练示例

如果只想跳过主入口脚本，直接对已经准备好的 joint cache 启动联合对比学习预训练，也可以直接调用 `run_train.py`。

单卡：
```bash
python run_train.py \
  --config configs/train_joint_contrastive.yaml \
  --manifest cache/joint_contrastive/manifest_all.csv \
  --root-dir cache/joint_contrastive \
  --output-dir outputs/joint_contrastive/contrastive
```

多卡：
```bash
torchrun --nproc_per_node=2 run_train.py \
  --config configs/train_joint_contrastive.yaml \
  --manifest cache/joint_contrastive/manifest_all.csv \
  --root-dir cache/joint_contrastive \
  --output-dir outputs/joint_contrastive/contrastive
```