# EEG-fMRI-Contrastive

## 环境约定

运行任何脚本前，请先手动激活你自己的 conda/mamba 环境。仓库内脚本不会强行写死环境名。

```bash
conda activate <your-mamba-env>
```

Windows PowerShell 同理：

```powershell
conda activate <your-mamba-env>
```

本仓库用于 EEG-fMRI 联合对比学习预训练，以及 EEG-only 分类微调。

当前支持两类任务：

1. EEG-fMRI 联合预训练
- 适用数据集：`ds002336`、`ds002338`、`ds002739`
- 使用 EEG encoder 和 fMRI encoder 做跨模态对比学习
- 当前预训练结构包含 `EEG shared head`、`EEG private head`、`fMRI shared head`
- 预训练损失由 `InfoNCE(shared)`、`band-power regression(private)`、`shared/private separation loss` 组成

2. EEG-only 微调分类
- 适用数据集：`ds002336`、`ds002338`、`ds002739`、`ds009999(SEED)`
- 支持 `classifier_mode=shared|private|concat`
- 默认 `classifier_mode=concat`
- 支持 `finetune.eeg_baseline` 切换到 EEG baseline
- 当前支持的 baseline：
  - traditional：`svm`、`eeg_deformer`、`eegnet`、`conformer`、`tsception`
  - foundation：`labram`、`cbramod`
- 微调模型按 `macro_f1` 选择最佳 checkpoint
- 分类评估默认输出 `accuracy`、`macro_f1`

`ds009999` 是仓库内部对 `SEED` 的命名，只用于 EEG-only 微调，不参与 EEG-fMRI 联合预训练。

## 仓库主要功能

### 1. 预处理

支持把原始数据预处理为训练可直接读取的缓存格式，包括：

- EEG 预处理
- fMRI 预处理
- subject pack 导出
- LOSO 划分
- EEG 公共通道映射
- EEG band-power 离线目标生成

band-power 的规则如下：

- 基于预处理后的 EEG 计算，不在训练时在线计算
- 当前仓库 EEG 预处理后的采样率是 `200 Hz`
- 默认 8 秒窗，对应 `1600` 点
- 输出 shape 为 `[N, 5]`
- 五个频带为：
  - `delta`: `0.5-4`
  - `theta`: `4-8`
  - `alpha`: `8-13`
  - `beta`: `13-30`
  - `gamma`: `30-40`

### 2. 训练

支持：

- EEG-fMRI 联合预训练
- 单 fold 微调
- LOSO 全折微调
- Optuna 自动搜索

### 3. 离线可视化

支持预训练后离线导出：

- `EEG shared / EEG private / fMRI shared` 的 t-SNE 图
- 跨模态相似度热力图

输出文件名会自动递增，例如：

- `tsne_shared_private_001.png`
- `cross_modal_similarity_heatmap_001.png`
- `visualization_summary_001.json`

## 目录约定

推荐目录结构：

```text
OpenNeuro/
  EEG-fMRI-Contrastive/
  ds002336/
  ds002338/
  ds002739/
  SEED/
```

说明：

- `ds002336`、`ds002338`、`ds002739` 用于联合预训练和微调
- `SEED` 用于 `ds009999` EEG-only 微调

## 关键配置说明

### 1. 预训练损失权重

在 [train_joint_contrastive.yaml](/d:/OpenNeuro/EEG-fMRI-Contrastive/configs/train_joint_contrastive.yaml) 的 `train` 段中设置：

```yaml
train:
  band_power_loss_weight: 1.0
  separation_loss_weight: 0.1
```

### 2. 头部 dropout

`shared/private` 头的 dropout 用 `head_dropout` 控制：

```yaml
train:
  head_dropout: 0.0
```

或者微调时：

```yaml
finetune:
  head_dropout: 0.0
```

最后分类器 MLP 的 dropout 用：

```yaml
finetune:
  dropout: 0.2
```

### 3. 微调模式

```yaml
finetune:
  fusion: eeg_only
  classifier_mode: concat
```

支持：

- `shared`
- `private`
- `concat`

### 4. 无 contrastive checkpoint 时的微调 fallback

如果：

- 没有设置 `finetune.contrastive_checkpoint_path`
- 但设置了 `eeg_model.checkpoint_path`

则当前代码会直接使用 pretrained EEG backbone 特征做分类，不会强行使用未初始化的 `shared/private` 头。

### 5. EEG baseline 配置

最小配置示例：

```yaml
finetune:
  fusion: eeg_only
  eeg_baseline:
    enabled: true
    model_name: eegnet
    load_pretrained_weights: false
    checkpoint_path: ""
```

说明：

- `traditional` baseline 只支持 `finetune.fusion=eeg_only`
- `svm` 不支持 DDP，建议单卡或 `--force-cpu`
- `foundation` baseline 中，`labram`、`cbramod` 可通过 `finetune.eeg_baseline.checkpoint_path` 加载预训练权重
- 当 `finetune.eeg_baseline.enabled=true` 时，常规的 contrastive checkpoint 不再是主路径；传统 baseline 会直接走自身分类分支
- `labram` 依赖 `timm`，如果缺失需要额外安装

## Windows 操作指南

### 1. 环境安装

```powershell
conda activate mamba
cd D:\OpenNeuro\EEG-fMRI-Contrastive
python -m pip install -r requirements.txt
```

### 2. 预处理 ds002336、ds002338、ds002739 的联合预训练缓存

```powershell
.\scripts\prepare_joint_contrastive.ps1 -Ds002336Root ..\ds002336 -Ds002338Root ..\ds002338 -Ds002739Root ..\ds002739 -OutputRoot cache\joint_contrastive -Datasets ds002336,ds002338,ds002739 -EegWindowSec 8.0 -NumWorkers 2
```

### 3. 仅补 band-power 目标

```powershell
python preprocess\compute_eeg_band_power_targets.py --manifest-csv cache\joint_contrastive\manifest_all.csv --root-dir cache\joint_contrastive --sample-rate-hz 200 --window-sec 8 --overwrite
```

### 4. 预处理 ds002336

```powershell
.\scripts\ds00233x\prepare_ds00233x.ps1 -DatasetName ds002336 -DsRoot ..\ds002336 -OutputRoot cache\ds002336
```

### 5. 预处理 ds002338

```powershell
.\scripts\ds00233x\prepare_ds00233x.ps1 -DatasetName ds002338 -DsRoot ..\ds002338 -OutputRoot cache\ds002338
```

### 6. 预处理 ds002739

```powershell
.\scripts\ds002739\prepare_ds002739.ps1 -DsRoot ..\ds002739 -OutputRoot cache\ds002739
```

### 7. 预处理 SEED 为 ds009999

默认会先尝试读取 `SEED\label.mat`，如果不存在，再尝试 `SEED\Preprocessed_EEG\label.mat`，并递归搜索其余 `.mat` EEG 文件：

```powershell
.\scripts\ds009999\prepare_ds009999.ps1 -DsRoot ..\SEED -OutputRoot cache\ds009999 -SplitMode loso -WindowSec 8.0 -EegTargetSfreq 200.0
```

如果 `label.mat` 不在默认位置：

```powershell
.\scripts\ds009999\prepare_ds009999.ps1 -DsRoot ..\SEED -LabelsMat ..\SEED\label.mat -OutputRoot cache\ds009999
```

### 8. 联合预训练

```powershell
python run_train.py --config configs\train_joint_contrastive.yaml
```

### 9. 单 fold 微调 ds002338

```powershell
python run_finetune.py --config configs\finetune_ds002338.yaml
```

### 10. 单 fold 微调 ds009999

```powershell
python run_finetune.py --config configs\finetune_ds009999.yaml
```

### 11. 运行单个 EEG baseline

例如，在 `ds009999` 上运行 `EEGNet`：

```powershell
python run_finetune.py --config configs\finetune_ds009999.yaml --set "finetune.fusion='eeg_only'" --set "finetune.contrastive_checkpoint_path=''" --set "finetune.eeg_baseline.enabled=true" --set "finetune.eeg_baseline.model_name='eegnet'" --set "finetune.eeg_baseline.load_pretrained_weights=false" --set "finetune.eeg_baseline.checkpoint_path=''"
```

例如，在 `ds009999` 上运行 `SVM`：

```powershell
python run_finetune.py --config configs\finetune_ds009999.yaml --force-cpu --set "finetune.fusion='eeg_only'" --set "finetune.contrastive_checkpoint_path=''" --set "finetune.eeg_baseline.enabled=true" --set "finetune.eeg_baseline.model_name='svm'" --set "finetune.eeg_baseline.load_pretrained_weights=false" --set "finetune.eeg_baseline.checkpoint_path=''"
```

如果要运行 foundation baseline，例如 `LaBraM`，可以再补：

```powershell
--set "finetune.eeg_baseline.model_name='labram'" --set "finetune.eeg_baseline.load_pretrained_weights=true" --set "finetune.eeg_baseline.checkpoint_path='pretrained_weights\\labram.pth'"
```

### 12. LOSO 全折微调 ds009999

先预处理生成 `cache\ds009999\loso_subjectwise\fold_*`，然后对每个 fold 调用 `run_finetune.py`。单个 fold 的调用格式如下：

```powershell
python run_finetune.py --config configs\finetune_ds009999.yaml --train-manifest cache\ds009999\loso_subjectwise\fold_ds009999_sub01\manifest_train.csv --val-manifest cache\ds009999\loso_subjectwise\fold_ds009999_sub01\manifest_val.csv --test-manifest cache\ds009999\loso_subjectwise\fold_ds009999_sub01\manifest_test.csv --root-dir cache\ds009999 --output-dir outputs\ds009999\finetune\fold_ds009999_sub01
```

### 13. 混淆矩阵

`run_finetune.py` 只要拿到了 `test_manifest`，测试阶段就会自动输出混淆矩阵，不需要额外参数。最小用法：

```powershell
python run_finetune.py --config configs\finetune_ds002338.yaml
```

如果要对已有 LOSO 权重直接做离线复现，并输出每个 fold 和总 LOSO 的混淆矩阵，可以运行：

```powershell
python run_offline_loso_eval.py --dataset-name ds009999 --config configs\finetune_ds009999.yaml --checkpoints-root outputs\ds009999\finetune --output-dir outputs\ds009999\offline_eval
```

要求 `--checkpoints-root` 下存在 `fold_*\checkpoints\best.pth`。

### 14. Optuna 搜索 ds002338

```powershell
python run_optuna_search.py --study-config configs\optuna_ds002338.yaml --mode full
```

### 15. Optuna 搜索 ds009999

`ds009999` 只支持 finetune-only：

```powershell
python run_optuna_search.py --study-config configs\optuna_ds009999.yaml --mode finetune_only
```

### 16. 预训练结果离线可视化

```powershell
python .\run_visualize_contrastive.py --config configs\train_joint_contrastive.yaml --checkpoint outputs\joint_contrastive\contrastive\checkpoints\best.pth --output-dir outputs\visualizations\contrastive --batch-size 32 --tsne-max-points 2000 --heatmap-max-points 128
```

## Linux 操作指南

### 1. 环境安装

```bash
conda activate mamba
cd /path/to/EEG-fMRI-Contrastive
python -m pip install -r requirements.txt
chmod +x scripts_linux/*.sh scripts_linux/ds00233x/*.sh scripts_linux/ds002739/*.sh scripts_linux/ds009999/*.sh
```

### 2. 预处理 ds002336、ds002338、ds002739 的联合预训练缓存

```bash
./scripts_linux/prepare_joint_contrastive.sh --ds002336-root ../ds002336 --ds002338-root ../ds002338 --ds002739-root ../ds002739 --output-root cache/joint_contrastive --datasets ds002336,ds002338,ds002739 --eeg-window-sec 8.0 --num-workers 2
```

### 3. 仅补 band-power 目标

```bash
python preprocess/compute_eeg_band_power_targets.py --manifest-csv cache/joint_contrastive/manifest_all.csv --root-dir cache/joint_contrastive --sample-rate-hz 200 --window-sec 8 --overwrite
```

### 4. 预处理 ds002336

```bash
./scripts_linux/ds00233x/prepare_ds00233x.sh --dataset-name ds002336 --ds-root ../ds002336 --output-root cache/ds002336
```

### 5. 预处理 ds002338

```bash
./scripts_linux/ds00233x/prepare_ds00233x.sh --dataset-name ds002338 --ds-root ../ds002338 --output-root cache/ds002338
```

### 6. 预处理 ds002739

```bash
./scripts_linux/ds002739/prepare_ds002739.sh --ds-root ../ds002739 --output-root cache/ds002739
```

### 7. 预处理 SEED 为 ds009999

```bash
./scripts_linux/ds009999/prepare_ds009999.sh --ds-root ../SEED --output-root cache/ds009999 --split-mode loso --window-sec 8.0 --eeg-target-sfreq 200.0
```

如果 `label.mat` 不在默认位置：

```bash
./scripts_linux/ds009999/prepare_ds009999.sh --ds-root ../SEED --labels-mat ../SEED/label.mat --output-root cache/ds009999
```

### 8. 联合预训练

```bash
python run_train.py --config configs/train_joint_contrastive.yaml
```

### 9. 单 fold 微调 ds002338

```bash
python run_finetune.py --config configs/finetune_ds002338.yaml
```

### 10. 单 fold 微调 ds009999

```bash
python run_finetune.py --config configs/finetune_ds009999.yaml
```

### 11. 运行 EEG baseline

例如，在 `ds009999` 上运行 `EEGNet`：

```bash
python run_finetune.py --config configs/finetune_ds009999.yaml --set "finetune.fusion='eeg_only'" --set "finetune.contrastive_checkpoint_path=''" --set "finetune.eeg_baseline.enabled=true" --set "finetune.eeg_baseline.model_name='eegnet'" --set "finetune.eeg_baseline.load_pretrained_weights=false" --set "finetune.eeg_baseline.checkpoint_path=''"
```

例如，在 `ds009999` 上运行 `SVM`：

```bash
python run_finetune.py --config configs/finetune_ds009999.yaml --force-cpu --set "finetune.fusion='eeg_only'" --set "finetune.contrastive_checkpoint_path=''" --set "finetune.eeg_baseline.enabled=true" --set "finetune.eeg_baseline.model_name='svm'" --set "finetune.eeg_baseline.load_pretrained_weights=false" --set "finetune.eeg_baseline.checkpoint_path=''"
```

例如，在 `ds002336` 上批量跑全部 LOSO EEG baselines：

```bash
./scripts_linux/run_all_eeg_baselines.sh --config configs/finetune_ds002336.yaml --models svm,labram,cbramod,eeg_deformer,eegnet,conformer,tsception --output-root outputs/finetune_ds002336_all_baselines
```

如果要运行 foundation baseline，例如 `LaBraM`，可以再补：

```bash
--set "finetune.eeg_baseline.model_name='labram'" --set "finetune.eeg_baseline.load_pretrained_weights=true" --set "finetune.eeg_baseline.checkpoint_path='pretrained_weights/labram.pth'"
```

### 12. LOSO 单 fold 微调 ds009999

```bash
python run_finetune.py --config configs/finetune_ds009999.yaml --train-manifest cache/ds009999/loso_subjectwise/fold_ds009999_sub01/manifest_train.csv --val-manifest cache/ds009999/loso_subjectwise/fold_ds009999_sub01/manifest_val.csv --test-manifest cache/ds009999/loso_subjectwise/fold_ds009999_sub01/manifest_test.csv --root-dir cache/ds009999 --output-dir outputs/ds009999/finetune/fold_ds009999_sub01
```

### 13. 混淆矩阵

`run_finetune.py` 只要拿到了 `test_manifest`，测试阶段就会自动输出混淆矩阵，不需要额外参数。最小用法：

```bash
python run_finetune.py --config configs/finetune_ds002338.yaml
```

如果要对已有 LOSO 权重直接做离线复现，并输出每个 fold 和总 LOSO 的混淆矩阵，可以运行：

```bash
python run_offline_loso_eval.py --dataset-name ds009999 --config configs/finetune_ds009999.yaml --checkpoints-root outputs/ds009999/finetune --output-dir outputs/ds009999/offline_eval
```

要求 `--checkpoints-root` 下存在 `fold_*/checkpoints/best.pth`。

### 14. Optuna 搜索 ds002338

```bash
python run_optuna_search.py --study-config configs/optuna_ds002338_linux.yaml --mode full
```

### 15. Optuna 搜索 ds009999

```bash
python run_optuna_search.py --study-config configs/optuna_ds009999_linux.yaml --mode finetune_only
```

### 16. 预训练结果离线可视化

```bash
python run_visualize_contrastive.py --config configs/train_joint_contrastive.yaml --checkpoint outputs/joint_contrastive/contrastive/checkpoints/best.pth --output-dir outputs/visualizations/contrastive --batch-size 32 --tsne-max-points 2000 --heatmap-max-points 128
```

## 脚本索引与用法

下面只列脚本入口。执行前请先手动 `conda activate <your-mamba-env>`。

### 1. 核心 Python 入口

`run_train.py`

```bash
python run_train.py --config configs/train_joint_contrastive.yaml
```

`run_finetune.py`

```bash
```

`run_optuna_search.py`

```bash
```

`run_visualize_contrastive.py`

```bash
python run_visualize_contrastive.py --config configs/train_joint_contrastive.yaml --checkpoint outputs/joint_contrastive/contrastive/checkpoints/best.pth --output-dir outputs/visualizations/contrastive
```

`run_offline_loso_eval.py`

```bash
```

### 2. Linux 脚本

`scripts_linux/prepare_joint_contrastive.sh`

```bash
./scripts_linux/prepare_joint_contrastive.sh --ds002336-root ../ds002336 --ds002338-root ../ds002338 --ds002739-root ../ds002739 --output-root cache/joint_contrastive
```

`scripts_linux/ds00233x/prepare_ds00233x.sh`

```bash
./scripts_linux/ds00233x/prepare_ds00233x.sh --dataset-name ds002336 --ds-root ../ds002336 --output-root cache/ds002336
```

`scripts_linux/ds00233x/prepare_ds00233x_spm.sh`

```bash
./scripts_linux/ds00233x/prepare_ds00233x_spm.sh --dataset-name ds002336 --ds-root ../ds002336
```

`scripts_linux/ds002739/prepare_ds002739.sh`

```bash
./scripts_linux/ds002739/prepare_ds002739.sh --ds-root ../ds002739 --output-root cache/ds002739
```

`scripts_linux/ds009999/prepare_ds009999.sh`

```bash
./scripts_linux/ds009999/prepare_ds009999.sh --ds-root ../SEED --output-root cache/ds009999 --split-mode loso
```


```bash
```

`scripts_linux/run_pretrain_and_finetune.sh`

```bash
./scripts_linux/run_pretrain_and_finetune.sh --target-dataset ds002739 --joint-train-config configs/train_joint_contrastive.yaml --ds002739-finetune-config configs/finetune_ds002739.yaml
```

`scripts_linux/run_optuna_pretrain_and_finetune.sh`

```bash
./scripts_linux/run_optuna_pretrain_and_finetune.sh --target-dataset ds002739 --finetune-config configs/finetune_ds002739.yaml --output-root outputs/optuna_run
```

`scripts_linux/run_finetune_ds009999.sh`

```bash
./scripts_linux/run_finetune_ds009999.sh --finetune-config configs/finetune_ds009999.yaml --cache-root cache/ds009999 --output-root outputs/ds009999
```


```bash
```

`scripts_linux/run_all_eeg_baselines.sh`

```bash
./scripts_linux/run_all_eeg_baselines.sh --config configs/finetune_ds002336.yaml --models svm,labram,cbramod,eeg_deformer,eegnet,conformer,tsception --output-root outputs/finetune_ds002336_all_baselines
```

`scripts_linux/run_finetune_from_pretrain_trials.sh`

```bash
```

### 3. Windows 脚本

`scripts/prepare_joint_contrastive.ps1`

```powershell
.\scripts\prepare_joint_contrastive.ps1 -Ds002336Root ..\ds002336 -Ds002338Root ..\ds002338 -Ds002739Root ..\ds002739 -OutputRoot cache\joint_contrastive
```

`scripts/ds00233x/prepare_ds00233x.ps1`

```powershell
.\scripts\ds00233x\prepare_ds00233x.ps1 -DatasetName ds002336 -DsRoot ..\ds002336 -OutputRoot cache\ds002336
```

`scripts/ds00233x/prepare_ds00233x_spm.ps1`

```powershell
.\scripts\ds00233x\prepare_ds00233x_spm.ps1 -DatasetName ds002336 -DsRoot ..\ds002336
```

`scripts/ds002739/prepare_ds002739.ps1`

```powershell
.\scripts\ds002739\prepare_ds002739.ps1 -DsRoot ..\ds002739 -OutputRoot cache\ds002739
```

`scripts/ds009999/prepare_ds009999.ps1`

```powershell
.\scripts\ds009999\prepare_ds009999.ps1 -DsRoot ..\SEED -OutputRoot cache\ds009999 -SplitMode loso
```


```powershell
```

`scripts/run_pretrain_and_finetune.ps1`

```powershell
.\scripts\run_pretrain_and_finetune.ps1 -TargetDataset ds002739 -JointTrainConfig configs\train_joint_contrastive.yaml -Ds002739FinetuneConfig configs\finetune_ds002739.yaml
```

`scripts/run_optuna_pretrain_and_finetune.ps1`

```powershell
.\scripts\run_optuna_pretrain_and_finetune.ps1 -TargetDataset ds002739 -FinetuneConfig configs\finetune_ds002739.yaml -OutputRoot outputs\optuna_run
```

`scripts/run_finetune_ds009999.ps1`

```powershell
.\scripts\run_finetune_ds009999.ps1 -FinetuneConfig configs\finetune_ds009999.yaml -CacheRoot cache\ds009999 -OutputRoot outputs\ds009999
```


```powershell
```

### 4. Optuna 配置入口

`configs/optuna_ds002338.yaml`

```powershell
python run_optuna_search.py --study-config configs\optuna_ds002338.yaml --mode full
```

`configs/optuna_ds009999.yaml`

```powershell
python run_optuna_search.py --study-config configs\optuna_ds009999.yaml --mode finetune_only
```


```powershell
```

`configs/optuna_ds002338_linux.yaml`

```bash
python run_optuna_search.py --study-config configs/optuna_ds002338_linux.yaml --mode full
```

`configs/optuna_ds009999_linux.yaml`

```bash
python run_optuna_search.py --study-config configs/optuna_ds009999_linux.yaml --mode finetune_only
```


```bash
```

## 结果文件说明

### 1. 联合预训练

常见输出：

- `outputs/joint_contrastive/contrastive/checkpoints/best.pth`
- `outputs/joint_contrastive/contrastive/final_metrics.json`

### 2. 微调

每个 fold 常见输出：

- `checkpoints/best.pth`
- `val_metrics.json`
- `test_metrics.json`
- `test_confusion_matrix.png`
- `test_confusion_matrix.svg`
- `test_confusion_matrix.json`
- `test_logits.csv`
- `svm_summary.json`（仅 `svm` baseline）

LOSO 汇总一般写到：

- `loso_finetune_summary.csv`
- `confusion_matrix_<dataset_name>_loso.png`（离线 LOSO 总混淆矩阵）
- `confusion_matrix_<dataset_name>_loso.svg`（离线 LOSO 总混淆矩阵矢量图）
- `baseline_summary.csv`（单个 baseline 的全 fold 汇总）
- `all_baselines_summary.csv`（批量 baseline 的总汇总，Linux 脚本生成）

### 3. 可视化

输出在你指定的 `--output-dir` 下：

- `tsne_shared_private_XXX.png`
- `cross_modal_similarity_heatmap_XXX.png`
- `visualization_summary_XXX.json`

## 常见说明

### 1. `common-channel overlap=40/40` 不是输入只有 40 通道

它表示：

- 当前数据原始通道数在另一条日志里单独打印
- 公共通道集合总数是 40
- 当前数据和公共集合匹配成功了 40 个

### 2. fMRI 预训练权重 `loaded=0`

如果日志出现：

```text
loaded=0, shape_mismatch=..., missing_keys=..., unexpected_keys=...
```

通常不是“没执行加载”，而是 checkpoint 的模型配置和当前 `NeuroSTORM` 配置不一致。

### 3. SEED 不参与 joint pretraining

这是设计上的限制，不是漏接。  
因为 `SEED` 没有 fMRI，不能构造真实 EEG-fMRI 正样本配对。
