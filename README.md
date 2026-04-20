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
- 额外支持两个最简单的跨模态 baseline：
  - `Pure InfoNCE`：只保留 EEG shared encoder 和 fMRI encoder，只用 symmetric InfoNCE
  - `Barlow Twins`：只保留 EEG shared encoder 和 fMRI encoder，只用跨模态 Barlow Twins loss

2. EEG-only 微调分类
- 适用数据集：`ds002336`、`ds002338`、`ds002739`、`ds009999(SEED)`
- 支持 `classifier_mode=shared|private|concat|add`
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
- `add`

### 4. 无 contrastive checkpoint 时的微调 fallback

如果：

- 没有设置 `finetune.contrastive_checkpoint_path`
- 但设置了 `eeg_model.checkpoint_path`

则当前代码会直接使用 pretrained EEG backbone 特征做分类，不会强行使用未初始化的 `shared/private` 头。

如果希望微调阶段按离线预训练模式自动定位权重，可在配置里直接写：

```yaml
finetune:
  contrastive_checkpoint_path: ""
  pretrain_mode: strict
  pretrain_objective: contrastive
  pretrain_output_root: pretrained_weights
```

说明：

- `pretrain_mode=full|strict` 控制自动查找哪一类离线预训练权重
- `pretrain_objective` 可选 `contrastive`、`infonce`、`barlow_twins`
- 如果命令行没有手动传 `--contrastive-checkpoint`，则会按这里的设置自动定位 checkpoint
- 默认配置中 `ds002336`、`ds002338` 使用 `strict`，`ds009999` 使用 `full`

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

### 6. Cross-Modal baseline 配置

目前额外提供两个最简单的跨模态 baseline：

- `train.pretrain_objective=infonce`
- `train.pretrain_objective=barlow_twins`

它们都只保留 `EEG shared encoder + fMRI encoder`，不使用 `EEG private encoder`、`band-power loss`、`separation loss` 或其他额外损失。

如果要把这类预训练权重用于 EEG-only 微调，需要额外设置：

```yaml
finetune:
  fusion: eeg_only
  eeg_encoder_variant: shared_only
  classifier_mode: shared
```

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
python run_pretrain.py --config configs\train_joint_contrastive.yaml
```

离线预训练现在支持两种模式：

- `full`：直接使用 `cache\joint_contrastive\manifest_all.csv` 中的全部样本做联合预训练。
- `strict`：仍然直接使用现有 `cache\joint_contrastive` 缓存，不重新做预处理；只是在启动预训练时，从 `manifest_all.csv` 中排除“目标数据集中的目标测试被试”，生成一份临时过滤 manifest，再用这份 manifest 做预训练。
- 预训练权重默认写到 `pretrained_weights`：`full` 写到 `pretrained_weights\full\<objective>`，`strict` 写到 `pretrained_weights\strict\<dataset>\<subject>\<objective>`。
- 如果 `strict` 只传 `--target-dataset`，程序会自动从 `cache\joint_contrastive\manifest_all.csv` 中找出该数据集的全部被试，并连续为每个被试各跑一次 strict 预训练。

例如，对整个 `ds002336` 生成 strict 预训练权重：

```powershell
python run_pretrain.py --config configs\train_joint_contrastive.yaml --pretrain-mode strict --target-dataset ds002336
```

Pure InfoNCE baseline：

```powershell
python run_pretrain.py --config configs\train_joint_infonce.yaml
```

Barlow Twins baseline：

```powershell
python run_pretrain.py --config configs\train_joint_barlow_twins.yaml
```

### 9. LOSO 全折微调 ds009999

先预处理生成 `cache\ds009999\loso_subjectwise\fold_*`，然后可直接用 `run_finetune.py --loso` 跑完整 LOSO：

```powershell
python run_finetune.py --config configs\finetune_ds009999.yaml --loso --root-dir cache\ds009999 --output-dir outputs\ds009999\finetune
```

`run_finetune.py` 现在就是完整 LOSO 入口：

- 不加 `--loso`：按当前 config 跑单 fold
- 加 `--loso`：自动遍历 `<root_dir>\loso_subjectwise\fold_*`，并在输出目录下生成 `loso_finetune_summary.csv`
- 如需自定义 LOSO 划分目录，可额外传 `--split-root`
- 因此 `ds009999` 的常规微调和 Optuna 微调都直接推荐使用 `run_finetune.py`
- 微调阶段仍与预训练解耦；如果没有手动传 `--contrastive-checkpoint`，则会按配置中的 `finetune.pretrain_mode` 自动定位对应预训练权重。默认配置中 `ds002336`、`ds002338` 使用 `strict`，`ds009999` 使用 `full`

如果只想临时跑单 fold，可再补充：

```powershell
python run_finetune.py --config configs\finetune_ds009999.yaml
python run_finetune.py --config configs\finetune_ds002338.yaml
```

如果要评估 `Pure InfoNCE / Barlow Twins` 预训练后的 EEG-only 微调，可在现有微调配置上追加：

```powershell
python run_finetune.py --config configs\finetune_ds009999.yaml --loso --root-dir cache\ds009999 --output-dir outputs\ds009999\finetune --set "finetune.eeg_encoder_variant='shared_only'" --set "finetune.fusion='eeg_only'" --set "finetune.classifier_mode='shared'" --set "finetune.contrastive_checkpoint_path='path\\to\\best.pth'"
```

### 10. 运行单个 EEG baseline

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

### 14. Optuna 搜索 ds002338

```powershell
python run_optuna_search.py --study-config configs\optuna_ds002338.yaml --mode full
```

### 15. Optuna 搜索 ds009999

`ds009999` 只支持 finetune-only：

```powershell
python run_optuna_search.py --study-config configs\optuna_ds009999.yaml --mode finetune_only
```

说明：`ds009999` 现在和另外三个数据集一样，也是两层 metric 配置。
- 顶层 `metric.column=accuracy`
- `modes.finetune_only.metric.column=macro_f1`

### 16. 可视化

当前项目的可视化输出分为三部分：

1. 微调训练过程自动生成训练/验证损失曲线

```powershell
python run_finetune.py --config configs\finetune_ds002338.yaml --save-train-curve
```

默认不会生成。只有显式传 `--save-train-curve` 后，输出目录下才会生成：

- `train_loss_curve_XXX.png`
- `train_loss_history_XXX.json`
- `train_loss_history_XXX.csv`

2. 混淆矩阵

默认不会生成。只有显式传 `--save-confusion-matrix`，或在配置里设置 `finetune.visualization.confusion_matrix.enabled=true` 后，测试阶段才会保存单次运行的混淆矩阵：

```powershell
python run_finetune.py --config configs\finetune_ds002338.yaml --save-confusion-matrix
```

如果需要对已有 LOSO 权重做离线汇总，可运行：

```powershell
python .\run_visualize.py offline-loso --dataset-name ds009999 --config configs\finetune_ds009999.yaml --checkpoints-root outputs\ds009999\finetune --output-dir outputs\ds009999\offline_eval
```

3. 预训练表征可视化

这里实际会生成两张图：`t-SNE` 和跨模态相似度热图。

```powershell
python .\run_visualize.py contrastive --config configs\train_joint_contrastive.yaml --checkpoint outputs\joint_contrastive\contrastive\checkpoints\best.pth --output-dir outputs\visualizations\contrastive --batch-size 32 --tsne-max-points 2000 --heatmap-max-points 128
```

如果只想快速抽样查看，可额外加 `--max-samples`。这个参数会在前面的数据集中随机选择一个 batch 起点，然后按原顺序取连续的一段样本，而不打乱样本顺序；如果还想复现同一段样本，可再加 `--sample-seed`：

```powershell
python .\run_visualize.py contrastive --config configs\train_joint_contrastive.yaml --checkpoint pretrained_weights\contrastive_best.pth --output-dir outputs\visualizations\contrastive --batch-size 128 --max-samples 512 --sample-seed 42 --tsne-max-points 200 --heatmap-max-points 128
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
python run_pretrain.py --config configs/train_joint_contrastive.yaml
```

离线预训练现在支持两种模式：

- `full`：直接使用 `cache/joint_contrastive/manifest_all.csv` 中的全部样本做联合预训练。
- `strict`：仍然直接使用现有 `cache/joint_contrastive` 缓存，不重新做预处理；只是在启动预训练时，从 `manifest_all.csv` 中排除“目标数据集中的目标测试被试”，生成一份临时过滤 manifest，再用这份 manifest 做预训练。
- 预训练权重默认写到 `pretrained_weights`：`full` 写到 `pretrained_weights/full/<objective>`，`strict` 写到 `pretrained_weights/strict/<dataset>/<subject>/<objective>`。
- 如果 `strict` 只传 `--target-dataset`，程序会自动从 `cache/joint_contrastive/manifest_all.csv` 中找出该数据集的全部被试，并连续为每个被试各跑一次 strict 预训练。

例如，对整个 `ds002336` 生成 strict 预训练权重：

```bash
python run_pretrain.py --config configs/train_joint_contrastive.yaml --pretrain-mode strict --target-dataset ds002336
```

Pure InfoNCE baseline：

```bash
python run_pretrain.py --config configs/train_joint_infonce.yaml
```

Barlow Twins baseline：

```bash
python run_pretrain.py --config configs/train_joint_barlow_twins.yaml
```

### 9. LOSO 全折微调 ds009999

先预处理生成 `cache/ds009999/loso_subjectwise/fold_*`，然后可直接用 `run_finetune.py --loso` 跑完整 LOSO：

```bash
python run_finetune.py --config configs/finetune_ds009999.yaml --loso --root-dir cache/ds009999 --output-dir outputs/ds009999/finetune
```

`run_finetune.py` 现在就是完整 LOSO 入口：

- 不加 `--loso`：按当前 config 跑单 fold
- 加 `--loso`：自动遍历 `<root_dir>/loso_subjectwise/fold_*`，并在输出目录下生成 `loso_finetune_summary.csv`
- 如需自定义 LOSO 划分目录，可额外传 `--split-root`
- 因此 `ds009999` 的常规微调和 Optuna 微调都直接推荐使用 `run_finetune.py`
- 微调阶段仍与预训练解耦；如果没有手动传 `--contrastive-checkpoint`，则会按配置中的 `finetune.pretrain_mode` 自动定位对应预训练权重。默认配置中 `ds002336`、`ds002338` 使用 `strict`，`ds009999` 使用 `full`

如果只想临时跑单 fold，可再补充：

```bash
python run_finetune.py --config configs/finetune_ds009999.yaml
python run_finetune.py --config configs/finetune_ds002338.yaml
```

如果要评估 `Pure InfoNCE / Barlow Twins` 预训练后的 EEG-only 微调，可在现有微调配置上追加：

```bash
python run_finetune.py --config configs/finetune_ds009999.yaml --loso --root-dir cache/ds009999 --output-dir outputs/ds009999/finetune --set "finetune.eeg_encoder_variant='shared_only'" --set "finetune.fusion='eeg_only'" --set "finetune.classifier_mode='shared'" --set "finetune.contrastive_checkpoint_path='path/to/best.pth'"
```

### 10. 运行 EEG baseline

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

### 14. Optuna 搜索 ds002338

```bash
python run_optuna_search.py --study-config configs/optuna_ds002338_linux.yaml --mode full
```

### 15. Optuna 搜索 ds009999

```bash
python run_optuna_search.py --study-config configs/optuna_ds009999_linux.yaml --mode finetune_only
```

说明：`ds009999` 现在和另外三个数据集一样，也是两层 metric 配置。
- 顶层 `metric.column=accuracy`
- `modes.finetune_only.metric.column=macro_f1`

### 16. 可视化

当前项目的可视化输出分为三部分：

1. 微调训练过程自动生成训练/验证损失曲线

```bash
python run_finetune.py --config configs/finetune_ds002338.yaml --save-train-curve
```

默认不会生成。只有显式传 `--save-train-curve` 后，输出目录下才会生成：

- `train_loss_curve_XXX.png`
- `train_loss_history_XXX.json`
- `train_loss_history_XXX.csv`

2. 混淆矩阵

默认不会生成。只有显式传 `--save-confusion-matrix`，或在配置里设置 `finetune.visualization.confusion_matrix.enabled=true` 后，测试阶段才会保存单次运行的混淆矩阵：

```bash
python run_finetune.py --config configs/finetune_ds002338.yaml --save-confusion-matrix
```

如果需要对已有 LOSO 权重做离线汇总，可运行：

```bash
python run_visualize.py offline-loso --dataset-name ds009999 --config configs/finetune_ds009999.yaml --checkpoints-root outputs/ds009999/finetune --output-dir outputs/ds009999/offline_eval
```

3. 预训练表征可视化

这里实际会生成两张图：`t-SNE` 和跨模态相似度热图。

```bash
python run_visualize.py contrastive --config configs/train_joint_contrastive.yaml --checkpoint outputs/joint_contrastive/contrastive/checkpoints/best.pth --output-dir outputs/visualizations/contrastive --batch-size 32 --tsne-max-points 2000 --heatmap-max-points 128
```

如果只想快速抽样查看，可额外加 `--max-samples`。这个参数会在前面的数据集中随机选择一个 batch 起点，然后按原顺序取连续的一段样本，而不打乱样本顺序；如果还想复现同一段样本，可再加 `--sample-seed`：

```bash
python run_visualize.py contrastive --config configs/train_joint_contrastive.yaml --checkpoint pretrained_weights/contrastive_best.pth --output-dir outputs/visualizations/contrastive --batch-size 128 --max-samples 512 --sample-seed 42 --tsne-max-points 200 --heatmap-max-points 128
```

## 脚本索引与用法

执行前请先手动 `conda activate <your-mamba-env>`，再进入项目根目录。

### 1. 核心 Python 入口

`run_pretrain.py`

联合预训练入口，可用于主方法、Pure InfoNCE baseline、Barlow Twins baseline。

```bash
python run_pretrain.py --config configs/train_joint_contrastive.yaml
python run_pretrain.py --config configs/train_joint_infonce.yaml
python run_pretrain.py --config configs/train_joint_barlow_twins.yaml
```

`run_finetune.py`

统一微调入口：

- 不加 `--loso`：跑单 fold
- 加 `--loso`：自动遍历 `<root_dir>/loso_subjectwise/fold_*` 跑完整 LOSO
- 自动在输出目录写出 `loso_finetune_summary.csv`
- `--save-train-curve`：显式开启微调训练/验证损失曲线输出
- `--save-confusion-matrix`：显式开启测试混淆矩阵输出
- `--train-curve-output-dir`：可选指定损失曲线输出目录

```bash
python run_finetune.py --config configs/finetune_ds002338.yaml
python run_finetune.py --config configs/finetune_ds009999.yaml --loso --root-dir cache/ds009999 --output-dir outputs/ds009999/finetune
python run_finetune.py --config configs/finetune_ds002338.yaml --save-train-curve
python run_finetune.py --config configs/finetune_ds002338.yaml --save-confusion-matrix
```

`run_optuna_search.py`

```bash
python run_optuna_search.py --study-config configs/optuna_ds002338_linux.yaml --mode full
python run_optuna_search.py --study-config configs/optuna_ds009999_linux.yaml --mode finetune_only
```

`run_visualize.py`

对比预训练可视化入口，用于生成 t-SNE 和跨模态相似度热力图。

```bash
python run_visualize.py contrastive --config configs/train_joint_contrastive.yaml --checkpoint outputs/joint_contrastive/contrastive/checkpoints/best.pth --output-dir outputs/visualizations/contrastive
```

如果只想快速抽样查看，可额外加 `--max-samples 256` 或 `--max-samples 512`；默认是随机 batch 起点的连续切片，想复现可再加 `--sample-seed 42`。

```bash
python run_visualize.py offline-loso --dataset-name ds009999 --config configs/finetune_ds009999.yaml --checkpoints-root outputs/ds009999/finetune --output-dir outputs/ds009999/offline_eval
```

### 2. Linux 脚本

`scripts_linux/prepare_joint_contrastive.sh`

准备 `ds002336/ds002338/ds002739` 的联合预训练缓存。

```bash
./scripts_linux/prepare_joint_contrastive.sh --ds002336-root ../ds002336 --ds002338-root ../ds002338 --ds002739-root ../ds002739 --output-root cache/joint_contrastive
```

`scripts_linux/ds00233x/prepare_ds00233x.sh`

准备 `ds002336` 或 `ds002338` 的 EEG-only 微调缓存。

```bash
./scripts_linux/ds00233x/prepare_ds00233x.sh --dataset-name ds002336 --ds-root ../ds002336 --output-root cache/ds002336
./scripts_linux/ds00233x/prepare_ds00233x.sh --dataset-name ds002338 --ds-root ../ds002338 --output-root cache/ds002338
```

`scripts_linux/ds00233x/prepare_ds00233x_spm.sh`

`ds00233x` 的 SPM 预处理辅助脚本。

```bash
./scripts_linux/ds00233x/prepare_ds00233x_spm.sh --dataset-name ds002336 --ds-root ../ds002336
```

`scripts_linux/ds002739/prepare_ds002739.sh`

准备 `ds002739` 的 EEG-only 微调缓存。

```bash
./scripts_linux/ds002739/prepare_ds002739.sh --ds-root ../ds002739 --output-root cache/ds002739
```

`scripts_linux/ds009999/prepare_ds009999.sh`

准备 `ds009999(SEED)` 的 LOSO 微调缓存。

```bash
./scripts_linux/ds009999/prepare_ds009999.sh --ds-root ../SEED --output-root cache/ds009999 --split-mode loso
```

`scripts_linux/run_pretrain_and_finetune.sh`

串联执行联合预训练缓存准备、联合预训练、目标数据集微调缓存准备，并最终调用 `run_finetune.py --loso` 跑完整 LOSO。

```bash
./scripts_linux/run_pretrain_and_finetune.sh --target-dataset ds002739 --joint-train-config configs/train_joint_contrastive.yaml --ds002739-finetune-config configs/finetune_ds002739.yaml
```

说明：适用于 `ds002336/ds002338/ds002739` 这类 joint-pretrain 体系下的数据集。

`scripts_linux/run_optuna_pretrain_and_finetune.sh`

给 Optuna 用的 joint pipeline 包装脚本，本质上会再调用 `run_pretrain_and_finetune.sh`。

```bash
./scripts_linux/run_optuna_pretrain_and_finetune.sh --target-dataset ds002739 --finetune-config configs/finetune_ds002739.yaml --output-root outputs/optuna_run
```

说明：适用于 `ds002336/ds002338/ds002739`，不用于 `ds009999`。

`scripts_linux/run_all_eeg_baselines.sh`

批量跑某个数据集的所有 EEG baselines。

```bash
./scripts_linux/run_all_eeg_baselines.sh --config configs/finetune_ds002336.yaml --models svm,labram,cbramod,eeg_deformer,eegnet,conformer,tsception --output-root outputs/finetune_ds002336_all_baselines
```

`scripts_linux/run_finetune_from_pretrain_trials.sh`

```bash
./scripts_linux/run_finetune_from_pretrain_trials.sh --config configs/finetune_ds009999.yaml --pretrain-root ../pretrain_save --output-root outputs/finetune_from_pretrain_trials
```

### 3. Windows 脚本

`scripts/prepare_joint_contrastive.ps1`

准备 `ds002336/ds002338/ds002739` 的联合预训练缓存。

```powershell
.\scripts\prepare_joint_contrastive.ps1 -Ds002336Root ..\ds002336 -Ds002338Root ..\ds002338 -Ds002739Root ..\ds002739 -OutputRoot cache\joint_contrastive
```

`scripts/ds00233x/prepare_ds00233x.ps1`

准备 `ds002336` 或 `ds002338` 的 EEG-only 微调缓存。

```powershell
.\scripts\ds00233x\prepare_ds00233x.ps1 -DatasetName ds002336 -DsRoot ..\ds002336 -OutputRoot cache\ds002336
.\scripts\ds00233x\prepare_ds00233x.ps1 -DatasetName ds002338 -DsRoot ..\ds002338 -OutputRoot cache\ds002338
```

`scripts/ds00233x/prepare_ds00233x_spm.ps1`

`ds00233x` 的 SPM 预处理辅助脚本。

```powershell
.\scripts\ds00233x\prepare_ds00233x_spm.ps1 -DatasetName ds002336 -DsRoot ..\ds002336
```

`scripts/ds002739/prepare_ds002739.ps1`

准备 `ds002739` 的 EEG-only 微调缓存。

```powershell
.\scripts\ds002739\prepare_ds002739.ps1 -DsRoot ..\ds002739 -OutputRoot cache\ds002739
```

`scripts/ds009999/prepare_ds009999.ps1`

准备 `ds009999(SEED)` 的 LOSO 微调缓存。

```powershell
.\scripts\ds009999\prepare_ds009999.ps1 -DsRoot ..\SEED -OutputRoot cache\ds009999 -SplitMode loso
```

`scripts/run_pretrain_and_finetune.ps1`

串联执行联合预训练缓存准备、联合预训练、目标数据集微调缓存准备，并最终调用 `run_finetune.py --loso` 跑完整 LOSO。

```powershell
.\scripts\run_pretrain_and_finetune.ps1 -TargetDataset ds002739 -JointTrainConfig configs\train_joint_contrastive.yaml -Ds002739FinetuneConfig configs\finetune_ds002739.yaml
```

说明：适用于 `ds002336/ds002338/ds002739` 这类 joint-pretrain 体系下的数据集。

`scripts/run_optuna_pretrain_and_finetune.ps1`

给 Optuna 用的 joint pipeline 包装脚本，本质上会再调用 `run_pretrain_and_finetune.ps1`。

```powershell
.\scripts\run_optuna_pretrain_and_finetune.ps1 -TargetDataset ds002739 -FinetuneConfig configs\finetune_ds002739.yaml -OutputRoot outputs\optuna_run
```

说明：适用于 `ds002336/ds002338/ds002739`，不用于 `ds009999`。

### 4. Optuna 配置入口

Windows:

`configs/optuna_ds002338.yaml`

```powershell
python run_optuna_search.py --study-config configs\optuna_ds002338.yaml --mode full
```

`configs/optuna_ds009999.yaml`

```powershell
python run_optuna_search.py --study-config configs\optuna_ds009999.yaml --mode finetune_only
```

说明：`ds009999` 的 Optuna 也是通过 `run_optuna_search.py` 启动，但现在直接调用 `run_finetune.py --loso`。

Linux:

`configs/optuna_ds002338_linux.yaml`

```bash
python run_optuna_search.py --study-config configs/optuna_ds002338_linux.yaml --mode full
```

`configs/optuna_ds009999_linux.yaml`

```bash
python run_optuna_search.py --study-config configs/optuna_ds009999_linux.yaml --mode finetune_only
```

说明：`ds009999` 的 Optuna 也是通过 `run_optuna_search.py` 启动，但现在直接调用 `run_finetune.py --loso`。

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
- `test_logits.csv`
- `svm_summary.json`（仅 `svm` baseline）

只有显式开启混淆矩阵可视化后，才会额外生成：

- `test_confusion_matrix.png`
- `test_confusion_matrix.svg`
- `test_confusion_matrix.json`

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
