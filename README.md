# EEG-fMRI-Contrastive

一个用于 EEG-fMRI 配对样本学习、LOSO 交叉验证和 Optuna 自动搜参的工程。当前仓库内置 EEG 编码器 CBraMod 和 fMRI 编码器 NeuroSTORM，默认工作流已经完全收敛到本仓库内部，不再依赖外部 NeuroSTORM 路径。

当前支持的主流程：

- ds002336 与 ds002739 的预处理与缓存构建
- EEG-fMRI 双塔对比学习
- 基于对比学习骨干的分类微调
- subject-wise LOSO 交叉验证
- 基于已有 finetune checkpoint 的 TestOnly 评估
- 基于 LOSO 全流程的 Optuna 搜参和自动汇总

## 目录概览

```text
EEG-fMRI-Contrastive/
  README.md
  requirements.txt
  run_train.py
  run_finetune.py
  run_optuna_search.py
  configs/
    train_ds002336.yaml
    finetune_ds002336.yaml
    train_ds002739.yaml
    finetune_ds002739.yaml
    optuna_loso_ds002336.yaml
    optuna_loso_ds002739.yaml
  preprocess/
    prepare_ds002336.py
    prepare_ds002739.py
    preprocess_common.py
  mmcontrast/
    baselines/
      eeg_baseline.py
    contrastive_runner.py
    contrastive_trainer.py
    finetune_runner.py
    finetune_trainer.py
    datasets/
    models/
    losses.py
  scripts/
    run_prepare_all.ps1
    ds002336/
      prepare_ds002336.ps1
      prepare_ds002336_spm.ps1
      run_ds002336.ps1
    ds002739/
      prepare_ds002739.ps1
      run_ds002739.ps1
    test/
      run_ds002336_contrastive.ps1
      run_ds002336_finetune.ps1
      run_ds002739_contrastive.ps1
      run_ds002739_finetune.ps1
```

## 默认约定

- 数据缓存目录：cache/ds002336、cache/ds002739
- LOSO manifest 目录：cache/<dataset>/loso_subjectwise/fold_*
- 训练输出目录：outputs/ds002336、outputs/ds002739
- 主批量入口：scripts/ds002336/run_ds002336.ps1、scripts/ds002739/run_ds002739.ps1
- 主评估方式：subject-wise LOSO
- scripts/test 下的脚本只用于单次调试，不是主批量入口
- PowerShell 脚本默认直接调用当前终端里的 python，所以先激活你想用的环境

## 环境准备

建议先激活你的 Conda 环境，再安装依赖。当前仓库里的 Optuna 冒烟测试是在 mamba 环境下通过的。

```powershell
conda activate mamba
cd D:\OpenNeuro\EEG-fMRI-Contrastive
pip install -r requirements.txt
```

如果你只打算跑训练脚本而不做 Optuna，也可以使用任意满足 requirements 的 Python 环境。但如果当前环境里没有安装 optuna，run_optuna_search.py 会直接失败。

## 输入形状与默认配置

### EEG

- 输入形状为 [C, S, P]
- C 为通道数
- S 为序列块数
- P 为 patch 长度

### fMRI

- 当前默认走 volume 路径
- 单样本原始数组通常为 [H, W, D, T]
- 模型输入整理为 [B, H, W, D, T]

当前两个数据集的默认导出形状：

- ds002336：EEG [63, 20, 200]，fMRI [48, 48, 48, 10]
- ds002739：EEG [53, 2, 200]，fMRI [48, 48, 48, 3]

这些默认值分别写在：

- configs/train_ds002336.yaml
- configs/finetune_ds002336.yaml
- configs/train_ds002739.yaml
- configs/finetune_ds002739.yaml

## 运行流程总览

日常完整流程建议按下面顺序执行：

1. 激活环境并安装依赖。
2. 对目标数据集执行预处理，生成 cache/<dataset>。
3. 用对应的 LOSO 主脚本执行整套训练或评估。
4. 到 outputs/<dataset> 查看对比学习、微调和跨折汇总结果。
5. 如果要搜参，再调用 run_optuna_search.py 包装 LOSO 主脚本。

主入口只有两个：

- scripts/ds002336/run_ds002336.ps1
- scripts/ds002739/run_ds002739.ps1

这两个脚本是正式批量入口。它们会自动遍历 cache/<dataset>/loso_subjectwise/fold_*，并在每一折上运行对比学习、微调或 TestOnly，然后自动汇总。

## 第一步：数据预处理

下面命令默认都在仓库根目录执行。

### ds002336

默认情况下，prepare_ds002336.ps1 会直接读取已有的 SPM 预处理结果，不会重新跑 SPM。

```powershell
.\scripts\ds002336\prepare_ds002336.ps1
```

如果你要强制改回原始 fMRI：

```powershell
.\scripts\ds002336\prepare_ds002336.ps1 -FmriSource raw
```

如果你希望从头调用 SPM12 预处理，再写入当前仓库的缓存，走下面这个入口：

```powershell
.\scripts\ds002336\prepare_ds002336_spm.ps1 -ParallelJobs 4
```

说明：

- prepare_ds002336_spm.ps1 依赖本机可直接调用 matlab
- 需要事先配置好 SPM12
- prepare_ds002336_spm.ps1 会先跑 SPM，再调用 prepare_ds002336.ps1
- prepare_ds002336.ps1 默认读取 derivatives/spm12_preproc 下的最终 NIfTI
- 默认输出到 cache/ds002336

prepare_ds002336.ps1 的常用参数：

- -OutputRoot：修改缓存输出目录
- -Subjects：只处理指定被试
- -Tasks：只处理指定任务
- -SplitMode：none、subject、loso，默认 loso
- -FmriSource：raw、spm_unsmoothed、spm_smoothed，默认 spm_smoothed
- -TrainingReady：是否在 preprocess 阶段直接做训练所需规范化，并在 manifest 中标记为训练就绪；默认开启，后续 dataset 初始化会跳过规范化和 fMRI 形状适配

如果你想直接导出训练就绪版缓存：

```powershell
.\scripts\ds002336\prepare_ds002336.ps1
```

### ds002739

```powershell
.\scripts\ds002739\prepare_ds002739.ps1
```

如果你想指定并行 worker 和输出目录：

```powershell
.\scripts\ds002739\prepare_ds002739.ps1 -NumWorkers 8 -OutputRoot cache\ds002739_parallel
```

prepare_ds002739.ps1 的常用参数：

- -OutputRoot：修改缓存输出目录
- -Subjects：只处理指定被试
- -Runs：只处理指定 run
- -NumWorkers：预处理 worker 数
- -SplitMode：none、subject、loso，默认 loso
- fMRI volume 预处理默认先重采样到 2x2x2 mm，再中心裁剪到 48x48x48
- -TrainingReady：是否在 preprocess 阶段直接做训练所需规范化，并在 manifest 中标记为训练就绪；默认开启，后续 dataset 初始化会跳过规范化和 fMRI 形状适配

如果你想直接导出训练就绪版缓存：

```powershell
.\scripts\ds002739\prepare_ds002739.ps1
```

### training-ready 的含义

- 现在默认就是训练就绪模式：preprocess 会直接产出训练就绪数据，并把 `training_ready=true` 写进 manifest；dataset 初始化时会跳过 EEG/fMRI 规范化以及 fMRI 的 target_shape、pad/crop 处理。
- 如果你想退回旧行为，可以显式关闭：PowerShell 脚本里传 `-TrainingReady:$false`，或直接调用 Python 预处理入口时传 `--no-training-ready`。
- 这个开关只负责把训练前的数据准备职责前移到 preprocess，不会改变 LOSO 切分逻辑，也不会改变模型配置本身。

### 顺序处理两个数据集

```powershell
.\scripts\run_prepare_all.ps1
```

## 第二步：主 LOSO 训练与评估

### LOSO 主脚本做什么

scripts/ds002336/run_ds002336.ps1 和 scripts/ds002739/run_ds002739.ps1 会自动：

1. 遍历 cache/<dataset>/loso_subjectwise/fold_*。
2. 为每一折读取 manifest_train.csv、manifest_val.csv、manifest_test.csv。
3. 先跑对比学习，再把 contrastive checkpoint 传给微调。
4. 最终在 outputs/<dataset>/contrastive 和 outputs/<dataset>/finetune 下写出各折结果。
5. 自动写出跨折汇总：
   - contrastive/loso_contrastive_summary.csv
   - finetune/loso_finetune_summary.csv

主脚本本身只负责流程编排与少量通用覆盖项。模型的大部分超参数仍然以 YAML 配置为准。

### 主脚本支持的常用参数

两个数据集的主脚本参数保持同步，常用项如下：

- -TrainConfig：指定对比学习 YAML，默认分别是 configs/train_ds002336.yaml 或 configs/train_ds002739.yaml
- -FinetuneConfig：指定微调 YAML，默认分别是 configs/finetune_ds002336.yaml 或 configs/finetune_ds002739.yaml
- -DataRoot：缓存根目录
- -LosoDir：LOSO fold 目录
- -OutputRoot：输出根目录
- -TrainEpochs：覆盖 contrastive epoch
- -FinetuneEpochs：覆盖 finetune epoch
- -BatchSize：覆盖 train 和 finetune 的 batch size
- -EvalBatchSize：覆盖评估 batch size
- -NumWorkers：覆盖 DataLoader worker 数
- -ForceCpu：强制 CPU
- -SkipContrastive：跳过对比学习
- -SkipFinetune：跳过微调
- -TestOnly：只做 finetune checkpoint 的 test-only 评估

补充约束：

- -SkipContrastive 和 -SkipFinetune 不能同时设置
- -TestOnly 不能和 -SkipFinetune 同时设置
- 如果只传 -TrainEpochs 而不传 -FinetuneEpochs，主脚本会把这个 epoch 数同时用于 contrastive 和 finetune

### 模式 1：完整 LOSO

这是默认模式，会先做对比学习，再做微调。

```powershell
.\scripts\ds002336\run_ds002336.ps1
```

```powershell
.\scripts\ds002739\run_ds002739.ps1
```

如果你想先做一次短跑检查：

```powershell
.\scripts\ds002336\run_ds002336.ps1 -TrainEpochs 5 -BatchSize 8 -NumWorkers 2
```

### 模式 2：只做微调

跳过对比学习，直接做微调。

```powershell
.\scripts\ds002336\run_ds002336.ps1 -SkipContrastive
```

```powershell
.\scripts\ds002739\run_ds002739.ps1 -SkipContrastive
```

行为说明：

- 如果某折已经存在 contrastive checkpoint，会自动复用
- 如果不存在，会从随机初始化开始微调

### 模式 3：只做对比学习

跳过微调，只跑 contrastive，并在最后输出跨折 retrieval 汇总。

```powershell
.\scripts\ds002336\run_ds002336.ps1 -SkipFinetune
```

```powershell
.\scripts\ds002739\run_ds002739.ps1 -SkipFinetune
```

这一模式最终关注的是：

- outputs/<dataset>/contrastive/loso_contrastive_summary.csv
- 其中 CROSS_FOLD_MEAN_STD 这一行的 mean_r1

### 模式 4：只做 TestOnly

TestOnly 会对每一折自动查找：

- outputs/<dataset>/finetune/<fold_dir>/checkpoints/best.pth

然后直接跑 test-only，不重新训练。

```powershell
.\scripts\ds002336\run_ds002336.ps1 -TestOnly
```

```powershell
.\scripts\ds002739\run_ds002739.ps1 -TestOnly
```

说明：

- TestOnly 不会重跑 contrastive 或 finetune
- 如果某一折缺少 finetune checkpoint，脚本会直接报错
- 可以叠加 -ForceCpu、-EvalBatchSize、-NumWorkers 等参数

## 第三步：查看输出结果

典型输出结构如下：

```text
outputs/
  ds002336/
    contrastive/
      fold_sub-xp101/
        checkpoints/best.pth
        final_metrics.json
      loso_contrastive_summary.csv
    finetune/
      fold_sub-xp101/
        checkpoints/best.pth
        test_metrics.json
        test_logits.csv
        test_logits_summary.json
      loso_finetune_summary.csv
  ds002739/
    contrastive/
    finetune/
```

其中：

- contrastive/fold_*/final_metrics.json 保存单折对比学习最终指标
- contrastive/loso_contrastive_summary.csv 保存跨折对比学习汇总
- finetune/fold_*/test_metrics.json 保存单折测试指标
- finetune/fold_*/test_logits.csv 保存逐样本 logits 明细
- finetune/fold_*/test_logits_summary.json 保存 logits 数值范围、NaN/Inf 检查和极值样本摘要
- finetune/loso_finetune_summary.csv 保存跨折分类汇总

日常最常看的通常是：

- 完整流程或微调流程：finetune/loso_finetune_summary.csv 里的 CROSS_FOLD_MEAN_STD
- 纯对比学习流程：contrastive/loso_contrastive_summary.csv 里的 CROSS_FOLD_MEAN_STD

## 第四步：Optuna 自动搜参与汇总

仓库提供了统一入口 run_optuna_search.py。它不会改训练器内部逻辑，而是直接调用现有 LOSO 主脚本，自动读取指标文件，并在 study 目录下输出汇总文件。

### Optuna 的基本用法

ds002336：

```powershell
python .\run_optuna_search.py --study-config configs\optuna_loso_ds002336.yaml --mode full
```

ds002739：

```powershell
python .\run_optuna_search.py --study-config configs\optuna_loso_ds002739.yaml --mode full
```

当前 Optuna 的核心约束是：每个 trial 都在完整的 LOSO 流程上评估，同一个 trial 里的同一套参数会应用到该次 LOSO 的所有 fold。

### 三种搜索模式

- --mode full：先对比学习，再微调，目标值读取 finetune/loso_finetune_summary.csv 里的 macro_f1
- --mode finetune_only：跳过对比学习，只搜索微调，目标值也是 macro_f1
- --mode contrastive_only：跳过微调，只搜索对比学习，目标值读取 contrastive/loso_contrastive_summary.csv 里的 mean_r1

示例：

```powershell
python .\run_optuna_search.py --study-config configs\optuna_loso_ds002336.yaml --mode finetune_only
python .\run_optuna_search.py --study-config configs\optuna_loso_ds002336.yaml --mode contrastive_only
```

```powershell
python .\run_optuna_search.py --study-config configs\optuna_loso_ds002739.yaml --mode finetune_only
python .\run_optuna_search.py --study-config configs\optuna_loso_ds002739.yaml --mode contrastive_only
```

### Optuna 的配置方式

现在的 Optuna 不再把一堆超参数直接拼到 PowerShell 命令行，而是：

1. 读取 configs/optuna_loso_ds002336.yaml 或 configs/optuna_loso_ds002739.yaml。
2. 为每个 trial 生成运行时 train YAML 和 finetune YAML。
3. 只把 -TrainConfig 和 -FinetuneConfig 传给 LOSO 主脚本。

也就是说，主流程现在是 YAML 驱动，而不是大段 CLI 超参透传。

当前 Optuna 配置默认搜索的主要参数包括：

- train_epochs
- batch_size
- lr
- weight_decay
- min_lr
- hidden_dim
- grad_clip
- early_stop_patience

其中：

- hidden_dim 和 early_stop_patience 只影响 finetune
- contrastive_only 模式不会搜索 hidden_dim 和 early_stop_patience

### 只重建汇总

如果 trial 已经跑完，只想重新导出 study 汇总：

```powershell
python .\run_optuna_search.py --study-config configs\optuna_loso_ds002336.yaml --summary-only
```

同理也可以替换成 ds002739 的配置文件。

### Optuna 输出内容

每个 study 输出目录下会自动生成：

- trials.csv
- study_summary.json
- best_trial.json
- study_summary.md
- 每个 trial 的 stdout.log
- 每个 trial 的 stderr.log
- 每个 trial 的 trial_plan.json
- 每个 trial 的 trial_result.json

## 直接调用 Python 入口

如果你不想走 PowerShell 封装，也可以直接调用 Python 入口。这更适合调单个配置、单个 fold 或单个 checkpoint。

### 对比学习

```powershell
python run_train.py --config configs/train_ds002336.yaml
```

```powershell
python run_train.py --config configs/train_ds002739.yaml
```

### 微调

```powershell
python run_finetune.py --config configs/finetune_ds002336.yaml
```

```powershell
python run_finetune.py --config configs/finetune_ds002739.yaml
```

### 从已有 checkpoint 做 test-only

```powershell
python run_finetune.py --config configs/finetune_ds002336.yaml --finetune-checkpoint outputs/ds002336/finetune/fold_sub-xp101/checkpoints/best.pth --test-only
```

```powershell
python run_finetune.py --config configs/finetune_ds002739.yaml --finetune-checkpoint outputs/ds002739/finetune/fold_sub-01/checkpoints/best.pth --test-only
```

注意：四个主 YAML 里的默认 manifest 路径只是为了让单次运行有一个可启动的默认值。真正走 LOSO 主脚本时，train、val、test manifest 与 output_dir 都会在运行时按折覆盖。

## 单次调试脚本

scripts/test 下保留了单次运行脚本，适合调单个配置或单个 checkpoint：

- scripts/test/run_ds002336_contrastive.ps1
- scripts/test/run_ds002336_finetune.ps1
- scripts/test/run_ds002739_contrastive.ps1
- scripts/test/run_ds002739_finetune.ps1

这些脚本不是主批量入口。批量训练、批量评估和 Optuna 搜索，应优先使用 scripts/ds002336、scripts/ds002739 和 run_optuna_search.py。

## Manifest 与样本组织

训练与评估由 manifest CSV 驱动。常见列包括：

- eeg_path
- fmri_path
- sample_id
- label

如果 manifest 包含 subject_path 和 sample_count，说明使用的是 subject-packed 形式。当前默认会把每个被试写成一个目录，目录内包含 eeg.npy、fmri.npy、labels.npy 等 memmap 友好文件。运行时会先展开成逐样本索引，再由 DataLoader 按 batch 取样。

## 训练与选择机制

### 对比学习

- 训练对象是 EEG-fMRI 配对样本
- 每个 batch 同时取出 EEG 和 fMRI
- 双塔编码器得到 eeg_embed 和 fmri_embed
- 使用对称 InfoNCE 做 EEG->fMRI 与 fMRI->EEG 两个方向的对比损失
- 正样本是同一索引位置的 EEG-fMRI 配对
- 负样本是当前 batch 中其他样本
- 当前支持在验证集上计算 retrieval 指标，并在 LOSO 汇总里使用 mean_r1

### 微调

- 在对比学习骨干上接分类头
- 读取 train、val、test manifest
- 默认按 validation 指标选择 best.pth
- 最终对 test split 评估，并写出 test_metrics.json、test_logits.csv、test_logits_summary.json
- finetune 目前支持 early stopping

### EEG Baseline 微调

微调阶段现在支持一个可选的 EEG baseline 接口，只作用于 finetune，不作用于 contrastive。

配置入口在两个 finetune YAML 里：

- configs/finetune_ds002336.yaml
- configs/finetune_ds002739.yaml

对应配置段为：

```yaml
finetune:
  fusion: eeg_only
  eeg_baseline:
    enabled: true
    category: traditional
    model_name: conv1d
```

两大类 baseline：

- traditional：传统 EEG 方法，输入会从 [B,C,S,P] 折叠成 [B,C,T]，模型自带分类头，直接输出 logits，不再额外接当前的 finetune 分类头
- foundation：基础模型 EEG 方法，继续使用当前 patch 输入 [B,C,S,P]，只负责输出特征，再复用我们现有的 finetune 分类头

当前可选名字：

- traditional：conv1d、cnn、shallowconv1d、lstm、bilstm
- foundation：cbramod、patch_mlp、mlp

额外约束：

- traditional baseline 目前只支持 finetune.fusion=eeg_only
- foundation baseline 可以继续走现有的 eeg_only 路径；如果你后面扩展到 concat，也会继续复用现有分类头逻辑

你可以直接在这两个位置看当前支持的 baseline 名字：

- mmcontrast/baselines/eeg_baseline.py：实际模型实现和名字分支
- mmcontrast/config.py：配置校验时允许的名字白名单

也可以通过命令行直接切换 baseline：

```powershell
python run_finetune.py --config configs/finetune_ds002336.yaml --eeg-baseline-category traditional --eeg-baseline-model conv1d
```

```powershell
python run_finetune.py --config configs/finetune_ds002336.yaml --eeg-baseline-category foundation --eeg-baseline-model cbramod
```

如果只想改 YAML，不走命令行覆盖，也可以直接编辑 finetune.eeg_baseline 这一段。

### Early stopping

finetune 配置中已经提供：

- finetune.early_stop_patience
- finetune.early_stop_min_delta

默认值已经写入两个 finetune YAML。当前只有 finetune 支持 early stopping，contrastive 还没有这一机制。

## 最小复现命令

### ds002336：推荐完整流程

```powershell
conda activate mamba
cd D:\OpenNeuro\EEG-fMRI-Contrastive
.\scripts\ds002336\prepare_ds002336_spm.ps1 -ParallelJobs 4
.\scripts\ds002336\run_ds002336.ps1
```

### ds002739：推荐完整流程

```powershell
conda activate mamba
cd D:\OpenNeuro\EEG-fMRI-Contrastive
.\scripts\ds002739\prepare_ds002739.ps1 -NumWorkers 8
.\scripts\ds002739\run_ds002739.ps1
```

### 只复查已有模型

```powershell
.\scripts\ds002336\run_ds002336.ps1 -TestOnly
```

```powershell
.\scripts\ds002739\run_ds002739.ps1 -TestOnly
```

## 已知边界

- 当前默认 fMRI 路径是 NeuroSTORM volume，不再以旧的 Brain-JEPA 入口为主
- 在线 pad、crop、interpolate 只能解决尺寸不一致，不能修复采集阶段的信息缺失
- Windows 下某些加速依赖不一定总能安装，但默认流程通常仍可运行
- Optuna 依赖当前 Python 环境中已经安装 optuna
