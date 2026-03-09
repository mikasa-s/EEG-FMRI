# EEG-fMRI-Contrastive

一个自包含融合工程，用于把 `eeg-CBraMod` 和 `fmri-Brain-JEPA` 作为双塔编码器，进行 EEG-fMRI 跨模态对比学习（InfoNCE）。

当前目录已经内置两套基础模型的核心骨干源码，不再依赖去其他仓库跨目录导入模型。

## 1. 支持能力

- 内置 EEG 基础模型 `CBraMod` 完整骨干。
- 内置 fMRI 基础模型 `Brain-JEPA VisionTransformer` 完整骨干。
- 双塔投影到共享嵌入空间，做对称 InfoNCE 对比学习。
- 支持 `torchrun` 多 GPU 分布式训练（DDP）。
- 提供外接数据接口：
  - Manifest CSV 数据集接口（开箱即用）。
  - 自定义 Dataset 接口模板（你可以接任何自有数据格式）。

## 2. 文件夹结构

```text
EEG-fMRI-Contrastive/
  README.md
  requirements.txt
  run_train.py
  run_finetune.py
  configs/
    train_contrastive_binary_block.yaml
    finetune_classifier_binary_block.yaml
    train_contrastive_true450_rawfmri.yaml
    finetune_classifier_true450_rawfmri.yaml
    train_contrastive_true450_5patch.yaml
    finetune_classifier_true450_5patch.yaml
  scripts/
    train_multigpu.sh
    train_multigpu.bat
    build_subject_splits.py
    build_loso_splits.py
    demo_loso_pipeline.py
    summarize_loso_metrics.py
    inspect_checkpoints.py
  assets/
    gradient_mapping_450.csv        # 你自己的 ROI gradient 文件
  pretrained_weights/
    eeg_cbramod.pth                 # 可选，EEG 预训练权重
    fmri_brainjepa.pth              # 可选，fMRI 预训练权重
  mmcontrast/
    __init__.py
    config.py
    distributed.py
    losses.py
    metrics.py
    runner.py
    finetune_runner.py
    trainer.py
    finetune_trainer.py
    datasets/
      __init__.py
      paired_manifest_dataset.py
      custom_interface.py
    backbones/
      __init__.py
      eeg_cbramod/
        __init__.py
        cbramod.py
        criss_cross_transformer.py
      fmri_brainjepa/
        __init__.py
        vision_transformer.py
        tensors.py
        mask_utils.py
    models/
      __init__.py
      classifier.py
      eeg_adapter.py
      fmri_adapter.py
      multimodal_model.py
```

## 3. 环境安装

建议使用与你现有项目一致的 conda 环境。

```bash
pip install -r requirements.txt
```

如果你想用 `flash_attn`，请自行安装对应版本；默认配置里已经把 `attn_mode` 设置成 `normal`，即使不安装也能运行。

## 4. 你的数据如何接入

### 4.1 推荐方式：Manifest CSV

在 `data.root_dir` 下组织文件，例如：

```text
my_data/
  eeg/
    sample_0001.npy
  fmri/
    sample_0001.npy
  manifest_train.csv
```

`manifest_train.csv` 必需列：

- `eeg_path`
- `fmri_path`

可选列：

- `sample_id`
- `label`

示例：

```csv
sample_id,eeg_path,fmri_path,label
s0001,eeg/sample_0001.npy,fmri/sample_0001.npy,0
s0002,eeg/sample_0002.npy,fmri/sample_0002.npy,1
```

数据张量形状要求：

- EEG: `numpy` 形状 `[C, S, P]`，对应 `[通道, patch段数, patch长度]`。
- fMRI: `numpy` 形状 `[1, ROI, T]` 或 `[ROI, T]`（后者会自动补成 `[1, ROI, T]`）。

支持后缀：`.npy`、`.npz`（默认读取 `arr_0`）、`.pt`。

### 4.2 自定义方式：自己写 Dataset

参考 `mmcontrast/datasets/custom_interface.py`，输出字段保持：

```python
{
  "eeg": Tensor[C, S, P],
  "fmri": Tensor[1, ROI, T],
  "sample_id": str,
  "label": int  # optional
}
```

然后在 `mmcontrast/trainer.py` 中把数据集替换为你的类即可。

## 5. 关键配置说明

对比学习默认配置文件：`configs/train_contrastive_binary_block.yaml`

- `eeg_model.checkpoint_path`: EEG 预训练权重路径。
- `fmri_model.gradient_csv_path`: fMRI gradient csv 路径。
- `fmri_model.checkpoint_path`: fMRI 预训练权重（可留空）。
- `data.train_manifest_csv`: 训练集配对清单。
- `data.val_manifest_csv`: 验证集配对清单。
- `data.test_manifest_csv`: 测试集配对清单。
- `data.root_dir`: 清单中相对路径的根目录。

分类微调默认配置文件：`configs/finetune_classifier_binary_block.yaml`

- `finetune.contrastive_checkpoint_path`: 对比学习得到的 `best.pth` 或其它 checkpoint。
- `finetune.num_classes`: 分类类别数。
- `finetune.selection_metric`: 用哪个验证集指标选择最佳微调模型，可选 `accuracy` / `acc` / `macro_f1` / `f1`，默认 `accuracy`。
- `finetune.fusion`: `concat`、`eeg_only`、`fmri_only`。
- `finetune.freeze_encoders`: 是否冻结编码器，仅训练分类头。

形状对齐规则：

- `eeg_model.seq_len` 必须等于 EEG 的 patch 数 `S`。
- `eeg_model.in_dim` 必须等于 EEG 的 patch 长度 `P`。
- `fmri_model.crop_size` 必须等于 fMRI 的 `[ROI, T]`。
- 如果你在 `data.expected_eeg_shape` / `data.expected_fmri_shape` 里显式填写了期望形状，训练入口会在启动前把它和 manifest 首个样本做一致性校验。
- 现在新增了一套通用 block 二分类配置：`configs/train_contrastive_binary_block.yaml` 和 `configs/finetune_classifier_binary_block.yaml`。
- 对 ds002336 这类 TR=2s、20s block 的数据，原生 fMRI 时间长度应是 10，而不是 160。把 10 插值成 160 只是旧模型适配技巧，不属于原始预处理结果。
- 如果要使用 450 ROI，请提供真实的 450 区标签图，例如 `50 Tian Scale III subcortical + 400 Schaefer cortical` 的合并 atlas；不要把 400 ROI 直接插值成 450。
- 如果要做留一被试交叉验证，可以用 `scripts/build_loso_splits.py` 按需生成每个 fold 的 train/val/test manifest。脚本默认只写 split 清单，不再默认输出每折 YAML；只有显式传入 `--write-fold-configs --config-dir ...` 时才会额外生成派生配置文件。`--val-subjects` 默认控制每折验证被试数。

## 6. 训练命令（多 GPU）

对比学习主入口和训练函数现在分别在：

- 主函数入口：`run_train.py -> main()`
- 运行调度：`mmcontrast/runner.py -> run_training()`
- 总训练循环：`mmcontrast/trainer.py -> fit()`
- 单轮训练：`mmcontrast/trainer.py -> train_one_epoch()`

当前默认实验逻辑：

- 对比学习只在训练集上更新模型。
- 对比学习使用验证集 `loss` 选择 `best.pth`。
- 对比学习评估默认关注验证集，不再默认对测试集做最终评估。

分类微调主入口和训练函数现在分别在：

- 主函数入口：`run_finetune.py -> main()`
- 运行调度：`mmcontrast/finetune_runner.py -> run_finetuning()`
- 总训练循环：`mmcontrast/finetune_trainer.py -> fit()`
- 单轮训练：`mmcontrast/finetune_trainer.py -> train_one_epoch()`

微调默认实验逻辑：

- 微调只在训练集上更新模型。
- 微调使用验证集指标选择 `best.pth`，默认是 `accuracy`，也可以改成 `macro_f1`。
- 微调只对验证集选出的最佳模型执行最终测试集评估。

### Linux/macOS

```bash
bash scripts/train_multigpu.sh 4 configs/train_contrastive_binary_block.yaml
```

### Windows

```bat
scripts\train_multigpu.bat 2 configs\train_contrastive_binary_block.yaml
```

或直接：

```bat
python -m torch.distributed.run --nproc_per_node=2 run_train.py --config configs/train_contrastive_binary_block.yaml
```

## 7. 当前实验入口

1. 生成 ds002336 block 二分类数据：

```bash
python preprocess/prepare_ds002336.py --ds-root D:/OpenNeuro/ds002336 --output-root D:/OpenNeuro/EEG-fMRI-Contrastive/outputs/ds002336_binary_block_model_ready --tasks eegNF fmriNF eegfmriNF --sample-mode block --label-mode binary_rest_task --eeg-mode patched --drop-ecg --n-rois 400
```

2. 生成被试级切分：

```bash
python scripts/build_subject_splits.py --manifest D:/OpenNeuro/EEG-fMRI-Contrastive/outputs/ds002336_binary_block_model_ready/manifest_all.csv --output-dir D:/OpenNeuro/EEG-fMRI-Contrastive/outputs/ds002336_binary_block_model_ready/splits_subjectwise
```

3. 运行对比学习：

```bash
python run_train.py --config configs/train_contrastive_binary_block.yaml
```

也可以直接在主入口覆盖常用字段，而不再额外手写临时 YAML：

```bash
python run_train.py --config configs/train_contrastive_binary_block.yaml --train-manifest outputs/ds002336_binary_block_model_ready/splits_subjectwise/manifest_train.csv --val-manifest outputs/ds002336_binary_block_model_ready/splits_subjectwise/manifest_val.csv --root-dir outputs/ds002336_binary_block_model_ready --output-dir outputs/contrastive_binary_block_run2 --epochs 30 --batch-size 8 --set data.expected_fmri_shape=[450,10]
```

4. 运行分类微调：

```bash
python run_finetune.py --config configs/finetune_classifier_binary_block.yaml
```

同样支持直接覆盖常用实验参数：

```bash
python run_finetune.py --config configs/finetune_classifier_binary_block.yaml --train-manifest outputs/ds002336_binary_block_model_ready/splits_subjectwise/manifest_train.csv --val-manifest outputs/ds002336_binary_block_model_ready/splits_subjectwise/manifest_val.csv --test-manifest outputs/ds002336_binary_block_model_ready/splits_subjectwise/manifest_test.csv --root-dir outputs/ds002336_binary_block_model_ready --contrastive-checkpoint outputs/contrastive_binary_block/checkpoints/best.pth --output-dir outputs/finetune_binary_block_run2 --selection-metric macro_f1
```

5. 留一被试和汇总工具：

```bash
python scripts/build_loso_splits.py --manifest outputs/ds002336_binary_block_model_ready/manifest_all.csv --output-dir outputs/ds002336_binary_block_model_ready/loso_subjectwise --val-subjects 1
python scripts/summarize_loso_metrics.py --finetune-root outputs/finetune_binary_block_loso
python scripts/demo_loso_pipeline.py
```

6. 输出内容：

- `outputs/exp01/resolved_config.yaml`
- `outputs/exp01/checkpoints/epoch_XXX.pth`
- `outputs/exp01/checkpoints/best.pth`

评估指标：

- 对比学习会输出 `loss`、`eeg_to_fmri_r1`、`fmri_to_eeg_r1`、`eeg_to_fmri_r5`、`fmri_to_eeg_r5`、`mean_r1`，以及对应的 `*_std` 字段。
- 微调会输出 `loss`、`accuracy`、`macro_f1`，以及 `accuracy_std`、`macro_f1_std`。

对比学习会在验证/测试集上输出：

- `loss`
- `eeg_to_fmri_r1`
- `fmri_to_eeg_r1`
- `eeg_to_fmri_r5`
- `fmri_to_eeg_r5`

分类微调会在验证/测试集上输出：

- `loss`
- `accuracy`
- `macro_f1`

## 8. 说明

- 对比学习默认使用对称 InfoNCE。
- 两个基础模型默认都可训练；可在配置中设置 `freeze_backbone: true` 固定其中某个编码器。
- 如果你需要加入监督任务（比如分类头、多任务 loss），建议在 `mmcontrast/trainer.py` 中扩展。
- 现在支持从 `train.resume_path` 恢复训练。
- 分类微调默认把 EEG 和 fMRI 特征拼接后做分类，也支持单模态分类。
- 仓库已移除 smoke/demo 路径，当前默认工作流是：预处理 -> 切分脚本 -> `run_train.py` -> `run_finetune.py`。`scripts/` 目录只保留切分、检查、汇总和端到端 demo 工具。

## 9. 模型源码在哪里

你要找的完整基础模型源码现在在这里：

- EEG 骨干：`mmcontrast/backbones/eeg_cbramod/cbramod.py`
- EEG Transformer 依赖：`mmcontrast/backbones/eeg_cbramod/criss_cross_transformer.py`
- fMRI 骨干：`mmcontrast/backbones/fmri_brainjepa/vision_transformer.py`
- fMRI 依赖：`mmcontrast/backbones/fmri_brainjepa/tensors.py`
- fMRI mask 工具：`mmcontrast/backbones/fmri_brainjepa/mask_utils.py`

`mmcontrast/models/eeg_adapter.py` 和 `mmcontrast/models/fmri_adapter.py` 现在只负责加载本地骨干和权重，不再从外部仓库导入。

## 10. 目前还差什么

从“代码结构完整性”来说，当前已经具备完整实验主链路。

离你真正用自己数据稳定跑起来，还剩的关键项是：

- 把你真实的 EEG/fMRI 配对数据整理成 manifest 或自定义 Dataset。
- 把真实的 `gradient_mapping_450.csv` 放到 `assets/`。
- 如果你要加载预训练模型，把权重放到 `pretrained_weights/` 并在配置里填写路径。
- 如果你后续需要更复杂的评估，比如 AUC、混淆矩阵、患者级投票、多模态缺失鲁棒性，还可以继续扩展。
- 如果你需要完全复现原始两篇工作的预训练细节，目前还没有把两边原始 trainer 的全部策略逐项迁入这个融合工程。
