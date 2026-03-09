#!/usr/bin/env bash
set -e

# Linux/macOS 下的多卡启动脚本。
# 用法示例：bash scripts/train_multigpu.sh 4 configs/train_contrastive_binary_block.yaml
NPROC=${1:-4}
CFG=${2:-configs/train_contrastive_binary_block.yaml}

python -m torch.distributed.run \
  --nproc_per_node=${NPROC} \
  run_train.py \
  --config ${CFG}
