@echo off
REM Windows 下的多卡启动脚本。
set NPROC=%1
if "%NPROC%"=="" set NPROC=2

set CFG=%2
if "%CFG%"=="" set CFG=configs/train_contrastive_binary_block.yaml

REM 调用 torch.distributed.run 启动 DDP 训练。
python -m torch.distributed.run --nproc_per_node=%NPROC% run_train.py --config %CFG%
