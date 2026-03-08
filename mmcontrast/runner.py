from __future__ import annotations

"""对比学习的流程调度层。"""

from pathlib import Path

from .config import TrainConfig
from .distributed import get_rank
from .trainer import ContrastiveTrainer


def run_training(config_path: str) -> None:
    """完成配置加载、校验、配置落盘，再交给训练器执行。"""
    cfg = TrainConfig.load(config_path)
    cfg.validate(base_dir=str(Path(config_path).resolve().parent.parent))

    if get_rank() == 0:
        # 只在主进程写配置，避免多卡下重复覆盖文件。
        cfg.dump(cfg.section("train").get("output_dir", "outputs/run1"))

    trainer = ContrastiveTrainer(cfg.raw)
    trainer.fit()
