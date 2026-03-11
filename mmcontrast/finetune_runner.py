from __future__ import annotations

"""微调分类的流程调度层。"""

from pathlib import Path

from .config import TrainConfig
from .distributed import get_rank
from .finetune_trainer import FinetuneTrainer


def run_finetuning(config_path: str) -> None:
    """完成配置加载、校验、配置落盘，再交给微调训练器执行。"""
    cfg = TrainConfig.load(config_path)
    cfg.validate(base_dir=str(Path(config_path).resolve().parent.parent))

    dump_resolved_config = bool(cfg.section("finetune").get("dump_resolved_config", False))
    if get_rank() == 0 and dump_resolved_config:
        # 只在主进程写配置，避免多卡下重复覆盖文件。
        cfg.dump(cfg.section("finetune").get("output_dir", "outputs/finetune"))

    trainer = FinetuneTrainer(cfg.raw)
    if bool(cfg.section("finetune").get("test_only", False)):
        trainer.test_only()
    else:
        trainer.fit()
