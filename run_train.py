from __future__ import annotations

"""对比学习训练入口脚本。"""

from mmcontrast.runner import run_training

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
# 允许直接从项目根目录执行脚本，而不依赖外部 PYTHONPATH 配置。
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    """解析命令行参数，只暴露最核心的配置文件入口。"""
    parser = argparse.ArgumentParser("EEG-fMRI contrastive training")
    parser.add_argument("--config", type=str, default="configs/train_contrastive.yaml")
    return parser.parse_args()


def main() -> None:
    """加载配置并启动对比学习训练流程。"""
    args = parse_args()
    run_training(args.config)


if __name__ == "__main__":
    main()
