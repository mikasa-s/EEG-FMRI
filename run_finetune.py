from __future__ import annotations

"""分类微调入口脚本。"""

from mmcontrast.finetune_runner import run_finetuning

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
# 允许直接从项目根目录执行脚本，而不依赖外部 PYTHONPATH 配置。
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    """解析微调所需的配置文件路径。"""
    parser = argparse.ArgumentParser("EEG-fMRI finetuning for classification")
    parser.add_argument("--config", type=str, default="configs/finetune_classifier.yaml")
    return parser.parse_args()


def main() -> None:
    """加载配置并启动下游分类微调。"""
    args = parse_args()
    run_finetuning(args.config)


if __name__ == "__main__":
    main()
