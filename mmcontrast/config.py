from __future__ import annotations

"""配置加载、落盘与基础校验逻辑。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml


@dataclass
class TrainConfig:
    """对 YAML 配置做轻量封装，避免在训练器里直接处理原始字典。"""

    raw: dict[str, Any]

    @staticmethod
    def load(path: str) -> "TrainConfig":
        """从 YAML 文件读取原始配置。"""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return TrainConfig(raw=data)

    def get(self, key: str, default: Any = None) -> Any:
        """读取顶层字段。"""
        return self.raw.get(key, default)

    def section(self, key: str) -> dict[str, Any]:
        """读取某个配置分区，不存在时返回空字典。"""
        return self.raw.get(key, {})

    def dump(self, output_dir: str) -> None:
        """把解析后的配置写入输出目录，方便复现实验。"""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "resolved_config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(self.raw, f, sort_keys=False)

    def validate(self, base_dir: str | None = None) -> None:
        """检查关键路径和字段是否齐全，尽量在训练前失败而不是训练中途失败。"""
        root = Path(base_dir).resolve() if base_dir else Path.cwd()

        data_cfg = self.section("data")
        eeg_cfg = self.section("eeg_model")
        fmri_cfg = self.section("fmri_model")
        train_cfg = self.section("train")

        required_sections = ["train", "data", "eeg_model", "fmri_model"]
        for section_name in required_sections:
            if section_name not in self.raw:
                raise ValueError(f"Missing required config section: {section_name}")

        # 对比学习和微调都依赖训练集 manifest，因此这里先统一检查。
        manifest_value = str(data_cfg.get("train_manifest_csv", data_cfg.get("manifest_csv", "")))
        manifest_path = root / manifest_value
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest CSV not found: {manifest_path}")

        for split_key in ["val_manifest_csv", "test_manifest_csv"]:
            split_path = str(data_cfg.get(split_key, "")).strip()
            if split_path:
                resolved = root / split_path
                if not resolved.exists():
                    raise FileNotFoundError(f"Manifest CSV not found: {resolved}")

        gradient_path = root / str(fmri_cfg.get("gradient_csv_path", ""))
        if not gradient_path.exists():
            raise FileNotFoundError(f"Gradient CSV not found: {gradient_path}")

        for checkpoint_key, cfg_section in [("EEG", eeg_cfg), ("fMRI", fmri_cfg)]:
            checkpoint_path = str(cfg_section.get("checkpoint_path", "")).strip()
            if checkpoint_path:
                resolved = root / checkpoint_path
                if not resolved.exists():
                    raise FileNotFoundError(f"{checkpoint_key} checkpoint not found: {resolved}")

        resume_path = str(train_cfg.get("resume_path", "")).strip()
        if resume_path:
            resolved = root / resume_path
            if not resolved.exists():
                raise FileNotFoundError(f"Resume checkpoint not found: {resolved}")

        if "finetune" in self.raw:
            finetune_cfg = self.section("finetune")
            if "num_classes" not in finetune_cfg:
                raise ValueError("Missing finetune.num_classes in config")
            contrastive_checkpoint = str(finetune_cfg.get("contrastive_checkpoint_path", "")).strip()
            if contrastive_checkpoint:
                resolved = root / contrastive_checkpoint
                if not resolved.exists():
                    raise FileNotFoundError(f"Contrastive checkpoint not found: {resolved}")
