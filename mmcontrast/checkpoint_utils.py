from __future__ import annotations

"""预训练权重检查与兼容加载工具。"""

from collections import OrderedDict
from typing import Iterable

import torch


def load_checkpoint_file(checkpoint_path: str):
    """从磁盘读取 checkpoint 原始对象。"""
    return torch.load(checkpoint_path, map_location="cpu")


def extract_state_dict(checkpoint, preferred_keys: Iterable[str] | None = None) -> OrderedDict[str, torch.Tensor]:
    """从常见 checkpoint 结构中提取真正的参数字典。"""
    if isinstance(checkpoint, OrderedDict):
        return checkpoint

    if isinstance(checkpoint, dict):
        if preferred_keys is not None:
            for key in preferred_keys:
                value = checkpoint.get(key)
                if isinstance(value, (dict, OrderedDict)):
                    return OrderedDict(value)

        for key in ["state_dict", "model", "encoder", "target_encoder", "backbone"]:
            value = checkpoint.get(key)
            if isinstance(value, (dict, OrderedDict)):
                return OrderedDict(value)

        if all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
            return OrderedDict(checkpoint)

    raise ValueError("Unable to extract a state_dict from checkpoint.")


def strip_prefixes(state_dict: OrderedDict[str, torch.Tensor], prefixes: Iterable[str]) -> OrderedDict[str, torch.Tensor]:
    """去掉常见的键名前缀，例如 module.。"""
    normalized = OrderedDict()
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        normalized[new_key] = value
    return normalized


def filter_compatible_state_dict(model: torch.nn.Module, state_dict: OrderedDict[str, torch.Tensor]) -> tuple[OrderedDict[str, torch.Tensor], dict]:
    """只保留名字和形状都匹配的参数，并返回检查报告。"""
    model_state = model.state_dict()
    compatible = OrderedDict()
    skipped_missing = []
    skipped_shape = []

    for key, value in state_dict.items():
        if key not in model_state:
            skipped_missing.append(key)
            continue
        if model_state[key].shape != value.shape:
            skipped_shape.append((key, tuple(value.shape), tuple(model_state[key].shape)))
            continue
        compatible[key] = value

    missing_in_checkpoint = [key for key in model_state.keys() if key not in compatible]
    report = {
        "loaded_count": len(compatible),
        "skipped_missing_count": len(skipped_missing),
        "skipped_shape_count": len(skipped_shape),
        "missing_in_checkpoint_count": len(missing_in_checkpoint),
        "skipped_missing": skipped_missing,
        "skipped_shape": skipped_shape,
        "missing_in_checkpoint": missing_in_checkpoint,
    }
    return compatible, report


def load_compatible_state_dict(
    model: torch.nn.Module,
    checkpoint_path: str,
    preferred_keys: Iterable[str] | None = None,
    prefixes: Iterable[str] = ("module.",),
) -> dict:
    """读取 checkpoint，提取 state_dict，过滤到与当前模型兼容的参数，再实际加载。"""
    checkpoint = load_checkpoint_file(checkpoint_path)
    state_dict = extract_state_dict(checkpoint, preferred_keys=preferred_keys)
    state_dict = strip_prefixes(state_dict, prefixes=prefixes)
    compatible, report = filter_compatible_state_dict(model, state_dict)
    model.load_state_dict(compatible, strict=False)
    return report
