from __future__ import annotations

"""分布式训练相关的轻量工具函数。"""

import os
from typing import Any

import torch
import torch.distributed as dist


def configure_runtime_devices(train_cfg: dict | None = None) -> bool:
    """根据配置设置可见 GPU，并返回是否强制使用 CPU。"""
    train_cfg = train_cfg or {}
    force_cpu = bool(train_cfg.get("force_cpu", False))
    if force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return True

    gpu_ids = train_cfg.get("gpu_ids", [])
    gpu_count = train_cfg.get("gpu_count", None)

    normalized_gpu_ids: list[str] = []
    if isinstance(gpu_ids, str):
        normalized_gpu_ids = [token.strip() for token in gpu_ids.split(",") if token.strip()]
    elif isinstance(gpu_ids, (list, tuple)):
        normalized_gpu_ids = [str(item).strip() for item in gpu_ids if str(item).strip()]
    elif gpu_ids not in (None, ""):
        normalized_gpu_ids = [str(gpu_ids).strip()]

    if normalized_gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(normalized_gpu_ids)
        return False

    if gpu_count not in (None, ""):
        gpu_count_int = int(gpu_count)
        if gpu_count_int <= 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            return True
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in range(gpu_count_int))

    return False


def configure_cudnn(train_cfg: dict | None = None, device: torch.device | None = None) -> bool:
    """按配置启用 cuDNN benchmark，并返回最终是否启用。"""
    train_cfg = train_cfg or {}
    requested = bool(train_cfg.get("cudnn_benchmark", True))
    use_cuda = (device is not None and device.type == "cuda") or torch.cuda.is_available()
    enabled = bool(requested and use_cuda)
    torch.backends.cudnn.benchmark = enabled
    return enabled


def runtime_summary(train_cfg: dict | None, device: torch.device, world_size: int) -> dict[str, Any]:
    """汇总当前训练实际使用的运行时环境。"""
    train_cfg = train_cfg or {}
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    return {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_visible_devices": visible_devices,
        "world_size": int(world_size),
        "num_workers": int(train_cfg.get("num_workers", 4)),
        "pin_memory": bool(train_cfg.get("pin_memory", True)),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


def init_distributed(force_cpu: bool = False) -> tuple[int, int, int, torch.device]:
    """根据环境变量初始化单卡或多卡运行上下文。"""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    use_cuda = torch.cuda.is_available() and not force_cpu
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if world_size > 1 and not dist.is_initialized():
        # Windows 或 CPU 下优先使用 gloo，Linux GPU 多卡时使用 nccl。
        backend = "nccl" if use_cuda and os.name != "nt" else "gloo"
        dist.init_process_group(backend=backend)

    return world_size, rank, local_rank, device


def is_dist_initialized() -> bool:
    """判断当前是否已经进入分布式上下文。"""
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """获取总进程数。"""
    if is_dist_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_rank() -> int:
    """获取当前进程编号。"""
    if is_dist_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


def is_main_process() -> bool:
    """只有主进程负责打印日志和写文件。"""
    return get_rank() == 0


def barrier() -> None:
    """在多卡模式下同步所有进程。"""
    if is_dist_initialized():
        dist.barrier()


def gather_with_grad(x: torch.Tensor) -> torch.Tensor:
    """带梯度聚合张量，主要给对比学习 loss 使用。"""
    if not is_dist_initialized():
        return x

    world_size = get_world_size()
    rank = get_rank()
    gathered = [torch.zeros_like(x) for _ in range(world_size)]
    dist.all_gather(gathered, x)
    gathered[rank] = x
    return torch.cat(gathered, dim=0)


def gather_tensor(x: torch.Tensor) -> torch.Tensor:
    """无梯度聚合张量，主要给评估指标计算使用。"""
    if not is_dist_initialized():
        return x

    world_size = get_world_size()
    gathered = [torch.zeros_like(x) for _ in range(world_size)]
    dist.all_gather(gathered, x)
    return torch.cat(gathered, dim=0)


def cleanup_distributed() -> None:
    """训练结束后释放分布式进程组。"""
    if is_dist_initialized():
        dist.destroy_process_group()
