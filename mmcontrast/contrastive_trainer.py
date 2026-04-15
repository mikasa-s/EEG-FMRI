from __future__ import annotations

"""对比学习训练器。"""

from contextlib import nullcontext
from pathlib import Path
import time
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from .datasets import PairedEEGfMRIDataset
from .distributed import cleanup_distributed, configure_cudnn, configure_runtime_devices, gather_tensor, init_distributed, is_dist_initialized, is_main_process, runtime_summary
from .losses import BarlowTwinsPretrainLoss, PureInfoNCEPretrainLoss, SharedPrivatePretrainLoss
from .metrics import contrastive_retrieval_metrics
from .models import EEGfMRIContrastiveModel, EEGfMRISharedOnlyContrastiveModel


def _build_grad_scaler(enabled: bool) -> torch.amp.GradScaler | torch.cuda.amp.GradScaler:
    grad_scaler_cls = getattr(torch.amp, "GradScaler", None)
    if grad_scaler_cls is not None:
        return grad_scaler_cls("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


class ContrastiveTrainer:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        # 需要在 CPU 上运行或显存不足时，可通过配置强制走 CPU。
        force_cpu = configure_runtime_devices(cfg.get("train", {}))
        self.world_size, self.rank, self.local_rank, self.device = init_distributed(force_cpu=force_cpu)
        self.cudnn_benchmark = configure_cudnn(cfg.get("train", {}), device=self.device)
        self.project_root = Path(__file__).resolve().parent.parent

        train_cfg = cfg["train"]
        data_cfg = cfg["data"]
        self.pretrain_objective = str(train_cfg.get("pretrain_objective", "shared_private")).strip().lower()

        train_dataset = self.build_dataset(data_cfg, split="train")
        # 多卡时由 DistributedSampler 负责切分样本，避免重复读同一数据。
        self.sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False) if is_dist_initialized() else None
        self.train_loader = self.build_loader(
            train_dataset,
            batch_size=int(train_cfg.get("batch_size", 16)),
            sampler=self.sampler,
            shuffle=self.sampler is None,
            drop_last=False,
            num_workers=int(train_cfg.get("num_workers", 4)),
            pin_memory=bool(train_cfg.get("pin_memory", True)),
        )

        if is_main_process():
            train_eeg_shape, train_fmri_shape = self.describe_loader_modal_shapes(self.train_loader)
            print(
                "Dataset train shapes for pretrain: "
                f"eeg={train_eeg_shape}, fmri={train_fmri_shape}",
                flush=True,
            )

        self.model = self.build_model(cfg)
        if is_main_process():
            total_params, trainable_params = self.count_parameters(self.model)
            # frozen_params = self.list_frozen_parameters(self.model)
            runtime = runtime_summary(train_cfg, self.device, self.world_size)
            print(
                "Runtime: "
                f"device={runtime['device']}, "
                f"CUDA_VISIBLE_DEVICES={runtime['cuda_visible_devices']}, "
                f"world_size={runtime['world_size']}, "
                f"num_workers={runtime['num_workers']}, "
                f"pin_memory={runtime['pin_memory']}, "
                f"cudnn_benchmark={runtime['cudnn_benchmark']}"
            )
            if getattr(self.model, "initialization_summary", ""):
                print(self.model.initialization_summary)
            print(f"Model params: total={total_params:,}, trainable={trainable_params:,}")
            # if frozen_params:
            #     frozen_total = sum(item[1] for item in frozen_params)
            #     frozen_desc = "; ".join([f"{name} ({count})" for name, count in frozen_params])
            #     print(f"Non-trainable params: total={frozen_total:,}; {frozen_desc}")
        if is_dist_initialized():
            # 只有真正进入分布式模式时才包装 DDP。
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank] if self.device.type == "cuda" else None,
                output_device=self.local_rank if self.device.type == "cuda" else None,
                find_unused_parameters=False,
            )

        if self.pretrain_objective == "shared_private":
            self.criterion = SharedPrivatePretrainLoss(
                temperature=float(train_cfg.get("temperature", 0.07)),
                band_power_weight=float(train_cfg.get("band_power_loss_weight", 1.0)),
                separation_weight=float(train_cfg.get("separation_loss_weight", 0.1)),
            )
        elif self.pretrain_objective == "infonce":
            self.criterion = PureInfoNCEPretrainLoss(
                temperature=float(train_cfg.get("temperature", 0.07)),
            )
        elif self.pretrain_objective == "barlow_twins":
            self.criterion = BarlowTwinsPretrainLoss(
                lambda_offdiag=float(train_cfg.get("barlow_lambda_offdiag", 5e-3)),
                eps=float(train_cfg.get("barlow_eps", 1e-9)),
            )
        else:
            raise ValueError(f"Unsupported pretrain objective: {self.pretrain_objective}")
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(train_cfg.get("lr", 1e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        )
        self.epochs = int(train_cfg.get("epochs", 20))
        total_steps = max(1, self.epochs * len(self.train_loader))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=float(train_cfg.get("min_lr", 1e-6)),
        )

        self.grad_clip = float(train_cfg.get("grad_clip", 0.0))
        self.use_amp = bool(train_cfg.get("use_amp", True))
        self.amp_dtype = str(train_cfg.get("amp_dtype", "fp16"))
        self.scaler = _build_grad_scaler(enabled=self.use_amp and self.amp_dtype == "fp16")
        self.log_interval = int(train_cfg.get("log_interval", 20))
        self.output_dir = self.resolve_path(str(train_cfg.get("output_dir", "outputs/run1")))
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.resume_path = str(train_cfg.get("resume_path", "")).strip()
        self.start_epoch = 1

        if is_main_process():
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        if self.resume_path:
            self.load_checkpoint(self.resolve_path(self.resume_path))

    @staticmethod
    def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
        total = sum(param.numel() for param in model.parameters())
        trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
        return total, trainable

    @staticmethod
    def list_frozen_parameters(model: torch.nn.Module) -> list[tuple[str, int]]:
        return [(name, int(param.numel())) for name, param in model.named_parameters() if not param.requires_grad]

    @staticmethod
    def describe_loader_modal_shapes(loader: DataLoader | None) -> tuple[str, str]:
        if loader is None:
            return "None", "None"
        dataset = loader.dataset
        sample_count = len(dataset)
        if sample_count == 0:
            return "[0]", "[0]"
        try:
            sample = dataset[0]
        except Exception as exc:
            unavailable = f"unavailable({exc})"
            return unavailable, unavailable
        if not isinstance(sample, dict):
            return "None", "None"

        eeg = sample.get("eeg")
        fmri = sample.get("fmri")
        eeg_shape = str([sample_count, *(list(eeg.shape) if hasattr(eeg, "shape") else [])]) if eeg is not None else "None"
        fmri_shape = str([sample_count, *(list(fmri.shape) if hasattr(fmri, "shape") else [])]) if fmri is not None else "None"
        return eeg_shape, fmri_shape

    def resolve_path(self, path_str: str) -> Path:
        """把相对路径统一转成基于项目根目录的绝对路径。"""
        path = Path(path_str)
        return path if path.is_absolute() else self.project_root / path

    def build_dataset(self, data_cfg: dict[str, Any], split: str) -> PairedEEGfMRIDataset:
        """读取单一 manifest，构建配对数据集。"""
        dataset_cfg = dict(data_cfg)
        manifest_path = dataset_cfg.get("manifest_csv", dataset_cfg.get("train_manifest_csv", ""))
        dataset_cfg["manifest_csv"] = str(self.resolve_path(str(manifest_path)))
        if dataset_cfg.get("root_dir"):
            dataset_cfg["root_dir"] = str(self.resolve_path(str(dataset_cfg["root_dir"])))
        dataset_cfg.pop("train_manifest_csv", None)
        dataset_cfg.pop("val_manifest_csv", None)
        dataset_cfg.pop("test_manifest_csv", None)
        dataset_cfg.pop("expected_eeg_shape", None)
        dataset_cfg.pop("expected_fmri_shape", None)
        dataset_cfg["require_band_power"] = self.pretrain_objective == "shared_private"
        return PairedEEGfMRIDataset(**dataset_cfg)

    def build_loader(self, dataset, batch_size: int, sampler=None, shuffle=False, drop_last=False, num_workers=4, pin_memory=True):
        """统一封装 DataLoader 创建逻辑，避免训练和评估重复写参数。"""
        effective_num_workers = 0 if getattr(dataset, "is_preloaded", False) else num_workers
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=effective_num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=effective_num_workers > 0,
        )

    def build_model(self, cfg: dict[str, Any]) -> torch.nn.Module:
        """解析路径后实例化双塔对比学习模型。"""
        model_cfg = {
            "train": dict(cfg["train"]),
            "data": dict(cfg["data"]),
            "eeg_model": dict(cfg["eeg_model"]),
            "fmri_model": dict(cfg["fmri_model"]),
        }
        eeg_ckpt = str(model_cfg["eeg_model"].get("checkpoint_path", "")).strip()
        fmri_ckpt = str(model_cfg["fmri_model"].get("checkpoint_path", "")).strip()
        model_cfg["eeg_model"]["checkpoint_path"] = str(self.resolve_path(eeg_ckpt)) if eeg_ckpt else ""
        model_cfg["fmri_model"]["checkpoint_path"] = str(self.resolve_path(fmri_ckpt)) if fmri_ckpt else ""
        if self.pretrain_objective == "shared_private":
            return EEGfMRIContrastiveModel(model_cfg).to(self.device)
        return EEGfMRISharedOnlyContrastiveModel(model_cfg).to(self.device)

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """恢复模型、优化器和学习率调度器状态。"""
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_state = checkpoint["model"]
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(model_state, strict=False)
        else:
            self.model.load_state_dict(model_state, strict=False)

        if checkpoint.get("optimizer") is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scheduler") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.start_epoch = int(checkpoint.get("epoch", 0)) + 1

        if is_main_process():
            print(f"Resumed from checkpoint: {checkpoint_path} (next epoch: {self.start_epoch})")

    def save_checkpoint(self, epoch: int, metric: float, name: str, extra: dict[str, Any] | None = None) -> None:
        """仅主进程保存 checkpoint，避免多卡重复写盘。"""
        if not is_main_process():
            return

        model_state = self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict()
        payload = {
            "epoch": epoch,
            "model": model_state,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "metric": metric,
            "cfg": self.cfg,
        }
        if extra is not None:
            payload.update(extra)
        torch.save(payload, self.ckpt_dir / name)

    def save_metrics(self, name: str, metrics: dict[str, Any]) -> None:
        """仅主进程保存最终指标与训练摘要。"""
        if not is_main_process():
            return
        import json

        with open(self.output_dir / name, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2)

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        """执行一个 epoch 的对比学习训练。"""
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)

        self.model.train()
        running = {
            "loss": 0.0,
            "contrastive_loss": 0.0,
            "band_power_loss": 0.0,
            "separation_loss": 0.0,
        }

        for step, batch in enumerate(self.train_loader, start=1):
            eeg = batch["eeg"].to(self.device, non_blocking=True)
            fmri = batch["fmri"].to(self.device, non_blocking=True)
            band_power = batch["band_power"].to(self.device, non_blocking=True) if self.pretrain_objective == "shared_private" else None
            self.optimizer.zero_grad(set_to_none=True)
            optimizer_stepped = False

            amp_enabled = self.use_amp and self.device.type == "cuda"
            amp_dtype = torch.bfloat16 if self.amp_dtype == "bf16" else torch.float16
            autocast_ctx = torch.autocast(device_type=self.device.type, dtype=amp_dtype) if amp_enabled else nullcontext()
            with autocast_ctx:
                out = self.model(eeg=eeg, fmri=fmri)
                if self.pretrain_objective == "shared_private":
                    loss_dict = self.criterion(
                        eeg_shared=out["eeg_embed"],
                        fmri_shared=out["fmri_embed"],
                        eeg_private=out["eeg_private"],
                        band_power_pred=out["band_power_pred"],
                        band_power_target=band_power,
                    )
                else:
                    eeg_for_loss = out["eeg_shared"] if self.pretrain_objective == "barlow_twins" else out["eeg_embed"]
                    fmri_for_loss = out["fmri_shared"] if self.pretrain_objective == "barlow_twins" else out["fmri_embed"]
                    loss_dict = self.criterion(
                        eeg_shared=eeg_for_loss,
                        fmri_shared=fmri_for_loss,
                    )
                loss = loss_dict["loss"]

            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                prev_scale = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                optimizer_stepped = self.scaler.get_scale() >= prev_scale
            else:
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                optimizer_stepped = True

            if optimizer_stepped:
                self.scheduler.step()
            for key in running:
                running[key] += float(loss_dict[key].item())

        denom = max(1, len(self.train_loader))
        return {key: value / denom for key, value in running.items()}

    def evaluate_retrieval(self) -> dict[str, float]:
        """在当前训练集上计算双向检索指标（R@1/R@5）。"""
        self.model.eval()
        gathered_eeg: list[torch.Tensor] = []
        gathered_fmri: list[torch.Tensor] = []

        with torch.no_grad():
            for batch in self.train_loader:
                eeg = batch["eeg"].to(self.device, non_blocking=True)
                fmri = batch["fmri"].to(self.device, non_blocking=True)
                out = self.model(eeg=eeg, fmri=fmri)
                eeg_embed = gather_tensor(out["eeg_embed"].detach())
                fmri_embed = gather_tensor(out["fmri_embed"].detach())
                if is_main_process():
                    gathered_eeg.append(eeg_embed.cpu())
                    gathered_fmri.append(fmri_embed.cpu())

        if not is_main_process() or not gathered_eeg or not gathered_fmri:
            return {}

        eeg_all = torch.cat(gathered_eeg, dim=0)
        fmri_all = torch.cat(gathered_fmri, dim=0)
        return contrastive_retrieval_metrics(eeg_all, fmri_all)

    def fit(self) -> None:
        """执行完整训练流程，并按训练 loss 选择最好 checkpoint。"""
        stage_start_time = time.perf_counter()
        best = float("inf")
        best_epoch = 0
        last_train_loss = float("nan")
        last_train_losses: dict[str, float] = {}
        last_retrieval_metrics: dict[str, float] = {}
        for epoch in range(self.start_epoch, self.epochs + 1):
            train_losses = self.train_one_epoch(epoch)
            last_train_loss = float(train_losses["loss"])
            last_train_losses = train_losses
            last_retrieval_metrics = self.evaluate_retrieval()

            if is_main_process():
                summary_parts = [
                    f"epoch={epoch:03d}/{self.epochs:03d}",
                    f"train_loss={train_losses['loss']:.6f}",
                    f"contrastive_loss={train_losses['contrastive_loss']:.6f}",
                    f"band_power_loss={train_losses['band_power_loss']:.6f}",
                    f"separation_loss={train_losses['separation_loss']:.6f}",
                    f"lr={self.optimizer.param_groups[0]['lr']:.3e}",
                ]
                if last_retrieval_metrics:
                    summary_parts.append(f"mean_r1={last_retrieval_metrics.get('mean_r1', float('nan')):.4f}")
                    summary_parts.append(f"mean_r5={last_retrieval_metrics.get('mean_r5', float('nan')):.4f}")
                print("Contrastive " + ", ".join(summary_parts), flush=True)

            if train_losses["loss"] < best:
                best = float(train_losses["loss"])
                best_epoch = epoch
                self.save_checkpoint(epoch, best, "best.pth", extra={"train_loss": train_losses["loss"], "train_losses": train_losses})

        fold_elapsed_seconds = time.perf_counter() - stage_start_time
        final_metrics = {
            "epochs": self.epochs,
            "best_epoch": best_epoch,
            "best_score": best,
            "selection_mode": "train_loss",
            "last_train_loss": last_train_loss,
            "last_train_losses": last_train_losses,
            "last_retrieval_metrics": last_retrieval_metrics,
            "fold_elapsed_seconds": fold_elapsed_seconds,
        }
        if is_main_process():
            print(
                "Contrastive fold summary: "
                f"fold_elapsed={fold_elapsed_seconds:.1f}s",
                flush=True,
            )
        self.save_metrics("final_metrics.json", final_metrics)
        cleanup_distributed()

    def train(self) -> None:
        """兼容外部习惯调用 train() 的写法。"""
        self.fit()
