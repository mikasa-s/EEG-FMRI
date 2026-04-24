from __future__ import annotations

"""对比学习训练器。"""

from contextlib import nullcontext
import os
from pathlib import Path
import time
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

from .dataset_batching import GroupedBatchSampler, resolve_sample_group_values
from .datasets import PairedEEGfMRIDataset
from .distributed import cleanup_distributed, configure_cudnn, configure_runtime_devices, gather_tensor, init_distributed, is_dist_initialized, is_main_process, runtime_summary
from .losses import BarlowTwinsPretrainLoss, PureInfoNCEPretrainLoss, SharedPrivatePretrainLoss
from .metrics import contrastive_retrieval_metrics
from .models import EEGfMRIContrastiveModel, EEGfMRISharedOnlyContrastiveModel
from .pretrain_online_monitor import PretrainOnlineMonitor


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
        train_dataset = self.maybe_limit_train_dataset(train_dataset, train_cfg)
        self.train_dataset = train_dataset
        # 多卡时由 DistributedSampler 负责切分样本，避免重复读同一数据。
        self.sampler = None
        self.batch_sampler = GroupedBatchSampler(
            train_dataset,
            batch_size=int(train_cfg.get("batch_size", 16)),
            group_field="dataset",
            shuffle=True,
            drop_last=False,
            world_size=self.world_size,
            rank=self.rank,
            seed=42,
        )
        self.train_loader = self.build_loader(
            train_dataset,
            batch_size=int(train_cfg.get("batch_size", 16)),
            sampler=None,
            batch_sampler=self.batch_sampler,
            shuffle=False,
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
            grouped_shape_summary = self.describe_loader_modal_shapes_by_group(self.train_loader)
            if grouped_shape_summary:
                print(f"Per-dataset shapes for pretrain: {grouped_shape_summary}", flush=True)

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
        self.retrieval_eval_max_samples = max(1, int(train_cfg.get("retrieval_eval_max_samples", 2000) or 2000))
        self.retrieval_eval_seed = int(train_cfg.get("retrieval_eval_seed", 42) or 42)
        self._retrieval_eval_force_sampled = False
        self._retrieval_eval_permutation: list[int] | None = None
        self._retrieval_eval_seen_indices: set[int] = set()
        self.use_amp = bool(train_cfg.get("use_amp", True))
        self.amp_dtype = str(train_cfg.get("amp_dtype", "fp16"))
        self.scaler = _build_grad_scaler(enabled=self.use_amp and self.amp_dtype == "fp16")
        self.log_interval = int(train_cfg.get("log_interval", 20))
        self.output_dir = self.resolve_path(str(train_cfg.get("output_dir", "outputs/run1")))
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.resume_path = str(train_cfg.get("resume_path", "")).strip()
        self.start_epoch = 1
        self.online_monitor: PretrainOnlineMonitor | None = None
        self.online_monitor_enabled = False

        if is_main_process():
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            visualization_cfg = train_cfg.get("visualization", {}) or {}
            online_cfg = visualization_cfg.get("online_monitor", {}) or {}
            if bool(online_cfg.get("enabled", False)):
                self.online_monitor_enabled = True
                online_output_dir = str(online_cfg.get("output_dir", "")).strip()
                resolved_online_output_dir = self.resolve_path(online_output_dir) if online_output_dir else (self.output_dir / "online_monitor")
                self.online_monitor = PretrainOnlineMonitor(
                    output_dir=resolved_online_output_dir,
                    dataset=train_dataset,
                    batch_size=int(online_cfg.get("batch_size", train_cfg.get("eval_batch_size", train_cfg.get("batch_size", 16)))),
                    num_workers=int(online_cfg.get("num_workers", train_cfg.get("num_workers", 4))),
                    pin_memory=bool(online_cfg.get("pin_memory", train_cfg.get("pin_memory", True))),
                    update_interval_steps=int(online_cfg.get("update_interval_steps", self.log_interval)),
                    max_samples=int(online_cfg.get("max_samples", 1000)),
                    random_seed=int(online_cfg.get("random_seed", 42)),
                    tsne_interval_epochs=int(online_cfg.get("tsne_interval_epochs", 1)),
                    tsne_max_points=int(online_cfg.get("tsne_max_points", online_cfg.get("max_samples", 1000))),
                    refresh_interval_sec=int(online_cfg.get("refresh_interval_sec", 10)),
                    objective_name=self.pretrain_objective,
                    projection_method=str(online_cfg.get("projection_method", "pca")),
                )
                print(f"Online monitor enabled: {resolved_online_output_dir}", flush=True)
        if (not is_main_process()) and bool((train_cfg.get("visualization", {}) or {}).get("online_monitor", {}).get("enabled", False)):
            self.online_monitor_enabled = True

        if self.online_monitor_enabled and os.name == "nt" and int(train_cfg.get("num_workers", 4)) > 0:
            self.train_loader = self.build_loader(
                train_dataset,
                batch_size=int(train_cfg.get("batch_size", 16)),
                sampler=None,
                batch_sampler=self.batch_sampler,
                shuffle=False,
                drop_last=False,
                num_workers=0,
                pin_memory=bool(train_cfg.get("pin_memory", True)),
            )
            if is_main_process():
                print("Online monitor on Windows: forcing pretrain DataLoader num_workers=0 for stability.", flush=True)

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

    @staticmethod
    def describe_loader_modal_shapes_by_group(loader: DataLoader | None) -> str:
        if loader is None:
            return ""
        dataset = loader.dataset
        try:
            group_values = resolve_sample_group_values(dataset, "dataset")
        except Exception:
            return ""
        first_indices: dict[str, int] = {}
        counts: dict[str, int] = {}
        for index, group_name in enumerate(group_values):
            group_name = str(group_name)
            counts[group_name] = counts.get(group_name, 0) + 1
            if group_name not in first_indices:
                first_indices[group_name] = index
        parts: list[str] = []
        for group_name in sorted(first_indices):
            sample = dataset[first_indices[group_name]]
            eeg = sample.get("eeg") if isinstance(sample, dict) else None
            fmri = sample.get("fmri") if isinstance(sample, dict) else None
            eeg_shape = [counts[group_name], *(list(eeg.shape) if hasattr(eeg, "shape") else [])] if eeg is not None else [counts[group_name]]
            fmri_shape = [counts[group_name], *(list(fmri.shape) if hasattr(fmri, "shape") else [])] if fmri is not None else [counts[group_name]]
            parts.append(f"{group_name}: eeg={eeg_shape}, fmri={fmri_shape}")
        return "; ".join(parts)

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

    def build_loader(self, dataset, batch_size: int, sampler=None, batch_sampler=None, shuffle=False, drop_last=False, num_workers=4, pin_memory=True):
        """统一封装 DataLoader 创建逻辑，避免训练和评估重复写参数。"""
        effective_num_workers = 0 if getattr(dataset, "is_preloaded", False) else num_workers
        common_kwargs = {
            "dataset": dataset,
            "num_workers": effective_num_workers,
            "pin_memory": pin_memory,
            "persistent_workers": effective_num_workers > 0,
        }
        if batch_sampler is not None:
            return DataLoader(batch_sampler=batch_sampler, **common_kwargs)
        return DataLoader(
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            drop_last=drop_last,
            **common_kwargs,
        )

    def maybe_limit_train_dataset(self, dataset, train_cfg: dict[str, Any]):
        visualization_cfg = train_cfg.get("visualization", {}) or {}
        online_monitor_cfg = visualization_cfg.get("online_monitor", {}) or {}
        max_samples = int(online_monitor_cfg.get("train_max_samples", train_cfg.get("max_samples", 0)) or 0)
        if max_samples <= 0 or max_samples >= len(dataset):
            return dataset
        subset_seed = int(online_monitor_cfg.get("train_max_samples_seed", train_cfg.get("max_samples_seed", 42)) or 42)
        generator = torch.Generator()
        generator.manual_seed(subset_seed)
        indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
        if is_main_process():
            print(
                f"Training dataset limited to {max_samples}/{len(dataset)} samples "
                f"(online monitor train subset, seed={subset_seed}).",
                flush=True,
            )
        return Subset(dataset, indices)

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
        if self.batch_sampler is not None and hasattr(self.batch_sampler, "set_epoch"):
            self.batch_sampler.set_epoch(epoch)

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
            if self.online_monitor is not None:
                step_interval = max(1, int(self.online_monitor.update_interval_steps))
                if step % step_interval == 0 or step == len(self.train_loader):
                    self.online_monitor.record_step(
                        epoch=epoch,
                        step=step,
                        steps_per_epoch=len(self.train_loader),
                        global_step=(epoch - 1) * max(1, len(self.train_loader)) + step,
                        train_losses={key: float(loss_dict[key].item()) for key in running.keys()},
                        lr=float(self.optimizer.param_groups[0]["lr"]),
                    )

        denom = max(1, len(self.train_loader))
        return {key: value / denom for key, value in running.items()}

    @staticmethod
    def _is_retrieval_memory_error(exc: RuntimeError) -> bool:
        message = str(exc).lower()
        return ("not enough memory" in message) or ("out of memory" in message)

    def _build_retrieval_eval_loader(self, dataset) -> DataLoader:
        batch_size = int(self.cfg["train"].get("eval_batch_size", self.cfg["train"].get("batch_size", 16)))
        eval_num_workers = 0
        if is_main_process():
            print("Retrieval eval loader: forcing num_workers=0 for stability.", flush=True)
        return self.build_loader(
            dataset,
            batch_size=batch_size,
            sampler=None,
            batch_sampler=GroupedBatchSampler(
                dataset,
                batch_size=batch_size,
                group_field="dataset",
                shuffle=False,
                drop_last=False,
                world_size=self.world_size,
                rank=self.rank,
                seed=self.retrieval_eval_seed,
            ),
            shuffle=False,
            drop_last=False,
            num_workers=eval_num_workers,
            pin_memory=bool(self.cfg["train"].get("pin_memory", True)),
        )

    def _evaluate_retrieval_on_dataset(self, dataset) -> dict[str, float]:
        self.model.eval()
        gathered_eeg: list[torch.Tensor] = []
        gathered_fmri: list[torch.Tensor] = []
        retrieval_loader = self._build_retrieval_eval_loader(dataset)

        with torch.no_grad():
            for batch in retrieval_loader:
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

    def _get_retrieval_eval_subset(self, epoch: int):
        total_samples = len(self.train_dataset)
        sample_count = min(self.retrieval_eval_max_samples, total_samples)
        if self._retrieval_eval_permutation is None or len(self._retrieval_eval_permutation) != total_samples:
            generator = torch.Generator()
            generator.manual_seed(self.retrieval_eval_seed)
            self._retrieval_eval_permutation = torch.randperm(total_samples, generator=generator).tolist()
        start = ((max(1, int(epoch)) - 1) * sample_count) % total_samples
        end = start + sample_count
        permutation = self._retrieval_eval_permutation
        if end <= total_samples:
            indices = permutation[start:end]
        else:
            indices = permutation[start:] + permutation[: end - total_samples]
        self._retrieval_eval_seen_indices.update(int(index) for index in indices)
        return Subset(self.train_dataset, indices), {
            "sample_count": int(sample_count),
            "total_samples": int(total_samples),
            "start_offset": int(start),
            "covered_samples": int(len(self._retrieval_eval_seen_indices)),
        }

    def evaluate_retrieval(self, epoch: int) -> dict[str, float]:
        if not self._retrieval_eval_force_sampled:
            try:
                return self._evaluate_retrieval_on_dataset(self.train_dataset)
            except RuntimeError as exc:
                if not self._is_retrieval_memory_error(exc):
                    raise
                self._retrieval_eval_force_sampled = True
                if is_main_process():
                    print(
                        "Retrieval eval full-set OOM: switching to sampled retrieval evaluation. "
                        f"From now on, each epoch will evaluate {self.retrieval_eval_max_samples} samples with rotating coverage across the whole training set.",
                        flush=True,
                    )
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        subset, subset_info = self._get_retrieval_eval_subset(epoch)
        if is_main_process():
            print(
                "Retrieval eval mode=sampled, "
                f"epoch={epoch:03d}, sampled={subset_info['sample_count']}/{subset_info['total_samples']}, "
                f"covered={subset_info['covered_samples']}/{subset_info['total_samples']}, "
                f"start_offset={subset_info['start_offset']}",
                flush=True,
            )
        return self._evaluate_retrieval_on_dataset(subset)

    def fit(self) -> None:
        """执行完整训练流程，并按训练 loss 选择最好 checkpoint。"""
        stage_start_time = time.perf_counter()
        best = float("inf")
        best_epoch = 0
        last_train_loss = float("nan")
        last_train_losses: dict[str, float] = {}
        last_retrieval_metrics: dict[str, float] = {}
        if self.online_monitor is not None:
            bootstrap_metrics = self.online_monitor.evaluate_retrieval(self.model, self.device)
            bootstrap_tsne = self.online_monitor.maybe_refresh_tsne(self.model, self.device, epoch=0)
            self.online_monitor.record_epoch(
                epoch=0,
                train_losses={"loss": float("nan"), "contrastive_loss": float("nan"), "band_power_loss": float("nan"), "separation_loss": float("nan")},
                retrieval_metrics=bootstrap_metrics,
                lr=float(self.optimizer.param_groups[0]["lr"]),
                tsne_report=bootstrap_tsne,
            )
        for epoch in range(self.start_epoch, self.epochs + 1):
            train_losses = self.train_one_epoch(epoch)
            last_train_loss = float(train_losses["loss"])
            last_train_losses = train_losses
            if self.online_monitor is not None:
                last_retrieval_metrics = self.online_monitor.evaluate_retrieval(self.model, self.device)
                tsne_report = self.online_monitor.maybe_refresh_tsne(self.model, self.device, epoch=epoch)
                self.online_monitor.record_epoch(
                    epoch=epoch,
                    train_losses=train_losses,
                    retrieval_metrics=last_retrieval_metrics,
                    lr=float(self.optimizer.param_groups[0]["lr"]),
                    tsne_report=tsne_report,
                )
            else:
                last_retrieval_metrics = self.evaluate_retrieval(epoch)

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
        if self.online_monitor is not None:
            self.online_monitor.mark_finished()
        self.save_metrics("final_metrics.json", final_metrics)
        cleanup_distributed()

    def train(self) -> None:
        """兼容外部习惯调用 train() 的写法。"""
        self.fit()
