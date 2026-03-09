from __future__ import annotations

"""分类微调训练器。"""

from contextlib import nullcontext
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from .datasets import PairedEEGfMRIDataset
from .distributed import cleanup_distributed, configure_runtime_devices, gather_tensor, init_distributed, is_dist_initialized, is_main_process
from .metrics import classification_metrics
from .models import EEGfMRIClassifier


class FinetuneTrainer:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        # 需要在 CPU 上运行或显存不足时，可通过配置强制走 CPU。
        force_cpu = configure_runtime_devices(cfg.get("train", {}))
        self.world_size, self.rank, self.local_rank, self.device = init_distributed(force_cpu=force_cpu)
        self.project_root = Path(__file__).resolve().parent.parent

        train_cfg = cfg["train"]
        finetune_cfg = cfg["finetune"]
        data_cfg = cfg["data"]
        self.fusion = str(finetune_cfg.get("fusion", "eeg_only"))
        selection_metric = str(finetune_cfg.get("selection_metric", "accuracy")).strip().lower()
        if selection_metric == "acc":
            selection_metric = "accuracy"
        if selection_metric == "f1":
            selection_metric = "macro_f1"
        if selection_metric not in {"accuracy", "macro_f1"}:
            raise ValueError("finetune.selection_metric must be one of: accuracy, acc, macro_f1, f1")
        self.selection_metric = selection_metric

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
        self.val_loader = self.build_optional_eval_loader(data_cfg, train_cfg, split="val")
        self.test_loader = self.build_optional_eval_loader(data_cfg, train_cfg, split="test")

        model_cfg = {
            "train": dict(train_cfg),
            "data": dict(data_cfg),
            "eeg_model": dict(cfg["eeg_model"]),
            "fmri_model": dict(cfg["fmri_model"]),
            "finetune": dict(finetune_cfg),
        }

        eeg_ckpt = str(model_cfg["eeg_model"].get("checkpoint_path", "")).strip()
        fmri_ckpt = str(model_cfg["fmri_model"].get("checkpoint_path", "")).strip()
        contrastive_ckpt = str(model_cfg["finetune"].get("contrastive_checkpoint_path", "")).strip()
        model_cfg["eeg_model"]["checkpoint_path"] = str(self.resolve_path(eeg_ckpt)) if eeg_ckpt else ""
        model_cfg["fmri_model"]["checkpoint_path"] = str(self.resolve_path(fmri_ckpt)) if fmri_ckpt else ""
        model_cfg["finetune"]["contrastive_checkpoint_path"] = str(self.resolve_path(contrastive_ckpt)) if contrastive_ckpt else ""

        self.model = EEGfMRIClassifier(model_cfg).to(self.device)
        if is_main_process():
            total_params, trainable_params = self.count_parameters(self.model)
            print(self.model.initialization_summary)
            print(f"Model params: total={total_params:,}, trainable={trainable_params:,}")
        if is_dist_initialized():
            # 只有真正进入分布式模式时才包装 DDP。
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank] if self.device.type == "cuda" else None,
                output_device=self.local_rank if self.device.type == "cuda" else None,
                find_unused_parameters=False,
            )

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(finetune_cfg.get("lr", 1e-4)),
            weight_decay=float(finetune_cfg.get("weight_decay", 1e-4)),
        )
        self.epochs = int(finetune_cfg.get("epochs", 10))
        self.eval_interval = int(finetune_cfg.get("eval_interval", 1))
        self.log_interval = int(finetune_cfg.get("log_interval", 20))
        total_steps = max(1, self.epochs * len(self.train_loader))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=float(finetune_cfg.get("min_lr", 1e-6)),
        )

        self.grad_clip = float(finetune_cfg.get("grad_clip", 0.0))
        self.use_amp = bool(finetune_cfg.get("use_amp", True))
        self.amp_dtype = str(finetune_cfg.get("amp_dtype", "fp16"))
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp and self.amp_dtype == "fp16")

        self.output_dir = self.resolve_path(str(finetune_cfg.get("output_dir", "outputs/finetune")))
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.best_checkpoint_path = self.ckpt_dir / "best.pth"
        if is_main_process():
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
        total = sum(param.numel() for param in model.parameters())
        trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
        return total, trainable

    def resolve_path(self, path_str: str) -> Path:
        """把相对路径统一转成基于项目根目录的绝对路径。"""
        path = Path(path_str)
        return path if path.is_absolute() else self.project_root / path

    def build_dataset(self, data_cfg: dict[str, Any], split: str) -> PairedEEGfMRIDataset:
        """根据 split 选择对应 manifest，构建配对数据集。"""
        dataset_cfg = dict(data_cfg)
        manifest_path = dataset_cfg.get(f"{split}_manifest_csv", dataset_cfg.get("manifest_csv", ""))
        dataset_cfg["manifest_csv"] = str(self.resolve_path(str(manifest_path)))
        if dataset_cfg.get("root_dir"):
            dataset_cfg["root_dir"] = str(self.resolve_path(str(dataset_cfg["root_dir"])))
        dataset_cfg.pop("train_manifest_csv", None)
        dataset_cfg.pop("val_manifest_csv", None)
        dataset_cfg.pop("test_manifest_csv", None)
        dataset_cfg.pop("expected_eeg_shape", None)
        dataset_cfg.pop("expected_fmri_shape", None)
        dataset_cfg["require_eeg"] = self.fusion != "fmri_only"
        dataset_cfg["require_fmri"] = self.fusion != "eeg_only"
        return PairedEEGfMRIDataset(**dataset_cfg)

    def build_loader(self, dataset, batch_size: int, sampler=None, shuffle=False, drop_last=False, num_workers=4, pin_memory=True):
        """统一封装 DataLoader 创建逻辑。"""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )

    def build_optional_eval_loader(self, data_cfg: dict[str, Any], train_cfg: dict[str, Any], split: str):
        """如果配置里存在验证/测试 manifest，就额外创建对应 loader。"""
        manifest_path = str(data_cfg.get(f"{split}_manifest_csv", "")).strip()
        if not manifest_path:
            return None
        dataset = self.build_dataset(data_cfg, split=split)
        return self.build_loader(
            dataset,
            batch_size=int(train_cfg.get("eval_batch_size", train_cfg.get("batch_size", 16))),
            shuffle=False,
            drop_last=False,
            num_workers=int(train_cfg.get("num_workers", 4)),
            pin_memory=bool(train_cfg.get("pin_memory", True)),
        )

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

    def save_metrics(self, name: str, metrics: dict[str, float]) -> None:
        """仅主进程保存评估指标，方便 LOSO 或实验汇总。"""
        if not is_main_process():
            return
        with open(self.output_dir / name, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2)

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """恢复最佳模型权重，用于最终测试集评估。"""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_state = checkpoint["model"]
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(model_state, strict=False)
        else:
            self.model.load_state_dict(model_state, strict=False)

    def train_one_epoch(self, epoch: int) -> float:
        """执行一个 epoch 的下游分类训练。"""
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)

        self.model.train()
        running = 0.0
        iterator = self.train_loader
        if is_main_process():
            iterator = tqdm(self.train_loader, desc=f"Finetune {epoch}/{self.epochs}", leave=False)

        for step, batch in enumerate(iterator, start=1):
            eeg = batch["eeg"].to(self.device, non_blocking=True) if "eeg" in batch else None
            fmri = batch["fmri"].to(self.device, non_blocking=True) if "fmri" in batch else None
            labels = batch["label"].to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            optimizer_stepped = False

            amp_enabled = self.use_amp and self.device.type == "cuda"
            amp_dtype = torch.bfloat16 if self.amp_dtype == "bf16" else torch.float16
            autocast_ctx = torch.autocast(device_type=self.device.type, dtype=amp_dtype) if amp_enabled else nullcontext()
            with autocast_ctx:
                out = self.model(eeg=eeg, fmri=fmri)
                loss = self.criterion(out["logits"], labels)

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
            running += float(loss.item())
            if is_main_process() and step % self.log_interval == 0:
                print(f"[finetune epoch {epoch:03d} | step {step:05d}] loss={running / step:.6f}")

        return running / max(1, len(self.train_loader))

    @torch.no_grad()
    def evaluate(self, loader: DataLoader | None, split_name: str) -> dict[str, float] | None:
        """在验证集或测试集上计算分类指标。"""
        if loader is None:
            return None

        self.model.eval()
        total_loss = 0.0
        logits_all = []
        labels_all = []

        for batch in loader:
            eeg = batch["eeg"].to(self.device, non_blocking=True) if "eeg" in batch else None
            fmri = batch["fmri"].to(self.device, non_blocking=True) if "fmri" in batch else None
            labels = batch["label"].to(self.device, non_blocking=True)
            out = self.model(eeg=eeg, fmri=fmri)
            loss = self.criterion(out["logits"], labels)
            total_loss += float(loss.item())
            # 评估时聚合 logits 和标签，统一在主进程上计算指标。
            logits_all.append(gather_tensor(out["logits"].detach()))
            labels_all.append(gather_tensor(labels.detach()))

        logits = torch.cat(logits_all, dim=0)
        labels = torch.cat(labels_all, dim=0)
        metrics = classification_metrics(logits, labels)
        metrics["loss"] = total_loss / max(1, len(loader))

        if is_main_process():
            summary = ", ".join([f"{key}={value:.4f}" for key, value in metrics.items()])
            print(f"[{split_name}] {summary}")
        return metrics

    def fit(self) -> None:
        """执行完整微调流程，使用验证集指标选择最佳模型，并只对最佳模型做测试。"""
        best = -1.0
        best_epoch = 0
        last_train_loss = float("nan")
        last_val_metrics = None
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            last_train_loss = train_loss
            if is_main_process():
                print(f"Finetune epoch {epoch:03d} done: train_loss={train_loss:.6f}")

            val_metrics = None
            if self.val_loader is not None and epoch % self.eval_interval == 0:
                val_metrics = self.evaluate(self.val_loader, split_name="val")
                last_val_metrics = val_metrics

            current_score = val_metrics[self.selection_metric] if val_metrics is not None else -train_loss
            if current_score > best:
                best = current_score
                best_epoch = epoch
                extra = {"train_loss": train_loss, "selection_metric": self.selection_metric}
                if val_metrics is not None:
                    extra["val_metrics"] = val_metrics
                self.save_checkpoint(epoch, best, "best.pth", extra=extra)

        final_metrics = {
            "epochs": self.epochs,
            "best_epoch": best_epoch,
            "best_score": best,
            "selection_metric": self.selection_metric,
            "last_train_loss": last_train_loss,
        }
        if last_val_metrics is not None:
            final_metrics["last_val_metrics"] = last_val_metrics

        if self.test_loader is not None:
            self.load_checkpoint(self.best_checkpoint_path)
            test_metrics = self.evaluate(self.test_loader, split_name="test")
            if test_metrics is not None:
                final_metrics["test_metrics"] = test_metrics
                self.save_metrics("test_metrics.json", test_metrics)
        self.save_metrics("final_metrics.json", final_metrics)
        cleanup_distributed()
