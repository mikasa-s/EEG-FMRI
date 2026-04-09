from __future__ import annotations

"""分类微调训练器。"""

from contextlib import nullcontext
import csv
import json
from pathlib import Path
import time
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from .datasets import PairedEEGfMRIDataset
from .distributed import cleanup_distributed, configure_cudnn, configure_runtime_devices, gather_tensor, init_distributed, is_dist_initialized, is_main_process, runtime_summary
from .metrics import classification_metrics
from .models import EEGfMRIClassifier
from .visualization import next_indexed_output_path, save_confusion_matrix, save_finetune_loss_curve


DATASET_CONFUSION_TITLES = {
    "ds002336": "XP1",
    "ds002338": "XP2",
    "ds009999": "SEED",
}


def _build_grad_scaler(enabled: bool) -> torch.amp.GradScaler | torch.cuda.amp.GradScaler:
    grad_scaler_cls = getattr(torch.amp, "GradScaler", None)
    if grad_scaler_cls is not None:
        return grad_scaler_cls("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


class FinetuneTrainer:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        # 需要在 CPU 上运行或显存不足时，可通过配置强制走 CPU。
        force_cpu = configure_runtime_devices(cfg.get("train", {}))
        self.world_size, self.rank, self.local_rank, self.device = init_distributed(force_cpu=force_cpu)
        self.cudnn_benchmark = configure_cudnn(cfg.get("train", {}), device=self.device)
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
        if is_main_process():
            fold_name = self.resolve_path(str(finetune_cfg.get("output_dir", "outputs/finetune"))).name
            train_shape = self.describe_loader_eeg_shape(self.train_loader)
            val_shape = self.describe_loader_eeg_shape(self.val_loader)
            test_shape = self.describe_loader_eeg_shape(self.test_loader)
            print(
                f"Dataset EEG shapes for {fold_name}: "
                f"train={train_shape}, val={val_shape}, test={test_shape}",
                flush=True,
            )

        model_cfg = {
            "train": dict(train_cfg),
            "data": dict(data_cfg),
            "eeg_model": dict(cfg["eeg_model"]),
            "fmri_model": dict(cfg["fmri_model"]),
            "finetune": dict(finetune_cfg),
        }
        eeg_input_shape = self.get_dataset_eeg_shape(train_dataset)
        baseline_cfg_runtime = dict(model_cfg["finetune"].get("eeg_baseline", {}) or {})
        if bool(baseline_cfg_runtime.get("enabled", False)) and eeg_input_shape is not None:
            baseline_cfg_runtime["num_channels"] = int(eeg_input_shape[0])
            if len(eeg_input_shape) >= 3:
                baseline_cfg_runtime["num_timepoints"] = int(eeg_input_shape[-2] * eeg_input_shape[-1])
            elif len(eeg_input_shape) >= 2:
                baseline_cfg_runtime["num_timepoints"] = int(eeg_input_shape[-1])
            model_cfg["finetune"]["eeg_baseline"] = baseline_cfg_runtime

        eeg_ckpt = str(model_cfg["eeg_model"].get("checkpoint_path", "")).strip()
        fmri_ckpt = str(model_cfg["fmri_model"].get("checkpoint_path", "")).strip()
        contrastive_ckpt = str(model_cfg["finetune"].get("contrastive_checkpoint_path", "")).strip()
        baseline_cfg = dict(model_cfg["finetune"].get("eeg_baseline", {}) or {})
        baseline_ckpt = str(baseline_cfg.get("checkpoint_path", "")).strip()
        model_cfg["eeg_model"]["checkpoint_path"] = str(self.resolve_path(eeg_ckpt)) if eeg_ckpt else ""
        model_cfg["fmri_model"]["checkpoint_path"] = str(self.resolve_path(fmri_ckpt)) if fmri_ckpt else ""
        model_cfg["finetune"]["contrastive_checkpoint_path"] = str(self.resolve_path(contrastive_ckpt)) if contrastive_ckpt else ""
        if baseline_ckpt:
            baseline_cfg["checkpoint_path"] = str(self.resolve_path(baseline_ckpt))
            model_cfg["finetune"]["eeg_baseline"] = baseline_cfg

        self.uses_svm_baseline = bool(baseline_cfg.get("enabled", False)) and str(baseline_cfg.get("model_name", "")).strip().lower() == "svm"
        self.model = EEGfMRIClassifier(model_cfg).to(self.device)
        if is_main_process():
            total_params, trainable_params = self.count_parameters(self.model)
            frozen_params = self.list_frozen_parameters(self.model)
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
            print(self.model.initialization_summary)
            print(f"Model params: total={total_params:,}, trainable={trainable_params:,}")
            # if frozen_params:
            #     frozen_total = sum(item[1] for item in frozen_params)
            #     frozen_desc = "; ".join([f"{name} ({count})" for name, count in frozen_params])
            #     print(f"Non-trainable params: total={frozen_total:,}; {frozen_desc}")
        if is_dist_initialized() and not self.uses_svm_baseline:
            # 只有真正进入分布式模式时才包装 DDP。
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank] if self.device.type == "cuda" else None,
                output_device=self.local_rank if self.device.type == "cuda" else None,
                find_unused_parameters=False,
            )

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = None
        self.epochs = int(finetune_cfg.get("epochs", 10))
        self.eval_interval = int(finetune_cfg.get("eval_interval", 1))
        self.log_interval = int(finetune_cfg.get("log_interval", 20))
        self.scheduler = None
        if not self.uses_svm_baseline:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=float(finetune_cfg.get("lr", 1e-4)),
                weight_decay=float(finetune_cfg.get("weight_decay", 1e-4)),
            )
            total_steps = max(1, self.epochs * len(self.train_loader))
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=float(finetune_cfg.get("min_lr", 1e-6)),
            )

        self.grad_clip = float(finetune_cfg.get("grad_clip", 0.0))
        self.early_stop_patience = int(finetune_cfg.get("early_stop_patience", 0))
        self.early_stop_min_delta = float(finetune_cfg.get("early_stop_min_delta", 0.0))
        self.use_amp = bool(finetune_cfg.get("use_amp", True))
        self.amp_dtype = str(finetune_cfg.get("amp_dtype", "fp16"))
        self.scaler = _build_grad_scaler(enabled=self.use_amp and self.amp_dtype == "fp16")

        self.output_dir = self.resolve_path(str(finetune_cfg.get("output_dir", "outputs/finetune")))
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.best_checkpoint_path = self.ckpt_dir / "best.pth"
        visualization_cfg = finetune_cfg.get("visualization", {}) or {}
        train_curve_cfg = visualization_cfg.get("train_curve", {}) or {}
        self.enable_train_curve_visualization = bool(train_curve_cfg.get("enabled", False))
        visual_output_dir = str(train_curve_cfg.get("output_dir", "")).strip()
        self.train_curve_output_dir = self.resolve_path(visual_output_dir) if visual_output_dir else (self.output_dir / "visualizations")
        eval_ckpt = str(finetune_cfg.get("eval_checkpoint_path", "")).strip()
        self.eval_checkpoint_path = self.resolve_path(eval_ckpt) if eval_ckpt else self.best_checkpoint_path
        self.test_only_mode = bool(finetune_cfg.get("test_only", False))
        if is_main_process():
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            if self.enable_train_curve_visualization:
                self.train_curve_output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
        total = sum(param.numel() for param in model.parameters())
        trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
        return total, trainable

    @staticmethod
    def list_frozen_parameters(model: torch.nn.Module) -> list[tuple[str, int]]:
        return [(name, int(param.numel())) for name, param in model.named_parameters() if not param.requires_grad]

    @staticmethod
    def describe_loader_eeg_shape(loader: DataLoader | None) -> str:
        if loader is None:
            return "None"
        dataset = loader.dataset
        sample_count = len(dataset)
        if sample_count == 0:
            return "[0]"
        try:
            sample = dataset[0]
        except Exception as exc:
            return f"unavailable({exc})"
        eeg = sample.get("eeg") if isinstance(sample, dict) else None
        if eeg is None:
            return "None"
        eeg_shape = list(eeg.shape) if hasattr(eeg, "shape") else []
        return str([sample_count, *eeg_shape])

    @staticmethod
    def get_dataset_eeg_shape(dataset) -> tuple[int, ...] | None:
        if len(dataset) == 0:
            return None
        sample = dataset[0]
        eeg = sample.get("eeg") if isinstance(sample, dict) else None
        if eeg is None or not hasattr(eeg, "shape"):
            return None
        return tuple(int(dim) for dim in eeg.shape)

    @staticmethod
    def flatten_eeg_features(eeg: torch.Tensor) -> torch.Tensor:
        if eeg.ndim <= 2:
            return eeg
        return eeg.reshape(eeg.shape[0], -1)

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

    def save_train_curve_artifacts(self, history: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not is_main_process() or not self.enable_train_curve_visualization:
            return None
        plot_path = next_indexed_output_path(self.train_curve_output_dir, "train_loss_curve", ".png")
        history_path = next_indexed_output_path(self.train_curve_output_dir, "train_loss_history", ".json")
        history_csv_path = next_indexed_output_path(self.train_curve_output_dir, "train_loss_history", ".csv")
        report = save_finetune_loss_curve(
            history=history,
            output_path=plot_path,
            title=f"Finetune Loss Curve ({self.output_dir.name})",
        )
        with open(history_path, "w", encoding="utf-8") as handle:
            json.dump(history, handle, ensure_ascii=False, indent=2)
        with open(history_csv_path, "w", encoding="utf-8", newline="") as handle:
            fieldnames = ["epoch", "train_loss", "val_loss", "val_accuracy", "val_macro_f1", "lr"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)
        report["history_path"] = str(history_path)
        report["history_csv_path"] = str(history_csv_path)
        return report

    def save_confusion_matrix_artifacts(self, split_name: str, logits: torch.Tensor, labels: torch.Tensor) -> None:
        if not is_main_process():
            return

        logits_cpu = logits.detach().float().cpu()
        labels_cpu = labels.detach().long().cpu()
        preds_cpu = logits_cpu.argmax(dim=1)
        unique_labels = sorted(set(labels_cpu.tolist()) | set(preds_cpu.tolist()))
        class_names = [str(index) for index in unique_labels]
        if len(unique_labels) == 2 and unique_labels == [0, 1]:
            class_names = ["0", "1"]

        report = save_confusion_matrix(
            labels=labels_cpu.numpy(),
            preds=preds_cpu.numpy(),
            output_path=self.output_dir / f"{split_name}_confusion_matrix.png",
            class_names=class_names,
            title=self.resolve_confusion_title(default_title=f"{split_name.title()} Confusion Matrix"),
            normalize=False,
        )
        with open(self.output_dir / f"{split_name}_confusion_matrix.json", "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)

    def resolve_confusion_title(self, default_title: str) -> str:
        data_cfg = self.cfg.get("data", {}) if isinstance(self.cfg, dict) else {}
        candidates = [
            str(data_cfg.get("root_dir", "")).strip(),
            str(data_cfg.get("train_manifest_csv", "")).strip(),
            str(data_cfg.get("val_manifest_csv", "")).strip(),
            str(data_cfg.get("test_manifest_csv", "")).strip(),
        ]
        haystack = " ".join(value.lower() for value in candidates if value)
        for dataset_name, title in DATASET_CONFUSION_TITLES.items():
            if dataset_name in haystack:
                return title
        return default_title

    def fit_svm_baseline(self) -> None:
        if hasattr(self.model, "module"):
            raise RuntimeError("SVM baseline does not support DDP execution.")
        if self.val_loader is None:
            raise ValueError("SVM baseline requires validation data for evaluation.")

        train_features = []
        train_labels = []
        for batch in self.train_loader:
            train_features.append(self.flatten_eeg_features(batch["eeg"].cpu()))
            train_labels.append(batch["label"].cpu())
        train_x = torch.cat(train_features, dim=0)
        train_y = torch.cat(train_labels, dim=0)
        self.model.eeg_encoder.fit(train_x, train_y)

        val_metrics = self.evaluate(self.val_loader, split_name="val")
        train_class_counts = {
            str(int(label.item())): int(count.item())
            for label, count in zip(*torch.unique(train_y, return_counts=True))
        }
        svm_summary = {
            "train_sample_count": int(train_x.shape[0]),
            "feature_dim": int(train_x.shape[1]) if train_x.ndim == 2 else int(train_x[0].numel()),
            "train_class_counts": train_class_counts,
            "selection_metric": self.selection_metric,
            "model_summary": self.model.eeg_encoder.summary(),
        }
        if val_metrics is not None:
            svm_summary["val_metrics"] = val_metrics
        final_metrics = {
            "epochs": 1,
            "completed_epochs": 1,
            "best_epoch": 1,
            "best_score": float(val_metrics[self.selection_metric]) if val_metrics is not None else float("nan"),
            "selection_metric": self.selection_metric,
            "last_train_loss": float("nan"),
            "early_stop_patience": self.early_stop_patience,
            "early_stop_min_delta": self.early_stop_min_delta,
            "early_stopped": False,
            "svm_baseline": True,
        }
        if val_metrics is not None:
            final_metrics["last_val_metrics"] = val_metrics
        if self.test_loader is not None:
            test_metrics = self.evaluate(self.test_loader, split_name="test", save_logits=True)
            if test_metrics is not None:
                final_metrics["test_metrics"] = test_metrics
                svm_summary["test_metrics"] = test_metrics
        self.save_metrics("svm_summary.json", svm_summary)
        self.save_metrics("final_metrics.json", final_metrics)
        cleanup_distributed()

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """恢复最佳模型权重，用于最终测试集评估。"""
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_state = checkpoint["model"]
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(model_state, strict=False)
        else:
            self.model.load_state_dict(model_state, strict=False)

    def save_logits_artifacts(self, split_name: str, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """保存分类 logits 明细与摘要，便于排查异常输出。"""
        if not is_main_process():
            return

        logits_cpu = logits.detach().float().cpu()
        labels_cpu = labels.detach().long().cpu()
        preds_cpu = logits_cpu.argmax(dim=1)
        probs_cpu = torch.softmax(logits_cpu, dim=1)
        prob_max_cpu = probs_cpu.max(dim=1).values
        topk_values = torch.topk(logits_cpu, k=min(2, logits_cpu.shape[1]), dim=1).values
        if logits_cpu.shape[1] > 1:
            margin_cpu = topk_values[:, 0] - topk_values[:, 1]
        else:
            margin_cpu = topk_values[:, 0]
        abs_max_per_row = logits_cpu.abs().max(dim=1).values
        top_row_count = min(10, logits_cpu.shape[0])
        top_indices = torch.topk(abs_max_per_row, k=top_row_count).indices.tolist() if top_row_count > 0 else []

        summary = {
            "sample_count": int(logits_cpu.shape[0]),
            "num_classes": int(logits_cpu.shape[1]) if logits_cpu.ndim == 2 else 0,
            "contains_nan": bool(torch.isnan(logits_cpu).any().item()),
            "contains_inf": bool(torch.isinf(logits_cpu).any().item()),
            "logit_min": float(logits_cpu.min().item()),
            "logit_max": float(logits_cpu.max().item()),
            "logit_mean": float(logits_cpu.mean().item()),
            "logit_std": float(logits_cpu.std(unbiased=False).item()) if logits_cpu.numel() > 1 else 0.0,
            "prob_max_mean": float(prob_max_cpu.mean().item()),
            "prob_max_min": float(prob_max_cpu.min().item()),
            "prob_max_max": float(prob_max_cpu.max().item()),
            "margin_mean": float(margin_cpu.mean().item()),
            "margin_min": float(margin_cpu.min().item()),
            "margin_max": float(margin_cpu.max().item()),
            "top_abs_logit_rows": [
                {
                    "sample_index": int(index),
                    "label": int(labels_cpu[index].item()),
                    "pred": int(preds_cpu[index].item()),
                    "abs_max_logit": float(abs_max_per_row[index].item()),
                    "prob_max": float(prob_max_cpu[index].item()),
                    "margin": float(margin_cpu[index].item()),
                    "logits": [float(value) for value in logits_cpu[index].tolist()],
                }
                for index in top_indices
            ],
        }

        with open(self.output_dir / f"{split_name}_logits_summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)

        header = ["sample_index", "label", "pred", "prob_max", "margin"] + [f"logit_{class_idx}" for class_idx in range(logits_cpu.shape[1])]
        with open(self.output_dir / f"{split_name}_logits.csv", "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for sample_index in range(logits_cpu.shape[0]):
                writer.writerow(
                    [
                        sample_index,
                        int(labels_cpu[sample_index].item()),
                        int(preds_cpu[sample_index].item()),
                        float(prob_max_cpu[sample_index].item()),
                        float(margin_cpu[sample_index].item()),
                        *[float(value) for value in logits_cpu[sample_index].tolist()],
                    ]
                )

    def train_one_epoch(self, epoch: int) -> float:
        """执行一个 epoch 的下游分类训练。"""
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)

        self.model.train()
        running = 0.0

        for step, batch in enumerate(self.train_loader, start=1):
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

        epoch_loss = running / max(1, len(self.train_loader))
        return epoch_loss

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader | None,
        split_name: str,
        save_logits: bool = False,
    ) -> dict[str, float] | None:
        """在验证集或测试集上计算分类指标。"""
        if loader is None:
            return None

        self.model.eval()
        total_loss = 0.0
        logits_all = []
        labels_all = []

        if self.uses_svm_baseline:
            for batch in loader:
                eeg = self.flatten_eeg_features(batch["eeg"].cpu())
                labels = batch["label"].to(self.device, non_blocking=True)
                probs = self.model.eeg_encoder.predict_proba(eeg).to(self.device)
                logits = torch.log(probs.clamp_min(1e-8))
                logits_all.append(gather_tensor(logits.detach()))
                labels_all.append(gather_tensor(labels.detach()))
            logits = torch.cat(logits_all, dim=0)
            labels = torch.cat(labels_all, dim=0)
            metrics = classification_metrics(logits, labels)
            metrics["loss"] = float("nan")
            if save_logits:
                self.save_logits_artifacts(split_name=split_name, logits=logits, labels=labels)
                self.save_confusion_matrix_artifacts(split_name=split_name, logits=logits, labels=labels)
            return metrics

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

        if save_logits:
            self.save_logits_artifacts(split_name=split_name, logits=logits, labels=labels)
            self.save_confusion_matrix_artifacts(split_name=split_name, logits=logits, labels=labels)

        return metrics

    def test_only(self) -> None:
        """直接加载已有微调 checkpoint 并在测试集上评估。"""
        if self.test_loader is None:
            raise ValueError("test_only requires data.test_manifest_csv to be configured.")
        if not self.eval_checkpoint_path.exists():
            raise FileNotFoundError(f"Finetune checkpoint not found for test_only: {self.eval_checkpoint_path}")

        self.load_checkpoint(self.eval_checkpoint_path)
        test_metrics = self.evaluate(self.test_loader, split_name="test", save_logits=True)
        payload = {
            "mode": "test_only",
            "checkpoint_path": str(self.eval_checkpoint_path),
            "test_metrics": test_metrics,
        }
        self.save_metrics("test_metrics.json", test_metrics or {})
        self.save_metrics("final_metrics.json", payload)
        cleanup_distributed()

    def fit(self) -> None:
        """执行完整微调流程，使用验证集指标选择最佳模型，并只对最佳模型做测试。"""
        if self.uses_svm_baseline:
            self.fit_svm_baseline()
            return
        fold_start_time = time.perf_counter()
        best = -1.0
        best_epoch = 0
        last_train_loss = float("nan")
        last_val_metrics = None
        epochs_without_improvement = 0
        early_stopped = False
        training_history: list[dict[str, Any]] = []
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            last_train_loss = train_loss

            val_metrics = None
            if self.val_loader is not None and epoch % self.eval_interval == 0:
                val_metrics = self.evaluate(self.val_loader, split_name="val")
                last_val_metrics = val_metrics
            training_history.append(
                {
                    "epoch": int(epoch),
                    "train_loss": float(train_loss),
                    "val_loss": None if val_metrics is None else float(val_metrics["loss"]),
                    "val_accuracy": None if val_metrics is None else float(val_metrics["accuracy"]),
                    "val_macro_f1": None if val_metrics is None else float(val_metrics["macro_f1"]),
                    "lr": float(self.optimizer.param_groups[0]["lr"]),
                }
            )

            if is_main_process():
                summary_parts = [
                    f"epoch={epoch:03d}/{self.epochs:03d}",
                    f"train_loss={train_loss:.6f}",
                    f"lr={self.optimizer.param_groups[0]['lr']:.3e}",
                ]
                if val_metrics is not None:
                    summary_parts.extend(
                        [
                            f"val_loss={float(val_metrics['loss']):.6f}",
                            f"val_accuracy={float(val_metrics['accuracy']):.4f}",
                            f"val_macro_f1={float(val_metrics['macro_f1']):.4f}",
                        ]
                    )
                print("Finetune " + ", ".join(summary_parts), flush=True)

            current_score = val_metrics[self.selection_metric] if val_metrics is not None else -train_loss
            improved = current_score > (best + self.early_stop_min_delta)
            if improved:
                best = current_score
                best_epoch = epoch
                epochs_without_improvement = 0
                extra = {"train_loss": train_loss, "selection_metric": self.selection_metric}
                if val_metrics is not None:
                    extra["val_metrics"] = val_metrics
                self.save_checkpoint(epoch, best, "best.pth", extra=extra)
            else:
                epochs_without_improvement += 1

            if self.early_stop_patience > 0 and epochs_without_improvement >= self.early_stop_patience:
                early_stopped = True
                if is_main_process():
                    print(
                        "Early stopping triggered: "
                        f"patience={self.early_stop_patience}, "
                        f"min_delta={self.early_stop_min_delta}, "
                        f"best_epoch={best_epoch}"
                    )
                break

        fold_elapsed_seconds = time.perf_counter() - fold_start_time
        final_metrics = {
            "epochs": self.epochs,
            "completed_epochs": epoch,
            "best_epoch": best_epoch,
            "best_score": best,
            "selection_metric": self.selection_metric,
            "last_train_loss": last_train_loss,
            "early_stop_patience": self.early_stop_patience,
            "early_stop_min_delta": self.early_stop_min_delta,
            "early_stopped": early_stopped,
            "fold_elapsed_seconds": fold_elapsed_seconds,
        }
        if last_val_metrics is not None:
            final_metrics["last_val_metrics"] = last_val_metrics
        curve_report = self.save_train_curve_artifacts(training_history)
        if curve_report is not None:
            final_metrics["train_curve_visualization"] = curve_report

        if self.test_loader is not None:
            self.load_checkpoint(self.best_checkpoint_path)
            test_metrics = self.evaluate(self.test_loader, split_name="test", save_logits=True)
            if test_metrics is not None:
                final_metrics["test_metrics"] = test_metrics
                self.save_metrics("test_metrics.json", test_metrics)
        if is_main_process():
            print(f"Finetune fold summary: fold_elapsed={fold_elapsed_seconds:.1f}s", flush=True)
        self.save_metrics("final_metrics.json", final_metrics)
        cleanup_distributed()
