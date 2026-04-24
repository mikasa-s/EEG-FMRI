from __future__ import annotations

import json
import math
import re
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data._utils.collate import default_collate

from .dataset_batching import resolve_sample_group_values
from .metrics import contrastive_retrieval_metrics
from .visualization import (
    save_embedding_groups_pca as _save_embedding_groups_pca,
    save_embedding_groups_tsne as _save_embedding_groups_tsne,
)


_PROJECTION_TITLE_PATTERN = re.compile(
    r"^\s*Online\s+(?P<method>PCA|t-SNE|TSNE)\b.*?\bepoch\s+(?P<epoch>\d+)\s*$",
    re.IGNORECASE,
)


def _normalize_projection_title(title: str) -> str:
    raw = str(title).strip()
    match = _PROJECTION_TITLE_PATTERN.match(raw)
    if not match:
        return raw
    method = str(match.group("method")).strip()
    epoch = int(match.group("epoch"))
    normalized_method = "t-SNE" if method.lower() in {"t-sne", "tsne"} else "PCA"
    return f"{normalized_method} (epoch {epoch})"


def save_embedding_groups_pca(*args, **kwargs):
    title = kwargs.get("title")
    if title is not None:
        kwargs["title"] = _normalize_projection_title(title)
    return _save_embedding_groups_pca(*args, **kwargs)


def save_embedding_groups_tsne(*args, **kwargs):
    title = kwargs.get("title")
    if title is not None:
        kwargs["title"] = _normalize_projection_title(title)
    return _save_embedding_groups_tsne(*args, **kwargs)


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


class PretrainOnlineMonitor:
    def __init__(
        self,
        *,
        output_dir: Path,
        dataset,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        update_interval_steps: int,
        max_samples: int = 1000,
        random_seed: int = 42,
        tsne_interval_epochs: int = 1,
        tsne_max_points: int = 1000,
        refresh_interval_sec: int = 10,
        objective_name: str = "contrastive",
        projection_method: str = "pca",
    ) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.output_dir / "monitor_state.json"
        self.html_path = self.output_dir / "index.html"
        self.tsne_path = self.output_dir / "tsne_latest.png"
        self.projection_epoch_dir = self.output_dir / "projection_epochs"
        self.projection_epoch_dir.mkdir(parents=True, exist_ok=True)
        self.loss_curve_path = self.output_dir / "loss_curve.png"
        self.retrieval_curve_path = self.output_dir / "retrieval_curve.png"
        self.max_samples = max(2, int(max_samples))
        self.random_seed = int(random_seed)
        self.tsne_interval_epochs = max(1, int(tsne_interval_epochs))
        self.tsne_max_points = max(3, int(tsne_max_points))
        self.refresh_interval_sec = max(2, int(refresh_interval_sec))
        self.update_interval_steps = max(1, int(update_interval_steps))
        self.objective_name = str(objective_name).strip() or "contrastive"
        projection_method_normalized = str(projection_method).strip().lower() or "pca"
        self.projection_method = projection_method_normalized if projection_method_normalized in {"pca", "tsne"} else "pca"
        self.projection_title = "PCA" if self.projection_method == "pca" else "t-SNE"
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self._pca_basis: dict[str, Any] | None = None

        sample_count = min(len(dataset), self.max_samples)
        generator = torch.Generator()
        generator.manual_seed(self.random_seed)
        if sample_count >= len(dataset):
            subset_indices = list(range(len(dataset)))
        else:
            subset_indices = torch.randperm(len(dataset), generator=generator)[:sample_count].tolist()
            subset_indices.sort()
        self.sample_indices = subset_indices
        all_group_values = resolve_sample_group_values(dataset, "dataset")
        grouped_indices: dict[str, list[int]] = {}
        for index in self.sample_indices:
            group_name = str(all_group_values[int(index)])
            grouped_indices.setdefault(group_name, []).append(int(index))
        self.grouped_sample_indices = grouped_indices
        self.state: dict[str, Any] = {
            "title": "Pretrain Online Monitor",
            "objective": self.objective_name,
            "sample_count": int(len(self.sample_indices)),
            "refresh_interval_sec": self.refresh_interval_sec,
            "update_interval_steps": self.update_interval_steps,
            "tsne_interval_epochs": self.tsne_interval_epochs,
            "tsne_max_points": self.tsne_max_points,
            "tsne_image": self.tsne_path.name,
            "tsne_version": 0,
            "tsne_report": {},
            "projection_method": self.projection_method,
            "projection_title": self.projection_title,
            "loss_curve_image": self.loss_curve_path.name,
            "retrieval_curve_image": self.retrieval_curve_path.name,
            "projection_epoch_dir": self.projection_epoch_dir.name,
            "status": {
                "phase": "initialized",
                "epoch": 0,
                "step": 0,
                "steps_per_epoch": 0,
                "global_step": 0,
                "last_updated": "",
            },
            "latest_train_losses": {},
            "latest_retrieval_metrics": {},
            "step_history": [],
            "epoch_history": [],
        }
        self.write_state()

    def _timestamp(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    def write_state(self) -> None:
        self.state["status"]["last_updated"] = self._timestamp()
        with self.state_path.open("w", encoding="utf-8") as handle:
            json.dump(self.state, handle, ensure_ascii=False, indent=2)
        with self.html_path.open("w", encoding="utf-8") as handle:
            handle.write(self.render_html())

    def record_step(
        self,
        *,
        epoch: int,
        step: int,
        steps_per_epoch: int,
        global_step: int,
        train_losses: dict[str, float],
        lr: float,
    ) -> None:
        history = self.state["step_history"]
        history.append(
            {
                "epoch": int(epoch),
                "step": int(step),
                "steps_per_epoch": int(steps_per_epoch),
                "global_step": int(global_step),
                "loss": _finite_float(train_losses.get("loss")),
                "contrastive_loss": _finite_float(train_losses.get("contrastive_loss")),
                "band_power_loss": _finite_float(train_losses.get("band_power_loss")),
                "separation_loss": _finite_float(train_losses.get("separation_loss")),
                "lr": _finite_float(lr),
            }
        )
        self.state["latest_train_losses"] = {key: _finite_float(value) for key, value in train_losses.items()}
        self.state["status"].update(
            {
                "phase": "training",
                "epoch": int(epoch),
                "step": int(step),
                "steps_per_epoch": int(steps_per_epoch),
                "global_step": int(global_step),
            }
        )
        self.write_state()

    def record_epoch(
        self,
        *,
        epoch: int,
        train_losses: dict[str, float],
        retrieval_metrics: dict[str, float],
        lr: float,
        tsne_report: dict[str, Any] | None = None,
    ) -> None:
        epoch_history = self.state["epoch_history"]
        epoch_history.append(
            {
                "epoch": int(epoch),
                "train_loss": _finite_float(train_losses.get("loss")),
                "contrastive_loss": _finite_float(train_losses.get("contrastive_loss")),
                "band_power_loss": _finite_float(train_losses.get("band_power_loss")),
                "separation_loss": _finite_float(train_losses.get("separation_loss")),
                "mean_r1": _finite_float(retrieval_metrics.get("mean_r1")),
                "mean_r5": _finite_float(retrieval_metrics.get("mean_r5")),
                "lr": _finite_float(lr),
            }
        )
        self.state["latest_train_losses"] = {key: _finite_float(value) for key, value in train_losses.items()}
        self.state["latest_retrieval_metrics"] = {key: _finite_float(value) for key, value in retrieval_metrics.items()}
        if tsne_report is not None:
            self.state["tsne_report"] = tsne_report
        self._save_summary_curves()
        self.state["status"].update({"phase": "epoch_end", "epoch": int(epoch), "step": 0})
        self.write_state()

    def mark_finished(self) -> None:
        self.state["status"]["phase"] = "finished"
        self.write_state()

    def collect_subset_outputs(self, model: torch.nn.Module, device: torch.device) -> dict[str, torch.Tensor]:
        previous_mode = model.training
        model.eval()
        outputs: dict[str, list[torch.Tensor]] = {
            "eeg_shared": [],
            "eeg_private": [],
            "fmri_shared": [],
            "eeg_embed": [],
            "fmri_embed": [],
        }
        with torch.no_grad():
            for batch_indices_by_group in self.grouped_sample_indices.values():
                for batch_start in range(0, len(batch_indices_by_group), self.batch_size):
                    batch_indices = batch_indices_by_group[batch_start : batch_start + self.batch_size]
                    samples = [self.dataset[index] for index in batch_indices]
                    batch = default_collate(samples)
                    eeg = batch["eeg"].to(device, non_blocking=True)
                    fmri = batch["fmri"].to(device, non_blocking=True)
                    batch_out = model(eeg=eeg, fmri=fmri)
                    outputs["eeg_shared"].append(batch_out["eeg_shared"].detach().cpu())
                    outputs["fmri_shared"].append(batch_out["fmri_shared"].detach().cpu())
                    outputs["eeg_embed"].append(batch_out["eeg_embed"].detach().cpu())
                    outputs["fmri_embed"].append(batch_out["fmri_embed"].detach().cpu())
                    if "eeg_private" in batch_out:
                        outputs["eeg_private"].append(batch_out["eeg_private"].detach().cpu())
        if previous_mode:
            model.train()
        return {key: torch.cat(value, dim=0) for key, value in outputs.items() if value}

    def _save_curve_plot(self, series_list: list[dict[str, Any]], output_path: Path, title: str, x_label: str = "Epoch") -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return
        valid_series = []
        for series in series_list:
            xs = []
            ys = []
            for item in series.get("values", []):
                x = item.get("x")
                y = item.get("y")
                if y is None or (isinstance(y, float) and math.isnan(y)):
                    continue
                xs.append(float(x))
                ys.append(float(y))
            if xs:
                valid_series.append({"label": series.get("label", "series"), "color": series.get("color", "#1f77b4"), "x": xs, "y": ys})
        if not valid_series:
            return
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
        for series in valid_series:
            ax.plot(series["x"], series["y"], linewidth=2.0, color=series["color"], label=series["label"])
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)

    def _save_summary_curves(self) -> None:
        epoch_history = self.state.get("epoch_history", []) or []
        self._save_curve_plot(
            [
                {
                    "label": "Total loss",
                    "color": "#1f77b4",
                    "values": [{"x": item.get("epoch"), "y": item.get("train_loss")} for item in epoch_history],
                }
            ],
            self.loss_curve_path,
            title="Train Loss Curve",
        )
        self._save_curve_plot(
            [
                {
                    "label": "Mean R@1",
                    "color": "#2ca02c",
                    "values": [{"x": item.get("epoch"), "y": item.get("mean_r1")} for item in epoch_history],
                },
                {
                    "label": "Mean R@5",
                    "color": "#ff7f0e",
                    "values": [{"x": item.get("epoch"), "y": item.get("mean_r5")} for item in epoch_history],
                },
            ],
            self.retrieval_curve_path,
            title="Retrieval Curve",
        )

    def evaluate_retrieval(self, model: torch.nn.Module, device: torch.device) -> dict[str, float]:
        outputs = self.collect_subset_outputs(model, device)
        eeg_embed = outputs.get("eeg_embed")
        fmri_embed = outputs.get("fmri_embed")
        if eeg_embed is None or fmri_embed is None:
            return {}
        return contrastive_retrieval_metrics(eeg_embed, fmri_embed)

    def maybe_refresh_tsne(self, model: torch.nn.Module, device: torch.device, epoch: int) -> dict[str, Any] | None:
        if epoch % self.tsne_interval_epochs != 0:
            return None
        outputs = self.collect_subset_outputs(model, device)
        groups: dict[str, torch.Tensor] = {
            "EEG shared": outputs["eeg_shared"],
            "fMRI shared": outputs["fmri_shared"],
        }
        if "eeg_private" in outputs:
            groups["EEG private"] = outputs["eeg_private"]
        epoch_projection_path = self.projection_epoch_dir / f"{self.projection_method}_epoch_{int(epoch):03d}.png"
        if self.projection_method == "pca":
            report = save_embedding_groups_pca(
                groups,
                output_path=epoch_projection_path,
                max_points=self.tsne_max_points,
                title=f"Online PCA ({self.objective_name}) epoch {int(epoch):03d}",
                basis=self._pca_basis,
            )
            if report.get("saved") and isinstance(report.get("basis"), dict):
                self._pca_basis = dict(report["basis"])
        else:
            report = save_embedding_groups_tsne(
                groups,
                output_path=epoch_projection_path,
                max_points=self.tsne_max_points,
                random_state=self.random_seed,
                title=f"Online t-SNE ({self.objective_name}) epoch {int(epoch):03d}",
            )
        if report.get("saved"):
            shutil.copyfile(epoch_projection_path, self.tsne_path)
            report["epoch_projection_path"] = str(epoch_projection_path)
        self.state["tsne_version"] = int(self.state.get("tsne_version", 0)) + 1
        return report

    def render_html(self) -> str:
        state_json = json.dumps(self.state, ensure_ascii=False)
        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate" />
  <meta http-equiv="Pragma" content="no-cache" />
  <meta http-equiv="Expires" content="0" />
  <title>Pretrain Online Monitor</title>
  <style>
    :root {{
      --bg: #f4f8ff;
      --panel: #ffffff;
      --panel-2: #f8fbff;
      --ink: #162033;
      --muted: #62748d;
      --line: rgba(130, 154, 196, 0.2);
      --shadow: 0 20px 48px rgba(63, 90, 160, 0.12);
    }}
    body {{
      margin: 0;
      padding: 24px;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(47, 125, 246, 0.12), transparent 24%),
        radial-gradient(circle at bottom right, rgba(15, 181, 167, 0.10), transparent 20%),
        linear-gradient(180deg, #fbfdff 0%, #f5f9ff 48%, #f1f6ff 100%);
      font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans SC", "Segoe UI", sans-serif;
      line-height: 1.55;
    }}
    h1 {{
      margin: 0 0 8px 0;
      font-size: 34px;
      font-weight: 700;
    }}
    .meta {{
      margin-bottom: 18px;
      color: var(--muted);
      font-size: 14px;
    }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 22px; }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      box-shadow: var(--shadow);
    }}
    .card h2 {{
      margin: 0 0 14px 0;
      font-size: 22px;
      font-weight: 700;
    }}
    canvas {{
      width: 100%;
      height: 360px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #fff;
    }}
    img {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #fff;
      display: block;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin: 10px 0 14px 0;
    }}
    .stat {{
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
    }}
    .label {{ font-size: 13px; color: var(--muted); }}
    .value {{
      font-size: 20px;
      font-weight: 700;
      margin-top: 6px;
      line-height: 1.35;
    }}
    code {{
      background: #eef3ff;
      padding: 2px 6px;
      border-radius: 6px;
      font-family: "Cascadia Code", "Consolas", monospace;
    }}
    .wide {{ grid-column: 1 / -1; }}
    @media (max-width: 980px) {{
      .grid {{ grid-template-columns: 1fr; }}
      .stats {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      canvas {{ height: 320px; }}
    }}
    @media (max-width: 640px) {{
      body {{ padding: 16px; }}
      .stats {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 28px; }}
      .card h2 {{ font-size: 20px; }}
    }}
  </style>
</head>
<body>
  <h1>预训练在线监控</h1>
  <div class="meta">固定抽样 <code>{self.state["sample_count"]}</code> 个样本；页面每隔 <code>{self.refresh_interval_sec}</code> 秒局部更新一次。</div>

  <div class="stats">
    <div class="stat"><div class="label">Status</div><div class="value" id="statStatus">--</div></div>
    <div class="stat"><div class="label">Epoch</div><div class="value" id="statEpoch">--</div></div>
    <div class="stat"><div class="label">Samples</div><div class="value" id="statSamples">--</div></div>
    <div class="stat"><div class="label">Updated</div><div class="value" id="statUpdated">--</div></div>
    <div class="stat"><div class="label">Train Loss</div><div class="value" id="statTrainLoss">--</div></div>
    <div class="stat"><div class="label">R@1</div><div class="value" id="statR1">--</div></div>
    <div class="stat"><div class="label">R@5</div><div class="value" id="statR5">--</div></div>
    <div class="stat"><div class="label">Objective</div><div class="value" id="statObjective">--</div></div>
  </div>

  <div class="grid">
    <div class="card"><h2>???????</h2><img id="lossCurveImage" alt="loss curve" /></div>
    <div class="card"><h2>R@1 / R@5 ??</h2><img id="retrievalCurveImage" alt="retrieval curve" /></div>
    <div class="card wide"><h2 id="projectionTitle">固定样本投影</h2><img id="tsneImage" alt="projection" /></div>
  </div>

  <script>
    let state = {state_json};

    function formatNumber(value, digits) {{
      if (digits === undefined) digits = 4;
      if (value === null || value === undefined || Number.isNaN(value)) return "--";
      return Number(value).toFixed(digits);
    }}

    function setText(id, value) {{
      const node = document.getElementById(id);
      if (!node) return;
      node.textContent = value;
    }}

    function renderState() {{
      const status = state.status || {{}};
      const latestTrain = state.latest_train_losses || {{}};
      const latestRetrieval = state.latest_retrieval_metrics || {{}};
      const statusText = status.phase === "finished" ? "end" : "training";

      setText("statStatus", statusText);
      setText("statEpoch", status.epoch != null ? String(status.epoch) : "--");
      setText("statSamples", String(state.sample_count || "--"));
      setText("statUpdated", status.last_updated || "--");
      setText("statTrainLoss", formatNumber(latestTrain.loss));
      setText("statR1", formatNumber(latestRetrieval.mean_r1));
      setText("statR5", formatNumber(latestRetrieval.mean_r5));
      setText("statObjective", state.objective || "--");
      setText("projectionTitle", "固定样本 " + (state.projection_title || "PCA"));

      const tsneImage = document.getElementById("tsneImage");
      const nextSrc = state.tsne_image + '?v=' + String(state.tsne_version || 0);
      if (tsneImage.dataset.src !== nextSrc) {{
        tsneImage.src = nextSrc;
        tsneImage.dataset.src = nextSrc;
      }}

      const lossCurveImage = document.getElementById("lossCurveImage");
      const lossCurveSrc = (state.loss_curve_image || "loss_curve.png") + '?v=' + String(state.status && state.status.epoch ? state.status.epoch : 0);
      if (lossCurveImage && lossCurveImage.dataset.src !== lossCurveSrc) {{
        lossCurveImage.src = lossCurveSrc;
        lossCurveImage.dataset.src = lossCurveSrc;
      }}

      const retrievalCurveImage = document.getElementById("retrievalCurveImage");
      const retrievalCurveSrc = (state.retrieval_curve_image || "retrieval_curve.png") + '?v=' + String(state.status && state.status.epoch ? state.status.epoch : 0);
      if (retrievalCurveImage && retrievalCurveImage.dataset.src !== retrievalCurveSrc) {{
        retrievalCurveImage.src = retrievalCurveSrc;
        retrievalCurveImage.dataset.src = retrievalCurveSrc;
      }}
    }}

    async function refreshState() {{
      try {{
        const response = await fetch('monitor_state.json?ts=' + Date.now(), {{ cache: 'no-store' }});
        if (!response.ok) return;
        state = await response.json();
        renderState();
      }} catch (error) {{
        console.error(error);
      }}
    }}

    renderState();
    window.setInterval(refreshState, {self.refresh_interval_sec * 1000});
  </script>
</body>
</html>"""
