from __future__ import annotations

import argparse
import json
import mimetypes
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ONLINE_MONITOR_CURVE_COMPAT_SCRIPT = """<script>
(function () {
  if (window.__cmclCurveCompat) return;
  window.__cmclCurveCompat = true;

  function collectSeries(items, xKey, yKey) {
    return (items || [])
      .filter(function (item) {
        return item && item[xKey] !== undefined && item[yKey] !== undefined && item[yKey] !== null && !Number.isNaN(Number(item[yKey]));
      })
      .map(function (item) {
        return { x: Number(item[xKey]), y: Number(item[yKey]) };
      });
  }

  function ensureSectionLabel(targetId, label) {
    var target = document.getElementById(targetId);
    if (!target || !target.parentNode) return;
    var previous = target.previousElementSibling;
    if (previous && previous.dataset && previous.dataset.cmclLabel === "true") {
      previous.textContent = label;
      return;
    }
    if (previous && /loss curve|retrieval|r@1|r@5|projection|pca|t-sne|\?{2,}|[éèå]/i.test((previous.textContent || "").trim())) {
      previous.textContent = label;
      previous.dataset.cmclLabel = "true";
      return;
    }
    var heading = document.createElement("h3");
    heading.textContent = label;
    heading.dataset.cmclLabel = "true";
    heading.style.margin = "0 0 10px 0";
    heading.style.font = '600 16px "Microsoft YaHei", "PingFang SC", "Noto Sans SC", sans-serif';
    heading.style.color = "#25344c";
    target.parentNode.insertBefore(heading, target);
  }

  function normalizeTextLabels() {
    document.title = "Pretrain Online Monitor";
    ensureSectionLabel("lossCanvas", "训练损失曲线");
    ensureSectionLabel("lossCurveImage", "训练损失曲线");
    ensureSectionLabel("retrievalCanvas", "R@1 / R@5 检索曲线");
    ensureSectionLabel("retrievalCurveImage", "R@1 / R@5 检索曲线");

    var projectionTitle = document.getElementById("projectionTitle");
    var projectionName = ((window.state && state.projection_title) || "PCA");
    var desiredProjectionTitle = projectionName + "表征";
    if (projectionTitle) {
      projectionTitle.textContent = desiredProjectionTitle;
    }
    ensureSectionLabel("tsneImage", desiredProjectionTitle);

    var textNodes = document.querySelectorAll("h1, h2, h3, h4, p, div, span, strong");
    textNodes.forEach(function (node) {
      var text = (node.textContent || "").trim();
      if (!text) return;
      if (/^Online\s+(PCA|t-SNE|TSNE)\b.*\bepoch\b/i.test(text) || /^Online\s+.*\bepoch\s+\d+\b/i.test(text)) {
        node.textContent = desiredProjectionTitle;
      }
    });
  }

  function drawChart(canvas, seriesList, title) {
    if (!canvas || !canvas.getContext) return;
    var validSeries = (seriesList || []).filter(function (series) {
      return Array.isArray(series.values) && series.values.length > 0;
    });
    var ctx = canvas.getContext("2d");
    var width = canvas.width || 720;
    var height = canvas.height || 320;
    var margin = { top: 28, right: 24, bottom: 38, left: 54 };
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#f7f8fa";
    ctx.fillRect(0, 0, width, height);
    if (!validSeries.length) {
      ctx.fillStyle = "#6b7280";
      ctx.font = '14px "Microsoft YaHei", "PingFang SC", "Noto Sans SC", sans-serif';
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("Waiting for data", width / 2, height / 2);
      return;
    }

    var xs = [];
    var ys = [];
    validSeries.forEach(function (series) {
      series.values.forEach(function (point) {
        xs.push(point.x);
        ys.push(point.y);
      });
    });

    var xMin = Math.min.apply(null, xs);
    var xMax = Math.max.apply(null, xs);
    var yMinRaw = Math.min.apply(null, ys);
    var yMaxRaw = Math.max.apply(null, ys);
    var ySpan = Math.max(1e-6, yMaxRaw - yMinRaw);
    var yMin = yMinRaw - ySpan * 0.08;
    var yMax = yMaxRaw + ySpan * 0.08;
    var chartW = width - margin.left - margin.right;
    var chartH = height - margin.top - margin.bottom;
    var xScale = function (x) {
      return margin.left + ((x - xMin) / Math.max(1, xMax - xMin)) * chartW;
    };
    var yScale = function (y) {
      return margin.top + (1 - (y - yMin) / Math.max(1e-6, yMax - yMin)) * chartH;
    };

    ctx.strokeStyle = "#d6deea";
    ctx.lineWidth = 1;
    ctx.fillStyle = "#5b6578";
    ctx.font = '12px "Microsoft YaHei", "PingFang SC", "Noto Sans SC", sans-serif';
    for (var i = 0; i < 5; i += 1) {
      var y = margin.top + (chartH / 4) * i;
      var tickValue = yMax - ((yMax - yMin) / 4) * i;
      ctx.beginPath();
      ctx.moveTo(margin.left, y);
      ctx.lineTo(width - margin.right, y);
      ctx.stroke();
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillText(Number(tickValue).toFixed(3), margin.left - 8, y);
    }

    ctx.strokeStyle = "#25344c";
    ctx.beginPath();
    ctx.moveTo(margin.left, margin.top);
    ctx.lineTo(margin.left, height - margin.bottom);
    ctx.lineTo(width - margin.right, height - margin.bottom);
    ctx.stroke();
    ctx.fillStyle = "#25344c";
    ctx.font = '14px "Microsoft YaHei", "PingFang SC", "Noto Sans SC", sans-serif';
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    ctx.fillText(title, margin.left, 14);
    ctx.font = '12px "Microsoft YaHei", "PingFang SC", "Noto Sans SC", sans-serif';
    ctx.fillText(String(xMin), margin.left, height - margin.bottom + 20);
    ctx.textAlign = "right";
    ctx.fillText(String(xMax), width - margin.right, height - margin.bottom + 20);

    validSeries.forEach(function (series, index) {
      ctx.strokeStyle = series.color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      series.values.forEach(function (point, pointIndex) {
        var x = xScale(point.x);
        var y = yScale(point.y);
        if (pointIndex === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();

      var legendX = margin.left + index * 150;
      ctx.fillStyle = series.color;
      ctx.fillRect(legendX, height - 18, 18, 4);
      ctx.fillStyle = "#374151";
      ctx.textAlign = "left";
      ctx.fillText(series.label, legendX + 24, height - 12);
    });
  }

  function renderCurvesCompat() {
    try {
      normalizeTextLabels();
      var lossCanvas = document.getElementById("lossCanvas");
      if (lossCanvas) {
        drawChart(
          lossCanvas,
          [
            { label: "total", color: "#1d4ed8", values: collectSeries(state.step_history, "global_step", "loss") },
            { label: "contrastive", color: "#0f766e", values: collectSeries(state.step_history, "global_step", "contrastive_loss") },
            { label: "band_power", color: "#d97706", values: collectSeries(state.step_history, "global_step", "band_power_loss") },
            { label: "separation", color: "#b91c1c", values: collectSeries(state.step_history, "global_step", "separation_loss") }
          ],
          "训练损失"
        );
      }

      var retrievalCanvas = document.getElementById("retrievalCanvas");
      if (retrievalCanvas) {
        drawChart(
          retrievalCanvas,
          [
            { label: "R@1", color: "#7c3aed", values: collectSeries(state.epoch_history, "epoch", "mean_r1") },
            { label: "R@5", color: "#059669", values: collectSeries(state.epoch_history, "epoch", "mean_r5") }
          ],
          "R@1 / R@5 检索"
        );
      }
    } catch (error) {
      console.warn("CMCL monitor curve compat failed", error);
    }
  }

  var originalRenderState = typeof renderState === "function" ? renderState : null;
  if (originalRenderState) {
    window.renderState = function () {
      try {
        originalRenderState();
      } catch (error) {
        console.warn("CMCL monitor original renderState failed", error);
      }
      renderCurvesCompat();
    };
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", renderCurvesCompat);
  } else {
    renderCurvesCompat();
  }
  window.setTimeout(renderCurvesCompat, 0);
  window.setInterval(renderCurvesCompat, 2000);
})();
</script>"""

from mmcontrast.pretrain_pathing import FULL_MODE, STRICT_MODE, infer_pretrain_objective_name, resolve_pretrain_output_dir


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return payload


def list_train_configs() -> list[str]:
    config_dir = PROJECT_ROOT / "configs"
    return sorted(str(path.relative_to(PROJECT_ROOT)).replace("\\", "/") for path in config_dir.glob("train_joint*.yaml"))


def list_train_config_summaries() -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for rel_path in list_train_configs():
        config_path = (PROJECT_ROOT / rel_path).resolve()
        cfg = load_yaml(config_path)
        train_cfg = cfg.get("train", {}) or {}
        summaries.append(
            {
                "path": rel_path,
                "batch_size": int(train_cfg.get("batch_size", 128)),
            }
        )
    return summaries


def patch_online_monitor_html(content: str) -> str:
    content = content.replace("loss_curve.png", "loss_curve.svg")
    content = content.replace("retrieval_curve.png", "retrieval_curve.svg")
    content = content.replace("tsne_latest.png", "tsne_latest.svg")
    if "__cmclCurveCompat" in content:
        return content
    if "</body>" in content:
        return content.replace("</body>", ONLINE_MONITOR_CURVE_COMPAT_SCRIPT + "\n</body>")
    return content + "\n" + ONLINE_MONITOR_CURVE_COMPAT_SCRIPT


@dataclass
class JobState:
    job_id: int
    command: list[str]
    config_path: str
    pretrain_mode: str
    target_dataset: str
    held_out_subject: str
    objective_name: str
    output_root: str
    output_dir: str
    monitor_root: str
    created_at: float
    process: subprocess.Popen[str]
    logs: list[str] = field(default_factory=list)
    returncode: int | None = None
    completed_at: float | None = None


class JobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current_job: JobState | None = None
        self._job_counter = 0

    def _find_latest_monitor_dir(self) -> Path | None:
        search_roots = [
            PROJECT_ROOT / "pretrained_weights",
            PROJECT_ROOT / "outputs",
        ]
        candidates: list[Path] = []
        for root in search_roots:
            if not root.exists():
                continue
            for path in root.rglob("online_monitor"):
                if path.is_dir() and (path / "index.html").exists():
                    candidates.append(path)
        if not candidates:
            return None
        candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        return candidates[0]

    def start_job(self, payload: dict[str, Any]) -> JobState:
        with self._lock:
            if self._current_job is not None and self._current_job.process.poll() is None:
                raise RuntimeError("A pretrain job is already running.")

            config_rel = str(payload.get("config", "configs/train_joint_contrastive.yaml")).strip() or "configs/train_joint_contrastive.yaml"
            config_path = (PROJECT_ROOT / config_rel).resolve()
            cfg = load_yaml(config_path)
            pretrain_mode = str(payload.get("pretrain_mode", "")).strip().lower() or FULL_MODE
            if pretrain_mode not in {FULL_MODE, STRICT_MODE}:
                raise ValueError(f"Unsupported pretrain mode: {pretrain_mode}")
            target_dataset = str(payload.get("target_dataset", "")).strip()
            held_out_subject = str(payload.get("held_out_subject", "")).strip()
            output_root = str(payload.get("pretrain_output_root", "pretrained_weights")).strip() or "pretrained_weights"
            objective_name = infer_pretrain_objective_name(config_path, cfg)

            output_dir = resolve_pretrain_output_dir(
                project_root=PROJECT_ROOT,
                mode=pretrain_mode,
                objective_name=objective_name,
                target_dataset=target_dataset,
                held_out_subject=held_out_subject,
                output_root=output_root,
            )
            monitor_root = self._resolve_monitor_root(
                pretrain_mode=pretrain_mode,
                target_dataset=target_dataset,
                held_out_subject=held_out_subject,
                objective_name=objective_name,
                output_root=output_root,
            )

            command = [
                sys.executable,
                str(PROJECT_ROOT / "run_pretrain.py"),
                "--config",
                str(config_path),
                "--pretrain-mode",
                pretrain_mode,
            ]
            if target_dataset:
                command.extend(["--target-dataset", target_dataset])
            if held_out_subject:
                command.extend(["--held-out-subject", held_out_subject])
            if output_root:
                command.extend(["--pretrain-output-root", output_root])
            if bool(payload.get("force_cpu", False)):
                command.append("--force-cpu")

            max_samples = int(payload.get("max_samples", 1000))
            train_max_samples = int(payload.get("train_max_samples", max_samples))
            if train_max_samples <= 0:
                train_max_samples = max_samples
            tsne_interval_epochs = int(payload.get("tsne_interval_epochs", 1))
            refresh_interval_sec = int(payload.get("refresh_interval_sec", 10))
            update_interval_steps = int(payload.get("update_interval_steps", 20))
            tsne_max_points = int(payload.get("tsne_max_points", max_samples))
            projection_method = str(payload.get("projection_method", "pca")).strip().lower() or "pca"
            command.extend(
                [
                    "--set",
                    f"train.visualization.online_monitor.train_max_samples={train_max_samples}",
                    "--set",
                    "train.visualization.online_monitor.enabled=true",
                    "--set",
                    f"train.visualization.online_monitor.max_samples={max_samples}",
                    "--set",
                    f"train.visualization.online_monitor.tsne_interval_epochs={tsne_interval_epochs}",
                    "--set",
                    f"train.visualization.online_monitor.refresh_interval_sec={refresh_interval_sec}",
                    "--set",
                    f"train.visualization.online_monitor.update_interval_steps={update_interval_steps}",
                    "--set",
                    f"train.visualization.online_monitor.tsne_max_points={tsne_max_points}",
                    "--set",
                    f"train.visualization.online_monitor.projection_method={projection_method}",
                ]
            )

            extra_sets = payload.get("extra_set", []) or []
            for item in extra_sets:
                if str(item).strip():
                    command.extend(["--set", str(item).strip()])

            process = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            self._job_counter += 1
            job = JobState(
                job_id=self._job_counter,
                command=command,
                config_path=str(config_path),
                pretrain_mode=pretrain_mode,
                target_dataset=target_dataset,
                held_out_subject=held_out_subject,
                objective_name=objective_name,
                output_root=output_root,
                output_dir=str(output_dir),
                monitor_root=str(monitor_root),
                created_at=time.time(),
                process=process,
            )
            self._current_job = job
            self._start_log_threads(job)
            return job

    def _resolve_monitor_root(
        self,
        *,
        pretrain_mode: str,
        target_dataset: str,
        held_out_subject: str,
        objective_name: str,
        output_root: str,
    ) -> Path:
        if pretrain_mode == FULL_MODE:
            return resolve_pretrain_output_dir(
                project_root=PROJECT_ROOT,
                mode=pretrain_mode,
                objective_name=objective_name,
                output_root=output_root,
            ) / "online_monitor"
        if held_out_subject:
            return resolve_pretrain_output_dir(
                project_root=PROJECT_ROOT,
                mode=pretrain_mode,
                objective_name=objective_name,
                target_dataset=target_dataset,
                held_out_subject=held_out_subject,
                output_root=output_root,
            ) / "online_monitor"
        root = PROJECT_ROOT / output_root / "pretrain_strict" / target_dataset
        return root.resolve()

    def _start_log_threads(self, job: JobState) -> None:
        def _reader() -> None:
            assert job.process.stdout is not None
            for line in job.process.stdout:
                with self._lock:
                    job.logs.append(line.rstrip("\n"))
                    if len(job.logs) > 2000:
                        job.logs = job.logs[-2000:]
            returncode = job.process.wait()
            with self._lock:
                job.returncode = int(returncode)
                job.completed_at = time.time()

        thread = threading.Thread(target=_reader, daemon=True)
        thread.start()

    def stop_current_job(self) -> bool:
        with self._lock:
            job = self._current_job
            if job is None or job.process.poll() is not None:
                return False
            job.process.terminate()
            return True

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            job = self._current_job
            if job is None:
                latest_monitor_dir = self._find_latest_monitor_dir()
                monitor_url = "/monitor/index.html" if latest_monitor_dir is not None else ""
                return {
                    "running": False,
                    "monitor_url": monitor_url,
                    "monitor_root": str(latest_monitor_dir) if latest_monitor_dir is not None else "",
                    "available_configs": list_train_config_summaries(),
                }
            running = job.process.poll() is None
            monitor_dir = self._find_monitor_dir(job)
            monitor_url = "/monitor/index.html" if (monitor_dir / "index.html").exists() else ""
            return {
                "running": running,
                "job_id": job.job_id,
                "command": job.command,
                "config_path": job.config_path,
                "pretrain_mode": job.pretrain_mode,
                "target_dataset": job.target_dataset,
                "held_out_subject": job.held_out_subject,
                "objective_name": job.objective_name,
                "output_dir": job.output_dir,
                "monitor_root": job.monitor_root,
                "monitor_url": monitor_url,
                "returncode": job.returncode,
                "created_at": job.created_at,
                "completed_at": job.completed_at,
                "log_tail": job.logs[-200:],
                "available_configs": list_train_config_summaries(),
            }

    def _find_monitor_dir(self, job: JobState) -> Path:
        root = Path(job.monitor_root)
        if job.pretrain_mode == FULL_MODE or job.held_out_subject:
            return root
        if not root.exists():
            return root
        candidates = sorted(
            (path for path in root.glob(f"*/{job.objective_name}/online_monitor") if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else root

    def resolve_monitor_file(self, rel_path: str) -> Path | None:
        with self._lock:
            job = self._current_job
            if job is None:
                monitor_dir = self._find_latest_monitor_dir()
                if monitor_dir is None:
                    return None
            else:
                monitor_dir = self._find_monitor_dir(job)
        safe_rel = rel_path.lstrip("/").strip() or "index.html"
        target = (monitor_dir / safe_rel).resolve()
        try:
            target.relative_to(monitor_dir.resolve())
        except ValueError:
            return None
        if target.exists():
            return target
        if target.suffix.lower() == ".svg":
            fallback = target.with_suffix(".png")
            if fallback.exists():
                return fallback
        return None


JOB_MANAGER = JobManager()


class AppHandler(BaseHTTPRequestHandler):
    server_version = "CMCLPretrainServer/0.1"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: Path) -> None:
        mime_type, _ = mimetypes.guess_type(str(path))
        if path.suffix.lower() == ".html":
            html = path.read_text(encoding="utf-8")
            if "online_monitor" in {part.lower() for part in path.parts}:
                html = patch_online_monitor_html(html)
            content = html.encode("utf-8")
            mime_type = "text/html; charset=utf-8"
        else:
            content = path.read_bytes()
            if path.suffix.lower() == ".svg":
                mime_type = "image/svg+xml"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object.")
        return payload

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            return self._send_file(Path(__file__).resolve().parent / "index.html")
        if parsed.path == "/api/status":
            return self._send_json(JOB_MANAGER.get_status())
        if parsed.path.startswith("/monitor"):
            rel_path = parsed.path[len("/monitor") :].lstrip("/") or "index.html"
            target = JOB_MANAGER.resolve_monitor_file(rel_path)
            if target is None:
                return self._send_json({"error": "Monitor artifact not found."}, status=HTTPStatus.NOT_FOUND)
            return self._send_file(target)
        return self._send_json({"error": f"Unsupported route: {parsed.path}"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/start":
            try:
                payload = self._read_json_body()
                job = JOB_MANAGER.start_job(payload)
                return self._send_json(
                    {
                        "ok": True,
                        "job_id": job.job_id,
                        "output_dir": job.output_dir,
                        "monitor_root": job.monitor_root,
                    }
                )
            except RuntimeError as exc:
                return self._send_json({"error": str(exc)}, status=HTTPStatus.CONFLICT)
            except Exception as exc:  # noqa: BLE001
                return self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        if parsed.path == "/api/stop":
            stopped = JOB_MANAGER.stop_current_job()
            return self._send_json({"ok": bool(stopped)})
        return self._send_json({"error": f"Unsupported route: {parsed.path}"}, status=HTTPStatus.NOT_FOUND)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Local web UI for CMCL-EEG pretraining")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), AppHandler)
    print(f"Server running at http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
