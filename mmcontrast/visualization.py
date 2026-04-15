from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


TITLE_FONTSIZE = 15
LABEL_FONTSIZE = 13
TICK_FONTSIZE = 11
LEGEND_FONTSIZE = 11
ANNOTATION_FONTSIZE = 11
COLORBAR_TICK_FONTSIZE = 11


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def next_indexed_output_path(output_dir: str | Path, stem: str, suffix: str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = f"{stem}_*{suffix}"
    max_index = 0
    for path in output_dir.glob(pattern):
        suffix_text = path.stem[len(stem) + 1 :]
        if suffix_text.isdigit():
            max_index = max(max_index, int(suffix_text))
    return output_dir / f"{stem}_{max_index + 1:03d}{suffix}"


def _to_numpy(x: torch.Tensor, max_items: int | None = None) -> np.ndarray:
    if max_items is not None and x.shape[0] > max_items:
        x = x[:max_items]
    return x.detach().float().cpu().numpy()


def save_shared_private_tsne(
    eeg_shared: torch.Tensor,
    eeg_private: torch.Tensor,
    fmri_shared: torch.Tensor,
    output_path: str | Path,
    max_points: int = 200,
    random_state: int = 42,
) -> dict[str, Any]:
    try:
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise ModuleNotFoundError(
            "Embedding visualization requires matplotlib and scikit-learn."
        ) from exc

    eeg_shared_np = _to_numpy(eeg_shared, max_items=max_points)
    eeg_private_np = _to_numpy(eeg_private, max_items=max_points)
    fmri_shared_np = _to_numpy(fmri_shared, max_items=max_points)

    labels = (
        ["EEG shared"] * len(eeg_shared_np)
        + ["EEG private"] * len(eeg_private_np)
        + ["fMRI shared"] * len(fmri_shared_np)
    )
    features = np.concatenate([eeg_shared_np, eeg_private_np, fmri_shared_np], axis=0)
    if features.shape[0] < 3:
        return {"saved": False, "reason": "not_enough_points"}

    perplexity = min(30, max(2, features.shape[0] // 6))
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=random_state)
    coords = tsne.fit_transform(features)

    output = Path(output_path)
    _ensure_parent(output)
    fig, ax = plt.subplots(figsize=(8, 7), dpi=160)
    color_map = {
        "EEG shared": "#1f77b4",
        "EEG private": "#d62728",
        "fMRI shared": "#2ca02c",
    }
    for label in ["EEG shared", "EEG private", "fMRI shared"]:
        mask = np.array([item == label for item in labels], dtype=bool)
        ax.scatter(coords[mask, 0], coords[mask, 1], s=18, alpha=0.75, label=label, c=color_map[label])
    ax.set_title("t-SNE of EEG/FMRI Shared-Private Representations", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("t-SNE 1", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("t-SNE 2", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return {
        "saved": True,
        "path": str(output),
        "num_points_per_group": {
            "eeg_shared": int(len(eeg_shared_np)),
            "eeg_private": int(len(eeg_private_np)),
            "fmri_shared": int(len(fmri_shared_np)),
        },
    }


def save_cross_modal_similarity_heatmap(
    eeg_shared: torch.Tensor,
    fmri_shared: torch.Tensor,
    output_path: str | Path,
    max_points: int = 48,
) -> dict[str, Any]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ModuleNotFoundError(
            "Similarity heatmap visualization requires matplotlib."
        ) from exc

    eeg = eeg_shared[:max_points].detach().float().cpu()
    fmri = fmri_shared[:max_points].detach().float().cpu()
    if eeg.shape[0] == 0 or fmri.shape[0] == 0:
        return {"saved": False, "reason": "empty_embeddings"}

    eeg = torch.nn.functional.normalize(eeg, dim=-1)
    fmri = torch.nn.functional.normalize(fmri, dim=-1)
    sim = eeg @ fmri.t()
    sim_np = sim.numpy()

    output = Path(output_path)
    _ensure_parent(output)
    fig, ax = plt.subplots(figsize=(8, 7), dpi=160)
    im = ax.imshow(sim_np, cmap="coolwarm", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_title("Cross-Modal Similarity Heatmap (EEG shared vs fMRI shared)", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("fMRI sample index", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("EEG sample index", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    diag_count = min(sim_np.shape[0], sim_np.shape[1])
    ax.plot(np.arange(diag_count), np.arange(diag_count), color="black", linewidth=0.8, linestyle="--")
    colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)

    diagonal_mean = float(np.diag(sim_np[:diag_count, :diag_count]).mean()) if diag_count > 0 else float("nan")
    off_diag_mask = ~np.eye(sim_np.shape[0], sim_np.shape[1], dtype=bool)
    off_diagonal_mean = float(sim_np[off_diag_mask].mean()) if np.any(off_diag_mask) else float("nan")
    return {
        "saved": True,
        "path": str(output),
        "num_points": int(sim_np.shape[0]),
        "diagonal_mean": diagonal_mean,
        "off_diagonal_mean": off_diagonal_mean,
    }


def save_finetune_loss_curve(
    history: list[dict[str, Any]],
    output_path: str | Path,
    title: str | None = None,
) -> dict[str, Any]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ModuleNotFoundError(
            "Finetune loss curve visualization requires matplotlib."
        ) from exc

    if not history:
        return {"saved": False, "reason": "empty_history"}

    epochs = [int(item["epoch"]) for item in history if "epoch" in item]
    train_loss = [float(item["train_loss"]) for item in history if "train_loss" in item]
    val_epochs = [
        int(item["epoch"])
        for item in history
        if item.get("val_loss") is not None and np.isfinite(float(item["val_loss"]))
    ]
    val_loss = [
        float(item["val_loss"])
        for item in history
        if item.get("val_loss") is not None and np.isfinite(float(item["val_loss"]))
    ]
    if not epochs or not train_loss:
        return {"saved": False, "reason": "missing_loss_values"}

    output = Path(output_path)
    _ensure_parent(output)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    ax.plot(epochs, train_loss, color="#1f77b4", linewidth=2.0, label="Train loss")
    if val_epochs and val_loss:
        ax.plot(val_epochs, val_loss, color="#d62728", linewidth=1.8, linestyle="--", label="Val loss")
    ax.set_title(title or "Finetune Loss Curve", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Epoch", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Loss", fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.legend(frameon=False, fontsize=LEGEND_FONTSIZE)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return {
        "saved": True,
        "path": str(output),
        "epoch_count": int(len(epochs)),
        "has_val_loss": bool(val_epochs),
        "min_train_loss": float(min(train_loss)),
        "min_val_loss": float(min(val_loss)) if val_loss else float("nan"),
    }


def save_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    output_path: str | Path,
    class_names: list[str] | None = None,
    title: str | None = None,
    normalize: bool = False,
) -> dict[str, Any]:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as pe
    except ImportError as exc:
        raise ModuleNotFoundError(
            "Confusion matrix visualization requires matplotlib."
        ) from exc

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    preds = np.asarray(preds, dtype=np.int64).reshape(-1)
    if labels.size == 0 or preds.size == 0:
        return {"saved": False, "reason": "empty_labels_or_preds"}
    if labels.shape != preds.shape:
        raise ValueError(f"labels and preds must have the same shape, got {labels.shape} and {preds.shape}")

    num_classes = int(max(labels.max(initial=0), preds.max(initial=0)) + 1)
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for label, pred in zip(labels.tolist(), preds.tolist()):
        matrix[int(label), int(pred)] += 1

    display_matrix = matrix.astype(np.float32)
    if normalize:
        row_sums = display_matrix.sum(axis=1, keepdims=True)
        valid_rows = row_sums.squeeze(-1) > 0
        display_matrix[valid_rows] = display_matrix[valid_rows] / row_sums[valid_rows]

    if class_names is None:
        class_names = [str(index) for index in range(num_classes)]
    else:
        class_names = [str(name) for name in class_names]
        if len(class_names) < num_classes:
            class_names = class_names + [str(index) for index in range(len(class_names), num_classes)]

    output = Path(output_path)
    _ensure_parent(output)
    fig, ax = plt.subplots(figsize=(7, 6), dpi=160)
    im = ax.imshow(display_matrix, cmap="Blues", aspect="auto")
    ax.set_title(title or "Confusion Matrix", fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Predicted label", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("True label", fontsize=LABEL_FONTSIZE)
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(class_names, rotation=0, ha="center", fontsize=TICK_FONTSIZE)
    ax.set_yticklabels(class_names, fontsize=TICK_FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)

    for row in range(num_classes):
        for col in range(num_classes):
            value = display_matrix[row, col]
            if normalize:
                text_value = f"{value:.2f}"
            else:
                text_value = str(int(matrix[row, col]))
            text = ax.text(
                col,
                row,
                text_value,
                ha="center",
                va="center",
                color="white" if value > display_matrix.max() * 0.5 else "black",
                fontsize=ANNOTATION_FONTSIZE,
                fontweight="semibold",
            )
            if value > display_matrix.max() * 0.5:
                text.set_path_effects([pe.Stroke(linewidth=1.8, foreground="black"), pe.Normal()])

    colorbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    vector_output = output.with_suffix(".svg")
    fig.savefig(vector_output, bbox_inches="tight")
    plt.close(fig)
    return {
        "saved": True,
        "path": str(output),
        "vector_path": str(vector_output),
        "num_classes": int(num_classes),
        "sample_count": int(labels.size),
        "normalized": bool(normalize),
        "matrix": matrix.tolist(),
    }
