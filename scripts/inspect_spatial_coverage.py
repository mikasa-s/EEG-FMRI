from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import nibabel as nib
import numpy as np
from nilearn.datasets import load_mni152_brain_mask, load_mni152_template
from nilearn.image import math_img, mean_img, resample_to_img
from nilearn.masking import compute_epi_mask
from nilearn import plotting


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Inspect whether a BOLD file is in standard space and whether it covers the whole brain")
    parser.add_argument("--bold", required=True, help="Path to the target BOLD NIfTI file")
    parser.add_argument("--output-dir", required=True, help="Directory for report files and figures")
    return parser.parse_args()


def voxel_bbox_world(mask_data: np.ndarray, affine: np.ndarray) -> dict[str, list[float]]:
    indices = np.argwhere(mask_data > 0)
    if indices.size == 0:
        raise ValueError("Mask is empty, cannot compute bounding box")
    world = nib.affines.apply_affine(affine, indices)
    return {
        "min": [float(value) for value in world.min(axis=0)],
        "max": [float(value) for value in world.max(axis=0)],
    }


def summarize_header(img: nib.spatialimages.SpatialImage) -> dict[str, object]:
    return {
        "shape": [int(dim) for dim in img.shape],
        "affine": np.asarray(img.affine).round(6).tolist(),
        "qform_code": int(img.header["qform_code"]),
        "sform_code": int(img.header["sform_code"]),
        "qform": np.asarray(img.get_qform()).round(6).tolist(),
        "sform": np.asarray(img.get_sform()).round(6).tolist(),
    }


def save_overlay_figures(
    mean_bold_img: nib.spatialimages.SpatialImage,
    template_img: nib.spatialimages.SpatialImage,
    template_mask_in_bold: nib.spatialimages.SpatialImage,
    bold_resampled_to_template: nib.spatialimages.SpatialImage,
    output_dir: Path,
) -> None:
    template_display = plotting.plot_anat(
        template_img,
        title="MNI template with mean BOLD overlay",
        display_mode="ortho",
        cut_coords=(0, -20, 20),
    )
    template_display.add_overlay(bold_resampled_to_template, cmap="cold_hot", alpha=0.55)
    template_display.savefig(str(output_dir / "mni_with_bold_overlay.png"))
    template_display.close()

    bold_display = plotting.plot_anat(
        mean_bold_img,
        title="Mean BOLD with MNI brain mask contour",
        display_mode="ortho",
        cut_coords=(0, -20, 20),
    )
    bold_display.add_contours(template_mask_in_bold, levels=[0.5], colors="cyan")
    bold_display.savefig(str(output_dir / "bold_with_mni_contour.png"))
    bold_display.close()


def main() -> None:
    args = parse_args()
    bold_path = Path(args.bold).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bold_img = nib.load(str(bold_path))
    mean_bold_img = mean_img(bold_img) if len(bold_img.shape) == 4 else bold_img
    mean_bold_data = mean_bold_img.get_fdata(dtype=np.float32)

    template_img = load_mni152_template()
    template_mask_img = load_mni152_brain_mask()
    bold_mask_img = compute_epi_mask(mean_bold_img)

    bold_mask_resampled = resample_to_img(bold_mask_img, template_mask_img, interpolation="nearest")
    bold_mask_resampled_data = bold_mask_resampled.get_fdata(dtype=np.float32) > 0
    template_mask_data = template_mask_img.get_fdata(dtype=np.float32) > 0
    intersection = np.logical_and(bold_mask_resampled_data, template_mask_data)
    coverage_fraction = float(intersection.sum() / template_mask_data.sum())

    z_coverages: list[dict[str, float]] = []
    template_affine = template_mask_img.affine
    for z_index in range(template_mask_data.shape[2]):
        template_slice = template_mask_data[:, :, z_index]
        denom = int(template_slice.sum())
        if denom == 0:
            continue
        covered = int(np.logical_and(bold_mask_resampled_data[:, :, z_index], template_slice).sum())
        z_mm = float(nib.affines.apply_affine(template_affine, np.array([[0, 0, z_index]]))[0, 2])
        z_coverages.append(
            {
                "z_index": float(z_index),
                "z_mm": z_mm,
                "coverage_fraction": float(covered / denom),
            }
        )

    z_nonzero = [row for row in z_coverages if row["coverage_fraction"] > 0]
    top_slice = max(z_nonzero, key=lambda row: row["z_mm"]) if z_nonzero else None
    bottom_slice = min(z_nonzero, key=lambda row: row["z_mm"]) if z_nonzero else None

    bold_bbox = voxel_bbox_world(bold_mask_img.get_fdata(dtype=np.float32) > 0, mean_bold_img.affine)
    template_bbox = voxel_bbox_world(template_mask_data, template_mask_img.affine)

    template_mask_in_bold = resample_to_img(template_mask_img, mean_bold_img, interpolation="nearest")
    bold_resampled_to_template = resample_to_img(mean_bold_img, template_img, interpolation="continuous")
    save_overlay_figures(
        mean_bold_img=mean_bold_img,
        template_img=template_img,
        template_mask_in_bold=template_mask_in_bold,
        bold_resampled_to_template=bold_resampled_to_template,
        output_dir=output_dir,
    )

    report = {
        "bold_path": str(bold_path),
        "bold_header": summarize_header(bold_img),
        "mean_bold_header": summarize_header(mean_bold_img),
        "template_header": summarize_header(template_img),
        "bold_world_bbox_mm": bold_bbox,
        "template_world_bbox_mm": template_bbox,
        "template_brain_coverage_fraction": coverage_fraction,
        "top_covered_slice": top_slice,
        "bottom_covered_slice": bottom_slice,
        "notes": [
            "qform_code or sform_code equal to 1 means the header does not explicitly declare MNI space.",
            "Visual overlay and coverage fraction should be used together with header fields for interpretation.",
        ],
    }

    with open(output_dir / "spatial_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    with open(output_dir / "z_coverage.csv", "w", encoding="utf-8") as handle:
        handle.write("z_index,z_mm,coverage_fraction\n")
        for row in z_coverages:
            handle.write(f"{int(row['z_index'])},{row['z_mm']:.3f},{row['coverage_fraction']:.6f}\n")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()