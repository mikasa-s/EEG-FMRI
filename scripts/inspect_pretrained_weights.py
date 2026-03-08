from __future__ import annotations
from mmcontrast.checkpoint_utils import extract_state_dict, filter_compatible_state_dict, load_checkpoint_file, strip_prefixes
from mmcontrast.models.eeg_adapter import EEGCBraModAdapter
from mmcontrast.models.fmri_adapter import FMRIBrainJEPAAdapter

"""检查 EEG 和 fMRI 预训练权重格式，并验证与当前工程配置的兼容性。"""

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Inspect pretrained checkpoint formats")
    parser.add_argument("--eeg-path", type=str, default="pretrained_weights/CBraMod_pretrained_weights.pth")
    parser.add_argument("--fmri-path", type=str, default="pretrained_weights/jepa-ep300.pth.tar")
    parser.add_argument("--gradient-csv", type=str, default="assets/gradient_mapping_450.csv")
    parser.add_argument("--fmri-model-name", type=str, default="vit_base")
    parser.add_argument("--crop-size", type=int, nargs=2, default=[450, 160])
    parser.add_argument("--patch-size", type=int, default=16)
    return parser.parse_args()


def print_top_level_summary(name: str, path: Path) -> None:
    checkpoint = load_checkpoint_file(str(path))
    print(f"\n=== {name} ===")
    print(f"path: {path}")
    print(f"top_level_type: {type(checkpoint).__name__}")
    if isinstance(checkpoint, dict):
        keys = list(checkpoint.keys())
        print(f"top_level_keys_count: {len(keys)}")
        print(f"top_level_keys_head: {keys[:15]}")


def inspect_eeg(path: Path) -> None:
    print_top_level_summary("EEG checkpoint", path)
    raw_state = extract_state_dict(load_checkpoint_file(str(path)), preferred_keys=["state_dict", "model"])
    state = strip_prefixes(raw_state, prefixes=("module.",))
    model = EEGCBraModAdapter(checkpoint_path="")
    compatible, report = filter_compatible_state_dict(model.backbone, state)
    print(f"state_dict_keys: {len(state)}")
    print(f"compatible_keys: {len(compatible)}")
    print(f"shape_mismatch_keys: {report['skipped_shape_count']}")
    print(f"missing_name_keys: {report['skipped_missing_count']}")
    if report["skipped_shape"]:
        print("shape_mismatch_examples:")
        for key, src_shape, dst_shape in report["skipped_shape"][:5]:
            print(f"  - {key}: checkpoint={src_shape}, model={dst_shape}")


def inspect_fmri(path: Path, gradient_csv: Path, fmri_model_name: str, crop_size: tuple[int, int], patch_size: int) -> None:
    print_top_level_summary("fMRI checkpoint", path)
    raw_state = extract_state_dict(load_checkpoint_file(str(path)), preferred_keys=["target_encoder", "encoder", "state_dict"])
    state = strip_prefixes(raw_state, prefixes=("module.",))

    gradient = np.loadtxt(gradient_csv, delimiter=",", dtype=np.float32)
    print(f"gradient_shape: {tuple(gradient.shape)}")

    model = FMRIBrainJEPAAdapter(
        gradient_csv_path=str(gradient_csv),
        checkpoint_path="",
        model_name=fmri_model_name,
        crop_size=crop_size,
        patch_size=patch_size,
        attn_mode="normal",
        add_w="mapping",
    )
    compatible, report = filter_compatible_state_dict(model.backbone, state)
    print(f"state_dict_keys: {len(state)}")
    print(f"compatible_keys: {len(compatible)}")
    print(f"shape_mismatch_keys: {report['skipped_shape_count']}")
    print(f"missing_name_keys: {report['skipped_missing_count']}")
    if report["skipped_shape"]:
        print("shape_mismatch_examples:")
        for key, src_shape, dst_shape in report["skipped_shape"][:8]:
            print(f"  - {key}: checkpoint={src_shape}, model={dst_shape}")


def main() -> None:
    args = parse_args()
    eeg_path = PROJECT_ROOT / args.eeg_path
    fmri_path = PROJECT_ROOT / args.fmri_path
    gradient_csv = PROJECT_ROOT / args.gradient_csv

    inspect_eeg(eeg_path)
    inspect_fmri(fmri_path, gradient_csv, args.fmri_model_name, tuple(args.crop_size), args.patch_size)


if __name__ == "__main__":
    main()
