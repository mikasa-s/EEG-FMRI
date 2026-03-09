from __future__ import annotations
from mmcontrast.checkpoint_utils import extract_state_dict, filter_compatible_state_dict, load_checkpoint_file, strip_prefixes
from mmcontrast.models.eeg_adapter import EEGCBraModAdapter
from mmcontrast.models.fmri_adapter import FMRINeuroSTORMAdapter

"""检查 EEG 和 fMRI 预训练权重格式，并验证与当前工程配置的兼容性。"""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Inspect pretrained checkpoint formats")
    parser.add_argument("--eeg-path", type=str, default="pretrained_weights/CBraMod_pretrained_weights.pth")
    parser.add_argument("--fmri-path", type=str, default="pretrained_weights/neurostorm.pth")
    parser.add_argument("--img-size", type=int, nargs=4, default=[96, 96, 96, 20])
    parser.add_argument("--patch-size", type=int, nargs=4, default=[6, 6, 6, 1])
    parser.add_argument("--first-window-size", type=int, nargs=4, default=[2, 2, 2, 2])
    parser.add_argument("--window-size", type=int, nargs=4, default=[4, 4, 4, 4])
    parser.add_argument("--depths", type=int, nargs=4, default=[2, 2, 6, 2])
    parser.add_argument("--num-heads", type=int, nargs=4, default=[3, 6, 12, 24])
    parser.add_argument("--embed-dim", type=int, default=24)
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


def inspect_fmri(
    path: Path,
    img_size: tuple[int, int, int, int],
    patch_size: tuple[int, int, int, int],
    first_window_size: tuple[int, int, int, int],
    window_size: tuple[int, int, int, int],
    depths: tuple[int, int, int, int],
    num_heads: tuple[int, int, int, int],
    embed_dim: int,
) -> None:
    print_top_level_summary("fMRI checkpoint", path)
    raw_state = extract_state_dict(load_checkpoint_file(str(path)), preferred_keys=["state_dict", "model", "backbone", "encoder"])
    state = strip_prefixes(raw_state, prefixes=("module.",))

    model = FMRINeuroSTORMAdapter(
        checkpoint_path="",
        img_size=img_size,
        embed_dim=embed_dim,
        patch_size=patch_size,
        first_window_size=first_window_size,
        window_size=window_size,
        depths=depths,
        num_heads=num_heads,
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

    inspect_eeg(eeg_path)
    inspect_fmri(
        fmri_path,
        tuple(args.img_size),
        tuple(args.patch_size),
        tuple(args.first_window_size),
        tuple(args.window_size),
        tuple(args.depths),
        tuple(args.num_heads),
        args.embed_dim,
    )


if __name__ == "__main__":
    main()
