"""Run the pinned, unmodified CFLOW-AD implementation for one Figure 6 sample.

The baseline model and anomaly map are computed by the upstream repository;
this wrapper only validates the source image, invokes upstream entry points,
and collects the upstream visualization with provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


UPSTREAM_URL = "https://github.com/gudovskiy/cflow-ad.git"
UPSTREAM_COMMIT = "b2ebf9e673a0aa46992a3b18367ec066a57bba89"
# At 128 px, the deepest 4x4 feature map yields only 8*4*4=128 fibers for
# the default batch size, below upstream CFLOW-AD's fixed 256-fiber decoder
# batch (train.py: FIB = E//N). Use the same 256 px setting as the other classes.
INPUT_SIZES = {
    "cable": 256, "transistor": 256, "hazelnut": 256, "metal_nut": 256,
    # Documented aliases in prepare_figure6_data.py; the upstream loader
    # accepts only names from its original MVTec whitelist.
    "bottle": 256, "carpet": 256, "leather": 256, "tile": 256,
    "pill": 256, "toothbrush": 256,
}
NATIVE_CLASSES = {"cable", "transistor", "hazelnut", "metal_nut"}
ADAPTER_MANIFEST = "cflow_figure6_adapter.json"
CONFIRMED_FIGURE6_SAMPLES = {
    Path("cable/test/bent_wire/001.png"),
    Path("transistor/test/cut_lead/005.png"),
}
PIXEL_MATCHED_FIGURE6_SAMPLES = {
    Path("metal_nut/test/color/000.png"),
    Path("metal_nut/test/bent/005.png"),
    Path("hazelnut/test/crack/000.png"),
    Path("transistor/test/damaged_case/009.png"),
    Path("cable/test/missing_cable/000.png"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_images(data_root: Path, class_name: str) -> list[Path]:
    """Reproduce upstream MVTecDataset.load_dataset_folder test ordering."""
    test_dir = data_root / class_name / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Missing MVTec AD test directory: {test_dir}")
    images = []
    for defect_dir in sorted(test_dir.iterdir()):
        if defect_dir.is_dir():
            images.extend(sorted(p for p in defect_dir.iterdir() if p.name.endswith(".png")))
    return images


def sample_info(data_root: Path, relative_sample: Path) -> tuple[str, Path, Path | None, int]:
    if relative_sample.is_absolute() or ".." in relative_sample.parts:
        raise ValueError("--sample must be a relative path within --data-root")
    parts = relative_sample.parts
    if len(parts) != 4 or parts[1] != "test":
        raise ValueError("Expected --sample CLASS/test/DEFECT/FILE.png")
    class_name, _, defect_name, file_name = parts
    if class_name not in INPUT_SIZES:
        raise ValueError(f"Supported Figure 6 classes/aliases: {sorted(INPUT_SIZES)}")
    if not file_name.endswith(".png"):
        raise ValueError("The upstream MVTec AD loader selects lowercase .png files")
    image = data_root / relative_sample
    mask = None if defect_name == "good" else data_root / class_name / "ground_truth" / defect_name / f"{image.stem}_mask.png"
    if not image.is_file() or (mask is not None and not mask.is_file()):
        raise FileNotFoundError(f"Input/GT pair incomplete: {image}; {mask}")
    images = test_images(data_root, class_name)
    try:
        index = images.index(image)
    except ValueError as error:
        raise ValueError(f"Input is not selected by upstream test loader: {image}") from error
    return class_name, image, mask, index


def adapter_entry(data_root: Path, relative_sample: Path, image: Path, mask: Path | None,
                  upstream_index: int | None = None) -> dict | None:
    """Validate staged sample provenance when a Figure 6 adapter is used."""
    manifest_path = data_root / ADAPTER_MANIFEST
    if not manifest_path.is_file():
        if relative_sample.parts[0] not in NATIVE_CLASSES:
            raise FileNotFoundError(
                f"Alias {relative_sample.parts[0]} requires an adapter manifest: {manifest_path}"
            )
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unknown adapter manifest version: {manifest_path}")
    matches = [entry for entry in manifest["figure6_inputs"]
               if entry["staged_image"] == relative_sample.as_posix()]
    if not matches:
        raise ValueError(f"Sample is not one of the audited Figure 6 inputs: {relative_sample}")
    source_paths = {entry["source_image"] for entry in matches}
    if len(source_paths) != 1 or any(entry["source_image_sha256"] != sha256(image) for entry in matches):
        raise ValueError(f"Staged source image does not match its Figure 6 provenance: {image}")
    expected_mask_hashes = {entry["staged_mask_sha256"] for entry in matches}
    if len(expected_mask_hashes) != 1 or (sha256(mask) if mask else None) not in expected_mask_hashes:
        raise ValueError(f"Staged mask does not match its Figure 6 provenance: {mask}")
    if upstream_index is not None and any(entry["upstream_test_index"] != upstream_index for entry in matches):
        raise ValueError(f"CFLOW sample order differs from adapter manifest: {relative_sample}")
    source_root = Path(manifest["datasets_root"])
    original = source_root / matches[0]["source_image"]
    if not original.is_file() or sha256(original) != matches[0]["source_image_sha256"]:
        raise ValueError(f"Original source image changed or is missing: {original}")
    return {
        "adapter_manifest": str(manifest_path),
        "adapter_manifest_sha256": sha256(manifest_path),
        "source_dataset": matches[0]["dataset"],
        "source_class": matches[0]["source_class"],
        "source_image_original": str(original),
        "source_masks_original": [str(source_root / p) for p in matches[0]["source_masks"]],
        "ppt_input_ids": [entry["ppt_id"] for entry in matches],
        "upstream_class_alias": matches[0]["upstream_alias"],
        "adapter_method": manifest["adapter"],
    }


def git_head(repo: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def ensure_upstream_repo(repo: Path) -> str:
    if not (repo / "main.py").is_file() or not (repo / ".git").exists():
        raise FileNotFoundError(f"Clone the official repository first: {repo}")
    commit = git_head(repo)
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError(
            f"Unexpected CFLOW-AD revision {commit}; required {UPSTREAM_COMMIT}. "
            "Checkout the pinned revision rather than silently changing the method."
        )
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
        check=True, capture_output=True, text=True,
    )
    if status.stdout.strip():
        raise RuntimeError("Official CFLOW-AD tracked files are modified; use a clean checkout")
    return commit


def ensure_data_link(repo: Path, data_root: Path) -> None:
    """The upstream main.py reads ./data/MVTec-AD; do not edit its code."""
    link = repo / "data" / "MVTec-AD"
    if link.exists() or link.is_symlink():
        if link.resolve() != data_root:
            raise RuntimeError(f"Existing upstream data path points elsewhere: {link}")
        return
    link.parent.mkdir(exist_ok=True)
    link.symlink_to(data_root, target_is_directory=True)


def run_upstream(repo: Path, args: list[str]) -> None:
    command = [sys.executable, "main.py", *args]
    print("Running official CFLOW-AD:", " ".join(command), flush=True)
    subprocess.run(command, cwd=repo, check=True)


def newest_new_directory(parent: Path, prefix: str, old: set[Path]) -> Path:
    candidates = [p for p in parent.glob(prefix + "*") if p.is_dir() and p not in old]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one new {prefix} directory under {parent}; found {len(candidates)}. "
            "Inspect the upstream viz output before collecting anything."
        )
    return candidates[0]


def single_existing_directory(parent: Path, prefix: str) -> Path:
    candidates = [p for p in parent.glob(prefix + "*") if p.is_dir()]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one existing {prefix} directory under {parent}; "
            f"found {len(candidates)}. Specify a fresh upstream run if ambiguous."
        )
    return candidates[0]


def upstream_svg(directory: Path, index: int) -> Path:
    """The pinned upstream writer emits SVG bytes without a .svg suffix."""
    stem = f"{index:08d}"
    candidates = [p for p in (directory / stem, directory / (stem + ".svg")) if p.is_file()]
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one upstream SVG for sample {stem} in {directory}; "
            f"found {len(candidates)}"
        )
    source = candidates[0]
    with source.open("rb") as handle:
        if b"<svg" not in handle.read(8192).lower():
            raise ValueError(f"Upstream visualization is not SVG data: {source}")
    return source


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-repo", type=Path, help="Clean clone of gudovskiy/cflow-ad")
    parser.add_argument("--data-root", required=True, type=Path, help="MVTec AD root or Figure 6 staged root")
    parser.add_argument("--sample", required=True, type=Path, help="CLASS/test/DEFECT/FILE.png")
    parser.add_argument("--output-root", type=Path, default=Path("results_figure6/cflow_ad"))
    parser.add_argument("--checkpoint", type=Path, help="Previously trained CFLOW-AD checkpoint; skips training")
    parser.add_argument("--collect-existing-viz", action="store_true",
                        help="Collect one already completed upstream --viz run without rerunning inference; requires --checkpoint")
    parser.add_argument("--score-viz-dir", type=Path, help="Explicit sc_images directory for --collect-existing-viz")
    parser.add_argument("--gt-viz-dir", type=Path, help="Explicit gt_images directory for --collect-existing-viz")
    parser.add_argument("--meta-epochs", type=int, default=25)
    parser.add_argument("--sub-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--dry-run", action="store_true", help="Validate sample and print plan, without training")
    args = parser.parse_args()
    if args.collect_existing_viz and not args.checkpoint:
        parser.error("--collect-existing-viz requires --checkpoint")
    if bool(args.score_viz_dir) != bool(args.gt_viz_dir):
        parser.error("Specify both --score-viz-dir and --gt-viz-dir")
    if args.score_viz_dir and not args.collect_existing_viz:
        parser.error("Explicit viz directories require --collect-existing-viz")

    data_root = args.data_root.resolve(strict=True)
    class_name, image, mask, index = sample_info(data_root, args.sample)
    adapted = adapter_entry(data_root, args.sample, image, mask, index)
    input_size = INPUT_SIZES[class_name]
    model_name = f"mvtec_wide_resnet50_2_freia-cflow_pl3_cb8_inp{input_size}_run0_{class_name}"
    out_dir = args.output_root.resolve() / f"{class_name}_{image.parent.name}_{image.stem}"
    plan = {
        "official_repository": UPSTREAM_URL,
        "pinned_revision": UPSTREAM_COMMIT,
        "class": class_name,
        "source_dataset": adapted["source_dataset"] if adapted else "mvtec-ad",
        "source_class": adapted["source_class"] if adapted else class_name,
        "input_size": input_size,
        "sample_index_in_upstream_test_loader": index,
        "sample_index_stem": f"{index:08d}",
        "figure6_source_status": (
            "byte-identical input confirmed in supplied PPT"
            if args.sample in CONFIRMED_FIGURE6_SAMPLES
            else (
                "high-confidence pixel match after vertical flip and resize; not byte-identical"
                if args.sample in PIXEL_MATCHED_FIGURE6_SAMPLES or adapted
                else "MVTec AD sample only; PPT identity not confirmed"
            )
        ),
        "source_image": str(image),
        "ground_truth": str(mask) if mask else None,
        "image_sha256": sha256(image),
        "gt_sha256": sha256(mask) if mask else None,
        "model_name": model_name,
        "output_directory": str(out_dir),
        "checkpoint": str(args.checkpoint) if args.checkpoint else "train from official code",
        "training_configuration": (
            None if args.checkpoint else {
                "meta_epochs": args.meta_epochs,
                "sub_epochs": args.sub_epochs,
                "batch_size": args.batch_size,
                "workers": args.workers,
                "gpu": args.gpu,
                "encoder": "wide_resnet50_2",
                "decoder": "freia-cflow",
                "pool_layers": 3,
                "coupling_blocks": 8,
            }
        ),
    }
    if adapted:
        plan.update(adapted)
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return
    if os.name == "nt":
        raise RuntimeError("Run training/inference on Linux GPU server; Windows is supported for --dry-run only")
    if args.official_repo is None:
        parser.error("--official-repo is required unless --dry-run is used")
    if out_dir.exists():
        raise FileExistsError(f"Output directory already exists: {out_dir}")
    repo = args.official_repo.resolve(strict=True)
    ensure_upstream_repo(repo)
    ensure_data_link(repo, data_root)
    if package_version("Pillow") is not None:
        from PIL import Image
        if not hasattr(Image, "ANTIALIAS"):
            raise RuntimeError("Upstream loader uses Image.ANTIALIAS; install Pillow<10 in baseline environment")
    if args.meta_epochs < 1 or args.sub_epochs < 1 or args.batch_size < 1:
        parser.error("Epoch and batch-size values must be positive")

    common = [
        "--gpu", args.gpu, "--dataset", "mvtec", "--class-name", class_name,
        "-enc", "wide_resnet50_2", "-inp", str(input_size),
        "-bs", str(args.batch_size), "--workers", str(args.workers),
    ]
    if args.checkpoint:
        checkpoint = args.checkpoint.resolve(strict=True)
    else:
        weight_dir = repo / "weights"
        old_weights = set(weight_dir.glob(model_name + "_*.pt"))
        run_upstream(repo, [
            *common, "--action-type", "norm-train", "--meta-epochs", str(args.meta_epochs),
            "--sub-epochs", str(args.sub_epochs),
        ])
        new_weights = [p for p in weight_dir.glob(model_name + "_*.pt") if p not in old_weights]
        if len(new_weights) != 1:
            raise RuntimeError(f"Expected one new official checkpoint; found {len(new_weights)}")
        checkpoint = new_weights[0]

    viz_root = repo / "viz" / model_name
    if args.collect_existing_viz:
        if args.score_viz_dir:
            score_dir = args.score_viz_dir.resolve(strict=True)
            gt_dir = args.gt_viz_dir.resolve(strict=True)
            if (score_dir.parent != viz_root.resolve() or gt_dir.parent != viz_root.resolve() or
                    not score_dir.name.startswith("sc_images_") or not gt_dir.name.startswith("gt_images_")):
                raise ValueError("Explicit viz directories must belong to this upstream model")
        else:
            score_dir = single_existing_directory(viz_root, "sc_images_")
            gt_dir = single_existing_directory(viz_root, "gt_images_")
    else:
        old_scores = set(viz_root.glob("sc_images_*"))
        old_ground_truth = set(viz_root.glob("gt_images_*"))
        run_upstream(repo, [
            *common, "--action-type", "norm-test", "--checkpoint", str(checkpoint), "--viz",
        ])
        score_dir = newest_new_directory(viz_root, "sc_images_", old_scores)
        gt_dir = newest_new_directory(viz_root, "gt_images_", old_ground_truth)
    score_svg = upstream_svg(score_dir, index)
    gt_svg = upstream_svg(gt_dir, index)

    out_dir.mkdir(parents=True)
    outputs = {
        "source_image": Path(adapted["source_image_original"]) if adapted else image,
        "cflow_heat_overlay_and_boundary": score_svg,
        "cflow_ground_truth_boundary": gt_svg,
    }
    if mask is not None:
        outputs["ground_truth"] = mask
    for label, source in outputs.items():
        suffix = ".svg" if label.startswith("cflow_") else source.suffix
        destination = out_dir / (label + suffix)
        shutil.copy2(source, destination)
        plan[label + "_output"] = str(destination)
        plan[label + "_output_sha256"] = sha256(destination)
    plan.update({
        "official_revision_actual": git_head(repo),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_origin": "local official training" if not args.checkpoint else "supplied checkpoint",
        "upstream_visualization_origin": "existing completed upstream run" if args.collect_existing_viz else "new upstream run",
        "upstream_score_svg": str(score_svg),
        "upstream_gt_svg": str(gt_svg),
        "sample_index_filename": score_svg.name,
        "python_version": sys.version,
        "torch_version": package_version("torch"),
        "torchvision_version": package_version("torchvision"),
        "pillow_version": package_version("Pillow"),
        "note": "The upstream sc_images SVG has a heatmap overlay on top and a thresholded boundary image below.",
    })
    (out_dir / "provenance.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Collected official CFLOW-AD visualization: {out_dir}")


if __name__ == "__main__":
    main()
