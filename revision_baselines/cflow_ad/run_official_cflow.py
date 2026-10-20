"""Run the pinned, unmodified CFLOW-AD implementation for one MVTec AD sample.

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
INPUT_SIZES = {"cable": 256, "transistor": 128, "hazelnut": 256, "metal_nut": 256}
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


def sample_info(data_root: Path, relative_sample: Path) -> tuple[str, Path, Path, int]:
    if relative_sample.is_absolute() or ".." in relative_sample.parts:
        raise ValueError("--sample must be a relative path within --data-root")
    parts = relative_sample.parts
    if len(parts) != 4 or parts[1] != "test" or parts[2] == "good":
        raise ValueError("Expected --sample CLASS/test/DEFECT/FILE.png (an anomalous sample)")
    class_name, _, defect_name, file_name = parts
    if class_name not in INPUT_SIZES:
        raise ValueError(f"Supported MVTec AD classes: {sorted(INPUT_SIZES)}")
    if not file_name.endswith(".png"):
        raise ValueError("The upstream MVTec AD loader selects lowercase .png files")
    image = data_root / relative_sample
    mask = data_root / class_name / "ground_truth" / defect_name / f"{image.stem}_mask.png"
    if not image.is_file() or not mask.is_file():
        raise FileNotFoundError(f"Input/GT pair incomplete: {image}; {mask}")
    images = test_images(data_root, class_name)
    try:
        index = images.index(image)
    except ValueError as error:
        raise ValueError(f"Input is not selected by upstream test loader: {image}") from error
    return class_name, image, mask, index


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


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-repo", type=Path, help="Clean clone of gudovskiy/cflow-ad")
    parser.add_argument("--data-root", required=True, type=Path, help="MVTec AD dataset root")
    parser.add_argument("--sample", required=True, type=Path, help="CLASS/test/DEFECT/FILE.png")
    parser.add_argument("--output-root", type=Path, default=Path("results_figure6/cflow_ad"))
    parser.add_argument("--checkpoint", type=Path, help="Previously trained CFLOW-AD checkpoint; skips training")
    parser.add_argument("--meta-epochs", type=int, default=25)
    parser.add_argument("--sub-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--dry-run", action="store_true", help="Validate sample and print plan, without training")
    args = parser.parse_args()

    data_root = args.data_root.resolve(strict=True)
    class_name, image, mask, index = sample_info(data_root, args.sample)
    input_size = INPUT_SIZES[class_name]
    model_name = f"mvtec_wide_resnet50_2_freia-cflow_pl3_cb8_inp{input_size}_run0_{class_name}"
    out_dir = args.output_root.resolve() / f"{class_name}_{image.parent.name}_{image.stem}"
    plan = {
        "official_repository": UPSTREAM_URL,
        "pinned_revision": UPSTREAM_COMMIT,
        "class": class_name,
        "input_size": input_size,
        "sample_index_in_upstream_test_loader": index,
        "sample_index_filename": f"{index:08d}.svg",
        "figure6_source_status": (
            "byte-identical input confirmed in supplied PPT"
            if args.sample in CONFIRMED_FIGURE6_SAMPLES
            else (
                "high-confidence pixel match after vertical flip and resize; not byte-identical"
                if args.sample in PIXEL_MATCHED_FIGURE6_SAMPLES
                else "MVTec AD sample only; PPT identity not confirmed"
            )
        ),
        "source_image": str(image),
        "ground_truth": str(mask),
        "image_sha256": sha256(image),
        "gt_sha256": sha256(mask),
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
    old_scores = set(viz_root.glob("sc_images_*"))
    old_ground_truth = set(viz_root.glob("gt_images_*"))
    run_upstream(repo, [
        *common, "--action-type", "norm-test", "--checkpoint", str(checkpoint), "--viz",
    ])
    score_dir = newest_new_directory(viz_root, "sc_images_", old_scores)
    gt_dir = newest_new_directory(viz_root, "gt_images_", old_ground_truth)
    score_svg = score_dir / f"{index:08d}.svg"
    gt_svg = gt_dir / f"{index:08d}.svg"
    if not score_svg.is_file() or not gt_svg.is_file():
        raise FileNotFoundError(f"Expected upstream visualization missing: {score_svg}; {gt_svg}")

    out_dir.mkdir(parents=True)
    outputs = {
        "source_image": image,
        "ground_truth": mask,
        "cflow_heat_overlay_and_boundary": score_svg,
        "cflow_ground_truth_boundary": gt_svg,
    }
    for label, source in outputs.items():
        destination = out_dir / (label + source.suffix)
        shutil.copy2(source, destination)
        plan[label + "_output"] = str(destination)
        plan[label + "_output_sha256"] = sha256(destination)
    plan.update({
        "official_revision_actual": git_head(repo),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_origin": "local official training" if not args.checkpoint else "supplied checkpoint",
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
