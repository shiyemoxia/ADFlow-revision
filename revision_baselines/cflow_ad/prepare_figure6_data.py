"""Stage the original Figure 6 datasets for the unmodified CFLOW-AD loader.

Upstream CFLOW-AD only accepts its MVTec class names and lowercase ``.png``
filenames. This adapter changes neither pixels nor the upstream model: it
creates file links (or copies), uses documented class aliases for VisA/LOCO,
and unions LOCO's multiple ground-truth masks when needed.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil

from PIL import Image, ImageChops


MANIFEST_NAME = "cflow_figure6_adapter.json"


@dataclass(frozen=True)
class ClassSpec:
    dataset: str
    source_class: str
    alias: str


# All aliases are in the pinned upstream MVTec class whitelist. Avoid its
# special grayscale aliases (grid, screw, zipper) for RGB VisA/LOCO images.
CLASS_SPECS = (
    ClassSpec("mvtec-ad", "cable", "cable"),
    ClassSpec("mvtec-ad", "transistor", "transistor"),
    ClassSpec("mvtec-ad", "hazelnut", "hazelnut"),
    ClassSpec("mvtec-ad", "metal_nut", "metal_nut"),
    ClassSpec("visa", "pcb2", "bottle"),
    ClassSpec("visa", "cashew", "carpet"),
    ClassSpec("visa", "fryum", "leather"),
    ClassSpec("visa", "pcb4", "tile"),
    ClassSpec("mvtec-loco", "pushpins", "pill"),
    ClassSpec("mvtec-loco", "screw_bag", "toothbrush"),
)


# The 19 SVG-embedded input pictures identified in the supplied PPT. The
# source audit explains the pixel-level matching and PPT-label discrepancies.
FIGURE6_INPUTS = {
    "rId4_01": "visa/pcb2/Data/Images/Normal/0690.JPG",
    "rId5_01": "mvtec-ad/metal_nut/test/good/017.png",
    "rId6_01": "mvtec-ad/metal_nut/test/color/000.png",
    "rId7_01": "visa/pcb2/Data/Images/Anomaly/013.JPG",
    "rId8_01": "mvtec-ad/cable/test/good/046.png",
    "rId10_01": "mvtec-ad/transistor/test/good/050.png",
    "rId13_01": "mvtec-loco/pushpins/test/logical_anomalies/071.png",
    "rId14_01": "mvtec-loco/pushpins/test/good/131.png",
    "rId15_01": "mvtec-loco/screw_bag/test/good/046.png",
    "rId16_01": "mvtec-ad/hazelnut/test/crack/000.png",
    "rId17_01": "mvtec-ad/metal_nut/test/bent/005.png",
    "rId18_01": "mvtec-ad/transistor/test/damaged_case/009.png",
    "rId19_01": "mvtec-ad/cable/test/missing_cable/000.png",
    "rId20_01": "visa/cashew/Data/Images/Anomaly/019.JPG",
    "rId21_01": "visa/fryum/Data/Images/Anomaly/074.JPG",
    "rId22_01": "visa/pcb4/Data/Images/Anomaly/011.JPG",
    "rId28_01": "mvtec-loco/screw_bag/test/logical_anomalies/105.png",
    "rId33_01": "mvtec-ad/transistor/test/cut_lead/005.png",
    "rId36_01": "mvtec-ad/cable/test/bent_wire/001.png",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_within(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve(strict=True)
    resolved_root = root.resolve(strict=True)
    if resolved_root not in candidate.parents or not candidate.is_file():
        raise ValueError(f"Dataset file is missing or outside its root: {relative}")
    return candidate


def place_file(source: Path, target: Path, method: str) -> None:
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Staging collision: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if method == "hardlink":
        try:
            os.link(source, target)
        except OSError as error:
            raise RuntimeError(
                f"Hardlink failed for {source}; place the stage on the same filesystem "
                "or retry with --link-mode symlink/copy"
            ) from error
    elif method == "symlink":
        target.symlink_to(source)
    elif method == "copy":
        shutil.copy2(source, target)
    else:
        raise ValueError(f"Unsupported link mode: {method}")


def union_masks(sources: list[Path], target: Path) -> None:
    if not sources:
        raise ValueError(f"No LOCO masks for {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    merged = None
    size = None
    for source in sources:
        with Image.open(source) as opened:
            mask = opened.convert("L")
            if size is None:
                size = mask.size
            elif mask.size != size:
                raise ValueError(f"Mask dimensions differ for {target}: {source}")
            mask = mask.point(lambda value: 255 if value else 0)
            merged = mask if merged is None else ImageChops.lighter(merged, mask)
    assert merged is not None
    merged.save(target, format="PNG")


def stage_mvtec(spec: ClassSpec, datasets_root: Path, stage_root: Path, method: str) -> dict:
    source_root = datasets_root / spec.dataset / spec.source_class
    counts = {"train_good": 0, "test_good": 0, "test_anomaly": 0}
    for phase in ("train", "test"):
        phase_dir = source_root / phase
        if not phase_dir.is_dir():
            raise FileNotFoundError(phase_dir)
        for image in sorted(phase_dir.glob("*/*.png")):
            defect = image.parent.name
            if phase == "train" and defect != "good":
                raise ValueError(f"Unexpected anomalous training image: {image}")
            place_file(image, stage_root / spec.alias / phase / defect / image.name, method)
            bucket = "train_good" if phase == "train" else ("test_good" if defect == "good" else "test_anomaly")
            counts[bucket] += 1
            if phase == "test" and defect != "good":
                mask = source_root / "ground_truth" / defect / f"{image.stem}_mask.png"
                if not mask.is_file():
                    raise FileNotFoundError(mask)
                place_file(mask, stage_root / spec.alias / "ground_truth" / defect / mask.name, method)
    return counts


def stage_visa(spec: ClassSpec, datasets_root: Path, stage_root: Path, method: str) -> dict:
    visa_root = datasets_root / "visa"
    csv_path = visa_root / "split_csv" / "1cls.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    counts = {"train_good": 0, "test_good": 0, "test_anomaly": 0}
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = [row for row in csv.DictReader(stream) if row["object"] == spec.source_class]
    if not rows:
        raise ValueError(f"VisA CSV has no rows for {spec.source_class}")
    for row in rows:
        split, label = row["split"], row["label"]
        if (split, label) not in {("train", "normal"), ("test", "normal"), ("test", "anomaly")}:
            raise ValueError(f"Unexpected VisA split/label: {row}")
        image = require_within(visa_root, row["image"])
        if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            raise ValueError(f"Unexpected VisA image format: {image}")
        defect = "anomaly" if label == "anomaly" else "good"
        # The pinned upstream loader filters lowercase .png names, but PIL
        # decodes by file signature. A link preserves the original JPEG bytes.
        target = stage_root / spec.alias / split / defect / f"{image.stem}.png"
        place_file(image, target, method)
        bucket = "train_good" if split == "train" else ("test_good" if defect == "good" else "test_anomaly")
        counts[bucket] += 1
        if defect == "anomaly":
            mask = require_within(visa_root, row["mask"])
            place_file(mask, stage_root / spec.alias / "ground_truth" / defect / f"{image.stem}_mask.png", method)
    counts["source_split_csv_sha256"] = sha256(csv_path)
    return counts


def stage_loco(spec: ClassSpec, datasets_root: Path, stage_root: Path, method: str) -> dict:
    source_root = datasets_root / "mvtec-loco" / spec.source_class
    counts = {"train_good": 0, "test_good": 0, "test_anomaly": 0, "multi_mask_samples": 0}
    for phase in ("train", "test"):
        phase_dir = source_root / phase
        if not phase_dir.is_dir():
            raise FileNotFoundError(phase_dir)
        for image in sorted(phase_dir.glob("*/*.png")):
            defect = image.parent.name
            if phase == "train" and defect != "good":
                raise ValueError(f"Unexpected anomalous LOCO training image: {image}")
            place_file(image, stage_root / spec.alias / phase / defect / image.name, method)
            bucket = "train_good" if phase == "train" else ("test_good" if defect == "good" else "test_anomaly")
            counts[bucket] += 1
            if phase == "test" and defect != "good":
                source_masks = sorted((source_root / "ground_truth" / defect / image.stem).glob("*.png"))
                if not source_masks:
                    raise FileNotFoundError(f"No masks for {image}")
                if len(source_masks) > 1:
                    counts["multi_mask_samples"] += 1
                target = stage_root / spec.alias / "ground_truth" / defect / f"{image.stem}_mask.png"
                union_masks(source_masks, target)
    return counts


def staged_selected_image(spec: ClassSpec, source_relative: Path) -> Path:
    parts = source_relative.parts
    if spec.dataset == "visa":
        defect = "good" if parts[4] == "Normal" else "anomaly"
        return Path(spec.alias, "test", defect, source_relative.stem + ".png")
    return Path(spec.alias, *parts[2:])


def upstream_test_images(stage_root: Path, alias: str) -> list[Path]:
    images = []
    for defect_dir in sorted((stage_root / alias / "test").iterdir()):
        if defect_dir.is_dir():
            images.extend(sorted(p for p in defect_dir.iterdir() if p.name.endswith(".png")))
    return images


def selected_entries(datasets_root: Path, stage_root: Path, specs: tuple[ClassSpec, ...]) -> list[dict]:
    by_source = {(spec.dataset, spec.source_class): spec for spec in specs}
    entries = []
    for ppt_id, source_name in FIGURE6_INPUTS.items():
        source_relative = Path(source_name)
        spec = by_source.get((source_relative.parts[0], source_relative.parts[1]))
        if spec is None:
            continue
        original_image = require_within(datasets_root, source_name)
        relative_stage = staged_selected_image(spec, source_relative)
        staged_image = stage_root / relative_stage
        if not staged_image.is_file() or sha256(staged_image) != sha256(original_image):
            raise ValueError(f"Selected input changed during staging: {source_name}")
        try:
            upstream_index = upstream_test_images(stage_root, spec.alias).index(staged_image)
        except ValueError as error:
            raise ValueError(f"CFLOW test loader would skip {staged_image}") from error
        mask_parts = []
        staged_mask = None
        if relative_stage.parts[2] != "good":
            staged_mask = Path(spec.alias, "ground_truth", relative_stage.parts[2], relative_stage.stem + "_mask.png")
            if not (stage_root / staged_mask).is_file():
                raise FileNotFoundError(stage_root / staged_mask)
            if spec.dataset == "mvtec-loco":
                mask_parts = [str(p.relative_to(datasets_root).as_posix()) for p in sorted(
                    (datasets_root / spec.dataset / spec.source_class / "ground_truth" / relative_stage.parts[2] / relative_stage.stem).glob("*.png")
                )]
            elif spec.dataset == "mvtec-ad":
                mask_parts = [f"{spec.dataset}/{spec.source_class}/ground_truth/{relative_stage.parts[2]}/{relative_stage.stem}_mask.png"]
            else:
                mask_parts = [f"visa/{spec.source_class}/Data/Masks/Anomaly/{source_relative.stem}.png"]
        entries.append({
            "ppt_id": ppt_id,
            "dataset": spec.dataset,
            "source_class": spec.source_class,
            "upstream_alias": spec.alias,
            "source_image": source_relative.as_posix(),
            "staged_image": relative_stage.as_posix(),
            "source_image_sha256": sha256(original_image),
            "staged_mask": staged_mask.as_posix() if staged_mask else None,
            "source_masks": mask_parts,
            "source_mask_sha256": [sha256(datasets_root / p) for p in mask_parts],
            "staged_mask_sha256": sha256(stage_root / staged_mask) if staged_mask else None,
            "upstream_test_index": upstream_index,
            "upstream_viz_stem": f"{upstream_index:08d}",
        })
    return entries


def prepare(datasets_root: Path, stage_root: Path, method: str = "hardlink", specs: tuple[ClassSpec, ...] = CLASS_SPECS) -> dict:
    datasets_root = datasets_root.resolve(strict=True)
    stage_root = stage_root.resolve()
    if stage_root.exists():
        raise FileExistsError(f"Staging output exists; will not overwrite: {stage_root}")
    if stage_root == datasets_root or stage_root in datasets_root.parents:
        raise ValueError("Staging root cannot contain the source datasets")
    aliases = [spec.alias for spec in specs]
    if len(aliases) != len(set(aliases)):
        raise ValueError("CFLOW aliases must be unique")
    stage_root.mkdir(parents=True)
    classes = []
    for spec in specs:
        if spec.dataset == "mvtec-ad":
            counts = stage_mvtec(spec, datasets_root, stage_root, method)
        elif spec.dataset == "visa":
            counts = stage_visa(spec, datasets_root, stage_root, method)
        elif spec.dataset == "mvtec-loco":
            counts = stage_loco(spec, datasets_root, stage_root, method)
        else:
            raise ValueError(f"Unexpected dataset: {spec.dataset}")
        classes.append({"dataset": spec.dataset, "source_class": spec.source_class, "upstream_alias": spec.alias, **counts})
        print(f"Staged {spec.dataset}/{spec.source_class} as {spec.alias}: {counts}", flush=True)
    manifest = {
        "schema_version": 1,
        "adapter": "MVTec-folder staging only; upstream CFLOW-AD model and loader unchanged",
        "link_mode": method,
        "visa_image_note": "VisA JPEG bytes are linked under .png filenames; PIL detects JPEG content by signature",
        "loco_mask_note": "All masks for an anomalous LOCO image are binarized and unioned into one PNG",
        "datasets_root": str(datasets_root),
        "stage_root": str(stage_root),
        "classes": classes,
        "figure6_inputs": selected_entries(datasets_root, stage_root, specs),
    }
    (stage_root / MANIFEST_NAME).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify_stage(stage_root: Path, deep: bool = False) -> dict:
    """Check every staged loader path, selected provenance, and optional pixels."""
    stage_root = stage_root.resolve(strict=True)
    manifest_path = stage_root / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unknown adapter manifest version: {manifest_path}")
    actual_total = 0
    for item in manifest["classes"]:
        alias = item["upstream_alias"]
        root = stage_root / alias
        for bucket, phase, defect in (
            ("train_good", "train", "good"),
            ("test_good", "test", "good"),
        ):
            paths = list((root / phase / defect).glob("*.png"))
            if len(paths) != item[bucket]:
                raise ValueError(f"{alias}/{phase}/{defect}: {len(paths)} != {item[bucket]}")
            actual_total += len(paths)
            if deep:
                for path in paths:
                    with Image.open(path) as opened:
                        if opened.mode != "RGB":
                            raise ValueError(f"CFLOW expects RGB input: {path} ({opened.mode})")
                        opened.verify()
        anomaly_images = [p for defect_dir in (root / "test").iterdir()
                          if defect_dir.is_dir() and defect_dir.name != "good"
                          for p in defect_dir.glob("*.png")]
        if len(anomaly_images) != item["test_anomaly"]:
            raise ValueError(f"{alias}: {len(anomaly_images)} anomalies != {item['test_anomaly']}")
        actual_total += len(anomaly_images)
        for path in anomaly_images:
            mask = root / "ground_truth" / path.parent.name / f"{path.stem}_mask.png"
            if not mask.is_file():
                raise FileNotFoundError(mask)
            if deep:
                with Image.open(path) as image, Image.open(mask) as opened_mask:
                    if image.mode != "RGB" or opened_mask.mode != "L" or image.size != opened_mask.size:
                        raise ValueError(f"Image/mask shape or mode mismatch: {path}, {mask}")
                    image.verify()
                    opened_mask.verify()
    if __package__:
        from .run_official_cflow import adapter_entry, sample_info
    else:
        from run_official_cflow import adapter_entry, sample_info
    for entry in manifest["figure6_inputs"]:
        source = Path(manifest["datasets_root"]) / entry["source_image"]
        staged = stage_root / entry["staged_image"]
        if sha256(source) != entry["source_image_sha256"] or sha256(staged) != entry["source_image_sha256"]:
            raise ValueError(f"Figure 6 source/staged input mismatch: {entry['ppt_id']}")
        mask = entry["staged_mask"]
        if mask and sha256(stage_root / mask) != entry["staged_mask_sha256"]:
            raise ValueError(f"Figure 6 GT mismatch: {entry['ppt_id']}")
        if len(entry["source_masks"]) != len(entry["source_mask_sha256"]):
            raise ValueError(f"Figure 6 GT provenance length mismatch: {entry['ppt_id']}")
        for relative, expected_hash in zip(entry["source_masks"], entry["source_mask_sha256"]):
            if sha256(Path(manifest["datasets_root"]) / relative) != expected_hash:
                raise ValueError(f"Figure 6 original GT changed: {entry['ppt_id']} {relative}")
        current_index = upstream_test_images(stage_root, entry["upstream_alias"]).index(staged)
        if current_index != entry["upstream_test_index"]:
            raise ValueError(f"CFLOW sample order changed: {entry['ppt_id']}")
        relative = Path(entry["staged_image"])
        _, image, loaded_mask, loaded_index = sample_info(stage_root, relative)
        provenance = adapter_entry(stage_root, relative, image, loaded_mask, loaded_index)
        if provenance is None or entry["ppt_id"] not in provenance["ppt_input_ids"]:
            raise ValueError(f"CFLOW wrapper provenance failed: {entry['ppt_id']}")
    return {"classes": len(manifest["classes"]), "staged_images": actual_total,
            "figure6_inputs": len(manifest["figure6_inputs"]), "deep": deep}


def refresh_manifest(stage_root: Path) -> None:
    """Recalculate selected-sample hashes/indices without rewriting staged data."""
    stage_root = stage_root.resolve(strict=True)
    path = stage_root / MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    datasets_root = Path(manifest["datasets_root"]).resolve(strict=True)
    manifest["figure6_inputs"] = selected_entries(datasets_root, stage_root, CLASS_SPECS)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", type=Path, default=Path("datasets"))
    parser.add_argument("--output-root", type=Path, default=Path("datasets/cflow_figure6_stage"))
    parser.add_argument("--link-mode", choices=("hardlink", "symlink", "copy"), default="hardlink")
    parser.add_argument("--verify-only", action="store_true", help="Check an existing stage; do not create files")
    parser.add_argument("--refresh-manifest", action="store_true", help="Recalculate an existing manifest only")
    parser.add_argument("--deep", action="store_true", help="Decode every staged image and compare image/mask sizes")
    args = parser.parse_args()
    if args.refresh_manifest:
        refresh_manifest(args.output_root)
    elif not args.verify_only:
        prepare(args.datasets_root, args.output_root, args.link_mode)
    result = verify_stage(args.output_root, deep=args.deep)
    print(f"Validated stage: {result}; manifest: {args.output_root / MANIFEST_NAME}")


if __name__ == "__main__":
    main()
