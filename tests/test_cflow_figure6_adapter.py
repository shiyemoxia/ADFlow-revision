import csv
from pathlib import Path

from PIL import Image
import pytest

from revision_baselines.cflow_ad.prepare_figure6_data import (
    ClassSpec, sha256, stage_loco, stage_visa,
)
from revision_baselines.cflow_ad.run_official_cflow import adapter_entry, sample_info
from revision_baselines.cflow_ad.run_figure6_batch import grouped_samples, output_dir


def save_rgb(path: Path, color=(30, 80, 120), format="PNG") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path, format=format)


def save_mask(path: Path, x: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", (8, 8), 0)
    image.putpixel((x, 3), 255)
    image.save(path)


def test_visa_jpeg_alias_preserves_bytes_and_split(tmp_path):
    datasets_root = tmp_path / "datasets"
    visa = datasets_root / "visa"
    rows = []
    for split, label, stem in (("train", "normal", "0001"),
                               ("test", "normal", "0002"),
                               ("test", "anomaly", "0003")):
        image = f"cashew/Data/Images/{'Anomaly' if label == 'anomaly' else 'Normal'}/{stem}.JPG"
        save_rgb(visa / image, format="JPEG")
        mask = f"cashew/Data/Masks/Anomaly/{stem}.png" if label == "anomaly" else ""
        if mask:
            save_mask(visa / mask, 2)
        rows.append({"object": "cashew", "split": split, "label": label, "image": image, "mask": mask})
    csv_path = visa / "split_csv" / "1cls.csv"
    csv_path.parent.mkdir(parents=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    stage = tmp_path / "stage"
    counts = stage_visa(ClassSpec("visa", "cashew", "carpet"), datasets_root, stage, "hardlink")
    assert [counts[key] for key in ("train_good", "test_good", "test_anomaly")] == [1, 1, 1]
    aliased = stage / "carpet/test/anomaly/0003.png"
    assert sha256(aliased) == sha256(visa / rows[2]["image"])
    with Image.open(aliased) as opened:
        assert opened.format == "JPEG" and opened.mode == "RGB"
    assert (stage / "carpet/ground_truth/anomaly/0003_mask.png").is_file()


def test_loco_multiple_masks_are_unioned(tmp_path):
    datasets_root = tmp_path / "datasets"
    source = datasets_root / "mvtec-loco/pushpins"
    save_rgb(source / "train/good/001.png")
    save_rgb(source / "test/good/002.png")
    save_rgb(source / "test/logical_anomalies/071.png")
    save_mask(source / "ground_truth/logical_anomalies/071/000.png", 1)
    save_mask(source / "ground_truth/logical_anomalies/071/001.png", 5)
    stage = tmp_path / "stage"
    counts = stage_loco(ClassSpec("mvtec-loco", "pushpins", "pill"), datasets_root, stage, "hardlink")
    assert counts == {"train_good": 1, "test_good": 1, "test_anomaly": 1, "multi_mask_samples": 1}
    with Image.open(stage / "pill/ground_truth/logical_anomalies/071_mask.png") as mask:
        assert mask.getpixel((1, 3)) == 255
        assert mask.getpixel((5, 3)) == 255
        assert mask.getpixel((3, 3)) == 0


def test_official_order_includes_normal_samples_and_alias_needs_manifest(tmp_path):
    save_rgb(tmp_path / "bottle/test/anomaly/001.png")
    save_mask(tmp_path / "bottle/ground_truth/anomaly/001_mask.png", 2)
    save_rgb(tmp_path / "bottle/test/good/002.png")
    alias, _, mask, index = sample_info(tmp_path, Path("bottle/test/good/002.png"))
    assert (alias, mask, index) == ("bottle", None, 1)
    with pytest.raises(FileNotFoundError, match="adapter manifest"):
        adapter_entry(tmp_path, Path("bottle/test/good/002.png"), tmp_path / "bottle/test/good/002.png", None)


def test_batch_groups_same_class_for_one_training(tmp_path):
    manifest = {"figure6_inputs": [
        {"upstream_alias": "bottle", "staged_image": "bottle/test/good/002.png"},
        {"upstream_alias": "bottle", "staged_image": "bottle/test/anomaly/003.png"},
        {"upstream_alias": "bottle", "staged_image": "bottle/test/anomaly/003.png"},
        {"upstream_alias": "pill", "staged_image": "pill/test/good/004.png"},
    ]}
    assert grouped_samples(manifest) == {
        "bottle": ["bottle/test/good/002.png", "bottle/test/anomaly/003.png"],
        "pill": ["pill/test/good/004.png"],
    }
    assert output_dir(tmp_path, "bottle/test/anomaly/003.png") == tmp_path / "bottle_anomaly_003"
