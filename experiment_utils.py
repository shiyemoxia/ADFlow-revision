"""Small, dependency-light helpers for the revision experiments."""

import csv
import json
from pathlib import Path

import numpy as np


CURVE_FIELDS = (
    "epoch",
    "train_loss",
    "train_nll",
    "train_bce",
    "test_loss",
    "det_auroc",
    "seg_auroc",
    "det_ap",
    "seg_ap",
)


class CurveRecorder:
    def __init__(self, path, metadata):
        self.path = Path(path)
        self.metadata_path = self.path.with_suffix(".json")
        self.rows = []
        self.metadata = dict(metadata)

    def append(self, **values):
        row = {field: values.get(field, np.nan) for field in CURVE_FIELDS}
        self.rows.append(row)
        self.save()

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CURVE_FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        with self.metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(self.metadata, handle, ensure_ascii=False, indent=2)


def save_latent_sample(path, z, labels, log_prob, metadata, max_samples, seed):
    """Save a deterministic random subset of latent vectors."""
    z = np.asarray(z, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    log_prob = np.asarray(log_prob, dtype=np.float32).reshape(-1)
    if not (len(z) == len(labels) == len(log_prob)):
        raise ValueError("z, labels and log_prob must contain the same number of rows")

    if len(z) > max_samples:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(len(z), size=max_samples, replace=False))
        z = z[indices]
        labels = labels[indices]
        log_prob = log_prob[indices]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        z=z,
        labels=labels,
        log_prob=log_prob,
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    return path
