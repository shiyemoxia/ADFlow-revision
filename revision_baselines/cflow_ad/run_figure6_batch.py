"""Run one official CFLOW-AD training/inference per Figure 6 class.

The adapter manifest supplies the exact 19 audited inputs. Additional inputs
from a class reuse the same checkpoint and the same completed upstream viz.
No baseline model or upstream loader file is modified.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

if __package__:
    from .prepare_figure6_data import MANIFEST_NAME, verify_stage
    from .run_official_cflow import UPSTREAM_COMMIT, ensure_upstream_repo
else:
    from prepare_figure6_data import MANIFEST_NAME, verify_stage
    from run_official_cflow import UPSTREAM_COMMIT, ensure_upstream_repo


WRAPPER = Path(__file__).with_name("run_official_cflow.py")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def grouped_samples(manifest: dict) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for entry in manifest["figure6_inputs"]:
        sample = entry["staged_image"]
        alias = entry["upstream_alias"]
        if sample not in groups[alias]:
            groups[alias].append(sample)
    return dict(groups)


def output_dir(root: Path, sample: str) -> Path:
    path = Path(sample)
    return root / f"{path.parts[0]}_{path.parts[2]}_{path.stem}"


def prior_provenance(root: Path, data_root: Path, sample: str, revision: str) -> dict | None:
    directory = output_dir(root, sample)
    if not directory.exists():
        return None
    path = directory / "provenance.json"
    if not path.is_file():
        raise RuntimeError(f"Existing output has no provenance; inspect manually: {directory}")
    plan = json.loads(path.read_text(encoding="utf-8"))
    if (plan.get("pinned_revision") != revision or
            plan.get("class") != Path(sample).parts[0] or
            Path(plan.get("source_image", "")).resolve() != (data_root / sample).resolve() or
            plan.get("image_sha256") != sha256(data_root / sample)):
        raise RuntimeError(f"Existing output does not match requested sample: {directory}")
    checkpoint = Path(plan.get("checkpoint_path", ""))
    if not checkpoint.is_file() or plan.get("checkpoint_sha256") != sha256(checkpoint):
        raise RuntimeError(f"Existing checkpoint changed or is missing: {checkpoint}")
    return plan


def run_wrapper(args: argparse.Namespace, sample: str, checkpoint: Path | None,
                collect_existing: bool, viz_source: dict | None = None) -> dict:
    command = [sys.executable, str(WRAPPER),
               "--official-repo", str(args.official_repo),
               "--data-root", str(args.data_root),
               "--sample", sample,
               "--output-root", str(args.output_root),
               "--meta-epochs", str(args.meta_epochs),
               "--sub-epochs", str(args.sub_epochs),
               "--batch-size", str(args.batch_size),
               "--workers", str(args.workers),
               "--gpu", args.gpu]
    if checkpoint is not None:
        command.extend(("--checkpoint", str(checkpoint)))
    if collect_existing:
        command.append("--collect-existing-viz")
        if viz_source is not None:
            command.extend(("--score-viz-dir", str(Path(viz_source["upstream_score_svg"]).parent),
                            "--gt-viz-dir", str(Path(viz_source["upstream_gt_svg"]).parent)))
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    provenance = prior_provenance(args.output_root, args.data_root, sample, UPSTREAM_COMMIT)
    if provenance is None:
        raise RuntimeError(f"Wrapper returned but result is missing: {sample}")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Output of prepare_figure6_data.py")
    parser.add_argument("--official-repo", type=Path, help="Clean checkout of pinned official CFLOW-AD")
    parser.add_argument("--output-root", type=Path, default=Path("results_figure6/cflow_ad"))
    parser.add_argument("--meta-epochs", type=int, default=25)
    parser.add_argument("--sub-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--dry-run", action="store_true", help="Print training/collection plan, no GPU work")
    args = parser.parse_args()
    args.data_root = args.data_root.resolve(strict=True)
    manifest = json.loads((args.data_root / MANIFEST_NAME).read_text(encoding="utf-8"))
    groups = grouped_samples(manifest)
    if len(groups) != 10 or sum(map(len, groups.values())) != 19:
        raise ValueError(f"Expected 10 classes and 19 audited inputs; got {len(groups)} classes, {sum(map(len, groups.values()))} inputs")
    print(json.dumps({"classes": groups, "meta_epochs": args.meta_epochs,
                      "sub_epochs": args.sub_epochs, "batch_size": args.batch_size},
                     indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        verify_stage(args.data_root)
        return
    if os.name == "nt":
        raise RuntimeError("Train on Linux GPU server; Windows supports --dry-run only")
    if args.official_repo is None:
        parser.error("--official-repo is required for training")
    args.official_repo = args.official_repo.resolve(strict=True)
    args.output_root = args.output_root.resolve()
    ensure_upstream_repo(args.official_repo)
    verify_stage(args.data_root)
    for alias, samples in groups.items():
        existing = {sample: prior_provenance(args.output_root, args.data_root, sample, UPSTREAM_COMMIT)
                    for sample in samples}
        checkpoint = None
        viz_source = None
        for plan in existing.values():
            if plan is not None:
                candidate = Path(plan["checkpoint_path"])
                if checkpoint is not None and checkpoint != candidate:
                    raise RuntimeError(f"Different checkpoints found for {alias}; inspect outputs")
                checkpoint = candidate
                if (viz_source is None and Path(plan["upstream_score_svg"]).is_file() and
                        Path(plan["upstream_gt_svg"]).is_file()):
                    viz_source = plan
        pending = [sample for sample in samples if existing[sample] is None]
        if not pending:
            print(f"Completed already: {alias}", flush=True)
            continue
        print(f"Class {alias}: {len(pending)} input(s) pending", flush=True)
        if checkpoint is None:
            first = pending.pop(0)
            plan = run_wrapper(args, first, None, False)
            checkpoint = Path(plan["checkpoint_path"])
            viz_source = plan
        if pending and viz_source is None:
            first = pending.pop(0)
            viz_source = run_wrapper(args, first, checkpoint, False)
        for sample in pending:
            run_wrapper(args, sample, checkpoint, True, viz_source)
    print("All 19 audited Figure 6 CFLOW-AD samples collected.", flush=True)


if __name__ == "__main__":
    main()
