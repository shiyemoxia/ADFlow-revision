"""Run the MACA multi-scale ablation under matched training settings.

The middle-scale arm keeps the same MACA block only at the middle backbone
feature scale. It is not an unchanged upstream ACmix implementation or a
parameter-matched ACmix comparison.
"""

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path


FULL_CLASSES = {
    'mvtec': ('bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut',
              'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush',
              'transistor', 'wood', 'zipper'),
    'visa': ('candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1',
             'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'),
}
PILOT_CLASSES = {'mvtec': ('carpet', 'screw'), 'visa': ('capsules', 'pcb1')}


def _last_result(output_root, dataset, class_name, run_name, *, scales,
                 seed, meta_epochs, data_root):
    path = (output_root / 'curves' / dataset / class_name / 'adaptive' /
            'run_{}.csv'.format(run_name))
    metadata_path = path.with_suffix('.json')
    if not path.exists() and not metadata_path.exists():
        return None
    if not path.is_file() or not metadata_path.is_file():
        raise RuntimeError('Incomplete run files at {}. Preserve them and use a new '
                           '--output-dir or --run-name to rerun this arm.'.format(path))
    with path.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    expected = {
        'dataset': dataset, 'class_name': class_name,
        'distribution': 'adaptive', 'maca_scales': scales,
        'run_name': run_name, 'seed': seed, 'meta_epochs': meta_epochs,
    }
    mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
    if Path(metadata.get('data_path', '')).resolve() != data_root.resolve():
        mismatches.append('data_path')
    if mismatches:
        raise RuntimeError('Existing run at {} has different settings ({}). '
                           'Use a new --output-dir or --run-name.'.format(
                               path, ', '.join(mismatches)))
    try:
        epochs = [int(row['epoch']) for row in rows]
        for row in rows:
            if not all(math.isfinite(float(row[key])) for key in
                       ('det_auroc', 'seg_auroc', 'test_fps')):
                raise ValueError('non-finite metric')
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError('Invalid run curve at {}: {}'.format(path, exc)) from exc
    if epochs != list(range(meta_epochs)):
        raise RuntimeError('Run at {} has {} of {} epochs. This script cannot '
                           'resume model/optimizer state mid-run; preserve the partial '
                           'files and use a new --output-dir or --run-name.'.format(
                               path, len(rows), meta_epochs))
    return rows[-1], metadata, rows


def _summary_row(dataset, class_name, scales, result, metadata, rows, seed):
    best_det = max(rows, key=lambda row: float(row['det_auroc']))
    best_seg = max(rows, key=lambda row: float(row['seg_auroc']))
    return {
        'dataset': dataset, 'class_name': class_name,
        'arm': scales, 'epoch': result['epoch'],
        'det_auroc': result['det_auroc'],
        'seg_auroc': result['seg_auroc'],
        'best_det_auroc': best_det['det_auroc'],
        'best_det_epoch': best_det['epoch'],
        'best_seg_auroc': best_seg['seg_auroc'],
        'best_seg_epoch': best_seg['epoch'],
        'test_fps': result['test_fps'],
        'peak_vram_gib': result['peak_vram_gib'],
        'trainable_parameters': metadata['trainable_parameters'],
        'maca_parameters': metadata['maca_parameters'],
        'seed': seed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true',
                        help='run all 15 MVTec AD and 12 VisA classes; default is a four-class pilot')
    parser.add_argument('--datasets', nargs='+', choices=['mvtec', 'visa'],
                        default=['mvtec', 'visa'])
    parser.add_argument('--mvtec-root', default=None,
                        help='MVTec AD root; auto-detects datasets/mvtec-ad or ../mvtec-ad')
    parser.add_argument('--visa-root', default=None,
                        help='VisA root; auto-detects datasets/visa or ../visa')
    parser.add_argument('--output-dir', default='results_maca_ablation')
    parser.add_argument('--meta-epochs', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--run-name', type=int, default=0)
    parser.add_argument('--save-checkpoints', action='store_true')
    parser.add_argument('--resume', action='store_true',
                        help='skip verified complete arms; partial arms require a new run name or output directory')
    args = parser.parse_args()
    if args.meta_epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error('--meta-epochs and --batch-size must be positive; --workers cannot be negative')
    classes = FULL_CLASSES if args.full else PILOT_CLASSES
    repo_root = Path(__file__).resolve().parent
    data_roots = {
        'mvtec': Path(args.mvtec_root) if args.mvtec_root else next(
            (path for path in (repo_root / 'datasets' / 'mvtec-ad', repo_root.parent / 'mvtec-ad')
             if path.is_dir()), repo_root / 'datasets' / 'mvtec-ad'),
        'visa': Path(args.visa_root) if args.visa_root else next(
            (path for path in (repo_root / 'datasets' / 'visa', repo_root.parent / 'visa')
             if path.is_dir()), repo_root / 'datasets' / 'visa'),
    }
    for dataset in args.datasets:
        if not data_roots[dataset].is_dir():
            parser.error('{} dataset root does not exist: {}. Pass --{}-root.'.format(
                dataset, data_roots[dataset], dataset))
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = []

    for dataset in args.datasets:
        for class_name in classes[dataset]:
            for scales in ('middle', 'all'):
                arm_root = output_root / scales
                prior = _last_result(
                    arm_root, dataset, class_name, args.run_name, scales=scales,
                    seed=args.seed, meta_epochs=args.meta_epochs,
                    data_root=data_roots[dataset],
                )
                if prior is not None and not args.resume:
                    raise RuntimeError('Completed run already exists for {}/{} {}. '
                                       'Use --resume to skip it, or choose a new '
                                       '--output-dir or --run-name.'.format(
                                           dataset, class_name, scales))
                command = [
                    sys.executable, 'ADFlow_Start.py',
                    '--dataset', dataset, '--class-name', class_name,
                    '--data-path', str(data_roots[dataset]),
                    '--distribution', 'adaptive', '--maca-scales', scales,
                    '--pool-layers', '3', '--meta-epochs', str(args.meta_epochs),
                    '--batch-size', str(args.batch_size), '--workers', str(args.workers),
                    '--seed', str(args.seed), '--run-name', str(args.run_name),
                    '--output-dir', str(arm_root),
                ]
                if args.save_checkpoints:
                    command.append('--save-checkpoint')
                if prior is None:
                    print('\n>>> {}'.format(' '.join(command)), flush=True)
                    subprocess.run(command, check=True)
                    prior = _last_result(
                        arm_root, dataset, class_name, args.run_name, scales=scales,
                        seed=args.seed, meta_epochs=args.meta_epochs,
                        data_root=data_roots[dataset],
                    )
                    if prior is None:
                        raise RuntimeError('Training returned without a curve: {}'.format(arm_root))
                else:
                    print('Skipping verified complete {}/{} {}'.format(
                        dataset, class_name, scales), flush=True)
                result, metadata, rows = prior
                summary.append(_summary_row(
                    dataset, class_name, scales, result, metadata, rows, args.seed))
                summary_path = output_root / 'summary.csv'
                temporary = summary_path.with_suffix('.csv.tmp')
                with temporary.open('w', newline='', encoding='utf-8') as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
                    writer.writeheader()
                    writer.writerows(summary)
                temporary.replace(summary_path)
                print('Updated {}'.format(summary_path), flush=True)


if __name__ == '__main__':
    main()
