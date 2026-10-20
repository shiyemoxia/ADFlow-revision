"""Run the five exact Figure 9 samples, reusing matching ablation weights if present.

This script never chooses replacement images. It verifies the supplied image/GT
manifests before launching any training, and skips only complete matching
outputs when restarted after an interruption.
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath

from verify_figure9_sources import verify


REUSABLE_ABLATION_CLASSES = {('mvtec', 'screw'), ('visa', 'capsules')}
OUTPUT_FILES = ('input.png', 'gt.png', 'heatmap.png', 'overlay.png',
                'anomaly_score.npy', 'metadata.json')


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _load_rows(index_path):
    with index_path.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    keys = [(row['dataset'], row['class_name']) for row in rows]
    if len(rows) != 5 or len(set(keys)) != 5:
        raise ValueError('Expected five distinct Figure 9 dataset/class targets')
    return rows


def _reusable_checkpoint(root, dataset, class_name, run_name):
    if root is None or (dataset, class_name) not in REUSABLE_ABLATION_CLASSES:
        return None
    path = (root / 'all' / 'checkpoints' / dataset / class_name / 'adaptive' /
            'run_{}_last.pt'.format(run_name))
    return path.resolve() if path.is_file() else None


def _validate_checkpoint(path, row, args):
    # The checkpoint comes from our own ablation run. Import torch only on the
    # GPU machine, leaving local --dry-run usable without a PyTorch install.
    import torch

    payload = torch.load(path, map_location='cpu')
    expected = {
        'format': 'adflow-revision-v1',
        'dataset': row['dataset'],
        'class_name': row['class_name'],
        'distribution': 'adaptive',
        'maca_scales': 'all',
        'pool_layers': 3,
        'seed': args.seed,
        'epoch': args.meta_epochs - 1,
        'input_size': 256,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError('Cannot reuse {}: {} is {!r}, expected {!r}'.format(
                path, key, payload.get(key), value))
    return _sha256(path)


def _sample_dir(output_root, row, run_name, checkpoint=None):
    if checkpoint is not None:
        output_root = output_root / 'inference'
    stem = PurePosixPath(row['image']).with_suffix('')
    return (output_root / 'heatmaps' / row['dataset'] / row['class_name'] /
            'adaptive' / 'run_{}'.format(run_name)).joinpath(*stem.parts)


def _complete_sample(sample_dir, row, expected_epoch):
    if not sample_dir.exists():
        return False
    missing = [name for name in OUTPUT_FILES if not (sample_dir / name).is_file()]
    if missing:
        raise RuntimeError('Partial Figure 9 output at {}: missing {}. Inspect it; '
                           'use a new --run-name to rerun without overwriting.'.format(
                               sample_dir, missing))
    metadata = json.loads((sample_dir / 'metadata.json').read_text(encoding='utf-8'))
    expected = {
        'dataset': row['dataset'], 'class_name': row['class_name'],
        'relative_image_path': row['image'], 'epoch': expected_epoch,
        'checkpoint_variant': 'all',
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise RuntimeError('Existing Figure 9 output metadata does not match this run: {}. '
                           'Use a new --run-name.'.format(sample_dir))
    return True


def _command(row, args, data_root, manifest, checkpoint):
    command = [
        sys.executable, str(Path(__file__).with_name('ADFlow_Start.py')),
        '--dataset', row['dataset'], '--class-name', row['class_name'],
        '--data-path', str(data_root), '--distribution', 'adaptive',
        '--maca-scales', 'all', '--pool-layers', '3',
        '--meta-epochs', str(args.meta_epochs), '--batch-size', str(args.batch_size),
        '--workers', str(args.workers), '--seed', str(args.seed),
        '--run-name', str(args.run_name), '--output-dir', str(args.output_dir),
        '--heatmap-targets', str(manifest),
    ]
    if checkpoint is None:
        command.append('--save-checkpoint')
    else:
        command.extend(['--action-type', 'norm-test', '--checkpoint', str(checkpoint)])
    return command


def _read_existing_manifest(path):
    if not path.is_file():
        return {}
    records = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(records, list):
        raise ValueError('Existing Figure 9 run manifest must contain a list')
    by_key = {}
    for record in records:
        key = (record['dataset'], record['class_name'], record['run_name'])
        if key in by_key:
            raise ValueError('Duplicate Figure 9 run manifest entry: {}'.format(key))
        by_key[key] = record
    return by_key


def _write_manifest(path, records):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(list(records.values()), indent=2, ensure_ascii=False) + '\n',
                         encoding='utf-8')
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mvtec-root', type=Path, default=Path('datasets/mvtec-ad'))
    parser.add_argument('--visa-root', type=Path, default=Path('datasets/visa'))
    parser.add_argument('--loco-root', type=Path, default=Path('datasets/mvtec-loco'))
    parser.add_argument('--index', type=Path, default=Path('revision_manifests/figure9_sources.csv'))
    parser.add_argument('--manifest-root', type=Path, default=Path('revision_manifests/figure9'))
    parser.add_argument('--output-dir', type=Path, default=Path('results_figure9'))
    parser.add_argument('--reuse-ablation-dir', type=Path,
                        help='reuse matching final checkpoints from the all-scale ablation arm')
    parser.add_argument('--meta-epochs', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--run-name', type=int, default=0)
    parser.add_argument('--only', nargs='+', metavar='DATASET/CLASS',
                        help='optional subset of the five exact targets')
    parser.add_argument('--dry-run', action='store_true',
                        help='check original images/GT and show commands without starting the model')
    args = parser.parse_args()
    if args.meta_epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error('epochs and batch size must be positive; workers must be nonnegative')

    roots = {'mvtec': args.mvtec_root, 'visa': args.visa_root,
             'mvtec-loco': args.loco_root}
    verify(args.index, args.manifest_root, roots)
    rows = _load_rows(args.index)
    available = {'{}/{}'.format(row['dataset'], row['class_name']) for row in rows}
    selected = set(args.only or available)
    unknown = selected - available
    if unknown:
        parser.error('Unknown Figure 9 target(s): {}'.format(sorted(unknown)))

    output_root = args.output_dir.resolve()
    reuse_root = args.reuse_ablation_dir.resolve() if args.reuse_ablation_dir else None
    run_manifest = output_root / 'figure9_run_manifest.json'
    existing_records = _read_existing_manifest(run_manifest)
    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for row in rows:
        label = '{}/{}'.format(row['dataset'], row['class_name'])
        if label not in selected:
            continue
        manifest = args.manifest_root / '{}_{}.txt'.format(row['dataset'], row['class_name'])
        checkpoint = _reusable_checkpoint(reuse_root, row['dataset'],
                                          row['class_name'], args.run_name)
        sample_dir = _sample_dir(output_root, row, args.run_name, checkpoint)
        command = _command(row, args, roots[row['dataset']].resolve(),
                           manifest.resolve(), checkpoint)
        record = {
            'dataset': row['dataset'], 'class_name': row['class_name'],
            'run_name': args.run_name,
            'source_image': row['image'], 'source_sha256': row['image_sha256'],
            'sample_dir': str(sample_dir),
            'source_checkpoint': str(checkpoint) if checkpoint else None,
            'mode': 'reuse_and_test' if checkpoint else 'train_and_export',
            'command': command, 'status': 'planned',
        }
        print('\n>>> {}'.format(' '.join(command)), flush=True)
        if args.dry_run:
            records.append(record)
            continue
        key = (row['dataset'], row['class_name'], args.run_name)
        previous = existing_records.get(key)
        if previous is not None and (previous.get('sample_dir') != str(sample_dir) or
                                     previous.get('command') != command):
            raise RuntimeError('Existing Figure 9 run uses different settings or output path: {}. '
                               'Use a new --run-name.'.format(label))
        if _complete_sample(sample_dir, row, args.meta_epochs - 1):
            if previous is None or previous.get('source_image') != row['image']:
                raise RuntimeError('Complete Figure 9 output has no matching provenance '
                                   'entry in {}: {}'.format(run_manifest, sample_dir))
            if checkpoint and previous.get('source_checkpoint_sha256') != _validate_checkpoint(
                    checkpoint, row, args):
                raise RuntimeError('Reused checkpoint has changed since Figure 9 export: {}'.format(
                    checkpoint))
            record = dict(previous)
            record['status'] = 'already_complete'
        else:
            run_output_root = output_root / 'inference' if checkpoint else output_root
            curve = (run_output_root / 'curves' / row['dataset'] / row['class_name'] /
                     'adaptive' / 'run_{}.csv'.format(args.run_name))
            if curve.exists():
                raise RuntimeError('Existing partial training curve at {}. Inspect it; '
                                   'use a new --run-name to avoid overwriting.'.format(curve))
            if checkpoint:
                record['source_checkpoint_sha256'] = _validate_checkpoint(checkpoint, row, args)
            subprocess.run(command, check=True)
            if not _complete_sample(sample_dir, row, args.meta_epochs - 1):
                raise RuntimeError('Model finished but Figure 9 heatmap is missing: {}'.format(sample_dir))
            if checkpoint is None:
                saved_checkpoint = (output_root / 'checkpoints' / row['dataset'] /
                                    row['class_name'] / 'adaptive' /
                                    'run_{}_last.pt'.format(args.run_name))
                if not saved_checkpoint.is_file():
                    raise RuntimeError('Heatmap exists but trained checkpoint is missing: {}'.format(
                        saved_checkpoint))
                record['trained_checkpoint'] = str(saved_checkpoint)
                record['trained_checkpoint_sha256'] = _sha256(saved_checkpoint)
            record['status'] = 'complete'
        records.append(record)
        existing_records[key] = record
        _write_manifest(run_manifest, existing_records)
    if args.dry_run:
        print('Dry-run verified {} Figure 9 target(s); no models were run'.format(len(records)))
    else:
        print('Finished {} Figure 9 target(s); manifest: {}'.format(
            len(records), run_manifest))


if __name__ == '__main__':
    main()
