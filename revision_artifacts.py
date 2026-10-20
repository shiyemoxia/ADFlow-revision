"""Reproducible checkpoints and sample-matched anomaly-map exports."""

import json
import os
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image
try:
    from matplotlib import colormaps
    _TURBO = colormaps['turbo']
except ImportError:  # older Matplotlib on some GPU images
    from matplotlib import cm
    _TURBO = cm.get_cmap('turbo')


def _cpu_state(module):
    return {name: tensor.detach().cpu() for name, tensor in module.state_dict().items()}


def save_revision_checkpoint(path, config, epoch, acas, variancenets, decoders, metrics):
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'format': 'adflow-revision-v1',
        'dataset': config.dataset,
        'class_name': config.class_name,
        'distribution': config.distribution,
        'maca_scales': config.maca_scales,
        'pool_layers': config.pool_layers,
        'enc_arch': config.enc_arch,
        'dec_arch': config.dec_arch,
        'coupling_blocks': config.coupling_blocks,
        'input_size': config.input_size,
        'num_components': config.num_components,
        'seed': config.seed,
        'epoch': int(epoch),
        'metrics': dict(metrics),
        'acas': [_cpu_state(module) for module in acas],
        'variancenets': [_cpu_state(module) for module in variancenets],
        'decoders': [_cpu_state(module) for module in decoders],
    }
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def load_revision_checkpoint(path, config, acas, variancenets, decoders):
    import torch

    payload = torch.load(path, map_location='cpu')
    if not isinstance(payload, dict) or payload.get('format') != 'adflow-revision-v1':
        raise ValueError('This is not a revision checkpoint: {}'.format(path))
    for key in ('dataset', 'class_name', 'distribution', 'maca_scales', 'pool_layers',
                'enc_arch', 'dec_arch', 'coupling_blocks', 'input_size', 'num_components'):
        if payload[key] != getattr(config, key):
            raise ValueError('Checkpoint {} mismatch: saved {!r}, requested {!r}'.format(
                key, payload[key], getattr(config, key)))
    for name, modules in (('acas', acas), ('variancenets', variancenets), ('decoders', decoders)):
        if len(payload[name]) != len(modules):
            raise ValueError('Checkpoint {} layer count mismatch'.format(name))
        for module, state in zip(modules, payload[name]):
            module.load_state_dict(state, strict=True)
    return payload


def dataset_image_paths(dataset):
    for attribute in ('image_paths', 'img_paths', 'x'):
        paths = getattr(dataset, attribute, None)
        if paths is not None:
            if len(paths) != len(dataset):
                raise ValueError('Dataset {} path count does not match sample count'.format(attribute))
            return [Path(path) for path in paths]
    raise ValueError('Cannot get ordered image paths from {}'.format(type(dataset).__name__))


def _relative_image_path(path, data_root):
    return path.resolve().relative_to(Path(data_root).resolve()).as_posix()


def resolve_heatmap_targets(manifest_path, dataset, data_root):
    """Resolve exact dataset-root-relative paths; reject missing or ambiguous entries."""
    entries = []
    for raw in Path(manifest_path).read_text(encoding='utf-8-sig').splitlines():
        entry = raw.strip().replace('\\', '/')
        if entry and not entry.startswith('#'):
            parts = PurePosixPath(entry).parts
            if entry.startswith('/') or '..' in parts or ':' in entry:
                raise ValueError('Use a dataset-root-relative path: {}'.format(entry))
            entries.append(PurePosixPath(entry).as_posix())
    if not entries:
        raise ValueError('Heatmap target file is empty: {}'.format(manifest_path))
    if len(set(entries)) != len(entries):
        raise ValueError('Heatmap target file contains duplicate paths')
    ordered = dataset_image_paths(dataset)
    lookup = {_relative_image_path(path, data_root): index for index, path in enumerate(ordered)}
    missing = [entry for entry in entries if entry not in lookup]
    if missing:
        raise ValueError('Target images are not in this test split: {}'.format(missing))
    return [(lookup[entry], entry) for entry in entries]


def export_heatmaps(output_dir, config, epoch, dataset, anomaly_maps, targets):
    """Keep raw scores and make display-only per-image normalized heatmaps."""
    output_dir = Path(output_dir)
    maps = np.asarray(anomaly_maps)
    if len(maps) != len(dataset):
        raise ValueError('Anomaly-map count does not match the test dataset')
    output_paths = []
    for index, relative_path in targets:
        stem = PurePosixPath(relative_path).with_suffix('')
        sample_dir = output_dir.joinpath(*stem.parts)
        sample_dir.mkdir(parents=True, exist_ok=True)
        scores = np.asarray(maps[index], dtype=np.float32)
        if not np.isfinite(scores).all():
            raise ValueError('Non-finite anomaly scores for {}'.format(relative_path))
        np.save(sample_dir / 'anomaly_score.npy', scores)
        lo, hi = float(scores.min()), float(scores.max())
        display = (scores - lo) / (hi - lo) if hi > lo else np.zeros_like(scores)
        heatmap = Image.fromarray((_TURBO(display)[..., :3] * 255).astype(np.uint8))
        heatmap.save(sample_dir / 'heatmap.png')
        image, label, mask = dataset[index]
        image = image.detach().cpu().numpy().transpose(1, 2, 0)
        image = np.clip(image * np.asarray(config.norm_std) + np.asarray(config.norm_mean), 0, 1)
        input_image = Image.fromarray((image * 255).astype(np.uint8))
        input_image.save(sample_dir / 'input.png')
        mask_array = np.asarray(mask.detach().cpu().numpy()).squeeze()
        Image.fromarray((mask_array > 0.5).astype(np.uint8) * 255).save(sample_dir / 'gt.png')
        Image.blend(input_image, heatmap, 0.45).save(sample_dir / 'overlay.png')
        metadata = {
            'dataset': config.dataset, 'class_name': config.class_name,
            'relative_image_path': relative_path, 'test_index': int(index),
            'label': int(label), 'epoch': int(epoch),
            'checkpoint_variant': config.maca_scales,
            'raw_score_min': lo, 'raw_score_max': hi,
            'display_normalization': 'per-image min-max; use anomaly_score.npy for quantitative values',
        }
        (sample_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        output_paths.append(sample_dir)
    return output_paths
