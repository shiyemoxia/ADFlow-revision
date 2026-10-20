"""No-GPU checks for the exact Figure 9 orchestration."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from run_figure9_heatmaps import (_command, _complete_sample, _read_existing_manifest,
                                  _sample_dir, _write_manifest)


ROW = {
    'dataset': 'mvtec', 'class_name': 'screw',
    'image': 'screw/test/thread_side/019.png',
}


def test_reused_weights_run_inference_only(tmp_path):
    args = SimpleNamespace(meta_epochs=25, batch_size=8, workers=2, seed=0,
                           run_name=0, output_dir=tmp_path)
    command = _command(ROW, args, tmp_path / 'data', tmp_path / 'target.txt',
                       tmp_path / 'ablation.pt')
    assert command[command.index('--action-type') + 1] == 'norm-test'
    assert '--checkpoint' in command
    assert '--save-checkpoint' not in command
    train_command = _command(ROW, args, tmp_path / 'data', tmp_path / 'target.txt', None)
    assert '--save-checkpoint' in train_command
    assert '--action-type' not in train_command


def test_completed_sample_must_match_exact_target_and_epoch(tmp_path):
    sample_dir = _sample_dir(tmp_path, ROW, 0)
    sample_dir.mkdir(parents=True)
    with pytest.raises(RuntimeError, match='Partial'):
        _complete_sample(sample_dir, ROW, 24)
    for filename in ('input.png', 'gt.png', 'heatmap.png', 'overlay.png',
                     'anomaly_score.npy'):
        (sample_dir / filename).write_bytes(b'x')
    metadata = {
        'dataset': 'mvtec', 'class_name': 'screw',
        'relative_image_path': ROW['image'], 'epoch': 24,
        'checkpoint_variant': 'all',
    }
    (sample_dir / 'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
    assert _complete_sample(sample_dir, ROW, 24)
    with pytest.raises(RuntimeError, match='metadata'):
        _complete_sample(sample_dir, ROW, 99)


def test_reused_checkpoint_outputs_are_isolated_from_training(tmp_path):
    checkpoint = tmp_path / 'ablation.pt'
    training_sample = _sample_dir(tmp_path, ROW, 0)
    inference_sample = _sample_dir(tmp_path, ROW, 0, checkpoint)
    assert inference_sample != training_sample
    assert inference_sample.relative_to(tmp_path / 'inference')


def test_figure9_manifest_merges_subset_runs(tmp_path):
    path = tmp_path / 'figure9_run_manifest.json'
    first = ('mvtec', 'screw', 0)
    second = ('visa', 'capsules', 0)
    records = {first: {'dataset': first[0], 'class_name': first[1],
                       'run_name': first[2], 'status': 'complete'}}
    _write_manifest(path, records)
    merged = _read_existing_manifest(path)
    merged[second] = {'dataset': second[0], 'class_name': second[1],
                      'run_name': second[2], 'status': 'complete'}
    _write_manifest(path, merged)
    assert set(_read_existing_manifest(path)) == {first, second}
