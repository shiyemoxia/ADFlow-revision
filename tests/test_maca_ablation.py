import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import run_maca_ablation


def write_curve(output_root, data_root, class_name, arm, epochs=2, seed=0):
    path = (output_root / arm / 'curves' / 'mvtec' / class_name /
            'adaptive' / 'run_0.csv')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            'epoch', 'det_auroc', 'seg_auroc', 'test_fps', 'peak_vram_gib'])
        writer.writeheader()
        for epoch in range(epochs):
            writer.writerow({
                'epoch': epoch, 'det_auroc': 70 + epoch,
                'seg_auroc': 80 + epoch, 'test_fps': 12,
                'peak_vram_gib': 6,
            })
    path.with_suffix('.json').write_text(json.dumps({
        'dataset': 'mvtec', 'class_name': class_name,
        'distribution': 'adaptive', 'maca_scales': arm,
        'run_name': 0, 'seed': seed, 'meta_epochs': 2,
        'data_path': str(data_root.resolve()),
        'trainable_parameters': 1234,
        'maca_parameters': 234,
    }), encoding='utf-8')
    return path


class MaccaAblationTests(unittest.TestCase):
    def test_completed_arms_can_be_resumed_without_retraining(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / 'mvtec-ad'
            data_root.mkdir()
            output_root = root / 'results'
            argv = ['run_maca_ablation.py', '--datasets', 'mvtec',
                    '--mvtec-root', str(data_root), '--output-dir', str(output_root),
                    '--meta-epochs', '2']

            def fake_run(command, check):
                self.assertTrue(check)
                class_name = command[command.index('--class-name') + 1]
                arm = Path(command[command.index('--output-dir') + 1]).name
                write_curve(output_root, data_root, class_name, arm)

            with mock.patch.object(sys, 'argv', argv), mock.patch.object(
                    run_maca_ablation.subprocess, 'run', side_effect=fake_run) as run:
                run_maca_ablation.main()
                self.assertEqual(run.call_count, 4)

            with mock.patch.object(sys, 'argv', argv + ['--resume']), mock.patch.object(
                    run_maca_ablation.subprocess, 'run') as run:
                run_maca_ablation.main()
                run.assert_not_called()

            with (output_root / 'summary.csv').open(newline='', encoding='utf-8') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[0]['best_det_epoch'], '1')
            self.assertEqual(rows[0]['best_seg_auroc'], '81')

    def test_partial_run_is_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / 'mvtec-ad'
            data_root.mkdir()
            output_root = root / 'results'
            write_curve(output_root, data_root, 'carpet', 'middle', epochs=1)
            with self.assertRaisesRegex(RuntimeError, 'cannot resume'):
                run_maca_ablation._last_result(
                    output_root / 'middle', 'mvtec', 'carpet', 0,
                    scales='middle', seed=0, meta_epochs=2,
                    data_root=data_root)

    def test_changed_seed_is_rejected_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / 'mvtec-ad'
            data_root.mkdir()
            output_root = root / 'results'
            write_curve(output_root, data_root, 'carpet', 'middle')
            with self.assertRaisesRegex(RuntimeError, 'different settings'):
                run_maca_ablation._last_result(
                    output_root / 'middle', 'mvtec', 'carpet', 0,
                    scales='middle', seed=1, meta_epochs=2,
                    data_root=data_root)


if __name__ == '__main__':
    unittest.main()
