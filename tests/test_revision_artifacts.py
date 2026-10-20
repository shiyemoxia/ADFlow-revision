import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from revision_artifacts import export_heatmaps, resolve_heatmap_targets


class FakeTensor:
    def __init__(self, array):
        self.array = np.asarray(array)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class FakeDataset:
    def __init__(self, path):
        self.x = [path]

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return (FakeTensor(np.full((3, 8, 8), 0.5)), 1,
                FakeTensor(np.ones((1, 8, 8))))


class ArtifactTests(unittest.TestCase):
    def test_exact_manifest_and_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'mvtec-ad'
            image_path = root / 'screw' / 'test' / 'scratch' / '001.png'
            image_path.parent.mkdir(parents=True)
            image_path.touch()
            manifest = Path(directory) / 'targets.txt'
            manifest.write_text('screw/test/scratch/001.png\n', encoding='utf-8')
            dataset = FakeDataset(image_path)
            targets = resolve_heatmap_targets(manifest, dataset, root)
            self.assertEqual(targets, [(0, 'screw/test/scratch/001.png')])
            config = SimpleNamespace(
                dataset='mvtec', class_name='screw', maca_scales='all',
                norm_mean=[0, 0, 0], norm_std=[1, 1, 1],
            )
            output = Path(directory) / 'output'
            exported = export_heatmaps(output, config, 3, dataset,
                                       np.arange(64).reshape(1, 8, 8), targets)
            self.assertEqual(len(exported), 1)
            for filename in ('anomaly_score.npy', 'heatmap.png', 'input.png',
                             'gt.png', 'overlay.png', 'metadata.json'):
                self.assertTrue((exported[0] / filename).is_file())
            np.testing.assert_array_equal(np.load(exported[0] / 'anomaly_score.npy'),
                                          np.arange(64).reshape(8, 8))

    def test_wrong_sample_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'mvtec-ad'
            image_path = root / 'screw' / 'test' / 'scratch' / '001.png'
            image_path.parent.mkdir(parents=True)
            image_path.touch()
            manifest = Path(directory) / 'targets.txt'
            manifest.write_text('screw/test/scratch/002.png\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'not in this test split'):
                resolve_heatmap_targets(manifest, FakeDataset(image_path), root)


if __name__ == '__main__':
    unittest.main()
