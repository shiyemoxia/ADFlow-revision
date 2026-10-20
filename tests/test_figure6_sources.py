"""Small, data-independent checks for Figure 6 source verification rules."""

import numpy as np
from PIL import Image

from verify_figure6_sources import MATCHES, _candidate_pixels, masked_rgb_mse


def test_candidate_is_flipped_vertically(tmp_path):
    original = np.zeros((171, 171, 3), dtype=np.uint8)
    original[0, :, :] = 20
    original[-1, :, :] = 200
    path = tmp_path / 'input.png'
    Image.fromarray(original).save(path)

    transformed = _candidate_pixels(path)

    assert np.all(transformed[0] == 200)
    assert np.all(transformed[-1] == 20)


def test_red_annotation_is_excluded_from_mse():
    candidate = np.full((2, 2, 3), 40.0, dtype=np.float32)
    reference = candidate.copy()
    reference[0, 0] = [255.0, 0.0, 0.0]

    assert masked_rgb_mse(reference, candidate) == 0.0


def test_manifest_has_expected_anomaly_balance():
    assert len(MATCHES) == 19
    assert sum(item.mask is not None for item in MATCHES) == 13
    assert len({item.relation_id for item in MATCHES}) == 19
