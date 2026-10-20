"""Read-only verification of the 19 Figure 6 input images recovered from PPT SVGs.

The original slide embeds resized, vertically flipped images in SVG shapes.
This script decodes the PPT itself, checks the previously extracted PNGs, then
compares each input against the specified test image in its dataset. With the
default full ranking, it also checks that the asserted image is the best match
within its dataset class, ahead of the next closest test image.

Neither the PPT nor dataset files are changed. Run from the repository root::

    python verify_figure6_sources.py

Use --skip-ranking for a faster identity/GT recheck, or override input paths
when the datasets are installed elsewhere.
"""

import argparse
import base64
import csv
import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree

import numpy as np
from PIL import Image, ImageOps
from pptx import Presentation


REL_EMBED = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed'
XLINK_HREF = '{http://www.w3.org/1999/xlink}href'
SVG_IMAGE = '{http://www.w3.org/2000/svg}image'
DATASET_ROOT_OPTIONS = {
    'mvtec-ad': 'mvtec_root',
    'visa': 'visa_root',
    'mvtec-loco': 'loco_root',
}


@dataclass(frozen=True)
class Match:
    relation_id: str
    image: str  # relative to datasets/
    mask: Optional[str]
    reported_mse: float
    reported_second_mse: float

    @property
    def dataset(self):
        return self.image.split('/', 1)[0]

    @property
    def class_name(self):
        return self.image.split('/', 2)[1]


# These are the claims in revision_manifests/figure6_source_audit.md, copied
# here so verification remains independent of the prose table's formatting.
MATCHES = (
    Match('rId4', 'visa/pcb2/Data/Images/Normal/0690.JPG', None, 17.0, 339.0),
    Match('rId5', 'mvtec-ad/metal_nut/test/good/017.png', None, 3.6, 395.6),
    Match('rId6', 'mvtec-ad/metal_nut/test/color/000.png',
          'mvtec-ad/metal_nut/ground_truth/color/000_mask.png', 3.7, 264.7),
    Match('rId7', 'visa/pcb2/Data/Images/Anomaly/013.JPG',
          'visa/pcb2/Data/Masks/Anomaly/013.png', 18.0, 309.9),
    Match('rId8', 'mvtec-ad/cable/test/good/046.png', None, 6.5, 771.4),
    Match('rId10', 'mvtec-ad/transistor/test/good/050.png', None, 3.8, 444.9),
    Match('rId13', 'mvtec-loco/pushpins/test/logical_anomalies/071.png',
          'mvtec-loco/pushpins/ground_truth/logical_anomalies/071/000.png', 13.2, 996.6),
    Match('rId14', 'mvtec-loco/pushpins/test/good/131.png', None, 11.6, 937.6),
    Match('rId15', 'mvtec-loco/screw_bag/test/good/046.png', None, 9.0, 710.2),
    Match('rId16', 'mvtec-ad/hazelnut/test/crack/000.png',
          'mvtec-ad/hazelnut/ground_truth/crack/000_mask.png', 1.5, 132.4),
    Match('rId17', 'mvtec-ad/metal_nut/test/bent/005.png',
          'mvtec-ad/metal_nut/ground_truth/bent/005_mask.png', 4.8, 250.1),
    Match('rId18', 'mvtec-ad/transistor/test/damaged_case/009.png',
          'mvtec-ad/transistor/ground_truth/damaged_case/009_mask.png', 5.5, 299.8),
    Match('rId19', 'mvtec-ad/cable/test/missing_cable/000.png',
          'mvtec-ad/cable/ground_truth/missing_cable/000_mask.png', 9.0, 1658.3),
    Match('rId20', 'visa/cashew/Data/Images/Anomaly/019.JPG',
          'visa/cashew/Data/Masks/Anomaly/019.png', 11.6, 344.8),
    Match('rId21', 'visa/fryum/Data/Images/Anomaly/074.JPG',
          'visa/fryum/Data/Masks/Anomaly/074.png', 6.3, 576.6),
    Match('rId22', 'visa/pcb4/Data/Images/Anomaly/011.JPG',
          'visa/pcb4/Data/Masks/Anomaly/011.png', 13.0, 308.9),
    Match('rId28', 'mvtec-loco/screw_bag/test/logical_anomalies/105.png',
          'mvtec-loco/screw_bag/ground_truth/logical_anomalies/105/000.png', 11.2, 1013.9),
    Match('rId33', 'mvtec-ad/transistor/test/cut_lead/005.png',
          'mvtec-ad/transistor/ground_truth/cut_lead/005_mask.png', 4.1, 264.4),
    Match('rId36', 'mvtec-ad/cable/test/bent_wire/001.png',
          'mvtec-ad/cable/ground_truth/bent_wire/001_mask.png', 5.6, 831.4),
)

# The two originals also occur as ordinary PPT picture objects. A matching
# SHA-256 of their original encoded bytes is stronger than pixel similarity.
BYTE_IDENTICAL = (
    ('mvtec-ad/cable/test/bent_wire/001.png', (13, 78),
     '6f906f05817d54a62532add0da5f56314a5508babd5849d5a3fa571001ad1a88'),
    ('mvtec-ad/transistor/test/cut_lead/005.png', (16, 17, 63),
     '37d6d29ececc6e1c497d3e67a625c318a2024a0a865aa29df44028d528f35395'),
)


def _inside(root, relative_path):
    root = Path(root).resolve()
    path = (root / relative_path).resolve()
    path.relative_to(root)
    return path


def _ppt_inputs(pptx_path, slide_number=6):
    presentation = Presentation(pptx_path)
    slide = presentation.slides[slide_number - 1]
    decoded = {}
    for shape in slide.shapes:
        for node in shape._element.iter():
            if not node.tag.endswith('svgBlip'):
                continue
            relation_id = node.get(REL_EMBED)
            if not relation_id or relation_id in decoded:
                continue
            svg_part = slide.part.related_part(relation_id)
            root = ElementTree.fromstring(svg_part.blob)
            for raster in root.iter(SVG_IMAGE):
                uri = raster.get(XLINK_HREF, '')
                if not uri.startswith('data:image/') or ';base64,' not in uri:
                    continue
                payload = base64.b64decode(uri.split(',', 1)[1], validate=True)
                decoded[relation_id] = payload  # first raster is the input image
                break
    return slide, decoded


def _reference_pixels(payload):
    with Image.open(BytesIO(payload)) as image:
        if image.size != (171, 171):
            raise ValueError(f'Expected a 171x171 SVG input, got {image.size}')
        return np.asarray(image.convert('RGB'), dtype=np.float32)


def _candidate_pixels(path):
    with Image.open(path) as image:
        resized = image.convert('RGB').resize((171, 171), Image.Resampling.LANCZOS)
        return np.asarray(ImageOps.flip(resized), dtype=np.float32)


def masked_rgb_mse(reference, candidate):
    """Compare after ignoring red anomaly-outline pixels painted over the PPT."""
    if reference.shape != candidate.shape:
        raise ValueError(f'Image shapes differ: {reference.shape} != {candidate.shape}')
    red, green, blue = reference[:, :, 0], reference[:, :, 1], reference[:, :, 2]
    marked = (red > 1.3 * green) & (red > 1.3 * blue) & (red > 80)
    valid = ~marked
    if not np.any(valid):
        raise ValueError('No unmarked pixels remain for comparison')
    return float(np.mean((reference[valid] - candidate[valid]) ** 2))


def _candidate_files(match, dataset_root):
    class_root = _inside(dataset_root, match.class_name)
    if match.dataset == 'visa':
        folders = (class_root / 'Data' / 'Images' / 'Normal',
                   class_root / 'Data' / 'Images' / 'Anomaly')
        paths = (path for folder in folders for path in folder.glob('*'))
    else:
        paths = (class_root / 'test').glob('*/*')
    return sorted(path for path in paths if path.is_file()
                  and path.suffix.lower() in {'.png', '.jpg', '.jpeg'})


def _check_mask(match, dataset_root):
    if match.mask is None:
        if match.dataset == 'visa':
            normal = '/Normal/' in match.image
        else:
            normal = '/test/good/' in match.image
        if not normal:
            raise ValueError(f'Unmasked image is not a normal test sample: {match.image}')
        return
    if match.dataset == 'visa':
        anomalous = '/Anomaly/' in match.image
    else:
        anomalous = '/test/' in match.image and '/test/good/' not in match.image
    if not anomalous:
        raise ValueError(f'Masked image is not an anomalous test sample: {match.image}')
    relative = match.mask.split('/', 1)[1]
    mask_path = _inside(dataset_root, relative)
    if not mask_path.is_file():
        raise FileNotFoundError(f'Missing GT: {mask_path}')
    with Image.open(mask_path) as mask:
        if not np.any(np.asarray(mask)):
            raise ValueError(f'GT is empty: {mask_path}')


def _check_byte_identical(slide, roots):
    for relative_image, shape_numbers, expected_digest in BYTE_IDENTICAL:
        dataset, inside_dataset = relative_image.split('/', 1)
        image_path = _inside(roots[dataset], inside_dataset)
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if digest != expected_digest:
            raise ValueError(f'Original image SHA-256 mismatch: {relative_image}')
        for number in shape_numbers:
            shape_digest = hashlib.sha256(slide.shapes[number - 1].image.blob).hexdigest()
            if shape_digest != digest:
                raise ValueError(f'PPT shape {number} differs from {relative_image}')


def verify(pptx_path, raster_root, index_path, roots, rank_all=True):
    slide, payloads = _ppt_inputs(pptx_path)
    with Path(index_path).open(newline='', encoding='utf-8') as handle:
        indexed = {row['relation_id']: row for row in csv.DictReader(handle)
                   if row['embedded_index'] == '1'}
    references = {}
    by_class = {}
    visa_split = None
    for match in MATCHES:
        label = f'{match.relation_id} {match.image}'
        if match.relation_id not in payloads or match.relation_id not in indexed:
            raise ValueError(f'Missing SVG input in PPT or raster index: {label}')
        payload = payloads[match.relation_id]
        index = indexed[match.relation_id]
        digest = hashlib.sha256(payload).hexdigest()
        if digest != index['sha256']:
            raise ValueError(f'PPT SVG/index SHA-256 differs: {label}')
        extracted = _inside(raster_root, index['raster_file'])
        if extracted.read_bytes() != payload:
            raise ValueError(f'Extracted raster differs from PPT SVG: {label}')
        dataset_root = roots[match.dataset]
        image_path = _inside(dataset_root, match.image.split('/', 1)[1])
        if not image_path.is_file():
            raise FileNotFoundError(f'Missing original image: {image_path}')
        _check_mask(match, dataset_root)
        if match.dataset == 'visa':
            if visa_split is None:
                split_path = _inside(dataset_root, 'split_csv/1cls.csv')
                with split_path.open(newline='', encoding='utf-8') as handle:
                    visa_split = {row['image']: row for row in csv.DictReader(handle)}
            row = visa_split.get(match.image.split('/', 1)[1])
            expected_mask = match.mask.split('/', 1)[1] if match.mask else ''
            if row is None or row['split'] != 'test' or row['mask'] != expected_mask:
                raise ValueError(f'VisA test split or mask disagrees: {label}')
        reference = _reference_pixels(payload)
        mse = masked_rgb_mse(reference, _candidate_pixels(image_path))
        if abs(mse - match.reported_mse) > 0.11:
            raise ValueError(f'MSE disagrees with audit for {label}: {mse:.3f} vs '
                             f'{match.reported_mse:.1f}')
        references[match.relation_id] = reference
        by_class.setdefault((match.dataset, match.class_name), []).append(match)
        print(f'PASS {label}: MSE={mse:.2f}, GT={"yes" if match.mask else "normal"}')

    _check_byte_identical(slide, roots)
    print('PASS 2 originals are SHA-256-identical to PPT picture objects')

    if rank_all:
        second_disagreements = []
        for (dataset, class_name), group in by_class.items():
            dataset_root = Path(roots[dataset]).resolve()
            candidates = _candidate_files(group[0], dataset_root)
            if len(candidates) < 2:
                raise ValueError(f'Too few test candidates for {dataset}/{class_name}')
            top = {match.relation_id: [] for match in group}
            for candidate_path in candidates:
                pixels = _candidate_pixels(candidate_path)
                for match in group:
                    score = masked_rgb_mse(references[match.relation_id], pixels)
                    leaders = top[match.relation_id]
                    leaders.append((score, candidate_path))
                    leaders.sort(key=lambda pair: pair[0])
                    del leaders[2:]
            for match in group:
                best, second = top[match.relation_id]
                original = _inside(dataset_root, match.image.split('/', 1)[1])
                if best[1] != original:
                    raise ValueError(f'{match.relation_id}: nearest image is {best[1]}, not '
                                     f'{original}')
                if abs(second[0] - match.reported_second_mse) > 0.11:
                    second_disagreements.append(
                        f'{match.relation_id}: second MSE is {second[0]:.3f}, '
                        f'audit says {match.reported_second_mse:.1f} '
                        f'({second[1].relative_to(dataset_root).as_posix()})'
                    )
                print(f'RANK {match.relation_id}: 1st={best[0]:.2f}, '
                      f'2nd={second[0]:.2f}, {len(candidates)} same-class tests')

    abnormal = sum(match.mask is not None for match in MATCHES)
    print(f'Verified {len(MATCHES)} SVG inputs ({abnormal} anomalous, '
          f'{len(MATCHES) - abnormal} normal) and 2 byte-identical originals')
    print('Evidence: high-confidence pixel identity; SVG inputs are not byte-identical '
          'to original dataset files.')
    if rank_all and second_disagreements:
        raise RuntimeError('Reported runner-up scores need correction:\n' +
                           '\n'.join(second_disagreements))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pptx', default='实验图源.pptx')
    parser.add_argument('--raster-root', default='results_figure_matching/svg_rasters')
    parser.add_argument('--index', default='results_figure_matching/svg_rasters/slide06_svg_rasters.csv')
    parser.add_argument('--mvtec-root', default='datasets/mvtec-ad')
    parser.add_argument('--visa-root', default='datasets/visa')
    parser.add_argument('--loco-root', default='datasets/mvtec-loco')
    parser.add_argument('--skip-ranking', action='store_true',
                        help='skip same-class first/second-place search')
    args = parser.parse_args()
    roots = {name: getattr(args, option) for name, option in DATASET_ROOT_OPTIONS.items()}
    verify(args.pptx, args.raster_root, args.index, roots, not args.skip_ranking)


if __name__ == '__main__':
    main()
