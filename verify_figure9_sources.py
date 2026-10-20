"""Check Figure 9 target manifests against exact images and ground truth."""

import argparse
import csv
import hashlib
from pathlib import Path


ROOT_OPTIONS = {
    'mvtec': 'mvtec_root',
    'visa': 'visa_root',
    'mvtec-loco': 'loco_root',
}


def _inside(root, relative_path):
    path = (root / relative_path).resolve()
    path.relative_to(root.resolve())
    return path


def verify(index_path, manifest_root, roots):
    with Path(index_path).open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError('Figure 9 source index is empty')
    visa_rows = None
    failures = []
    for row in rows:
        dataset, class_name = row['dataset'], row['class_name']
        root = Path(roots[dataset]).resolve()
        label = f'{dataset}/{class_name}'
        try:
            image = _inside(root, row['image'])
            mask = _inside(root, row['mask'])
            manifest = Path(manifest_root) / f'{dataset}_{class_name}.txt'
            entries = [line.strip() for line in manifest.read_text(encoding='utf-8').splitlines()
                       if line.strip() and not line.lstrip().startswith('#')]
            if entries != [row['image']]:
                raise ValueError(f'manifest has unexpected targets: {entries}')
            if not image.is_file() or not mask.is_file():
                raise FileNotFoundError(f'image={image.is_file()}, mask={mask.is_file()}')
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if digest != row['image_sha256']:
                raise ValueError(f'image hash mismatch: {digest}')
            if dataset == 'visa':
                if visa_rows is None:
                    split_csv = root / 'split_csv' / '1cls.csv'
                    with split_csv.open(newline='', encoding='utf-8') as handle:
                        visa_rows = {item['image']: item for item in csv.DictReader(handle)}
                split = visa_rows.get(row['image'])
                if split is None or split['split'] != 'test' or split['mask'] != row['mask']:
                    raise ValueError('image/mask is not the expected VisA test sample')
            print(f'OK {label}: {row["image"]}')
        except (KeyError, OSError, ValueError) as error:
            failures.append(f'{label}: {error}')
    if failures:
        raise RuntimeError('Figure 9 source checks failed:\n' + '\n'.join(failures))
    print(f'Verified {len(rows)} Figure 9 images, manifests, and masks')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', default='revision_manifests/figure9_sources.csv')
    parser.add_argument('--manifest-root', default='revision_manifests/figure9')
    parser.add_argument('--mvtec-root', default='datasets/mvtec-ad')
    parser.add_argument('--visa-root', default='datasets/visa')
    parser.add_argument('--loco-root', default='datasets/mvtec-loco')
    args = parser.parse_args()
    roots = {dataset: getattr(args, option) for dataset, option in ROOT_OPTIONS.items()}
    verify(args.index, args.manifest_root, roots)


if __name__ == '__main__':
    main()
