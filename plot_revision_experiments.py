"""Create the revision figures from saved convergence curves and Z samples."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def _read_curve(path, metric):
    epochs = []
    values = []
    with Path(path).open('r', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            value = float(row[metric])
            if np.isfinite(value):
                epochs.append(int(row['epoch']))
                values.append(value)
    return np.asarray(epochs), np.asarray(values, dtype=np.float64)


def plot_average_curve(curve_root, output, metric, distribution, normalize):
    curve_root = Path(curve_root)
    paths = sorted(curve_root.glob('**/{}/run_*.csv'.format(distribution)))
    if not paths:
        raise FileNotFoundError(
            "No '{}' curve CSV files found under {}".format(distribution, curve_root)
        )

    curves = []
    labels = []
    common_length = None
    for path in paths:
        epochs, values = _read_curve(path, metric)
        if len(values) < 2:
            continue
        if normalize == 'first':
            denominator = values[0] if abs(values[0]) > 1e-12 else 1.0
            values = values / denominator
        elif normalize == 'minmax':
            span = values.max() - values.min()
            values = (values - values.min()) / (span if span > 1e-12 else 1.0)
        common_length = len(values) if common_length is None else min(common_length, len(values))
        curves.append(values)
        labels.append('{}/{}'.format(path.parents[2].name, path.parents[1].name))

    if not curves:
        raise ValueError('Curve files were found, but none contained at least two valid values')
    curves = np.stack([curve[:common_length] for curve in curves], axis=0)
    epochs = np.arange(common_length)
    mean = curves.mean(axis=0)
    std = curves.std(axis=0)

    fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    for curve, label in zip(curves, labels):
        ax.plot(epochs, curve, color='#8fb6d9', alpha=0.28, linewidth=1.0, label=label)
    ax.fill_between(epochs, mean - std, mean + std, color='#2468a2', alpha=0.18,
                    label='Mean +/- 1 SD')
    ax.plot(epochs, mean, color='#123f68', linewidth=2.8, label='Overall mean')
    ax.set_xlabel('Epoch')
    ylabel = metric.replace('_', ' ').title()
    if normalize != 'none':
        ylabel = '{} ({})'.format(ylabel, normalize)
    ax.set_ylabel(ylabel)
    ax.set_title('Overall Average Convergence Curve')
    ax.grid(alpha=0.22)
    ax.legend(loc='best', fontsize=8, ncol=2)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    fig.savefig(output.with_suffix('.pdf'))
    plt.close(fig)
    print('Saved {}'.format(output))


def _load_latent(path, max_points, seed):
    payload = np.load(path, allow_pickle=False)
    z = payload['z'].astype(np.float64, copy=False)
    labels = payload['labels'].astype(np.uint8, copy=False)
    metadata = json.loads(str(payload['metadata']))
    if len(z) > max_points:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(z), max_points, replace=False))
        z = z[idx]
        labels = labels[idx]
    return z, labels, metadata


def _kde_surface(points, x_grid, y_grid, bandwidth):
    kde = gaussian_kde(points.T, bw_method=bandwidth)
    positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
    density = kde(positions).reshape(x_grid.shape)
    return density / max(density.max(), np.finfo(np.float64).eps)


def plot_z_surfaces(before, after, output, max_points, grid_size, bandwidth, seed):
    z_before, labels_before, meta_before = _load_latent(before, max_points, seed)
    z_after, labels_after, meta_after = _load_latent(after, max_points, seed)
    if z_before.shape[1] != z_after.shape[1]:
        raise ValueError('Before/after Z dimensions differ: {} vs {}'.format(
            z_before.shape[1], z_after.shape[1]))

    combined = np.concatenate([z_before, z_after], axis=0)
    combined = StandardScaler().fit_transform(combined)
    projected = PCA(n_components=2, random_state=seed).fit_transform(combined)
    before_2d = projected[:len(z_before)]
    after_2d = projected[len(z_before):]

    lower = np.percentile(projected, 1.0, axis=0)
    upper = np.percentile(projected, 99.0, axis=0)
    padding = 0.08 * np.maximum(upper - lower, 1e-6)
    x = np.linspace(lower[0] - padding[0], upper[0] + padding[0], grid_size)
    y = np.linspace(lower[1] - padding[1], upper[1] + padding[1], grid_size)
    x_grid, y_grid = np.meshgrid(x, y)
    density_before = _kde_surface(before_2d, x_grid, y_grid, bandwidth)
    density_after = _kde_surface(after_2d, x_grid, y_grid, bandwidth)

    fig = plt.figure(figsize=(13.0, 5.6), constrained_layout=True)
    titles = ('Before AD (standard Gaussian)', 'After AD (adaptive mixture)')
    densities = (density_before, density_after)
    point_sets = (before_2d, after_2d)
    label_sets = (labels_before, labels_after)
    for index, (title, density, points, labels) in enumerate(
        zip(titles, densities, point_sets, label_sets), start=1
    ):
        ax = fig.add_subplot(1, 2, index, projection='3d')
        ax.plot_surface(
            x_grid,
            y_grid,
            density,
            cmap=cm.viridis,
            linewidth=0,
            antialiased=True,
            alpha=0.92,
        )
        if labels.any():
            anomaly_points = points[labels.astype(bool)]
            ax.scatter(
                anomaly_points[:, 0],
                anomaly_points[:, 1],
                np.zeros(len(anomaly_points)),
                color='#d94841',
                s=5,
                alpha=0.35,
                label='Anomalous patches',
            )
            ax.legend(loc='upper right', fontsize=8)
        ax.set_title(title, pad=12)
        ax.set_xlabel('Joint PCA 1')
        ax.set_ylabel('Joint PCA 2')
        ax.set_zlabel('Normalized KDE density')
        ax.set_zlim(0.0, 1.05)
        ax.view_init(elev=31, azim=-58)

    dataset = meta_after.get('dataset', 'dataset')
    class_name = meta_after.get('class_name', 'class')
    fig.suptitle('Latent Z Density Surface: {} / {}'.format(dataset, class_name), fontsize=15)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    fig.savefig(output.with_suffix('.pdf'))
    plt.close(fig)
    print('Saved {}'.format(output))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)

    curve = subparsers.add_parser('curve', help='plot the overall average convergence curve')
    curve.add_argument('--curve-root', default='results_revision/curves')
    curve.add_argument('--output', default='results_revision/figures/overall_average_convergence.png')
    curve.add_argument('--metric', default='train_loss')
    curve.add_argument('--distribution', default='adaptive', choices=['adaptive', 'standard'])
    curve.add_argument('--normalize', default='none', choices=['none', 'first', 'minmax'],
                       help='raw per-dimension loss is the default because flow NLL can be negative')

    surface = subparsers.add_parser('z-surface', help='plot before/after AD Z density surfaces')
    surface.add_argument('--before', required=True)
    surface.add_argument('--after', required=True)
    surface.add_argument('--output', required=True)
    surface.add_argument('--max-points', default=5000, type=int)
    surface.add_argument('--grid-size', default=90, type=int)
    surface.add_argument('--bandwidth', default=None, type=float)
    surface.add_argument('--seed', default=0, type=int)
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == 'curve':
        plot_average_curve(
            args.curve_root,
            args.output,
            args.metric,
            args.distribution,
            args.normalize,
        )
    else:
        plot_z_surfaces(
            args.before,
            args.after,
            args.output,
            args.max_points,
            args.grid_size,
            args.bandwidth,
            args.seed,
        )


if __name__ == '__main__':
    main()
