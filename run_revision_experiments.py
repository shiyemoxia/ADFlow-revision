"""Run the selected revision experiments sequentially on one GPU."""

import argparse
import subprocess
import sys
from pathlib import Path


EXPERIMENT_3 = {
    'mvtec': ('carpet', 'screw'),
    'mvtec-loco': ('pushpins', 'screw_bag'),
    'visa': ('pcb1', 'capsules'),
}

EXPERIMENT_4 = {
    'mvtec': ('screw',),
    'mvtec-loco': ('pushpins',),
    'visa': ('pcb1',),
}


def run_one(args, dataset, class_name, distribution, save_z):
    command = [
        sys.executable,
        'ADFlow_Start.py',
        '--dataset', dataset,
        '--class-name', class_name,
        '--distribution', distribution,
        '--meta-epochs', str(args.meta_epochs),
        '--sub-epochs', str(args.sub_epochs),
        '--batch-size', str(args.batch_size),
        '--workers', str(args.workers),
        '--run-name', str(args.run_name),
        '--seed', str(args.seed),
        '--output-dir', args.output_dir,
    ]
    if save_z:
        command.extend([
            '--save-z',
            '--z-layer', str(args.z_layer),
            '--z-max-samples', str(args.z_max_samples),
        ])
    print('\n>>> {}'.format(' '.join(command)), flush=True)
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True, choices=['3', '4'])
    parser.add_argument('--datasets', nargs='+',
                        choices=['mvtec', 'mvtec-loco', 'visa'],
                        default=['mvtec', 'mvtec-loco', 'visa'])
    parser.add_argument('--meta-epochs', default=15, type=int,
                        help='arbitrary converged state is sufficient for revision plots')
    parser.add_argument('--sub-epochs', default=1, type=int)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--run-name', default=0, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--z-layer', default=1, type=int)
    parser.add_argument('--z-max-samples', default=5000, type=int)
    parser.add_argument('--output-dir', default='results_revision')
    args = parser.parse_args()

    selected = EXPERIMENT_3 if args.experiment == '3' else EXPERIMENT_4
    for dataset in args.datasets:
        for class_name in selected[dataset]:
            if args.experiment == '3':
                run_one(args, dataset, class_name, 'adaptive', save_z=False)
            else:
                # Same seed and settings make this an interpretable ablation.
                run_one(args, dataset, class_name, 'standard', save_z=True)
                run_one(args, dataset, class_name, 'adaptive', save_z=True)

    if args.experiment == '3':
        output = Path(args.output_dir) / 'figures' / 'overall_average_convergence.png'
        command = [
            sys.executable,
            'plot_revision_experiments.py',
            'curve',
            '--curve-root', str(Path(args.output_dir) / 'curves'),
            '--output', str(output),
        ]
        print('\n>>> {}'.format(' '.join(command)), flush=True)
        subprocess.run(command, check=True)
    else:
        for dataset in args.datasets:
            for class_name in selected[dataset]:
                latent_dir = Path(args.output_dir) / 'latents' / dataset / class_name
                output = Path(args.output_dir) / 'figures' / 'z_surfaces' / '{}_{}.png'.format(
                    dataset, class_name
                )
                command = [
                    sys.executable,
                    'plot_revision_experiments.py',
                    'z-surface',
                    '--before', str(latent_dir / 'standard_run{}_layer{}.npz'.format(
                        args.run_name, args.z_layer)),
                    '--after', str(latent_dir / 'adaptive_run{}_layer{}.npz'.format(
                        args.run_name, args.z_layer)),
                    '--output', str(output),
                    '--seed', str(args.seed),
                ]
                print('\n>>> {}'.format(' '.join(command)), flush=True)
                subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
