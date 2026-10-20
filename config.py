from __future__ import print_function
import argparse

__all__ = ['get_args']

def get_args():
    parser = argparse.ArgumentParser(description='ADFlow revision experiments')
    parser.add_argument('--dataset', default='mvtec', type=str,
                        choices=['mvtec', 'mvtec-loco', 'visa', 'visa-new', 'btad'])
    parser.add_argument('--data-path', default=None, type=str,
                        help='dataset root; defaults to the sibling dataset folder')
    parser.add_argument('--checkpoint', default='', type=str, metavar='D',
                        help='file with saved checkpoint')
    parser.add_argument('-cl', '--class-name', required=True, type=str, metavar='C')
    parser.add_argument('-enc', '--enc-arch', default='wide_resnet50_2', type=str, metavar='A')
    parser.add_argument('-dec', '--dec-arch', default='daflow', type=str, metavar='A')
    parser.add_argument('-pl', '--pool-layers', default=3, type=int, metavar='L',
                        help='number of layers used in NF model (default: 3)')
    parser.add_argument('--maca-scales', choices=['all', 'middle'], default='all',
                        help='apply the ACmix-like block at all three feature scales or only the middle scale; the flow always uses all scales')
    parser.add_argument('-cb', '--coupling-blocks', default=8, type=int, metavar='L',
                        help='number of layers used in NF model (default: 8)')
    parser.add_argument('-run', '--run-name', default=0, type=int, metavar='C',
                        help='name of the run (default: 0)')
    parser.add_argument('-inp', '--input-size', default=256, type=int, metavar='C',
                        help='image resize dimensions (default: 256)')
    parser.add_argument("--action-type", default='norm-train', type=str, metavar='T',
                        help='norm-train/norm-test (default: norm-train)')
    parser.add_argument('-bs', '--batch-size', default=8, type=int, metavar='B',
                        help='train batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 2e-4)')
    parser.add_argument('--meta-epochs', type=int, default=25, metavar='N',
                        help='number of reported training epochs')
    parser.add_argument('--sub-epochs', type=int, default=1, metavar='N',
                        help='optimizer passes per reported epoch')
    parser.add_argument('--pro', action='store_true', default=False,
                        help='enables estimation of AUPRO metric')
    parser.add_argument('--viz', action='store_true', default=False,
                        help='saves test data visualizations')
    parser.add_argument('--workers', default=4, type=int, metavar='G',
                        help='number of data loading workers (default: 4)')
    parser.add_argument("--gpu", default='0', type=str, metavar='G',
                        help='GPU device number')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--video-path', default='.', type=str, metavar='D',
                        help='video file path')
    parser.add_argument('--anomaly-g', action='store_true',
                        help='compatibility flag for pseudo-anomaly training')
    parser.add_argument('--num-anomalies', default=0, type=int,
                        help='number of real anomalies mixed into MVTec training (default: 0)')
    parser.add_argument('--distribution', choices=['adaptive', 'standard'],
                        default='adaptive',
                        help='adaptive mixture (AD) or standard Gaussian baseline')
    parser.add_argument('--num-components', default=0, type=int,
                        help='AD mixture components; 0 selects the paper setting automatically')
    parser.add_argument('--lambda-bce', default=0.2, type=float,
                        help='weight of the pseudo-anomaly BCE term')
    parser.add_argument('--output-dir', default='results_revision', type=str,
                        help='directory for curves, latent samples and figures')
    parser.add_argument('--save-checkpoint', action='store_true',
                        help='save final-epoch model weights for later heatmap inference')
    parser.add_argument('--heatmap-targets', default='', type=str,
                        help='UTF-8 text file containing exact dataset-root-relative test image paths, one per line; exports final-epoch maps only')
    parser.add_argument('--save-z', action='store_true',
                        help='save a sample of final-epoch latent vectors')
    parser.add_argument('--z-layer', default=1, type=int,
                        help='zero-based feature scale exported by --save-z')
    parser.add_argument('--z-max-samples', default=5000, type=int,
                        help='maximum latent vectors saved per run')
    parser.add_argument('--seed', default=0, type=int,
                        help='fixed seed; use the same seed for before/after AD comparisons')
    parser.add_argument('--detect-anomaly', action='store_true',
                        help='enable slow PyTorch autograd anomaly detection')

    args = parser.parse_args()
    
    return args
