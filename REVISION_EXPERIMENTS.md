# ADFlow revision experiments

The supplied source snapshot was incomplete. This branch removes unused imports
for the absent MAE code and reconstructs the adaptive-distribution likelihood from
Eqs. (10)-(16) of the paper.

## Server layout

When the repository is `/root/autodl-tmp/ADFlow-revision`, the default paths are:

```text
/root/autodl-tmp/mvtec-ad
/root/autodl-tmp/mvtec-loco
/root/autodl-tmp/visa
```

## Install

The AutoDL PyTorch image already provides PyTorch and torchvision.

```bash
cd /root/autodl-tmp/ADFlow-revision
pip install -r requirements.txt
python smoke_test.py
```

## Smoke test

Use an available LOCO category for one short epoch:

```bash
python ADFlow_Start.py \
  --dataset mvtec-loco \
  --class-name pushpins \
  --distribution adaptive \
  --meta-epochs 1 \
  --batch-size 2
```

## Experiment 3: overall average convergence

The default representative categories are:

- MVTec AD: `carpet`, `screw`
- MVTec LOCO AD: `pushpins`, `screw_bag`
- VisA: `pcb1`, `capsules`

Run all available datasets sequentially on one GPU:

```bash
python run_revision_experiments.py --experiment 3 --meta-epochs 100
```

If MVTec AD has not finished downloading, start with LOCO and VisA:

```bash
python run_revision_experiments.py \
  --experiment 3 \
  --datasets mvtec-loco visa \
  --meta-epochs 100
```

Experiment 3 defaults to 100 epochs when `--meta-epochs` is omitted. The script
writes one CSV per category and creates PNG and PDF versions of both the raw and
per-category min-max normalized overall mean curves under
`results_revision/figures/`. Flow log densities include the Jacobian term, so a
valid converged raw loss may be negative. The normalized curve is intended for the
paper because it emphasizes the common convergence trend across datasets.

The learnable MACA weights are also recorded at every epoch under
`results_revision/maca_weights/`. Each MACA layer contains scalar alpha and beta
weights; concatenating the three layer-wise values produces the exported alpha and
beta vectors. The JSON file includes the final vectors and the vectors associated
with the best image-level and pixel-level AUROC.

## Experiment 4: Z before/after AD

The AD module changes the base-density objective; it does not apply another
deterministic transformation to a fixed Z. Therefore, the defensible comparison is
between two independently trained settings with identical seed and hyperparameters:

- `standard`: CNF trained against a standard Gaussian base distribution;
- `adaptive`: CNF trained against the adaptive Gaussian mixture.

Run one representative category from every available dataset:

```bash
python run_revision_experiments.py \
  --experiment 4 \
  --datasets mvtec-loco visa \
  --meta-epochs 10
```

The final test epoch saves sampled Z vectors. A joint PCA basis is fitted to the
before/after samples, then Gaussian KDE is used to create comparable 3D density
surfaces under `results_revision/figures/z_surfaces/`.

Experiment 4 still defaults to 10 epochs because its purpose is to extract Z at a
representative converged state rather than reproduce the final paper metrics.
