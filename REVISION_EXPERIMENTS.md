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
python run_revision_experiments.py --experiment 3 --meta-epochs 15
```

If MVTec AD has not finished downloading, start with LOCO and VisA:

```bash
python run_revision_experiments.py \
  --experiment 3 \
  --datasets mvtec-loco visa \
  --meta-epochs 15
```

The script writes one CSV per category and creates both PNG and PDF versions of
the overall mean curve under `results_revision/figures/`. The curve averages the
raw per-feature-dimension loss. Flow log densities include the Jacobian term, so a
valid converged loss may be negative and must not be divided by its first value.

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
before/after samples, then Gaussian KDE is used to create comparable multi-peak 3D
density surfaces under `results_revision/figures/z_surfaces/`.

The revision request explicitly permits any converged state, so increasing epochs is
optional and should be based on whether the loss curve has visibly stabilized.
