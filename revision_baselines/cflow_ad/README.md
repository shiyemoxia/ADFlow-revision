# Figure 6: CFLOW-AD comparison on the original images

This directory **does not reimplement CFLOW-AD**. It prepares the 19 source
inputs verified in `revision_manifests/figure6_source_audit.md` for the pinned,
unmodified [official CFLOW-AD code](https://github.com/gudovskiy/cflow-ad) at
`b2ebf9e673a0aa46992a3b18367ec066a57bba89`, runs one model per class,
and collects the official `--viz` output for each original Figure 6 input.
The dataset adapter is an extension of the *data*, not of the CFLOW model.
No new CFLOW-AD result is implied by this preparation; GPU training and
inference have not been verified in this checkout.

## What is adapted

The official loader accepts only 15 MVTec class names and lowercase `.png`
filenames. The old Figure 6 contains MVTec AD, VisA and MVTec LOCO inputs.
`prepare_figure6_data.py` stages the 10 required classes as follows:

| Original | CFLOW loader name | Stage operation |
| --- | --- | --- |
| MVTec AD cable, transistor, hazelnut, metal_nut | unchanged | link original PNG images and masks |
| VisA pcb2, cashew, fryum, pcb4 | bottle, carpet, leather, tile | use official VisA `1cls.csv` split; link original JPEG bytes under `.png` names, which PIL opens by content signature; link masks |
| MVTec LOCO pushpins, screw_bag | pill, toothbrush | link original PNG images; binarize and union all masks of each anomalous test image into one PNG |

These aliases are bookkeeping required by the upstream whitelist, **not** a
claim that VisA pcb2 is the MVTec bottle class, etc. The adapter manifest and
every result's `provenance.json` give the true dataset/class, original file,
ground-truth source, staged file, hash, and upstream test index. The selected
19 input image bytes remain unchanged. Normal samples have no anomalous GT.

The old PowerPoint column headings are not trustworthy; some headings do not
match the images below them. Rebuild the comparison panel using the verified
source names in the audit and the same input/GT for each method. Two MVTec
images are byte-identical to ordinary PPT image objects; the other matches
are high-confidence pixel matches after accounting for the PPT's vertical
flip, resize and red markings. Do not describe all 19 as byte-identical.

## Local preparation (no GPU needed)

Data root must contain `mvtec-ad/`, `visa/` and `mvtec-loco/` with the original
datasets, including VisA `split_csv/1cls.csv`. On this checkout:

```bash
python revision_baselines/cflow_ad/prepare_figure6_data.py \
  --datasets-root datasets \
  --output-root datasets/cflow_figure6_stage
python revision_baselines/cflow_ad/prepare_figure6_data.py \
  --output-root datasets/cflow_figure6_stage --verify-only --deep
python revision_baselines/cflow_ad/run_figure6_batch.py \
  --data-root datasets/cflow_figure6_stage --dry-run
```

The default `hardlink` mode costs little extra disk space but requires the
source and stage on the same filesystem. On different filesystems, use
`--link-mode symlink` or `--link-mode copy`. The script refuses to overwrite an
existing stage. If files are moved to a different machine, regenerate the
stage there rather than copying a manifest with machine-specific paths.

The local deep check on 2026-10-29 decoded all **6,312** staged RGB images,
verified image/mask sizes, all 19 original-image hashes and the selected
CFLOW test indices. This does **not** validate PyTorch/CUDA training.

## Linux GPU run

1. Put the datasets under one data root. Prepare and verify a new stage there,
   using the commands above with absolute paths. Keep the stage and its source
   datasets on the data disk, not the small system disk.
2. Clone the official repository in a separate directory and pin it:

   ```bash
   git clone https://github.com/gudovskiy/cflow-ad.git /root/autodl-tmp/cflow-ad-official
   git -C /root/autodl-tmp/cflow-ad-official checkout b2ebf9e673a0aa46992a3b18367ec066a57bba89
   ```

3. Use a separate Python environment. The official README reports testing
   Python 3.8 / PyTorch 1.8, which predates RTX 4090; a modern CUDA PyTorch
   requires a smoke check. The pinned loader uses `Image.ANTIALIAS`, so use
   `Pillow<10`. Its FrEIA requirement uses a legacy `git://` URL; if that
   fails, install the *same* pinned FrEIA commit over HTTPS. Also install
   `timm`, `numpy<2`, `scipy`, `scikit-learn`, `scikit-image`, `matplotlib`,
   and `tqdm`. Do not silently change the official model code.
4. Run a dry plan, then the batch runner:

   ```bash
   python revision_baselines/cflow_ad/run_figure6_batch.py \
     --data-root /root/autodl-tmp/cflow_figure6_stage --dry-run
   python revision_baselines/cflow_ad/run_figure6_batch.py \
     --official-repo /root/autodl-tmp/cflow-ad-official \
     --data-root /root/autodl-tmp/cflow_figure6_stage \
     --output-root results_figure6/cflow_ad
   ```

The batch runner trains and runs inference **once per class**, then collects
all selected samples from that class's completed visualization. Default
settings are upstream 25 meta-epochs, 8 sub-epochs, batch size 8 and 256-pixel
inputs. The 256-pixel setting is required because a previous 128-pixel test
failed at the start of training with the upstream fixed 256-fiber decoder.
If memory is insufficient, reduce `--batch-size` and disclose that change.
The runner can skip already complete outputs if their provenance and checkpoint
still match; it never overwrites an existing output directory. It does not
auto-recover a partial upstream training run without a completed result.

Each output directory contains the original input, original GT if anomalous,
the upstream score and GT SVGs, and `provenance.json`. The upstream score SVG
has **two panels**: a heatmap overlay on top and a thresholded outline below.
Crop/use the top panel for the heatmap row. The pinned writer may save SVG
bytes without `.svg` suffix; `run_official_cflow.py` verifies and renames the
collected copy. The official training code selects its checkpoint by maximum
test-set segmentation AUROC; disclose this rule if reporting quantitative
performance.

The [official pretrained-link issue](https://github.com/gudovskiy/cflow-ad/issues/51)
means this workflow assumes local training, not availability of old weights.
Do not claim the adapted VisA/LOCO runs are official out-of-the-box benchmarks;
label them as CFLOW-AD run with the documented dataset adapter.
