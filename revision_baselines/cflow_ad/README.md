# Figure 6: official CFLOW-AD comparison (preparation only)

This directory **does not implement CFLOW-AD itself**. The wrapper calls the
unmodified [official implementation](https://github.com/gudovskiy/cflow-ad)
at commit `b2ebf9e673a0aa46992a3b18367ec066a57bba89`, then collects the
official `--viz` outputs. No CFLOW-AD result or heatmap has been generated yet.

## Confirmed same-image comparisons

The two inputs below are byte-identical to images in the supplied Figure 6
PowerPoint; their original masks are present. Do not assume that the other
columns of the PowerPoint have identified source files.

| Class | Original input relative to MVTec AD root | GT relative to MVTec AD root | Official test-loader index |
| --- | --- | --- | ---: |
| cable | `cable/test/bent_wire/001.png` | `cable/ground_truth/bent_wire/001_mask.png` | 1 |
| transistor | `transistor/test/cut_lead/005.png` | `transistor/ground_truth/cut_lead/005_mask.png` | 15 |

The source audit also has high-confidence **pixel-level** matches for anomalous
MVTec AD inputs: `metal_nut/test/color/000.png`,
`metal_nut/test/bent/005.png`, `hazelnut/test/crack/000.png`,
`transistor/test/damaged_case/009.png`, and
`cable/test/missing_cable/000.png`. Their PPT versions were vertically flipped
and resized, so these are not byte-identical matches. The wrapper accepts the
four relevant MVTec AD classes and labels this weaker evidence separately in
its provenance. The source audit also lists VisA/LOCO samples, but the official
CFLOW-AD loader does not cover those datasets without adaptation.

The index comes from the upstream `MVTecDataset.load_dataset_folder`: sorted
defect subdirectories, then sorted lowercase `.png` names. The wrapper
independently recomputes it and checks input/mask hashes before execution.

## On a Linux GPU machine

1. Clone and pin the official code in a **separate checkout** (not inside this
   ADFlow source tree):

   ```bash
   git clone https://github.com/gudovskiy/cflow-ad.git /root/autodl-tmp/cflow-ad-official
   git -C /root/autodl-tmp/cflow-ad-official checkout b2ebf9e673a0aa46992a3b18367ec066a57bba89
   ```

2. Use a separate Python environment. The official README reports testing
   Python 3.8 / PyTorch 1.8, which predates RTX 4090; a modern CUDA-enabled
   PyTorch may need compatibility checks. The official loader calls
   `PIL.Image.ANTIALIAS`, so use `Pillow<10`. The upstream requirements also
   refer to FrEIA through a legacy `git://` URL; if that fails, install the
   same pinned FrEIA commit over HTTPS. Install `timm`, `numpy<2`, `scipy`,
   `scikit-learn`, `scikit-image`, `matplotlib`, and `tqdm` in that environment.
   Do not silently change the official model code to solve dependency issues.

3. Preview the exact sample mapping without using a GPU:

   ```bash
   python revision_baselines/cflow_ad/run_official_cflow.py \
     --data-root /root/autodl-tmp/mvtec-ad \
     --sample cable/test/bent_wire/001.png --dry-run
   ```

4. Run the official model for cable; repeat with
   `transistor/test/cut_lead/005.png` for the second sample:

   ```bash
   python revision_baselines/cflow_ad/run_official_cflow.py \
     --official-repo /root/autodl-tmp/cflow-ad-official \
     --data-root /root/autodl-tmp/mvtec-ad \
     --sample cable/test/bent_wire/001.png \
     --output-root results_figure6/cflow_ad
   ```

   The wrapper creates an upstream `data/MVTec-AD` symlink to the supplied
   dataset, runs the official `norm-train` entry point (25 meta-epochs,
   8 sub-epochs, batch size 8 by default), then runs the official `norm-test
   --viz` entry point with the locally trained checkpoint. The original
   paper's training batch size and any local change are recorded by the
   invocation. If CUDA memory is insufficient, retry with a smaller
   `--batch-size` and report that change.

5. If a compatible checkpoint is available, add
   `--checkpoint /absolute/path/to/model.pt` to skip training. Record its
   original source separately; the wrapper records the checkpoint SHA-256.
   The [official README's pretrained Google Drive link](https://github.com/gudovskiy/cflow-ad#testing-pretrained-models)
   has a [public invalid-link report](https://github.com/gudovskiy/cflow-ad/issues/51),
   so the workflow does not depend on those weights being available.

Each completed sample is collected in a fresh output directory with the raw
input, original GT, upstream GT visualization, upstream score visualization,
and `provenance.json` (source hashes, checkpoint hash, official revision,
image index, dependency versions). The official `sc_images_*.svg` consists of
**two panels**: heatmap overlay on top, thresholded outline underneath.
Use/crop the upper panel for a heatmap column; do not mislabel the lower panel
as a raw heatmap. This wrapper does not create performance claims or synthetic
scores.

## Scope and caveats

- CFLOW-AD officially supports MVTec AD; the current confirmed same-image
  comparison covers cable/transistor only, not the VisA or LOCO columns.
- The old Figure 6 collage contains inconsistent column labels. The audit now
  identifies 19 SVG inputs by strong pixel evidence after a vertical flip and
  resize, but the labels remain unreliable; see
  `revision_manifests/figure6_source_audit.md`. Rebuild any comparison panel
  with correct class/sample labels. A MVTec-only CFLOW-AD comparison must not
  be presented as covering the old VisA/LOCO columns.
- The official training code saves the checkpoint that maximizes test-set
  segmentation AUROC. This is upstream behavior. Document the checkpoint
  selection rule when reporting any quantitative comparison.
- Training, dependency compatibility, and SVG output collection remain
  **unverified** until the Linux GPU run. Local verification performed so far:
  Python syntax compilation and `--dry-run` on both source/GT pairs.
