# 第二轮实验：GPU 服务器执行顺序

本文件是**待执行方案**，不是实验结果。最后核验：2026-10-20。本地没有 PyTorch/GPU；以下训练命令必须在新租用的 Linux GPU 服务器上验证。数据、代码、权重和生成图应保留到下载并核对完成后。

## 启动服务器之前

1. 使用本轮实验分支 `codex/revision-2026-10-experiments` 的代码；不能只克隆 GitHub 默认 `main` 分支就直接运行。不要上传 `datasets/`、大权重、旧结果、论文附件或原始 PPT 到公开仓库。
2. 上传或重新下载并解压三套数据，最终目录建议为 `/root/autodl-tmp/mvtec-ad`、`/root/autodl-tmp/visa`、`/root/autodl-tmp/mvtec-loco`。本地这三套解压数据约 **12.4 GiB**，另外还要留模型权重、日志、输出和临时空间。使用此前验证过的单卡 4090D 24GB 配置即可先试跑；新机器仍须重新做 smoke test。
3. 在服务器进入包含本轮代码的 `ADFlow-revision` 项目目录。下面命令假设数据根目录采用第 2 步路径；若不同，替换三个 `--*-root` 参数。不要在数据盘核实完成前删除下载档案。
4. 以下带 `tee` 的长任务建议先在同一 Bash 终端执行 `set -o pipefail`，否则训练失败时命令行可能只显示日志程序的成功状态。中断后保留旧结果；不完整运行不能从中间 epoch 接续，需换输出目录或运行编号重跑。

## 先核验，再付费训练

```bash
python -m pip install -r requirements.txt
python smoke_test.py
python -m pytest -q tests
python verify_figure9_sources.py \
  --mvtec-root /root/autodl-tmp/mvtec-ad \
  --visa-root /root/autodl-tmp/visa \
  --loco-root /root/autodl-tmp/mvtec-loco
python run_figure9_heatmaps.py --dry-run \
  --mvtec-root /root/autodl-tmp/mvtec-ad \
  --visa-root /root/autodl-tmp/visa \
  --loco-root /root/autodl-tmp/mvtec-loco
```

若 smoke test 或文件核验失败，先修复环境/目录，不开始长时间训练。图 6 来源及像素匹配口径见 `revision_manifests/figure6_source_audit.md`。`verify_figure6_sources.py` 的完整复核已在本地完成；它需要未公开的 `实验图源.pptx` 和从 PPT 提取的光栅图，因此**不能**在只克隆公开 GitHub 代码的服务器上直接运行。无需为训练上传 PPT；如确需服务器重复来源核验，应通过私有渠道提供这些输入。

## 一、MACA 多尺度消融试跑

先跑四类（MVTec AD 的 carpet、screw；VisA 的 capsules、pcb1），三尺度与仅中间尺度各一组。加 `--save-checkpoints`，这样 screw 和 capsules 的三尺度权重可以在图 9 出图时复用，避免重复训练。

```bash
python run_maca_ablation.py \
  --mvtec-root /root/autodl-tmp/mvtec-ad \
  --visa-root /root/autodl-tmp/visa \
  --meta-epochs 25 --batch-size 8 --seed 0 \
  --save-checkpoints --output-dir results_maca_ablation \
  2>&1 | tee maca_ablation_pilot.log
```

先核对 `results_maca_ablation/summary.csv`、各组 `curves/` 和 `checkpoints/`，再决定是否运行 `--full`。如果完整运行已完成、设置完全相同，`--resume` 可以跳过已完成组；部分运行不会被覆盖，但也不能从中间 epoch 接着训。试跑四类的均值只能表述为“四类采样均值”，不能写作两套数据集的全类均值。仅中间尺度组不是未经修改的原版 ACmix。

## 二、图 9 的五张 Ours 热力图

脚本先检查五张原图的哈希及 GT，之后依次生成图。若上一步保留了设置一致的三尺度权重，会对 screw、capsules 做推理出图，其余类别重新训练；权重的类别、尺度、seed、epoch 与输入尺寸必须通过脚本检查。运行中断后，脚本只跳过文件和元数据完整的样本；不完整输出会停止并提示检查，不会擅自覆盖。

```bash
python run_figure9_heatmaps.py \
  --mvtec-root /root/autodl-tmp/mvtec-ad \
  --visa-root /root/autodl-tmp/visa \
  --loco-root /root/autodl-tmp/mvtec-loco \
  --reuse-ablation-dir results_maca_ablation \
  --meta-epochs 25 --batch-size 8 --seed 0 \
  --output-dir results_figure9 \
  2>&1 | tee figure9_heatmaps.log
```

每张样本需要同时保留 `input.png`、`gt.png`、`heatmap.png`、`overlay.png`、`anomaly_score.npy`、`metadata.json`。展示用 PNG 是逐图归一化的；定量值看原始 NPY。`results_figure9/figure9_run_manifest.json` 记录实际是复用权重还是新训练。复用权重时会把推理曲线和图片写进 `results_figure9/inference/`，与训练输出隔离；不要把手工 `norm-test` 的 `inference/` 曲线当作完整训练曲线。

## 三、图 6 官方方法对照

按 `revision_baselines/cflow_ad/README.md` 在**独立环境/仓库**中运行官方 CFLOW-AD。先只做两张与 PPT 逐字节相同的 MVTec AD 异常样本（cable、transistor），核对官方 SVG 的上半幅确为热力图叠加，再考虑扩展到像素级高置信度匹配的 MVTec AD 类别。VisA/LOCO 不在该官方仓库直接支持的范围内，不能把两张 MVTec 图写成“旧图 6 全部类别对照”。记录每张图的模型权重哈希、官方代码版本、输入和 GT 路径、渲染方式。

## 收尾与关机前检查

- [ ] 下载并备份 `results_maca_ablation/summary.csv`、曲线、日志和用到的权重。
- [ ] 下载并备份 `results_figure9/heatmaps/`、`results_figure9/inference/heatmaps/`、`figure9_run_manifest.json` 和日志。
- [ ] 下载并备份 `results_figure6/cflow_ad/` 的官方输出、权重来源和日志（仅在实际跑出后）。
- [ ] 检查图与同一原图、同一 GT 对得上；表格注明采样类别数、epoch、seed、指标口径。
- [ ] 备份核对无误后再关闭服务器，不把“命令执行结束”直接等同于“可交论文”。
