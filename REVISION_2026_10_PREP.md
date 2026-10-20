# ADFlow 第二轮修改：本地准备与服务器执行

当前状态（2026-10-20）：MVTec AD、VisA、MVTec LOCO 三套数据已下载并解压到本机 `datasets/`，但本机没有用于正式训练的 GPU。图 9 的五张输入图已与 PPT 嵌入图片逐一进行 SHA-256 核对，原图和 GT 路径见 `revision_manifests/figure9_sources.csv`。**尚未产生本轮实验的性能表或新热力图**。旧服务器权重未保留，图 9 热力图需重新训练模型后导出。数据压缩包和解压目录都不要提交到 GitHub。

## 实验 1：MACA 多尺度消融（三尺度与单尺度）

入口：`run_maca_ablation.py`。默认先在 MVTec AD 的 `carpet,screw` 和 VisA 的 `capsules,pcb1` 上做四类试跑；加 `--full` 才运行 MVTec AD 全 15 类和 VisA 全 12 类。每类依次运行：

- `all`：现有模块作用于三层特征；
- `middle`：相同模块只作用于中间层，其余两层为恒等映射；三层骨干特征、三条 Flow 分支、AD 分布、数据划分与训练设置均保持不变。

使用同一随机种子、输入分辨率、epoch 和 batch size。结果写到 `results_maca_ablation/summary.csv`，包含最终 epoch 的图像 AUROC、像素 AUROC、测试集端到端 FPS、训练/测试峰值显存、总可训练参数和模块参数。两个实验臂的输出分别放在 `all/` 和 `middle/` 下，不互相覆盖。`--save-checkpoints` 可另外保存最终权重，但会占用较多磁盘。

**解释边界：**老师已确认将此项表述为 MACA 的多尺度消融。`middle` 是现有模块仅作用于中间层的单尺度组；它不是未经修改的原版 ACmix，两组参数量也不相等。结果只能用于分析多尺度设计的影响，不能写成“MACA 对比原版 ACmix 的同参数量实验”。FPS 包含数据读取与评分，是端到端测试速度，不是纯模型前向延迟。

服务器具备数据和依赖之后，在仓库目录运行：

```bash
python smoke_test.py
python run_maca_ablation.py --mvtec-root /root/autodl-tmp/mvtec-ad --visa-root /root/autodl-tmp/visa --meta-epochs 25
```

四类试跑正常后，再决定是否运行 `--full`。全量训练时间会明显增加，按租用 GPU 的实际进度安排。不要把试跑四类的平均值写成数据集全类平均值，也不要与论文已发表全类结果直接等同。

## 实验 2：图 6 增加一个已有方法的定位图

采用 [CFLOW-AD 官方仓库](https://github.com/gudovskiy/cflow-ad) 的公开实现作为至少一种方法对照。官方代码直接支持 MVTec AD；VisA 和 MVTec LOCO 不能假设无需适配即可覆盖。`revision_baselines/cflow_ad/` 已准备固定官方版本的独立调用脚本与说明，两张逐字节确认的 MVTec AD 输入本地干运行通过。**尚未在 GPU 上训练 CFLOW-AD，也尚未生成对照图。** 对每个方法必须使用同一张输入图与同一份 GT，并记录模型设置、权重来源和图像归一化方式。

`match_figure_sources.py` 会提取 PPT 中的嵌入图片、位置，并在有完整数据后给出相似图候选：

```bash
python -m pip install python-pptx
python match_figure_sources.py --pptx 实验图源.pptx --slides 4 6 7 --mvtec-root /root/autodl-tmp/mvtec-ad --visa-root /root/autodl-tmp/visa --loco-root /root/autodl-tmp/mvtec-loco
```

输出目录 `results_figure_matching/all_datasets/` 含 `reference_images.csv`、提取图和 `candidate_matches.csv`。旧候选排序**未考虑 PPT 嵌入图的上下翻转，不能作为来源证据**。第 6 页 SVG 形状填充中的 19 张输入图已按翻转和缩放后的像素误差完成高置信度匹配，其中 13 张异常图、6 张正常图；相应原图和异常 GT 均存在，详见 `revision_manifests/figure6_source_audit.md`。另有两张普通图片与数据集逐字节相同。PPT 原有列标题仍存在错位风险，最终面板必须按已核实的类别和样本重新标注，不能直接复用旧标题。

图 6 的可重复核验入口为 `python verify_figure6_sources.py --mvtec-root ... --visa-root ... --loco-root ...`。本地完整同类排名已通过：19/19 张均是第一名；审计中榛子图第二名 MSE 的旧笔误已更正。该检查会遍历相关类别的测试图，服务器上运行时间会长于普通文件存在性检查。

## 实验 3：图 9 补 Ours 热力图

新参数 `--save-checkpoint` 保存最终 epoch 权重；`--heatmap-targets` 指向一个 UTF-8 文本，每行填写**数据集根目录相对路径**，不能只写文件名。程序会核对它确实属于当前类别测试集，防止图片错配。每个目标输出 `input.png`、`gt.png`、`heatmap.png`、`overlay.png`、原始 `anomaly_score.npy` 和记录来源、epoch、归一化范围的 `metadata.json`。PNG 热力图采用逐图 min-max 归一化，仅用于展示；定量分数在 NPY 中。必须沿用已确认的图 9 原图，不应选“看起来更好”的别的样本替换。

本机或服务器在数据准备好后，先核查五个样本、标注和哈希值；服务器上需把三个根目录参数改成实际数据路径：

```bash
python verify_figure9_sources.py --mvtec-root /root/autodl-tmp/mvtec-ad --visa-root /root/autodl-tmp/visa --loco-root /root/autodl-tmp/mvtec-loco
```

训练时即可同时保存权重并导出该类别的热力图。例如 MVTec AD 的 screw：

```bash
python ADFlow_Start.py --dataset mvtec --class-name screw --data-path /root/autodl-tmp/mvtec-ad --distribution adaptive --maca-scales all --meta-epochs 25 --save-checkpoint --heatmap-targets revision_manifests/figure9/mvtec_screw.txt --output-dir results_figure9
```

现已确认图 9 的五张原图（文件内容与 PPT 完全相同），目标文件在 `revision_manifests/figure9/`，无需重新猜测图片编号。如需之后从已保存权重重新导出 screw，可执行：

```bash
python ADFlow_Start.py --dataset mvtec --class-name screw --data-path /root/autodl-tmp/mvtec-ad --distribution adaptive --maca-scales all --action-type norm-test --checkpoint results_figure9/checkpoints/mvtec/screw/adaptive/run_0_last.pt --heatmap-targets revision_manifests/figure9/mvtec_screw.txt --output-dir results_figure9
```

`norm-test` 会自动把输出放在 `results_figure9/inference/` 下，不会覆盖训练曲线或 MACA 权重历史。五类批量入口是 `run_figure9_heatmaps.py`，完整服务器命令与产物清单见 `SERVER_RUNBOOK_2026_10.md`。该批量入口可以在权重设置匹配时复用消融实验的 MVTec screw 与 VisA capsules 三尺度最终权重，并记录实际来源。

其他四类分别为 VisA 的 capsules、macaroni2、pcb3，以及 MVTec LOCO 的 splicing_connectors，需逐类训练、逐类导出。不能把不同类别的模型权重混用；加载时会检查类别、架构、尺度、分布等设置。图 9 是论文的典型失败案例，应忠实展示同一失败样本的结果。重新训练的热力图可能与旧稿不同，不应换图来制造相同结果。

## 数据和保存空间

- [MVTec AD 官方下载页](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)：下载并解压到 `/root/autodl-tmp/mvtec-ad/`，最终应存在 `mvtec-ad/screw/test/` 等类别目录。
- [MVTec LOCO AD 官方下载页](https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad)：解压到 `/root/autodl-tmp/mvtec-loco/`，最终应存在 `mvtec-loco/pushpins/test/` 等类别目录。
- [VisA 官方项目](https://github.com/amazon-science/spot-diff) 与[官方数据包](https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar)：解压到 `/root/autodl-tmp/visa/`，最终应存在 `visa/pcb1/Data/Images/` 和 `visa/split_csv/1cls.csv`。VisA 数据包本身的顶层是各类别，**没有额外的 `VisA/` 目录**。

先核对解压后的目录与可读图片，再考虑清理下载包，勿在验证之前删除档案。数据集不要提交到 GitHub。服务器配置沿用此前已能完成 smoke test 的单卡 4090D 24GB 即可；正式运行前还需检查剩余数据盘空间和实际类别数。

## 本轮仍需完成

1. 在 GPU 服务器上实跑 MACA 消融、图 9 Ours 热力图，以及官方 CFLOW-AD 的 MVTec AD 同图对照；本地准备不等于已经得到实验结果。
2. 图 6 新面板按已核实的输入图与 GT 重新排版、标注。若只展示 CFLOW-AD 官方支持的 MVTec AD 子集，必须明确样本与数据集范围；不能说覆盖了 VisA/LOCO 的旧图全部列。
