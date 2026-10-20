# 图 6 原图核对记录（2026-10-20）

来源：`实验图源.pptx` 第 6 页。普通图片位于
`results_figure_matching/all_datasets/`；PPT 形状的 SVG 填充中还恢复出
57 张独立位图，位于 `results_figure_matching/svg_rasters/`。

## 已确认的原图

以下两张图片与 PPT 的普通图片对象逐字节相同，并且数据集中的 GT 文件存在。

| PPT 对象 | 数据集原图（相对数据集根目录） | GT | SHA-256 |
| --- | --- | --- | --- |
| 第 6 页 shape 13、78 | `mvtec-ad/cable/test/bent_wire/001.png` | `mvtec-ad/cable/ground_truth/bent_wire/001_mask.png` | `6f906f05817d54a62532add0da5f56314a5508babd5849d5a3fa571001ad1a88` |
| 第 6 页 shape 16、17、63 | `mvtec-ad/transistor/test/cut_lead/005.png` | `mvtec-ad/transistor/ground_truth/cut_lead/005_mask.png` | `37d6d29ececc6e1c497d3e67a625c318a2024a0a865aa29df44028d528f35395` |

## SVG 内嵌图的像素级复核

第 6 页 SVG 中的 171×171 输入图相对数据集原图**上下翻转**，且已经缩放，部分还画有红色异常轮廓。因此它们不可能与原图字节哈希相同；此前未考虑翻转的视觉候选排名并不可靠。复核时将每张候选原图以 Lanczos 缩放到 171×171，再上下翻转；比较 RGB 像素均方误差（MSE），并排除明显红色标注像素（`R > 1.3G`、`R > 1.3B`、`R > 80`）。下表是各自所属类别中得分最低的候选；括号内是第二名 MSE。所有第一名与第二名都有明显间隔。MSE 单位为 8-bit RGB 像素值的平方，数值受缩放插值、JPEG 编码及标注影响，不要求为 0。

| SVG 输入图 | 原图（相对 `datasets/`） | 状态 | MSE（第二名） | GT（相对 `datasets/`） |
| --- | --- | --- | ---: | --- |
| `rId4_01` | `visa/pcb2/Data/Images/Normal/0690.JPG` | 正常 | 17.0 (339.0) | 无异常 GT |
| `rId5_01` | `mvtec-ad/metal_nut/test/good/017.png` | 正常 | 3.6 (395.6) | 无异常 GT |
| `rId6_01` | `mvtec-ad/metal_nut/test/color/000.png` | 异常 | 3.7 (264.7) | `mvtec-ad/metal_nut/ground_truth/color/000_mask.png` |
| `rId7_01` | `visa/pcb2/Data/Images/Anomaly/013.JPG` | 异常 | 18.0 (309.9) | `visa/pcb2/Data/Masks/Anomaly/013.png` |
| `rId8_01` | `mvtec-ad/cable/test/good/046.png` | 正常 | 6.5 (771.4) | 无异常 GT |
| `rId10_01` | `mvtec-ad/transistor/test/good/050.png` | 正常 | 3.8 (444.9) | 无异常 GT |
| `rId13_01` | `mvtec-loco/pushpins/test/logical_anomalies/071.png` | 异常 | 13.2 (996.6) | `mvtec-loco/pushpins/ground_truth/logical_anomalies/071/000.png` |
| `rId14_01` | `mvtec-loco/pushpins/test/good/131.png` | 正常 | 11.6 (937.6) | 无异常 GT |
| `rId15_01` | `mvtec-loco/screw_bag/test/good/046.png` | 正常 | 9.0 (710.2) | 无异常 GT |
| `rId16_01` | `mvtec-ad/hazelnut/test/crack/000.png` | 异常 | 1.5 (132.4) | `mvtec-ad/hazelnut/ground_truth/crack/000_mask.png` |
| `rId17_01` | `mvtec-ad/metal_nut/test/bent/005.png` | 异常 | 4.8 (250.1) | `mvtec-ad/metal_nut/ground_truth/bent/005_mask.png` |
| `rId18_01` | `mvtec-ad/transistor/test/damaged_case/009.png` | 异常 | 5.5 (299.8) | `mvtec-ad/transistor/ground_truth/damaged_case/009_mask.png` |
| `rId19_01` | `mvtec-ad/cable/test/missing_cable/000.png` | 异常 | 9.0 (1658.3) | `mvtec-ad/cable/ground_truth/missing_cable/000_mask.png` |
| `rId20_01` | `visa/cashew/Data/Images/Anomaly/019.JPG` | 异常 | 11.6 (344.8) | `visa/cashew/Data/Masks/Anomaly/019.png` |
| `rId21_01` | `visa/fryum/Data/Images/Anomaly/074.JPG` | 异常 | 6.3 (576.6) | `visa/fryum/Data/Masks/Anomaly/074.png` |
| `rId22_01` | `visa/pcb4/Data/Images/Anomaly/011.JPG` | 异常 | 13.0 (308.9) | `visa/pcb4/Data/Masks/Anomaly/011.png` |
| `rId28_01` | `mvtec-loco/screw_bag/test/logical_anomalies/105.png` | 异常 | 11.2 (1013.9) | `mvtec-loco/screw_bag/ground_truth/logical_anomalies/105/000.png` |
| `rId33_01` | `mvtec-ad/transistor/test/cut_lead/005.png` | 异常 | 4.1 (264.4) | `mvtec-ad/transistor/ground_truth/cut_lead/005_mask.png` |
| `rId36_01` | `mvtec-ad/cable/test/bent_wire/001.png` | 异常 | 5.6 (831.4) | `mvtec-ad/cable/ground_truth/bent_wire/001_mask.png` |

证据等级：上表为**高置信度像素级身份匹配**，并非 PNG/JPEG 字节一致；其中 `rId33_01`、`rId36_01` 还分别与上方两张已作 SHA-256 确认的普通图片指向同一原图。共核对 19 张 SVG 输入图，含 13 张异常、6 张正常。异常图及其 GT 可作为“同一输入、同一 GT”的方法热力图对照候选；正常图可用于正常样本展示，但不宜充当异常定位的主体。

复核更正：`rId16_01` 的第二名此前误记为 120.0。按同一类别全部测试图重新排名，第二名是 `hazelnut/test/hole/008.png`，MSE 为 132.371，表中按一位小数记为 132.4；第一名 `hazelnut/test/crack/000.png` 和 GT 不变。可用 `python verify_figure6_sources.py` 重算全部 19 张图的前两名。

## 尚不能直接使用的部分

第 6 页不是一组简单的普通图片：多列输入图存在于 SVG 形状填充中。
`extract_ppt_svg_rasters.py` 已恢复这些位图。旧的 `rank_ppt_svg_sources.py`
候选排名未考虑上下翻转，因此不能当作来源证据；以上像素级复核结果应优先采用。

PPT 上方的文字与下方图片也有明显表面不一致：例如标作
“Transistor”“Cable”的前两列，恢复出的输入位图外观看起来都是金属螺母；
标作“Metalnut”的列则是电路板。可能是 PPT 的拼贴布局或标签尚未定稿。
即使输入图来源已经确定，也不能按这些错位标题替它们指定论文列名。
制作对照图前，应以相同原图和 GT 重建一个标签与样本一致的新面板；
不能把当前 PPT 的列顺序直接当作最终排版。

下一步应确定最终图 6 保留哪些列、对照方法及其数据集支持范围；
针对每列使用同一张输入图、同一份 GT 生成对照热力图。
