# 数据说明

## 目录结构

| 目录 | 说明 |
| --- | --- |
| `gen` | 采用 `generator/ts_argos_dp` 生成的时序数据图像 |
| `labeled` | 线上采集的业务绘制的时序图像 |
| `jsons` | 各数据集的标签文件，详见 [jsons](#jsons) |
| `prompts` | 线上使用的 prompt 文件，详见 [prompts](#prompts) |

## jsons

| 文件 | 说明 | 总条数 |
| --- | --- | --- |
| `argos-dp-gens.json` | 生成时序数据图像的标签 | 1535 |
| `argos-dp-trains.json` | 人工标注的训练数据标签 | 1750 |
| `argos-dp-valids.json` | 验证数据标签 | 763 |
| `seed-sft.json` | 基于 `argos-dp-trains.json` 改造的 SFT 数据 | 1751 |
| `argos-trains.json` | 最终训练集：`argos-dp-gens.json` 与 `argos-dp-valids.json` 合并打乱后，除抽取 800 条验证集外的剩余部分（1498）再合并 `argos-dp-trains.json`（1750），由 `preprocess.py` 的 `merge_labeled_gen` 生成 | 3248 |
| `argos-valids.json` | 最终验证集：从 `argos-dp-gens.json` 与 `argos-dp-valids.json` 合并打乱后随机抽取的 800 条，由 `preprocess.py` 的 `merge_labeled_gen` 生成 | 800 |

走势标签分布：

| 文件 | normal | offset | increase | decline | spike | fluctuate |
| --- | --- | --- | --- | --- | --- | --- |
| `argos-dp-gens.json` | 256 | 256 | 256 | 255 | 256 | 256 |
| `argos-dp-trains.json` | 221 | 446 | 522 | 126 | 435 | - |
| `argos-dp-valids.json` | 277 | 150 | 171 | 11 | 74 | 80 |
| `argos-trains.json` | 583 | 698 | 809 | 290 | 643 | 225 |
| `argos-valids.json` | 171 | 154 | 140 | 102 | 122 | 111 |

恢复标签分布：

| 文件 | recovered | recovering | no_recover |
| --- | --- | --- | --- |
| `argos-dp-gens.json` | 256 | 315 | 964 |
| `argos-dp-trains.json` | 527 | 341 | 882 |
| `argos-dp-valids.json` | 344 | 41 | 378 |
| `argos-trains.json` | 927 | 567 | 1754 |
| `argos-valids.json` | 200 | 130 | 470 |

## prompts

| 文件 | 说明 |
| --- | --- |
| `prompt.json` | argos 智能盯盘线上使用的 prompt |
