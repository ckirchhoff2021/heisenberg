# heisenberg-trainer

AIOps 智能运维领域的时序曲线工具集，涵盖时序图像的**生成、预处理标注、网页预览**，以及基于 Qwen3.5 的**多模态模型训练**（GRPO / OPD / JEV）。

## 目录结构

```
heisenberg-trainer/
├── pyproject.toml            # 项目元数据与依赖声明（uv 管理）
├── dataset/                  # 训练数据构造（grpo.py / jev.py）
├── models/                   # 自定义模型（jev.py）
├── train/                    # 训练入口（grpo.py / grpo_opd.py / jev.py）
├── scripts/                  # 训练启动脚本
├── datas/
│   └── aiops/
│       ├── generator/        # 时序数据与图像生成
│       ├── preprocess.py     # 数据预处理与标签整理
│       └── files/
│           ├── gen/          # 生成的时序图像
│           ├── labeled/      # 线上采集的业务图像
│           ├── jsons/        # 各数据集标签文件
│           ├── prompts/      # 线上 prompt
│           └── viewer/       # 数据预览网站
```

## 环境要求

- Python >= 3.10
- [uv](https://docs.astral.sh/uv/)（用于管理 `.venv` 与依赖）

依赖在 [pyproject.toml](pyproject.toml) 中声明：

| 依赖 | 用途 |
| --- | --- |
| numpy | 数据预处理与时序计算 |
| matplotlib | 时序图像生成 |
| tqdm | 生成过程进度条 |

## 快速开始

在项目根目录下创建虚拟环境并安装依赖：

```bash
uv sync          # 自动创建 .venv 并按 pyproject.toml 安装依赖
```

### 启动数据预览网站

预览服务仅使用 Python 标准库，通过 `.venv` 启动：

```bash
uv run python datas/aiops/files/viewer/server.py
```

启动后访问 `http://127.0.0.1:8000`，可在页面切换数据集（`argos-trains.json`、`argos-valids.json` 等）并按 `curve`、`recover` 筛选。

## JEV 模型

JEV（Joint Evaluation / 判别式选择模型）把"曲线分类"从**生成式回答**改造成**对候选选项的平行判别**：模型不逐 token 生成答案，而是给一道选择题的每个候选各打一个标量分数，softmax 后得到各类别置信度。

### 设计思想

按 `STATE → QUESTION → CANDIDATE` 的层级组织输入。每一个 `(状态, 问题, 候选)` 三元组各自拼成一条**多模态因果序列**，用 Qwen3.5 骨干（视觉塔 + 语言层，无 LM 头）编码：

```
[STATE] <分类规则 system> <曲线图像 vision 块>
[QUESTION] <问题>
[CANDIDATE] <候选内容>
```

- 拼进序列的是候选**内容文本**（如 `normal`、`offset`），不是索引——模型需要基于候选语义来判断；索引只用于寻址与监督。
- 骨干读取每条路径 `[CANDIDATE]` **末位 token 的隐状态**作为该候选的向量（此时它已看全状态、问题与候选自身，但还没看到其它候选）。
- 一个**无位置编码的候选集合编码器**让同题候选横向比较，保证打分与选项排列顺序无关。

### 网络结构

```
候选末位隐状态 cand_vecs [Bq, Cmax, H]      # H = 1024
  → proj_in        : Linear H → 256
  → set_encoder    : 2 层 Transformer（无位置编码，padding mask 屏蔽无效位）
  → proj_out       : Linear 256 → H
  → 残差 rep       : cand_vecs + proj_out(enc)
  → scorer         : RMSNorm → Linear H→256 → SiLU → Linear 256→1
  → logits [Bq, Cmax]                          # 每个候选 1 个标量，无效位 = -1e4
```

关键组件：

| 组件 | 作用 |
| --- | --- |
| `PathEncoder` | Qwen3.5 多模态骨干前向，按 `cand_end_pos` 取各候选末位隐状态，并由 `q_index/cand_index` 散射回 `[题号, 候选号]` 二维结构 |
| `CandidateSetEncoder` | 无位置编码的 2 层 Transformer，在同题候选之间做 self-attention，实现**置换等变**（打乱候选顺序，输出只跟着同样打乱，概率不变） |
| `ScalarScorer` | 把每个候选的 H 维表示压成 **1 个标量 logit**，与候选文本长度无关 |

### 路径是如何分组的

collate 按"题目优先"摊平路径：`题0的所有候选 → 题1的所有候选 → …`，同题路径连续排列；同时每条路径携带 `q_index`（所属题号）与 `cand_index`（题内序号）。`cand_mask [Bq,Cmax]` 标记每题有效候选（不足 `Cmax` 的补空位为 `False`）。

> argos 每张图产出 **trend、recover 两道独立的题**（7 候选 / 4 候选），即使共用同一张图也分别成组、各自 softmax。

### 路径示例

以一张曲线图的 trend 题、候选 `normal` 为例（图像降采样后为 55 个 vision token）：

```
[STATE] # 角色
你是一个 AIOps 智能运维领域的时序曲线图理解专家……
（# 输入约束 / # 术语 / # 输出目标 / # trend 定义 / # recover 定义 / # 强制判断流程）
<|vision_start|><|image_pad|>×55<|vision_end|>
[QUESTION] 对比参考曲线，当前曲线的趋势（trend）属于哪一类？
[CANDIDATE] normal
```

同题 7 条路径的前缀逐字相同，仅末尾候选词不同，例如 `...[CANDIDATE] offset`、`...[CANDIDATE] other`。

### 置信度与损失函数

**置信度**：对每题的有效候选做 softmax（padding 位先用 `-1e4` 排除）：

$$
p_k = \frac{e^{z_k}}{\sum_{j\in valid} e^{z_j}}
$$

`p_k` 即"属于第 k 类"的置信度，和为 1；预测类别取 `argmax p`。

**损失**：交叉熵（逼高正确类置信度）+ Brier 校准项（默认权重 0.1），另含默认关闭的置换一致性 KL：

$$
L = \underbrace{-\sum_k t_k\log p_k}_{\text{soft 交叉熵}}
  + 0.1\underbrace{\sum_k (p_k - t_k)^2}_{\text{Brier}}
  + w_{perm}\,\underbrace{KL(p\,\|\,p')}_{\text{置换 KL}}
$$

其中 `t` 为正确答案的 one-hot，`p'` 为候选顺序随机打乱后重新打分的分布。训练用两参组 AdamW（backbone `1e-5` / head `1e-4`）、warmup + cosine、梯度裁剪 1.0、bf16，按 **dev 交叉熵**选优保存 `best.pt`。

### 数据构造与训练

输入 json 支持两种：标准 JEV 选择题，或 argos 标签数据（`curve/recover/image/from`）。后者会自动按每张图出 trend/recover 两题，并把 [argos-dp-prompt.md](datas/aiops/files/prompts/argos-dp-prompt.md) 的**分类规则部分**（去掉生成式专用的"输出格式"与 few-shot）作为 system 注入。

```bash
# 从 argos 标签数据直接训练
python train/jev.py \
    --backbone checkpoints/Qwen3.5-0.8B \
    --train-file datas/aiops/files/jsons/argos-trains.json \
    --dev-file   datas/aiops/files/jsons/argos-valids.json \
    --output-dir runs/jev

# 小规模调试：每个文件只取前 N 条、降低图像分辨率
python train/jev.py \
    --backbone checkpoints/Qwen3.5-0.8B \
    --train-file datas/aiops/files/jsons/argos-trains.json \
    --limit 16 --max-pixels 262144 --max-steps 10
```

相关实现：

| 文件 | 内容 |
| --- | --- |
| [models/jev.py](models/jev.py) | `RMSNorm` / `CandidateSetEncoder` / `ScalarScorer` / `PathEncoder` / `JevModel` |
| [dataset/jev.py](dataset/jev.py) | `JEVDatasets`、`JevTorchDataset`、argos → 选择题转换与 collate |
| [train/jev.py](train/jev.py) | 两参组训练循环、tensorboard、dev 选优 |

> 说明：v1 对每个候选路径重算共享前缀与视觉编码（数值正确但有冗余），后续可通过共享前缀 / 视觉特征缓存优化。

更多说明见：

- 数据与标签说明：[datas/aiops/files/README.md](datas/aiops/files/README.md)
- 预览服务详细用法：[datas/aiops/files/viewer/README.md](datas/aiops/files/viewer/README.md)

## GRPO 训练（曲线分类）

用 Qwen3.5-0.8B 做「时序曲线 trend / recover 分类」的生成式 RL 训练。与 JEV 的判别式思路互补：这里让模型直接按 prompt 里的强制判断流程，输出结构化标签（`<reason>` / `<trend>` / `<recover>`），通过奖励函数驱动模型学会分类。

相关实现：

| 文件 | 内容 |
| --- | --- |
| [train/grpo.py](train/grpo.py) | GRPO 训练入口 + 全部 reward 函数 |
| [dataset/grpo.py](dataset/grpo.py) | `TRLDatasets`，argos 标签 → 多模态对话数据 |
| [scripts/train_grpo.sh](scripts/train_grpo.sh) | 训练启动脚本（vLLM server 模式） |

### 任务定义

- **trend**（7 类）：`normal` / `offset` / `increase` / `decline` / `spike` / `fluctuate` / `other`
- **recover**（4 类）：`recovered` / `recovering` / `no_recover` / `other`

标签分布**严重不均衡**（3248 样本）：trend 从 `fluctuate` 225 到 `increase` 809（3.6×），recover 从 `recovering` 567 到 `no_recover` 1754（3.1×）。ground truth 中 **`other` 出现次数为 0**（它是 prompt 定义的兜底标签，真实标注从不使用）。

### Reward 设计

奖励由 4 个函数相加（`train/grpo.py` 的 `reward_funcs` 列表），核心目标：**保证 GRPO 组内 reward 几乎总有差异，让 advantage 非零、梯度持续流动**。

| 函数 | 权重/幅度 | 说明 |
| --- | --- | --- |
| `format_reward_func` | +0.2 | 正确输出 `<reason>/<trend>/<recover>` 三段结构 |
| `curve_accuracy_reward` | +1.0 / -0.2 / -0.5 | trend 标签三级梯度（主任务） |
| `recover_accuracy_reward` | +0.5 / -0.2 / -0.5 | recover 标签三级梯度（次任务） |
| `consistency_reward_func` | -0.3 | trend/recover 逻辑一致性强约束 |

**三级梯度**（`_tag_accuracy_reward` 的核心）：

```
命中真值              → +weight
输出具体但错误的标签  → -0.2   (WRONG_LABEL_PENALTY)
输出 other / 空 / 非法 → -0.5   (OTHER_PENALTY)
```

设计动机（对应训练中实际观察到的病态现象）：

1. **「错误标签」不能给 0 分**。GRPO 靠组内 reward 的**相对差异**算 advantage（组内标准化）。若「答具体标签但错」给 0 分，则当一个 group 的 4 条 completion 全错时，组内 reward 全 0 → `std=0` → advantage 全 0 → 这一 group 的梯度完全消失。早期日志里 `frac_reward_zero_std` 高达 0.4（40% 的 group 零方差白训）正是这个原因。改成 -0.2 后，只要组内有一两条蒙对，reward 就拉开差距。
2. **`other` 要给最重的惩罚**。ground truth 从不标注 `other`，模型一旦「遇到不确定就输出 other」即可规避分类风险。用 -0.5 压住这种投机行为，逼模型给出具体判断。

**逻辑一致性强约束**（`consistency_reward_func`）：

与 ground truth 无关，只检查模型输出内部自洽性。当前唯一硬约束是数据 100% 验证的确定关系：

```
trend=normal ⟹ recover=recovered   （583/583 样本无一例外）
```

若模型输出 `trend=normal` 但 `recover≠recovered`，逻辑上必然自相矛盾，额外罚 -0.3。这驱动模型学会标签配对，而不是把两个标签当独立变量。注意：**提取不到标签时不触发**（交给 tag 精度 reward 去罚），避免双重扣分。

### 已知的优化方向（尚未实现）

- **类别逆频率加权**：fluctuate / decline / recovering 等少数类被多数类淹没，可对少数类命中给更高权重（如 fluctuate 3.6×、recovering 3.09×）。当前未实现，因为直接调 weight 会改变 reward 量纲、需重新调 `beta` 与 `OTHER_PENALTY` 的平衡，留待验证主流程收敛后再做。
- **reason 质量信号**：模型花了 ~335 token 写 `<reason>`，目前完全不打分。若后续发现「标签对但 reason 瞎编」，可加轻量的 reason 长度 / 关键特征词奖励。
- **更完整的 label-pair 约束**：`recover=other ⟹ trend=other`（prompt 定义）也是一条硬约束，但数据中无 `other` 样本，暂未加入 consistency。

### 训练启动

```bash
# vLLM server 模式（加速生成，~7×）
bash scripts/train_grpo.sh
```

关键参数：`--per-device-batch-size 4 --num-generations 4`（16 completions/step）、`lr=5e-6`、`--vllm-mode server --vllm-server-base-url http://localhost:8010`。

> 训练镜像 `vllm027-trl-train:v0.0.2` 已内置 Qwen3.5 GatedDeltaNet 所需的两个 kernel（`fla` + `causal-conv1d`），构建脚本见 [scripts/build_vllm027_trl_image.sh](scripts/build_vllm027_trl_image.sh)。
