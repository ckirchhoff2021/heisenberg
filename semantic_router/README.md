# semantic_router 评测汇总

> 目标：评估「复用 vLLM Semantic Router 推理模块」这条路径的可行性。
> 方法：用 d3-mini（Decision 3.0，Qwen3.5-4B backbone + 255 路 code-readout）在
> jev-mixed 验证集上做 zero-shot 评测，与 heisenberg 自己训练的模型做横向对比。

---

## 一、核心结论（TL;DR）

1. **推理模块复用可行**：d3-mini 本地 transformers 5.19 即可加载（8.6s），
   `system_one` 接口标准，vLLM Semantic Router 原生支持（151 req/s，README 实测）。
2. **zero-shot 精度不足**：d3-mini 零样本总 acc **0.5677**，比训练后的 jev_4b_mixed
   （0.8442）低 27.7 个点。差距源于领域规则知识，不是推理模块。
3. **正确路径**：复用 vLLM Semantic Router 的 serving 吞吐，配自己的 argos 训练权重，
   而非直接用 d3-mini 零样本分类。

---

## 二、模型对比总览（jev-mixed 验证集 2600 条）

| 模型 | 训练方式 | 总 acc | legacy.trend | legacy.recover | noise_argos.trend | noise_kepler.trend |
|---|---|---|---|---|---|---|
| **d3-mini-finetuned** | d3-mini 微调（readout + LoRA，冻结 vision） | **0.8481** | 0.9075 | 0.7688 | 0.8770 | 0.8436 |
| **jev_4b_mixed** | 4B LoRA + PointerHead，4任务混合 | **0.8442** | 0.8962 | 0.7750 | 0.8803 | 0.8231 |
| **argos-only** | 4B LoRA + PointerHead，仅 legacy | 0.7292 | 0.9137 | 0.7550 | 0.5098 | 0.6410 |
| **d3-mini** | 零样本（未训 argos） | 0.5677 | 0.4950 | 0.5775 | 0.6443 | 0.5769 |

### 关键观察

- **d3-mini 微调是强基线**：用 d3-mini 自己的 readout + 英文 prompt 框架微调
  readout 头 + LoRA backbone（冻结 vision），总 acc 从 zero-shot 的 0.5677 提升到
  **0.8481**（+28 点），甚至略超 jev_4b_mixed（0.8442）。
- **混合训练是净收益**：jev_4b_mixed 在源任务（legacy）不掉点（甚至 recover 略赢），
  在跨任务（noise）大幅领先 argos-only 30 个点。
- **d3-mini 零样本已超随机**：4 任务选项数 7/4/6/4（随机 0.19~0.25），d3 到 0.5677，
  证明其通用视觉决策能力真实有效。
- **recovering 是共同盲区**：d3-mini 微调后 recovering 类仍只有 0.077（130 个对 10 个），
  与 jev_4b_mixed（recovering F1 0.268）同属「恢复中」这个精细时序判断的难点。

---

## 三、各模型 per-task 准确率详情

### 3.1 jev_4b_mixed（混合训练，best_step=500）

| 任务 | acc | mean_conf | 最短板 |
|---|---|---|---|
| legacy_argos.trend | 0.8962 | 0.9403 | spike F1 0.833 |
| legacy_argos.recover | 0.7750 | 0.8355 | **recovering F1 0.268** |
| noise_argos.trend | 0.8803 | 0.9110 | noise F1 0.843 |
| noise_kepler.trend | 0.8231 | 0.9034 | same_trend F1 0.813 |

### 3.2 argos-only（单任务训练，best_step=200，之后过拟合）

| 任务 | acc | 备注 |
|---|---|---|
| legacy_argos.trend | 0.9137 | 源任务略赢 mixed（+1.8pp） |
| legacy_argos.recover | 0.7550 | 略输 mixed（-2.0pp） |
| noise_argos.trend | 0.5098 | **same_trend F1 0.144 崩塌** |
| noise_kepler.trend | 0.6410 | **same_trend F1 0.335 崩塌** |

> 崩塌根因：argos-only 训练集无 `noise` 类，零样本跨任务时把「有噪声的平稳」
> 强行套成 legacy 语义（up/down），导致 same_trend 系统性误判。

### 3.3 d3-mini（zero-shot）

| 任务 | acc | 关键盲区 |
|---|---|---|
| legacy_argos.trend | 0.4950 | spike 0.164 / offset 0.338 / normal 0.427 |
| legacy_argos.recover | 0.5775 | **recovering 0.031（130 个只对 4 个）** |
| noise_argos.trend | 0.6443 | **noise 0.916 / spike 0.945 反而很强** |
| noise_kepler.trend | 0.5769 | same_trend 0.255 |

---

## 四、实验配置（口径对齐说明）

### 4.1 数据

- 训练/验证集：`datas/aiops/noise/jev-mixed.json`
  - train 11755 条，eval 2600 条
  - 4 任务：legacy_argos.trend(7类) / recover(4类) / noise_argos.trend(6类) / noise_kepler.trend(4类)
- 图像路径：`schema.data_roots[source] + state.images[0]`

### 4.2 三个模型评测口径

| 项 | jev_4b_mixed / argos-only | d3-mini |
|---|---|---|
| state | 中文分类规则全文 | 同左 |
| images | 曲线图绝对路径 | 同左 |
| question | PointerHead 判别式（options + instr） | d3 choice（criteria 选项名做 key） |
| 预测 | PointerHead 打分 argmax | d3 readout 打分 argmax |
| prompt 框架 | 中文规则 + 中文问题 | **d3 英文 system prompt + State/Question/Options** |

> ⚠️ d3-mini 是「换 prompt 框架」的 zero-shot：它不认 jev 的中文规则格式，
> 测的是通用视觉决策能力，与 jev 训练口径不直接可比。

### 4.3 训练配置（jev_4b_mixed / argos-only 一致）

```
backbone = Qwen3.5-4B（4573M 参数）
LoRA r=16 + PointerHead，冻结 vision tower
batch 1 × world 4 × accum 8 = 32 record/step
lr 1e-4 / head-lr 1e-4 / max-steps 500 / eval-every 100
max-state 3072 / max-pixels 262144
```

- jev_4b_mixed：500 步 ≈ 1.36 epoch（11755 条），best_step=500，无过拟合
- argos-only：500 步 ≈ 4.9 epoch（3248 条），best_step=200，之后过拟合

---

## 五、产物清单

### 5.1 训练 checkpoint

| 目录 | 内容 |
|---|---|
| `outputs/jev_4b_mixed/` | 混合训练 LoRA adapter(124M) + head.pt(5M)，best_step=500 |
| `outputs/jev_4b_argos_only/` | 单任务训练，best_step=200 |
| `outputs/jev_4b_mixed_vision/` | vision 解冻版（已证伪 acc 0.334） |

### 5.2 评测结果

| 文件 | 内容 |
|---|---|
| `semantic_router/results/d3_mini_zero_shot_eval.json` | d3-mini zero-shot（本次新增） |
| `semantic_router/results/per_task_eval_jev_4b_mixed.json` | 混合模型 per-task |
| `semantic_router/results/noise_eval_argos_only_4b.json` | argos-only noise 跨任务 |
| `semantic_router/results/legacy_eval_argos_only_4b.json` | argos-only legacy 源任务 |

### 5.3 评测脚本

| 文件 | 用途 |
|---|---|
| `semantic_router/eval_d3_mini_zero_shot.py` | d3-mini zero-shot 评测（持久化版，支持 --limit/--device） |
| `semantic_router/eval_d3_mini_finetuned.py` | d3-mini 微调后评测（--ckpt 指定 checkpoint） |

### 5.4 d3-mini 微调训练脚本

| 文件 | 用途 |
|---|---|
| `semantic_router/train_d3_mini.py` | d3-mini 微调入口（readout + LoRA，冻结 vision，4卡 DDP） |
| `semantic_router/train_d3_mini_4gpu.sh` | 启动脚本（torchrun 4卡 + tee 日志落盘 + NCCL 超时兜底） |

训练产物：`outputs/d3_mini_finetuned/`（LoRA adapter 129MB + readout.safetensors 2.6MB + finetune_meta.pt + train.log）

---

## 六、d3-mini 微调实验（2025-11）

### 6.1 训练配置

```
backbone = d3-mini 的 Qwen3.5-4B（完整 9.1GB 权重 + 255路 readout）
训练对象 = readout 头(255×2560) + LoRA r=16（冻结 vision）
前向 = d3 noncausal_full_attention（PromptEncoder，训练模式可反传）
batch 1 × world 4 × accum 8 = 32 record/step
lr 1e-4 / readout-lr 1e-4 / max-steps 500 / eval-every 100
可训练参数 33.1M（LoRA 32.5M + readout 0.65M）
```

### 6.2 训练曲线（dev 子集 256 条）

| step | loss | dev_ce | dev_acc |
|---|---|---|---|
| initial | — | 1.2044 | 0.5352 |
| 100 | 0.8477 | 0.5164 | 0.8203 |
| 200 | 0.1542 | 0.5282 | 0.8125 |
| 300 | 0.6700 | **0.4488** | 0.8164 |
| 400 | 0.2743 | 0.4629 | 0.8125 |
| 500 | 0.4946 | 0.4659 | 0.8203 |

> best_step=300（按 dev_ce），step 400/500 ce 轻微回升（0.4629/0.4659），
> 有轻微过拟合；acc 稳定在 0.81~0.82。

### 6.3 结论

- d3-mini 微调后总 acc 从 zero-shot 的 **0.5677 提升到 0.8481**（+28 点），
  证明 readout + LoRA 微调路径有效，且略超 jev_4b_mixed（0.8442）。
- per-task：legacy.trend 0.9075 / recover 0.7688 / noise_argos 0.8770 /
  noise_kepler 0.8436。恢复类 recovering 仍是短板（0.077）。
- 完整验证集结果见 `results/d3_mini_finetuned_eval.json`。

---

## 七、下一步建议

1. **验证 vLLM Semantic Router 加载自己的 checkpoint**：jev_4b_mixed 是 LoRA + PointerHead
   增量，需确认能否 merge 成完整权重后喂给 vLLM Semantic Router Engine mode。
2. **d3-mini 微调产物部署**：`outputs/d3_mini_finetuned/` 的 LoRA + readout 可直接
   merge 回 d3-mini backbone 或作为 adapter 挂载，复用以 vLLM 部署链路。
3. **补 epoch-matched 对照**：argos-only 与 mixed 的 epoch 数不对称（4.9 vs 1.36），
   若要做严谨消融需补一个 ~140 步的 argos-only 对照。
