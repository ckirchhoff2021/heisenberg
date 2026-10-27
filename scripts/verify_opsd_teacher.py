"""验证 OPSD 教师轨迹预测的图像相关性。

核心假设：教师（base 冻结 + gold 标签注入）对轨迹 token 的 logits
几乎不依赖图像（因为"照着 gold 抄"不需要视觉），因此 reverse KL 的
监督信号不含「图→标签」判别信息，学生只能赌先验坍缩。

实验：固定 gold 和 traj（traj == gold，学生采对的理想情况），只换图像
（4 张不同 trend 的图），比较教师对 traj 部分 logits 的两两差异。
若差异 ~0 => 教师预测图无关 => 假设成立。
"""
import json
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

from train.opsd import ModelHolder, encode_image, gold_text

BACKBONE = "checkpoints/Qwen3.5-0.8B"
PROMPT = "datas/aiops/files/prompts/argos-dp-prompt-noreason.md"
VALIDS = "datas/aiops/files/jsons/argos-valids.json"
FILES_ROOT = "datas/aiops/files"

device = "cuda"
print("加载 base 模型（教师）...")
holder = ModelHolder(BACKBONE, device, grad_checkpointing=False, load_teacher=True)
teacher = holder.teacher
teacher.eval()

with open(PROMPT, encoding="utf-8") as f:
    prompt_text = f.read().strip()

recs = json.load(open(VALIDS, encoding="utf-8"))
# 找 4 张不同 trend 的图（recover 尽量也不同，但 focus trend）
targets = ["normal", "spike", "increase", "decline"]
picked = []
seen = set()
for r in recs:
    c = r["curve"]
    if c in targets and c not in seen:
        picked.append(r)
        seen.add(c)
    if len(picked) == len(targets):
        break
print("选中图:", [(r["curve"], r["recover"]) for r in picked])

# 固定 gold / traj（traj == gold 理想情况）
gold = gold_text("normal", "recovered")
gold_ids = holder.tok.encode(gold, add_special_tokens=False)
traj_ids = holder.tok.encode(gold, add_special_tokens=False)
Lg = len(gold_ids)
Lt = len(traj_ids)
# 内容 token 的位置（类别词，非格式 token）
# traj = <trend>normal</trend>\n<recover>recovered</recover>
# 找 'normal', 're', 'covered' 等词的 token 位置
traj_toks = holder.tok.convert_ids_to_tokens(traj_ids)
print("\ntraj token 结构:")
for i, tk in enumerate(traj_toks):
    print(f"  pos{i}: {tk!r}")

logits_list = []
for r in picked:
    img_path = os.path.join(FILES_ROOT, r["from"], r["image"])
    pv, gthw, n_pad = encode_image(img_path, holder.ip, 512 * 512)
    prompt_ids = holder._prompt_ids(prompt_text, holder._img_block(n_pad))
    Lp = len(prompt_ids)
    teacher_ids = torch.tensor([prompt_ids + gold_ids + traj_ids], device=device)
    with torch.no_grad():
        t_logits = holder.forward_logits(teacher, teacher_ids, pv.to(device), gthw.to(device))
    teacher_traj = t_logits[0, Lp + Lg - 1: Lp + Lg - 1 + Lt]  # [Lt, V]
    logits_list.append(teacher_traj.cpu())

print("\n===== 教师轨迹 logits 两两差异（mean abs diff）=====")
names = [r["curve"] for r in picked]
for i in range(len(logits_list)):
    for j in range(i + 1, len(logits_list)):
        diff = (logits_list[i] - logits_list[j]).abs().mean().item()
        print(f"  {names[i]:10s} vs {names[j]:10s}: {diff:.6f}")

# 内容 token（类别词）的 top1 一致性
print("\n===== 教师对内容 token 的 top1 预测（若抄答案则完全一致）=====")
content_pos = [i for i, tk in enumerate(traj_toks) if tk.strip("Ġ") in ("normal", "re", "covered", "sp", "ike", "decl", "ine", "no", "_re", "cover")]
print("内容 token 位置:", content_pos)
for pos in content_pos:
    tops = []
    for l in logits_list:
        top1 = int(l[pos].argmax().item())
        tops.append(holder.tok.decode([top1]))
    print(f"  pos{pos} ({traj_toks[pos]!r}): top1 -> {tops}")

# 关键：如果图像相关，同一 pos 的 logits 应该随图变化；如果"抄答案"，则不变
print("\n===== 结论判断 =====")
# 用 logits 的总变化量 vs 单个 logit 的典型 scale 对比
all_diffs = []
for i in range(len(logits_list)):
    for j in range(i + 1, len(logits_list)):
        all_diffs.append((logits_list[i] - logits_list[j]).abs().mean().item())
avg_diff = sum(all_diffs) / len(all_diffs)
# logits 自身的平均幅度
avg_mag = sum(l.abs().mean().item() for l in logits_list) / len(logits_list)
print(f"  平均图间 logits 差异: {avg_diff:.6f}")
print(f"  logits 平均幅度:     {avg_mag:.6f}")
print(f"  差异/幅度 比值:      {avg_diff/avg_mag:.4%}")
if avg_diff / avg_mag < 0.05:
    print("  => 教师轨迹预测基本图无关（比值 <5%），reverse KL 无判别信号")
else:
    print("  => 教师轨迹预测图相关，假设不成立，需另找原因")
