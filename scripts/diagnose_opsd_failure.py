"""诊断 OPSD 失败根因：检查训练采样出的学生轨迹 vs 教师复述轨迹 vs 贪心解码。

三个关键问题：
1. 学生 on-policy 采样出的 traj 长什么样？（是标签还是垃圾？）
2. 教师"复述"的 gold 轨迹（在 traj 位置上）到底分给正确答案多少概率？
3. reverse KL 的 mode-seeking 是否把学生逼到了错误的 mode？

用 v2 训练产物（如果 best.pt 存在则加载，否则用 base 学生+修复教师做前向诊断）。
"""
import json
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

from train.opsd import ModelHolder, encode_image, gold_text, gold_text_annotated, parse_tags

BACKBONE = "checkpoints/Qwen3.5-0.8B"
PROMPT = "datas/aiops/files/prompts/argos-dp-prompt-noreason.md"
VALIDS = "datas/aiops/files/jsons/argos-valids.json"
FILES_ROOT = "datas/aiops/files"

device = "cuda"
print("加载 base 学生 + base 教师（OPSD v2 无 best.pt，用 base 前向诊断机制）...")
holder = ModelHolder(BACKBONE, device, grad_checkpointing=False, load_teacher=True)
student = holder.student
teacher = holder.teacher

with open(PROMPT, encoding="utf-8") as f:
    prompt_text = f.read().strip()

recs = json.load(open(VALIDS, encoding="utf-8"))
# 取 3 张不同 trend 的图
targets = ["normal", "spike", "decline"]
picked = []
seen = set()
for r in recs:
    if r["curve"] in targets and r["curve"] not in seen:
        picked.append(r)
        seen.add(r["curve"])
    if len(picked) == len(targets):
        break

print("\n" + "=" * 70)
print("核心诊断：reverse KL 到底在让学生学什么？")
print("=" * 70)

for r in picked:
    curve, recover = r["curve"], r["recover"]
    img_path = os.path.join(FILES_ROOT, r["from"], r["image"])
    pv, gthw, n_pad = encode_image(img_path, holder.ip, 512 * 512)
    prompt_ids = holder._prompt_ids(prompt_text, holder._img_block(n_pad))
    Lp = len(prompt_ids)

    # 1. 学生 on-policy 采样（训练时用 sample_temperature=1.0）
    traj_ids = holder.sample(prompt_ids, pv, gthw, 64, 1.0)
    traj_text = holder.tok.decode(traj_ids, skip_special_tokens=True)
    print(f"\n图 {curve}/{recover} (GT)")
    print(f"  学生采样轨迹: {traj_text!r}")
    print(f"  学生轨迹 parse: {parse_tags(traj_text)}")

    # 2. 学生贪心解码（评估用）
    input_ids = torch.tensor([prompt_ids], device=device)
    attn = torch.ones_like(input_ids)
    mm = (input_ids == holder.vpad).long()
    gen = student.generate(
        input_ids=input_ids, attention_mask=attn,
        pixel_values=pv.to(device), image_grid_thw=gthw.to(device),
        mm_token_type_ids=mm, max_new_tokens=64, do_sample=False,
        pad_token_id=holder.pad_id, eos_token_id=holder.eos_id,
    )
    greedy_text = holder.tok.decode(gen[0][len(prompt_ids):], skip_special_tokens=True)
    print(f"  学生贪心解码: {greedy_text!r}")
    print(f"  贪心 parse: {parse_tags(greedy_text)}")

    # 3. 教师（带引导语）在 traj 位置上的正确 token 概率
    gold_ann_ids = holder.tok.encode(gold_text_annotated(curve, recover), add_special_tokens=False)
    Lg = len(gold_ann_ids)
    teacher_ids = torch.tensor([prompt_ids + gold_ann_ids + traj_ids], device=device)
    with torch.no_grad():
        t_logits = holder.forward_logits(teacher, teacher_ids, pv.to(device), gthw.to(device))
    teacher_traj = t_logits[0, Lp + Lg - 1: Lp + Lg - 1 + len(traj_ids)]
    # 正确标签的 token 序列
    gold_pure_ids = holder.tok.encode(gold_text(curve, recover), add_special_tokens=False)
    print(f"  教师对正确标签各 token 的 top1 概率:")
    for i, tid in enumerate(gold_pure_ids):
        if i < teacher_traj.shape[0]:
            probs = torch.softmax(teacher_traj[i], dim=-1)
            p = probs[tid].item()
            top1 = holder.tok.decode([int(probs.argmax().item())])
            print(f"    pos{i} (GT token {holder.tok.convert_ids_to_tokens([tid])[0]!r}): "
                  f"P(正确)={p:.4f}  top1={top1!r}")
