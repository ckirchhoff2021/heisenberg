"""对照实验：验证给 gold_text 加「这是正确答案」标注能否救回教师 top1。

背景：当前 gold_text 是纯标签 `<trend>x</trend>\n<recover>y</recover>`，无任何
「这是答案」标注。前面验证发现教师对 trend 类别词 top1 恒为 other（主动答错）。

本实验：固定同一张图（normal/recovered），固定 traj=gold，对比 4 种 gold 标注格式
下，教师对 traj 各 token 的 top1 预测。若加标注后 top1 从 other 变为正确标签，
则说明「标注」是可行补救；若仍为 other，则问题更深（base 视觉塔看不懂图/不认引导）。
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
holder = ModelHolder(BACKBONE, device, grad_checkpointing=False, load_teacher=True)
teacher = holder.teacher
teacher.eval()

with open(PROMPT, encoding="utf-8") as f:
    prompt_text = f.read().strip()

recs = json.load(open(VALIDS, encoding="utf-8"))
# 找 3 张不同 trend 的图
targets = ["normal", "spike", "decline"]
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

# 4 种 gold 标注格式
def gold_variants(curve, recover):
    base = gold_text(curve, recover)
    return {
        "A.纯标签(现状)": base,
        "B.引导语前缀": f"正确答案如下：\n{base}",
        "C.强引导few-shot": f"已知该时序曲线图的正确分类为：\n{base}\n请据此复述答案。",
        "D.答案token标注": f"<answer>\n{base}\n</answer>",
    }

# 对每张图，用其真实标签，traj = gold（理想学生采对的情况）
for r in picked:
    curve, recover = r["curve"], r["recover"]
    img_path = os.path.join(FILES_ROOT, r["from"], r["image"])
    pv, gthw, n_pad = encode_image(img_path, holder.ip, 512 * 512)
    prompt_ids = holder._prompt_ids(prompt_text, holder._img_block(n_pad))
    Lp = len(prompt_ids)

    print(f"\n{'='*70}")
    print(f"图: {curve}/{recover}  (GT)")
    print(f"{'='*70}")

    variants = gold_variants(curve, recover)
    for name, gold_str in variants.items():
        gold_ids = holder.tok.encode(gold_str, add_special_tokens=False)
        traj_ids = holder.tok.encode(gold_text(curve, recover), add_special_tokens=False)
        Lg = len(gold_ids)
        Lt = len(traj_ids)
        teacher_ids = torch.tensor([prompt_ids + gold_ids + traj_ids], device=device)
        with torch.no_grad():
            t_logits = holder.forward_logits(teacher, teacher_ids, pv.to(device), gthw.to(device))
        traj_logits = t_logits[0, Lp + Lg - 1: Lp + Lg - 1 + Lt]
        traj_toks = holder.tok.convert_ids_to_tokens(traj_ids)
        # 内容 token 位置
        content_pos = [i for i, tk in enumerate(traj_toks)
                       if tk.strip("Ġ") in ("normal", "re", "covered", "sp", "ike",
                                            "decl", "ine", "no", "_re", "cover",
                                            "recover", "ing")]
        tops = []
        for pos in content_pos:
            top1 = holder.tok.decode([int(traj_logits[pos].argmax().item())])
            tops.append(f"{traj_toks[pos]}→{top1}")
        print(f"  {name:18s} | " + " | ".join(tops))
