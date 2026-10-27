"""最终诊断：加载 OPSD v2 训练后的学生（best.pt），看它到底输出什么。

对比 base 学生 vs OPSD v2 训练 300 步后的学生，在 8 张验证图上：
  - 贪心解码文本（评估口径）
  - on-policy 采样文本（训练口径）
若两者仍输出空/乱码，则 reverse KL 完全没学会「图→标签」映射。
"""
import json
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

from train.opsd import ModelHolder, encode_image, parse_tags

BACKBONE = "checkpoints/Qwen3.5-0.8B"
PROMPT = "datas/aiops/files/prompts/argos-dp-prompt-noreason.md"
VALIDS = "datas/aiops/files/jsons/argos-valids.json"
FILES_ROOT = "datas/aiops/files"
CKPT = "outputs/opsd_4gpu_v2/best.pt"

device = "cuda"
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
print("checkpoint step:", ck.get("step"), "mode:", ck.get("mode"),
      "backbone:", ck.get("backbone"), "best_dev_acc:", ck.get("best_dev_acc", "N/A"))
print("state_dict keys:", len(ck["state_dict"]))

holder = ModelHolder(ck["backbone"], device, grad_checkpointing=False, load_teacher=False)
missing, unexpected = holder.student.load_state_dict(ck["state_dict"], strict=True)
print(f"load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
holder.student.eval()

with open(PROMPT, encoding="utf-8") as f:
    prompt_text = f.read().strip()
recs = json.load(open(VALIDS, encoding="utf-8"))

# 8 张图，覆盖多种 trend
picked = []
seen = set()
for r in recs:
    if r["curve"] not in seen:
        picked.append(r)
        seen.add(r["curve"])
    if len(picked) >= 8:
        break

print("\n" + "=" * 72)
print("OPSD v2 训练后学生输出（8 张验证图）")
print("=" * 72)
for r in picked:
    curve, recover = r["curve"], r["recover"]
    img_path = os.path.join(FILES_ROOT, r["from"], r["image"])
    pv, gthw, n_pad = encode_image(img_path, holder.ip, 512 * 512)
    prompt_ids = holder._prompt_ids(prompt_text, holder._img_block(n_pad))
    input_ids = torch.tensor([prompt_ids], device=device)
    attn = torch.ones_like(input_ids)
    mm = (input_ids == holder.vpad).long()
    with torch.no_grad():
        gen = holder.student.generate(
            input_ids=input_ids, attention_mask=attn,
            pixel_values=pv.to(device), image_grid_thw=gthw.to(device),
            mm_token_type_ids=mm, max_new_tokens=64, do_sample=False,
            pad_token_id=holder.pad_id, eos_token_id=holder.eos_id,
        )
    out_text = holder.tok.decode(gen[0][len(prompt_ids):], skip_special_tokens=True)
    trend, recov = parse_tags(out_text)
    ok = (trend == curve and recov == recover)
    print(f"  {curve:9s}/{recover:12s} -> {out_text[:60]!r}")
    print(f"      parse=({trend!r},{recov!r}) {'OK' if ok else 'X'}")
