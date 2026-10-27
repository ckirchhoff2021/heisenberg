"""最终验证：safetensors 导出权重 == best.pt 训练后权重（tie 语义正确处理）。"""
import numpy as np
import torch
from safetensors import safe_open

best = torch.load("outputs/sft_full_4gpu/best.pt", map_location="cpu", weights_only=False)
best_sd = best["state_dict"]

f = safe_open("outputs/sft_full_4gpu/safetensors_model/model.safetensors", framework="pt")
sft_keys = set(f.keys())
best_keys = set(best_sd.keys())

only_in_best = best_keys - sft_keys
only_in_sft = sft_keys - best_keys
print("best.pt 独有 key:", sorted(only_in_best))
print("safetensors 独有 key:", sorted(only_in_sft))

# 唯一差异应该是 lm_head.weight，且因 tie_word_embeddings 应等于 embed_tokens.weight
assert only_in_best == {"lm_head.weight"}, f"意外差异: {only_in_best}"
assert not only_in_sft, f"safetensors 多出 key: {only_in_sft}"

lm_head = best_sd["lm_head.weight"].to(torch.float32).numpy()
embed = f.get_tensor("model.language_model.embed_tokens.weight").to(torch.float32).numpy()
tie_ok = np.allclose(lm_head, embed, atol=1e-3)
print("lm_head == embed_tokens (tie 语义):", "PASS" if tie_ok else "FAIL")

# 抽查所有关键模块（用真实存在的 key）
sample_keys = [
    "model.visual.patch_embed.proj.weight",
    "model.visual.blocks.11.attn.proj.weight",  # 最后一个视觉 block
    "model.language_model.embed_tokens.weight",
    "model.language_model.layers.0.linear_attn.out_proj.weight",  # linear_attn 层
    "model.language_model.layers.3.self_attn.o_proj.weight",      # full_attention 层(第3层)
    "model.language_model.layers.23.mlp.down_proj.weight",         # 最后一层
]
all_match = True
for k in sample_keys:
    if k not in sft_keys:
        print(f"  [缺失] {k}")
        all_match = False
        continue
    a = best_sd[k].to(torch.float32).numpy()
    b = f.get_tensor(k).to(torch.float32).numpy()
    match = np.allclose(a, b, atol=1e-3)
    print(f"  {k}: {'MATCH' if match else 'MISMATCH'}")
    all_match = all_match and match

print()
print("=" * 50)
print("最终结论:", "✅ 全部通过，safetensors 导出正确" if (tie_ok and all_match) else "❌ 失败")
