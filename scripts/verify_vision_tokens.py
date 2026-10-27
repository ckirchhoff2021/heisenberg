"""精确计算 chat_template 的文本增量（role 标记 + generation prompt + think 块）。

serving prompt_tokens(2636) = 训练口径总 token(2624) + chat_template 文本增量。
本脚本算出这个增量，坐实「剩余掉点来自 chat_template 角色标记」这一假设。
"""
from transformers import AutoTokenizer

CKPT = "checkpoints/Qwen3.5-0.8B"
tok = AutoTokenizer.from_pretrained(CKPT, local_files_only=True)

prompt_text = open("datas/aiops/files/prompts/argos-dp-prompt-noreason.md", encoding="utf-8").read().strip()

# 训练口径：纯文本 ids
train_text_ids = tok.encode(prompt_text, add_special_tokens=False)

# serving 口径：chat_template 渲染（无图，纯看文本模板的增量）
messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
serving_str = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
serving_ids = tok.encode(serving_str, add_special_tokens=False)

print(f"训练口径纯文本 token: {len(train_text_ids)}")
print(f"serving chat_template 纯文本 token: {len(serving_ids)}")
print(f"chat_template 文本增量: {len(serving_ids) - len(train_text_ids)} token")

print("\n--- 增量拆解 ---")
# 前缀 role 标记
prefix = "<|im_start|>user\n"
print(f"前缀 '<|im_start|>user\\n': {len(tok.encode(prefix, add_special_tokens=False))} token")

# 后缀 generation prompt + 空 think 块
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
print(f"后缀 (im_end + assistant + 空think): {len(tok.encode(suffix, add_special_tokens=False))} token")
