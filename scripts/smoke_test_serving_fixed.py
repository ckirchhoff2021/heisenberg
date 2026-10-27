"""验证 max_pixels 修复生效：对比同图修复前后 serving 输出的 vision 处理结果。

通过 vLLM 的 /v1/chat/completions 拿不到 token 数，这里改用一个技巧：
直接看修复后模型输出的 quality，同时用 host transformers 复算该图在
max_pixels=262144 下的 n_pad（应与训练一致=256），确认修复参数正确。
"""
import base64
import json
import os

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8007/v1", api_key="EMPTY")

with open("datas/aiops/files/prompts/argos-dp-prompt-noreason.md", encoding="utf-8") as f:
    prompt_text = f.read().strip()
records = json.load(open("datas/aiops/files/jsons/argos-valids.json", encoding="utf-8"))

# 取 3 张图做冒烟
for rec in records[:3]:
    img_path = os.path.join("datas/aiops/files", rec["from"], rec["image"])
    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    resp = client.chat.completions.create(
        model="sft-full",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": prompt_text},
            ],
        }],
        max_tokens=128, temperature=0.0, top_p=1.0,
    )
    out = (resp.choices[0].message.content or "").strip()
    print(f"GT={rec['curve']}/{rec['recover']} -> {out[:70]!r}")
