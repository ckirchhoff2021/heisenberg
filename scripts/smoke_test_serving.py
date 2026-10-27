"""vLLM serving 冒烟测试：单张验证图，确认能输出 <trend>/<recover> 标签。"""
import base64
import json
import os

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8007/v1", api_key="EMPTY")

# 读 prompt + 取第一张验证图
with open("datas/aiops/files/prompts/argos-dp-prompt-noreason.md", encoding="utf-8") as f:
    prompt_text = f.read().strip()
records = json.load(open("datas/aiops/files/jsons/argos-valids.json", encoding="utf-8"))
rec = records[0]
img_path = os.path.join("datas/aiops/files", rec["from"], rec["image"])
print(f"测试图: {rec['from']}/{rec['image']}")
print(f"GT: curve={rec['curve']} recover={rec['recover']}")

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
    max_tokens=128,
    temperature=0.0,
    top_p=1.0,
)

out = resp.choices[0].message.content.strip()
print(f"\n模型输出:\n{out!r}")
print(f"\nfinish_reason: {resp.choices[0].finish_reason}")
