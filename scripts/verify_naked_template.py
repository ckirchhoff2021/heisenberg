"""验证裸 chat_template 生效：prompt_tokens 应 == 训练口径 2624（无 role 标记增量）。"""
import base64
import json
import os

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8007/v1", api_key="EMPTY")

prompt_text = open("datas/aiops/files/prompts/argos-dp-prompt-noreason.md", encoding="utf-8").read().strip()
records = json.load(open("datas/aiops/files/jsons/argos-valids.json", encoding="utf-8"))

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
        max_tokens=16, temperature=0.0, top_p=1.0,
    )
    pt = resp.usage.prompt_tokens
    out = (resp.choices[0].message.content or "").strip()[:60]
    print(f"图 {rec['image']}: prompt_tokens={pt}  out={out!r}")
