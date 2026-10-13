"""
JEV模型单条VLLM推理测试脚本
对齐evaluate/grpo_eval_vllm.py实现逻辑，支持本地/远程两种vLLM模式
随机从验证集取1条数据，输出推理结果和真实标签对比
"""
import re
import random
import argparse
import os
import json
import base64
from typing import Dict, Any
from openai import OpenAI
from vllm import LLM, SamplingParams

# --------------------------
# 配置常量（对齐grpo_eval_vllm.py）
# --------------------------
DEFAULT_VALID_DATA_PATH = "datas/aiops/files/jsons/argos-valids.json"
DEFAULT_PROMPT_FILE = "datas/aiops/files/prompts/argos-dp-prompt.md"
DEFAULT_FILES_ROOT = "datas/aiops/files"

# 标签列表
CURVE_LABELS = ["spike", "decline", "increase", "normal", "offset", "fluctuate", "other"]
RECOVER_LABELS = ["recovered", "recovering", "no_recover", "other"]

# 正则匹配规则
RESULT_TAG_PATTERN = r'<trend>(.*?)</trend>'
RECOVER_TAG_PATTERN = r'<recover>(.*?)</recover>'
FULL_OUTPUT_PATTERN = r'<reason>(.*?)</reason>\s*<trend>(.*?)</trend>\s*<recover>(.*?)</recover>'

# --------------------------
# 工具函数（复用grpo_eval_vllm.py）
# --------------------------
def extract_tag_content(text: str, pattern: str) -> str:
    """提取标签内容，匹配失败返回空字符串"""
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""


def encode_image_to_base64(image_path: str) -> str:
    """将图像编码为base64，用于VLLM多模态输入"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def load_random_sample(data_path: str, prompt_text: str, files_root: str, sample_idx: int = None) -> Dict[str, Any]:
    """加载验证集，随机取1条或指定下标的样本"""
    with open(data_path, encoding="utf-8") as f:
        records = json.load(f)
    
    if sample_idx is not None:
        if sample_idx < 0 or sample_idx >= len(records):
            raise ValueError(f"Sample index must be between 0 and {len(records)-1}")
        record = records[sample_idx]
    else:
        record = random.choice(records)
    
    image_path = os.path.join(files_root, record["from"], record["image"])
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    return {
        "prompt": prompt_text.strip(),
        "image_path": image_path,
        "curve_gt": record["curve"],
        "recover_gt": record["recover"],
        "index": sample_idx if sample_idx is not None else records.index(record)
    }


def inference_local_vllm(model_path: str, sample: Dict[str, Any], max_new_tokens: int = 1024) -> str:
    """本地VLLM引擎推理"""
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.8,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
    )
    
    # 图像转base64
    image_base64 = encode_image_to_base64(sample["image_path"])
    prompt = f"<|im_start|>user\n<image>\n{sample['prompt']}<|im_end|>\n<|im_start|>assistant\n"
    
    outputs = llm.generate(
        {
            "prompt": prompt,
            "multi_modal_data": {
                "image": image_base64
            }
        },
        sampling_params=sampling_params
    )
    return outputs[0].outputs[0].text.strip()


def inference_remote_vllm(api_base: str, api_key: str, model_name: str, sample: Dict[str, Any], max_new_tokens: int = 1024) -> str:
    """远程VLLM OpenAI兼容接口推理"""
    client = OpenAI(
        base_url=api_base,
        api_key=api_key,
    )
    
    # 图像转base64
    image_base64 = encode_image_to_base64(sample["image_path"])
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                    {"type": "text", "text": sample["prompt"]}
                ]
            }
        ],
        max_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
    )
    return response.choices[0].message.content.strip()


def parse_args():
    parser = argparse.ArgumentParser(description="JEV single sample VLLM inference test")
    
    # 公共参数
    parser.add_argument("--valid-data-path", type=str, default=DEFAULT_VALID_DATA_PATH,
                        help="Path to validation dataset JSON")
    parser.add_argument("--prompt-file", type=str, default=DEFAULT_PROMPT_FILE,
                        help="Path to prompt markdown file")
    parser.add_argument("--files-root", type=str, default=DEFAULT_FILES_ROOT,
                        help="Root directory for image files")
    parser.add_argument("--sample-index", type=int, default=None,
                        help="Specify sample index to test, default random pick")
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="Max new tokens for generation")
    
    # 模式选择
    parser.add_argument("--mode", type=str, default="remote", choices=["local", "remote"],
                        help="Inference mode: local (VLLM engine) / remote (VLLM OpenAI API)")
    
    # 本地模式参数
    parser.add_argument("--model-path", type=str, help="Path to trained model (for local mode)")
    
    # 远程模式参数
    parser.add_argument("--api-base", type=str, default="http://localhost:8000/v1",
                        help="Remote VLLM API base URL, e.g. http://localhost:8000/v1")
    parser.add_argument("--api-key", type=str, default="EMPTY", help="Remote VLLM API key")
    parser.add_argument("--model-name", type=str, default="jev-model", help="Remote model name")
    
    return parser.parse_args()


# --------------------------
# 主逻辑
# --------------------------
if __name__ == "__main__":
    args = parse_args()
    
    # 加载prompt
    with open(args.prompt_file, encoding="utf-8") as f:
        prompt_text = f.read().strip()
    
    # 加载样本
    sample = load_random_sample(args.valid_data_path, prompt_text, args.files_root, args.sample_index)
    print(f"="*60)
    print(f"✅ 加载样本成功，索引: {sample['index']}")
    print(f"📸 图片路径: {sample['image_path']}")
    print(f"🏷️  真实标签: 曲线趋势={sample['curve_gt']}, 恢复状态={sample['recover_gt']}")
    print(f"="*60)
    
    # 推理
    print(f"\n🚀 开始{args.mode}模式推理...")
    if args.mode == "local":
        if not args.model_path:
            raise ValueError("Please provide --model-path for local mode")
        response = inference_local_vllm(args.model_path, sample, args.max_new_tokens)
    else:  # remote
        response = inference_remote_vllm(args.api_base, args.api_key, args.model_name, sample, args.max_new_tokens)
    
    # 解析结果
    curve_pred = extract_tag_content(response, RESULT_TAG_PATTERN)
    recover_pred = extract_tag_content(response, RECOVER_TAG_PATTERN)
    reason = extract_tag_content(response, r'<reason>(.*?)</reason>')
    
    # 输出结果
    print(f"\n📝 模型完整输出:")
    print(f"{response}")
    print(f"\n🔍 解析结果:")
    print(f"推理理由: {reason if reason else '未检测到<reason>标签'}")
    print(f"预测曲线趋势: {curve_pred if curve_pred else '未检测到<result>标签'}")
    print(f"预测恢复状态: {recover_pred if recover_pred else '未检测到<recover>标签'}")
    print(f"\n✅ 对比结果:")
    curve_match = curve_pred == sample['curve_gt']
    recover_match = recover_pred == sample['recover_gt']
    print(f"曲线趋势匹配: {'✅ 正确' if curve_match else '❌ 错误'}")
    print(f"恢复状态匹配: {'✅ 正确' if recover_match else '❌ 错误'}")
    print(f"整体结果: {'✅ 全部正确' if curve_match and recover_match else '❌ 存在错误'}")
    print(f"="*60)

