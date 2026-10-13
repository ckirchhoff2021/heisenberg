"""
GRPO训练模型评测脚本（VLLM加速版）
比原生HuggingFace推理速度提升5~10倍，支持批量推理
支持两种模式：
1. 本地VLLM引擎模式：直接在本地加载模型用VLLM推理
2. 远程VLLM服务模式：调用已部署的VLLM OpenAI兼容接口
"""
import re
import argparse
import os
import json
import base64
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from openai import OpenAI

# --------------------------
# 配置常量
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
# 工具函数
# --------------------------
def extract_tag_content(text: str, pattern: str) -> str:
    """提取标签内容，匹配失败返回空字符串"""
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""


def encode_image_to_base64(image_path: str) -> str:
    """将图像编码为base64，用于VLLM多模态输入"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def load_valid_dataset(data_path: str, prompt_text: str, files_root: str, check_images: bool = True) -> List[Dict[str, Any]]:
    """加载验证集，格式和训练集保持一致"""
    with open(data_path, encoding="utf-8") as f:
        records = json.load(f)
    
    valid_examples = []
    for record in tqdm(records, desc="Loading validation dataset"):
        image_path = os.path.join(files_root, record["from"], record["image"])
        if check_images and not os.path.isfile(image_path):
            print(f"Warning: Image not found {image_path}, skip this sample")
            continue
        
        valid_examples.append({
            "prompt": prompt_text.strip(),
            "image_path": image_path,
            "curve_gt": record["curve"],
            "recover_gt": record["recover"]
        })
    
    print(f"Loaded {len(valid_examples)} valid samples")
    return valid_examples


def calculate_metrics(all_curve_gt: List[str], all_curve_pred: List[str], 
                      all_recover_gt: List[str], all_recover_pred: List[str],
                      output_file: str = None) -> Dict[str, Any]:
    """计算评测指标，和原生版完全一致"""
    metrics = {}
    
    # 整体准确率
    metrics["curve_accuracy"] = accuracy_score(all_curve_gt, all_curve_pred)
    metrics["recover_accuracy"] = accuracy_score(all_recover_gt, all_recover_pred)
    metrics["joint_accuracy"] = np.mean([
        1 if c_gt == c_pred and r_gt == r_pred else 0 
        for c_gt, c_pred, r_gt, r_pred in zip(all_curve_gt, all_curve_pred, all_recover_gt, all_recover_pred)
    ])
    
    # Curve分类详细指标
    print("\n===== Curve Trend Classification Metrics =====")
    print(f"Overall Accuracy: {metrics['curve_accuracy']:.4f}")
    curve_report = classification_report(all_curve_gt, all_curve_pred, labels=CURVE_LABELS, target_names=CURVE_LABELS, zero_division=0)
    print(curve_report)
    metrics["curve_report"] = classification_report(all_curve_gt, all_curve_pred, labels=CURVE_LABELS, target_names=CURVE_LABELS, zero_division=0, output_dict=True)
    
    # Recover分类详细指标
    print("\n===== Recover State Classification Metrics =====")
    print(f"Overall Accuracy: {metrics['recover_accuracy']:.4f}")
    recover_report = classification_report(all_recover_gt, all_recover_pred, labels=RECOVER_LABELS, target_names=RECOVER_LABELS, zero_division=0)
    print(recover_report)
    metrics["recover_report"] = classification_report(all_recover_gt, all_recover_pred, labels=RECOVER_LABELS, target_names=RECOVER_LABELS, zero_division=0, output_dict=True)
    
    # 联合准确率
    print("\n===== Joint Metrics =====")
    print(f"Curve + Recover Both Correct Accuracy: {metrics['joint_accuracy']:.4f}")
    
    # 保存结果到文件
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"\nMetrics saved to {output_file}")
    
    return metrics


def inference_local_vllm(model_path: str, valid_examples: List[Dict[str, Any]], max_new_tokens: int = 1024, batch_size: int = 32) -> List[str]:
    """本地VLLM引擎批量推理，速度最快"""
    from vllm import LLM, SamplingParams
    
    print(f"Loading VLLM model from {model_path}...")
    # 初始化VLLM引擎，Qwen3.5多模态原生支持
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        max_model_len=8192,
        # 多模态配置
        limit_mm_per_prompt={"image": 1},
    )
    
    # 采样参数，和原生版对齐（Greedy解码）
    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        stop_token_ids=[llm.get_tokenizer().eos_token_id],
        skip_special_tokens=True,
    )
    
    # 构造批量输入
    prompts = []
    for example in valid_examples:
        # Qwen多模态输入格式
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": example["image_path"]},
                    {"type": "text", "text": example["prompt"]}
                ]
            }
        ]
        # 应用聊天模板
        prompt = llm.get_tokenizer().apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
    
    # 批量推理
    print(f"Start batch inference on {len(prompts)} samples, batch size={batch_size}...")
    outputs = llm.generate(prompts, sampling_params)
    
    # 提取输出
    responses = [output.outputs[0].text.strip() for output in outputs]
    return responses


def inference_remote_vllm(api_base: str, api_key: str, model_name: str, valid_examples: List[Dict[str, Any]], max_new_tokens: int = 1024) -> List[str]:
    """调用远程VLLM OpenAI兼容接口推理，适合已经部署好服务的场景"""
    client = OpenAI(
        base_url=api_base,
        api_key=api_key,
    )
    
    responses = []
    print(f"Start inference with remote VLLM service {api_base}...")
    for example in tqdm(valid_examples, desc="Inferencing"):
        # 图像转base64
        image_base64 = encode_image_to_base64(example["image_path"])
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                        {"type": "text", "text": example["prompt"]}
                    ]
                }
            ],
            max_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
        )
        responses.append(response.choices[0].message.content.strip())
    return responses


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GRPO trained curve classification model (VLLM accelerated)")
    
    # 公共参数
    parser.add_argument("--valid-data-path", type=str, default=DEFAULT_VALID_DATA_PATH,
                        help="Path to validation dataset JSON")
    parser.add_argument("--prompt-file", type=str, default=DEFAULT_PROMPT_FILE,
                        help="Path to prompt markdown file")
    parser.add_argument("--files-root", type=str, default=DEFAULT_FILES_ROOT,
                        help="Root directory for image files")
    parser.add_argument("--output-file", type=str, default=None,
                        help="Path to save evaluation metrics JSON")
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="Max new tokens for generation")
    
    # 模式选择
    parser.add_argument("--mode", type=str, default="local", choices=["local", "remote"],
                        help="Inference mode: local (VLLM engine) / remote (VLLM OpenAI API)")
    
    # 本地模式参数
    parser.add_argument("--model-path", type=str, help="Path to trained model (for local mode)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for local VLLM inference")
    
    # 远程模式参数
    parser.add_argument("--api-base", type=str, help="Remote VLLM API base URL, e.g. http://localhost:8000/v1")
    parser.add_argument("--api-key", type=str, default="EMPTY", help="Remote VLLM API key")
    parser.add_argument("--model-name", type=str, help="Remote model name")
    
    return parser.parse_args()


# --------------------------
# 主逻辑
# --------------------------
if __name__ == "__main__":
    args = parse_args()
    
    # 加载prompt
    with open(args.prompt_file, encoding="utf-8") as f:
        prompt_text = f.read().strip()
    
    # 加载验证集
    valid_examples = load_valid_dataset(args.valid_data_path, prompt_text, args.files_root)
    
    # 推理
    if args.mode == "local":
        if not args.model_path:
            raise ValueError("Please provide --model-path for local mode")
        responses = inference_local_vllm(args.model_path, valid_examples, args.max_new_tokens, args.batch_size)
    else:  # remote
        if not args.api_base or not args.model_name:
            raise ValueError("Please provide --api-base and --model-name for remote mode")
        responses = inference_remote_vllm(args.api_base, args.api_key, args.model_name, valid_examples, args.max_new_tokens)
    
    # 解析输出
    all_curve_gt = []
    all_curve_pred = []
    all_recover_gt = []
    all_recover_pred = []
    invalid_output_count = 0
    
    for example, response in zip(valid_examples, responses):
        curve_pred = extract_tag_content(response, RESULT_TAG_PATTERN)
        recover_pred = extract_tag_content(response, RECOVER_TAG_PATTERN)
        
        # 统计无效输出
        if not curve_pred or not recover_pred:
            invalid_output_count += 1
            # 无效输出归为other类
            curve_pred = curve_pred if curve_pred else "other"
            recover_pred = recover_pred if recover_pred else "other"
        
        all_curve_gt.append(example["curve_gt"])
        all_curve_pred.append(curve_pred)
        all_recover_gt.append(example["recover_gt"])
        all_recover_pred.append(recover_pred)
    
    # 统计无效输出率
    invalid_rate = invalid_output_count / len(valid_examples)
    print(f"\nInvalid output rate: {invalid_rate:.4f} ({invalid_output_count}/{len(valid_examples)})")
    
    # 计算指标
    calculate_metrics(all_curve_gt, all_curve_pred, all_recover_gt, all_recover_pred, args.output_file)
