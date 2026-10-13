"""
GRPO训练模型评测脚本
计算curve分类和recover分类的准确率、precision、recall、F1指标
支持LoRA微调模型和全量模型评测
"""
import re
import argparse
import os
import json
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

import torch
from peft import PeftModel, PeftConfig
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor

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
            "prompt": f"<image> {prompt_text}",
            "image_path": image_path,
            "curve_gt": record["curve"],
            "recover_gt": record["recover"]
        })
    
    print(f"Loaded {len(valid_examples)} valid samples")
    return valid_examples


def load_model_and_processor(model_path: str, device: str = "cuda") -> Tuple[Any, Any]:
    """加载模型和处理器，自动识别Peft LoRA模型"""
    # 检查是否为Peft模型
    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        print(f"Loading PEFT LoRA model from {model_path}")
        peft_config = PeftConfig.from_pretrained(model_path)
        base_model_path = peft_config.base_model_name_or_path
        # 加载基础模型
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
        # 加载LoRA权重
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()  # 合并权重加速推理
    else:
        print(f"Loading full model from {model_path}")
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device
        )
    
    processor = AutoProcessor.from_pretrained(model_path)
    model.eval()
    return model, processor


def inference_single_sample(model: Any, processor: Any, prompt: str, image_path: str, max_new_tokens: int = 1024) -> str:
    """单样本推理"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt.replace("<image>", "").strip()}
            ]
        }
    ]
    
    # 处理输入
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image_path],
        return_tensors="pt"
    ).to(model.device)
    
    # 推理
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            top_p=1.0,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id
        )
    
    # 解码输出，跳过输入部分
    generated_ids = outputs[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return response


def calculate_metrics(all_curve_gt: List[str], all_curve_pred: List[str], 
                      all_recover_gt: List[str], all_recover_pred: List[str],
                      output_file: str = None) -> Dict[str, Any]:
    """计算评测指标"""
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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GRPO trained curve classification model")
    
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to trained model (support LoRA adapter and full model)")
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
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference on (cuda/cpu)")
    
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
    
    # 加载模型和处理器
    model, processor = load_model_and_processor(args.model_path, args.device)
    
    # 批量推理
    all_curve_gt = []
    all_curve_pred = []
    all_recover_gt = []
    all_recover_pred = []
    invalid_output_count = 0
    
    print(f"\nStart inference on {len(valid_examples)} samples...")
    for example in tqdm(valid_examples, desc="Inferencing"):
        response = inference_single_sample(model, processor, example["prompt"], example["image_path"], args.max_new_tokens)
        
        # 解析输出
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
