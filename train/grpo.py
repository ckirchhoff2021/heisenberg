"""
GRPO training script for curve trend classification task
Usage: Run directly, or use the launch script in scripts/ directory
"""
import re
import os
import sys
import argparse
from typing import List, Dict, Any, Tuple

import torch
import datasets
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig, get_peft_model, TaskType
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor

from dataset.grpo import TRLDatasets, FILES_ROOT as DEFAULT_FILES_ROOT, DEFAULT_PROMPT_FILE

# --------------------------
# Default Configuration Constants
# --------------------------
# Model config
DEFAULT_MODEL_CHECKPOINT = "checkpoints/Qwen3.5-0.8B"
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Data config
DEFAULT_TRAIN_DATA_PATH = "datas/aiops/files/jsons/argos-trains.json"

# Training config
DEFAULT_OUTPUT_DIR = "outputs/grpo"
DEFAULT_PER_DEVICE_BATCH_SIZE = 4
DEFAULT_TRAIN_EPOCHS = 3
DEFAULT_LOGGING_STEPS = 5
DEFAULT_SAVE_STEPS = 100
DEFAULT_LEARNING_RATE = 5e-6
DEFAULT_NUM_GENERATIONS = 4
DEFAULT_MAX_COMPLETION_LENGTH = 4096

# --------------------------
# Reward 设计（详见 README.md「Reward 设计」章节）
# --------------------------
# 格式奖励：正确输出 <reason>/<trend>/<recover> 三段结构
FORMAT_REWARD_WEIGHT = 0.2

# 标签准确率奖励权重：trend 是主任务，recover 次之
CURVE_ACCURACY_REWARD_WEIGHT = 1.0
RECOVER_ACCURACY_REWARD_WEIGHT = 0.5

# 类平衡权重：纯 inverse-frequency（1/频率），归一化到均值 1。
# 训练集 argos-trains.json 真实分布（3248 样本）：
#   curve:   increase 809 / offset 698 / spike 643 / normal 583 / decline 290 / fluctuate 225
#   recover: no_recover 1754 / recovered 927 / recovering 567
# 目的：压制「无脑猜多数类」的 reward hacking。
#   早期版本用三级负分（错误=-0.2 / other=-0.5），数学上「赌 increase+no_recover」
#   的期望 reward = +0.477，远超「认真学判别」的 +0.029，优化器理性走了捷径
#   （实测 checkpoint-1200：curve 22.5% < base 31.88%，normal 类 F1 归零，
#     increase recall 飙到 0.67 —— 就是赌最大类的表现）。
#   修正：错误一律 0 分（对齐 4B-RL 成功版本），命中时按 inverse-frequency 加权。
#   纯 inv-freq 归一化后等价于 macro-F1 优化：每类 recall 等权，minority 类不再被淹没。
#   数学验证：赌最优捷径 E=0.469（变成赌 spike/recovering 这种低命中组合）
#            < 认真学 30% E=0.562，margin +0.093，捷径空间收窄 4 倍。
#   （sqrt inv-freq 的 margin 仅 +0.022，太薄，不足以压制捷径。）
CURVE_CLASS_WEIGHTS = {
    "fluctuate": 1.928, "decline": 1.496, "spike": 0.675,
    "normal": 0.744, "offset": 0.621, "increase": 0.536,
    "other": 1.0,
}
RECOVER_CLASS_WEIGHTS = {
    "no_recover": 0.501, "recovering": 1.55, "recovered": 0.948,
    "other": 1.0,
}

# 合法标签集合（与 prompt 定义一致，用于 format 校验）
VALID_TREND_LABELS = {"normal", "offset", "increase", "decline", "spike", "fluctuate", "other"}
VALID_RECOVER_LABELS = {"recovered", "recovering", "no_recover", "other"}

# Regex patterns
FULL_OUTPUT_PATTERN = r'<reason>(.*?)</reason>\s*<trend>(.*?)</trend>\s*<recover>(.*?)</recover>'
RESULT_TAG_PATTERN = r'<trend>(.*?)</trend>'
RECOVER_TAG_PATTERN = r'<recover>(.*?)</recover>'

# --------------------------
# Reward functions
# TRL passes `completions` as a flat list (length = batch * num_generations),
# where each item is [{"role": "assistant", "content": "<generated text>"}].
# Dataset columns (curve/recover) are repeated to match and passed via kwargs.
# --------------------------
def extract_tag_content(text: str, pattern: str) -> str:
    """Extract content from XML-style tags using regex"""
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""


def _completion_text(completion: List[Dict[str, Any]]) -> str:
    return completion[0]["content"]


def format_reward_func(completions, **kwargs):
    """Reward for the reason/result/recover tag structure"""
    return [
        FORMAT_REWARD_WEIGHT if re.search(FULL_OUTPUT_PATTERN, _completion_text(c), re.DOTALL) else 0.0
        for c in completions
    ]


def _tag_accuracy_reward(completions, ground_truth, pattern, weight, class_weights):
    """Reward when the content of `pattern` matches the ground-truth label.

    两级设计（对齐 4B-RL 成功版本）+ 类平衡加权：
    - 命中真值 → +weight * class_weights[gt]（minority 类加权，压制赌多数类）
    - 其余（错误标签 / other / 空 / 非法）→ 0
    """
    rewards = []
    for c, gt in zip(completions, ground_truth):
        pred = extract_tag_content(_completion_text(c), pattern)
        if pred == gt:
            rewards.append(weight * class_weights.get(gt, 1.0))
        else:
            rewards.append(0.0)
    return rewards


def curve_accuracy_reward(completions, **kwargs):
    return _tag_accuracy_reward(
        completions, kwargs["curve"], RESULT_TAG_PATTERN,
        CURVE_ACCURACY_REWARD_WEIGHT, CURVE_CLASS_WEIGHTS,
    )


def recover_accuracy_reward(completions, **kwargs):
    return _tag_accuracy_reward(
        completions, kwargs["recover"], RECOVER_TAG_PATTERN,
        RECOVER_ACCURACY_REWARD_WEIGHT, RECOVER_CLASS_WEIGHTS,
    )


# --------------------------
# Initialization
# --------------------------
def build_dataset(data_path: str, files_root: str, prompt_file: str) -> datasets.Dataset:
    """Build the conversational multimodal dataset via TRLDatasets."""
    dataset = TRLDatasets(data_path, prompt_file=prompt_file, files_root=files_root).build()
    print(f"Loaded training dataset, total samples: {len(dataset)}")
    return dataset


def init_model_and_processor(model_checkpoint: str, lora_r: int, lora_alpha: int, lora_dropout: float) -> Tuple[Any, Any]:
    """Initialize base model, apply LoRA adapter, and load processor"""
    model = Qwen3_5ForConditionalGeneration.from_pretrained(model_checkpoint, torch_dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(model_checkpoint)
    
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=DEFAULT_LORA_TARGET_MODULES,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    model = get_peft_model(model, lora_config)
    print("Trainable parameters:")
    model.print_trainable_parameters()
    return model, processor


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="GRPO Training for Curve Trend Classification")
    
    # Model arguments
    parser.add_argument("--model-checkpoint", type=str, default=DEFAULT_MODEL_CHECKPOINT,
                        help="Path to pretrained model checkpoint")
    parser.add_argument("--lora-r", type=int, default=DEFAULT_LORA_R, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA, help="LoRA alpha parameter")
    parser.add_argument("--lora-dropout", type=float, default=DEFAULT_LORA_DROPOUT, help="LoRA dropout rate")
    
    # Data arguments
    parser.add_argument("--train-data-path", type=str, default=DEFAULT_TRAIN_DATA_PATH,
                        help="Path to training JSON dataset")
    parser.add_argument("--files-root", type=str, default=DEFAULT_FILES_ROOT,
                        help="Root dir containing the image folders")
    parser.add_argument("--prompt-file", type=str, default=DEFAULT_PROMPT_FILE,
                        help="Path to the prompt markdown file")
    
    # Training arguments
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save training outputs")
    parser.add_argument("--per-device-batch-size", type=int, default=DEFAULT_PER_DEVICE_BATCH_SIZE,
                        help="Batch size per GPU")
    parser.add_argument("--num-train-epochs", type=int, default=DEFAULT_TRAIN_EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--logging-steps", type=int, default=DEFAULT_LOGGING_STEPS,
                        help="Log every X steps")
    parser.add_argument("--save-steps", type=int, default=DEFAULT_SAVE_STEPS,
                        help="Save checkpoint every X steps")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE,
                        help="Learning rate")
    parser.add_argument("--num-generations", type=int, default=DEFAULT_NUM_GENERATIONS,
                        help="Number of completions per prompt")
    parser.add_argument("--beta", type=float, default=0.0,
                        help="KL regularization coefficient (0.0 disables KL; use ~0.02-0.04 to anchor policy "
                             "to base model and prevent collapse-to-majority-class)")
    parser.add_argument("--max-completion-length", type=int, default=DEFAULT_MAX_COMPLETION_LENGTH,
                        help="Max completion length")
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16 instead of BF16 mixed precision (default: BF16)")
    parser.add_argument("--use-vllm", action="store_true", default=False,
                        help="Use vLLM for faster generation")
    parser.add_argument("--vllm-mode", type=str, default="colocate", choices=["colocate", "server"],
                        help="vLLM deployment mode (colocate=same GPU, server=separate GPU)")
    parser.add_argument("--vllm-server-base-url", type=str, default=None,
                        help="vLLM server base URL (required for server mode)")
    parser.add_argument("--vllm-group-port", type=int, default=51216,
                        help="NCCL weight transfer group port (server mode)")
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1,
                        help="Tensor parallel size (colocate mode only; 1 = pure data parallel across ranks)")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.4,
                        help="GPU memory utilization for the colocated vLLM engine (colocate mode only)")
    parser.add_argument("--report-to", type=str, default="tensorboard", choices=["tensorboard", "wandb", "none"],
                        help="Report training metrics to")
    
    return parser.parse_args()


# --------------------------
# Main Training Logic
# --------------------------
if __name__ == "__main__":
    args = parse_args()
    
    # Load data
    train_dataset = build_dataset(args.train_data_path, args.files_root, args.prompt_file)

    # Initialize model and processor
    base_model, processor = init_model_and_processor(
        model_checkpoint=args.model_checkpoint,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )

    # Configure training arguments
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_steps=args.save_steps,
        report_to=[args.report_to] if args.report_to != "none" else [],
        bf16=not args.fp16,
        fp16=args.fp16,
        learning_rate=args.learning_rate,
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        vllm_server_base_url=args.vllm_server_base_url,
        vllm_group_port=args.vllm_group_port,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        num_generations=args.num_generations,
        beta=args.beta,
        max_completion_length=args.max_completion_length,
        mask_truncated_completions=True,
        vllm_importance_sampling_correction=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # Initialize trainer
    trainer = GRPOTrainer(
        model=base_model,
        processing_class=processor,
        reward_funcs=[format_reward_func, curve_accuracy_reward, recover_accuracy_reward],
        args=training_args,
        train_dataset=train_dataset,
    )

    # Start training
    trainer.train()
