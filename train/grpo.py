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

# 三级惩罚梯度（关键设计：保证 GRPO 组内 reward 几乎总有差异 → advantage 非零）
# - 命中真值                    → +weight
# - 输出具体但错误的标签        → WRONG_LABEL_PENALTY（小负，而非 0）
#     原因：若「错误标签」给 0 分，4 条 completion 全错时组内 reward 全 0 → std=0
#           → advantage 全 0 → 这一 group 白训（日志中 frac_reward_zero_std 高达 0.4 的根源）
# - 输出 other / 空 / 非法标签  → OTHER_PENALTY（大负，逼模型别「遇到不确定就放弃」）
WRONG_LABEL_PENALTY = -0.2
OTHER_PENALTY = -0.5

# 逻辑一致性强约束（与 ground truth 无关，检查模型输出内部自洽性）
# 数据验证 100% 确定：trend=normal ⟹ recover=recovered（583/583 样本无一例外）
# trend=normal 但 recover≠recovered 是自相矛盾 → 独立惩罚，驱动模型学会标签配对
CONSISTENCY_PENALTY = -0.3

# 合法标签集合（与 prompt 定义一致，用于 format 校验与 other 判定）
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


def _tag_accuracy_reward(completions, ground_truth, pattern, weight, valid_labels):
    """Reward when the content of `pattern` matches the ground-truth label.

    三级梯度（保证 GRPO 组内 reward 几乎总有差异 → advantage 非零）：
    - 命中真值 → +weight
    - 输出具体但错误的标签 → WRONG_LABEL_PENALTY（小负，而非 0，否则全错组 std=0 白训）
    - 输出 `other` / 空 / 非法标签 → OTHER_PENALTY（大负，逼模型别放弃）
    """
    rewards = []
    for c, gt in zip(completions, ground_truth):
        pred = extract_tag_content(_completion_text(c), pattern)
        if pred == gt:
            rewards.append(weight)
        elif pred in valid_labels and pred != "other":
            rewards.append(WRONG_LABEL_PENALTY)   # 具体但错误 → 小负
        else:
            rewards.append(OTHER_PENALTY)          # other / 空 / 非法 → 大负
    return rewards


def curve_accuracy_reward(completions, **kwargs):
    return _tag_accuracy_reward(
        completions, kwargs["curve"], RESULT_TAG_PATTERN,
        CURVE_ACCURACY_REWARD_WEIGHT, VALID_TREND_LABELS,
    )


def recover_accuracy_reward(completions, **kwargs):
    return _tag_accuracy_reward(
        completions, kwargs["recover"], RECOVER_TAG_PATTERN,
        RECOVER_ACCURACY_REWARD_WEIGHT, VALID_RECOVER_LABELS,
    )


def consistency_reward_func(completions, **kwargs):
    """Reward for internal logical consistency of the model's output.

    与 ground truth 无关，仅检查 trend/recover 两标签是否满足已知的硬约束：
    - trend=normal ⟹ recover=recovered（数据 100% 确定，违反必错）

    返回值语义：满足约束 → 0（不额外加分）；违反约束 → CONSISTENCY_PENALTY。
    注意：只有「能提取到两个标签」时才检查；提取不到则让 tag 精度 reward 去惩罚。
    """
    rewards = []
    for c in completions:
        text = _completion_text(c)
        trend = extract_tag_content(text, RESULT_TAG_PATTERN)
        recover = extract_tag_content(text, RECOVER_TAG_PATTERN)
        # 约束 1：trend=normal 必须 paired recovered（prompt 定义 + 数据 100% 确定）
        if trend == "normal" and recover not in ("", "recovered"):
            rewards.append(CONSISTENCY_PENALTY)
        else:
            rewards.append(0.0)
    return rewards


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
        num_generations=args.num_generations,
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
        reward_funcs=[format_reward_func, curve_accuracy_reward, recover_accuracy_reward, consistency_reward_func],
        args=training_args,
        train_dataset=train_dataset,
    )

    # Start training
    trainer.train()
