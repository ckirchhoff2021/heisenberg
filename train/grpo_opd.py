"""
GRPO + OPD (On-Policy Distillation) 混合训练脚本
在学生 on-policy 采样的完成序列上，同时优化：
  1. GRPO 策略梯度损失（基于 reward 的强化学习信号）
  2. OPD 在线蒸馏损失：冻结的强教师在同一批 on-policy 序列上给出 token 分布，
     学生通过 KL 逼近教师分布，缓解 RL 训练中的能力退化 / 分布漂移。
总目标： L = L_GRPO + lambda * KL(teacher || student)
注意：这不是 DPO——OPD 不需要 chosen/rejected 偏好对，教师分布来自其在学生
当前采样序列上的 forward（on-policy），而不是离线标注。
"""
import re
import os
import sys
import argparse
from typing import List, Dict, Any, Tuple

import torch
import torch.nn.functional as F
import datasets
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig, get_peft_model, TaskType
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor

from dataset.grpo import TRLDatasets, FILES_ROOT as DEFAULT_FILES_ROOT, DEFAULT_PROMPT_FILE

# --------------------------
# Default Configuration Constants
# --------------------------
# Student model config
DEFAULT_MODEL_CHECKPOINT = "checkpoints/Qwen3.5-0.8B"
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Teacher model config (stronger, frozen). 请按实际可用的教师模型修改此路径。
DEFAULT_TEACHER_CHECKPOINT = "checkpoints/Qwen3.5-7B"

# Data config
DEFAULT_TRAIN_DATA_PATH = "datas/aiops/files/jsons/seed-grpo.json"

# Training config
DEFAULT_OUTPUT_DIR = "outputs/grpo_opd"
DEFAULT_PER_DEVICE_BATCH_SIZE = 4
DEFAULT_TRAIN_EPOCHS = 3
DEFAULT_LOGGING_STEPS = 5
DEFAULT_SAVE_STEPS = 100
DEFAULT_LEARNING_RATE = 5e-6
DEFAULT_NUM_GENERATIONS = 4
DEFAULT_MAX_COMPLETION_LENGTH = 4096

# OPD config
DEFAULT_OPD_LOSS_WEIGHT = 1.0  # 蒸馏项权重 lambda，0 表示退化为纯 GRPO
DEFAULT_OPD_KL_DIRECTION = "forward"  # forward: KL(teacher||student)；reverse: KL(student||teacher)

# Reward weights
FORMAT_REWARD_WEIGHT = 0.2
CURVE_ACCURACY_REWARD_WEIGHT = 1.0
RECOVER_ACCURACY_REWARD_WEIGHT = 0.5

# Regex patterns
FULL_OUTPUT_PATTERN = r'<reason>(.*?)</reason>\s*<result>(.*?)</result>\s*<recover>(.*?)</recover>'
RESULT_TAG_PATTERN = r'<result>(.*?)</result>'
RECOVER_TAG_PATTERN = r'<recover>(.*?)</recover>'

# Qwen 系多模态 forward 需要透传的额外字段
MULTIMODAL_KEYS = ("pixel_values", "image_grid_thw", "mm_token_type_ids")

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


def _tag_accuracy_reward(completions, ground_truth, pattern, weight):
    """Reward when the content of `pattern` matches the ground-truth label"""
    return [
        weight if extract_tag_content(_completion_text(c), pattern) == gt else 0.0
        for c, gt in zip(completions, ground_truth)
    ]


def curve_accuracy_reward(completions, **kwargs):
    return _tag_accuracy_reward(completions, kwargs["curve"], RESULT_TAG_PATTERN, CURVE_ACCURACY_REWARD_WEIGHT)


def recover_accuracy_reward(completions, **kwargs):
    return _tag_accuracy_reward(completions, kwargs["recover"], RECOVER_TAG_PATTERN, RECOVER_ACCURACY_REWARD_WEIGHT)


# --------------------------
# GRPO + OPD Trainer
# --------------------------
class GRPOOPDTrainer(GRPOTrainer):
    """Adds an on-policy distillation term to the standard GRPO objective."""

    def __init__(
        self,
        *args,
        teacher_model=None,
        opd_loss_weight: float = DEFAULT_OPD_LOSS_WEIGHT,
        opd_kl_direction: str = DEFAULT_OPD_KL_DIRECTION,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.opd_loss_weight = opd_loss_weight
        self.opd_kl_direction = opd_kl_direction
        self.teacher_model = teacher_model
        if self.teacher_model is not None:
            self.teacher_model.to(self.accelerator.device)
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad_(False)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOOPDTrainer does not support returning outputs")

        # Keep all of TRL's clipping / loss_type / normalization logic untouched.
        grpo_loss = self._compute_loss(model, inputs)

        if self.teacher_model is None or self.opd_loss_weight == 0:
            return grpo_loss

        opd_loss, opd_kl = self._compute_opd_loss(model, inputs)
        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["opd/kl"].append(self.accelerator.gather(opd_kl).nanmean().item())
        self._metrics[mode]["opd/loss"].append(self.accelerator.gather(opd_loss).nanmean().item())

        return grpo_loss + self.opd_loss_weight * opd_loss

    def _compute_opd_loss(self, model, inputs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Token-level KL between teacher and student over on-policy completions."""
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        n_completion = completion_ids.size(1)
        mask = completion_mask if "tool_mask" not in inputs else completion_mask * inputs["tool_mask"]

        mm_kwargs = {k: inputs[k] for k in MULTIMODAL_KEYS if inputs.get(k) is not None}

        # Teacher sees exactly the same on-policy sequence + image, but is frozen.
        with torch.no_grad():
            teacher_logits = self.teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                logits_to_keep=n_completion,
                **mm_kwargs,
            ).logits.float()
        # Student logits carry the gradient for the distillation term.
        student_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=n_completion,
            **mm_kwargs,
        ).logits.float()

        if teacher_logits.shape != student_logits.shape:
            raise ValueError(
                f"Teacher/student logits shape mismatch: {teacher_logits.shape} vs {student_logits.shape}. "
                "OPD requires identical vocabulary / sequence length."
            )

        log_p_teacher = F.log_softmax(teacher_logits, dim=-1)
        log_p_student = F.log_softmax(student_logits, dim=-1)
        if self.opd_kl_direction == "forward":  # KL(teacher || student), mode-covering
            per_token_kl = (log_p_teacher.exp() * (log_p_teacher - log_p_student)).sum(dim=-1)
        else:  # reverse KL(student || teacher), mode-seeking
            per_token_kl = (log_p_student.exp() * (log_p_student - log_p_teacher)).sum(dim=-1)

        active = mask.sum().clamp(min=1.0)
        kl_mean = (per_token_kl * mask).sum() / active
        # Match gradient-accumulation scaling used for the policy term.
        accumulation = self.current_gradient_accumulation_steps if self.model.training else 1.0
        opd_loss = kl_mean / accumulation
        return opd_loss, kl_mean.detach()


# --------------------------
# Initialization
# --------------------------
def build_dataset(data_path: str, files_root: str, prompt_file: str) -> datasets.Dataset:
    """Build the conversational multimodal dataset via TRLDatasets."""
    dataset = TRLDatasets(data_path, prompt_file=prompt_file, files_root=files_root).build()
    print(f"Loaded training dataset, total samples: {len(dataset)}")
    return dataset


def init_model(model_checkpoint: str, lora_r: int, lora_alpha: int, lora_dropout: float):
    """Initialize student model with LoRA, and the tokenizer/processor."""
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
    parser = argparse.ArgumentParser(description="GRPO + OPD Training for Curve Trend Classification")

    # Model arguments
    parser.add_argument("--model-checkpoint", type=str, default=DEFAULT_MODEL_CHECKPOINT,
                        help="Path to the student model checkpoint")
    parser.add_argument("--teacher-checkpoint", type=str, default=DEFAULT_TEACHER_CHECKPOINT,
                        help="Path to the frozen teacher model (same architecture family)")
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
    parser.add_argument("--report-to", type=str, default="tensorboard", choices=["tensorboard", "wandb", "none"],
                        help="Report training metrics to")

    # OPD arguments
    parser.add_argument("--opd-loss-weight", type=float, default=DEFAULT_OPD_LOSS_WEIGHT,
                        help="On-policy distillation weight lambda, set to 0 for pure GRPO")
    parser.add_argument("--opd-kl-direction", type=str, default=DEFAULT_OPD_KL_DIRECTION,
                        choices=["forward", "reverse"], help="KL direction for the distillation term")

    return parser.parse_args()


# --------------------------
# Main Training Logic
# --------------------------
if __name__ == "__main__":
    args = parse_args()

    # Load data
    train_dataset = build_dataset(args.train_data_path, args.files_root, args.prompt_file)

    # Initialize student model and processor
    student_model, processor = init_model(
        args.model_checkpoint, args.lora_r, args.lora_alpha, args.lora_dropout
    )

    # Teacher only needed when distillation is enabled.
    teacher_model = None
    if args.opd_loss_weight > 0:
        teacher_model = Qwen3_5ForConditionalGeneration.from_pretrained(
            args.teacher_checkpoint, torch_dtype=torch.bfloat16
        )

    # Configure training arguments
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        logging_dir=f"{args.output_dir}/runs",
        save_steps=args.save_steps,
        report_to=[args.report_to] if args.report_to != "none" else [],
        bf16=not args.fp16,
        fp16=args.fp16,
        learning_rate=args.learning_rate,
        use_vllm=args.use_vllm,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        mask_truncated_completions=True,
    )

    # Initialize trainer
    trainer = GRPOOPDTrainer(
        model=student_model,
        teacher_model=teacher_model,
        opd_loss_weight=args.opd_loss_weight,
        opd_kl_direction=args.opd_kl_direction,
        processing_class=processor,
        reward_funcs=[format_reward_func, curve_accuracy_reward, recover_accuracy_reward],
        args=training_args,
        train_dataset=train_dataset,
    )

    # Start training
    trainer.train()
