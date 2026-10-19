"""
SFT training script for curve trend classification task.

与 GRPO 的关键差异：
- 纯 cross-entropy 监督，直接逼近黄金标签（<trend>/<recover>），无采样、无 reward hacking 风险；
- 数据级调整见 dataset/sft.py：有 output 的借 <reason> 但标签以 JSON 字段为准，
  无 output 的用换行占位 reason + 拼接 curve/recover。

Usage: python train/sft.py [--model-checkpoint ...] [--output-dir ...]
"""
import os
import argparse

import torch
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, TaskType
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor

from dataset.sft import SFTDatasets, FILES_ROOT as DEFAULT_FILES_ROOT, DEFAULT_PROMPT_FILE

# --------------------------
# Default Configuration Constants
# --------------------------
DEFAULT_MODEL_CHECKPOINT = "checkpoints/Qwen3.5-0.8B"
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

DEFAULT_TRAIN_DATA_PATH = "datas/aiops/files/jsons/argos-trains.json"
DEFAULT_OUTPUT_DIR = "outputs/sft"
DEFAULT_PER_DEVICE_BATCH_SIZE = 4
DEFAULT_GRAD_ACCUM = 4
DEFAULT_TRAIN_EPOCHS = 3
DEFAULT_LOGGING_STEPS = 5
DEFAULT_SAVE_STEPS = 100
DEFAULT_LEARNING_RATE = 5e-5
# 注意：Qwen3.5-VL 图片 token 约 3900~4100，prompt 部分就超过 2048，
# 必须 >= prompt+completion 总长，否则 completion 被截断、labels 全 -100、loss=0
DEFAULT_MAX_LENGTH = 8192


def build_dataset(data_path: str, files_root: str, prompt_file: str):
    dataset = SFTDatasets(data_path, prompt_file=prompt_file, files_root=files_root).build()
    print(f"Loaded training dataset, total samples: {len(dataset)}")
    return dataset


def init_model_and_processor(model_checkpoint: str, lora_r: int, lora_alpha: int, lora_dropout: float):
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
    parser = argparse.ArgumentParser(description="SFT Training for Curve Trend Classification")
    parser.add_argument("--model-checkpoint", type=str, default=DEFAULT_MODEL_CHECKPOINT)
    parser.add_argument("--lora-r", type=int, default=DEFAULT_LORA_R)
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--lora-dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    parser.add_argument("--train-data-path", type=str, default=DEFAULT_TRAIN_DATA_PATH)
    parser.add_argument("--files-root", type=str, default=DEFAULT_FILES_ROOT)
    parser.add_argument("--prompt-file", type=str, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--per-device-batch-size", type=int, default=DEFAULT_PER_DEVICE_BATCH_SIZE)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=DEFAULT_GRAD_ACCUM)
    parser.add_argument("--num-train-epochs", type=int, default=DEFAULT_TRAIN_EPOCHS)
    parser.add_argument("--logging-steps", type=int, default=DEFAULT_LOGGING_STEPS)
    parser.add_argument("--save-steps", type=int, default=DEFAULT_SAVE_STEPS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--report-to", type=str, default="tensorboard", choices=["tensorboard", "wandb", "none"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train_dataset = build_dataset(args.train_data_path, args.files_root, args.prompt_file)

    base_model, processor = init_model_and_processor(
        model_checkpoint=args.model_checkpoint,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to=[args.report_to] if args.report_to != "none" else [],
        bf16=True,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        # 只在 completion 上算 loss（prompt 部分 -100），避免学 prompt 里的冗余 token
        completion_only_loss=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # 多模态 LoRA DDP：视觉 encoder 冻结、部分分支可能不被激活，容忍 unused params
        ddp_find_unused_parameters=True,
    )

    trainer = SFTTrainer(
        model=base_model,
        processing_class=processor,
        args=training_args,
        train_dataset=train_dataset,
    )

    print(f"Starting SFT training ...")
    print(f"  model: {args.model_checkpoint}")
    print(f"  output: {args.output_dir}")
    print(f"  per-device batch: {args.per_device_batch_size} | grad accum: {args.gradient_accumulation_steps}")
    print(f"  epochs: {args.num_train_epochs} | lr: {args.learning_rate} | max_length: {args.max_length}")
    trainer.train()
    trainer.save_model(args.output_dir)
    print("SFT training done.")
