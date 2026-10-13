"""
JEV 选择题数据集
标准格式: system（任务说明） + state（状态，可含图像） + question（问题）
        + choices（选项） + answer（正确答案索引）
支持纯文本和多模态（单图/多图）；并提供从 argos 标签数据
(curve/recover/image/from) 直接构造选择题的转换入口。
"""
import os
import json
from typing import List, Dict
import torch
from PIL import Image
from datasets import Dataset

DEFAULT_FILES_ROOT = "datas/aiops/files"
DEFAULT_PROMPT_FILE = "datas/aiops/files/prompts/argos-dp-prompt.md"
DEFAULT_MAX_PIXELS = 512 * 512  # 曲线图降采样上限，控制 vision token 数
DEFAULT_MAX_TOKENS = 3072

# 与 argos-dp-prompt.md 的输出空间一致（含 other），顺序对齐
CURVE_CHOICES = ["normal", "offset", "increase", "decline", "spike", "fluctuate", "other"]
RECOVER_CHOICES = ["recovered", "recovering", "no_recover", "other"]
CURVE_QUESTION = "对比参考曲线，当前曲线的趋势（trend）属于哪一类？"
RECOVER_QUESTION = "当前曲线相对基准水位的异常恢复状态（recover）属于哪一类？"

VISION_START_TOKEN = "<|vision_start|>"
IMAGE_PAD_TOKEN = "<|image_pad|>"
VISION_END_TOKEN = "<|vision_end|>"


class JEVDatasets:
    """
    SystemOne JEV 格式数据集加载器
    输入JSON标准格式要求：
    [
        {
            "system": "<system_prompt>",
            "state": {"images": ["<image_path>"], "from": "<image_subdir>"},
            "question": "请问当前曲线的趋势属于哪一类？",  
            "choices": ["spike", "normal", "increase", "decline"],
            "answer": 2,                                   
        },
        ...
    ]
    """

    def __init__(
        self,
        data_files: str,
        files_root: str = DEFAULT_FILES_ROOT,
        check_images: bool = False,
        answer_format: str = "index"  # 输出answer格式：index=索引 / content=选项内容
    ):
        """
        Args:
            data_files: JEV格式JSON数据集路径
            files_root: 图像文件根目录
            check_images: 是否检查图像文件存在
            answer_format: 输出答案格式，可选index/content
        """
        self.data_path = data_files
        self.files_root = files_root
        self.check_images = check_images
        self.answer_format = answer_format
        
        if self.answer_format not in ["index", "content"]:
            raise ValueError("answer_format must be one of 'index' or 'content'")

    def _resolve_image_paths(self, state: dict) -> List[str]:
        """解析状态中的图像路径，返回绝对路径列表"""
        image_paths = []
        if "images" not in state or not state["images"]:
            return image_paths
        
        image_subdir = state.get("from", "")
        for img_name in state["images"]:
            img_path = os.path.join(self.files_root, image_subdir, img_name)
            if self.check_images and not os.path.isfile(img_path):
                raise FileNotFoundError(f"Image not found: {img_path}")
            image_paths.append(img_path)

        return image_paths

    def _to_example(self, record: dict) -> dict:
        """将单条记录转为 JEV 标准训练格式"""
        required_fields = ["state", "question", "choices", "answer"]
        for field in required_fields:
            if field not in record:
                raise ValueError(f"Missing required field: {field}")
        
        if len(record["choices"]) < 2:
            raise ValueError(f"Choices must have at least 2 options, got {len(record['choices'])}")
        
        example = {}

        # 任务说明（可选）
        system = str(record.get("system") or "").strip()
        if system:
            example["system"] = system

        # 处理状态
        example["state_text"] = record["state"].get("text", "")
        example["state_images"] = self._resolve_image_paths(record["state"])
        if example["state_images"]:
            example["images"] = example["state_images"]  # 兼容TRL多模态字段
        
        # 处理问题和选项
        example["question"] = record["question"].strip()
        example["choices"] = [c.strip() for c in record["choices"]]
        
        # 处理答案
        if self.answer_format == "content":
            if isinstance(record["answer"], int):
                example["answer"] = example["choices"][record["answer"]]
            else:
                example["answer"] = str(record["answer"]).strip()
        else:  # index格式
            if isinstance(record["answer"], int):
                if record["answer"] < 0 or record["answer"] >= len(example["choices"]):
                    raise ValueError(f"Answer index {record['answer']} out of range for {len(example['choices'])} choices")
                example["answer"] = record["answer"]
            else:
                # 答案是内容，转为索引
                answer_str = str(record["answer"]).strip()
                try:
                    example["answer"] = example["choices"].index(answer_str)
                except ValueError:
                    raise ValueError(f"Answer '{answer_str}' not found in choices: {example['choices']}")
        
        # 保留元信息
        if "meta" in record:
            example["meta"] = record["meta"]
       
        return example

    def build(self, records=None) -> Dataset:
        """构建 HuggingFace Dataset；records 给定时直接使用，否则读 data_files。"""
        if records is None:
            with open(self.data_path, encoding="utf-8") as f:
                records = json.load(f)

        valid_records = []
        for idx, r in enumerate(records):
            try:
                valid_records.append(self._to_example(r))
            except Exception as e:
                print(f"Skip record #{idx}: {str(e)}")
                continue

        print(f"Loaded {len(valid_records)} valid JEV samples, skipped {len(records)-len(valid_records)} invalid samples")
        return Dataset.from_list(valid_records)


class JevTorchDataset(torch.utils.data.Dataset):
    """把 ``JEVDatasets.build()`` 的选择题行编码成多模态候选路径。

    每个样本（一道选择题）产出:
      paths          : List[List[int]]，每个候选一条 token 序列
      target         : List[float]，长度=选项数（one-hot）
      pixel_values   : torch.Tensor [n_pad, C]，无图时为 None
      image_grid_thw : torch.Tensor [n_img, 3]，无图时为 None
    需要 JEVDatasets 以 answer_format="index" 构建（answer 为整数索引）。
    """

    def __init__(
        self,
        hf_dataset,
        tokenizer,
        image_processor=None,
        max_tokens: int = 2048,
        max_pixels: int = DEFAULT_MAX_PIXELS,
    ):
        self.tok = tokenizer
        self.img_proc = image_processor
        self.max_tokens = max_tokens
        self.max_pixels = max_pixels
        self.merge_size = getattr(image_processor, "merge_size", 1) if image_processor else 1
        self.pad_id = tokenizer.pad_token_id
        self.vision_start = tokenizer.convert_tokens_to_ids(VISION_START_TOKEN)
        self.image_pad = tokenizer.convert_tokens_to_ids(IMAGE_PAD_TOKEN)
        self.vision_end = tokenizer.convert_tokens_to_ids(VISION_END_TOKEN)
        self.samples = [self._encode(row) for row in hf_dataset]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]

    def _encode(self, row: dict) -> dict:
        system = row.get("system") or ""
        state_text = row.get("state_text") or ""
        image_paths = row.get("state_images") or []
        choices = row["choices"]
        try:
            answer = int(row["answer"])
        except (TypeError, ValueError):
            raise ValueError("answer must be an integer index; build JEVDatasets with answer_format='index'")

        pixel_values = None
        image_grid_thw = None
        image_blocks = []
        if image_paths and self.img_proc is not None:
            images = [Image.open(p).convert("RGB") for p in image_paths]
            feat = self.img_proc(images=images, return_tensors="pt", max_pixels=self.max_pixels)
            pixel_values = feat["pixel_values"]
            image_grid_thw = feat["image_grid_thw"]
            for grid in image_grid_thw:
                # image_grid_thw 是合并前网格；视觉塔内部做 merge×merge 合并。
                n_pad = int(grid.prod() // (self.merge_size ** 2))
                image_blocks.append([self.vision_start] + [self.image_pad] * n_pad + [self.vision_end])

        prefix = self.tok.encode("[STATE]", add_special_tokens=False)
        if system:
            prefix += self.tok.encode(" " + system, add_special_tokens=False)
        for block in image_blocks:
            prefix += block
        if state_text:
            prefix += self.tok.encode(" " + state_text, add_special_tokens=False)
        prefix += self.tok.encode("\n[QUESTION] " + row["question"], add_special_tokens=False)

        paths = [
            prefix + self.tok.encode("\n[CANDIDATE] " + choice, add_special_tokens=False)
            for choice in choices
        ]
        if any(len(p) > self.max_tokens for p in paths):
            raise ValueError(f"a path exceeds {self.max_tokens} tokens; shorten the input (nothing truncated)")

        target = [0.0] * len(choices)
        target[answer] = 1.0
        return {"paths": paths, "target": target,
                "pixel_values": pixel_values, "image_grid_thw": image_grid_thw}

    def collate(self, batch: List[dict]) -> dict:
        all_paths = [seq for s in batch for seq in s["paths"]]
        max_len = max(map(len, all_paths))
        cmax = max(len(s["target"]) for s in batch)

        input_ids = torch.full((len(all_paths), max_len), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        cand_mask = torch.zeros(len(batch), cmax, dtype=torch.bool)
        target = torch.zeros(len(batch), cmax)

        pv_parts, grid_parts = [], []
        q_index, cand_index, cand_end_pos = [], [], []
        p = 0
        for qi, s in enumerate(batch):
            n = len(s["target"])
            cand_mask[qi, :n] = True
            target[qi, :n] = torch.tensor(s["target"])
            for ci, seq in enumerate(s["paths"]):
                input_ids[p, :len(seq)] = torch.tensor(seq)
                attention_mask[p, :len(seq)] = 1
                q_index.append(qi)
                cand_index.append(ci)
                cand_end_pos.append(len(seq) - 1)
                # State images are shared by all candidates, so repeat their
                # tensors once per candidate path.
                if s["pixel_values"] is not None:
                    pv_parts.append(s["pixel_values"])
                    grid_parts.append(s["image_grid_thw"])
                p += 1

        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mm_token_type_ids": (input_ids == self.image_pad).long(),
            "cand_mask": cand_mask,
            "target": target,
            "q_index": torch.tensor(q_index),
            "cand_index": torch.tensor(cand_index),
            "cand_end_pos": torch.tensor(cand_end_pos),
        }
        if pv_parts:
            out["pixel_values"] = torch.cat(pv_parts, dim=0)
            out["image_grid_thw"] = torch.cat(grid_parts, dim=0)
        return out


def load_rule_prompt(path: str) -> str:
    """读取判别式所需的任务/分类规则：去掉生成式专用的“输出格式”及之后。"""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    return text.split("# 输出格式")[0].strip()


def argos_to_jev_records(rows, system: str = "") -> List[dict]:
    """argos (curve/recover/image/from) 记录 → 标准 JEV 选择题记录。

    每条 argos 记录（一张图）产出 trend、recover 两道选择题。
    """
    records = []
    for r in rows:
        state = {"images": [r["image"]], "from": r["from"]}
        records += [
            {"system": system, "state": state, "question": CURVE_QUESTION,
             "choices": CURVE_CHOICES, "answer": CURVE_CHOICES.index(r["curve"])},
            {"system": system, "state": state, "question": RECOVER_QUESTION,
             "choices": RECOVER_CHOICES, "answer": RECOVER_CHOICES.index(r["recover"])},
        ]
    return records


def build_argos_jev_dataset(
    argos_path: str,
    files_root: str,
    tokenizer,
    image_processor,
    prompt_file: str = "",
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    limit=None,
) -> JevTorchDataset:
    """从 argos 标签 json 一键构建多模态 JevTorchDataset。"""
    with open(argos_path, encoding="utf-8") as f:
        rows = json.load(f)
    if limit is not None:
        rows = rows[:limit]
    system = load_rule_prompt(prompt_file) if prompt_file else ""
    records = argos_to_jev_records(rows, system)
    raw = JEVDatasets(argos_path, files_root=files_root).build(records)
    return JevTorchDataset(raw, tokenizer, image_processor,
                           max_tokens=max_tokens, max_pixels=max_pixels)


if __name__ == "__main__":
    from transformers import AutoImageProcessor, AutoTokenizer

    ckpt = "checkpoints/Qwen3.5-0.8B"
    tokenizer = AutoTokenizer.from_pretrained(ckpt, local_files_only=True)
    image_processor = AutoImageProcessor.from_pretrained(ckpt, local_files_only=True)

    ds = build_argos_jev_dataset(
        "datas/aiops/files/jsons/argos-trains.json",
        files_root=DEFAULT_FILES_ROOT,
        tokenizer=tokenizer,
        image_processor=image_processor,
        prompt_file=DEFAULT_PROMPT_FILE,
        limit=4,  # 每张图产出 trend/recover 两题，共 8 条
    )
    print(f"dataset size (questions): {len(ds)}")
    for i in (0, 1):  # 同一张图的 trend(7候选) 与 recover(4候选)
        s = ds[i]
        print(s)
        print(f"sample {i}: paths={len(s['paths'])} target={s['target']}")
        break