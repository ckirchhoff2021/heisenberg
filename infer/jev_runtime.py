"""Minimal optimized JEV inference runtime.

This module intentionally depends only on PyTorch, Transformers, Pillow, and
``models.jev`` so it can be copied into a small serving image without training
or dataset code.
"""
from __future__ import annotations

import io
import os
import time
from typing import Sequence

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer

from models.jev import JevModel

CURVE_CHOICES = ["normal", "offset", "increase", "decline", "spike", "fluctuate", "other"]
RECOVER_CHOICES = ["recovered", "recovering", "no_recover", "other"]
CURVE_QUESTION = "对比参考曲线，当前曲线的趋势（trend）属于哪一类？"
RECOVER_QUESTION = "当前曲线相对基准水位的异常恢复状态（recover）属于哪一类？"

VISION_START_TOKEN = "<|vision_start|>"
IMAGE_PAD_TOKEN = "<|image_pad|>"
VISION_END_TOKEN = "<|vision_end|>"


def load_rule_prompt(path: str) -> str:
    with open(path, encoding="utf-8") as file:
        return file.read().split("# 输出格式")[0].strip()


def longest_common_prefix(sequences: Sequence[Sequence[int]]) -> list[int]:
    if not sequences:
        return []
    limit = min(map(len, sequences))
    end = 0
    while end < limit and all(sequence[end] == sequences[0][end] for sequence in sequences[1:]):
        end += 1
    return list(sequences[0][:end])


class FastJevInference:
    """JEV inference with one vision pass and one cached prefix per question."""

    def __init__(
        self,
        checkpoint: str,
        backbone: str | None = None,
        device: str | None = None,
        prompt_file: str | None = None,
    ):
        if not checkpoint or not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"JEV checkpoint not found: {checkpoint}")

        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.backbone = backbone or state.get("backbone")
        if not self.backbone or not os.path.isdir(self.backbone):
            raise FileNotFoundError(f"JEV backbone not found: {self.backbone}")

        self.max_pixels = int(state.get("max_pixels", 512 * 512))
        self.max_tokens = int(state.get("max_tokens", 3072))
        self.step = state.get("step")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {self.device}")
        self.amp_dtype = torch.bfloat16
        self.last_latency_ms: float | None = None

        print(
            f"[JEV] checkpoint={checkpoint} step={self.step} "
            f"backbone={self.backbone} device={self.device}",
            flush=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone, local_files_only=True)
        self.image_processor = AutoImageProcessor.from_pretrained(
            self.backbone,
            local_files_only=True,
        )
        self.model = JevModel(self.backbone, dtype=self.amp_dtype).to(self.device)
        self.model.load_state_dict(state["state_dict"])
        self.model.eval()
        del state

        self.system = load_rule_prompt(prompt_file) if prompt_file else ""
        self.merge_size = getattr(self.image_processor, "merge_size", 1)
        self.vision_start = self.tokenizer.convert_tokens_to_ids(VISION_START_TOKEN)
        self.image_pad = self.tokenizer.convert_tokens_to_ids(IMAGE_PAD_TOKEN)
        self.vision_end = self.tokenizer.convert_tokens_to_ids(VISION_END_TOKEN)
        self._state_ids = self.tokenizer.encode("[STATE]", add_special_tokens=False)
        if self.system:
            self._state_ids += self.tokenizer.encode(" " + self.system, add_special_tokens=False)
        self._templates = [
            self._build_template(CURVE_QUESTION, CURVE_CHOICES),
            self._build_template(RECOVER_QUESTION, RECOVER_CHOICES),
        ]

    def _build_template(self, question: str, choices: Sequence[str]) -> dict:
        question_ids = self.tokenizer.encode(
            "\n[QUESTION] " + question,
            add_special_tokens=False,
        )
        suffixes = [
            self.tokenizer.encode("\n[CANDIDATE] " + choice, add_special_tokens=False)
            for choice in choices
        ]
        common_ids = longest_common_prefix(suffixes)
        candidate_tails = [suffix[len(common_ids):] for suffix in suffixes]
        if not common_ids or any(not tail for tail in candidate_tails):
            raise ValueError("candidate labels must have a shared prefix and unique token tail")
        if any(common_ids + tail != suffix for suffix, tail in zip(suffixes, candidate_tails)):
            raise RuntimeError("candidate token split changed the original path")
        return {
            "question_ids": question_ids,
            "candidate_common_ids": common_ids,
            "candidate_tails": candidate_tails,
        }

    def _prepare_image(
        self,
        image: Image.Image,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        feature = self.image_processor(
            images=[image.convert("RGB")],
            return_tensors="pt",
            max_pixels=self.max_pixels,
        )
        pixel_values = feature["pixel_values"].to(self.device)
        image_grid_thw = feature["image_grid_thw"].to(self.device)
        if image_grid_thw.shape != (1, 3):
            raise ValueError(f"expected one image grid, got {tuple(image_grid_thw.shape)}")

        n_pad = int(image_grid_thw[0].prod().item() // (self.merge_size**2))
        image_block = [self.vision_start] + [self.image_pad] * n_pad + [self.vision_end]
        return pixel_values, image_grid_thw, image_block

    @staticmethod
    def _continuation_position_ids(
        attention_mask: torch.Tensor,
        prefix_length: int,
        rope_delta: torch.Tensor,
    ) -> torch.Tensor:
        positions = attention_mask.long().cumsum(-1) - 1 + prefix_length
        positions = positions.masked_fill(attention_mask == 0, 0)

        delta = rope_delta.reshape(rope_delta.shape[0], -1)
        if delta.shape[1] != 1:
            raise ValueError(f"unexpected rope_delta shape: {tuple(rope_delta.shape)}")
        if delta.shape[0] == 1 and positions.shape[0] > 1:
            delta = delta.expand(positions.shape[0], -1)
        if delta.shape[0] != positions.shape[0]:
            raise ValueError(
                f"rope_delta batch {delta.shape[0]} != candidate batch {positions.shape[0]}"
            )
        positions = positions + delta.to(device=positions.device, dtype=positions.dtype)
        return positions.unsqueeze(0).expand(3, -1, -1)

    def _encode_question(
        self,
        template: dict,
        image_block: Sequence[int],
        image_grid_thw: torch.Tensor,
        vision_outputs,
    ) -> torch.Tensor:
        backbone = self.model.path_encoder.backbone
        prefix_ids = (
            self._state_ids
            + list(image_block)
            + template["question_ids"]
            + template["candidate_common_ids"]
        )
        tails = template["candidate_tails"]
        if any(len(prefix_ids) + len(tail) > self.max_tokens for tail in tails):
            raise ValueError(f"a path exceeds {self.max_tokens} tokens")

        prefix = torch.tensor([prefix_ids], dtype=torch.long, device=self.device)
        prefix_out = backbone(
            input_ids=prefix,
            attention_mask=torch.ones_like(prefix),
            use_cache=True,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=(prefix == self.image_pad).long(),
            mm_encoder_outputs={"image": vision_outputs},
        )
        if prefix_out.past_key_values is None or prefix_out.rope_deltas is None:
            raise RuntimeError("Qwen3.5 did not return the prefix cache or M-RoPE delta")

        candidate_count = len(tails)
        cache = prefix_out.past_key_values
        cache.reorder_cache(
            torch.zeros(candidate_count, dtype=torch.long, device=self.device)
        )

        tail_length = max(map(len, tails))
        tail_ids = torch.full(
            (candidate_count, tail_length),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        tail_mask = torch.zeros_like(tail_ids)
        for index, tail in enumerate(tails):
            tail_ids[index, : len(tail)] = torch.tensor(tail, device=self.device)
            tail_mask[index, : len(tail)] = 1

        full_attention_mask = torch.cat(
            [
                torch.ones(
                    candidate_count,
                    len(prefix_ids),
                    dtype=tail_mask.dtype,
                    device=self.device,
                ),
                tail_mask,
            ],
            dim=1,
        )
        tail_out = backbone(
            input_ids=tail_ids,
            attention_mask=full_attention_mask,
            position_ids=self._continuation_position_ids(
                tail_mask,
                len(prefix_ids),
                prefix_out.rope_deltas,
            ),
            past_key_values=cache,
            use_cache=False,
        )
        end_positions = tail_mask.sum(-1) - 1
        return tail_out.last_hidden_state[
            torch.arange(candidate_count, device=self.device),
            end_positions,
        ]

    @staticmethod
    def _answer(probabilities: torch.Tensor, choices: Sequence[str]) -> dict:
        best = int(probabilities.argmax())
        return {
            "label": choices[best],
            "index": best,
            "prob": float(probabilities[best]),
            "probs": {
                choice: float(probabilities[index])
                for index, choice in enumerate(choices)
            },
        }

    @torch.inference_mode()
    def _predict(self, image: Image.Image) -> dict:
        pixel_values, image_grid_thw, image_block = self._prepare_image(image)
        device_type = "cuda" if self.device.startswith("cuda") else "cpu"
        with torch.autocast(device_type=device_type, dtype=self.amp_dtype):
            backbone = self.model.path_encoder.backbone
            vision_outputs = backbone.get_image_features(
                pixel_values,
                image_grid_thw,
                return_dict=True,
            )
            vectors = [
                self._encode_question(
                    template,
                    image_block,
                    image_grid_thw,
                    vision_outputs,
                )
                for template in self._templates
            ]

            candidate_mask = torch.zeros(
                2,
                len(CURVE_CHOICES),
                dtype=torch.bool,
                device=self.device,
            )
            candidate_vectors = torch.zeros(
                2,
                len(CURVE_CHOICES),
                self.model.hidden_size,
                dtype=vectors[0].dtype,
                device=self.device,
            )
            for question_index, question_vectors in enumerate(vectors):
                count = question_vectors.shape[0]
                candidate_mask[question_index, :count] = True
                candidate_vectors[question_index, :count] = question_vectors
            logits = self.model._score(candidate_vectors, candidate_mask).float()

        probabilities = torch.softmax(
            logits.masked_fill(~candidate_mask, float("-inf")),
            dim=-1,
        ).cpu()
        return {
            "trend": self._answer(probabilities[0, : len(CURVE_CHOICES)], CURVE_CHOICES),
            "recover": self._answer(
                probabilities[1, : len(RECOVER_CHOICES)],
                RECOVER_CHOICES,
            ),
        }

    def predict_image(self, image_path: str) -> dict:
        started = time.perf_counter()
        with Image.open(image_path) as image:
            result = self._predict(image)
        self.last_latency_ms = (time.perf_counter() - started) * 1000
        return result

    def predict_bytes(self, image_bytes: bytes) -> dict:
        started = time.perf_counter()
        with Image.open(io.BytesIO(image_bytes)) as image:
            result = self._predict(image)
        self.last_latency_ms = (time.perf_counter() - started) * 1000
        return result
