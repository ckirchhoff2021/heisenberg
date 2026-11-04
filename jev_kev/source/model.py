"""指针式决策模型（自包含，无外部 kev 依赖），支持纯文本与多模态 backbone。

组成：因果语言模型骨干 + 每题独立 causal row + PointerHead 打分。

纯文本路径（kev 原始）：
  - encode()       将 record 打包成 [<state>...][<q>...<opt> o </opt>...<decide>]
  - rows_of()      拆成「state + 每题分支」的 causal rows（1D 位置 id 手工续编）
  - forward_batch  每行独立前向后，用 PointerHead 在 <decide> 与各 </opt> 间打分

多模态路径（本文件新增）：
  - encode() 在 state 中插入 [<vision_start> <image_pad>×N <vision_end>] 块
  - 每行 forward 时传 pixel_values / image_grid_thw / mm_token_type_ids，
    位置编码（M-RoPE 3D）交给 Qwen3_5Model 内部根据 mm_token_type_ids 计算，
    不再手工维护 1D pos。

移除的 serving-only 逻辑：state prefix/KV 缓存、CUDA graphs、MLX 后端、
packed block-causal mask、option isolation、批量 temperature 推理等。
"""
from __future__ import annotations

import math
import re

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3_5Model

# 复用 Qwen 中很少使用的特殊 token 作为分隔符（state/q/opt//opt/decide），
# 无需新增或训练 embedding 行；LoRA 适配它们的含义。
SPECIAL = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]

# 训练上下文上限：state token 数、单题分支 token 数、整条 packed record token 数。
# 多模态 argos 任务 state 很大（分类规则 ~2041 + 图像 ~244），由 train 用 --max-state 覆盖。
MAX_STATE, MAX_BRANCH, MAX_PACKED = 384, 1024, 2048

# 评测时一次前向最多处理的 row token 数。
ROW_PASS_TOKENS = 16384

VISION_START, IMAGE_PAD, VISION_END = "<|vision_start|>", "<|image_pad|>", "<|vision_end|>"


def load_tokenizer(name, revision=None):
    return AutoTokenizer.from_pretrained(name, revision=revision)


def pad_id(tok):
    """右填充 id（从不被 attend）；Qwen tokenizer 自带，其他情况回退到 0。"""
    return tok.pad_token_id if tok.pad_token_id is not None else 0


def is_hybrid(config):
    """config 是否含 Gated DeltaNet 层（Qwen3.5），用于选择 LoRA target。"""
    tc = getattr(config, "text_config", config)
    return "linear_attention" in set(getattr(tc, "layer_types", None) or [])


_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


def user_tokens(tok, text):
    """标记调用方文本，保证无法伪造分隔符 token。

    快分词器忽略 split_special_tokens，所以先把 `<|name|>` 改写成 `<¦name¦>`。
    """
    return tok(_SPECIAL_RE.sub(r"<¦\1¦>", text), add_special_tokens=False).input_ids


class ContextOverflow(ValueError):
    """record 无法在上下文上限内编码。state 超限时附带 state_tokens / max_state。"""

    def __init__(self, message, state_tokens=None, max_state=None):
        super().__init__(message)
        self.state_tokens, self.max_state = state_tokens, max_state


def encode(tok, rec, max_state=MAX_STATE, max_branch=MAX_BRANCH, strict=False,
           image_pad_ids=None):
    """打包一条 record：[<state> ...]（多模态时含 vision 块），随后每题
    [<q> instr <opt> o </opt>... <decide>]。

    返回 ids（packed 序列）、seg（0=state，k=第 k 题）、pos（1D 位置 id，仅纯文本
    路径使用；多模态时由模型内部重算）、decide_idx、opt_idx、mm_token_type_ids
    （多模态时非 None：image_pad 处=1 其余=0）、image_grid_thw（多模态时非 None）。

    多模态：record 可带 "images" 字段（PIL 图像或路径列表），encode 时在 state
    文本后为每张图插入 [<vision_start> <image_pad>×n <vision_end>] 块；image_pad
    数量 n 由调用方通过 image_pad_ids（每张图的 pad token 数列表）传入。

    strict=True 时 state 超过 max_state 抛 ContextOverflow；否则截断到 max_state。
    """
    state_tokens = user_tokens(tok, rec["state"])
    # 多模态：state 文本 + 各图 vision 块
    image_blocks = []
    images = rec.get("images") or []
    image_grid_thw = rec.get("image_grid_thw")
    if images:
        if image_pad_ids is None or len(image_pad_ids) != len(images):
            raise ValueError("multimodal record requires image_pad_ids (one per image)")
        vs, ip, ve = (tok.convert_tokens_to_ids(t) for t in (VISION_START, IMAGE_PAD, VISION_END))
        for n_pad in image_pad_ids:
            image_blocks.append([vs] + [ip] * n_pad + [ve])
    state_len = len(state_tokens) + 1 + sum(len(b) for b in image_blocks)
    if strict and state_len > max_state:
        raise ContextOverflow(
            f"state exceeds {max_state} tokens: {state_len}",
            state_tokens=state_len, max_state=max_state)
    S = [tok.convert_tokens_to_ids(SPECIAL[0])] + state_tokens[: max_state - 1]
    ids = list(S)
    seg = [0] * len(S)
    pos = list(range(len(S)))
    mm = [0] * len(S)
    # 插入图像块（state 文本之后）
    for block in image_blocks:
        ids += block
        seg += [0] * len(block)
        pos += list(range(len(S), len(S) + len(block)))
        # block = [vs, ip×n, ve]：仅 ip 是 image token（type 1）
        mm += [0] + [1] * (len(block) - 2) + [0]

    q_id, o_id, c_id, d_id = (tok.convert_tokens_to_ids(t) for t in SPECIAL[1:])
    decide_idx, opt_idx = [], []
    for k, q in enumerate(rec["questions"], start=1):
        instr = [q_id] + user_tokens(tok, q["instr"])
        spans = [[o_id] + user_tokens(tok, o) + [c_id] for o in q["options"]]
        br = instr + [t for sp in spans for t in sp] + [d_id]
        if len(br) > max_branch - len(S):
            raise ContextOverflow(
                f"branch too long: {len(br)} tokens with a {len(S)}-token state "
                f"(row limit {max_branch})")
        base = len(ids)
        p0 = len(S)
        br_pos = list(range(p0, p0 + len(br)))
        ends, cursor = [], len(instr)
        for sp in spans:
            cursor += len(sp)
            ends.append(cursor - 1)
        ids += br
        seg += [k] * len(br)
        pos += br_pos
        mm += [0] * len(br)
        decide_idx.append(base + len(br) - 1)
        opt_idx.append([base + e for e in ends])
    return {
        "ids": ids, "seg": seg, "pos": pos,
        "decide_idx": decide_idx, "opt_idx": opt_idx,
        "labels": [q["label"] for q in rec["questions"]],
        "state_tokens": state_len,
        "state_truncated": state_len > max_state,
        "mm_token_type_ids": mm if images else None,
        "image_grid_thw": image_grid_thw,
    }


def rows_per_pass(rows, prefix_len=0, budget=ROW_PASS_TOKENS):
    """评测时一次前向能容纳多少条 causal row（按每行 token 数与预算，至少 1）。"""
    return max(1, budget // (prefix_len + max(len(r) for r in rows)))


def rows_of(enc):
    """把 packed 编码拆成 state 与每题分支。

    返回 (state_ids, state_pos, rows)；rows[k] 含分支 ids、从 state 续编的
    pos、<decide> 与各 </opt> 在分支内的下标。多模态 enc 时 rows 额外带
    mm_token_type_ids 切片。
    """
    seg = enc["seg"]
    Ls = seg.count(0)
    mm = enc.get("mm_token_type_ids")
    rows, start = [], Ls
    for k, (d, oi) in enumerate(zip(enc["decide_idx"], enc["opt_idx"]), start=1):
        end = d + 1  # <decide> 是分支最后一个 token
        if seg[start] != k or seg[end - 1] != k:
            raise ValueError("branch layout mismatch")
        row = {
            "ids": enc["ids"][start:end],
            "pos": enc["pos"][start:end],
            "decide": d - start,
            "opts": [o - start for o in oi],
        }
        if mm is not None:
            row["mm"] = mm[:Ls] + mm[start:end]
        rows.append(row)
        start = end
    return enc["ids"][:Ls], enc["pos"][:Ls], rows


class PointerHead(nn.Module):
    def __init__(self, d, dp=256):
        """dp = pointer 维度（分类头容量旋钮）。"""
        super().__init__()
        self.q, self.k = nn.Linear(d, dp), nn.Linear(d, dp)
        self.scale = 1 / math.sqrt(dp)
        # 校准温度：仅 eval 时除；1.0 为原始 logits。argmax 不随温度改变。
        self.temperature = 1.0

    def forward(self, h_decide, h_opts):  # [d], [K,d] -> logits [K]
        z = (self.k(h_opts) @ self.q(h_decide)) * self.scale
        return z if self.training or self.temperature == 1.0 else z / self.temperature


class DecisionModel(nn.Module):
    def __init__(self, name, tok, device, lora=None, revision=None, attn=None,
                 head_dim=256, lora_targets="all", dtype=torch.float32,
                 multimodal=False, image_processor=None):
        """name: 骨干 HF hub id 或本地路径；lora: LoRA rank（None=不加 LoRA）。

        multimodal=True 时用多模态 Qwen3_5Model（visual + language_model），
        否则用 AutoModelForCausalLM().model（纯文本 Qwen3_5TextModel）。
        """
        super().__init__()
        attn = attn or ("sdpa" if str(device).startswith("cuda") else "eager")
        self.multimodal = multimodal
        if multimodal:
            self.lm = Qwen3_5Model.from_pretrained(
                name, revision=revision, dtype=dtype, attn_implementation=attn)
            self.image_processor = image_processor
        else:
            self.lm = AutoModelForCausalLM.from_pretrained(
                name, revision=revision, dtype=dtype, attn_implementation=attn).model
            self.image_processor = None
        self.pad_id = pad_id(tok)
        self.hybrid = is_hybrid(self.lm.config)
        if lora:
            from peft import LoraConfig, get_peft_model
            targets = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]
            if self.hybrid and lora_targets == "all":
                # Gated DeltaNet 投影（Qwen3_5TextModel 的 5 个名字）+ mixer out_proj。
                targets += ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]
            cfg = LoraConfig(
                task_type="FEATURE_EXTRACTION", r=lora, lora_alpha=2 * lora,
                lora_dropout=0.05, target_modules=targets)
            if multimodal:
                # LoRA 只挂在语言模型上（vision tower 模块名不重叠，但显式限定更稳）。
                self.lm.language_model = get_peft_model(self.lm.language_model, cfg)
                # 冻结 vision tower：只训 LoRA + head。否则 vision tower 会被 AdamW 全量
                # 更新（100M 参数），而 save_checkpoint 只存 language_model 的 LoRA + head，
                # 导致训练效果（图像特征判别力）全部丢在未保存的 vision tower 里。
                for p in self.lm.visual.parameters():
                    p.requires_grad = False
            else:
                self.lm = get_peft_model(self.lm, cfg)
        # PointerHead 输入维度 = 语言模型 hidden_size（多模态时取 text_config.hidden_size）
        tc = getattr(self.lm.config, "text_config", self.lm.config)
        self.head = PointerHead(tc.hidden_size, dp=head_dim)
        self.device = device
        self.to(device)

    def encode(self, tok, rec, **kw):
        """record -> packed 编码。多模态时自动计算 image_pad_ids 与 pixel_values。"""
        images = rec.get("images") or []
        image_pad_ids = None
        pixel_values = None
        image_grid_thw = None
        if images and self.multimodal:
            from PIL import Image
            imgs = [im.convert("RGB") if isinstance(im, Image.Image) else Image.open(im).convert("RGB")
                    for im in images]
            ip = self.image_processor
            mp = kw.pop("max_pixels", None)
            feat = ip(images=imgs, return_tensors="pt",
                      **({"max_pixels": mp} if mp is not None else {}))
            merge = getattr(ip, "merge_size", 1) or 1
            image_grid_thw = feat["image_grid_thw"]
            pixel_values = feat["pixel_values"]
            image_pad_ids = [int(g.prod() // (merge ** 2)) for g in image_grid_thw]
            rec = {**rec, "image_grid_thw": image_grid_thw}
        enc = encode(tok, rec, image_pad_ids=image_pad_ids, **kw)
        if pixel_values is not None:
            enc["pixel_values"] = pixel_values
        return enc

    def _pad_rows(self, rows):
        """causal rows 右填充成 [N,L] id/position tensor 与 [N,L] mask（1=真实 token）。
        填充位作为 key 被 mask，不改变真实 token 的隐状态。多模态时额外填充
        mm_token_type_ids。rows 元素为 dict：{ids, pos, mm?}。
        """
        L = max(len(r["ids"]) for r in rows)
        ids = torch.full((len(rows), L), self.pad_id, device=self.device)
        pos = torch.zeros((len(rows), L), dtype=torch.long, device=self.device)
        att = torch.zeros((len(rows), L), dtype=torch.long, device=self.device)
        mm = torch.zeros((len(rows), L), dtype=torch.long, device=self.device)
        has_mm = any(r.get("mm") is not None for r in rows)
        for i, r in enumerate(rows):
            ids[i, :len(r["ids"])] = torch.tensor(r["ids"], device=self.device)
            pos[i, :len(r["pos"])] = torch.tensor(r["pos"], device=self.device)
            att[i, :len(r["ids"])] = 1
            if r.get("mm") is not None:
                mm[i, :len(r["ids"])] = torch.tensor(r["mm"], device=self.device)
        return ids, pos, att, (mm if has_mm else None)

    def _rows_hidden(self, rows):
        """逐 causal row 前向，每条返回 [L_i,d] 隐状态。

        训练一次跑完整 batch（autograd 需要完整图）；评测按 rows_per_pass 分块。
        rows 元素为 dict：{ids, pos, mm?, pixel_values?, image_grid_thw?}。
        纯文本：一次批量前向（ids/pos 已填充）。
        多模态：批量前向，pixel_values/image_grid_thw 按 row 顺序拼接（每个 row
        携带自己 record 的图像副本，对齐 dataset.jev 的 collate 逻辑）。
        """
        chunk = len(rows) if self.training else rows_per_pass([r["ids"] for r in rows])
        out = []
        for start in range(0, len(rows), chunk):
            part = rows[start:start + chunk]
            ids, pos, att, mm = self._pad_rows(part)
            if self.multimodal:
                kw = {"mm_token_type_ids": mm}
                pvs, grids = [], []
                for r in part:
                    if r.get("pixel_values") is not None:
                        pvs.append(r["pixel_values"])
                        grids.append(r["image_grid_thw"])
                if pvs:
                    kw["pixel_values"] = torch.cat(pvs, dim=0).to(ids.device)
                    kw["image_grid_thw"] = torch.cat(grids, dim=0).to(ids.device)
                h = self.lm(
                    input_ids=ids, attention_mask=att, use_cache=False, **kw,
                ).last_hidden_state.float()
            else:
                h = self.lm(
                    input_ids=ids, position_ids=pos, attention_mask=att
                ).last_hidden_state.float()  # 分类头保持 fp32
            out += [h[i, :len(r["ids"])] for i, r in enumerate(part)]
        return out

    def forward_batch(self, encs, shared_prefix=False):
        """返回每 record、每题的 logits。

        统一走 causal row：state 与每题分支拼成独立行，行间互不影响。
        shared_prefix 仅为兼容旧调用签名（本模块不做 serving prefix 复用）。
        """
        del shared_prefix
        rows, readouts = [], []
        for b, e in enumerate(encs):
            S, Sp, brs = rows_of(e)
            for r in brs:
                row = {"ids": S + r["ids"], "pos": Sp + r["pos"]}
                if r.get("mm") is not None:
                    row["mm"] = r["mm"]
                if e.get("pixel_values") is not None:
                    row["pixel_values"] = e["pixel_values"]
                    row["image_grid_thw"] = e["image_grid_thw"]
                rows.append(row)
                readouts.append((b, len(S) + r["decide"],
                                 [len(S) + o for o in r["opts"]]))
        out = [[] for _ in encs]
        for h, (b, d, oi) in zip(self._rows_hidden(rows), readouts):
            out[b].append(self.head(h[d], h[torch.tensor(oi, device=self.device)]))
        return out

    def forward(self, encs):
        """encs：单条 record 的编码 dict，或编码 dict 的列表。

        DDP 兼容：DDP 包装后 `model(encs)` 会走此 forward，梯度正确 all-reduce。
        返回每 record、每题的 logits 列表（单条输入时返回 logits 列表的列表）。
        """
        single = isinstance(encs, dict)
        out = self.forward_batch([encs] if single else encs, shared_prefix=False)
        return out[0] if single else out

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


def build_model(backbone: str, device: str, lora: int = 16, head_dim: int = 256,
                revision: str | None = None, multimodal: bool = False,
                image_processor=None) -> DecisionModel:
    """构建 jev_kev 模型：骨干 + LoRA + 从零初始化的 PointerHead。"""
    tok = load_tokenizer(backbone, revision=revision)
    model = DecisionModel(
        backbone, tok, device, lora=lora, revision=revision, head_dim=head_dim,
        dtype=torch.bfloat16, multimodal=multimodal, image_processor=image_processor)
    model.lm.config.use_cache = False
    return model
