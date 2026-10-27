"""指针式决策模型（自包含，无外部 kev 依赖）。

组成：因果语言模型骨干 + 每题独立 causal row + PointerHead 打分。

代码摘编自 kev（github.com/jaredpalmer/kev）的 model.py，只保留 jev_kev
训练/评测实际使用的路径：
  - encode()       将 record 打包成 [<state>...][<q>...<opt> o </opt>...<decide>]
  - rows_of()      拆成「state + 每题分支」的 causal rows
  - forward_batch  每行独立前向后，用 PointerHead 在 <decide> 与各 </opt> 间打分

移除的 serving-only 逻辑：state prefix/KV 缓存、CUDA graphs、MLX 后端、
packed block-causal mask、option isolation、批量 temperature 推理等。
"""
from __future__ import annotations

import math
import re

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# 复用 Qwen 中很少使用的特殊 token 作为分隔符（state/q/opt//opt/decide），
# 无需新增或训练 embedding 行；LoRA 适配它们的含义。
SPECIAL = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]

# 训练上下文上限：state token 数、单题分支 token 数、整条 packed record token 数。
MAX_STATE, MAX_BRANCH, MAX_PACKED = 384, 1024, 2048

# 评测时一次前向最多处理的 row token 数。
ROW_PASS_TOKENS = 16384


def load_tokenizer(name, revision=None):
    return AutoTokenizer.from_pretrained(name, revision=revision)


def pad_id(tok):
    """右填充 id（从不被 attend）；Qwen tokenizer 自带，其他情况回退到 0。"""
    return tok.pad_token_id if tok.pad_token_id is not None else 0


def is_hybrid(config):
    """config 是否含 Gated DeltaNet 层（Qwen3.5），用于选择 LoRA target。"""
    return "linear_attention" in set(getattr(config, "layer_types", None) or [])


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


def encode(tok, rec, max_state=MAX_STATE, max_branch=MAX_BRANCH, strict=False):
    """打包一条 record：[<state> ...]，随后每题 [<q> instr <opt> o </opt>... <decide>]。

    返回 ids（packed 序列）、seg（0=state，k=第 k 题）、pos（位置 id，
    state 后续编）、decide_idx（每题 <decide> 全局下标）、opt_idx（每题
    各 </opt> 全局下标）。

    strict=True 时 state 超过 max_state 抛 ContextOverflow；否则截断到 max_state。
    """
    state_tokens = user_tokens(tok, rec["state"])
    if strict and len(state_tokens) + 1 > max_state:
        raise ContextOverflow(
            f"state exceeds {max_state} tokens: {len(state_tokens) + 1}",
            state_tokens=len(state_tokens) + 1, max_state=max_state)
    S = [tok.convert_tokens_to_ids(SPECIAL[0])] + state_tokens[: max_state - 1]
    ids = list(S)
    seg = [0] * len(S)
    pos = list(range(len(S)))
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
        decide_idx.append(base + len(br) - 1)
        opt_idx.append([base + e for e in ends])
    return {
        "ids": ids, "seg": seg, "pos": pos,
        "decide_idx": decide_idx, "opt_idx": opt_idx,
        "labels": [q["label"] for q in rec["questions"]],
        "state_tokens": len(state_tokens) + 1,
        "state_truncated": len(state_tokens) + 1 > max_state,
    }


def rows_per_pass(rows, prefix_len=0, budget=ROW_PASS_TOKENS):
    """评测时一次前向能容纳多少条 causal row（按每行 token 数与预算，至少 1）。"""
    return max(1, budget // (prefix_len + max(len(r) for r in rows)))


def rows_of(enc):
    """把 packed 编码拆成 state 与每题分支。

    返回 (state_ids, state_pos, rows)；rows[k] 含分支 ids、从 state 续编的
    pos、<decide> 与各 </opt> 在分支内的下标。

    「state + rows[k]」作为一条 causal row 与 packed 形式对第 k 题等价：
    row 中恰好是该题可见的 token，位置也一致，任何骨干架构都成立。
    """
    seg = enc["seg"]
    Ls = seg.count(0)
    rows, start = [], Ls
    for k, (d, oi) in enumerate(zip(enc["decide_idx"], enc["opt_idx"]), start=1):
        end = d + 1  # <decide> 是分支最后一个 token
        if seg[start] != k or seg[end - 1] != k:
            raise ValueError("branch layout mismatch")
        rows.append({
            "ids": enc["ids"][start:end],
            "pos": enc["pos"][start:end],
            "decide": d - start,
            "opts": [o - start for o in oi],
        })
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
                 head_dim=256, lora_targets="all", dtype=torch.float32):
        """name: 骨干 HF hub id 或本地路径；lora: LoRA rank（None=不加 LoRA）。"""
        super().__init__()
        # 只取骨干（无 LM 头）：本模型从不生成文本。CUDA 用 SDPA，其余 eager。
        attn = attn or ("sdpa" if str(device).startswith("cuda") else "eager")
        self.lm = AutoModelForCausalLM.from_pretrained(
            name, revision=revision, dtype=dtype, attn_implementation=attn).model
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
            self.lm = get_peft_model(self.lm, cfg)
        self.head = PointerHead(self.lm.config.hidden_size, dp=head_dim)
        self.device = device
        self.to(device)

    def encode(self, tok, rec, **kw):
        """record -> packed 编码。"""
        return encode(tok, rec, **kw)

    def _pad_rows(self, rows):
        """causal rows 右填充成 [N,L] id/position tensor 与 [N,L] mask（1=真实 token）。
        填充位作为 key 被 mask，不改变真实 token 的隐状态。"""
        L = max(len(ids) for ids, _ in rows)
        ids = torch.full((len(rows), L), self.pad_id, device=self.device)
        pos = torch.zeros((len(rows), L), dtype=torch.long, device=self.device)
        att = torch.zeros((len(rows), L), dtype=torch.long, device=self.device)
        for i, (rid, rpos) in enumerate(rows):
            ids[i, :len(rid)] = torch.tensor(rid, device=self.device)
            pos[i, :len(rpos)] = torch.tensor(rpos, device=self.device)
            att[i, :len(rid)] = 1
        return ids, pos, att

    def _rows_hidden(self, rows):
        """逐 causal row 前向，每条返回 [L_i,d] 隐状态。
        训练一次跑完整 batch（autograd 需要完整图）；评测按 rows_per_pass 分块。"""
        chunk = len(rows) if self.training else rows_per_pass([r[0] for r in rows])
        out = []
        for start in range(0, len(rows), chunk):
            part = rows[start:start + chunk]
            ids, pos, att = self._pad_rows(part)
            h = self.lm(
                input_ids=ids, position_ids=pos, attention_mask=att
            ).last_hidden_state.float()  # 分类头保持 fp32
            out += [h[i, :len(rid)] for i, (rid, _) in enumerate(part)]
        return out

    def forward_batch(self, encs, shared_prefix=False):
        """返回每 record、每题的 logits。

        统一走 causal row：state 与每题分支拼成独立行，行间互不影响。
        shared_prefix 仅为兼容旧调用签名（本模块不做 serving prefix 复用）。"""
        del shared_prefix
        rows, readouts = [], []
        for b, e in enumerate(encs):
            S, Sp, brs = rows_of(e)
            for r in brs:
                rows.append((S + r["ids"], Sp + r["pos"]))
                readouts.append((b, len(S) + r["decide"],
                                 [len(S) + o for o in r["opts"]]))
        out = [[] for _ in encs]
        for h, (b, d, oi) in zip(self._rows_hidden(rows), readouts):
            out[b].append(self.head(h[d], h[torch.tensor(oi, device=self.device)]))
        return out

    def forward(self, enc):
        """单条 record：每题一个 logits tensor。"""
        return self.forward_batch([enc])[0]

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


def build_model(backbone: str, device: str, lora: int = 16, head_dim: int = 256,
                revision: str | None = None) -> DecisionModel:
    """构建 jev_kev 模型：骨干 + LoRA + 从零初始化的 PointerHead。"""
    tok = load_tokenizer(backbone, revision=revision)
    model = DecisionModel(
        backbone, tok, device, lora=lora, revision=revision, head_dim=head_dim,
        dtype=torch.bfloat16)
    model.lm.config.use_cache = False
    return model
