"""JEV decision model (multimodal).

Architecture: Qwen3.5 backbone (vision tower + language model, no LM head) +
a permutation-equivariant candidate head.

STATE -> QUESTION -> CANDIDATE hierarchy. Every (state, question, candidate)
triple is encoded as one multimodal causal sequence

    [STATE] <state images/text> [QUESTION] <question> [CANDIDATE] <candidate>

and the backbone hidden state at the candidate's last token is that
candidate's vector. All candidate paths of one question reuse the same state
image(s), so the shared prefix (and vision encoding) is recomputed per
candidate; that is wasteful but numerically correct (a shared-prefix / cached
vision encoder is a possible later optimisation).

Candidate head (per question):
  candidate vectors [C, H]
    -> Linear H->D
    -> CandidateSetEncoder: small transformer, no positional embeddings, so it
       is permutation-equivariant over the candidate set
    -> Linear D->H, residual with the backbone vector
    -> ScalarScorer: RMSNorm -> Linear H->D -> SiLU -> Linear D->1
  -> one scalar logit per candidate; the per-question softmax is applied by
     the loss, not the model.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import Qwen3_5Model


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


class CandidateSetEncoder(nn.Module):
    """Transformer over one question's candidate set.

    No positional embeddings, so permuting candidates permutes the outputs
    identically (permutation-equivariant).
    """

    def __init__(self, d_model: int = 256, nhead: int = 4, num_layers: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor, cand_mask: torch.Tensor) -> torch.Tensor:
        # x: [Bq, Cmax, d]; cand_mask: [Bq, Cmax] True = valid.
        # src_key_padding_mask: True = position to ignore.
        return self.encoder(x, src_key_padding_mask=~cand_mask)


class ScalarScorer(nn.Module):
    """RMSNorm -> Linear H->D -> SiLU -> Linear D->1."""

    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.norm = RMSNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(self.norm(x)))).squeeze(-1)


class PathEncoder(nn.Module):
    """Encodes the packed per-candidate paths and reads each candidate's end."""

    def __init__(self, backbone: Qwen3_5Model):
        super().__init__()
        self.backbone = backbone

    def forward(self, batch: dict) -> torch.Tensor:
        out = self.backbone(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
            pixel_values=batch.get("pixel_values"),
            image_grid_thw=batch.get("image_grid_thw"),
            mm_token_type_ids=batch.get("mm_token_type_ids"),
        )
        hs = out.last_hidden_state  # [P, L, H]
        p = hs.size(0)
        h = hs[torch.arange(p, device=hs.device), batch["cand_end_pos"]]  # [P, H]
        bq, cmax = batch["cand_mask"].shape
        cand_vecs = hs.new_zeros(bq, cmax, hs.size(-1))
        cand_vecs[batch["q_index"], batch["cand_index"]] = h
        return cand_vecs


class JevModel(nn.Module):
    def __init__(
        self,
        backbone_path: str,
        set_dim: int = 256,
        set_layers: int = 2,
        set_heads: int = 4,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        backbone = Qwen3_5Model.from_pretrained(backbone_path, torch_dtype=dtype)
        text_config = getattr(backbone.config, "text_config", backbone.config)
        hidden = text_config.hidden_size
        self.hidden_size = hidden

        self.path_encoder = PathEncoder(backbone)
        self.proj_in = nn.Linear(hidden, set_dim)
        self.set_encoder = CandidateSetEncoder(set_dim, set_heads, set_layers)
        self.proj_out = nn.Linear(set_dim, hidden)
        self.scorer = ScalarScorer(hidden, set_dim)

    def encode_candidates(self, batch: dict) -> torch.Tensor:
        return self.path_encoder(batch)

    def _score(self, cand_vecs: torch.Tensor, cand_mask: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(cand_vecs)
        x = x * cand_mask.unsqueeze(-1).to(x.dtype)
        enc = self.set_encoder(x, cand_mask)
        rep = cand_vecs + self.proj_out(enc)
        logits = self.scorer(rep)  # [Bq, Cmax]
        return logits.masked_fill(~cand_mask, -1e4)

    def forward(self, batch: dict, perm_reg: bool = False) -> dict:
        cand_mask = batch["cand_mask"]
        cand_vecs = self.encode_candidates(batch)
        logits = self._score(cand_vecs, cand_mask)

        out = {"logits": logits, "cand_mask": cand_mask}

        if perm_reg:
            # Permutation-invariance regularizer: re-score a randomly permuted
            # candidate order and align the logits back. With an exactly
            # equivariant set encoder this KL is ~0; it guards the property
            # under bf16 numerics.
            bq, cmax = cand_mask.shape
            device = cand_vecs.device
            perm = torch.arange(cmax, device=device).unsqueeze(0).repeat(bq, 1)
            for q in range(bq):
                c = int(cand_mask[q].sum())
                if c > 1:
                    perm[q, :c] = torch.randperm(c, device=device)
            idx_h = perm.unsqueeze(-1).expand(-1, -1, cand_vecs.size(-1))
            vecs_p = cand_vecs.gather(1, idx_h)
            mask_p = cand_mask.gather(1, perm)
            logits_p = self._score(vecs_p, mask_p)
            out["logits_perm"] = logits_p.gather(1, torch.argsort(perm, dim=1))

        return out
