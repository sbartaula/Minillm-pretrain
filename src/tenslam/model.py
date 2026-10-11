"""Tenslam MiniLM: a decoder-only Transformer, written from scratch in plain PyTorch.

Architecture (pre-norm, LLaMA-style):
    tokens -> Embedding -> [ x + Attn(RMSNorm(x)) ; x + SwiGLU(RMSNorm(x)) ] * L
           -> RMSNorm -> Linear(d_model -> vocab)   (weights tied with the embedding)
"""
from __future__ import annotations

import math
import tomllib
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    n_layers: int = 8
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 1408
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    init_std: float = 0.02
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert (self.d_model // self.n_heads) % 2 == 0, "head_dim must be even for RoPE"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @classmethod
    def from_toml(cls, path: str = "configs/model.toml") -> "ModelConfig":
        with open(path, "rb") as f:
            return cls(**tomllib.load(f)["model"])

    def to_dict(self) -> dict:
        return asdict(self)


class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x^2) + eps) * g.  Like LayerNorm without mean-centering or bias."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()  # normalise in fp32 even under bf16/fp16 autocast
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.to(x.dtype) * self.weight


def build_rope_cache(seq_len: int, head_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables of shape (seq_len, head_dim/2): angle[t, i] = t * theta^(-2i/head_dim)."""
    inv_freq = theta ** (-torch.arange(0, head_dim, 2).float() / head_dim)
    angles = torch.outer(torch.arange(seq_len).float(), inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate each (x_i, x_{i+d/2}) pair of x:(B,H,T,D) by its position's angle.

    Because rotations compose, q_m . k_n depends only on (m - n): relative position
    information enters the attention score without any learned position embedding.
    """
    t = x.shape[-2]
    cos, sin = cos[:t].to(x.dtype), sin[:t].to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads, self.head_dim = cfg.n_heads, cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        # softmax(q k^T / sqrt(d) + causal_mask) v, fused by PyTorch (flash / mem-efficient kernels).
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(b, t, c))


class SwiGLU(nn.Module):
    """down( silu(gate(x)) * up(x) ): a gated MLP; 3 matrices instead of 2."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.mlp(self.mlp_norm(x))


class TenslamLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight  # one shared (vocab x d_model) matrix
        cos, sin = build_rope_cache(cfg.max_seq_len, cfg.head_dim, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)
        # GPT-2 trick: each block adds 2 residual branches, so shrink their output
        # projections by 1/sqrt(2L) to keep the residual stream variance ~constant at init.
        for name, p in self.named_parameters():
            if name.endswith(("attn.proj.weight", "mlp.down.weight")):
                nn.init.normal_(p, mean=0.0, std=cfg.init_std / math.sqrt(2 * cfg.n_layers))

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)

    def num_params(self, exclude_embeddings: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())  # parameters() de-duplicates tied weights
        return n - self.tok_emb.weight.numel() if exclude_embeddings else n

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """idx:(B,T) int token ids -> logits:(B,T,V); loss only if targets:(B,T) given.

        targets are the *same sequence shifted left by one* (done by the caller); position
        t is trained to predict token t+1. Use -100 in targets to ignore a position.
        """
        assert idx.shape[1] <= self.cfg.max_seq_len, "sequence longer than max_seq_len"
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), targets.reshape(-1),
                                   ignore_index=-100)
        return logits, loss

    def configure_optimizer(self, lr: float, weight_decay: float = 0.1,
                            betas: tuple[float, float] = (0.9, 0.95)) -> torch.optim.AdamW:
        """AdamW; weight decay only on matrices (>=2D), not on norm gains / biases."""
        decay = [p for p in self.parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for p in self.parameters() if p.requires_grad and p.dim() < 2]
        return torch.optim.AdamW(
            [{"params": decay, "weight_decay": weight_decay},
             {"params": no_decay, "weight_decay": 0.0}], lr=lr, betas=betas)
