"""EAGLE-3 draft head in the layout of the engine's `LlamaForCausalLMEagle3`.

The draft reads three intermediate residual streams of the target (layers 2, 32 and 61 of 64,
the engine's default), normalises each, fuses them with one linear layer, and runs a single
decoder layer whose attention sees [norm(embedding of the next token), norm(fused state)]. It
has no embedding and no output layer of its own: the engine then shares the target's, so they
cost nothing in the memory metric and the draft predicts over the full vocabulary.

Drafting in the engine (vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py): step 1
takes combine_hidden_states(aux) at the target's positions; every later step takes the draft's
own pre-norm output, while the post-norm output goes to the shared LM head. Positions and the
draft KV cache follow the same rules as the MTP head, so `emcdraft.head.unroll` applies as is.

Trained online during feature extraction (`OnlineTrainer`), from the same target forward
passes that produce the MTP features.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from .head import QLinear, rope, save_safetensors_raw


@dataclass
class EagleConfig:
    hidden: int = 5120
    heads: int = 32
    kv_heads: int = 8
    head_dim: int = 128
    inter: int = 6144              # keeps the bf16 weights inside the memory cap (see README)
    eps: float = 1e-6
    theta: float = 1e6
    aux_layers: tuple = (2, 32, 61)
    vocab: int = 248320
    max_pos: int = 262144


class RMSNorm(nn.Module):
    """x * rsqrt(mean(x^2) + eps) * w, in fp32 (the engine's RMSNorm)."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.float()).to(x.dtype)


class Eagle3Head(nn.Module):
    def __init__(self, c: EagleConfig):
        super().__init__()
        self.c = c
        H, n = c.hidden, len(c.aux_layers)
        self.fc_norm = nn.ModuleList([RMSNorm(H, c.eps) for _ in range(n)])
        self.fc = QLinear(n * H, H)
        self.input_layernorm = RMSNorm(H, c.eps)
        self.hidden_norm = RMSNorm(H, c.eps)
        self.q_proj = QLinear(2 * H, c.heads * c.head_dim)
        self.k_proj = QLinear(2 * H, c.kv_heads * c.head_dim)
        self.v_proj = QLinear(2 * H, c.kv_heads * c.head_dim)
        self.o_proj = QLinear(c.heads * c.head_dim, H)
        self.post_attention_layernorm = RMSNorm(H, c.eps)
        self.gate_proj = QLinear(H, c.inter)
        self.up_proj = QLinear(H, c.inter)
        self.down_proj = QLinear(c.inter, H)
        self.norm = RMSNorm(H, c.eps)
        for m in self.modules():
            if isinstance(m, QLinear):
                nn.init.normal_(m.weight, std=0.02)

    def combine(self, aux):
        """[.., n*H] auxiliary states -> [.., H] (the engine's combine_hidden_states, fc_norm=True)."""
        H = self.c.hidden
        return self.fc(torch.cat([nm(ch) for nm, ch in zip(self.fc_norm, aux.split(H, -1))], -1))

    def step(self, emb, hid, pos, mem=(), mask=None):
        """One drafting step. Returns (post-norm output for the LM head, (k, v), pre-norm output
        that the next step consumes)."""
        c = self.c
        B, N, _ = emb.shape
        x = torch.cat([self.input_layernorm(emb), self.hidden_norm(hid)], -1)
        q = self.q_proj(x).view(B, N, c.heads, c.head_dim)
        k = self.k_proj(x).view(B, N, c.kv_heads, c.head_dim)
        v = self.v_proj(x).view(B, N, c.kv_heads, c.head_dim)
        q = rope(q, pos, c.head_dim, c.theta).transpose(1, 2)
        k = rope(k, pos, c.head_dim, c.theta).transpose(1, 2)
        v = v.transpose(1, 2)
        keys = torch.cat([m[0] for m in mem] + [k], 2)
        vals = torch.cat([m[1] for m in mem] + [v], 2)
        rep = c.heads // c.kv_heads
        o = F.scaled_dot_product_attention(q, keys.repeat_interleave(rep, 1), vals.repeat_interleave(rep, 1),
                                           attn_mask=mask)
        r = self.o_proj(o.transpose(1, 2).reshape(B, N, c.heads * c.head_dim)) + hid
        h = self.post_attention_layernorm(r)
        pre = self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h)) + r
        return self.norm(pre), (k, v), pre


def engine_names(head: Eagle3Head) -> dict:
    """Our parameter name -> checkpoint name read by the engine's Eagle3LlamaForCausalLM."""
    out = {}
    for n, _ in head.named_parameters():
        if n.startswith(("fc.", "fc_norm.", "norm.")):
            out[n] = n
        elif n.split(".")[0] in ("q_proj", "k_proj", "v_proj", "o_proj"):
            out[n] = "midlayer.self_attn." + n
        elif n.split(".")[0] in ("gate_proj", "up_proj", "down_proj"):
            out[n] = "midlayer.mlp." + n
        else:
            out[n] = "midlayer." + n
    return out


def engine_config(c: EagleConfig) -> dict:
    return {
        "architectures": ["LlamaForCausalLMEagle3"], "model_type": "llama",
        "hidden_size": c.hidden, "intermediate_size": c.inter, "num_attention_heads": c.heads,
        "num_key_value_heads": c.kv_heads, "head_dim": c.head_dim, "num_hidden_layers": 1,
        "hidden_act": "silu", "rms_norm_eps": c.eps, "max_position_embeddings": c.max_pos,
        "rope_theta": c.theta, "rope_parameters": {"rope_type": "default", "rope_theta": c.theta},
        "attention_bias": False, "mlp_bias": False, "tie_word_embeddings": False,
        "vocab_size": c.vocab, "draft_vocab_size": c.vocab, "target_hidden_size": c.hidden,
        "fc_norm": True, "eagle_aux_hidden_state_layer_ids": list(c.aux_layers),
        "eagle_config": {"eagle_aux_hidden_state_layer_ids": list(c.aux_layers), "use_aux_hidden_state": True},
        "torch_dtype": "bfloat16",
    }


@torch.no_grad()
def export_eagle(head: Eagle3Head, out_dir, metadata: dict) -> dict:
    """Kit-format folder: config.json (the draft's own), model.safetensors (bf16), spec, metadata."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors, nbytes = {}, 0
    for ours, name in engine_names(head).items():
        t = head.state_dict()[ours].detach().float().to(torch.bfloat16).cpu().contiguous()
        tensors[name] = ("BF16", tuple(t.shape), t.view(torch.int16).numpy().tobytes())
        nbytes += t.numel() * 2
    save_safetensors_raw(out_dir / "model.safetensors", tensors)
    (out_dir / "config.json").write_text(json.dumps(engine_config(head.c), indent=2))
    (out_dir / "spec.json").write_text(json.dumps({"method": "eagle3"}))
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return {"weight_bytes": nbytes, "tensors": len(tensors),
            "params": sum(p.numel() for p in head.parameters())}


class OnlineTrainer:
    """Trains an Eagle3Head one batch at a time from target states computed elsewhere."""

    def __init__(self, cfg: EagleConfig, device, total_steps: int, lr=3e-4, warmup=100,
                 step_w=(1.0, 0.8, 0.64), topk=10, beta=0.5, floor=0.3, log=print):
        self.head = Eagle3Head(cfg).to(device)
        self.opt = torch.optim.AdamW(self.head.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
        self.a = SimpleNamespace(beta=beta, floor=floor, step_w=list(step_w), topk=topk)
        self.total, self.warm, self.lr = max(1, total_steps), warmup, lr
        self.n, self.skipped, self.log, self.t0 = 0, 0, log, time.time()
        self.hist = []

    def step(self, embed, W, tokens, aux, teacher, P, T):
        from .train import train_step
        f = (self.n + 1) / self.warm if self.n < self.warm else \
            0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, (self.n - self.warm) / max(1, self.total - self.warm))))
        for g in self.opt.param_groups:
            g["lr"] = self.lr * f
        try:
            self.head.train()
            loss, info = train_step(self.head, embed, W, (tokens, aux, P, T), self.a, teacher=teacher)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.head.parameters(), 1.0)
            self.opt.step()
        except torch.OutOfMemoryError:
            self.opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            self.skipped += 1
            return None
        self.n += 1
        if self.n % 25 == 0 or self.n == 1:
            est = 1 + info["acc1"] * (1 + info["acc2"] * (1 + info["acc3"]))
            self.log(f"  eagle step {self.n}/{self.total} ce {info['ce1']:.3f}/{info['ce2']:.3f}/{info['ce3']:.3f} "
                     f"acc {info['acc1']:.3f}/{info['acc2']:.3f}/{info['acc3']:.3f} batch-tau~{est:.3f} "
                     f"lr x{f:.2f} ({(time.time() - self.t0) / self.n:.2f}s/step incl. target)")
        return info

    def summary(self):
        return {"config": asdict(self.head.c), "steps": self.n, "skipped_oom": self.skipped,
                "lr": self.lr, "history": self.hist}
