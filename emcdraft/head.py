"""Training-time re-implementation of the Qwen3.5/3.8 multi-token-prediction (MTP) head.

The serving engine (vLLM v0.31.0, ``qwen3_5_mtp.py``) builds the head from the *target's*
config, so a replacement head must keep the baseline tensor layout exactly:

    mtp.pre_fc_norm_embedding / mtp.pre_fc_norm_hidden   RMSNorm (1 + w)
    mtp.fc                                               Linear(2H -> H)
    mtp.layers.0                                         one gated full-attention block
    mtp.norm                                             RMSNorm (1 + w)

This module mirrors that computation in plain PyTorch (parameter names are identical to the
checkpoint names without the ``mtp.`` prefix) and adds what training needs:

* ``unroll``     - the K-step recursive drafting pass, vectorised over every position of a
                   sequence with the same key/value visibility the engine's draft KV cache has;
* ``e4m3_round`` - FP8 (E4M3) rounding with a straight-through estimator for QAT;
* ``equalize_``  - an exact, function-preserving re-scaling that moves weights into the range
                   where E4M3 has relative (not absolute) precision.
"""
from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Tensors stored in FP8 in the compact variant (the big matrices); the rest stay bf16.
FP8_TENSORS = (
    "layers.0.mlp.gate_proj.weight",
    "layers.0.mlp.up_proj.weight",
    "layers.0.mlp.down_proj.weight",
    "layers.0.self_attn.o_proj.weight",
    "layers.0.self_attn.k_proj.weight",
    "layers.0.self_attn.v_proj.weight",
)


@dataclass
class HeadConfig:
    hidden: int = 5120
    inter: int = 17408
    heads: int = 24
    kv_heads: int = 4
    head_dim: int = 256
    rot_dim: int = 64
    theta: float = 1e7
    eps: float = 1e-6

    @classmethod
    def from_target(cls, cfg: dict) -> "HeadConfig":
        t = cfg.get("text_config", cfg)
        rp = t.get("rope_parameters") or {}
        prf = rp.get("partial_rotary_factor", t.get("partial_rotary_factor", 1.0))
        hd = t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"]
        return cls(hidden=t["hidden_size"], inter=t["intermediate_size"],
                   heads=t["num_attention_heads"],
                   kv_heads=t.get("num_key_value_heads", t["num_attention_heads"]),
                   head_dim=hd, rot_dim=int(hd * prf),
                   theta=float(rp.get("rope_theta", t.get("rope_theta", 10000.0))),
                   eps=float(t.get("rms_norm_eps", 1e-6)))


# --------------------------------------------------------------------------- FP8 (E4M3)
def e4m3_round(w: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest-even onto the finite FP8 E4M3 grid (saturating at 448)."""
    a = w.detach().abs().float().clamp(max=448.0)
    _, ex = torch.frexp(a)                                  # a = m * 2**ex, m in [0.5, 1)
    e = (ex - 1).clamp(min=-6).float()                      # binade; subnormals share e = -6
    step = torch.exp2(e - 3.0)                              # 3 mantissa bits
    q = (torch.round(a / step) * step).clamp(max=448.0)
    return (torch.sign(w.detach().float()) * q).to(w.dtype)


def e4m3_encode(w: torch.Tensor, chunk: int = 1 << 24) -> torch.Tensor:
    """uint8 bit patterns (sign | 4-bit exponent, bias 7 | 3-bit mantissa) of on-grid values.

    Works in chunks so the int64 temporaries stay small (host memory is limited)."""
    flat = w.detach().reshape(-1)
    out = torch.empty(flat.numel(), dtype=torch.uint8, device=flat.device)
    for s in range(0, flat.numel(), chunk):
        out[s:s + chunk] = _e4m3_encode(flat[s:s + chunk])
    return out.view(w.shape)


def _e4m3_encode(w: torch.Tensor) -> torch.Tensor:
    q = e4m3_round(w.float())
    a = q.abs()
    _, ex = torch.frexp(a)
    e = (ex - 1).clamp(min=-6)
    normal = a >= 2.0 ** -6
    mant_n = torch.round((a / torch.exp2(e.float()) - 1.0) * 8.0).long()
    mant_s = torch.round(a * 2.0 ** 9).long()
    bits = torch.where(normal, ((e.long() + 7) << 3) | mant_n, mant_s)
    bits = bits | ((q < 0).long() << 7)
    return bits.to(torch.uint8)


def e4m3_decode(b: torch.Tensor) -> torch.Tensor:
    b = b.long()
    sign = torch.where((b >> 7) > 0, -1.0, 1.0)
    ef, m = (b >> 3) & 0xF, (b & 0x7).float()
    val = torch.where(ef > 0, (1.0 + m / 8.0) * torch.exp2(ef.float() - 7.0), m * 2.0 ** -9)
    return sign * val


class QLinear(nn.Linear):
    """Bias-free linear layer; ``fq=True`` fake-quantises the weight to E4M3 (STE).

    ``add_lora(r)`` freezes the weight and learns a rank-r update B @ A instead (LoRA, Hu et
    al. 2021); ``merge_lora()`` folds it back into the weight, so exported files keep the
    engine's dense layout."""

    def __init__(self, i: int, o: int):
        super().__init__(i, o, bias=False)
        self.fq = False
        self.lora_a = self.lora_b = None

    def add_lora(self, r: int):
        o, i = self.weight.shape
        dev = self.weight.device
        self.weight.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.randn(r, i, device=dev) / math.sqrt(i))
        self.lora_b = nn.Parameter(torch.zeros(o, r, device=dev))

    @torch.no_grad()
    def merge_lora(self):
        if self.lora_a is not None:
            self.weight.copy_(self.dense_weight())
            self.lora_a = self.lora_b = None
            self.weight.requires_grad_(True)

    def dense_weight(self) -> torch.Tensor:
        w = self.weight
        return w if self.lora_a is None else w + (self.lora_b @ self.lora_a).to(w.dtype)

    def qweight(self) -> torch.Tensor:
        w = self.dense_weight()
        return w + (e4m3_round(w) - w).detach() if self.fq else w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.qweight().to(x.dtype))


class GRMSNorm(nn.Module):
    """RMSNorm with a (1 + w) gain, computed in fp32 (the engine's GemmaRMSNorm)."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * (1.0 + self.weight.float())).to(x.dtype)


def rope(x: torch.Tensor, pos: torch.Tensor, rot_dim: int, theta: float) -> torch.Tensor:
    """NeoX-style partial rotary embedding. x: [B, N, heads, head_dim], pos: [B, N]."""
    half = rot_dim // 2
    inv = 1.0 / (theta ** (torch.arange(0, rot_dim, 2, device=x.device, dtype=torch.float32) / rot_dim))
    ang = pos.float()[..., None] * inv
    cos, sin = ang.cos()[:, :, None, :], ang.sin()[:, :, None, :]
    xf = x.float()
    x1, x2 = xf[..., :half], xf[..., half:rot_dim]
    out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin, xf[..., rot_dim:]], -1)
    return out.to(x.dtype)


class Attention(nn.Module):
    def __init__(self, c: HeadConfig):
        super().__init__()
        self.c = c
        self.q_proj = QLinear(c.hidden, c.heads * c.head_dim * 2)      # per head: [query | gate]
        self.k_proj = QLinear(c.hidden, c.kv_heads * c.head_dim)
        self.v_proj = QLinear(c.hidden, c.kv_heads * c.head_dim)
        self.o_proj = QLinear(c.heads * c.head_dim, c.hidden)
        self.q_norm = GRMSNorm(c.head_dim, c.eps)
        self.k_norm = GRMSNorm(c.head_dim, c.eps)

    def forward(self, x, pos, mem, mask):
        c = self.c
        B, N, _ = x.shape
        qg = self.q_proj(x).view(B, N, c.heads, 2 * c.head_dim)
        q, gate = qg[..., :c.head_dim], qg[..., c.head_dim:]
        k = self.k_proj(x).view(B, N, c.kv_heads, c.head_dim)
        v = self.v_proj(x).view(B, N, c.kv_heads, c.head_dim)
        q = rope(self.q_norm(q), pos, c.rot_dim, c.theta).transpose(1, 2)
        k = rope(self.k_norm(k), pos, c.rot_dim, c.theta).transpose(1, 2)
        v = v.transpose(1, 2)
        keys = torch.cat([m[0] for m in mem] + [k], 2)
        vals = torch.cat([m[1] for m in mem] + [v], 2)
        rep = c.heads // c.kv_heads
        keys, vals = keys.repeat_interleave(rep, 1), vals.repeat_interleave(rep, 1)
        o = F.scaled_dot_product_attention(q, keys, vals, attn_mask=mask)
        o = o.transpose(1, 2) * torch.sigmoid(gate)
        return self.o_proj(o.reshape(B, N, c.heads * c.head_dim)), (k, v)


class MLP(nn.Module):
    def __init__(self, c: HeadConfig):
        super().__init__()
        self.gate_proj = QLinear(c.hidden, c.inter)
        self.up_proj = QLinear(c.hidden, c.inter)
        self.down_proj = QLinear(c.inter, c.hidden)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, c: HeadConfig):
        super().__init__()
        self.self_attn = Attention(c)
        self.mlp = MLP(c)
        self.input_layernorm = GRMSNorm(c.hidden, c.eps)
        self.post_attention_layernorm = GRMSNorm(c.hidden, c.eps)

    def forward(self, h, pos, mem=(), mask=None):
        a, kv = self.self_attn(self.input_layernorm(h), pos, list(mem), mask)
        h = h + a
        return h + self.mlp(self.post_attention_layernorm(h)), kv


class MTPHead(nn.Module):
    def __init__(self, c: HeadConfig):
        super().__init__()
        self.c = c
        self.pre_fc_norm_embedding = GRMSNorm(c.hidden, c.eps)
        self.pre_fc_norm_hidden = GRMSNorm(c.hidden, c.eps)
        self.fc = QLinear(2 * c.hidden, c.hidden)
        self.layers = nn.ModuleList([Layer(c)])
        self.norm = GRMSNorm(c.hidden, c.eps)

    def step(self, emb, hid, pos, mem=(), mask=None):
        """One drafting step. emb/hid: [B, N, H]; returns (post-norm hidden, (k, v))."""
        x = torch.cat([self.pre_fc_norm_embedding(emb), self.pre_fc_norm_hidden(hid)], -1)
        h, kv = self.layers[0](self.fc(x), pos, mem, mask)
        return self.norm(h), kv

    def set_fake_quant(self, on: bool, names=FP8_TENSORS):
        mods = dict(self.named_modules())
        for n in names:
            mods[n.rsplit(".", 1)[0]].fq = on


def unroll(head: MTPHead, embed, tokens: torch.Tensor, hidden: torch.Tensor, K: int = 3):
    """Vectorised K-step drafting for every base position of a batch of sequences.

    tokens [B, T]; hidden [B, T-1, H] = target final hidden states h_0..h_{T-2}.
    Step k (1-based) at base index i consumes token x_{i+k} and predicts x_{i+1+k}; it sees
    the "verified" keys/values of positions <= i (step-1 pass, true hidden states), plus the
    keys/values written by the earlier steps of the same round and its own - exactly what the
    engine's draft KV cache holds at that moment. Returns a generator of step outputs so the
    caller can interleave losses (and use correctness of step k-1 to weight step k).
    """
    B, T = tokens.shape
    N = T - 1
    dev = tokens.device
    base = torch.arange(N, device=dev)
    pos = base[None].expand(B, N)
    tri = torch.ones(N, N, dtype=torch.bool, device=dev).tril()
    eye = torch.eye(N, dtype=torch.bool, device=dev)
    # heads with their own input fusion (EAGLE-3: auxiliary target states -> fc) apply it first
    mem, hid = [], (head.combine(hidden) if hasattr(head, "combine") else hidden)
    for k in range(1, K + 1):
        tok_k = tokens[:, (base + k).clamp(max=T - 1)]
        mask = torch.cat([tri] + [eye] * (k - 1), 1)[None, None]
        res = head.step(embed(tok_k), hid, pos + (k - 1), mem, mask)
        out, kv = res[0], res[1]
        mem.append(kv)
        hid = res[2] if len(res) > 2 else out      # EAGLE-3 recurses on its pre-norm output
        yield k, out


def walk(correct, P: int, T: int, K: int = 3):
    """Greedy speculative decoding over per-position correctness flags.

    correct[k][i] says whether draft step k+1 at base index i matched the target's token.
    Returns (accepted draft tokens, drafting rounds) for one sequence of T tokens whose first
    P tokens are the prompt; acceptance length = 1 + accepted / rounds.
    """
    acc = drafts = 0
    cur = P
    while cur <= T - 2:
        i, n = cur - 1, 0
        for k in range(K):
            if i + 2 + k > T - 1 or not correct[k][i]:
                break
            n += 1
        drafts += 1
        acc += n
        cur += n + 1
    return acc, drafts


# --------------------------------------------------------------------------- equalisation
@torch.no_grad()
def equalize_(head: MTPHead, r: float = 0.25) -> dict:
    """Exact re-scaling that brings every FP8 tensor to RMS ~= r (function unchanged).

    Uses the scale symmetries of the block: RMSNorm inputs are scale-free, so the residual
    stream can be scaled by s (fc, o_proj, down_proj); value/output and up/down projections
    trade scale; the norm gains absorb a common factor of everything that reads them.
    """
    L = head.layers[0]
    A, M = L.self_attn, L.mlp
    rms = lambda w: w.float().pow(2).mean().sqrt().item()
    g = r / rms(M.gate_proj.weight)
    gam = rms(M.gate_proj.weight) / rms(M.up_proj.weight)
    s = r * gam / rms(M.down_proj.weight)
    beta = s * rms(A.o_proj.weight) / r
    cc = r / (rms(A.v_proj.weight) * beta)
    ak = r / rms(A.k_proj.weight)
    M.gate_proj.weight.mul_(g)
    M.up_proj.weight.mul_(g * gam)
    M.down_proj.weight.mul_(s / gam)
    L.post_attention_layernorm.weight.copy_((1 + L.post_attention_layernorm.weight) / g - 1)
    A.q_proj.weight.mul_(cc)
    A.v_proj.weight.mul_(cc * beta)
    A.o_proj.weight.mul_(s / beta)
    A.k_proj.weight.mul_(ak)
    L.input_layernorm.weight.copy_((1 + L.input_layernorm.weight) / cc - 1)
    head.fc.weight.mul_(s)
    return dict(g=g, gamma=gam, s=s, beta=beta, c=cc, a_k=ak)


# --------------------------------------------------------------------------- checkpoint IO
def read_safetensors(path, names=None, prefix=None) -> dict:
    from safetensors import safe_open
    out = {}
    with safe_open(str(path), framework="pt") as f:
        for n in f.keys():
            if (names is None or n in names) and (prefix is None or n.startswith(prefix)):
                out[n] = f.get_tensor(n)
    return out


def load_target_tensors(target_dir, names) -> dict:
    """Read named tensors straight from the target's shards (no model instantiation)."""
    target_dir = Path(target_dir)
    index = json.loads((target_dir / "model.safetensors.index.json").read_text())["weight_map"]
    out = {}
    for shard in sorted({index[n] for n in names}):
        out.update(read_safetensors(target_dir / shard, names=set(names)))
    return out


def load_mtp_state(src) -> dict:
    """mtp.* tensors (prefix stripped) from a submission folder or the target folder."""
    src = Path(src)
    if (src / "model.safetensors.index.json").exists():
        index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
        raw = {}
        for shard in sorted({f for n, f in index.items() if n.startswith("mtp.")}):
            raw.update(read_safetensors(src / shard, prefix="mtp."))
    else:
        raw = {}
        for p in sorted(src.glob("*.safetensors")):
            raw.update(read_safetensors(p, prefix="mtp."))
    return {n[4:]: (e4m3_decode(t.view(torch.uint8)) if "float8" in str(t.dtype) else t.float())
            for n, t in raw.items()}


def build_head(state: dict, cfg: HeadConfig, device="cpu") -> MTPHead:
    head = MTPHead(cfg)
    head.load_state_dict(state, strict=True)
    return head.to(device)


def save_safetensors_raw(path, tensors: dict):
    """Minimal safetensors writer (lets us emit F8_E4M3 without relying on torch float8)."""
    header, blobs, off = {}, [], 0
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [off, off + len(data)]}
        blobs.append(data)
        off += len(data)
    hj = json.dumps(header, separators=(",", ":")).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for b in blobs:
            f.write(b)


@torch.no_grad()
def export_submission(head: MTPHead, out_dir, target_config, fp8: bool, metadata: dict,
                      fp8_names=FP8_TENSORS) -> dict:
    """Write a kit-format submission folder: config.json, model.safetensors, spec, metadata."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors, nbytes = {}, 0
    for name, p in head.state_dict().items():
        w = p.detach().float().cpu()
        if fp8 and name in fp8_names:
            data = e4m3_encode(w).numpy().tobytes()
            tensors["mtp." + name] = ("F8_E4M3", w.shape, data)
        else:
            data = w.to(torch.bfloat16).view(torch.int16).numpy().tobytes()
            tensors["mtp." + name] = ("BF16", w.shape, data)
        nbytes += len(data)
    save_safetensors_raw(out_dir / "model.safetensors", tensors)
    (out_dir / "config.json").write_bytes(Path(target_config).read_bytes())
    (out_dir / "spec.json").write_text(json.dumps({"method": "mtp"}) + "\n")
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {"weight_bytes": nbytes, "tensors": len(tensors)}
