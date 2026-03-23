"""
Pure-PyTorch GatedDeltaNet module for ExLlamaV2.

Implements the recurrent linear attention mechanism used by Qwen 3.5 / Qwen3Next models.
No custom CUDA kernels required; optionally accelerated by flash-linear-attention (fla)
and causal-conv1d packages when available.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import math

from exllamav2.module import ExLlamaV2Module
from exllamav2.rmsnorm import ExLlamaV2RMSNorm
from exllamav2.layernorm import ExLlamaV2LayerNorm

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from exllamav2.model import ExLlamaV2


# ---------------------------------------------------------------------------
# causal_conv1d wrappers with PyTorch fallback
# ---------------------------------------------------------------------------

def _causal_conv1d_fwd_torch(x, weight, bias):
    """Full-sequence causal conv1d (prefill). x: [bsz, dim, seq_len]"""
    bsz, dim, seq_len = x.shape
    k = weight.shape[-1]
    x_padded = F.pad(x.to(weight.dtype), (k, 0))
    y = F.conv1d(x_padded, weight.unsqueeze(1), bias, padding=0, groups=dim)
    y = F.silu(y[:, :, -seq_len:])
    return y.to(x.dtype)


def _causal_conv1d_update_torch(x, conv_state, weight, bias=None):
    """Incremental causal conv1d (decode). conv_state updated in-place."""
    bsz, dim, seq_len = x.shape
    state_len = conv_state.shape[-1]
    y = torch.cat([conv_state, x.to(conv_state.dtype)], dim=-1)
    conv_state.copy_(y[:, :, -state_len:])
    y = F.conv1d(y.to(weight.dtype), weight.unsqueeze(1), bias, padding=0, groups=dim)
    y = F.silu(y[:, :, -seq_len:])
    return y.to(x.dtype)


try:
    import causal_conv1d_cuda
    def _causal_conv1d_fwd(x, weight, bias):
        y = torch.empty_like(x)
        causal_conv1d_cuda.causal_conv1d_fwd(x, weight, bias, None, None, y, None, True)
        return y
    def _causal_conv1d_update(x, conv_state, weight, bias=None):
        y = torch.empty_like(x)
        causal_conv1d_cuda.causal_conv1d_update(x, conv_state, weight, bias, y, True, None, None)
        return y
except (ModuleNotFoundError, ImportError):
    _causal_conv1d_fwd = _causal_conv1d_fwd_torch
    _causal_conv1d_update = _causal_conv1d_update_torch


# Try to import fla (flash-linear-attention)
try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    _has_fla_chunk = True
except (ModuleNotFoundError, ImportError, ValueError):
    chunk_gated_delta_rule = None
    _has_fla_chunk = False

try:
    from fla.ops.gated_delta_rule.fused_recurrent import fused_recurrent_gated_delta_rule_fwd
    _has_fla_recurrent = True
except (ModuleNotFoundError, ImportError, ValueError):
    fused_recurrent_gated_delta_rule_fwd = None
    _has_fla_recurrent = False


# ---------------------------------------------------------------------------
# RecurrentState – holds conv + recurrent state per GDN layer
# ---------------------------------------------------------------------------

class GDNRecurrentState:
    __slots__ = ("position", "last_conv_state", "last_recurrent_state")

    def __init__(self):
        self.position = 0
        self.last_conv_state = None
        self.last_recurrent_state = None

    def clone(self):
        new = GDNRecurrentState()
        new.position = self.position
        new.last_conv_state = None if self.last_conv_state is None else self.last_conv_state.clone()
        new.last_recurrent_state = (
            None if self.last_recurrent_state is None else self.last_recurrent_state.clone()
        )
        return new


# ---------------------------------------------------------------------------
# Pure-torch recurrent gated delta rule
# ---------------------------------------------------------------------------

def _torch_recurrent_gated_delta_rule(q, k, v, g, beta, initial_state, output_final_state):
    """Token-by-token recurrence. q/k/v: [bsz, seq, heads, dim], g/beta: [bsz, seq, heads]."""

    def _l2norm(x, eps=1e-6):
        return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)

    q = _l2norm(q.float())
    k = _l2norm(k.float())

    bsz, seqlen, n_heads, k_dim = k.shape
    v_dim = v.shape[-1]

    state = (
        torch.zeros(bsz, n_heads, k_dim, v_dim, dtype=torch.float, device=v.device)
        if initial_state is None else initial_state.float()
    )

    vf = v.float()
    bf = beta.float()
    gf = g.float()

    out = torch.zeros(bsz, seqlen, n_heads, v_dim, dtype=torch.float, device=v.device)

    for i in range(seqlen):
        q_t = q[:, i]
        k_t = k[:, i]
        v_t = vf[:, i]
        g_t = gf[:, i].exp().unsqueeze(-1)
        b_t = bf[:, i].unsqueeze(-1)

        kv_recall = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta_v = v_t - kv_recall * g_t
        state = state * g_t.unsqueeze(-1) + k_t.unsqueeze(-1) * delta_v.unsqueeze(-2) * b_t.unsqueeze(-1)
        out[:, i] = (state * q_t.unsqueeze(-1)).sum(dim=-2) / math.sqrt(k_dim)

    final = state if output_final_state else None
    return out, final


# ---------------------------------------------------------------------------
# Gated RMSNorm
# ---------------------------------------------------------------------------

def _gated_rmsnorm(x, gate, weight, eps=1e-6):
    """Matches HF Qwen3_5RMSNormGated exactly."""
    input_dtype = x.dtype
    xf = x.float()
    variance = xf.pow(2).mean(-1, keepdim=True)
    x_normed = xf * torch.rsqrt(variance + eps)
    x_normed = weight * x_normed.to(input_dtype)
    x_normed = x_normed * F.silu(gate.float())
    return x_normed.to(input_dtype)


# ---------------------------------------------------------------------------
# ExLlamaV2GatedDeltaNet
# ---------------------------------------------------------------------------

class ExLlamaV2GatedDeltaNet(ExLlamaV2Module):

    name: str = "GatedDeltaNet"

    def __init__(
        self,
        model: ExLlamaV2,
        key: str,
        layer_idx: int,
    ):
        super().__init__(model, key)

        cfg = self.model.config

        self.layer_idx = layer_idx
        self.has_norm = True
        self.has_residual = True
        self.device_idx = 0

        # GDN geometry
        self.hidden_size = cfg.hidden_size
        self.num_k_heads = cfg.gdn_num_key_heads
        self.k_head_dim = cfg.gdn_key_head_dim
        self.num_v_heads = cfg.gdn_num_value_heads
        self.v_head_dim = cfg.gdn_value_head_dim
        self.conv_kernel_size = cfg.gdn_conv_kernel_dim
        self.rms_norm_eps = cfg.norm_eps

        self.k_dim = self.k_head_dim * self.num_k_heads
        self.v_dim = self.v_head_dim * self.num_v_heads
        self.num_v_groups = self.num_v_heads // self.num_k_heads
        self.fdim_qkv = 2 * self.k_dim + self.v_dim

        # Pre-layernorm (input_layernorm, shared key prefix is layer key minus ".linear_attn")
        layer_base_key = key.rsplit(".linear_attn", 1)[0]
        km = self.archparams.keys
        if self.archparams.norm == "layernorm":
            self.pre_layernorm = ExLlamaV2LayerNorm(model, layer_base_key + km["norm_1"])
        else:
            self.pre_layernorm = ExLlamaV2RMSNorm(model, layer_base_key + km["norm_1"])

        # Raw weight tensors (loaded later)
        self.qkv_proj_weight = None
        self.z_proj_weight = None
        self.b_proj_weight = None
        self.a_proj_weight = None
        self.o_proj_weight = None
        self.gdn_norm_weight = None
        self.a_log = None
        self.dt_bias = None
        self.conv1d_weight = None
        self.conv1d_bias = None

        self.submodules = [self.pre_layernorm]

    def numel(self) -> int:
        n = self.pre_layernorm.numel()
        n += self.hidden_size * self.fdim_qkv       # qkv_proj
        n += self.hidden_size * self.v_dim           # z_proj
        n += self.hidden_size * self.num_v_heads     # b_proj
        n += self.hidden_size * self.num_v_heads     # a_proj
        n += self.v_dim * self.hidden_size           # o_proj
        n += self.v_head_dim                         # norm
        n += self.num_v_heads * 2                    # a_log + dt_bias
        n += self.fdim_qkv * self.conv_kernel_size   # conv1d
        return n

    def weight_footprint(self) -> int:
        return self.numel() * 2

    def scratch_space(self) -> int:
        return 0

    def scratch_space_fixed(self) -> int:
        return 0

    def scratch_space_tp(self) -> int:
        return 0

    def temp_attn_size(self) -> int:
        return 0

    def set_device_idx(self, idx):
        super().set_device_idx(idx)
        self.pre_layernorm.set_device_idx(idx)

    @torch.inference_mode()
    def load(self, device_context=True):
        cfg = self.model.config
        dev = self.device()

        self.pre_layernorm.load()

        # Load raw tensors from safetensors
        tensors = self.load_multi(
            self.key,
            [
                "in_proj_qkv.weight",
                "in_proj_z.weight",
                "in_proj_b.weight",
                "in_proj_a.weight",
                "out_proj.weight",
                "norm.weight",
                "conv1d.weight",
            ],
        )

        self.qkv_proj_weight = tensors["in_proj_qkv.weight"]
        self.z_proj_weight = tensors["in_proj_z.weight"]
        self.b_proj_weight = tensors["in_proj_b.weight"]
        self.a_proj_weight = tensors["in_proj_a.weight"]
        self.o_proj_weight = tensors["out_proj.weight"]
        self.gdn_norm_weight = tensors["norm.weight"]
        self.conv1d_weight = tensors["conv1d.weight"]

        # A_log and dt_bias don't have .weight suffix
        # Load them directly from safetensors files
        for st_path in cfg.tensor_files:
            from exllamav2.stloader import STFile
            stf = STFile.open(st_path, keymap=cfg.arch.keymap)
            keys_in_file = stf.get_dict()
            for tname in ["A_log", "dt_bias"]:
                full_key = self.key + "." + tname
                if full_key in keys_in_file:
                    t = stf.get_tensor(full_key, device=dev)
                    if tname == "A_log":
                        self.a_log = t.float()
                    else:
                        self.dt_bias = t

        # Try conv1d.bias
        try:
            bias_tensors = self.load_multi(self.key, ["conv1d.bias"])
            self.conv1d_bias = bias_tensors.get("conv1d.bias")
        except Exception:
            self.conv1d_bias = None

    def unload(self):
        self.pre_layernorm.unload()
        self.qkv_proj_weight = None
        self.z_proj_weight = None
        self.b_proj_weight = None
        self.a_proj_weight = None
        self.o_proj_weight = None
        self.gdn_norm_weight = None
        self.a_log = None
        self.dt_bias = None
        self.conv1d_weight = None
        self.conv1d_bias = None

    def get_weight_dict(self) -> dict:
        """Return all GDN weights as a flat dict keyed by safetensors-style keys."""
        d = {}
        d[self.key + ".in_proj_qkv.weight"] = self.qkv_proj_weight
        d[self.key + ".in_proj_z.weight"] = self.z_proj_weight
        d[self.key + ".in_proj_b.weight"] = self.b_proj_weight
        d[self.key + ".in_proj_a.weight"] = self.a_proj_weight
        d[self.key + ".out_proj.weight"] = self.o_proj_weight
        d[self.key + ".norm.weight"] = self.gdn_norm_weight
        d[self.key + ".conv1d.weight"] = self.conv1d_weight
        if self.conv1d_bias is not None:
            d[self.key + ".conv1d.bias"] = self.conv1d_bias
        d[self.key + ".A_log"] = self.a_log
        d[self.key + ".dt_bias"] = self.dt_bias
        return d

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache=None,
        attn_params=None,
        past_len=None,
        intermediates=False,
        loras=None,
        **kwargs,
    ):
        bsz, seqlen, _ = hidden_states.shape

        # Residual & pre-norm
        residual = hidden_states
        x = self.pre_layernorm.forward(hidden_states)

        # Get/create recurrent state
        recurrent_states = kwargs.get("recurrent_states")
        if recurrent_states is not None:
            rs = recurrent_states.get(self.layer_idx)
            if rs is None:
                rs = GDNRecurrentState()
                recurrent_states[self.layer_idx] = rs
        else:
            rs = GDNRecurrentState()

        if recurrent_states is not None and past_len is not None and rs.position != past_len:
            raise RuntimeError(
                f"Recurrent state for layer {self.layer_idx} is out of sync "
                f"(state at {rs.position}, cache at {past_len})."
            )

        device = x.device

        # 1. Linear projections (use bfloat16 to match HF reference precision)
        xbf = x.to(torch.bfloat16)
        qkv = F.linear(xbf, self.qkv_proj_weight.to(torch.bfloat16))
        z = F.linear(xbf, self.z_proj_weight.to(torch.bfloat16))
        b = F.linear(xbf, self.b_proj_weight.to(torch.bfloat16))
        a = F.linear(xbf, self.a_proj_weight.to(torch.bfloat16))

        z = z.view(bsz, seqlen, self.num_v_heads, self.v_head_dim)

        # 2. Beta and g (following HF reference)
        beta = torch.sigmoid(b.float())
        g = -self.a_log.exp() * F.softplus(a.float() + self.dt_bias.float())

        # 3. Causal conv1d
        mixed = qkv.transpose(1, 2).contiguous()  # [bsz, fdim_qkv, seqlen] in bf16
        conv_w = self.conv1d_weight.squeeze(1)

        if rs.last_conv_state is None:
            # Prefill: save conv state and run full-sequence conv
            conv_state = F.pad(mixed, (self.conv_kernel_size - seqlen, 0))
            rs.last_conv_state = conv_state[:, :, -self.conv_kernel_size:].clone()
            mixed = _causal_conv1d_fwd(mixed, conv_w, self.conv1d_bias)
        else:
            mixed = _causal_conv1d_update(mixed, rs.last_conv_state, conv_w, self.conv1d_bias)

        # 4. Split Q/K/V and run recurrence
        mixed_t = mixed.transpose(1, 2).contiguous()
        q, k, v = torch.split(mixed_t, [self.k_dim, self.k_dim, self.v_dim], dim=-1)
        q = q.view(bsz, seqlen, self.num_k_heads, self.k_head_dim)
        k = k.view(bsz, seqlen, self.num_k_heads, self.k_head_dim)
        v = v.view(bsz, seqlen, self.num_v_heads, self.v_head_dim)

        if self.num_v_groups > 1:
            q = q.repeat_interleave(self.num_v_groups, dim=2)
            k = k.repeat_interleave(self.num_v_groups, dim=2)

        recurrent_state = rs.last_recurrent_state
        if recurrent_state is None:
            recurrent_state = torch.zeros(
                bsz, self.num_v_heads, self.k_head_dim, self.v_head_dim,
                dtype=torch.float, device=device
            )

        # Select backend
        use_fla = False
        if _has_fla_chunk and seqlen > 1 and chunk_gated_delta_rule is not None:
            use_fla = True
        elif _has_fla_recurrent and seqlen == 1 and fused_recurrent_gated_delta_rule_fwd is not None:
            use_fla = True

        if use_fla and _has_fla_chunk and seqlen > 1:
            core_out, recurrent_state = chunk_gated_delta_rule(
                q, k, v, g=g, beta=beta.to(torch.bfloat16),
                initial_state=recurrent_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        elif use_fla and _has_fla_recurrent and seqlen == 1:
            scale = 1.0
            core_out, recurrent_state = fused_recurrent_gated_delta_rule_fwd(
                q, k, v, g, None, None, beta.to(torch.bfloat16), scale,
                recurrent_state.contiguous() if recurrent_state is not None else None,
                True, True, None,
            )
        else:
            core_out, recurrent_state = _torch_recurrent_gated_delta_rule(
                q, k, v, g, beta,
                initial_state=recurrent_state,
                output_final_state=True,
            )

        # 5. Gated RMSNorm (cast to bf16 first to match HF's dtype behavior)
        core_out = core_out.to(torch.bfloat16)
        core_out = _gated_rmsnorm(
            core_out.reshape(-1, self.v_head_dim),
            z.reshape(-1, self.v_head_dim),
            self.gdn_norm_weight, self.rms_norm_eps
        )
        core_out = core_out.reshape(bsz, seqlen, self.v_dim)

        # 6. Output projection
        output = F.linear(core_out.to(torch.bfloat16), self.o_proj_weight.to(torch.bfloat16))

        # 7. Residual connection (cast back to input dtype for compatibility with quantized modules)
        hidden_states = (output + residual).to(residual.dtype)

        # 8. Update state
        rs.last_recurrent_state = recurrent_state
        rs.position += seqlen

        if intermediates:
            return {"hidden_states": hidden_states}
        return hidden_states
