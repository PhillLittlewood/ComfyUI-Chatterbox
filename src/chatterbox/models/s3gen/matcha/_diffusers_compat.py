# coding=utf-8
# Copyright 2024 The HuggingFace Team and the diffusers authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Local replacements for the eight ``diffusers`` symbols the Matcha decoder uses.

Upstream Chatterbox pulls four activations, two adaptive layer norms, an
attention processor and an activation lookup out of ``diffusers``. That was a
loose end this repository inherited: ``diffusers`` is a heavyweight training
library pulled in by a purely inference-only decoder, and the version that would
otherwise be installed cannot even be imported against ``huggingface_hub`` 1.x.
Rather than depend on it, the eight symbols live here, copied from
``diffusers`` 0.27.2 (Apache-2.0, header above) and reduced to the parameters
``models/s3gen/matcha/`` actually reaches.

What was dropped, and why it is safe to drop it -- every reduction below is
justified by a call site in the vendored tree, not by convenience:

* ``Attention``'s processor indirection (``AttnProcessor``/``AttnProcessor2_0``,
  ``set_processor``, ``fuse_projections``, attention slicing, xformers, LoRA and
  custom-diffusion processors): ``matcha`` constructs ``Attention`` itself and
  never registers a processor, so the only processor that ever ran is
  ``AttnProcessor2_0``, and its body is inlined into :meth:`Attention.forward`.
* ``added_kv_proj_dim`` / ``add_k_proj`` / ``add_v_proj``: never passed.
* ``only_cross_attention``: ``matcha`` handles self- vs cross-attention with its
  own ``only_cross_attention`` branch and always passes
  ``cross_attention_dim=None`` to ``attn1``.
* ``norm_q`` / ``norm_k`` / ``norm_num_groups`` / ``group_norm``: never passed.
* ``cross_attention_norm`` / ``norm_cross``: never passed.
* ``spatial_norm``, ``residual_connection``, ``rescale_output_factor``,
  ``out_dim`` (attention-slicing head selection), ``upcast_softmax`` and the
  4-D ``(batch, channel, height, width)`` input rank: unreachable here --
  ``Decoder.forward`` always hands ``BasicTransformerBlock`` a 3-D
  ``(batch, time, channels)`` tensor, and every remaining flag is its default.

Two behaviours are *kept* rather than dropped, because upstream's own code
depends on them:

* ``Attention`` builds ``self.to_out`` as a ``ModuleList`` of ``[Linear, Dropout]``
  and ``Attention.forward`` indexes it -- the parameter names are part of the
  checkpoint.
* ``prepare_attention_mask`` upcasts the 2-D ``(batch, seq)`` padding mask to
  ``(batch, heads, 1, seq)`` by ``repeat_interleave``, which is what makes the
  matcha's ``bool`` padding mask line up with ``scaled_dot_product_attention``.

One behaviour is deliberately *improved*: ``scale_qk`` is honoured by passing
``scale=`` to ``scaled_dot_product_attention``. Upstream 0.27.2 could not do that
(``torch.nn.functional.scaled_dot_product_attention`` only gained ``scale`` in
torch 2.1) and silently ignored the flag on its SDPA path; torch 2.11 is a
requirement here, so the flag means what it says. With the default
``scale_qk=True`` the result is bit-identical to upstream.

Two upstream defects are kept *verbatim* rather than fixed, because the code that
would hit them is unreachable from this tree and a silent "improvement" would be
indistinguishable from a mistake:

* ``AdaLayerNorm.forward`` ends in ``torch.chunk(emb, 2)`` with no ``dim``, which
  splits the batch rather than the conditioning. Nothing constructs
  ``AdaLayerNorm`` here: ``BasicTransformerBlock`` only builds it when
  ``num_embeds_ada_norm`` is given, and matcha's single construction site never
  passes it.
* ``_CombinedTimestepLabelEmbeddings`` treats absent class labels as zeros
  instead of raising, which is what ``BasicTransformerBlock`` passes
  (``class_labels=None``).

``tests/test_diffusers_compat.py::TestUnreachableAdaptiveNorms`` asserts the
first point, so a re-sync that makes the path reachable says so.

``maybe_allow_in_graph`` is an identity decorator, which is exactly what
``diffusers`` resolves to when ``accelerate`` is absent -- and accelerate is not
a dependency of this repository.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "GEGLU",
    "GELU",
    "AdaLayerNorm",
    "AdaLayerNormZero",
    "ApproximateGELU",
    "Attention",
    "get_activation",
    "maybe_allow_in_graph",
]


# ---------------------------------------------------------------------------
# diffusers.utils.torch_utils
# ---------------------------------------------------------------------------


def maybe_allow_in_graph(cls):
    """Identity decorator (``diffusers`` returns the class unchanged)."""
    return cls


# ---------------------------------------------------------------------------
# diffusers.models.activations
# ---------------------------------------------------------------------------

ACTIVATION_FUNCTIONS = {
    "swish": nn.SiLU(),
    "silu": nn.SiLU(),
    "mish": nn.Mish(),
    "gelu": nn.GELU(),
    "relu": nn.ReLU(),
}


def get_activation(act_fn: str) -> nn.Module:
    """Helper function to get activation function from string.

    Args:
        act_fn (str): Name of activation function.

    Returns:
        nn.Module: Activation function.
    """
    act_fn = act_fn.lower()
    if act_fn in ACTIVATION_FUNCTIONS:
        return ACTIVATION_FUNCTIONS[act_fn]
    raise ValueError(f"Unsupported activation function: {act_fn}")


class GELU(nn.Module):
    r"""
    GELU activation function with tanh approximation support with `approximate="tanh"`.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        approximate (`str`, *optional*, defaults to `"none"`): If `"tanh"`, use tanh approximation.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none", bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)
        self.approximate = approximate

    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        if gate.device.type != "mps":
            return F.gelu(gate, approximate=self.approximate)
        # mps: gelu is not implemented for float16
        return F.gelu(gate.to(dtype=torch.float32), approximate=self.approximate).to(dtype=gate.dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        hidden_states = self.gelu(hidden_states)
        return hidden_states


class GEGLU(nn.Module):
    r"""
    A [variant](https://arxiv.org/abs/2002.05202) of the gated linear unit activation function.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)

    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        if gate.device.type != "mps":
            return F.gelu(gate)
        # mps: gelu is not implemented for float16
        return F.gelu(gate.to(dtype=torch.float32)).to(dtype=gate.dtype)

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * self.gelu(gate)


class ApproximateGELU(nn.Module):
    r"""
    The approximate form of the Gaussian Error Linear Unit (GELU). For more details, see section 2 of this
    [paper](https://arxiv.org/abs/1606.08415).

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x * torch.sigmoid(1.702 * x)


# ---------------------------------------------------------------------------
# diffusers.models.normalization
# ---------------------------------------------------------------------------


class _Timesteps(nn.Module):
    """Sinusoidal timestep projection.

    Private because :class:`AdaLayerNormZero` is its only caller. Mirrors
    ``diffusers.models.embeddings.Timesteps`` (``num_channels``,
    ``flip_sin_to_cos``, ``downscale_freq_shift``) and the free function it
    delegates to, minus the Karras/exponential timestep-spacing options that
    only a training pipeline selects.
    """

    def __init__(
        self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float = 1.0
    ):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim != 1:
            raise ValueError(
                f"Timesteps should be a 1d-array, got shape {tuple(timesteps.shape)}"
            )

        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(
            start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
        )
        exponent = exponent / (half_dim - self.downscale_freq_shift)

        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]

        # concat sine and cosine embeddings
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        # flip sine and cosine embeddings
        if self.flip_sin_to_cos:
            emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

        # zero pad
        if self.num_channels % 2 == 1:
            emb = F.pad(emb, (0, 1, 0, 0))
        return emb


class _TimestepEmbedding(nn.Module):
    """Two-layer MLP over the sinusoidal projection; ``diffusers``' equivalent."""

    def __init__(self, in_channels: int, time_embed_dim: int, act_fn: str = "silu"):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = get_activation(act_fn)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class _LabelEmbedding(nn.Module):
    """Class-label embedding.

    Reduced from ``diffusers.models.embeddings.LabelEmbedding``: the
    classifier-free-guidance dropout mask (``force_drop_ids`` / ``train``) only
    matters for text encoders, which this decoder has none of. The embedding
    table, its size and the dropout are kept, so the parameter shapes match.
    """

    def __init__(self, num_classes: int, embedding_dim: int, class_dropout_prob: float):
        super().__init__()
        use_cfg_embedding = class_dropout_prob > 0
        self.embedding_table = nn.Embedding(
            num_classes + int(use_cfg_embedding), embedding_dim
        )
        self.num_classes = num_classes
        self.dropout = nn.Dropout(class_dropout_prob)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.embedding_table(labels))


class _CombinedTimestepLabelEmbeddings(nn.Module):
    """Timestep + class-label embedding, as used by adaLN-Zero."""

    def __init__(self, num_classes: int, embedding_dim: int, class_dropout_prob: float = 0.1):
        super().__init__()
        self.time_proj = _Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1
        )
        self.timestep_embedder = _TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim
        )
        self.class_embedder = _LabelEmbedding(num_classes, embedding_dim, class_dropout_prob)

    def forward(
        self,
        timestep: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        hidden_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))

        if class_labels is None:
            # ``diffusers`` would raise here; the vendored decoder has no class
            # labels at all, so an all-zero conditioning is the faithful zero.
            class_labels = torch.zeros(
                timesteps_proj.shape[0], dtype=torch.long, device=timesteps_proj.device
            )
        return timesteps_emb + self.class_embedder(class_labels)


class AdaLayerNorm(nn.Module):
    r"""
    Norm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, num_embeddings: int):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        emb = self.linear(self.silu(self.emb(timestep)))
        scale, shift = torch.chunk(emb, 2)
        return self.norm(x) * (1 + scale) + shift


class AdaLayerNormZero(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, num_embeddings: int):
        super().__init__()
        self.emb = _CombinedTimestepLabelEmbeddings(num_embeddings, embedding_dim)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        class_labels: Optional[torch.LongTensor] = None,
        hidden_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


# ---------------------------------------------------------------------------
# diffusers.models.attention_processor.Attention (+ AttnProcessor2_0, inlined)
# ---------------------------------------------------------------------------


class Attention(nn.Module):
    r"""Scaled-dot-product attention, matching ``diffusers``' ``Attention``.

    Reduced to the parameter and path surface the Matcha decoder reaches; see
    the module docstring for the full list of what was dropped and why.

    Parameters:
        query_dim (`int`): The number of channels in the query.
        cross_attention_dim (`int`, *optional*): The number of channels in the encoder hidden states.
        heads (`int`, *optional*, defaults to 8): The number of heads to use for multi-head attention.
        dim_head (`int`, *optional*, defaults to 64): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        bias (`bool`, *optional*, defaults to False): Set to `True` for q/k/v to contain a bias parameter.
        upcast_attention (`bool`, *optional*, defaults to False): Accepted for signature parity.
            ``diffusers``' SDPA processor ignores it too; it is kept so a caller
            passing it gets upstream's behaviour rather than a ``TypeError``.
        out_bias (`bool`, *optional*, defaults to `True`): Bias in the output linear layer.
        scale_qk (`bool`, *optional*, defaults to `True`): Scale q/k by ``1 / sqrt(dim_head)``.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        out_bias: bool = True,
        scale_qk: bool = True,
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.is_cross_attention = cross_attention_dim is not None
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.dropout = dropout
        self.out_dim = query_dim

        self.scale_qk = scale_qk
        self.scale = dim_head**-0.5 if self.scale_qk else 1.0

        self.heads = heads
        self.sliceable_head_dim = heads

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, self.out_dim, bias=out_bias))
        self.to_out.append(nn.Dropout(dropout))

    def prepare_attention_mask(
        self, attention_mask: torch.Tensor, target_length: int, batch_size: int
    ) -> torch.Tensor:
        """Pad and per-head-expand ``(batch, seq)`` masks for SDPA.

        Returns ``(batch * heads, target_length)``, which
        :meth:`forward` reshapes to ``(batch, heads, -1, target_length)``.
        """
        if attention_mask is None:
            return attention_mask

        current_length: int = attention_mask.shape[-1]
        if current_length != target_length:
            if attention_mask.device.type == "mps":
                # HACK: MPS does not support padding by more than the input
                # dimension; build the padding tensor by hand instead.
                padding_shape = (attention_mask.shape[0], attention_mask.shape[1], target_length)
                padding = torch.zeros(padding_shape, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, padding], dim=2)
            else:
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)

        if attention_mask.shape[0] < batch_size * self.heads:
            attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)
        return attention_mask

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        """``AttnProcessor2_0.__call__`` inlined.

        ``cross_attention_kwargs`` is accepted and ignored: with the processor
        registry gone there is nothing to dispatch those to, and ``matcha``
        passes an empty dict.
        """
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, self.heads, -1, attention_mask.shape[-1])

        query = self.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads

        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        # the output of sdp is (batch, num_heads, seq_len, head_dim)
        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scale,
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj, then dropout
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)

        return hidden_states