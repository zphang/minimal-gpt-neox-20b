import torch.nn as nn
import torch
import torch.nn.functional as F
import math

from typing import Tuple
import minimal20b.rotary as rotary


class NeoX20BModel(nn.Module):
    def __init__(self, args, use_cache=False, device=None):
        super().__init__()
        self.use_cache = use_cache
        self.embed_in = nn.Embedding(args.vocab_size, args.hidden_size, device=device)
        self.layer_list = nn.ModuleList([])
        for layer_i in range(args.num_layers):
            self.layer_list.append(TransformerLayer(args, use_cache, device=device))
        self.final_layer_norm = LayerNormWithTPDuplication(
            args.hidden_size,
            eps=args.layernorm_epsilon,
            device=device,
        )
        self.logits_out = LinearWithTPMerge(
            args.hidden_size,
            args.vocab_size,
            bias=False,
            device=device,
        )

    def forward(self, x, attention_mask, layer_past=None):
        if layer_past is None:
            layer_past = [None] * len(self.layer_list)
        kv_cache_list = []
        hidden_states = self.embed_in(x)
        hidden_states = self.pre_transformer_transpose(hidden_states)
        hidden_states_replica_1, hidden_states_replica_2 = hidden_states, hidden_states
        for layer_i, layer in enumerate(self.layer_list):
            (hidden_states_replica_1, hidden_states_replica_2), kv_cache = layer(
                x_replica_1=hidden_states_replica_1,
                x_replica_2=hidden_states_replica_2,
                attention_mask=attention_mask,
                layer_past=layer_past[layer_i],
            )
            kv_cache_list.append(kv_cache)
        hidden_states_replica_1 = self.post_transformer_transpose(hidden_states_replica_1)
        hidden_states_replica_2 = self.post_transformer_transpose(hidden_states_replica_2)
        hidden_states_replica_1, hidden_states_replica_2 = self.final_layer_norm(
            hidden_states_replica_1,
            hidden_states_replica_2,
        )
        logits = self.logits_out(hidden_states_replica_1, hidden_states_replica_2)
        if self.use_cache:
            return logits, kv_cache_list
        else:
            return logits

    @classmethod
    def pre_transformer_transpose(cls, x):
        return x.transpose(0, 1).contiguous()

    @classmethod
    def post_transformer_transpose(cls, x):
        return x.transpose(0, 1).contiguous()


class TransformerLayer(nn.Module):
    def __init__(self, args, use_cache, device=None):
        super().__init__()
        self.use_cache = use_cache
        self.input_layernorm = LayerNormWithTPDuplication(
            args.hidden_size,
            eps=args.layernorm_epsilon,
            device=device,
        )
        self.post_attention_layernorm = LayerNormWithTPDuplication(
            args.hidden_size,
            eps=args.layernorm_epsilon,
            device=device,
        )
        self.attention = SelfAttention(args, self.use_cache, device=device)
        self.mlp = MLP(args)

    def forward(self, x_replica_1, x_replica_2, attention_mask, layer_past=None):
        residual_replica_1, residual_replica_2 = x_replica_1, x_replica_2
        ln_output_replica_1, ln_output_replica_2 = self.input_layernorm(
            x_replica_1,
            x_replica_2,
        )
        (attention_output_replica_1, attention_output_replica_2), kv_cache = self.attention(
            ln_output_replica_1,
            ln_output_replica_2,
            attention_mask,
            layer_past=layer_past,
        )
        post_attn_ln_replica_1, post_attn_ln_replica_2 = self.post_attention_layernorm(
            x_replica_1,
            x_replica_2,
        )
        mlp_output_replica_1, mlp_output_replica_2 = self.mlp(
            hidden_states_replica_1=post_attn_ln_replica_1,
            hidden_states_replica_2=post_attn_ln_replica_2,
        )
        output_replica_1 = residual_replica_1 + mlp_output_replica_1 + attention_output_replica_1
        output_replica_2 = residual_replica_2 + mlp_output_replica_2 + attention_output_replica_2
        return (output_replica_1, output_replica_2), kv_cache


class SelfAttention(nn.Module):
    def __init__(self, args, use_cache=False, device=None):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.use_cache = use_cache
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size_per_attention_head = args.hidden_size // args.num_attention_heads
        self.rotary_ndims = int(self.hidden_size_per_attention_head * args.rotary_pct)
        self.rotary_emb = rotary.RotaryEmbedding(
            self.rotary_ndims,
            base=args.rotary_emb_base,
            device=device,
        )
        self.query_key_value = LinearWithTPMerge(
            args.hidden_size,
            3 * args.hidden_size,
            device=device,
        )
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        self.dense = LinearWithTPSplitBias(
            args.hidden_size,
            args.hidden_size,
            device=device,
        )

    def forward(self, hidden_states_replica_1, hidden_states_replica_2, attention_mask, layer_past=None):
        has_layer_past = layer_past is not None and layer_past.numel() > 0

        # Compute QKV
        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        qkv = self.query_key_value(
            x_replica_1=hidden_states_replica_1,
            x_replica_2=hidden_states_replica_2,
        )

        # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
        new_qkv_shape = qkv.size()[:-1] + (
            self.num_attention_heads,
            3 * self.hidden_size_per_attention_head,
        )
        qkv = qkv.view(*new_qkv_shape)

        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        query_layer = qkv[..., :self.hidden_size_per_attention_head]
        key_layer = qkv[..., self.hidden_size_per_attention_head: 2 * self.hidden_size_per_attention_head]
        value_layer = qkv[..., 2 * self.hidden_size_per_attention_head:]

        # Compute rotary embeddings
        query_rot, query_pass = (
            query_layer[..., : self.rotary_ndims],
            query_layer[..., self.rotary_ndims:],
        )
        key_rot, key_pass = (
            key_layer[..., : self.rotary_ndims],
            key_layer[..., self.rotary_ndims:],
        )
        seq_len = key_layer.shape[0]
        offset = 0
        if has_layer_past:
            offset = layer_past[0].shape[0]
            seq_len += offset
        cos, sin = self.rotary_emb(value_layer, seq_len=seq_len)
        query_layer, key_layer = rotary.apply_rotary_pos_emb(
            query_rot, key_rot, cos, sin, offset=offset,
        )
        query_layer = torch.cat((query_layer, query_pass), dim=-1)
        key_layer = torch.cat((key_layer, key_pass), dim=-1)

        # Cache QKV values
        if has_layer_past:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer), key_layer), dim=0)
            value_layer = torch.cat((past_value.type_as(value_layer), value_layer), dim=0)
        if self.use_cache:
            kv_cache = torch.stack((key_layer, value_layer))
        else:
            kv_cache = None

        # Compute attention
        # noinspection PyTypeChecker
        context_layer = self.attention(
            query_layer, key_layer, value_layer, attention_mask
        )

        # Reshape outputs
        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.hidden_size,
        )
        context_layer = context_layer.view(*new_context_layer_shape)

        # =================
        # Output. [sq, b, h]
        # =================
        output_replica_1, output_replica_2 = self.dense(context_layer)

        return (output_replica_1, output_replica_2), kv_cache

    def attention(self, query_layer, key_layer, value_layer, attention_mask):
        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
        )

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(
            output_size[2], output_size[0] * output_size[1], -1
        )
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

        # preallocating result tensor: [b * np, sq, sk]
        matmul_result = torch.empty(
            output_size[0] * output_size[1],
            output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=query_layer.device,
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_result,
            query_layer.transpose(0, 1),  # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ==================================================
        # Update attention mask for inference. [b, np, sq, sk]
        # ==================================================

        if self.use_cache:
            attention_mask = attention_mask[
                             ...,
                             :attention_scores.size(3),
                             :attention_scores.size(3),
                             ]

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        masked_scores = attention_mask_func(attention_scores,
                                            attention_mask) if attention_mask is not None else attention_scores
        attention_probs = torch.nn.Softmax(dim=-1)(masked_scores)

        #         # This is actually dropping out entire tokens to attend to, which might
        #         # seem a bit unusual, but is taken from the original Transformer paper.
        #         attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (
            value_layer.size(1),
            value_layer.size(2),
            query_layer.size(0),
            value_layer.size(3),
        )

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(
            value_layer.size(0), output_size[0] * output_size[1], -1
        )

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(
            output_size[0] * output_size[1], output_size[2], -1
        )

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)
        return context_layer


class MLP(nn.Module):
    def __init__(self, args, device=None):
        super().__init__()
        ff_dim = 4 * args.hidden_size
        self.dense_h_to_4h = LinearWithTPMerge(
            args.hidden_size,
            ff_dim,
            device=device,
        )
        self.dense_4h_to_h = LinearWithTPSplitBias(
            ff_dim,
            args.hidden_size,
            device=device,
        )

    def forward(self, hidden_states_replica_1, hidden_states_replica_2):
        intermediate_parallel = self.dense_h_to_4h(
            x_replica_1=hidden_states_replica_1,
            x_replica_2=hidden_states_replica_2,
        )
        intermediate_parallel = bias_gelu_impl(intermediate_parallel)
        output_replica_1, output_replica_2 = self.dense_4h_to_h(intermediate_parallel)
        return output_replica_1, output_replica_2


class LinearWithTPMerge(nn.Module):

    def __init__(self, in_features: int, out_features: int, bias=True, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        half_out_features = self.out_features // 2
        self.linear_split_1 = nn.Linear(in_features, half_out_features, bias=bias, device=device)
        self.linear_split_2 = nn.Linear(in_features, half_out_features, bias=bias, device=device)

    def forward(self, x_replica_1: torch.Tensor, x_replica_2: torch.Tensor) -> torch.Tensor:
        output_from_replica_1 = self.linear_split_1(x_replica_1)
        output_from_replica_2 = self.linear_split_2(x_replica_2)
        merged_output = torch.cat([output_from_replica_1, output_from_replica_2], dim=-1)
        return merged_output


class LinearWithTPSplitBias(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device))
        self.bias_replica_1 = nn.Parameter(torch.empty(out_features, device=device))
        self.bias_replica_2 = nn.Parameter(torch.empty(out_features, device=device))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y_replica_1 = F.linear(x, self.weight, self.bias_replica_1)
        y_replica_2 = F.linear(x, self.weight, self.bias_replica_2)
        return y_replica_1, y_replica_2


class LayerNormWithTPDuplication(nn.Module):
    def __init__(self, hidden_size: int, eps: int, device=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.layernorm_epsilon = eps
        self.layer_norm_replica_1 = nn.LayerNorm(
            hidden_size,
            eps=eps,
            device=device,
        )
        self.layer_norm_replica_2 = nn.LayerNorm(
            hidden_size,
            eps=eps,
            device=device,
        )

    def forward(self, x_replica_1: torch.Tensor, x_replica_2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y_replica_1 = self.layer_norm_replica_1(x_replica_1)
        y_replica_2 = self.layer_norm_replica_2(x_replica_2)
        return y_replica_1, y_replica_2


# noinspection PyAbstractClass
class GeLUFunction(torch.autograd.Function):
    # noinspection PyMethodOverriding
    @staticmethod
    # bias is an optional argument
    def forward(ctx, inputs):
        ctx.save_for_backward(inputs)
        return gelu(inputs)

    # noinspection PyMethodOverriding
    @staticmethod
    def backward(ctx, grad_output):
        inputs = ctx.saved_tensors
        tmp = gelu_back(grad_output, inputs)
        return tmp, tmp


bias_gelu_impl = GeLUFunction.apply


def generate_mask(seq_len):
    return torch.tril(torch.ones((1, 1, seq_len, seq_len))) < 0.5


def attention_mask_func(attention_scores, ltor_mask):
    attention_scores.masked_fill_(ltor_mask, -10000.0)
    return attention_scores


# @torch.jit.script
def gelu(x):
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


# gradient of tanh approximation of gelu
# gradient of actual gelu is:
# 0.5 * (1. + torch.erf(x * 0.70710678)) + 0.3989423 * x * torch.exp(-0.5 * x * x)
# @torch.jit.script
def gelu_back(g, x):
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * x * (
            (1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)
    ) + 0.5 * (1 + tanh_out)
    return ff * g
