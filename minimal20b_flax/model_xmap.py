import numpy as np
from einops import repeat

import jax
import jax.numpy as jnp
# noinspection PyPep8Naming
from jax.experimental import PartitionSpec as P
from jax.experimental.pjit import pjit
from jax.nn.initializers import zeros

import flax.linen as nn
from flax import struct
from flax.core import frozen_dict
from flax import traverse_util
from minimal20b_flax.utils import f_psum, g_psum


@struct.dataclass
class NeoX20BConfig:
    vocab_size: int = 50432
    hidden_size: int = 6144
    num_attention_heads: int = 64
    rotary_pct: float = 0.25
    rotary_emb_base: int = 10000
    layernorm_epsilon: float = 1e-5
    num_layers: int = 44
    tp_num: int = 8


default_neox20b_config = NeoX20BConfig()


@struct.dataclass
class GPTNeoX20BModel:

    config: NeoX20BConfig = default_neox20b_config

    def eval(self, params, x, mask):
        embedded = ShardedEmbedIn(config=self.config).apply({"params": params["embed_in"]}, x)
        h = embedded
        for layer_i in range(self.config.num_layers):
            residual = ShardedTransformerLayer(config=self.config).apply(
                {"params": params[f"layer_{layer_i:02d}"]}, h, mask
            )
            h = h + residual
        return ShardedEmbedOut(config=self.config).apply({"params": params["embed_out"]}, h)

    def get_initial_decode_state(self, params, ctx, ctx_length):
        # Embed initial context
        embedded = ShardedEmbedIn(config=self.config).apply(
            {"params": params["embed_in"]}, ctx)

        # Set up scan function for creating decode_states for each layer
        decode_state_list = []
        h = embedded
        for layer_i in range(self.config.num_layers):
            new_residual, layer_decode_state = ShardedTransformerLayer(config=self.config).apply(
                {"params": params[f"layer_{layer_i:02d}"]},
                h, ctx_length,
                method=ShardedTransformerLayer.get_init_decode_state,
            )
            decode_state_list.append(layer_decode_state)
            h = h + new_residual
        print(h.shape)
        final_logit = ShardedEmbedOut(config=self.config).apply(
            {"params": params["embed_out"]},
            h[-1:, :],
        )
        return {"logits": final_logit, "decode_state": decode_state_list}

    def decode_once(self, params, single_x, decode_state):
        assert single_x.shape[0] == 1
        # Embed single token
        embedded = ShardedEmbedIn(config=self.config).apply(
            {"params": params["embed_in"]}, single_x)
        h = embedded
        new_decode_state_list = []
        for layer_i in range(self.config.num_layers):
            new_residual, new_layer_decode_state = ShardedTransformerLayer(config=self.config).apply(
                {"params": params[f"layer_{layer_i:02d}"]},
                decode_state[layer_i], h,
                method=ShardedTransformerLayer.decode_once,
            )
            h = h + new_residual
            new_decode_state_list.append(new_layer_decode_state)
        # Project to logits
        logits = ShardedEmbedOut(config=self.config).apply(
            {"params": params["embed_out"]},
            h,
        )
        return {
            "logits": logits,
            "new_decode_state": new_decode_state_list,
        }

    def generate(self, params, ctx, ctx_length, rng, sampler_args):
        GENERATE_LENGTH = 64
        init_out = self.get_initial_decode_state(
            params=params,
            ctx=ctx,
            ctx_length=ctx_length
        )
        # Add sampling logic here
        init_logits = init_out["logits"].swapaxes(0, 1).reshape(1, self.config.vocab_size)
        initial_token = init_logits.argmax(-1)

        init_carry = {
            "single_x": initial_token,
            "decode_state": init_out["decode_state"],
        }

        def _decode_once_scan_fn(decode_carry, step_rng):
            decode_out = self.decode_once(
                params=params,
                single_x=decode_carry["single_x"],
                decode_state=decode_carry["decode_state"],
            )

            # Add sampling logic here
            # next_token = decode_out["logits"].argmax(-1)
            next_token = temperature_sample(
                key=step_rng,
                logits=decode_out["logits"].swapaxes(0, 1).reshape(1, self.config.vocab_size),
                **sampler_args,
            )

            next_carry = {
                "single_x": next_token,
                "decode_state": decode_out["new_decode_state"]
            }
            outputs = {
                "logits": decode_out["logits"],
                "next_token": next_token,
            }
            return next_carry, outputs

        final_state, generation_outputs = jax.lax.scan(
            f=_decode_once_scan_fn,
            init=init_carry,
            xs=jax.random.split(rng, GENERATE_LENGTH),
        )
        # return {
        #     # "final_state": final_state,
        #     "initial_logits": init_out["logits"],
        #     "initial_token": initial_token,
        #     "generated_logits": generation_outputs["logits"],
        #     "generated_tokens": generation_outputs["next_token"],
        # }

        tokens = generation_outputs["next_token"][:, 0]
        logits = generation_outputs["logits"].swapaxes(0, 1).reshape(-1, self.config.vocab_size)
        return {
            # "final_state": final_state,
            "generated_logits": jnp.concatenate((init_logits, logits), axis=0),
            "generated_tokens": jnp.concatenate((initial_token, tokens), axis=0),
        }


class ShardedEmbedIn(nn.Module):

    config: NeoX20BConfig = default_neox20b_config

    @nn.compact
    def __call__(self, input_ids):
        config = self.config
        dims_per_shard = self.config.vocab_size // config.tp_num
        shard_start_index = jax.lax.axis_index('shard') * dims_per_shard

        # TODO: Check if this still works
        input_onehot = jax.nn.one_hot(input_ids - shard_start_index, dims_per_shard, dtype=jnp.float16)
        embedded = nn.Dense(
            features=config.hidden_size,
            kernel_init=zero_init_fp16(),
            use_bias=False,
            name="embed",
            dtype=jnp.float16,
        )(input_onehot)
        embedded = g_psum(embedded)
        return embedded


class ShardedTransformerLayer(nn.Module):
    """Sharded Transformer Layer.

    Note: This doesn't compute the full residual connection x + r(x), only r(x).
          The residual connection will be computed downstream.
    """
    config: NeoX20BConfig = default_neox20b_config

    # noinspection PyAttributeOutsideInit
    def setup(self):
        config = self.config
        self.dims_per_head = config.hidden_size // config.num_attention_heads
        self.heads_per_shard = config.num_attention_heads // config.tp_num
        self.dims_per_shard = config.hidden_size // config.tp_num
        self.attn_norm = ReplicatedLayerNorm(epsilon=config.layernorm_epsilon, dtype=jnp.float16)
        self.ff_norm = ReplicatedLayerNorm(epsilon=config.layernorm_epsilon, dtype=jnp.float16)
        self.qkv_proj = nn.Dense(
            self.dims_per_shard * 3,
            name="qkv_proj",
            dtype=jnp.float16,
            kernel_init=zero_init_fp16(),
            bias_init=zero_init_fp16(),
        )
        self.output_proj = nn.Dense(
            config.hidden_size,
            name="output_proj",
            dtype=jnp.float16,
            kernel_init=zero_init_fp16(),
            bias_init=zero_init_fp16(),
        )
        self.ff_up_proj = nn.Dense(
            self.dims_per_shard * 4,
            name="ff_up_proj",
            dtype=jnp.float16,
            kernel_init=zero_init_fp16(),
            bias_init=zero_init_fp16(),
        )
        self.ff_down_proj = nn.Dense(
            config.hidden_size,
            name="ff_down_proj",
            dtype=jnp.float16,
            kernel_init=zero_init_fp16(),
            bias_init=zero_init_fp16(),
        )

    def __call__(self, x, attn_bias):
        """
        :param x: [seq_len, hidden_size]
        :param attn_bias: [*, seq_len, seq_len]
        :return: [seq_len, hidden_size]
        """
        attn_in = self.attn_norm(x)
        # -> [seq_len, hidden_size]

        q, k, v = self.compute_qkv(attn_in)
        # -> 3 x [seq_len, heads, dims_per_head]

        seq_len = attn_in.shape[0]
        causal_mask = np.tril(np.ones((seq_len, seq_len)))[None, :, :]  # NumPy array gets cached
        # -> [1, seq_len, seq_len]

        bias = -1e4 * (1. - causal_mask)
        bias += attn_bias
        # -> [1, seq_len, seq_len]

        attn_out = self.compute_self_attn(q, k, v, bias)
        # -> [seq_len, hidden]

        ff_out = self.compute_ff(x)
        # -> [seq_len, hidden]

        return g_psum(attn_out + ff_out)

    def split_heads(self, x):
        reshaped = x.reshape(x.shape[:-1] + (self.heads_per_shard, self.dims_per_head))
        return reshaped

    def compute_qkv(self, x):
        # [seq, 3*dims_per_shard]
        qkv_arr = self.qkv_proj(x)
        q, k, v = jnp.split(qkv_arr, [self.dims_per_shard, self.dims_per_shard * 2], axis=-1)

        # [seq, heads, dims_per_head]
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        return q, k, v

    def compute_self_attn(self, q, k, v, attn_bias):
        """
        :param q: [q_len, heads, dims_per_head]
        :param k: [kv_len, heads, dims_per_head]
        :param v: [kv_len, heads, dims_per_head]
        :param attn_bias: [*, q_len, kv_len]
        :return: [q_len, hidden]
        """
        config = self.config
        rotary_dims = int(config.hidden_size // config.num_attention_heads * config.rotary_pct)
        k_rot = k[..., :rotary_dims]
        k_pass = k[..., rotary_dims:]

        q_rot = q[..., :rotary_dims]
        q_pass = q[..., rotary_dims:]

        sincos = fixed_pos_embedding(k_rot, seq_dim=0)  # TODO check this
        # return sincos
        q_rot = apply_rotary_pos_emb(q_rot, sincos)
        k_rot = apply_rotary_pos_emb(k_rot, sincos)

        k = jnp.concatenate([k_rot, k_pass], axis=-1)
        q = jnp.concatenate([q_rot, q_pass], axis=-1)

        attention_logits = jnp.einsum("thd,Thd->htT", q, k)
        # -> [heads, q_len, kv_len]

        sqrt_key_size = np.sqrt(config.hidden_size // config.num_attention_heads).astype(k.dtype)
        attention_logits = attention_logits / sqrt_key_size
        attention_logits += attn_bias
        # -> [heads, q_len, kv_len]

        attention_weights = jax.nn.softmax(attention_logits)
        # -> [heads, q_len, kv_len]

        attention_vec = jnp.einsum("htT,Thd->thd", attention_weights, v)
        # -> [q_len, heads, dims_per_head]

        attention_vec = attention_vec.reshape(-1, self.dims_per_shard)
        # -> [q_len, hidden]

        attn_out = self.output_proj(attention_vec)
        # -> [q_len, hidden]

        return attn_out

    def compute_ff(self, x):
        ff_out = self.ff_norm(x)
        ff_out = self.ff_up_proj(ff_out)
        ff_out = jax.nn.gelu(ff_out)
        ff_out = self.ff_down_proj(ff_out)
        return ff_out

    # iterate the decoding process by a single token
    def decode_once(self, decode_state, x):
        """
        :param decode_state:
        :param x: [batch, q_len=1, hidden_size]
        """
        attn_in = self.attn_norm(x)
        print("attn_in", attn_in)
        # -> [q_len=1, hidden_size]
        q, v, k = self.compute_qkv(attn_in)
        print("v", v)
        # -> 3 x [q_len=1, heads, dims_per_head]

        # ?? assert x.shape[0] == 1

        # add new kv to end, clip off the start
        v = jnp.concatenate((decode_state["v"], v), axis=0)[1:]
        print("v2", v)
        # -> [kv_len+1, heads, dims_per_head]
        k = jnp.concatenate((decode_state["k"], k), axis=0)[1:]
        # -> [kv_len+1, heads, dims_per_head]

        tokens_decoded = decode_state["tokens_decoded"] + 1

        length = v.shape[0]
        masked_tokens = (length - tokens_decoded)[None]
        print("masked_tokens", masked_tokens)
        attention_mask = (jnp.arange(0, length) < masked_tokens)[None, :]
        print("attention_mask", attention_mask)
        # -> [q_len=1, seq_len]

        bias = (-1e4 * attention_mask)
        print("bias", bias)
        # -> [q_len=1, seq_len]

        attn_out = self.compute_self_attn(q, v, k, bias)
        print("attn_out", attn_out)
        # -> 3 x [q_len=1, hidden]

        ff_out = self.compute_ff(x)
        print("ff_out", ff_out)
        # -> 3 x [q_len=1, hidden]

        return g_psum(attn_out + ff_out), {
            "tokens_decoded": tokens_decoded,
            "k": k,
            "v": v
        }

    # take in right aligned context tokens and generate an initial state
    def get_init_decode_state(self, x, given_length):
        """
        :param x: [batch, seq_len, hidden_size]
        :param given_length: [batch]
        """
        x = f_psum(x)
        attn_in = self.attn_norm(x)
        # -> [batch, seq_len, hidden_size]
        q, v, k = self.compute_qkv(attn_in)
        # -> 3 x [batch, seq_len, heads, dims_per_head]

        full_length = x.shape[0]
        masked_tokens = (full_length - given_length)[None]

        causal_mask = np.tril(np.ones((full_length, full_length)))
        # -> [seq_len, seq_len]

        bias = -1e4 * (1. - causal_mask)  # regular AR masking
        # -> [seq_len, seq_len]

        context_length_mask = (jnp.arange(0, full_length) < masked_tokens)[None, :]
        # -> [1, seq_len]

        bias -= 1e4 * context_length_mask  # mask out zero tokens before context starts
        # -> [seq_len, seq_len]

        attn_out = self.compute_self_attn(q, v, k, bias[None, :, :])
        # -> [seq_len, hidden]

        ff_out = self.compute_ff(x)
        # -> [seq_len, hidden]

        return g_psum(attn_out + ff_out), {
            "tokens_decoded": given_length.astype(jnp.uint32),
            "k": k,
            "v": v,
        }


class ShardedEmbedOut(nn.Module):

    config: NeoX20BConfig = default_neox20b_config

    # noinspection PyAttributeOutsideInit
    def setup(self):
        config = self.config
        self.vocab_per_shard = config.vocab_size // config.tp_num
        self.norm = ReplicatedLayerNorm(epsilon=config.layernorm_epsilon, dtype=jnp.float16)
        self.embed_out = nn.Dense(
            self.vocab_per_shard, use_bias=False, dtype=jnp.float16,
            kernel_init=zero_init_fp16(),
            bias_init=zero_init_fp16(),
        )

    def __call__(self, x):
        logits = self.predict(x)
        logits = jax.lax.all_gather(logits, 'shard')
        # Transpose?
        return logits

    def predict(self, x):
        x = self.norm(x)
        out = self.embed_out(x)
        return out

    def loss(self, x, targets, z_loss=1):
        x = f_psum(x)
        logits = self.predict(x)

        shard_start_index = jax.lax.axis_index('shard') * self.dim_per_shard
        global_max = jax.lax.pmax(jax.lax.stop_gradient(logits.max(-1, keepdims=True)), "shard")
        logits -= jax.lax.stop_gradient(global_max)

        gt_onehot = jax.nn.one_hot(targets - shard_start_index, self.dim_per_shard)
        predicted_logits = jnp.sum(jnp.multiply(gt_onehot, logits), axis=-1)
        predicted_logits = g_psum(predicted_logits)

        exp_logits = jnp.exp(logits)

        sum_exp_logits = exp_logits.sum(axis=-1)
        sum_exp_logits = g_psum(sum_exp_logits)

        loss = jnp.log(sum_exp_logits) - predicted_logits

        loss += (1e-4 * jnp.square(jnp.log(sum_exp_logits)) * z_loss).mean()

        correct = (0.0 == predicted_logits)

        return loss, correct


class ReplicatedLayerNorm(nn.Module):

    def __init__(self, epsilon, dtype):
        super().__init__()
        self.epsilon = epsilon
        self.dtype = dtype

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        mean = jnp.mean(inputs, axis=-1, keepdims=True)
        variance = jnp.var(inputs, axis=-1, keepdims=True)

        param_shape = inputs.shape[-1:]
        scale = self.param("scale", zeros, param_shape, self.dtype)
        scale = jax.lax.all_gather(scale, "shard")[0]

        offset = self.param("bias", zeros, param_shape, self.dtype)
        offset = jax.lax.all_gather(offset, "shard")[0]

        scale = jnp.broadcast_to(scale, inputs.shape)
        offset = jnp.broadcast_to(offset, inputs.shape)
        mean = jnp.broadcast_to(mean, inputs.shape)

        inv = scale * jax.lax.rsqrt(variance + self.epsilon)
        return inv * (inputs - mean) + offset


def fixed_pos_embedding(x, seq_dim=0):
    dim = x.shape[-1]
    inv_freq = 1. / (10000 ** (np.arange(0, dim, 2) / dim)) .astype(np.float16)
    sinusoid_inp = np.einsum('i , j -> i j', np.arange(x.shape[seq_dim]), inv_freq) .astype(np.float16)
    return np.sin(sinusoid_inp).astype(np.float16), np.cos(sinusoid_inp).astype(np.float16)


def apply_rotary_pos_emb(x, sincos):
    sin, cos = map(lambda t: repeat(t, '... b n -> ... b (j n)', j=2)[-x.shape[-3]:, None, :], sincos)
    return (x * cos) + (rotate_half(x) * sin)


def rotate_half(x):
    half_dim = x.shape[-1] // 2
    x1 = x[:, :, :half_dim]
    x2 = x[:, :, half_dim:]
    return jnp.concatenate((-x2, x1), axis=-1)


def temperature_sample(key, logits, temp=1):
    return jax.random.categorical(key, logits/temp, -1).astype(jnp.int32)


def zero_init_fp16():
    def zeros_(key, shape, dtype):
        return jnp.zeros(shape, jnp.float16)
    return zeros_
