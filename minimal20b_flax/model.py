import numpy as np
from einops import rearrange, repeat

import jax
import jax.numpy as jnp
# noinspection PyPep8Naming
from jax.experimental import PartitionSpec as P

import flax.linen as nn
from flax.linen.partitioning import with_sharding_constraint as shard_to
from flax import struct


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


neox20b_config = NeoX20BConfig()


@struct.dataclass
class ShardedTransformer:

    config: NeoX20BConfig

    def compute_embedding(self, x):
        x = shard_to(x, P("dp", None))
        return ShardedEmbedIn(config=self.config)(x)

    def compute_single_transformer_layer(self, x, mask):
        out = x + ShardedTransformerLayer(config=self.config)(x, mask)
        return shard_to(out, P("dp", None, "mp"))

    def compute_single_transformer_layer_with_grad_checkpoint(self, x, mask):
        return jax.checkpoint(self.compute_single_transformer_layer, prevent_cse=False)(x, mask)

    def init_decode(self, x, given_length, mask):
        residual, decode_state = ShardedTransformerLayer(
            config=self.config,
        ).get_init_decode_state(x, given_length, mask)
        out = x + residual
        return shard_to(out, P("dp", None, "mp")), decode_state

    def iter_decode(self, decode_state, x):
        residual, decode_state = ShardedTransformerLayer(
            config=self.config,
        ).decode_once(decode_state, x, 0)
        out = x + residual
        return shard_to(out, P("dp", None, "mp")), decode_state

    def eval_apply_scan_fn(self, layer_in, layer_state):
        x, mask = layer_in
        return (self.compute_single_transformer_layer_with_grad_checkpoint(layer_state, x, mask), mask), None

    # def setup(self):
    #     self.embed_in = ShardedEmbedIn(config=self.config)
    #     self.transformer_layer = ShardedTransformerLayer(config=self.config)
    #     self.embed_out = ShardedEmbedOut(config=self.config)
    #
    # def __call__(self, params, x, mask):
    #     embedded = self.embed_in.apply({"params": params["embed_in"]}, x)
    #
    #     def _transformer_layer_scan_fn(layer_in, layer_params):
    #         h_, mask_ = layer_in["h"], layer_in["mask"]
    #         h_out = h_ + self.transformer_layer.apply(
    #             {"params": layer_params["transformer"]},
    #             h_, mask_,
    #         )
    #         return {"h": h_out, "mask": mask_}, None
    #
    #     layers_out, _ = jax.lax.scan(
    #         f=_transformer_layer_scan_fn,
    #         init={"h": embedded, "mask": mask},
    #         xs=params["transformer"]
    #     )
    #     return self.embed_out.apply({"params": params["embed_out"]}, layers_out)

    def eval_apply_fn(self, params, x, mask):
        embedded = ShardedEmbedIn(config=self.config).apply({"params": params["embed_in"]}, x)

        def _transformer_layer_scan_fn(layer_in, layer_params):
            h_, mask_ = layer_in["h"], layer_in["mask"]
            h_out = h_ + ShardedTransformerLayer(config=self.config).apply(
                {"params": layer_params},
                h_, mask_,
            )
            return {"h": h_out, "mask": mask_}, None

        layers_out, _ = jax.lax.scan(
            f=_transformer_layer_scan_fn,
            init={"h": embedded, "mask": mask},
            xs=params["transformer"]
        )
        layers_out = layers_out["h"]
        return ShardedEmbedOut(config=self.config).apply({"params": params["embed_out"]}, layers_out)


class ShardedEmbedIn(nn.Module):

    config: NeoX20BConfig

    @nn.compact
    def __call__(self, input_ids):
        onehot_inputs = jax.nn.one_hot(input_ids, self.config.vocab_size, dtype=jnp.float16)
        onehot_inputs = shard_to(onehot_inputs, P("dp", None, "tp"))
        embedded = nn.Dense(
            features=self.config.hidden_size,
            use_bias=False,
            name="embed",
            dtype=jnp.float16,
        )(onehot_inputs)
        return embedded


class ShardedTransformerLayer(nn.Module):
    """Sharded Transformer Layer.

    Note: This doesn't compute the full residual connection x + r(x), only r(x).
          The residual connection will be computed downstream.
    """
    config: NeoX20BConfig

    # noinspection PyAttributeOutsideInit
    def setup(self):
        config = self.config
        self.attn_norm = nn.LayerNorm(epsilon=config.layernorm_epsilon, dtype=jnp.float16)
        self.ff_norm = nn.LayerNorm(epsilon=config.layernorm_epsilon, dtype=jnp.float16)
        self.qkv_proj = nn.Dense(
            config.hidden_size * 3,
            name="qkv_proj",
            dtype=jnp.float16,
        )
        self.output_proj = nn.Dense(
            config.hidden_size,
            name="output_proj",
            dtype=jnp.float16,
        )
        self.ff_up_proj = nn.Dense(
            config.hidden_size * 4,
            name="ff_up_proj",
            dtype=jnp.float16,
        )
        self.ff_down_proj = nn.Dense(
            config.hidden_size,
            name="ff_down_proj",
            dtype=jnp.float16,
        )

    def __call__(self, x, attn_bias):
        """
        :param x: [batch, seq_len, hidden_size]
        :param attn_bias: [*, seq_len, seq_len]
        :return: [batch, seq_len, hidden_size]
        """
        attn_in = self.attn_norm(x)
        # -> [batch, seq_len, hidden_size]

        q, k, v = self.compute_qkv(attn_in)
        # -> 3 x [batch, seq_len, heads, dims_per_head]

        seq_len = attn_in.shape[1]
        causal_mask = np.tril(np.ones((seq_len, seq_len)))[None, :, :]  # NumPy array gets cached
        # -> [1, seq_len, seq_len]

        bias = -1e10 * (1. - causal_mask)
        bias += attn_bias
        # -> [1, seq_len, seq_len]

        attn_out = self.compute_self_attn(q, k, v, bias)
        # -> [batch, seq_len, hidden]

        ff_out = self.compute_ff(x)
        # -> [batch, seq_len, hidden]

        return attn_out + ff_out

    def split_heads(self, x):
        config = self.config
        dims_per_head = config.hidden_size // config.num_attention_heads
        # reshaped = x.reshape(x.shape[:-1] + (heads_per_device, dims_per_head))
        reshaped = x.reshape(x.shape[:-2] + (config.num_attention_heads, dims_per_head))
        # reshaped = reshaped.reshape(x.shape[:-2] + (-1, ) + x.shape[-1:])
        return shard_to(reshaped, P("dp", None, "tp", None))

    def compute_qkv(self, x):
        config = self.config
        # [batch, seq, qkv_dims]
        qkv_arr = self.qkv_proj(x)

        # [batch, seq, mp, dim//mp]
        qkv_arr = shard_to(qkv_arr, P("dp", None, "tp"))
        mp_split = jnp.reshape(qkv_arr, qkv_arr.shape[:-1] + (config.tp_num, -1))
        mp_split = shard_to(mp_split, P("dp", None, "tp", None))

        local_dim = config.hidden_size // config.tp_num

        q, k, v = jnp.split(mp_split, [local_dim, local_dim * 2], axis=-1)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        # ->

        return q, k, v

    def compute_self_attn(self, q, k, v, attn_bias):
        """
        :param q: [batch, q_len, heads, dims_per_head]
        :param k: [batch, kv_len, heads, dims_per_head]
        :param v: [batch, kv_len, heads, dims_per_head]
        :param attn_bias: [*, q_len, kv_len]
        :return: [batch, q_len, hidden]
        """
        config = self.config
        rotary_dims = int(config.hidden_size // config.num_attention_heads * config.rotary_pct)
        k_rot = k[:, :, :, :rotary_dims]
        k_pass = k[:, :, :, rotary_dims:]

        q_rot = q[:, :, :, :rotary_dims]
        q_pass = q[:, :, :, rotary_dims:]

        sincos = fixed_pos_embedding(k_rot, seq_dim=1)
        q_rot = apply_rotary_pos_emb(q_rot, sincos)
        k_rot = apply_rotary_pos_emb(k_rot, sincos)
        q_rot = shard_to(q_rot, P("dp", None, "tp", None))
        k_rot = shard_to(k_rot, P("dp", None, "tp", None))

        k = jnp.concatenate([k_rot, k_pass], axis=-1)
        q = jnp.concatenate([q_rot, q_pass], axis=-1)

        k = shard_to(k, P("dp", None, "tp", None))
        q = shard_to(q, P("dp", None, "tp", None))

        attention_logits = jnp.einsum("bthd,bThd->bhtT", q, k)
        attention_logits = shard_to(attention_logits, P("dp", "tp", None, None))
        # -> [batch, heads, q_len, kv_len]

        sqrt_key_size = np.sqrt(config.hidden_size // config.num_attention_heads).astype(k.dtype)
        attention_logits = attention_logits / sqrt_key_size
        attention_logits += attn_bias
        attention_logits = shard_to(attention_logits, P("dp", "tp", None, None))
        # -> [batch, heads, q_len, kv_len]

        attention_weights = jax.nn.softmax(attention_logits)
        attention_weights = shard_to(attention_weights, P("dp", "tp", None, None))
        # -> [batch, heads, q_len, kv_len]

        attention_vec = jnp.einsum("bhtT,bThd->bthd", attention_weights, v)
        attention_vec = shard_to(attention_vec, P("dp", None, "tp", None))
        # -> [batch, q_len, heads, dims_per_head]

        attention_vec = attention_vec.reshape(attention_vec.shape[:2] + (-1,))
        attention_vec = shard_to(attention_vec, P("dp", None, "tp"))
        # -> [batch, q_len, hidden]

        attn_out = self.output_proj(attention_vec)
        attn_out = shard_to(attn_out, P("dp", None, "tp"))
        # -> [batch, q_len, hidden]

        return attn_out

    def compute_ff(self, x):
        ff_out = self.ff_norm(x)
        ff_out = self.ff_up_proj(ff_out)
        ff_out = shard_to(ff_out, P("dp", None, "tp"))
        ff_out = jax.nn.gelu(ff_out)
        ff_out = self.ff_down_proj(ff_out)
        ff_out = shard_to(ff_out, P("dp", None, None))
        return ff_out

    # iterate the decoding process by a single token
    def decode_once(self, decode_state, x):
        """
        :param decode_state:
        :param x: [batch, q_len=1, hidden_size]
        """
        attn_in = self.attn_norm(x)
        # -> [batch, q_len=1, hidden_size]
        q, v, k = self.compute_qkv(attn_in)
        # -> 3 x [batch, q_len=1, heads, dims_per_head]

        # ?? assert x.shape[0] == 1

        # add new kv to end
        v = jnp.concatenate((decode_state["v"], v), axis=1)[1:]
        # -> [batch, kv_len+1, heads, dims_per_head]
        k = jnp.concatenate((decode_state["k"], k), axis=1)[1:]
        # -> [batch, kv_len+1, heads, dims_per_head]

        tokens_decoded = decode_state["tokens_decoded"] + 1
        # -> [batch]

        length = v.shape[1]
        masked_tokens = (length - tokens_decoded)[:, None]
        # -> [batch, 1]
        attention_mask = (jnp.arange(0, length)[None, :] < masked_tokens)[:, None, :]
        # -> [batch, q_len=1, seq_len]

        bias = (-1e10 * attention_mask)
        # -> [batch, q_len=1, seq_len]

        attn_out = self.compute_self_attn(q, v, k, bias[:, None, :, :])
        # -> 3 x [batch, q_len=1, hidden]

        ff_out = self.compute_ff(x)
        # -> 3 x [batch, q_len=1, hidden]

        return (attn_out + ff_out), {
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
        attn_in = self.attn_norm(x)
        # -> [batch, seq_len, hidden_size]
        q, v, k = self.compute_qkv(attn_in)
        # -> 3 x [batch, seq_len, heads, dims_per_head]

        batch_size, full_length = x.shape[0], x.shape[1]
        masked_tokens = (full_length - given_length)[:, None]
        # -> [batch, 1]

        causal_mask = np.tril(np.ones((full_length, full_length)))
        # -> [seq_len, seq_len]

        bias = -1e10 * (1. - causal_mask)  # regular AR masking
        bias = jnp.repeat(bias[None, :], repeats=batch_size, axis=0)
        # -> [batch, seq_len, seq_len]

        context_length_mask = (jnp.arange(0, full_length)[None, :] < masked_tokens)[:, None, :]
        # -> [batch, 1, seq_len]

        bias -= 1e10 * context_length_mask  # mask out zero tokens before context starts
        # -> [batch, seq_len, seq_len]

        attn_out = self.compute_self_attn(q, v, k, bias[:, None, :, :])
        # -> [batch, seq_len, hidden]

        ff_out = self.compute_ff(x)
        # -> [batch, seq_len, hidden]

        return (attn_out + ff_out), {
            "tokens_decoded": given_length.astype(jnp.uint32),
            "k": k,
            "v": v,
        }


class ShardedEmbedOut(nn.Module):

    config: NeoX20BConfig

    # noinspection PyAttributeOutsideInit
    def setup(self):
        config = self.config
        self.norm = nn.LayerNorm(epsilon=config.layernorm_epsilon, dtype=jnp.float16)
        self.embed_out = nn.Dense(config.vocab_size, use_bias=False, dtype=jnp.float16)

    def __call__(self, x):
        return self.predict(x)

    def predict(self, x):
        x = self.norm(x)
        x = shard_to(x, P("dp", None, None))
        out = self.embed_out(x)
        out = shard_to(out, P("dp", None, "tp"))
        return out

    def loss(self, x, targets, z_loss=1):
        logits = self.predict(x)
        targets_onehot = jax.nn.one_hot(targets, self.dim, dtype=jnp.float16)
        logits_for_targets = jnp.sum(jnp.multiply(targets_onehot, logits), axis=-1)

        # softmax denominator
        exp_logits = jnp.exp(logits)
        sum_exp_logits = exp_logits.sum(axis=-1)

        # compute loss
        loss = jnp.log(sum_exp_logits) - logits_for_targets
        loss += (1e-4 * jnp.square(jnp.log(sum_exp_logits)) * z_loss).mean()
        correct = (0.0 == logits_for_targets)
        return loss, correct


def fixed_pos_embedding(x, seq_dim=0):
    dim = x.shape[-1]
    inv_freq = 1. / (10000 ** (np.arange(0, dim, 2) / dim))
    sinusoid_inp = np.einsum('i , j -> i j', np.arange(x.shape[seq_dim]), inv_freq)
    return np.sin(sinusoid_inp).astype(np.float16), np.cos(sinusoid_inp).astype(np.float16)


def apply_rotary_pos_emb(x, sincos):
    sin, cos = map(lambda t: repeat(t, '... b n -> ... b (n j)', j=2)[-x.shape[-3]:, None, :], sincos)
    return (x * cos) + (rotate_every_two(x) * sin)


def rotate_every_two(x):
    half_dim = x.shape[-1] // 2
    x1 = x[:, :, :, :half_dim]
    x2 = x[:, :, :, half_dim:]
    x = jnp.stack((-x2, x1), axis=-1)
    return rearrange(x, '... d j -> ... (d j)')
