import numpy as np
from einops import repeat

import jax
import jax.numpy as jnp
# noinspection PyPep8Naming
from jax.experimental import PartitionSpec as P
from jax.experimental.pjit import pjit

import flax.linen as nn
from flax.linen.partitioning import with_sharding_constraint as shard_to
from flax import struct
from flax.core import frozen_dict
from flax import traverse_util


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

    def _eval_apply_fn(self, params, x, mask):
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

    def eval_apply_fn_pjit(self):
        return pjit(
            self._eval_apply_fn,
            in_axis_resources=(
                self.get_sharding(),  # params
                P("dp", None),  # input [batch, seq_len]
                P("dp", None, None),  # mask [batch, seq_len, seq_len]
            ),
            out_axis_resources=P("dp", None, "tp"),  # [batch, seq_len, hidden]
        )

    def _get_initial_decode_state(self, params, ctx, ctx_length):
        # Embed initial context
        embedded = ShardedEmbedIn(config=self.config).apply(
            {"params": params["embed_in"]}, ctx)

        # Set up scan function for creating decode_states for each layer
        def _transformer_layer_init_decode_scan_fn(layer_in, layer_params):
            h_, ctx_length_ = layer_in["h"], layer_in["ctx_length"]
            new_residual, decode_state = ShardedTransformerLayer(config=self.config).apply(
                {"params": layer_params},
                h_, ctx_length,
                method=ShardedTransformerLayer.get_init_decode_state,
            )
            h_out = h_ + new_residual
            return {"h": h_out, "ctx_length": ctx_length_}, decode_state

        # Run scan over transformer layers
        layers_out, init_state = jax.lax.scan(
            f=_transformer_layer_init_decode_scan_fn,
            init={"h": embedded, "ctx_length": ctx_length},
            xs=params["transformer"],
        )
        final_logit = ShardedEmbedOut(config=self.config).apply(
            {"params": params["embed_out"]},
            layers_out["h"][:, -1:, :],
        )

        return {"logits": final_logit, "decode_state": init_state}

    def get_initial_decode_state_pjit(self):
        return pjit(
            self._get_initial_decode_state,
            in_axis_resources=(
                self.get_sharding(),
                P("dp", None),  # input_ids [batch, seq_len]
                P("dp"),  # ctx_length [batch]
            ),
            out_axis_resources={
                "logits": P("dp", None, "tp"),
                "decode_state": self.get_decode_state_sharding(),
            }
        )

    def _decode_once(self, params, single_x, decode_state):
        assert single_x.shape[1] == 1
        # Embed single token
        embedded = ShardedEmbedIn(config=self.config).apply(
            {"params": params["embed_in"]}, single_x)

        # Set up scan function for doing a single decoding step for each layer
        def _transformer_layer_decode_once_scan_fn(h, layer_params_and_decode_state):
            layer_params, layer_decode_state = layer_params_and_decode_state
            new_residual, new_layer_decode_state = ShardedTransformerLayer(config=self.config).apply(
                {"params": layer_params},
                layer_decode_state, h,
                method=ShardedTransformerLayer.decode_once,
            )
            h_out = h + new_residual
            return h_out, new_layer_decode_state

        # Run scan over transformer layers
        layers_out, new_decode_state = jax.lax.scan(
            f=_transformer_layer_decode_once_scan_fn,
            init=embedded,
            xs=(params["transformer"], decode_state),
        )

        # Project to logits
        logits = ShardedEmbedOut(config=self.config).apply(
            {"params": params["embed_out"]},
            layers_out,
        )
        return {
            "logits": logits,
            "new_decode_state": new_decode_state,
        }

    def decode_once_pjit(self):
        decode_state_sharding = self.get_decode_state_sharding()
        return pjit(
            self._decode_once,
            in_axis_resources=(
                self.get_sharding(),
                P("dp"),  # input_ids [batch, seq_len]
                decode_state_sharding,  # decode_state
            ),
            out_axis_resources={
                "logits": P("dp", None, "tp"),
                "new_decode_state": decode_state_sharding,
            }
        )

    def _generate(self, params, ctx, ctx_length, rng, generate_length, sampler_args):
        init_out = self._get_initial_decode_state(
            params=params,
            ctx=ctx,
            ctx_length=ctx_length
        )
        # Add sampling logic here
        initial_token = init_out["logits"].argmax(-1)

        init_carry = {
            "single_x": initial_token,
            "decode_state": init_out["decode_state"],
        }

        def _decode_once_scan_fn(decode_carry, step_rng):
            decode_out = self._decode_once(
                params=params,
                single_x=decode_carry["single_x"],
                decode_state=decode_carry["decode_state"],
            )

            # Add sampling logic here
            # next_token = decode_out["logits"].argmax(-1)
            next_token = temperature_sample(
                key=step_rng,
                logits=decode_out["logits"],
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
            xs=jax.random.split(rng, generate_length),
        )
        tokens = generation_outputs["next_token"].swapaxes(0, 1)[:, :, 0]
        logits = generation_outputs["logits"].swapaxes(0, 1)[:, :, 0]
        return {
            # "final_state": final_state,
            "generated_logits": jnp.concatenate((init_out["logits"], logits), axis=1),
            "generated_tokens": jnp.concatenate((initial_token, tokens), axis=1, ),
        }

    def generate_pjit(self):
        return pjit(
            self._generate,
            in_axis_resources=(
                self.get_sharding(),
                P("dp", None),  # ctx [batch, seq_len]
                P("dp"),  # ctx_length [batch]
                None,
            ),
            out_axis_resources={
                "generated_logits": P("dp", None, "tp"),
                "generated_tokens": P("dp"),
            },
            static_argnums=(4, 5),
        )

    @staticmethod
    def get_decode_state_sharding():
        return {
            "tokens_decoded": P(None, "dp"),  # [num_layers, batch]
            "k": P(None, "dp", None, "tp", None),  # [num_layers, batch, seq_len, heads, dim_per_head]
            "v": P(None, "dp", None, "tp", None),  # [num_layers, batch, seq_len, heads, dim_per_head]
        }

    @staticmethod
    def get_sharding():
        # 1. embed_in sharding
        embed_in_sharding = frozen_dict.freeze(traverse_util.unflatten_dict({
            ("embed", "kernel"): P("tp", None),
        }))

        # 2. layer_sharding
        flat_stacked_layers_sharding = {
            ('attn_norm', 'bias'): P(None, None, ),
            ('attn_norm', 'scale'): P(None, None, ),
            ('qkv_proj', 'bias'): P(None, None, ),
            ('qkv_proj', 'kernel'): P(None, None, 'tp'),
            ('output_proj', 'bias'): P(None, None, ),
            ('output_proj', 'kernel'): P(None, 'tp', None),
            ('ff_norm', 'bias'): P(None, None, ),
            ('ff_norm', 'scale'): P(None, None, ),
            ('ff_up_proj', 'bias'): P(None, None, ),
            ('ff_up_proj', 'kernel'): P(None, None, 'tp'),
            ('ff_down_proj', 'bias'): P(None, None),
            ('ff_down_proj', 'kernel'): P(None, 'tp', None),
        }
        stacked_layers_sharding = frozen_dict.freeze(traverse_util.unflatten_dict(
            flat_stacked_layers_sharding))

        # 3. embed_out sharding
        embed_out_sharding = {
            ('norm', 'bias'): P(None),
            ('norm', 'scale'): P(None),
            ('embed_out', 'kernel'): P(None, "tp"),
        }
        embed_out_sharding = frozen_dict.freeze(traverse_util.unflatten_dict(embed_out_sharding))

        # 4. Combine
        all_sharding = frozen_dict.freeze({
            "embed_in": embed_in_sharding,
            "transformer": stacked_layers_sharding,
            "embed_out": embed_out_sharding,
        })
        return all_sharding


class ShardedEmbedIn(nn.Module):

    config: NeoX20BConfig = default_neox20b_config

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
    config: NeoX20BConfig = default_neox20b_config

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

        bias = -1e4 * (1. - causal_mask)
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
        # return sincos
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

        # add new kv to end, clip off the start
        v = jnp.concatenate((decode_state["v"], v), axis=1)[:, 1:]
        # -> [batch, kv_len+1, heads, dims_per_head]
        k = jnp.concatenate((decode_state["k"], k), axis=1)[:, 1:]
        # -> [batch, kv_len+1, heads, dims_per_head]

        tokens_decoded = decode_state["tokens_decoded"] + 1
        # -> [batch]

        length = v.shape[1]
        masked_tokens = (length - tokens_decoded)[:, None]
        # -> [batch, 1]
        attention_mask = (jnp.arange(0, length)[None, :] < masked_tokens)[:, None, :]
        # -> [batch, q_len=1, seq_len]

        bias = (-1e4 * attention_mask)
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

        bias = -1e4 * (1. - causal_mask)  # regular AR masking
        bias = jnp.repeat(bias[None, :], repeats=batch_size, axis=0)
        # -> [batch, seq_len, seq_len]

        context_length_mask = (jnp.arange(0, full_length)[None, :] < masked_tokens)[:, None, :]
        # -> [batch, 1, seq_len]

        bias -= 1e4 * context_length_mask  # mask out zero tokens before context starts
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

    config: NeoX20BConfig = default_neox20b_config

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
    inv_freq = 1. / (10000 ** (np.arange(0, dim, 2) / dim)) .astype(np.float16)
    sinusoid_inp = np.einsum('i , j -> i j', np.arange(x.shape[seq_dim]), inv_freq) .astype(np.float16)
    return np.sin(sinusoid_inp).astype(np.float16), np.cos(sinusoid_inp).astype(np.float16)


def apply_rotary_pos_emb(x, sincos):
    sin, cos = map(lambda t: repeat(t, '... b n -> ... b (j n)', j=2)[-x.shape[-3]:, None, :], sincos)
    return (x * cos) + (rotate_half(x) * sin)


def rotate_half(x):
    half_dim = x.shape[-1] // 2
    x1 = x[:, :, :, :half_dim]
    x2 = x[:, :, :, half_dim:]
    return jnp.concatenate((-x2, x1), axis=-1)


def temperature_sample(key, logits, temp=1):
    return jax.random.categorical(key, logits/temp, -1).astype(jnp.int32)
