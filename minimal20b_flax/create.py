import os

from multiprocessing import Pool

import jax
from tqdm import auto as tqdm_lib

import numpy as np

# noinspection PyPep8Naming
from jax.experimental import PartitionSpec as P
from jax.experimental.pjit import pjit
from jax.experimental import maps
import jax.numpy as jnp
from flax.core import frozen_dict
from flax import traverse_util

import torch
import tokenizers

import minimal20b_flax.utils as utils
import minimal20b_flax.model as model


def load_model_weights(checkpoint_path, config: model.NeoX20BConfig = model.default_neox20b_config):
    """Loads the weights from a checkpoint and shard to 8 TPU devices."""
    pbar = tqdm_lib.tqdm(total=47)

    # 1. Load embed_in
    pbar.set_description("Loading embed_in")
    loaded_tp1 = load_to_numpy(os.path.join(checkpoint_path, "layer_00-model_00-model_states.pt"))
    loaded_tp2 = load_to_numpy(os.path.join(checkpoint_path, "layer_00-model_01-model_states.pt"))
    shared_embedding = np.concatenate([
        loaded_tp1["word_embeddings.weight"],
        loaded_tp2["word_embeddings.weight"],
    ], axis=0)
    del loaded_tp1
    del loaded_tp2
    # 1.1. Shard to device
    embed_in_params = traverse_util.unflatten_dict({
        ("embed", "kernel"): utils.shard_to_devices(shared_embedding, axis=0),
    })
    pbar.update(1)

    # 2. Load layer weights
    #    These are stacked because we will later run a jax.lax.scan over them to iterate
    #    over layers.
    # Note: this next line loads all the layers into CPU memory, which is a lot.
    layer_params_list = []
    for i in range(config.num_layers):
        pbar.set_description(f"Loading layer {i}")
        layer_params_list.append(traverse_util.flatten_dict(frozen_dict.unfreeze(
           load_single_layer_params(checkpoint_path, i)
        )))
        pbar.update(1)
    # 2.1. Shard to device
    sharding = model.GPTNeoX20BModel.get_sharding()
    flat_stacked_layers_sharding = traverse_util.flatten_dict(frozen_dict.unfreeze(
        sharding["transformer"]))
    pbar.set_description(f"Sharding transformer layers to TPUs")
    stacked_layer_params = {}
    for k, v in layer_params_list[0].items():
        stacked = np.stack([
            layer_params[k]
            for layer_params in layer_params_list
        ], axis=0)
        shard_strategy = flat_stacked_layers_sharding[k]
        if shard_strategy == P(None, None):
            stacked = utils.replicate_to_devices(stacked)
        elif shard_strategy == P(None, None, "tp"):
            stacked = utils.shard_to_devices(stacked, axis=2)
        elif shard_strategy == P(None, "tp", None):
            stacked = utils.shard_to_devices(stacked, axis=1)
        else:
            raise RuntimeError()
        stacked_layer_params[k] = stacked
    stacked_layer_params = frozen_dict.freeze(traverse_util.unflatten_dict(
        stacked_layer_params
    ))
    pbar.update(1)

    # 3. Load final layer norm and embed_out (jointly "embed_out")
    pbar.set_description(f"Load embed_out")
    loaded_tp1 = load_to_numpy(os.path.join(checkpoint_path, "layer_47-model_00-model_states.pt"))
    loaded_tp2 = load_to_numpy(os.path.join(checkpoint_path, "layer_47-model_01-model_states.pt"))
    # noinspection PyDictCreation
    embed_out_params = {
        ("norm", "bias"): (loaded_tp1["norm.bias"] + loaded_tp2["norm.bias"]) / 2,
        ("norm", "scale"): (loaded_tp1["norm.weight"] + loaded_tp2["norm.weight"]) / 2,
    }
    del loaded_tp1
    del loaded_tp2
    loaded_tp1 = load_to_numpy(os.path.join(checkpoint_path, "layer_48-model_00-model_states.pt"))
    loaded_tp2 = load_to_numpy(os.path.join(checkpoint_path, "layer_48-model_01-model_states.pt"))
    embed_out_params["embed_out", "kernel"] = np.concatenate([
        loaded_tp1["final_linear.weight"].T,
        loaded_tp2["final_linear.weight"].T,
    ], axis=1)
    del loaded_tp1
    del loaded_tp2
    # 3.1. Shard to device
    embed_out_params["norm", "bias"] = utils.replicate_to_devices(
        embed_out_params["norm", "bias"])
    embed_out_params["norm", "scale"] = utils.replicate_to_devices(
        embed_out_params["norm", "scale"])
    embed_out_params["embed_out", "kernel"] = utils.shard_to_devices(
        embed_out_params["embed_out", "kernel"], axis=1)
    embed_out_params = frozen_dict.freeze(traverse_util.unflatten_dict(embed_out_params))
    pbar.update(1)
    pbar.set_description("Done.")

    # 4. Combine
    all_params = frozen_dict.freeze({
        "embed_in": embed_in_params,
        "transformer": stacked_layer_params,
        "embed_out": embed_out_params
    })
    return all_params


def load_single_layer_params(checkpoint_path, layer_i):
    filename_tp1 = f"layer_{layer_i + 2:02d}-model_00-model_states.pt"
    filename_tp2 = f"layer_{layer_i + 2:02d}-model_01-model_states.pt"
    loaded_tp1 = load_to_numpy(os.path.join(checkpoint_path, filename_tp1))
    loaded_tp2 = load_to_numpy(os.path.join(checkpoint_path, filename_tp2))
    # noinspection PyDictCreation
    layer_params = {}
    layer_params["attn_norm", "bias"] = (
        loaded_tp1["input_layernorm.bias"]
        + loaded_tp2["input_layernorm.bias"]
    ) / 2
    layer_params["attn_norm", "scale"] = (
        loaded_tp1["input_layernorm.weight"]
        + loaded_tp2["input_layernorm.weight"]
    ) / 2
    layer_params["qkv_proj", "kernel"] = np.concatenate([
        loaded_tp1["attention.query_key_value.weight"].T,
        loaded_tp2["attention.query_key_value.weight"].T,
    ], axis=1).reshape((6144, 8, 8, 3, 96)).swapaxes(2, 3).reshape((6144, 18432))
    # input_dim, num_heads1(tp), numheads2(heads per device), qkv, dim_per_head
    layer_params["qkv_proj", "bias"] = np.concatenate([
        loaded_tp1["attention.query_key_value.bias"],
        loaded_tp2["attention.query_key_value.bias"],
    ]).reshape((8, 8, 3, 96)).swapaxes(1, 2).reshape(18432)
    layer_params["output_proj", "kernel"] = np.concatenate([
        loaded_tp1["attention.dense.weight"].T,
        loaded_tp2["attention.dense.weight"].T,
    ], axis=0)
    layer_params["output_proj", "bias"] = (
        loaded_tp1["attention.dense.bias"]
        + loaded_tp2["attention.dense.bias"]
    )
    layer_params["ff_norm", "bias"] = (
        loaded_tp1["post_attention_layernorm.bias"]
        + loaded_tp2["post_attention_layernorm.bias"]
    ) / 2
    layer_params["ff_norm", "scale"] = (
        loaded_tp1["post_attention_layernorm.weight"]
        + loaded_tp2["post_attention_layernorm.weight"]
    ) / 2
    layer_params["ff_up_proj", "kernel"] = np.concatenate([
        loaded_tp1["mlp.dense_h_to_4h.weight"].T,
        loaded_tp2["mlp.dense_h_to_4h.weight"].T,
    ], axis=1)
    layer_params["ff_up_proj", "bias"] = np.concatenate([
        loaded_tp1["mlp.dense_h_to_4h.bias"],
        loaded_tp2["mlp.dense_h_to_4h.bias"],
    ])
    layer_params["ff_down_proj", "kernel"] = np.concatenate([
        loaded_tp1["mlp.dense_4h_to_h.weight"].T,
        loaded_tp2["mlp.dense_4h_to_h.weight"].T,
    ], axis=0)
    layer_params["ff_down_proj", "bias"] = (
        loaded_tp1["mlp.dense_4h_to_h.bias"]
        + loaded_tp2["mlp.dense_4h_to_h.bias"]
    )
    layer_params = frozen_dict.freeze(traverse_util.unflatten_dict(layer_params))
    del loaded_tp1
    del loaded_tp2
    return layer_params


def load_to_numpy(path, **kwargs):
    return {k: v.numpy() for k, v in torch.load(path, **kwargs).items()}


def create_tokenizer(tokenizer_path):
    return tokenizers.Tokenizer.from_file(tokenizer_path)


# === Colab specific ===

def colab_load_model_weights(checkpoint_path, config: model.NeoX20BConfig = model.default_neox20b_config):
    """Loads the weights from a checkpoint and shard to 8 TPU devices."""
    pbar = tqdm_lib.tqdm(total=311)

    # 1. Load embed_in
    pbar.set_description("Loading embed_in")
    loaded_tp1 = load_to_numpy(os.path.join(checkpoint_path, "layer_00-model_00-model_states.pt"))
    loaded_tp2 = load_to_numpy(os.path.join(checkpoint_path, "layer_00-model_01-model_states.pt"))
    shared_embedding = np.concatenate([
        loaded_tp1["word_embeddings.weight"],
        loaded_tp2["word_embeddings.weight"],
    ], axis=0)
    del loaded_tp1
    del loaded_tp2
    # 1.1. Shard to device
    embed_in_params = traverse_util.unflatten_dict({
        ("embed", "kernel"): utils.shard_to_devices(shared_embedding, axis=0),
    })
    pbar.update(1)

    stacked_layer_params = {}
    sharding = model.GPTNeoX20BModel.get_sharding()
    flat_stacked_layers_sharding = traverse_util.flatten_dict(frozen_dict.unfreeze(
        sharding["transformer"]))

    # 2.1 Preallocate
    def initialize_layer_params():
        shape_dict = {
            ('attn_norm', 'scale'): (44, 6144,),
            ('attn_norm', 'bias'): (44, 6144,),
            ('qkv_proj', 'kernel'): (44, 6144, 18432),
            ('qkv_proj', 'bias'): (44, 18432,),
            ('output_proj', 'kernel'): (44, 6144, 6144),
            ('output_proj', 'bias'): (44, 6144,),
            ('ff_norm', 'scale'): (44, 6144,),
            ('ff_norm', 'bias'): (44, 6144,),
            ('ff_up_proj', 'kernel'): (44, 6144, 24576),
            ('ff_up_proj', 'bias'): (44, 24576,),
            ('ff_down_proj', 'kernel'): (44, 24576, 6144),
            ('ff_down_proj', 'bias'): (44, 6144,),
        }
        layer_params = {}
        for k, v in shape_dict.items():
            layer_params[k] = jnp.zeros(v, dtype=jnp.float16)
        return layer_params

    initialize_layer_params_pjit = pjit(
        initialize_layer_params,
        in_axis_resources=None,
        out_axis_resources=flat_stacked_layers_sharding,
    )
    mesh = utils.get_default_mesh()
    with maps.mesh(mesh.devices, mesh.axis_names):
        pbar.set_description(f"Initializing layer params on device")
        stacked_layer_params = initialize_layer_params_pjit()
        pbar.update(1)

    def assign_to_sharded_device_array(old_state, new_layer, layer_idx):
        new_state = old_state.at[layer_idx].set(new_layer)
        return new_state

    assign_funcs_dict = {}
    for k in stacked_layer_params:
        assign_funcs_dict[k] = pjit(
            assign_to_sharded_device_array,
            in_axis_resources=(
                flat_stacked_layers_sharding[k],
                P(*flat_stacked_layers_sharding[k][1:]),
            ),
            out_axis_resources=flat_stacked_layers_sharding[k],
            donate_argnums=(0,),
            static_argnums=(2,),
        )

    for layer_i in range(config.num_layers):
        pbar.set_description(f"Loading layer {layer_i}")
        single_layer_params = load_single_layer_params(checkpoint_path, layer_i)
        flattened_layer_params = traverse_util.flatten_dict(frozen_dict.unfreeze(single_layer_params))
        with maps.mesh(mesh.devices, mesh.axis_names):
            for k in stacked_layer_params:
                stacked_layer_params[k] = assign_funcs_dict[k](
                    stacked_layer_params[k],
                    flattened_layer_params[k],
                    layer_i,
                )
                pbar.update(1)

    stacked_layer_params = frozen_dict.freeze(traverse_util.unflatten_dict(stacked_layer_params))

    # 3. Load final layer norm and embed_out (jointly "embed_out")
    pbar.set_description(f"Load embed_out")
    loaded_tp1 = load_to_numpy(os.path.join(checkpoint_path, "layer_47-model_00-model_states.pt"))
    loaded_tp2 = load_to_numpy(os.path.join(checkpoint_path, "layer_47-model_01-model_states.pt"))
    # noinspection PyDictCreation
    embed_out_params = {
        ("norm", "bias"): (loaded_tp1["norm.bias"] + loaded_tp2["norm.bias"]) / 2,
        ("norm", "scale"): (loaded_tp1["norm.weight"] + loaded_tp2["norm.weight"]) / 2,
    }
    del loaded_tp1
    del loaded_tp2
    loaded_tp1 = load_to_numpy(os.path.join(checkpoint_path, "layer_48-model_00-model_states.pt"))
    loaded_tp2 = load_to_numpy(os.path.join(checkpoint_path, "layer_48-model_01-model_states.pt"))
    embed_out_params["embed_out", "kernel"] = np.concatenate([
        loaded_tp1["final_linear.weight"].T,
        loaded_tp2["final_linear.weight"].T,
    ], axis=1)
    del loaded_tp1
    del loaded_tp2
    # 3.1. Shard to device
    embed_out_params["norm", "bias"] = utils.replicate_to_devices(
        embed_out_params["norm", "bias"])
    embed_out_params["norm", "scale"] = utils.replicate_to_devices(
        embed_out_params["norm", "scale"])
    embed_out_params["embed_out", "kernel"] = utils.shard_to_devices(
        embed_out_params["embed_out", "kernel"], axis=1)
    embed_out_params = frozen_dict.freeze(traverse_util.unflatten_dict(embed_out_params))
    pbar.update(1)
    pbar.set_description("Done.")

    # 4. Combine
    all_params = frozen_dict.freeze({
        "embed_in": embed_in_params,
        "transformer": stacked_layer_params,
        "embed_out": embed_out_params
    })
    return all_params


def colab_load_single_layer_params(checkpoint_path, layer_i):
    filename_tp1 = f"layer_{layer_i + 2:02d}-model_00-model_states.pt"
    filename_tp2 = f"layer_{layer_i + 2:02d}-model_01-model_states.pt"
    loaded_tp1 = load_to_numpy(os.path.join(checkpoint_path, filename_tp1))
    loaded_tp2 = load_to_numpy(os.path.join(checkpoint_path, filename_tp2))
    # noinspection PyDictCreation
    layer_params = {}
    layer_params["attn_norm", "bias"] = (
        loaded_tp1["input_layernorm.bias"]
        + loaded_tp2["input_layernorm.bias"]
    ) / 2
    layer_params["attn_norm", "scale"] = (
        loaded_tp1["input_layernorm.weight"]
        + loaded_tp2["input_layernorm.weight"]
    ) / 2
    layer_params["qkv_proj", "kernel"] = np.concatenate([
        loaded_tp1["attention.query_key_value.weight"].T,
        loaded_tp2["attention.query_key_value.weight"].T,
    ], axis=1).reshape((6144, 8, 8, 3, 96)).swapaxes(2, 3).reshape((6144, 18432))
    # input_dim, num_heads1(tp), numheads2(heads per device), qkv, dim_per_head
    layer_params["qkv_proj", "bias"] = np.concatenate([
        loaded_tp1["attention.query_key_value.bias"],
        loaded_tp2["attention.query_key_value.bias"],
    ]).reshape((8, 8, 3, 96)).swapaxes(1, 2).reshape(18432)
    layer_params["output_proj", "kernel"] = np.concatenate([
        loaded_tp1["attention.dense.weight"].T,
        loaded_tp2["attention.dense.weight"].T,
    ], axis=0)
    layer_params["output_proj", "bias"] = (
        loaded_tp1["attention.dense.bias"]
        + loaded_tp2["attention.dense.bias"]
    )
    layer_params["ff_norm", "bias"] = (
        loaded_tp1["post_attention_layernorm.bias"]
        + loaded_tp2["post_attention_layernorm.bias"]
    ) / 2
    layer_params["ff_norm", "scale"] = (
        loaded_tp1["post_attention_layernorm.weight"]
        + loaded_tp2["post_attention_layernorm.weight"]
    ) / 2
    layer_params["ff_up_proj", "bias"] = np.concatenate([
        loaded_tp1["mlp.dense_h_to_4h.bias"],
        loaded_tp2["mlp.dense_h_to_4h.bias"],
    ])
    layer_params["ff_down_proj", "bias"] = (
        loaded_tp1["mlp.dense_4h_to_h.bias"]
        + loaded_tp2["mlp.dense_4h_to_h.bias"]
    )
    layer_params = frozen_dict.freeze(traverse_util.unflatten_dict(layer_params))
    del loaded_tp1
    del loaded_tp2
    return layer_params


def colab_load_single_layer_qkv_kernel_params(checkpoint_path, layer_i, original_shard: int):
    filename = f"layer_{layer_i + 2:02d}-model_{original_shard:02d}-model_states.pt"
    loaded = load_to_numpy(os.path.join(checkpoint_path, filename))
    return loaded["attention.query_key_value.weight"].T.reshape(
        (6144, 4, 8, 3, 96)
    ).swapaxes(2, 3).reshape((6144, 9216))


def colab_load_single_layer_ff_up_kernel_params(checkpoint_path, layer_i, original_shard: int):
    filename = f"layer_{layer_i + 2:02d}-model_{original_shard:02d}-model_states.pt"
    loaded = load_to_numpy(os.path.join(checkpoint_path, filename))
    return loaded["mlp.dense_h_to_4h.weight"].T


def colab_load_single_layer_ff_down_kernel_params(checkpoint_path, layer_i, original_shard: int):
    filename = f"layer_{layer_i + 2:02d}-model_{original_shard:02d}-model_states.pt"
    loaded = load_to_numpy(os.path.join(checkpoint_path, filename))
    return loaded["mlp.dense_4h_to_h.weight"].T


# === Xmap specific ===

def load_single_layer_params_for_xmap(checkpoint_path, layer_i):
    num_shards = 8
    transformer_layer_params = load_single_layer_params(checkpoint_path=checkpoint_path, layer_i=layer_i)
    fparams = frozen_dict.unfreeze(traverse_util.flatten_dict(transformer_layer_params))
    # noinspection PyDictCreation
    new_fparams = {}
    new_fparams[('attn_norm', 'bias')] = \
        stack_copies(fparams[('attn_norm', 'bias')], num_shards, axis=0)
    new_fparams[('attn_norm', 'scale')] = \
        stack_copies(fparams[('attn_norm', 'scale')], num_shards, axis=0)
    new_fparams[('ff_down_proj', 'bias')] = \
        stack_copies(fparams[('ff_down_proj', 'bias')], num_shards, axis=0) / 8
    new_fparams[('ff_down_proj', 'kernel')] = \
        fparams[('ff_down_proj', 'kernel')].reshape(8, 3072, 6144)
    new_fparams[('ff_norm', 'bias')] = \
        stack_copies(fparams[('ff_norm', 'bias')], num_shards, axis=0)
    new_fparams[('ff_norm', 'scale')] = \
        stack_copies(fparams[('ff_norm', 'scale')], num_shards, axis=0)
    new_fparams[('ff_up_proj', 'bias')] = \
        fparams[('ff_up_proj', 'bias')].reshape(8, 3072)
    new_fparams[('ff_up_proj', 'kernel')] = \
        fparams[('ff_up_proj', 'kernel')].reshape(6144, 8, 3072).swapaxes(0, 1)
    new_fparams[('output_proj', 'bias')] = \
        stack_copies(fparams[('output_proj', 'bias')], num_shards, axis=0) / 8
    new_fparams[('output_proj', 'kernel')] = \
        fparams[('output_proj', 'kernel')].reshape(8, 768, 6144)
    new_fparams[('qkv_proj', 'bias')] = \
        fparams[('qkv_proj', 'bias')].reshape(8, 2304)
    new_fparams[('qkv_proj', 'kernel')] = \
        fparams[('qkv_proj', 'kernel')].reshape(6144, 8, 2304).swapaxes(0, 1)
    return new_fparams


def load_model_weights_for_xmap(
        checkpoint_path,
        config: model.NeoX20BConfig = model.default_neox20b_config,
        pool_size=None):
    flattened_params = {}
    loaded_tp1 = load_to_numpy(os.path.join(checkpoint_path, "layer_00-model_00-model_states.pt"))
    loaded_tp2 = load_to_numpy(os.path.join(checkpoint_path, "layer_00-model_01-model_states.pt"))
    embed_in = np.concatenate([
        loaded_tp1["word_embeddings.weight"],
        loaded_tp2["word_embeddings.weight"],
    ], axis=0)
    flattened_params[('embed_in', 'embed', 'kernel')] = embed_in.reshape(8, 6304, 6144)
    del loaded_tp1
    del loaded_tp2

    loaded_tp1 = load_to_numpy(os.path.join(checkpoint_path, "layer_47-model_00-model_states.pt"))
    loaded_tp2 = load_to_numpy(os.path.join(checkpoint_path, "layer_47-model_01-model_states.pt"))
    # noinspection PyDictCreation
    final_norm_bias = (loaded_tp1["norm.bias"] + loaded_tp2["norm.bias"]) / 2
    final_norm_scale = (loaded_tp1["norm.weight"] + loaded_tp2["norm.weight"]) / 2
    num_shards = 8
    flattened_params[('embed_out', 'norm', 'bias')] = \
        stack_copies(final_norm_bias, num_shards, axis=0)
    flattened_params[('embed_out', 'norm', 'scale')] = \
        stack_copies(final_norm_scale, num_shards, axis=0)
    del loaded_tp1
    del loaded_tp2

    loaded_tp1 = load_to_numpy(os.path.join(checkpoint_path, "layer_48-model_00-model_states.pt"))
    loaded_tp2 = load_to_numpy(os.path.join(checkpoint_path, "layer_48-model_01-model_states.pt"))
    embed_out = np.concatenate([
        loaded_tp1["final_linear.weight"].T,
        loaded_tp2["final_linear.weight"].T,
    ], axis=1)
    flattened_params[('embed_out', 'embed_out', 'kernel')] = \
        embed_out.reshape(6144, 8, 6304).swapaxes(0, 1)
    del loaded_tp1
    del loaded_tp2

    if pool_size is None:
        for layer_i in tqdm_lib.trange(config.num_layers):
            layer_params = load_single_layer_params_for_xmap(
                checkpoint_path=checkpoint_path,
                layer_i=layer_i,
            )
            for k, v in layer_params.items():
                flattened_params[(f"layer_{layer_i:02d}",) + k] = v
    else:
        pool = Pool(processes=pool_size)

        pool_args = [(checkpoint_path, layer_i) for layer_i in range(config.num_layers)]
        for layer_i, layer_params in tqdm_lib.tqdm(pool.imap(pool_func, pool_args), total=len(pool_args)):
            for k, v in layer_params.items():
                flattened_params[(f"layer_{layer_i:02d}",) + k] = v

    params = traverse_util.unflatten_dict(flattened_params)
    return params


def colab_load_model_weights_for_xmap(
        checkpoint_path,
        config: model.NeoX20BConfig = model.default_neox20b_config):
    mesh = get_colab_mesh()
    pbar = tqdm_lib.tqdm(total=47)

    # Embedding
    pbar.set_description("Loading embed_in")
    flattened_params = {}
    loaded_tp1 = load_to_numpy(os.path.join(checkpoint_path, "layer_00-model_00-model_states.pt"))
    loaded_tp2 = load_to_numpy(os.path.join(checkpoint_path, "layer_00-model_01-model_states.pt"))
    embed_in = np.concatenate([
        loaded_tp1["word_embeddings.weight"],
        loaded_tp2["word_embeddings.weight"],
    ], axis=0)
    flattened_params[('embed_in', 'embed', 'kernel')] = shard_to_devices_v2(
        embed_in.reshape(8, 6304, 6144), mesh=mesh)
    del loaded_tp1
    del loaded_tp2
    pbar.update(1)

    # Load layers
    for layer_i in range(config.num_layers):
        pbar.set_description(f"Loading layer {layer_i}")
        layer_params = load_single_layer_params_for_xmap(
            checkpoint_path=checkpoint_path,
            layer_i=layer_i,
        )
        for k, v in layer_params.items():
            flattened_params[(f"layer_{layer_i:02d}",) + k] = shard_to_devices_v2(v, mesh=mesh)
        del layer_params
        pbar.update(1)

    # Final layer norm
    pbar.set_description(f"Load final layer norm")
    loaded_tp1 = load_to_numpy(os.path.join(checkpoint_path, "layer_47-model_00-model_states.pt"))
    loaded_tp2 = load_to_numpy(os.path.join(checkpoint_path, "layer_47-model_01-model_states.pt"))
    # noinspection PyDictCreation
    final_norm_bias = (loaded_tp1["norm.bias"] + loaded_tp2["norm.bias"]) / 2
    final_norm_scale = (loaded_tp1["norm.weight"] + loaded_tp2["norm.weight"]) / 2
    num_shards = 8
    flattened_params[('embed_out', 'norm', 'bias')] = shard_to_devices_v2(
        stack_copies(final_norm_bias, num_shards, axis=0), mesh=mesh)
    flattened_params[('embed_out', 'norm', 'scale')] = shard_to_devices_v2(
        stack_copies(final_norm_scale, num_shards, axis=0), mesh=mesh)
    del loaded_tp1
    del loaded_tp2
    pbar.update(1)

    # Output embeddings
    pbar.set_description(f"Load embed_out")
    loaded_tp1 = load_to_numpy(os.path.join(checkpoint_path, "layer_48-model_00-model_states.pt"))
    loaded_tp2 = load_to_numpy(os.path.join(checkpoint_path, "layer_48-model_01-model_states.pt"))
    embed_out = np.concatenate([
        loaded_tp1["final_linear.weight"].T,
        loaded_tp2["final_linear.weight"].T,
    ], axis=1)
    flattened_params[('embed_out', 'embed_out', 'kernel')] = shard_to_devices_v2(
        embed_out.reshape(6144, 8, 6304).swapaxes(0, 1), mesh=mesh)
    del loaded_tp1
    del loaded_tp2
    pbar.update(1)

    params = traverse_util.unflatten_dict(flattened_params)
    return params


def get_colab_mesh():
    return maps.Mesh(np.asarray(jax.local_devices()).reshape(1, 8), ('dp', 'tp'))


def identity(x):
    return x


def shard_to_devices_v2(x, mesh):
    shard_to = jax.experimental.maps.xmap(
        identity,
        in_axes=["shard", ...],
        out_axes=["shard", ...],
        axis_resources={'shard': 'tp', 'batch': 'dp'},
    )
    with mesh:
        return shard_to(x)


def shard_to_devices_v3(x):
    assert x.shape[0] == 8
    return jax.device_put_sharded(list(x), devices=jax.local_devices())


def pool_func(arg):
    checkpoint_path, layer_i = arg
    loaded_layer = load_single_layer_params_for_xmap(
        checkpoint_path=checkpoint_path,
        layer_i=layer_i,
    )
    return layer_i, loaded_layer


def stack_copies(x, num, axis=0):
    return np.stack([x] * num, axis=axis)