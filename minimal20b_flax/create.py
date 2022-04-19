import os
from tqdm import auto as tqdm_lib

import numpy as np

# noinspection PyPep8Naming
from jax.experimental import PartitionSpec as P
from flax.core import frozen_dict
from flax import traverse_util

import torch
import tokenizers

import minimal20b_flax.utils as utils
import minimal20b_flax.model as model


def load_model_weights(checkpoint_path):
    """Loads the weights from a checkpoint and shard to 8 TPU devices."""
    config = model.NeoX20BConfig()
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
    sharding = get_sharding()
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
        ('norm', 'scale'): (loaded_tp1["norm.weight"] + loaded_tp2["norm.weight"]) / 2,
    }
    del loaded_tp1
    del loaded_tp2
    loaded_tp1 = load_to_numpy(os.path.join(checkpoint_path, "layer_48-model_00-model_states.pt"))
    loaded_tp2 = load_to_numpy(os.path.join(checkpoint_path, "layer_48-model_01-model_states.pt"))
    embed_out_params['embed_out', 'kernel'] = np.concatenate([
        loaded_tp1["final_linear.weight"].T,
        loaded_tp2["final_linear.weight"].T,
    ], axis=1)
    del loaded_tp1
    del loaded_tp2
    # 3.1. Shard to device
    embed_out_params['norm', 'bias'] = utils.replicate_to_devices(
        embed_out_params['norm', 'bias'])
    embed_out_params['norm', 'scale'] = utils.replicate_to_devices(
        embed_out_params['norm', 'scale'])
    embed_out_params['embed_out', 'kernel'] = utils.shard_to_devices(
        embed_out_params['embed_out', 'kernel'], axis=1)
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
        ('ff_down_proj', 'bias'): P(None, None, ),
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


def load_single_layer_params(checkpoint_path, layer_i):
    filename_tp1 = f"layer_{layer_i + 2:02d}-model_00-model_states.pt"
    filename_tp2 = f"layer_{layer_i + 2:02d}-model_01-model_states.pt"
    loaded_tp1 = load_to_numpy(os.path.join(checkpoint_path, filename_tp1))
    loaded_tp2 = load_to_numpy(os.path.join(checkpoint_path, filename_tp2))
    # noinspection PyDictCreation
    layer_params = {}
    layer_params['attn_norm', 'bias'] = (
        loaded_tp1["input_layernorm.bias"]
        + loaded_tp2["input_layernorm.bias"]
    ) / 2
    layer_params['attn_norm', 'scale'] = (
        loaded_tp1["input_layernorm.weight"]
        + loaded_tp2["input_layernorm.weight"]
    ) / 2
    layer_params['qkv_proj', 'kernel'] = np.concatenate([
        loaded_tp1["attention.query_key_value.weight"].T,
        loaded_tp2["attention.query_key_value.weight"].T,
    ], axis=1).reshape((6144, 8, 8, 3, 96)).swapaxes(2, 3).reshape((6144, 18432))
    # input_dim, num_heads1(tp), numheads2(heads per device), qkv, dim_per_head
    layer_params['qkv_proj', 'bias'] = np.concatenate([
        loaded_tp1["attention.query_key_value.bias"],
        loaded_tp2["attention.query_key_value.bias"],
    ]).reshape((8, 8, 3, 96)).swapaxes(1, 2).reshape(18432)
    layer_params['output_proj', 'kernel'] = np.concatenate([
        loaded_tp1["attention.dense.weight"].T,
        loaded_tp2["attention.dense.weight"].T,
    ], axis=0)
    layer_params['output_proj', 'bias'] = (
        loaded_tp1["attention.dense.bias"]
        + loaded_tp2["attention.dense.bias"]
    )
    layer_params['ff_norm', 'bias'] = (
        loaded_tp1["post_attention_layernorm.bias"]
        + loaded_tp2["post_attention_layernorm.bias"]
    ) / 2
    layer_params['ff_norm', 'scale'] = (
        loaded_tp1["post_attention_layernorm.weight"]
        + loaded_tp2["post_attention_layernorm.weight"]
    ) / 2
    layer_params['ff_up_proj', 'kernel'] = np.concatenate([
        loaded_tp1["mlp.dense_h_to_4h.weight"].T,
        loaded_tp2["mlp.dense_h_to_4h.weight"].T,
    ], axis=1)
    layer_params['ff_up_proj', 'bias'] = np.concatenate([
        loaded_tp1["mlp.dense_h_to_4h.bias"],
        loaded_tp2["mlp.dense_h_to_4h.bias"],
    ])
    layer_params['ff_down_proj', 'kernel'] = np.concatenate([
        loaded_tp1["mlp.dense_4h_to_h.weight"].T,
        loaded_tp2["mlp.dense_4h_to_h.weight"].T,
    ], axis=0)
    layer_params['ff_down_proj', 'bias'] = (
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
