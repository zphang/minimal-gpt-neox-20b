import os
from tqdm import auto as tqdm_lib

import torch

import minimal20b.model as model20b
from minimal20b.constants import Args20b, ArgsDummy


def create_model(checkpoint_path, use_cache=False, device=torch.device("cuda:0")):
    """
    To prevent allocation memory on CPU, we initialize on 'meta' and individually
    port each module over to 'device' as we load each state dict.

    :param checkpoint_path: Path to the checkpoint folder
    :param use_cache: whether to use cache (i.e. for efficient generation)
    :param device: device that you want the model to end up on
    :return: model
    """
    # Instantiate model
    pbar = tqdm_lib.tqdm(total=48)
    pbar.set_description("Instantiating model (~1 min)")
    model = model20b.NeoX20BModel(Args20b, use_cache=use_cache, device="meta")
    model = model.half().to_empty(device=device)
    pbar.update(1)

    # Load transformer layers
    for layer_i in range(Args20b.num_layers):
        pbar.set_description(f"Loading layer {layer_i}")
        filename_tp1 = f"layer_{layer_i + 2:02d}-model_00-model_states.pt"
        filename_tp2 = f"layer_{layer_i + 2:02d}-model_01-model_states.pt"
        loaded_tp1 = torch.load(os.path.join(checkpoint_path, filename_tp1))
        loaded_tp2 = torch.load(os.path.join(checkpoint_path, filename_tp2))
        state_dict = {}
        # Good
        # Keys where we concatenate on the second dim
        for key in [
            "attention.dense.weight",
            "mlp.dense_4h_to_h.weight",
        ]:
            state_dict[key] = torch.cat([loaded_tp1[key], loaded_tp2[key]], dim=1)
        # Mapping individual split weights to custom split implementations
        # Layer Norms
        state_dict["input_layernorm.layer_norm_replica_1.weight"] = loaded_tp1["input_layernorm.weight"]
        state_dict["input_layernorm.layer_norm_replica_2.weight"] = loaded_tp2["input_layernorm.weight"]
        state_dict["input_layernorm.layer_norm_replica_1.bias"] = loaded_tp1["input_layernorm.bias"]
        state_dict["input_layernorm.layer_norm_replica_2.bias"] = loaded_tp2["input_layernorm.bias"]
        state_dict["post_attention_layernorm.layer_norm_replica_1.weight"] = \
            loaded_tp1["post_attention_layernorm.weight"]
        state_dict["post_attention_layernorm.layer_norm_replica_2.weight"] = \
            loaded_tp2["post_attention_layernorm.weight"]
        state_dict["post_attention_layernorm.layer_norm_replica_1.bias"] = loaded_tp1["post_attention_layernorm.bias"]
        state_dict["post_attention_layernorm.layer_norm_replica_2.bias"] = loaded_tp2["post_attention_layernorm.bias"]
        # LinearWithTPMerge
        state_dict["mlp.dense_h_to_4h.linear_split_1.weight"] = loaded_tp1["mlp.dense_h_to_4h.weight"]
        state_dict["mlp.dense_h_to_4h.linear_split_2.weight"] = loaded_tp2["mlp.dense_h_to_4h.weight"]
        state_dict["mlp.dense_h_to_4h.linear_split_1.bias"] = loaded_tp1["mlp.dense_h_to_4h.bias"]
        state_dict["mlp.dense_h_to_4h.linear_split_2.bias"] = loaded_tp2["mlp.dense_h_to_4h.bias"]
        state_dict["attention.query_key_value.linear_split_1.weight"] = loaded_tp1["attention.query_key_value.weight"]
        state_dict["attention.query_key_value.linear_split_2.weight"] = loaded_tp2["attention.query_key_value.weight"]
        state_dict["attention.query_key_value.linear_split_1.bias"] = loaded_tp1["attention.query_key_value.bias"]
        state_dict["attention.query_key_value.linear_split_2.bias"] = loaded_tp2["attention.query_key_value.bias"]
        # LinearWithTPSplitBias
        state_dict["mlp.dense_4h_to_h.bias_replica_1"] = loaded_tp1["mlp.dense_4h_to_h.bias"]
        state_dict["mlp.dense_4h_to_h.bias_replica_2"] = loaded_tp2["mlp.dense_4h_to_h.bias"]
        state_dict["attention.dense.bias_replica_1"] = loaded_tp1["attention.dense.bias"]
        state_dict["attention.dense.bias_replica_2"] = loaded_tp2["attention.dense.bias"]
        # Just take one
        state_dict["attention.rotary_emb.inv_freq"] = loaded_tp1["attention.rotary_emb.inv_freq"]
        model.layer_list[layer_i].load_state_dict(state_dict)
        del loaded_tp1
        del loaded_tp2
        pbar.update(1)

    # Load input embedding
    pbar.set_description(f"Loading input embedding")
    loaded_tp1 = torch.load(os.path.join(checkpoint_path, "layer_00-model_00-model_states.pt"))
    loaded_tp2 = torch.load(os.path.join(checkpoint_path, "layer_00-model_01-model_states.pt"))
    model.embed_in.load_state_dict({"weight": torch.cat([
        loaded_tp1["word_embeddings.weight"],
        loaded_tp2["word_embeddings.weight"],
    ], dim=0)})
    del loaded_tp1
    del loaded_tp2
    pbar.update(1)

    # Load final layer norm
    pbar.set_description(f"Loading final layer norm")
    loaded_tp1 = torch.load(os.path.join(checkpoint_path, "layer_47-model_00-model_states.pt"))
    loaded_tp2 = torch.load(os.path.join(checkpoint_path, "layer_47-model_01-model_states.pt"))
    model.final_layer_norm.load_state_dict({
        "layer_norm_replica_1.weight": loaded_tp1["norm.weight"],
        "layer_norm_replica_2.weight": loaded_tp2["norm.weight"],
        "layer_norm_replica_1.bias": loaded_tp1["norm.bias"],
        "layer_norm_replica_2.bias": loaded_tp2["norm.bias"],
    })
    del loaded_tp1
    del loaded_tp2
    pbar.update(1)

    # Load output embedding
    pbar.set_description(f"Loading output embedding")
    loaded_tp1 = torch.load(os.path.join(checkpoint_path, "layer_48-model_00-model_states.pt"))
    loaded_tp2 = torch.load(os.path.join(checkpoint_path, "layer_48-model_01-model_states.pt"))
    model.logits_out.load_state_dict({
        "linear_split_1.weight": loaded_tp1["final_linear.weight"],
        "linear_split_2.weight": loaded_tp2["final_linear.weight"],
    })
    del loaded_tp1
    del loaded_tp2
    pbar.update(1)
    pbar.set_description("Done.")

    return model


def create_dummy_model(use_cache=False, device=torch.device("cpu")):
    model = model20b.NeoX20BModel(ArgsDummy, use_cache=use_cache).half().to(device)
    return model
