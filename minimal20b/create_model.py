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
        filename = f"layer_{layer_i + 2:02d}-model_00-model_states.pt"
        loaded = torch.load(os.path.join(checkpoint_path, filename))
        model.layer_list[layer_i].load_state_dict(loaded)
        del loaded
        pbar.update(1)

    # Load input embedding
    pbar.set_description(f"Loading input embedding")
    loaded = torch.load(os.path.join(checkpoint_path, "layer_00-model_00-model_states.pt"))
    model.embed_in.load_state_dict({"weight": loaded["word_embeddings.weight"]})
    del loaded
    pbar.update(1)

    # Load final layer norm
    pbar.set_description(f"Loading final layer norm")
    loaded = torch.load(os.path.join(checkpoint_path, "layer_47-model_00-model_states.pt"))
    model.final_layer_norm.load_state_dict({
        "weight": loaded["norm.weight"],
        "bias": loaded["norm.bias"],
    })
    del loaded
    pbar.update(1)

    # Load output embedding
    pbar.set_description(f"Loading output embedding")
    loaded = torch.load(os.path.join(checkpoint_path, "layer_48-model_00-model_states.pt"))
    model.logits_out.load_state_dict({
        "weight": loaded["final_linear.weight"],
    })
    del loaded
    pbar.update(1)
    pbar.set_description("Done.")

    return model


def create_dummy_model(use_cache=False, device=torch.device("cpu")):
    model = model20b.NeoX20BModel(ArgsDummy, use_cache=use_cache).half().to(device)
    return model
