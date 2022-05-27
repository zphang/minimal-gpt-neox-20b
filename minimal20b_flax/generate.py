import jax
import numpy as np
import minimal20b_flax.model_xmap as model_xmap
import minimal20b_flax.create as create


# I have no idea if this helps
CACHED_FUNCS = {}


def generate(input_string: str,
             neox_model: model_xmap.GPTNeoX20BModel,
             params,
             tokenizer,
             maximum_context_length: int = None,
             rng: jax.random.PRNGKey = None,
             mesh=None):
    input_ids = tokenizer.encode(input_string).ids
    input_ctx_length = len(input_ids)
    # Specify a maximum_context_length to prevent re-jit-ing
    # Set it to None for the fastest inference for a fixed token length
    if maximum_context_length is not None:
        assert input_ctx_length < maximum_context_length
        padded_input_ids = np.zeros(maximum_context_length, dtype=int)
        padded_input_ids[-input_ctx_length:] = input_ids
    else:
        padded_input_ids = np.array([0] * neox_model.generate_length + input_ids)

    if rng is None:
        rng = jax.random.PRNGKey(np.random.randint(100000000))
    elif isinstance(rng, int):
        rng = jax.random.PRNGKey(rng)

    if "generate" not in CACHED_FUNCS:
        CACHED_FUNCS["generate"] = jax.experimental.maps.xmap(
            neox_model.generate,
            in_axes=(
                ["shard", ...],
                [...],
                [...],
                [...],
            ),
            out_axes={
                "generated_logits": [...],
                "generated_tokens": [...],

            },
            axis_resources={'shard': 'tp', 'batch': 'dp'},
        )
    if mesh is None:
        mesh = create.get_colab_mesh()
    with mesh:
        output = CACHED_FUNCS["generate"](
            params,
            padded_input_ids,
            input_ctx_length,
            rng,
        )
    return {
        "generated_string": tokenizer.decode(output["generated_tokens"]),
        "generated_tokens": np.array(output["generated_tokens"]),
        "generated_logits": np.array(output["generated_logits"]),
    }
