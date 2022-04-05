import torch
import torch.nn as nn
from tqdm import auto as tqdm_lib


def greedy_generate(model: nn.Module, input_ids: torch.Tensor, max_seq_len: int,
                    verbose=True):
    """Generate greedily from 20B.

    :param model: NeoX20BModel
    :param input_ids: token IDs [batch_size, seq_len]
    :param max_seq_len: max sequence length to generate up to (includes input_ids)
    :param verbose: whether to print progress

    :return: List of token IDs
    """
    initial_input_length = input_ids.shape[1]
    current_input_ids = input_ids
    layer_past = None
    layer_past_length = 0
    all_token_ids = input_ids.tolist()
    batch_size = len(all_token_ids)

    if verbose:
        trange = tqdm_lib.trange(initial_input_length, max_seq_len)
    else:
        trange = range(initial_input_length, max_seq_len)

    for _ in trange:
        input_length = current_input_ids.shape[1]
        model_out, layer_past = model(
            current_input_ids,
            layer_past=layer_past,
        )
        greedy_predicted_token_ids = model_out[:, -1].argmax(-1)
        current_input_ids = greedy_predicted_token_ids[:, None]
        for i in range(batch_size):
            all_token_ids[i].append(greedy_predicted_token_ids[i])
        layer_past_length += input_length
    return all_token_ids


def greedy_generate_text(model: nn.Module,
                         tokenizer,
                         initial_str: str,
                         max_seq_len: int,
                         device=torch.device("cuda:0"),
                         verbose=True):
    """Generate greedily from 20B.

    :param model: NeoX20BModel
    :param tokenizer: NeoX20B tokenizer
    :param initial_str: initial string to start generation from
    :param max_seq_len: max sequence length to generate up to (includes input_ids)
    :param device: device to use
    :param verbose: whether to print progress

    :return: List of token IDs
    """
    tokenized = tokenizer.encode(initial_str)
    input_ids = torch.LongTensor([tokenized.ids]).to(device)
    all_token_ids = greedy_generate(model=model, input_ids=input_ids, max_seq_len=max_seq_len, verbose=verbose)
    return tokenizer.decode(all_token_ids[0])
