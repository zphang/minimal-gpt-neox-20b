# Minimal GPT-NeoX-20B

This is a fairly minimal implementation of GPT-NeoX-20B in PyTorch. It is meant primarily as an educational/reference implementation, rather than an optimized or feature-full implementation. 

GPT-NeoX-20B is a 20B-parameter autoregressive Transformer model developed by [EleutherAI](https://www.eleuther.ai/) with the support of [CoreWeave](https://www.coreweave.com/), trained using the [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) library.

Some notes about the model:

* The model weights and activations come in half-precision (fp16).
* In fp16, loading the model weights requires about 40GB of GPU memory. Running inference on a single batch requires some more.
* The model supports up to a maximum sequence length of 2048 tokens.

## Setup

### Installation

Install PyTorch with your appropriate CUDA version, and then install from the `requirements.txt` (basically just `tokenizers`).

```bash
pip install -r requirements.txt
```

### Download weights

Following the [NeoX guide](https://github.com/EleutherAI/gpt-neox#download-links), download the model weights and tokenizer JSON file with the following command:

```bash
wget --cut-dirs=5 -nH -r --no-parent --reject "index.html*" https://mystic.the-eye.eu/public/AI/models/GPT-NeoX-20B/slim_weights/ -P 20B_checkpoints
```

You can also manually down them from [here](https://mystic.the-eye.eu/public/AI/models/GPT-NeoX-20B/slim_weights/). Because of the size of the model, the model weights are broken into multiple files, based on the DeepSpeed save format.

#### Generate text

Here is some sample code to generate text. Note that since we are greedily decoding with no fancy tricks, there tends to be quite some repetitiion in generations.

```python
import minimal20b
import torch
model = minimal20b.create_model(
    "/path/to/20B_checkpoints/global_step150000",
    use_cache=True,
    device="cuda:0",
)
tokenizer = minimal20b.create_tokenizer(
    "/path/to/20B_checkpoints/20B_tokenizer.json",
)
with torch.inference_mode():
    minimal20b.greedy_generate_text(
        model, tokenizer,
        "GPTNeoX20B is a 20B-parameter autoregressive Transformer model developed by EleutherAI.",
        max_seq_len=100,
    )
```