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

#### Evaluation

To run evaluation with the LM-eval-harness, you will need to install some additional dependencies (mostly just the eval harness library):

```bash
pip install -r scripts/eval/requirements.txt
```

Most datasets are automatically downloaded via Hugging Face `datasets`, but if you are evaluating on lambada, you will need to separately download the data.

```bash
mkdir -p data/lambada
wget http://eaidata.bmk.sh/data/lambada_test.jsonl -O data/lambada/lambada_test.jsonl
```

Then, you can run the following command.

```bash
python scripts/eval/eval_harness.py \
    --model_path /path/to/20B_checkpoints/global_step150000 \
    --tokenizer_path /path/to/20B_checkpoints/20B_tokenizer.json \
    --tasks lambada,anli_r1,anli_r2,anli_r3,wsc,winogrande,hellaswag,piqa
```

| Task       | Metric          | NeoX Impl (2 GPU) | This Repo (1 GPU) |
|------------|-----------------|-------------------|-------------------|
| anli_r1    | acc             | 0.3270            | 0.3300            | 
|            | acc_stderr      | 0.0148            | 0.0149            | 
| anli_r2    | acc             | 0.3410            | 0.3420            | 
|            | acc_stderr      | 0.0150            | 0.0150            | 
| anli_r3    | acc             | 0.3567            | 0.3617            | 
|            | acc_stderr      | 0.0138            | 0.0139            | 
| hellaswag  | acc             | 0.5351            | 0.5335            | 
|            | acc_stderr      | 0.0050            | 0.0050            | 
|            | acc_norm        | 0.7140            | 0.7126            | 
|            | acc_norm_stderr | 0.0045            | 0.0045            | 
| lambada    | acc             | 0.7211            | 0.7223            | 
|            | acc_stderr      | 0.0062            | 0.0062            | 
|            | ppl             | 3.6760            | 3.6559            | 
|            | ppl_stderr      | 0.0760            | 0.0757            | 
| piqa       | acc             | 0.7748            | 0.7758            | 
|            | acc_stderr      | 0.0097            | 0.0097            | 
|            | acc_norm        | 0.7786            | 0.7856            | 
|            | acc_norm_stderr | 0.0097            | 0.0096            | 
| winogrande | acc             | 0.6598            | 0.6598            | 
|            | acc_stderr      | 0.0133            | 0.0133            | 
| wsc        | acc             | 0.5096            | 0.4808            | 
|            | acc_stderr      | 0.0493            | 0.0492            | 






