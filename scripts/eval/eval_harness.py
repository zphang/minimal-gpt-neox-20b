import argparse
import json
from pprint import pprint

from tqdm import tqdm
import torch
import torch.nn.functional as F

from lm_eval.base import CacheHook
from lm_eval.models.gpt2 import GPT2LM
from lm_eval import tasks, evaluator, utils

import minimal20b


class TokenizerWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode(self, string: str):
        return self.tokenizer.encode(string).ids

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)


class EvalHarnessAdapter(GPT2LM):
    """
    An adapter to run NeoX models on LM Evaluation Harness (https://github.com/EleutherAI/lm-evaluation-harness) tasks.
    """

    def __init__(self, model, tokenizer):

        # self.device = torch.device(f"cuda:0")
        self.device = torch.device("cuda:0")
        self.VOCAB_SIZE = minimal20b.Args20b.vocab_size
        self.model = model
        self.tokenizer = TokenizerWrapper(tokenizer)
        self.EOT_TOKEN_ID = 0
        self.cache_hook = CacheHook(None)
        self.max_length = 2048
        self.max_gen_toks = 128

        self.batch_size = 4

        self.full_attention_mask = minimal20b.generate_mask(2048).to(self.device)

    def greedy_until(self, requests):
        raise NotImplementedError()

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        res = []
        res_len = 0  # storing the result length for later
        with torch.no_grad():

            def _collate(x):
                toks = x[1] + x[2]
                return -len(toks), tuple(toks)

            reord = utils.Reorderer(requests, _collate)
            for chunk in utils.chunks(
                tqdm(reord.get_reordered(), disable=disable_tqdm), self.batch_size
            ):
                inps, contlens, inplens, padding_length = [], [], [], None
                for _, context_enc, continuation_enc in chunk:
                    # when too long to fit in context, truncate from the left
                    inp = torch.tensor(
                        (context_enc + continuation_enc)[-(self.max_length + 1):][:-1],
                        dtype=torch.long,
                    ).to(self.device)
                    (inplen,) = inp.shape

                    cont = continuation_enc

                    # since in _collate we make sure length is descending, the longest is always the first one.
                    padding_length = (
                        padding_length if padding_length is not None else inplen
                    )

                    # pad to length
                    inp = torch.cat(
                        [
                            inp,  # [seq]
                            torch.zeros(padding_length - inplen, dtype=torch.long).to(
                                inp.device
                            ),  # [padding_length - seq]
                        ],
                        dim=0,
                    )

                    inps.append(inp.unsqueeze(0))
                    contlens.append(cont)
                    inplens.append(inplen)

                logits = self._model_call(torch.cat(inps, dim=0))
                res_len += len(chunk)

                if logits is not None:
                    multi_logits = F.log_softmax(logits, dim=-1)  # [batch, seq, vocab]
                    for (cache_key, _, _), logits, inp, inplen, cont_toks in zip(
                        chunk, multi_logits, inps, inplens, contlens
                    ):
                        contlen = len(cont_toks)
                        logits = logits[inplen - contlen:inplen].unsqueeze(
                            0
                        )  # [1, seq, vocab]
                        greedy_tokens = logits.argmax(dim=-1)
                        # cont_toks :: [1, seq]
                        cont_toks = (
                            torch.tensor(cont_toks, dtype=torch.long)
                            .unsqueeze(0)
                            .to(multi_logits.device)
                        )
                        # noinspection PyUnresolvedReferences
                        max_equal = (greedy_tokens == cont_toks).all()
                        logits = torch.gather(
                            logits, 2, cont_toks.unsqueeze(-1)
                        ).squeeze(
                            -1
                        )  # [1, seq]
                        answer = (float(logits.sum()), bool(max_equal))
                        res.append(answer)

        return reord.get_original(res)

    def _model_call(self, inps):
        length = inps.shape[1]
        model_out = self.model(
            inps.to(self.device),
            attention_mask=self.full_attention_mask[..., :length, :length],
        )
        return model_out

    @torch.no_grad()
    def run_eval(self, eval_tasks=None, num_fewshot=0, bootstrap_iters=2):
        self.model.eval()
        results = evaluator.evaluate(
            lm=self,
            task_dict=tasks.get_task_dict(eval_tasks),
            provide_description=False,
            num_fewshot=num_fewshot,
            limit=None,
            bootstrap_iters=bootstrap_iters,
        )
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--tokenizer_path', type=str, required=True)
    parser.add_argument('--tasks', type=str, required=True)
    parser.add_argument('--output_path', type=str, default=None)
    args = parser.parse_args()
    model = minimal20b.create_model(args.model_path)
    tokenizer = minimal20b.create_tokenizer(args.tokenizer_path)
    adapter = EvalHarnessAdapter(model, tokenizer)
    print("Running evaluation harness...")
    results = adapter.run_eval(
        eval_tasks=args.tasks.split(","),
        bootstrap_iters=10000,
    )
    pprint(results)
    if args.output_path:
        with open(args.output_path, "w") as f:
            f.write(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
