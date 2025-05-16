import torch
import torch.nn.functional as F
import tqdm
import gc
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import json
import argparse
import ast
from functions import *
import os

def get_model(args):
    device_map = {'' : int(args.device.split(':')[1])}
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16"
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        use_cache=False,
        device_map=device_map
    )

    return model,tokenizer
def get_likelihood(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    logits = logits.view(-1, logits.shape[-1])
    labels = labels.view(-1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_likelihood = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return log_likelihood.mean().item()

def get_rank(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
    assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

    ranks, timesteps = matches[:, -1], matches[:, -2]

    assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"

    ranks = ranks.float() + 1
    return -ranks.mean().item()

def get_logrank(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
    assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

    ranks, timesteps = matches[:, -1], matches[:, -2]

    assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"

    ranks = ranks.float() + 1
    ranks = torch.log(ranks)
    return -ranks.mean().item()

def get_entropy(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
    entropy = -entropy.sum(-1)
    return entropy.mean().item()

def get_baselines(args,model,tokenizer,test_df,subtask=None):
    texts = test_df['text'].values
    generated = test_df['generated'].values

    criterion_fns = {'likelihood': get_likelihood,
                     'rank': get_rank,
                     'logrank': get_logrank,
                     'entropy': get_entropy}

    for name in criterion_fns:
        eval_results = []
        criterion_fn = criterion_fns[name]
        for text in tqdm.tqdm(texts):
            tokenized = tokenizer(text, return_tensors="pt", padding=True,return_token_type_ids=False).to(model.device)
            labels = tokenized.input_ids[:, 1:]
            with torch.no_grad():
                logits = model(**tokenized).logits[:, :-1]
                text_crit = criterion_fn(logits, labels)

            eval_results.append({"text": text,
                                 "text_crit": text_crit,})

        predictions = [x["text_crit"] for x in eval_results]
        fpr, tpr, roc_auc = get_roc_metrics(generated, predictions)
        p, r, pr_auc = get_precision_recall_metrics(generated, predictions)

        print(f'{name}_roc_auc: {roc_auc:.4f}')


        results = {'name': f'{name}_threshold',
                   'predictions': predictions,
                   'raw_results': eval_results,
                   'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
                   'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r},
                   'loss': 1 - pr_auc}

        detect_save_dir = f'./detectors/baselines/results/{args.task}'
        if not os.path.exists(detect_save_dir):
            os.makedirs(detect_save_dir)
        with open(f'{detect_save_dir}/baselines_{args.dataset}_multi{args.multilen}.json', 'w') as fout:
            json.dump(results, fout)
            print(f'Results written into {detect_save_dir}/baselines_{args.dataset}_multi{args.multilen}.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--model_path', type=str, default='./models/Mistral-7B-v0.1')
    parser.add_argument('--task', type=str, choices=['cross-domain','cross-operation','cross-model'])
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--multilen', type=int, default=0)
    args = parser.parse_args()

    model,tokenizer = get_model(args)
    test_df = get_df(args).dropna().reset_index(drop=True)

    get_baselines(args, model, tokenizer, test_df)
