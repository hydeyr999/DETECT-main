# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pickle
import random
import time
import numpy as np
import torch
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
import torch.nn.functional as F
import tqdm
import argparse
import json
from data import *
from metrics import get_roc_metrics, get_precision_recall_metrics


def load_tokenizer(model_name, cache_dir):
    model_path = os.path.join(cache_dir, model_name)
    base_tokenizer = AutoTokenizer.from_pretrained(model_path)
    if base_tokenizer.pad_token_id is None:
        base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
    return base_tokenizer

def load_model(model_name, cache_dir):
    model_path = os.path.join(cache_dir, model_name)
    device_map = {'': int(args.device.split(':')[1])}
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16"
    )
    base_model = AutoModelForCausalLM.from_pretrained(model_path,
                                                       quantization_config=bnb_config,
                                                       use_cache=False,
                                                       device_map=device_map)

    return base_model
def load_mask_model(model_name, device, cache_dir):
    print(f'Loading mask filling model {model_name}...')
    model_path = os.path.join(cache_dir, model_name)

    mask_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    mask_model = mask_model.to(device)
    return mask_model

def load_mask_tokenizer(model_name, max_length, cache_dir):
    model_path = os.path.join(cache_dir, model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=max_length)
    return tokenizer

def get_samples(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    nsamples = 10000
    lprobs = torch.log_softmax(logits, dim=-1)
    distrib = torch.distributions.categorical.Categorical(logits=lprobs)
    samples = distrib.sample([nsamples]).permute([1, 2, 0])
    return samples

def get_likelihood(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
    lprobs = torch.log_softmax(logits, dim=-1)
    log_likelihood = lprobs.gather(dim=-1, index=labels)
    return log_likelihood.mean(dim=1)

def get_sampling_discrepancy(logits_ref, logits_score, labels):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    samples = get_samples(logits_ref, labels)
    log_likelihood_x = get_likelihood(logits_score, labels)
    log_likelihood_x_tilde = get_likelihood(logits_score, samples)

    miu_tilde = log_likelihood_x_tilde.mean(dim=-1)
    sigma_tilde = log_likelihood_x_tilde.std(dim=-1)
    discrepancy = (log_likelihood_x.squeeze(-1) - miu_tilde) / sigma_tilde
    return discrepancy.item()

def get_sampling_discrepancy_analytic(logits_ref, logits_score, labels):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref = torch.softmax(logits_ref, dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
    discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt()
    discrepancy = discrepancy.mean()
    return discrepancy.item()

def experiment(args):
    scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.cache_dir)
    scoring_model = load_model(args.scoring_model_name, args.cache_dir)
    scoring_model.eval()
    if args.reference_model_name != args.scoring_model_name:
        reference_tokenizer = load_tokenizer(args.reference_model_name, args.cache_dir)
        reference_model = load_model(args.reference_model_name, args.cache_dir)
        reference_model.eval()
    data = get_df(args).dropna().reset_index(drop=True)
    print(data)
    n_samples = len(data)

    if args.discrepancy_analytic:
        name = "sampling_discrepancy_analytic"
        criterion_fn = get_sampling_discrepancy_analytic
    else:
        name = "sampling_discrepancy"
        criterion_fn = get_sampling_discrepancy

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    results = []
    start_time = time.time()
    for idx in tqdm.tqdm(range(n_samples), desc=f"Computing {name} criterion"):
        text = data['text'][idx]

        tokenized = scoring_tokenizer(text, max_length=args.max_len, truncation=True, return_tensors="pt",
                                      padding=True, return_token_type_ids=False).to(args.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = scoring_model(**tokenized).logits[:, :-1]
            if args.reference_model_name == args.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = reference_tokenizer(text,max_length=args.max_len, truncation=True,return_tensors="pt", padding=True,
                                                return_token_type_ids=False).to(args.device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = reference_model(**tokenized).logits[:, :-1]
            try:
                text_crit = criterion_fn(logits_ref, logits_score, labels)
            except Exception as e:
                continue

        results.append({"text": text,
                        "text_crit": text_crit,
                        'generated':data['generated'][idx]})

    print(f"Total time: {time.time() - start_time:.4f}s")
    predictions = [x["text_crit"] for x in results]
    generated = [x["generated"] for x in results]

    fpr, tpr, roc_auc = get_roc_metrics(predictions, generated)
    p, r, pr_auc = get_precision_recall_metrics(predictions, generated)
    print(f"Criterion {name}_threshold ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")

    results = { 'name': f'{name}_threshold',
                'info': {'n_samples': n_samples},
                'predictions': predictions,
                'raw_results': results,
                'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
                'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r},
                'loss': 1 - pr_auc}

    detect_save_dir = f'./detectors/detectgpt/results/{args.task}'
    if not os.path.exists(detect_save_dir):
        os.makedirs(detect_save_dir)
    with open(f'{detect_save_dir}/fast_{args.dataset}_multi{args.multilen}.json', 'wb') as fout:
        pickle.dump(results, fout)
        print(f'Results written into {detect_save_dir}/fast_{args.dataset}_multi{args.multilen}.json')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--reference_model_name', type=str, default="gpt-neo-2.7B")
    parser.add_argument('--scoring_model_name', type=str, default="gpt-neo-2.7B")
    parser.add_argument('--discrepancy_analytic', action='store_true')
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--cache_dir', type=str, default="./models/")
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--multilen',type=int,default=0)
    args = parser.parse_args()

    experiment(args)
