import argparse
import json
import os

import pandas as pd
import numpy as np
import torch
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from nltk.tokenize import sent_tokenize
import gc
import joblib
from functions import *


cols = [
    'text_ppl', 'max_sent_ppl', 'sent_ppl_avg', 'sent_ppl_std', 'max_step_ppl',
    'step_ppl_avg', 'step_ppl_std', 'rank_0', 'rank_10', 'rank_100', 'rank_1000',
]
CROSS_ENTROPY = torch.nn.CrossEntropyLoss(reduction='none')


def gpt2_features(text, tokenizer, model, sent_cut,device):
    # Tokenize
    input_max_length = tokenizer.model_max_length - 2
    token_ids, offsets = list(), list()
    sentences = sent_cut(text)

    for s in sentences:
        tokens = tokenizer.tokenize(s)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        difference = len(token_ids) + len(ids) - input_max_length
        if difference > 0:
            ids = ids[:-difference]
        offsets.append((len(token_ids), len(token_ids) + len(ids)))
        token_ids.extend(ids)
        if difference >= 0:
            break

    input_ids = torch.tensor([[tokenizer.bos_token_id] + token_ids]).to(device)
    logits = model(input_ids).logits.squeeze()

    # Shift so that n-1 predict n
    input_ids = input_ids.squeeze()
    shift_logits = logits[:-1].contiguous()
    shift_target = input_ids[1:].contiguous()
    loss = CROSS_ENTROPY(shift_logits, shift_target)

    all_probs = torch.softmax(shift_logits, dim=-1)
    sorted_ids = torch.argsort(all_probs, dim=-1, descending=True)  # stable=True
    expanded_tokens = shift_target.unsqueeze(-1).expand_as(sorted_ids)
    indices = torch.where(sorted_ids == expanded_tokens)
    rank = indices[-1]
    counter = [
        rank < 10,
        (rank >= 10) & (rank < 100),
        (rank >= 100) & (rank < 1000),
        rank >= 1000
    ]
    counter = [c.long().sum(-1).item() for c in counter]

    # compute different-level ppl
    text_ppl = loss.mean().exp().item()
    sent_ppl = list()
    for start, end in offsets:
        nll = loss[start: end].sum() / (end - start)
        sent_ppl.append(nll.exp().item())

    max_sent_ppl = max(sent_ppl)
    sent_ppl_avg = sum(sent_ppl) / len(sent_ppl)
    if len(sent_ppl) > 1:
        sent_ppl_std = torch.std(torch.tensor(sent_ppl)).item()
    else:
        sent_ppl_std = 0

    mask = torch.tensor([1] * loss.size(0)).to(device)
    step_ppl = loss.cumsum(dim=-1).div(mask.cumsum(dim=-1)).exp()
    max_step_ppl = step_ppl.max(dim=-1)[0].item()
    step_ppl_avg = step_ppl.sum(dim=-1).div(loss.size(0)).item()
    if step_ppl.size(0) > 1:
        step_ppl_std = step_ppl.std().item()
    else:
        step_ppl_std = 0
    ppls = [
        text_ppl, max_sent_ppl, sent_ppl_avg, sent_ppl_std,
        max_step_ppl, step_ppl_avg, step_ppl_std
    ]
    return counter, ppls  # type: ignore

def get_gltr_feats(args,texts,tokenizer,device='cuda'):

    models_test_feats = []
    MODEL_EN = AutoModelForCausalLM.from_pretrained(args.model_small).to(device)
    test_ppl_feats  = []
    test_gltr_feats = []
    with torch.no_grad():
        for text in tqdm.tqdm(texts):
            gltr, ppl= gpt2_features(text, tokenizer, MODEL_EN, sent_tokenize,device)
            test_ppl_feats.append(ppl)
            test_gltr_feats.append(gltr)
    X_test = pd.DataFrame(
        np.concatenate((test_ppl_feats, test_gltr_feats), axis=1),
        columns=[f'tinyllama-{col}' for col in cols]
    )
    models_test_feats.append(X_test)
    del X_test
    del MODEL_EN
    del test_ppl_feats; del test_gltr_feats
    gc.collect()
    torch.cuda.empty_cache()

    model_path = args.model_large
    device_map = {'' : int(device.split(':')[1])}

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16"
    )

    MODEL_EN = AutoModelForCausalLM.from_pretrained(model_path,
                                                 quantization_config=bnb_config,
                                                 use_cache = False,
                                                device_map=device_map)
    test_ppl_feats  = []
    test_gltr_feats = []
    with torch.no_grad():
        for text in tqdm.tqdm(texts):
            gltr, ppl = gpt2_features(text,tokenizer, MODEL_EN, sent_tokenize,device)
            test_ppl_feats.append(ppl)
            test_gltr_feats.append(gltr)
    X_test = pd.DataFrame(
        np.concatenate((test_ppl_feats, test_gltr_feats), axis=1),
        columns=[f'llama2-{col}' for col in cols]
    )
    models_test_feats.append(X_test)
    del X_test
    del MODEL_EN
    del test_ppl_feats; del test_gltr_feats
    gc.collect()
    torch.cuda.empty_cache()

    gltrppl_feats_df = pd.concat(models_test_feats, axis=1)
    print('gltr_feats:')
    print(gltrppl_feats_df.head())

    gltr_feats_df = gltrppl_feats_df.iloc[:, gltrppl_feats_df.columns.str.contains('rank')]
    print(gltr_feats_df.head())

    ppl_feats_df = gltrppl_feats_df.iloc[:, gltrppl_feats_df.columns.str.contains('ppl')]
    print(ppl_feats_df.head())

    return gltr_feats_df.values, ppl_feats_df.values

def check_and_convert(x): # 如果是列表
    return np.array(x, dtype=float)  # 尝试将其转换为 numpy 数组

def run_train(args):
    print('training starting')
    name_small = args.model_small.split('/')[-1]
    name_large = args.model_large.split('/')[-1]
    train_df = get_gltr_df(args, name_small, name_large).dropna().reset_index(drop=True)

    if args.run_feats:
        tokenizer = AutoTokenizer.from_pretrained(args.model_small)
        tokenizer.pad_token = tokenizer.eos_token
        gltr_feats,ppl_feats = get_gltr_feats(args, train_df['text'].values, tokenizer, device=args.device)

        if args.save_feats:
            name_small = args.model_small.split('/')[-1]
            name_large = args.model_large.split('/')[-1]
            gltrppl_feats_df = pd.DataFrame({'id': train_df['id'],
                                             'text': train_df['text'],
                                             'gltr_feats': gltr_feats.tolist(),
                                             'ppl_feats': ppl_feats.tolist(),
                                             'generated': train_df['generated']})

            if train_df.columns.str.contains('model').any():
                gltrppl_feats_df['model'] = train_df['model']
            if train_df.columns.str.contains('operation').any():
                gltrppl_feats_df['operation'] = train_df['operation']
            if train_df.columns.str.contains('domain').any():
                gltrppl_feats_df['domain'] = train_df['domain']

            ling_save_dir = f'./detectors/ling-based/lingfeatures'
            if args.task == 'otherdata':
                gltrppl_feats_df.to_json(
                    f'{ling_save_dir}/{args.task}/{args.dataset}_gltrppl_{name_small}_{name_large}.json'
                    , orient="records", lines=True)
            else:
                gltrppl_feats_df.to_json(
                    f'{ling_save_dir}/{args.task}/train/{args.dataset}_gltrppl_{name_small}_{name_large}.json'
                    , orient="records", lines=True)

    else:
        if args.func == 'ppl':
            train_df['ppl_feats'] = train_df['ppl_feats'].apply(check_and_convert)
            train_df = train_df[train_df['ppl_feats'].apply(lambda x: not np.any(np.isnan(x)))]
            print(train_df)

        gltr_feats = np.array([np.array(feat) for feat in train_df['gltr_feats']])
        ppl_feats = np.array([np.array(feat) for feat in train_df['ppl_feats']])

    if args.func == 'gltr':
        features = gltr_feats
    elif args.func == 'ppl':
        features = ppl_feats
    else:
        raise ValueError(f'invalid func: {args.func}')

    print(f'running {args.func} ......')

    features = features[~np.isnan(features).any(axis=1)]
    classifier = get_classifier(features, train_df['generated'].values)

    model_save_dir = f'./detectors/ling-based/classifier'
    joblib.dump(classifier, f'{model_save_dir}/{args.task}/{args.func}_{args.dataset}_n{args.n_sample}.pkl')

def run_test(argse):
    print('testing starting')
    name_small = args.model_small.split('/')[-1]
    name_large = args.model_large.split('/')[-1]

    test_df = get_gltr_df(args,name_small,name_large).dropna().reset_index(drop=True)
    print(test_df)

    if args.run_feats:
        tokenizer = AutoTokenizer.from_pretrained(args.model_small)
        tokenizer.pad_token = tokenizer.eos_token
        gltr_feats,ppl_feats = get_gltr_feats(args, test_df['text'].values, tokenizer, device=args.device)

        if args.save_feats:
            gltrppl_feats_df = pd.DataFrame({'id': test_df['id'],
                                             'text': test_df['text'],
                                             'gltr_feats': gltr_feats.tolist(),
                                             'ppl_feats': ppl_feats.tolist(),
                                             'generated': test_df['generated']})

            if test_df.columns.str.contains('model').any():
                gltrppl_feats_df['model'] = test_df['model']
            if test_df.columns.str.contains('operation').any():
                gltrppl_feats_df['operation'] = test_df['operation']
            if test_df.columns.str.contains('domain').any():
                gltrppl_feats_df['domain'] = test_df['domain']

            ling_save_dir = f'./detectors/ling-based/lingfeatures'
            gltrppl_feats_df.to_json(f'{ling_save_dir}/{args.task}/test/{args.dataset}_gltrppl_{name_small}_{name_large}.json'
                , orient="records", lines=True)
    else:
        if args.func == 'ppl':
            test_df['ppl_feats'] = test_df['ppl_feats'].apply(check_and_convert)
            test_df = test_df[test_df['ppl_feats'].apply(lambda x: not np.any(np.isnan(x)))]
            print(test_df)

        gltr_feats = np.array([np.array(feat) for feat in test_df['gltr_feats']])
        ppl_feats = np.array([np.array(feat) for feat in test_df['ppl_feats']])

    if args.func == 'gltr':
        features = gltr_feats
    elif args.func == 'ppl':
        features = ppl_feats
    else:
        raise ValueError(f'invalid func: {args.func}')

    classifier = joblib.load(args.classifier)

    predictions = classifier.predict_proba(features)[:, 1]
    fpr, tpr, roc_auc = get_roc_metrics(test_df['generated'].values, predictions)
    p, r, pr_auc = get_precision_recall_metrics(test_df['generated'].values, predictions)
    print(f'ROC AUC: {roc_auc:.4f}')

    results = {'predictions': predictions.tolist(),
               'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
               'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r},
               'loss': 1 - pr_auc}

    res_save_dir = f'./detectors/ling-based/results/{args.task}'
    if not os.path.exists(res_save_dir):
        os.makedirs(res_save_dir)
    with open(f'{res_save_dir}/{args.func}_{args.dataset}_n{args.n_sample}_multi{args.multilen}.json', 'w') as fout:
        json.dump(results, fout)
        print(f'Results written into {res_save_dir}/{args.func}_{args.dataset}_n{args.n_sample}_multi{args.multilen}.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str,default='cross-domain',choices=['cross-domain','cross-model','cross-operation','otherdata'])
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'], help='train or val or test mode')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model_small', type=str, default='./models/tinyllama')
    parser.add_argument('--model_large', type=str, default='./models/llama2_7b')
    parser.add_argument('--run_feats', type=bool, default=False)
    parser.add_argument('--save_feats', type=bool, default=False)
    parser.add_argument('--func', type=str, default=None,choices=['gltr','ppl'])
    parser.add_argument('--classifier', type=str, default=None)
    parser.add_argument('--n_sample', type=int, default=2000)
    parser.add_argument('--multilen', type=int, default=0)
    args = parser.parse_args()

    if args.mode == 'train':
        run_train(args)
    elif args.mode == 'test':
        run_test(args)